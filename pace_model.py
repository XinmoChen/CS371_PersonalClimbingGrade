"""PACE: Personalization with Aggregate and Contextual Evidence.

Gaussian crossed random effects model of the deviation d = y - g (perceived minus official grade):

    d_n = b0 + b1 * gc_n + sum_k a^(k)[ z_k(n) ] + eps_n,    eps ~ N(0, sigma^2),   a^(k)_l ~ N(0, tau_k^2)

where gc_n is the centred official grade and each block k is a categorical grouping (climber u_i, route c_r,
setter e_s, and optional interaction blocks). The perceived grade prediction is  y_hat = g + d_hat, which equals
beta * g + u_i + c_r + e_s with beta = 1 + b1 (the intercept is absorbed).

Estimation is MAP (equivalently ridge regression with one penalty lambda_k = sigma^2 / tau_k^2 per block),
computed by backfitting (block Gauss-Seidel), which converges for this positive definite system
(Ghosh, Hastie and Owen, 2022). Prediction is prequential: for every calendar day D the model is refit on all
logs dated before D (warm started from day D-1), and logs within day D additionally absorb same-day logs with a
strictly earlier timestamp through one exact conditional update per block. No log ever sees itself or the future.
"""
import numpy as np
from common import prior_sum_count
import ash


class Block:
    """One random effect block. prior='gauss' uses linear shrinkage S/(n+lam); prior='ash' uses the adaptive
    shrinkage posterior mean under a zero centred normal scale mixture with weights pi and noise variance sigma2."""
    def __init__(self, name, code, lam, prior='gauss', pi=None, sigma2=None):
        self.name, self.code, self.lam = name, np.asarray(code, np.int64), float(lam)
        self.prior, self.pi, self.sigma2 = prior, pi, sigma2
        self.L = int(self.code.max()) + 1

    def shrink(self, num, cnt):
        if self.prior == 'gauss':
            return num / (cnt + self.lam)
        return ash.post_mean(num, cnt, self.sigma2, self.pi)


def fit_prefix(d, X, blocks, n, params=None, tol=1e-7, max_iter=500):
    """Backfitting on the first n rows. params = (beta, [effect arrays]) for warm start. Returns params, residual."""
    L = [b.L for b in blocks]
    if params is None:
        beta = np.zeros(X.shape[1]); eff = [np.zeros(l) for l in L]
    else:
        beta = params[0].copy(); eff = [e.copy() for e in params[1]]
    dd, XX = d[:n], X[:n]
    codes = [b.code[:n] for b in blocks]
    cnt = [np.bincount(c, minlength=l).astype(float) for c, l in zip(codes, L)]
    XtX = XX.T @ XX
    fitted_re = sum(e[c] for e, c in zip(eff, codes)) if blocks else np.zeros(n)
    for it in range(max_iter):
        delta = 0.0
        r = dd - fitted_re
        new_beta = np.linalg.solve(XtX + 1e-9 * np.eye(len(beta)), XX.T @ r)
        delta = max(delta, np.abs(new_beta - beta).max()); beta = new_beta
        xb = XX @ beta
        for k, b in enumerate(blocks):
            own = eff[k][codes[k]]
            partial = dd - xb - (fitted_re - own)
            num = np.bincount(codes[k], weights=partial, minlength=L[k])
            new = b.shrink(num, cnt[k])
            delta = max(delta, np.abs(new - eff[k]).max())
            fitted_re += new[codes[k]] - own
            eff[k] = new
        if delta < tol:
            break
    resid = dd - XX @ beta - fitted_re
    return (beta, eff), resid, it + 1


def level_sums(d, X, blocks, n, params):
    """Sum of partial residuals and count per level for every block, at the given parameters (first n rows)."""
    beta, eff = params
    codes = [b.code[:n] for b in blocks]
    xb = X[:n] @ beta
    fitted = sum(e[c] for e, c in zip(eff, codes)) if blocks else np.zeros(n)
    out = []
    for k, b in enumerate(blocks):
        partial = d[:n] - xb - (fitted - eff[k][codes[k]])
        out.append((np.bincount(codes[k], weights=partial, minlength=b.L), np.bincount(codes[k], minlength=b.L).astype(float)))
    return out


def prequential(d, X, blocks, day, t, gate_block=None, verbose=False, snapshot_days=False, hide_dateonly=False):
    """Prequential predictions for every row (rows must be sorted by time).

    Returns dict with
      pred        : predicted deviation using all blocks (within-day conditional updates applied)
      block_value : per block, the effect value used for each row
      block_count : per block, the number of earlier logs behind that effect for each row
      sigma2      : residual variance of the day-start fit, per row
      fixed       : fixed part b0 + b1*gc for each row
    hide_dateonly=True is the conservative sensitivity setting: a log stamped exactly 00:00:00 (a date without a time)
    becomes visible to other logs only from the next day, since its true order within its day is unknown.
    """
    N = len(d)
    days, starts = np.unique(day, return_index=True)
    ends = np.append(starts[1:], N)
    K = len(blocks)
    out_val = [np.zeros(N) for _ in range(K)]
    out_cnt = [np.zeros(N) for _ in range(K)]
    fixed = np.zeros(N); sigma2 = np.full(N, np.nan)
    params = None
    snaps = {} if snapshot_days else None
    iters_total = 0
    for j, (s, e) in enumerate(zip(starts, ends)):
        if s > 0:
            params, resid, it = fit_prefix(d, X, blocks, s, params)
            iters_total += it
            s2 = float(np.mean(resid ** 2))
        else:
            params = (np.zeros(X.shape[1]), [np.zeros(b.L) for b in blocks]); s2 = np.nan
        beta, eff = params
        if snapshot_days:
            snaps[days[j]] = (beta.copy(), [x.astype(np.float32).copy() for x in eff], s2)
        # day-start counts per level
        idx = slice(s, e)
        xb_day = X[idx] @ beta
        fixed[idx] = xb_day
        sigma2[idx] = s2
        # within-day update: for rows of this day, add same-day logs with strictly earlier timestamp
        dd = d[idx]; tt = t[idx]
        day_codes = [b.code[idx] for b in blocks]
        base_vals = [eff[k][day_codes[k]] for k in range(K)]
        sums = level_sums(d, X, blocks, s, params) if s > 0 else None
        for k, b in enumerate(blocks):
            if s > 0:
                num_prev = sums[k][0][day_codes[k]]; c_prev = sums[k][1][day_codes[k]]
            else:
                num_prev = np.zeros(e - s); c_prev = np.zeros(e - s)
            partial = dd - xb_day - sum(base_vals[kk] for kk in range(K) if kk != k)
            if hide_dateonly:
                vis = (tt % 86400 != 0).astype(float)
                ds, _ = prior_sum_count(day_codes[k], tt, partial * vis)
                dc, _ = prior_sum_count(day_codes[k], tt, vis)
            else:
                ds, dc = prior_sum_count(day_codes[k], tt, partial)
            out_val[k][idx] = b.shrink(num_prev + ds, c_prev + dc)
            out_cnt[k][idx] = c_prev + dc
        if verbose and j % 100 == 0:
            print(f'  day {j}/{len(days)} rows<{s} iters so far {iters_total}', flush=True)
    pred = fixed + sum(out_val)
    return dict(pred=pred, block_value={b.name: v for b, v in zip(blocks, out_val)},
                block_count={b.name: c for b, c in zip(blocks, out_cnt)}, sigma2=sigma2, fixed=fixed,
                snapshots=snaps, final_params=params, iters=iters_total)


def gated(res, block='climber', lam=None, z=1.96):
    """PACE-G: keep the personal effect only when its approximate posterior interval excludes zero.
    Conditional posterior of u_i given the other effects is N(u_hat, sigma^2 / (n_i + lambda_u))."""
    u = res['block_value'][block]; n = res['block_count'][block]
    sd = np.sqrt(np.nan_to_num(res['sigma2'], nan=np.nanmedian(res['sigma2'])) / (n + lam))
    keep = np.abs(u) > z * sd
    return res['pred'] - u + np.where(keep, u, 0.0), keep
