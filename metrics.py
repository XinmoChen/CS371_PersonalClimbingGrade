"""Evaluation metrics. All predictions are deviations d_hat (perceived minus official grade); the perceived grade
prediction is g + d_hat, so every error below equals the error on the perceived grade itself."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rounded(p):
    """What an app would display: the nearest whole grade step, clipped to the observed range [-3, 3]."""
    return np.clip(np.round(p), -3, 3)


def sign_class(v):
    return np.sign(v).astype(int)            # -1 softer, 0 same, +1 harder


def avg_user_rmse(y, p, users):
    per = pd.DataFrame({'u': users, 'e2': (y - p) ** 2}).groupby('u').e2.mean() ** .5
    return float(per.mean()), float(per.std(ddof=0))


def core_metrics(y, p, users):
    """The headline metric set used in every results table."""
    pr = rounded(p)
    dis = y != 0
    m = {
        'n': int(len(y)),
        'RMSE': rmse(y, p), 'MAE': mae(y, p),
        'RMSE_r': rmse(y, pr), 'MAE_r': mae(y, pr),
        'RMSE_dis': rmse(y[dis], p[dis]) if dis.any() else np.nan,
        'acc_r': float(np.mean(pr == y)),
    }
    m['userRMSE_r'], m['userRMSE_r_sd'] = avg_user_rmse(y, pr, users)
    yc, pc = sign_class(y), sign_class(pr)
    m['macroF1'] = float(f1_score(yc, pc, labels=[-1, 0, 1], average='macro', zero_division=0))
    P, R, F, _ = precision_recall_fscore_support(yc, pc, labels=[-1, 0, 1], zero_division=0)
    for lab, j in (('soft', 0), ('same', 1), ('hard', 2)):
        m[f'P_{lab}'], m[f'R_{lab}'], m[f'F1_{lab}'] = float(P[j]), float(R[j]), float(F[j])
    # ranking quality for "will this climber disagree", "harder", "softer" (scores from the continuous output)
    m['AP_dis'] = float(average_precision_score(dis, np.abs(p))) if dis.any() else np.nan
    m['AP_hard'] = float(average_precision_score(y > 0, p))
    m['AP_soft'] = float(average_precision_score(y < 0, -p))
    return m


# ---------------------------------------------------------------- cluster bootstrap over climbers
def climber_bootstrap(y, preds, users, B=1000, seed=0, ref=None):
    """95% percentile intervals for RMSE of each prediction (dict name -> array), resampling climbers.
    If ref is given, also returns intervals for RMSE(name) - RMSE(ref) (paired: same resampled climbers)."""
    uc, inv = np.unique(users, return_inverse=True)
    n_u = len(uc)
    cnt = np.bincount(inv, minlength=n_u).astype(float)
    sse = {k: np.bincount(inv, weights=(y - p) ** 2, minlength=n_u) for k, p in preds.items()}
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n_u, np.full(n_u, 1.0 / n_u), size=B).astype(float)
    den = W @ cnt
    boot = {k: np.sqrt((W @ s) / den) for k, s in sse.items()}
    out = {}
    for k in preds:
        out[k] = dict(RMSE=float(np.sqrt(sse[k].sum() / cnt.sum())),
                      lo=float(np.percentile(boot[k], 2.5)), hi=float(np.percentile(boot[k], 97.5)))
        if ref is not None and k != ref:
            diff = boot[k] - boot[ref]
            out[k].update(diff=float(np.sqrt(sse[k].sum() / cnt.sum()) - np.sqrt(sse[ref].sum() / cnt.sum())),
                          diff_lo=float(np.percentile(diff, 2.5)), diff_hi=float(np.percentile(diff, 97.5)),
                          p_better=float(np.mean(diff < 0)))
    return out


def helped_hurt(y, p_new, p_ref, users, tol=0.0, crossfit=False):
    """Share of climbers whose mean squared error goes down (helped), up (hurt) or stays within tol (unchanged).
    top10_share_of_gain ranks climbers by their gain on the same logs it measures (in sample, biased upward).
    With crossfit=True, each climber's test logs are split into random halves 40 times; climbers are ranked on one
    half and the top decile's share of the net gain is measured on the other half, in both directions."""
    df = pd.DataFrame({'u': users, 'd': (y - p_new) ** 2 - (y - p_ref) ** 2})
    per = df.groupby('u').d.mean()
    out = dict(helped=float((per < -tol).mean()), hurt=float((per > tol).mean()),
               unchanged=float((per.abs() <= tol).mean()), n_climbers=int(len(per)),
               top10_share_of_gain=float(_top_share(df)))
    if crossfit:
        out.update(_top_share_crossfit(np.asarray(users), df.d.values))
    return out


def _top_share_crossfit(users, dl, reps=40, seed=0):
    uc, inv = np.unique(users, return_inverse=True); n = len(uc); k = max(1, int(0.1 * n))
    rng = np.random.default_rng(seed); vals = []
    for _ in range(reps):
        h = rng.random(len(dl)) < 0.5
        ga = -np.bincount(inv[h], dl[h], minlength=n); gb = -np.bincount(inv[~h], dl[~h], minlength=n)
        for a, b in ((ga, gb), (gb, ga)):
            if b.sum() > 0:
                vals.append(b[np.argsort(-a)[:k]].sum() / b.sum())
    vals = np.array(vals) if vals else np.array([np.nan])
    return dict(top10_share_crossfit=float(np.mean(vals)), top10_share_crossfit_min=float(np.min(vals)),
                top10_share_crossfit_max=float(np.max(vals)))


def _top_share(df):
    tot = -df.d.sum()
    if tot <= 0:
        return np.nan
    per_sum = (-df.groupby('u').d.sum()).sort_values(ascending=False)
    k = max(1, int(len(per_sum) * 0.1))
    return per_sum.iloc[:k].sum() / tot


def expected_calibration_error(prob, y_idx, n_bins=10):
    """Top-label ECE for class probability matrix prob (n x C) and true class index y_idx."""
    conf = prob.max(1); pred = prob.argmax(1); correct = (pred == y_idx).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def average_outputs(outs):
    """Average a list of equally shaped outputs (dicts of numbers, nested) element by element."""
    x0 = outs[0]
    if isinstance(x0, dict):
        return {k: average_outputs([o[k] for o in outs]) for k in x0}
    if isinstance(x0, (int, float, np.floating, np.integer)) and not isinstance(x0, bool):
        return float(np.nanmean([float(o) for o in outs]))
    return x0


def per_seed(fn, preds):
    """preds: name -> list of prediction arrays (one per seed; deterministic models have one).
    Calls fn on each seed view (seed s of every seeded model, deterministic models repeated) and averages the
    numeric outputs, so seeded models are summarized as the mean over seeds, as in the main results table."""
    S = max(len(v) for v in preds.values())
    return average_outputs([fn({k: v[s % len(v)] for k, v in preds.items()}) for s in range(S)])
