"""Empirical Bayes for adaptive shrinkage blocks: alternate a static PACE fit with EM updates of the mixture
weights of each ash block, on training rows only."""
import numpy as np
import ash
from pace_model import fit_prefix, level_sums


def fit_ash_weights(d, X, blocks, n, rounds=4, verbose=False):
    params = None
    for r in range(rounds):
        params, resid, _ = fit_prefix(d, X, blocks, n, params, tol=1e-7, max_iter=1000)
        sums = level_sums(d, X, blocks, n, params)
        for b, (S, c) in zip(blocks, sums):
            if b.prior != 'ash':
                continue
            m = c > 0
            b.pi = ash.fit_pi(S[m] / c[m], b.sigma2 / c[m])
        if verbose:
            print('round', r, {b.name: np.round(b.pi, 3).tolist() for b in blocks if b.prior == 'ash'}, 'mse', round(float(np.mean(resid ** 2)), 5))
    return blocks
