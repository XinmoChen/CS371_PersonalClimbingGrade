"""Adaptive shrinkage (Stephens, 2017) for PACE blocks: a scale mixture of zero centred normals as the prior
g(a) = sum_k pi_k N(0, s_k^2) with s_0 = 0 (a point mass at zero). Given the sum S and count n of partial residuals
of a level and the noise variance sigma^2, the level mean xbar = S/n has standard error se^2 = sigma^2/n and
    E[a | S, n] = sum_k w_k * xbar * s_k^2 / (s_k^2 + se^2),   w_k ~ pi_k N(xbar; 0, s_k^2 + se^2).
Small, noisy evidence is pulled to zero almost completely, while consistent evidence is kept nearly in full,
which linear (Gaussian) shrinkage cannot do with a single penalty."""
import numpy as np

GRID = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.6])


def post_mean(S, n, sigma2, pi, grid=GRID):
    S = np.asarray(S, float); n = np.asarray(n, float)
    out = np.zeros_like(S)
    m = n > 0
    if not m.any():
        return out
    xbar = S[m] / n[m]; se2 = sigma2 / n[m]
    var = grid[None, :] ** 2 + se2[:, None]                                   # (L, K)
    logw = np.log(np.maximum(pi, 1e-300))[None, :] - 0.5 * np.log(2 * np.pi * var) - 0.5 * xbar[:, None] ** 2 / var
    logw -= logw.max(1, keepdims=True)
    w = np.exp(logw); w /= w.sum(1, keepdims=True)
    shrink = grid[None, :] ** 2 / var
    out[m] = xbar * (w * shrink).sum(1)
    return out


def fit_pi(xbar, se2, grid=GRID, iters=500, tol=1e-8):
    """EM for the mixture weights given level means and their squared standard errors."""
    K = len(grid); pi = np.full(K, 1.0 / K)
    var = grid[None, :] ** 2 + se2[:, None]
    lik = np.exp(-0.5 * xbar[:, None] ** 2 / var) / np.sqrt(2 * np.pi * var)
    for _ in range(iters):
        r = lik * pi[None, :]; r /= r.sum(1, keepdims=True)
        new = r.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new; break
        pi = new
    return pi
