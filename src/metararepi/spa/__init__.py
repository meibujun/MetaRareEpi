"""
metararepi.spa — Saddlepoint approximation (SPA) engine.

Computes ultra-precise p-values from the cumulant generating function (CGF)
of the test statistic distribution, leveraging x64 precision for distribution
tails down to P ≈ 10^{-300}.
"""

from metararepi.spa.saddlepoint import (  # noqa: F401
    spa_pvalue,
    spa_pvalues_batch,
)
