"""
test_saddlepoint_module.py — Tests for metararepi.spa.saddlepoint.

Validates the package-level SPA module (which completes the previously
stubbed saddlepoint.py) against known χ²(k) distribution cumulants.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from metararepi.spa.saddlepoint import (  # noqa: E402
    spa_pvalue,
    spa_pvalues_batch,
    _make_cgf,
    _solve_saddlepoint,
    _lugannani_rice,
)

import jax.numpy as jnp  # noqa: E402

K_DOF = 10


@pytest.fixture(scope="module")
def chi2_cumulants() -> np.ndarray:
    """Exact cumulants of χ²(10): κ_r = 2^{r-1} · (r-1)! · k."""
    k = K_DOF
    return np.array([k, 2 * k, 8 * k, 48 * k], dtype=np.float64)


class TestSPAModule:

    def test_spa_pvalue_returns_dict(self, chi2_cumulants):
        """spa_pvalue must return a dict with required keys."""
        result = spa_pvalue(15.0, chi2_cumulants)
        assert "pvalue" in result
        assert "saddlepoint" in result
        assert "cumulants" in result

    def test_pvalue_in_valid_range(self, chi2_cumulants):
        """P-value must be in [0, 1]."""
        for Q in [5.0, 10.0, 15.0, 25.0, 40.0]:
            result = spa_pvalue(Q, chi2_cumulants)
            assert 0.0 <= result["pvalue"] <= 1.0, f"p={result['pvalue']} at Q={Q}"

    def test_pvalue_monotone_decreasing(self, chi2_cumulants):
        """P(X > Q) must decrease as Q increases."""
        Q_values = [8.0, 12.0, 16.0, 20.0, 25.0]
        pvals = [spa_pvalue(q, chi2_cumulants)["pvalue"] for q in Q_values]
        for i in range(len(pvals) - 1):
            assert pvals[i] >= pvals[i + 1]

    @pytest.mark.parametrize("Q_val", [12.0, 15.0, 18.0, 22.0, 25.0])
    def test_pvalue_order_of_magnitude(self, chi2_cumulants, Q_val):
        """SPA p-value should be within 1 order of magnitude of scipy chi2.sf."""
        result = spa_pvalue(Q_val, chi2_cumulants)
        scipy_pval = scipy.stats.chi2.sf(Q_val, df=K_DOF)
        log_ratio = abs(
            np.log10(max(result["pvalue"], 1e-300))
            - np.log10(max(scipy_pval, 1e-300))
        )
        assert log_ratio < 1.0

    def test_near_mean_gaussian_fallback(self, chi2_cumulants):
        """Q ≈ mean → Gaussian fallback → p ≈ 0.5."""
        result = spa_pvalue(float(chi2_cumulants[0]), chi2_cumulants)
        np.testing.assert_allclose(result["pvalue"], 0.5, atol=0.05)

    def test_batch_matches_scalar(self, chi2_cumulants):
        """Batch API must reproduce scalar API for each Q."""
        Q_batch = np.array([10.0, 15.0, 20.0, 25.0])
        batch_result = spa_pvalues_batch(Q_batch, chi2_cumulants)
        for i, q in enumerate(Q_batch):
            scalar_result = spa_pvalue(q, chi2_cumulants)
            np.testing.assert_allclose(
                batch_result["pvalues"][i],
                scalar_result["pvalue"],
                rtol=1e-7,
            )

    def test_batch_shape(self, chi2_cumulants):
        """Output shapes must match input batch size."""
        Q_batch = np.array([8.0, 12.0, 16.0])
        result = spa_pvalues_batch(Q_batch, chi2_cumulants)
        assert result["pvalues"].shape == (3,)
        assert result["saddlepoints"].shape == (3,)
        assert result["cumulants"].shape == (4,)

    def test_cgf_at_zero(self, chi2_cumulants):
        """K(0) = 0."""
        cgf, _, _, _ = _make_cgf(jnp.asarray(chi2_cumulants))
        assert abs(float(cgf(jnp.float64(0.0)))) < 1e-15

    def test_halley_solver_converges(self, chi2_cumulants):
        """Halley solver must converge with small residual."""
        kappa = jnp.asarray(chi2_cumulants)
        Q_val = chi2_cumulants[0] + 2.0 * np.sqrt(chi2_cumulants[1])
        Q = jnp.float64(Q_val)
        t_hat = _solve_saddlepoint(kappa, Q)
        _, cgf_d1, _, _ = _make_cgf(kappa)
        residual = abs(float(cgf_d1(t_hat)) - Q_val)
        assert residual < 1e-10
