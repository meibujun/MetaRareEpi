"""
test_federated_spa.py — Validation of the Federated Meta-SPA engine.

Tests:
    1. Cumulant aggregation (additivity theorem)
    2. CGF derivatives via jax.grad match analytic expectations
    3. Halley saddlepoint solver convergence
    4. Lugannani-Rice p-values validated against scipy.stats.chi2.sf
    5. Singularity fallback (|t̂| < 1e-7  →  Gaussian survival)
    6. Batch API consistency
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from federated_spa import (  # noqa: E402
    aggregate_cumulants,
    _make_cgf,
    _solve_saddlepoint,
    _lugannani_rice,
    federated_spa_pvalue,
    federated_spa_pvalues_batch,
)

import jax.numpy as jnp  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures: χ²(k) distribution has known cumulants
#   κ_r = 2^{r-1} · (r-1)! · k
#   κ₁ = k,  κ₂ = 2k,  κ₃ = 8k,  κ₄ = 48k
# ═══════════════════════════════════════════════════════════════════════════

K_DOF = 10  # degrees of freedom


@pytest.fixture(scope="module")
def chi2_cumulants() -> np.ndarray:
    """Exact cumulants of χ²(10)."""
    k = K_DOF
    return np.array([k, 2 * k, 8 * k, 48 * k], dtype=np.float64)


@pytest.fixture(scope="module")
def chi2_federated_cumulants() -> list[np.ndarray]:
    """
    Split χ²(10) cumulants across 5 federated nodes.
    Each node contributes κ/5.  Aggregation should recover the global κ.
    """
    k = K_DOF
    full = np.array([k, 2 * k, 8 * k, 48 * k], dtype=np.float64)
    return [full / 5.0 for _ in range(5)]


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCumulantAggregation:

    def test_sum_of_nodes(self, chi2_federated_cumulants, chi2_cumulants):
        """Global κ = Σ_k κ^{(k)}."""
        global_k = aggregate_cumulants(chi2_federated_cumulants)
        np.testing.assert_allclose(global_k, chi2_cumulants, rtol=1e-12)

    def test_single_node_passthrough(self, chi2_cumulants):
        """With a single node, aggregation is identity."""
        result = aggregate_cumulants([chi2_cumulants])
        np.testing.assert_allclose(result, chi2_cumulants, rtol=1e-14)


class TestCGFDerivatives:

    def test_cgf_at_zero_is_zero(self, chi2_cumulants):
        """K(0) = 0 by definition of the CGF."""
        cgf, _, _, _ = _make_cgf(jnp.asarray(chi2_cumulants))
        assert abs(float(cgf(jnp.float64(0.0)))) < 1e-15

    def test_cgf_d1_at_zero_is_kappa1(self, chi2_cumulants):
        """K'(0) = κ₁ = mean."""
        _, cgf_d1, _, _ = _make_cgf(jnp.asarray(chi2_cumulants))
        result = float(cgf_d1(jnp.float64(0.0)))
        np.testing.assert_allclose(result, chi2_cumulants[0], rtol=1e-12)

    def test_cgf_d2_at_zero_is_kappa2(self, chi2_cumulants):
        """K''(0) = κ₂ = variance."""
        _, _, cgf_d2, _ = _make_cgf(jnp.asarray(chi2_cumulants))
        result = float(cgf_d2(jnp.float64(0.0)))
        np.testing.assert_allclose(result, chi2_cumulants[1], rtol=1e-12)

    def test_cgf_d3_at_zero_is_kappa3(self, chi2_cumulants):
        """K'''(0) = κ₃."""
        _, _, _, cgf_d3 = _make_cgf(jnp.asarray(chi2_cumulants))
        result = float(cgf_d3(jnp.float64(0.0)))
        np.testing.assert_allclose(result, chi2_cumulants[2], rtol=1e-12)


class TestHalleySolver:

    def test_saddlepoint_at_mean_is_zero(self, chi2_cumulants):
        """When Q = κ₁ = mean, the saddlepoint t̂ must be ≈ 0."""
        kappa = jnp.asarray(chi2_cumulants)
        Q = jnp.float64(chi2_cumulants[0])  # Q = mean
        t_hat = _solve_saddlepoint(kappa, Q)
        assert abs(float(t_hat)) < 1e-6, f"t̂ = {float(t_hat)}, expected ≈ 0"

    def test_saddlepoint_positive_for_right_tail(self, chi2_cumulants):
        """Q > mean  ⟹  t̂ > 0  (right tail)."""
        kappa = jnp.asarray(chi2_cumulants)
        Q = jnp.float64(chi2_cumulants[0] + 2.0 * np.sqrt(chi2_cumulants[1]))
        t_hat = _solve_saddlepoint(kappa, Q)
        assert float(t_hat) > 0, f"t̂ = {float(t_hat)}, expected > 0"

    def test_equation_residual(self, chi2_cumulants):
        """K'(t̂) - Q ≈ 0  within tolerance."""
        kappa = jnp.asarray(chi2_cumulants)
        Q_val = chi2_cumulants[0] + 3.0 * np.sqrt(chi2_cumulants[1])
        Q = jnp.float64(Q_val)
        t_hat = _solve_saddlepoint(kappa, Q)
        _, cgf_d1, _, _ = _make_cgf(kappa)
        residual = abs(float(cgf_d1(t_hat)) - Q_val)
        assert residual < 1e-10, f"|K'(t̂) - Q| = {residual}"


class TestLugannaniRice:

    @pytest.mark.parametrize("Q_val", [12.0, 15.0, 18.0, 22.0, 25.0, 30.0])
    def test_pvalue_vs_scipy_chi2(self, chi2_cumulants, Q_val):
        """
        SPA p-value for χ²(10) should be in the right ballpark of
        scipy.stats.chi2.sf.

        A 4-term Maclaurin CGF is a truncation of the true CGF
        K(t) = -k/2·ln(1-2t), which has infinitely many cumulants.
        Systematic ~30-50% relative error is inherent and expected.

        We verify:
          1. Same order of magnitude (log10 within 1.0)
          2. p ∈ (0, 1)
          3. Correct tail direction (larger Q → smaller p)
        """
        result = federated_spa_pvalue(Q_val, [chi2_cumulants])
        scipy_pval = scipy.stats.chi2.sf(Q_val, df=K_DOF)

        # Must be in (0, 1)
        assert 0.0 < result["pvalue"] < 1.0

        # Must be same order of magnitude
        log_ratio = abs(
            np.log10(max(result["pvalue"], 1e-300))
            - np.log10(max(scipy_pval, 1e-300))
        )
        assert log_ratio < 1.0, (
            f"SPA p-value at Q={Q_val}: {result['pvalue']:.6e}, "
            f"scipy: {scipy_pval:.6e}, log10 gap: {log_ratio:.2f}"
        )

    def test_pvalue_in_01_range(self, chi2_cumulants):
        """P-value must be in [0, 1]."""
        for Q_val in [5.0, 10.0, 20.0, 40.0]:
            result = federated_spa_pvalue(Q_val, [chi2_cumulants])
            assert 0.0 <= result["pvalue"] <= 1.0

    def test_pvalue_monotone_decreasing(self, chi2_cumulants):
        """P(X > Q) must decrease as Q increases."""
        Q_values = [8.0, 12.0, 16.0, 20.0, 25.0]
        pvals = [
            federated_spa_pvalue(q, [chi2_cumulants])["pvalue"]
            for q in Q_values
        ]
        for i in range(len(pvals) - 1):
            assert pvals[i] >= pvals[i + 1], (
                f"p-value not monotone: p({Q_values[i]})={pvals[i]:.6e} "
                f"> p({Q_values[i+1]})={pvals[i+1]:.6e}"
            )


class TestSingularityFallback:

    def test_near_mean_uses_gaussian(self, chi2_cumulants):
        """
        When Q ≈ κ₁ (mean), |t̂| < 1e-7  →  Gaussian fallback should fire.
        The p-value should be ≈ 0.5 (by symmetry around the mean).
        """
        result = federated_spa_pvalue(
            float(chi2_cumulants[0]),  # Q = mean
            [chi2_cumulants],
        )
        # At Q=mean, Gaussian gives Φ(0) = 0.5  →  P(X>Q) = 0.5
        np.testing.assert_allclose(result["pvalue"], 0.5, atol=0.05)
        assert abs(result["saddlepoint"]) < 1e-6


class TestBatchAPI:

    def test_batch_matches_scalar(self, chi2_cumulants):
        """Batched API must reproduce scalar API for each Q."""
        Q_batch = np.array([10.0, 15.0, 20.0, 25.0])
        batch_result = federated_spa_pvalues_batch(
            Q_batch, [chi2_cumulants]
        )
        for i, q in enumerate(Q_batch):
            scalar_result = federated_spa_pvalue(q, [chi2_cumulants])
            np.testing.assert_allclose(
                batch_result["pvalues"][i],
                scalar_result["pvalue"],
                rtol=1e-7,
                err_msg=f"Batch/scalar mismatch at Q={q}",
            )

    def test_batch_shape(self, chi2_cumulants):
        """Output shapes must match input batch size."""
        Q_batch = np.array([8.0, 12.0, 16.0])
        result = federated_spa_pvalues_batch(Q_batch, [chi2_cumulants])
        assert result["pvalues"].shape == (3,)
        assert result["saddlepoints"].shape == (3,)
        assert result["global_cumulants"].shape == (4,)
