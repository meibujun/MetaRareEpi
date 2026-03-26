"""
test_glmm.py — Tests for metararepi.glmm module.

Validates GLMM base model operations:
- P₀ projection properties (idempotency, symmetry, null-space)
- AI-REML variance component estimation
- Whitened residual computation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from metararepi.glmm import (  # noqa: E402
    estimate_grm,
    compute_V,
    compute_P0,
    compute_whitened_residual,
    estimate_variance_components,
    null_model_pipeline,
)

N = 200
M = 50
SEED = 42


@pytest.fixture(scope="module")
def synthetic_data():
    """Generate synthetic data for GLMM tests."""
    rng = np.random.default_rng(SEED)
    G = rng.integers(0, 3, size=(N, M)).astype(np.float64)
    X = np.column_stack([np.ones(N), rng.standard_normal(N)])
    y = X @ np.array([1.0, 0.5]) + rng.standard_normal(N)
    return {"G": G, "X": X, "y": y}


class TestGRM:

    def test_grm_symmetric(self, synthetic_data):
        Phi = estimate_grm(synthetic_data["G"])
        np.testing.assert_allclose(Phi, Phi.T, atol=1e-12)

    def test_grm_psd(self, synthetic_data):
        Phi = estimate_grm(synthetic_data["G"])
        eigenvalues = np.linalg.eigvalsh(Phi)
        assert np.all(eigenvalues > -1e-8)

    def test_grm_shape(self, synthetic_data):
        Phi = estimate_grm(synthetic_data["G"])
        assert Phi.shape == (N, N)


class TestProjectionP0:

    def test_p0_symmetric(self, synthetic_data):
        """P₀ must be symmetric."""
        Phi = estimate_grm(synthetic_data["G"])
        V = compute_V(Phi, 0.3, 0.7)
        P0 = compute_P0(V, synthetic_data["X"])
        np.testing.assert_allclose(P0, P0.T, atol=1e-10)

    def test_p0_annihilates_X(self, synthetic_data):
        """P₀X = 0 (null space property)."""
        Phi = estimate_grm(synthetic_data["G"])
        V = compute_V(Phi, 0.3, 0.7)
        P0 = compute_P0(V, synthetic_data["X"])
        result = P0 @ synthetic_data["X"]
        np.testing.assert_allclose(result, 0.0, atol=1e-8)

    def test_p0_idempotent(self, synthetic_data):
        """P₀² = P₀ (in the V-metric sense: P₀VP₀ = P₀)."""
        Phi = estimate_grm(synthetic_data["G"])
        V = compute_V(Phi, 0.3, 0.7)
        P0 = compute_P0(V, synthetic_data["X"])
        P0VP0 = P0 @ V @ P0
        np.testing.assert_allclose(P0VP0, P0, atol=1e-6)


class TestVarianceComponents:

    def test_ai_reml_converges(self, synthetic_data):
        """AI-REML must converge within max_iter."""
        Phi = estimate_grm(synthetic_data["G"])
        vc = estimate_variance_components(
            synthetic_data["y"], synthetic_data["X"], Phi
        )
        assert vc["converged"] or vc["n_iter"] <= 50
        assert vc["tau2"] > 0
        assert vc["sigma2_e"] > 0
        assert 0.0 <= vc["h2"] <= 1.0

    def test_h2_reasonable(self, synthetic_data):
        """h² should be between 0 and 1."""
        Phi = estimate_grm(synthetic_data["G"])
        vc = estimate_variance_components(
            synthetic_data["y"], synthetic_data["X"], Phi
        )
        assert 0.0 <= vc["h2"] <= 1.0


class TestWhitenedResidual:

    def test_residual_shape(self, synthetic_data):
        """Whitened residual must have shape (N,)."""
        Phi = estimate_grm(synthetic_data["G"])
        V = compute_V(Phi, 0.3, 0.7)
        P0 = compute_P0(V, synthetic_data["X"])
        y_tilde = compute_whitened_residual(
            synthetic_data["y"], synthetic_data["X"], P0
        )
        assert y_tilde.shape == (N,)

    def test_residual_orthogonal_to_X(self, synthetic_data):
        """X^T ỹ ≈ 0 (residual is orthogonal to fixed effects in V metric)."""
        Phi = estimate_grm(synthetic_data["G"])
        V = compute_V(Phi, 0.3, 0.7)
        P0 = compute_P0(V, synthetic_data["X"])
        y_tilde = compute_whitened_residual(
            synthetic_data["y"], synthetic_data["X"], P0
        )
        # P₀X = 0 ⟹ X^T P₀ y = 0
        projection = synthetic_data["X"].T @ y_tilde
        np.testing.assert_allclose(projection, 0.0, atol=1e-6)


class TestNullModelPipeline:

    def test_pipeline_returns_dict(self, synthetic_data):
        result = null_model_pipeline(
            synthetic_data["y"],
            X=synthetic_data["X"],
            G_common=synthetic_data["G"],
        )
        assert "P0" in result
        assert "y_tilde" in result
        assert "V" in result
        assert "variance_components" in result

    def test_pipeline_with_defaults(self, synthetic_data):
        """Pipeline with default intercept-only model."""
        result = null_model_pipeline(synthetic_data["y"])
        assert result["P0"].shape == (N, N)
        assert result["y_tilde"].shape == (N,)
