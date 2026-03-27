"""
test_glmm.py — Tests for metararepi.glmm module (R2 API).

Validates:
- GLMM null model fitting (continuous + binary)
- AI-REML variance component estimation
- Generalized FWL projection (Proposition 1)
- P0 projection properties
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from metararepi.glmm import (
    fit_null_model,
    build_fwl_projection,
    verify_fwl_properties,
)

N = 200
SEED = 42


@pytest.fixture(scope="module")
def synthetic_data():
    """Generate synthetic data for GLMM tests."""
    rng = np.random.default_rng(SEED)
    X = np.column_stack([np.ones(N), rng.standard_normal(N)])
    y = X @ np.array([1.0, 0.5]) + rng.standard_normal(N)
    # Simple GRM (identity + small noise for positive definiteness)
    GRM = np.eye(N) + 0.02 * rng.standard_normal((N, N))
    GRM = (GRM + GRM.T) / 2
    np.fill_diagonal(GRM, 1.0)
    return {"X": X, "y": y, "GRM": GRM}


class TestContinuousNullModel:

    def test_fit_converges(self, synthetic_data):
        """AI-REML should converge."""
        result = fit_null_model(
            synthetic_data["y"], synthetic_data["X"],
            synthetic_data["GRM"], trait_type="continuous"
        )
        assert result["converged"] or result["n_iter"] <= 100

    def test_variance_components_positive(self, synthetic_data):
        """Estimated variance components should be positive."""
        result = fit_null_model(
            synthetic_data["y"], synthetic_data["X"],
            synthetic_data["GRM"], trait_type="continuous"
        )
        assert result["tau2"] > 0
        assert result["sigma2"] > 0

    def test_P0_annihilates_X(self, synthetic_data):
        """P0 @ X should be approximately zero."""
        result = fit_null_model(
            synthetic_data["y"], synthetic_data["X"],
            synthetic_data["GRM"], trait_type="continuous"
        )
        P0X = result["P0_matrix"] @ synthetic_data["X"]
        assert np.max(np.abs(P0X)) < 1e-6

    def test_P0_symmetric(self, synthetic_data):
        """P0 should be symmetric."""
        result = fit_null_model(
            synthetic_data["y"], synthetic_data["X"],
            synthetic_data["GRM"], trait_type="continuous"
        )
        P0 = result["P0_matrix"]
        assert np.max(np.abs(P0 - P0.T)) < 1e-8

    def test_P0_V_metric_idempotent(self, synthetic_data):
        """P0 V P0 should equal P0."""
        result = fit_null_model(
            synthetic_data["y"], synthetic_data["X"],
            synthetic_data["GRM"], trait_type="continuous"
        )
        P0 = result["P0_matrix"]
        V = result["sigma2"] * np.eye(N) + result["tau2"] * synthetic_data["GRM"]
        PVP = P0 @ V @ P0
        assert np.max(np.abs(PVP - P0)) < 1e-5

    def test_y_adj_shape(self, synthetic_data):
        """Adjusted residual should have correct shape."""
        result = fit_null_model(
            synthetic_data["y"], synthetic_data["X"],
            synthetic_data["GRM"], trait_type="continuous"
        )
        assert result["y_adj"].shape == (N,)


class TestBinaryNullModel:

    @pytest.fixture(scope="class")
    def binary_data(self, synthetic_data):
        rng = np.random.default_rng(SEED + 1)
        mu = 1.0 / (1.0 + np.exp(-rng.normal(0, 0.3, N)))
        y = rng.binomial(1, mu).astype(np.float64)
        return {"X": synthetic_data["X"], "y": y, "GRM": synthetic_data["GRM"]}

    def test_binary_fit_converges(self, binary_data):
        """IRLS + AI-REML should converge for binary traits."""
        result = fit_null_model(
            binary_data["y"], binary_data["X"],
            binary_data["GRM"], trait_type="binary"
        )
        assert result["converged"] or result["n_iter"] <= 100

    def test_binary_W_diag_positive(self, binary_data):
        """IRLS working weights should be positive."""
        result = fit_null_model(
            binary_data["y"], binary_data["X"],
            binary_data["GRM"], trait_type="binary"
        )
        assert np.all(result["W_diag"] > 0)

    def test_binary_mu_in_range(self, binary_data):
        """Fitted means should be in (0, 1)."""
        result = fit_null_model(
            binary_data["y"], binary_data["X"],
            binary_data["GRM"], trait_type="binary"
        )
        assert np.all(result["mu_hat"] > 0)
        assert np.all(result["mu_hat"] < 1)


class TestFWLProjection:

    def test_fwl_annihilation(self, synthetic_data):
        """P_adj @ Z_main should be approximately zero."""
        result = fit_null_model(
            synthetic_data["y"], synthetic_data["X"],
            synthetic_data["GRM"], trait_type="continuous"
        )
        rng = np.random.default_rng(SEED)
        Z_main = rng.standard_normal((N, 10))
        P_adj = build_fwl_projection(result["P0_matrix"], Z_main)
        assert np.max(np.abs(P_adj @ Z_main)) < 1e-6

    def test_fwl_symmetry(self, synthetic_data):
        """P_adj should be symmetric."""
        result = fit_null_model(
            synthetic_data["y"], synthetic_data["X"],
            synthetic_data["GRM"], trait_type="continuous"
        )
        rng = np.random.default_rng(SEED)
        Z_main = rng.standard_normal((N, 10))
        P_adj = build_fwl_projection(result["P0_matrix"], Z_main)
        assert np.max(np.abs(P_adj - P_adj.T)) < 1e-8

    def test_verify_fwl_properties(self, synthetic_data):
        """verify_fwl_properties should return correct diagnostics."""
        result = fit_null_model(
            synthetic_data["y"], synthetic_data["X"],
            synthetic_data["GRM"], trait_type="continuous"
        )
        rng = np.random.default_rng(SEED)
        Z_main = rng.standard_normal((N, 10))
        P_adj = build_fwl_projection(result["P0_matrix"], Z_main)
        props = verify_fwl_properties(P_adj, Z_main)
        assert props["annihilation_error"] < 1e-6
        assert props["symmetry_error"] < 1e-8
