"""
test_weighting.py — Tests for metararepi.weighting module.

Validates annotation scoring backends and weight application.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from metararepi.weighting import (  # noqa: E402
    UniformWeighter,
    BetaWeighter,
    CADDScorer,
    AlphaMissenseScorer,
    apply_weights,
    compute_weighted_features,
)

SEED = 42


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.default_rng(SEED)
    N, m = 200, 30
    G = rng.integers(0, 3, size=(N, m)).astype(np.float64)
    mafs = rng.uniform(0.001, 0.01, size=m)
    return {"G": G, "mafs": mafs, "N": N, "m": m}


class TestUniformWeighter:

    def test_all_ones(self, synthetic_data):
        w = UniformWeighter()
        scores = w.score(synthetic_data["mafs"])
        np.testing.assert_array_equal(scores, 1.0)

    def test_weight_matrix_diagonal(self, synthetic_data):
        w = UniformWeighter()
        W = w.compute_weight_matrix(synthetic_data["mafs"])
        assert W.shape == (synthetic_data["m"], synthetic_data["m"])
        np.testing.assert_array_equal(np.diag(W), 1.0)


class TestBetaWeighter:

    def test_positive_weights(self, synthetic_data):
        w = BetaWeighter()
        scores = w.score(synthetic_data["mafs"])
        assert np.all(scores > 0)

    def test_rare_upweighted(self):
        """Rare variants (low MAF) should get higher weights."""
        w = BetaWeighter(a1=1.0, a2=25.0)
        rare_mafs = np.array([0.001, 0.005, 0.01, 0.05, 0.1])
        scores = w.score(rare_mafs)
        # MAF=0.001 should have higher weight than MAF=0.1
        assert scores[0] > scores[-1]


class TestCADDScorer:

    def test_with_cadd_scores(self, synthetic_data):
        scorer = CADDScorer()
        cadd_scores = np.random.default_rng(42).uniform(5, 35, size=synthetic_data["m"])
        weights = scorer.score(synthetic_data["mafs"], cadd_scores=cadd_scores)
        assert np.all(weights >= 0)
        assert np.all(weights <= 1)

    def test_fallback_without_cadd(self, synthetic_data):
        """Without CADD scores, falls back to Beta(MAF) weighting."""
        scorer = CADDScorer()
        weights = scorer.score(synthetic_data["mafs"])
        assert np.all(weights > 0)

    def test_higher_cadd_higher_weight(self):
        scorer = CADDScorer()
        mafs = np.array([0.005, 0.005])
        w = scorer.score(mafs, cadd_scores=np.array([5.0, 30.0]))
        assert w[1] > w[0]  # higher CADD → higher weight


class TestAlphaMissenseScorer:

    def test_with_am_scores(self, synthetic_data):
        scorer = AlphaMissenseScorer()
        am_scores = np.random.default_rng(42).uniform(0, 1, size=synthetic_data["m"])
        weights = scorer.score(synthetic_data["mafs"], alphamissense_scores=am_scores)
        np.testing.assert_array_equal(weights, am_scores)

    def test_fallback(self, synthetic_data):
        scorer = AlphaMissenseScorer()
        weights = scorer.score(synthetic_data["mafs"])
        assert np.all(weights > 0)


class TestApplyWeights:

    def test_output_shape(self, synthetic_data):
        weights = np.ones(synthetic_data["m"])
        Z = apply_weights(synthetic_data["G"], weights)
        assert Z.shape == (synthetic_data["N"], synthetic_data["m"])

    def test_standardised_mean_zero(self, synthetic_data):
        weights = np.ones(synthetic_data["m"])
        Z = apply_weights(synthetic_data["G"], weights, standardise=True)
        np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-10)

    def test_unstandardised(self, synthetic_data):
        weights = np.array([2.0] * synthetic_data["m"])
        Z = apply_weights(synthetic_data["G"], weights, standardise=False)
        np.testing.assert_allclose(Z, synthetic_data["G"] * 2.0)


class TestComputeWeightedFeatures:

    def test_pipeline(self, synthetic_data):
        G_A = synthetic_data["G"][:, :15]
        G_B = synthetic_data["G"][:, 15:]
        mafs_A = synthetic_data["mafs"][:15]
        mafs_B = synthetic_data["mafs"][15:]
        Z_A, Z_B = compute_weighted_features(G_A, G_B, mafs_A, mafs_B)
        assert Z_A.shape == (synthetic_data["N"], 15)
        assert Z_B.shape == (synthetic_data["N"], 15)
        # Standardised
        np.testing.assert_allclose(Z_A.mean(axis=0), 0.0, atol=1e-10)
