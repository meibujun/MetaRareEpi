"""
Shared pytest fixtures for MetaRareEpi test suite.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic data dimensions (shared across test modules)
# ---------------------------------------------------------------------------
N = 500
M_A = 20
M_B = 20


def _standardise(Z: np.ndarray) -> np.ndarray:
    """Column-wise standardisation: (Z - μ) / σ, with σ floored at 1e-12."""
    mu = Z.mean(axis=0)
    sigma = Z.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (Z - mu) / sigma


@pytest.fixture(scope="session")
def synthetic_genotypes() -> dict[str, np.ndarray]:
    """
    Session-scoped deterministic synthetic genotype matrices.

    Z_A: (500, 20), Z_B: (500, 20) — standardised, seed=42.
    """
    rng = np.random.default_rng(seed=42)
    Z_A = _standardise(rng.integers(0, 3, size=(N, M_A)).astype(np.float64))
    Z_B = _standardise(rng.integers(0, 3, size=(N, M_B)).astype(np.float64))
    return {"Z_A": Z_A, "Z_B": Z_B}
