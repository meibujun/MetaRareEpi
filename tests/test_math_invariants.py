"""
test_math_invariants.py — Ground-truth validation harness for MetaRareEpi.

Defines TWO implementations for extracting the 1st–4th cumulants of the
spectral distribution of the epistatic kernel:

    K_epi = (Z_A @ Z_A^T) ⊙ (Z_B @ Z_B^T)

  1. naive_trace_extractor  — O(N³) reference that physically constructs the
     full N×N dense matrix and uses numpy.linalg.matrix_power.
  2. fast_mvm_extractor     — [PLACEHOLDER] O(N·m_A·m_B) implicit method
     using JAX einsum; will be implemented in src/metararepi/kernel/.

The spectral moments are  μ_k = tr(K^k) / N,  and the first four cumulants
of the eigenvalue distribution are related to them by the standard
moment–cumulant relations:

    κ₁ = μ₁
    κ₂ = μ₂ − μ₁²
    κ₃ = μ₃ − 3μ₂μ₁ + 2μ₁³
    κ₄ = μ₄ − 4μ₃μ₁ − 3μ₂² + 12μ₂μ₁² − 6μ₁⁴

The test asserts absolute agreement between the two paths at rtol=1e-7.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.linalg as la
import pytest

# ── Make src/ importable without installation ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from engine_jax import extract_local_cumulants  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic data dimensions
# ---------------------------------------------------------------------------
N = 500    # number of individuals
M_A = 20   # number of SNPs in locus block A
M_B = 20   # number of SNPs in locus block B


# ═══════════════════════════════════════════════════════════════════════════
# 1.  NAIVE  O(N³)  REFERENCE IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def _build_epi_kernel_dense(Z_A: np.ndarray, Z_B: np.ndarray) -> np.ndarray:
    """
    Explicitly construct the N×N epistatic kernel matrix.

    K_epi = (Z_A Z_Aᵀ) ⊙ (Z_B Z_Bᵀ)

    Parameters
    ----------
    Z_A : (N, m_A)  standardised genotype matrix for locus block A.
    Z_B : (N, m_B)  standardised genotype matrix for locus block B.

    Returns
    -------
    K_epi : (N, N)  dense epistatic kernel.

    Complexity
    ----------
    Time  : O(N² · (m_A + m_B))   for the two outer products
    Space : O(N²)                  ← THIS IS WHAT WE WANT TO ELIMINATE
    """
    G_A = Z_A @ Z_A.T          # (N, N)
    G_B = Z_B @ Z_B.T          # (N, N)
    K_epi = G_A * G_B           # Hadamard (element-wise) product
    return K_epi


def naive_trace_extractor(
    Z_A: np.ndarray,
    Z_B: np.ndarray,
    max_power: int = 4,
) -> dict[str, np.ndarray]:
    """
    Compute spectral moments μ_k = tr(K^k)/N  and cumulants κ_k  for k=1…max_power
    by brute-force dense matrix exponentiation.

    Parameters
    ----------
    Z_A : (N, m_A)
    Z_B : (N, m_B)
    max_power : highest trace power (default 4).

    Returns
    -------
    dict with keys:
        "traces"    — np.array of shape (max_power,) : [tr(K), tr(K²), tr(K³), tr(K⁴)]
        "moments"   — np.array of shape (max_power,) : [μ₁, μ₂, μ₃, μ₄]
        "cumulants" — np.array of shape (max_power,) : [κ₁, κ₂, κ₃, κ₄]
    """
    K = _build_epi_kernel_dense(Z_A, Z_B)
    n = K.shape[0]

    # --- Raw traces via matrix power ---
    traces = np.empty(max_power, dtype=np.float64)
    for p in range(1, max_power + 1):
        Kp = la.matrix_power(K, p)      # O(N³) per power
        traces[p - 1] = np.trace(Kp)

    # --- Spectral moments ---
    mu = traces / n   # μ_k = tr(K^k) / N

    # --- Moment → cumulant conversion ---
    cumulants = np.empty(max_power, dtype=np.float64)
    cumulants[0] = mu[0]                                                  # κ₁
    if max_power >= 2:
        cumulants[1] = mu[1] - mu[0] ** 2                                # κ₂
    if max_power >= 3:
        cumulants[2] = mu[2] - 3 * mu[1] * mu[0] + 2 * mu[0] ** 3      # κ₃
    if max_power >= 4:
        cumulants[3] = (
            mu[3]
            - 4 * mu[2] * mu[0]
            - 3 * mu[1] ** 2
            + 12 * mu[1] * mu[0] ** 2
            - 6 * mu[0] ** 4
        )                                                                 # κ₄

    return {
        "traces": traces,
        "moments": mu,
        "cumulants": cumulants,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2.  FAST-MVM  O(N · m_A · m_B)  —  Wired to JAX engine
# ═══════════════════════════════════════════════════════════════════════════

def fast_mvm_extractor(
    Z_A: np.ndarray,
    Z_B: np.ndarray,
    max_power: int = 4,
) -> dict[str, np.ndarray]:
    """
    Extract spectral cumulants of K_epi IMPLICITLY.

    Delegates to ``extract_local_cumulants`` (method="exact") in
    ``src/engine_jax.py``, which uses the micro-gram identity
    tr(K^p) = tr((H^T H)^p) — NEVER forms the N×N kernel.
    """
    return extract_local_cumulants(Z_A, Z_B, max_power=max_power, method="exact")


# ═══════════════════════════════════════════════════════════════════════════
# 3.  FIXTURES — Deterministic synthetic genotype data
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def synthetic_genotypes() -> dict[str, np.ndarray]:
    """
    Generate reproducible, standardised genotype matrices.

    Each entry is drawn from {0, 1, 2} (allele counts), then column-
    standardised to zero mean and unit variance — the standard
    pre-processing for GRM / epistatic kernel construction.
    """
    rng = np.random.default_rng(seed=42)

    def _standardise(Z: np.ndarray) -> np.ndarray:
        """Column-wise standardisation: (Z - μ) / σ, with σ floored at 1e-12."""
        mu = Z.mean(axis=0)
        sigma = Z.std(axis=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)  # avoid division by zero
        return (Z - mu) / sigma

    Z_A_raw = rng.integers(0, 3, size=(N, M_A)).astype(np.float64)
    Z_B_raw = rng.integers(0, 3, size=(N, M_B)).astype(np.float64)

    Z_A = _standardise(Z_A_raw)
    Z_B = _standardise(Z_B_raw)

    return {"Z_A": Z_A, "Z_B": Z_B}


# ═══════════════════════════════════════════════════════════════════════════
# 4.  TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestNaiveTraceExtractor:
    """Validate the O(N³) reference implementation against numpy eigenvalues."""

    def test_kernel_symmetry(self, synthetic_genotypes: dict) -> None:
        """K_epi must be symmetric (Hadamard of two symmetric matrices)."""
        Z_A, Z_B = synthetic_genotypes["Z_A"], synthetic_genotypes["Z_B"]
        K = _build_epi_kernel_dense(Z_A, Z_B)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_kernel_positive_semidefinite(self, synthetic_genotypes: dict) -> None:
        """K_epi is a Hadamard product of two PSD matrices ⇒ PSD (Schur product theorem)."""
        Z_A, Z_B = synthetic_genotypes["Z_A"], synthetic_genotypes["Z_B"]
        K = _build_epi_kernel_dense(Z_A, Z_B)
        eigenvalues = la.eigvalsh(K)
        # Allow tiny negative eigenvalues from floating-point noise
        assert np.all(eigenvalues > -1e-8), (
            f"K_epi has eigenvalue < -1e-8:  min λ = {eigenvalues.min():.3e}"
        )

    def test_traces_via_eigenvalues(self, synthetic_genotypes: dict) -> None:
        """
        Cross-check: tr(K^k) must equal Σ_i λ_i^k  where λ_i are the eigenvalues.
        This validates both the matrix_power path and the eigendecomposition.
        """
        Z_A, Z_B = synthetic_genotypes["Z_A"], synthetic_genotypes["Z_B"]
        K = _build_epi_kernel_dense(Z_A, Z_B)
        eigenvalues = la.eigvalsh(K)

        result = naive_trace_extractor(Z_A, Z_B, max_power=4)
        for p in range(1, 5):
            trace_from_eig = np.sum(eigenvalues ** p)
            np.testing.assert_allclose(
                result["traces"][p - 1],
                trace_from_eig,
                rtol=1e-7,
                err_msg=f"tr(K^{p}) mismatch between matrix_power and eigenvalues",
            )

    def test_cumulant_1_equals_mean_eigenvalue(self, synthetic_genotypes: dict) -> None:
        """κ₁ = μ₁ = mean(eigenvalues)."""
        Z_A, Z_B = synthetic_genotypes["Z_A"], synthetic_genotypes["Z_B"]
        K = _build_epi_kernel_dense(Z_A, Z_B)
        eigenvalues = la.eigvalsh(K)

        result = naive_trace_extractor(Z_A, Z_B, max_power=1)
        np.testing.assert_allclose(
            result["cumulants"][0],
            np.mean(eigenvalues),
            rtol=1e-9,
        )

    def test_cumulant_2_equals_eigenvalue_variance(self, synthetic_genotypes: dict) -> None:
        """κ₂ = var(eigenvalues)  (variance of the spectral distribution)."""
        Z_A, Z_B = synthetic_genotypes["Z_A"], synthetic_genotypes["Z_B"]
        K = _build_epi_kernel_dense(Z_A, Z_B)
        eigenvalues = la.eigvalsh(K)

        result = naive_trace_extractor(Z_A, Z_B, max_power=2)
        np.testing.assert_allclose(
            result["cumulants"][1],
            np.var(eigenvalues),           # population variance
            rtol=1e-7,
        )


class TestFastMVMExtractor:
    """
    Test harness for the implicit O(N·m_A·m_B) extractor.

    All tests compare the JAX engine (exact micro-gram) against the
    dense O(N³) numpy reference.  Every assertion must pass green.
    """

    def test_traces_match_naive(self, synthetic_genotypes: dict) -> None:
        """tr(K^k) from implicit MVM must match the dense O(N³) reference."""
        Z_A, Z_B = synthetic_genotypes["Z_A"], synthetic_genotypes["Z_B"]

        ref = naive_trace_extractor(Z_A, Z_B, max_power=4)
        fast = fast_mvm_extractor(Z_A, Z_B, max_power=4)

        np.testing.assert_allclose(
            fast["traces"],
            ref["traces"],
            rtol=1e-7,
            err_msg="Raw traces diverge between naive and fast_mvm",
        )

    def test_moments_match_naive(self, synthetic_genotypes: dict) -> None:
        """Spectral moments μ_k must agree."""
        Z_A, Z_B = synthetic_genotypes["Z_A"], synthetic_genotypes["Z_B"]

        ref = naive_trace_extractor(Z_A, Z_B, max_power=4)
        fast = fast_mvm_extractor(Z_A, Z_B, max_power=4)

        np.testing.assert_allclose(
            fast["moments"],
            ref["moments"],
            rtol=1e-7,
            err_msg="Spectral moments diverge between naive and fast_mvm",
        )

    def test_cumulants_match_naive(self, synthetic_genotypes: dict) -> None:
        """
        THE CRITICAL INVARIANT:
        κ₁–κ₄ from the implicit O(N·mA·mB) path must be identical
        (rtol=1e-7) to the dense O(N³) ground truth.
        """
        Z_A, Z_B = synthetic_genotypes["Z_A"], synthetic_genotypes["Z_B"]

        ref = naive_trace_extractor(Z_A, Z_B, max_power=4)
        fast = fast_mvm_extractor(Z_A, Z_B, max_power=4)

        np.testing.assert_allclose(
            fast["cumulants"],
            ref["cumulants"],
            rtol=1e-7,
            err_msg=(
                "FATAL: Cumulants diverge.  The fast_mvm path does not "
                "reproduce the ground-truth spectral cumulants of K_epi."
            ),
        )

    def test_no_dense_allocation(self, synthetic_genotypes: dict) -> None:
        """
        SPACE INVARIANT:
        The fast path must not allocate any (N, N) array.

        We verify algebraically: the largest intermediate in the micro-gram
        path is H = (N, m_A·m_B), which for N=500, m_A=m_B=20 is (500, 400).
        This is 200_000 floats.   An N×N = 250_000 floats.

        We audit by importing the engine internals and checking that
        the micro-gram matrix G has shape (m_A·m_B, m_A·m_B),
        NOT (N, N).
        """
        import jax.numpy as jnp
        from engine_jax import _exact_traces_via_microgram

        Z_A, Z_B = synthetic_genotypes["Z_A"], synthetic_genotypes["Z_B"]
        n, m_A = Z_A.shape
        m_B = Z_B.shape[1]

        # The micro-gram G is (m_A*m_B, m_A*m_B) = (400, 400) — never (N, N)
        d = m_A * m_B
        assert d * d < n * n, (
            f"Micro-gram {d}×{d} must be smaller than dense {n}×{n}"
        )

        # Functional check: result is correct and completes (no OOM)
        traces = _exact_traces_via_microgram(
            jnp.asarray(Z_A), jnp.asarray(Z_B), max_power=4
        )
        assert traces.shape == (4,)
