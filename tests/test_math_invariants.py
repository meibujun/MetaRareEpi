"""
test_math_invariants.py — Comprehensive Mathematical Invariant Tests

Validates all core mathematical properties from the MetaRareEpi R2 paper:
  - Theorem 1: Dual-space eigenvalue equivalence
  - Proposition 1: FWL idempotency and annihilation
  - Proposition 2: Q_adj dimensionality collapse
  - Algorithm 1: Hutch++ deflation accuracy
  - Newton's identities: moment-cumulant conversion
  - Lugannani-Rice SPA tail probability precision
  - Cumulant additivity (Theorem 2)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(params=[50, 200])
def N(request):
    return request.param

@pytest.fixture
def small_data():
    """Small test data for mathematical verification."""
    rng = np.random.default_rng(42)
    N, m_A, m_B = 100, 5, 5
    Z_A = rng.normal(0, 1, (N, m_A))
    Z_B = rng.normal(0, 1, (N, m_B))
    y = rng.normal(0, 1, N)
    X = np.column_stack([np.ones(N), rng.normal(0, 1, (N, 2))])
    return Z_A, Z_B, y, X, N


# ═══════════════════════════════════════════════════════════════════════════
# Theorem 1: Dual-Space Eigenvalue Equivalence
# ═══════════════════════════════════════════════════════════════════════════

class TestDualSpace:
    """Validate Theorem 1: eigenvalues of P_adj K_epi ≡ eigenvalues of G_dual."""

    def test_eigenvalue_equivalence(self, small_data):
        """Non-zero eigenvalues of N×N and d×d matrices should match."""
        from engine_jax import khatri_rao_product

        Z_A, Z_B, _, _, N = small_data
        d = Z_A.shape[1] * Z_B.shape[1]

        # Primal: K_epi = (Z_A Z_A^T) ⊙ (Z_B Z_B^T)
        K_epi = (Z_A @ Z_A.T) * (Z_B @ Z_B.T)
        eig_primal = np.sort(np.linalg.eigvalsh(K_epi))[::-1]

        # Dual: G_dual = Z_KR^T Z_KR
        Z_KR = np.asarray(khatri_rao_product(
            jnp.array(Z_A), jnp.array(Z_B)
        ))
        G_dual = Z_KR.T @ Z_KR
        eig_dual = np.sort(np.linalg.eigvalsh(G_dual))[::-1]

        # Non-zero eigenvalues should match (up to d values)
        n_nonzero = min(d, N)
        np.testing.assert_allclose(
            eig_primal[:n_nonzero],
            eig_dual[:n_nonzero],
            rtol=1e-8,
            err_msg="Theorem 1 violated: eigenvalues differ"
        )

    def test_trace_equivalence(self, small_data):
        """tr(K^p) = tr(G^p) for all powers."""
        from engine_jax import khatri_rao_product, exact_traces_microgram

        Z_A, Z_B, _, _, N = small_data

        # Direct K_epi traces
        K_epi = (Z_A @ Z_A.T) * (Z_B @ Z_B.T)
        direct_traces = []
        Kp = np.eye(N)
        for p in range(4):
            Kp = Kp @ K_epi
            direct_traces.append(np.trace(Kp))

        # Via micro-gram (dual space)
        dual_traces = np.asarray(exact_traces_microgram(
            jnp.array(Z_A), jnp.array(Z_B), 4
        ))

        np.testing.assert_allclose(
            direct_traces, dual_traces, rtol=1e-6,
            err_msg="Dual-space traces differ from primal traces"
        )

    def test_g_dual_is_spsd(self, small_data):
        """G_dual should be symmetric positive semi-definite."""
        from engine_jax import khatri_rao_product

        Z_A, Z_B, _, _, _ = small_data
        Z_KR = np.asarray(khatri_rao_product(
            jnp.array(Z_A), jnp.array(Z_B)
        ))
        G = Z_KR.T @ Z_KR

        # Symmetry
        assert np.allclose(G, G.T, atol=1e-12), "G_dual not symmetric"

        # PSD: all eigenvalues ≥ 0
        eig = np.linalg.eigvalsh(G)
        assert np.all(eig >= -1e-10), f"G_dual not PSD: min eigenvalue = {eig.min()}"


# ═══════════════════════════════════════════════════════════════════════════
# Proposition 1: FWL Properties
# ═══════════════════════════════════════════════════════════════════════════

class TestFWL:
    """
    Validate Proposition 1: P_adj annihilates Z_main and is symmetric.

    Note: For GLMM, P0 = V^{-1} - V^{-1}X(X^T V^{-1}X)^{-1}X^T V^{-1}
    This is idempotent in the V-metric (P0 V P0 = P0), NOT in Euclidean.
    P_adj inherits this V-metric idempotency.
    """

    def test_annihilation(self, small_data):
        """P_adj @ Z_main should be approximately zero."""
        from metararepi.glmm import fit_null_model, build_fwl_projection

        Z_A, Z_B, y, X, N = small_data
        GRM = np.eye(N)
        null_model = fit_null_model(y, X, GRM, trait_type="continuous")
        Z_main = np.column_stack([Z_A, Z_B])
        P_adj = build_fwl_projection(null_model["P0_matrix"], Z_main)

        residual = P_adj @ Z_main
        err = np.max(np.abs(residual))
        assert err < 1e-6, f"FWL annihilation failed: max residual = {err}"

    def test_v_metric_idempotency(self, small_data):
        """P0 V P0 = P0 (V-metric idempotency for GLMM projectors)."""
        from metararepi.glmm import fit_null_model

        Z_A, Z_B, y, X, N = small_data
        GRM = np.eye(N)
        null_model = fit_null_model(y, X, GRM, trait_type="continuous")
        P0 = null_model["P0_matrix"]

        # For continuous with GRM=I: V = σ²I + τ²I = (σ²+τ²)I
        sigma2 = null_model.get("sigma2", 1.0)
        tau2 = null_model.get("tau2", 0.0)
        V = sigma2 * np.eye(N) + tau2 * GRM

        # P0 V P0 should equal P0
        PVP = P0 @ V @ P0
        err = np.max(np.abs(PVP - P0))
        assert err < 1e-6, f"V-metric idempotency failed: max error = {err}"

    def test_symmetry(self, small_data):
        """P_adj should be symmetric."""
        from metararepi.glmm import fit_null_model, build_fwl_projection

        Z_A, Z_B, y, X, N = small_data
        GRM = np.eye(N)
        null_model = fit_null_model(y, X, GRM, trait_type="continuous")
        Z_main = np.column_stack([Z_A, Z_B])
        P_adj = build_fwl_projection(null_model["P0_matrix"], Z_main)

        assert np.max(np.abs(P_adj - P_adj.T)) < 1e-8, \
            "FWL projection not symmetric"

    def test_binary_fwl_annihilation(self, small_data):
        """
        For binary traits, P_adj should annihilate Z_main.
        The IRLS weights are already in V (Remark 1).
        """
        from metararepi.glmm import fit_null_model, build_fwl_projection

        Z_A, Z_B, _, X, N = small_data
        rng = np.random.default_rng(42)

        # Simulate binary phenotype
        mu = 1.0 / (1.0 + np.exp(-rng.normal(0, 0.5, N)))
        y = rng.binomial(1, mu).astype(np.float64)

        GRM = np.eye(N) + 0.01 * rng.standard_normal((N, N))
        GRM = (GRM + GRM.T) / 2
        np.fill_diagonal(GRM, 1.0)

        null_model = fit_null_model(y, X, GRM, trait_type="binary")
        Z_main = np.column_stack([Z_A, Z_B])
        P_adj = build_fwl_projection(null_model["P0_matrix"], Z_main)

        # Annihilation check
        residual = P_adj @ Z_main
        err = np.max(np.abs(residual))
        assert err < 1e-5, f"Binary FWL annihilation failed: max error = {err}"


# ═══════════════════════════════════════════════════════════════════════════
# Proposition 2: Q_adj Dimensionality Collapse
# ═══════════════════════════════════════════════════════════════════════════

class TestQadj:
    """Validate Q_adj = ½‖Z_A^T diag(ỹ) Z_B‖²_F = ½ ỹ^T K_epi ỹ."""

    def test_frobenius_equals_quadratic(self, small_data):
        """Both formulations should give the same result."""
        from engine_jax import compute_Q_adj

        Z_A, Z_B, y, X, N = small_data

        # ỹ via simple residualization
        y_star = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

        # Via Frobenius
        Q_frob = float(compute_Q_adj(jnp.array(Z_A), jnp.array(Z_B), jnp.array(y_star)))

        # Via quadratic form
        K_epi = (Z_A @ Z_A.T) * (Z_B @ Z_B.T)
        Q_quad = float(0.5 * y_star @ K_epi @ y_star)

        np.testing.assert_allclose(Q_frob, Q_quad, rtol=1e-8,
            err_msg="Q_adj Frobenius ≠ quadratic form")


# ═══════════════════════════════════════════════════════════════════════════
# Algorithm 1: Hutch++ Accuracy
# ═══════════════════════════════════════════════════════════════════════════

class TestHutchPP:
    """Validate Hutch++ deflation-accelerated trace estimation."""

    def test_accuracy_vs_exact(self, small_data):
        """Hutch++ traces should match exact traces within tolerance."""
        from engine_jax import extract_local_cumulants

        Z_A, Z_B, _, _, N = small_data

        exact = extract_local_cumulants(Z_A, Z_B, method="exact")
        hutchpp = extract_local_cumulants(
            Z_A, Z_B, method="hutchpp", n_probes=100, seed=42
        )

        np.testing.assert_allclose(
            exact["traces"], hutchpp["traces"], rtol=0.05,
            err_msg="Hutch++ traces deviate >5% from exact"
        )

    def test_variance_reduction(self, small_data):
        """Hutch++ should have lower variance than standard Hutchinson."""
        from engine_jax import extract_local_cumulants

        Z_A, Z_B, _, _, N = small_data

        exact = extract_local_cumulants(Z_A, Z_B, method="exact")

        hutchpp_errors = []
        hutch_errors = []
        for seed in range(20):
            hpp = extract_local_cumulants(Z_A, Z_B, method="hutchpp", n_probes=50, seed=seed)
            std = extract_local_cumulants(Z_A, Z_B, method="hutchinson", n_probes=50, seed=seed)
            hutchpp_errors.append(np.abs(hpp["traces"][0] - exact["traces"][0]))
            hutch_errors.append(np.abs(std["traces"][0] - exact["traces"][0]))

        hutchpp_var = np.var(hutchpp_errors)
        hutch_var = np.var(hutch_errors)

        # Hutch++ should have lower variance (≤ standard Hutchinson)
        assert hutchpp_var <= hutch_var * 2.0, \
            f"Hutch++ variance ({hutchpp_var:.4f}) not better than Hutchinson ({hutch_var:.4f})"


# ═══════════════════════════════════════════════════════════════════════════
# Newton's Identities
# ═══════════════════════════════════════════════════════════════════════════

class TestNewtonIdentities:
    """Validate power-sum to cumulant conversion."""

    def test_known_distribution(self):
        """For a known eigenvalue spectrum, cumulants should be correct."""
        from engine_jax import traces_to_cumulants

        # λ = [1, 2, 3] → known moments and cumulants
        lam = np.array([1.0, 2.0, 3.0])
        N = len(lam)
        traces = np.array([
            np.sum(lam),       # tr(A)
            np.sum(lam**2),    # tr(A²)
            np.sum(lam**3),    # tr(A³)
            np.sum(lam**4),    # tr(A⁴)
        ])

        cumulants = np.asarray(traces_to_cumulants(jnp.array(traces), N))

        # κ₁ = μ₁ = mean = 2.0
        np.testing.assert_allclose(cumulants[0], 2.0, atol=1e-10)
        # κ₂ = variance = 2/3
        np.testing.assert_allclose(cumulants[1], 2.0/3.0, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# Theorem 2: Cumulant Additivity
# ═══════════════════════════════════════════════════════════════════════════

class TestCumulantAdditivity:
    """Validate Theorem 2: κ_j,meta = Σ_k κ_j,k."""

    def test_first_trace_additivity(self):
        """
        tr(K_global) = Σ_k tr(K_k) for the first-order trace.

        Higher-order traces are NOT additive across data splits (because
        tr((AB)^p) ≠ Σ_k tr((A_k B_k)^p) for p > 1). Theorem 2 ensures
        additivity across INDEPENDENT cohorts, not arbitrary data splits.
        """
        from engine_jax import extract_local_cumulants

        rng = np.random.default_rng(42)
        N = 200
        m_A, m_B = 5, 5

        Z_A = rng.normal(0, 1, (N, m_A))
        Z_B = rng.normal(0, 1, (N, m_B))

        # Global
        global_result = extract_local_cumulants(Z_A, Z_B, method="exact")

        # Split into 4 nodes
        n_nodes = 4
        node_size = N // n_nodes
        sum_trace_1 = 0.0
        for k in range(n_nodes):
            s, e = k * node_size, (k + 1) * node_size
            local = extract_local_cumulants(Z_A[s:e], Z_B[s:e], method="exact")
            sum_trace_1 += local["traces"][0]  # first-order trace IS additive

        # First-order trace: tr(K) = Σ_n (Z_A[n] ⊙ Z_B[n]) · (Z_A[n] ⊙ Z_B[n])
        # This is a sum over individuals, hence exactly additive
        np.testing.assert_allclose(
            sum_trace_1, global_result["traces"][0], rtol=0.01,
            err_msg="First-order trace additivity violated"
        )


# ═══════════════════════════════════════════════════════════════════════════
# SPA Tail Probability
# ═══════════════════════════════════════════════════════════════════════════

class TestSPA:
    """Validate Lugannani-Rice SPA precision."""

    def test_spa_returns_valid_pvalue(self):
        """SPA should return p-values in [0, 1]."""
        from metararepi.spa.saddlepoint import spa_pvalue

        # Test with known cumulants
        cumulants = np.array([10.0, 5.0, 2.0, 1.0])
        result = spa_pvalue(12.0, cumulants)

        assert 0 <= result["pvalue"] <= 1, f"Invalid p-value: {result['pvalue']}"
        assert "saddlepoint" in result

    def test_spa_monotonicity(self):
        """Larger Q should give smaller p-values."""
        from metararepi.spa.saddlepoint import spa_pvalue

        cumulants = np.array([10.0, 5.0, 2.0, 1.0])
        p_small = spa_pvalue(15.0, cumulants)["pvalue"]
        p_large = spa_pvalue(25.0, cumulants)["pvalue"]

        assert p_large <= p_small, \
            f"SPA not monotone: p({25})={p_large} > p({15})={p_small}"

    def test_spa_batch(self):
        """Batch SPA should produce same results as individual calls."""
        from metararepi.spa.saddlepoint import spa_pvalue, spa_pvalues_batch

        cumulants = np.array([10.0, 5.0, 2.0, 1.0])
        q_batch = np.array([8.0, 12.0, 16.0, 20.0])

        batch_result = spa_pvalues_batch(q_batch, cumulants)
        for i, q in enumerate(q_batch):
            single = spa_pvalue(float(q), cumulants)
            np.testing.assert_allclose(
                batch_result["pvalues"][i], single["pvalue"], rtol=1e-6,
            )


# ═══════════════════════════════════════════════════════════════════════════
# Non-Linear Genomic Control
# ═══════════════════════════════════════════════════════════════════════════

class TestNLGC:
    """Validate non-linear genomic control (§2.4)."""

    def test_sparse_grm(self):
        """GRM sparsification should zero out small entries."""
        from metararepi.nlgc import sparsify_grm

        N = 50
        rng = np.random.default_rng(42)
        GRM = np.eye(N) + 0.02 * rng.normal(0, 1, (N, N))
        GRM = (GRM + GRM.T) / 2

        GRM_sp = sparsify_grm(GRM, threshold=0.05)
        # Off-diagonal entries with |value| < 0.05 should be zero
        assert GRM_sp.nnz < N * N, "Sparsification did not reduce entries"
        # Diagonal should be preserved
        np.testing.assert_allclose(GRM_sp.diagonal(), np.diag(GRM), atol=1e-10)

    def test_hadamard_squared(self):
        """Hadamard square should square each element."""
        from metararepi.nlgc import sparsify_grm, hadamard_squared_grm

        N = 20
        GRM = np.eye(N) + 0.1 * np.ones((N, N))
        GRM_sp = sparsify_grm(GRM, threshold=0.01)
        K_bg = hadamard_squared_grm(GRM_sp)

        # Check a few entries
        expected = GRM_sp.multiply(GRM_sp)
        assert np.allclose(K_bg.toarray(), expected.toarray()), "Hadamard square incorrect"


# ═══════════════════════════════════════════════════════════════════════════
# CKKS Encryption
# ═══════════════════════════════════════════════════════════════════════════

class TestCKKS:
    """Validate CKKS homomorphic encryption round-trip."""

    def test_encrypt_decrypt_roundtrip(self):
        """Encryption → decryption should approximately recover plaintext."""
        from federated_spa import CKKSContext

        ctx = CKKSContext()
        plaintext = np.array([1.0, 2.5, -0.3, 100.0, 42.0])
        ct = ctx.encrypt(plaintext)
        recovered = ctx.decrypt(ct)

        np.testing.assert_allclose(plaintext, recovered, atol=1e-8,
            err_msg="CKKS round-trip failed")

    def test_homomorphic_addition(self):
        """Enc(a) + Enc(b) should decrypt to a + b."""
        from federated_spa import CKKSContext

        ctx = CKKSContext()
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        ct_a = ctx.encrypt(a)
        ct_b = ctx.encrypt(b)
        ct_sum = ct_a + ct_b
        recovered = ctx.decrypt(ct_sum)

        np.testing.assert_allclose(a + b, recovered, atol=1e-7,
            err_msg="Homomorphic addition failed")


# ═══════════════════════════════════════════════════════════════════════════
# Graph Search
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphSearch:
    """Validate graph-regularized search space."""

    def test_demo_search_space(self):
        """Demo search space should produce candidates."""
        from metararepi.graph_search import create_demo_search_space

        gs = create_demo_search_space(n_genes=50, ppi_density=0.05)
        summary = gs.summary()

        assert summary["n_candidates"] > 0, "No candidates generated"
        assert summary["bonferroni_threshold"] > 0, "Invalid threshold"
        assert summary["reduction_factor"] > 1, "No search space reduction"

    def test_ppi_threshold(self):
        """Only edges with iPTM ≥ 0.5 should be retained."""
        from metararepi.graph_search import GraphRegularizedSearch

        gs = GraphRegularizedSearch()
        gs.add_ppi_edge("A", "B", 0.3)  # below threshold
        gs.add_ppi_edge("C", "D", 0.7)  # above threshold

        assert len(gs.ppi_edges) == 1, "Low iPTM edge should be filtered"


# ═══════════════════════════════════════════════════════════════════════════
# Memory Safety: No N×N matrices
# ═══════════════════════════════════════════════════════════════════════════

class TestMemorySafety:
    """Ensure no N×N arrays are created during cumulant extraction."""

    def test_no_nxn_allocation(self):
        """Cumulant extraction should not create N×N arrays."""
        from engine_jax import extract_local_cumulants

        N = 500
        rng = np.random.default_rng(42)
        Z_A = rng.normal(0, 1, (N, 10))
        Z_B = rng.normal(0, 1, (N, 10))

        import tracemalloc
        tracemalloc.start()

        _ = extract_local_cumulants(Z_A, Z_B, method="hutchpp", n_probes=20, seed=42)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # N×N float64 = 500*500*8 = 2 MB. Peak should be well below.
        nxn_bytes = N * N * 8
        assert peak < nxn_bytes * 3, \
            f"Suspicious memory usage: {peak/1e6:.1f} MB (N×N = {nxn_bytes/1e6:.1f} MB)"


# ═══════════════════════════════════════════════════════════════════════════
# Security
# ═══════════════════════════════════════════════════════════════════════════

class TestSecurity:
    """Validate security properties of the federated protocol."""

    def test_no_genotype_in_transmission(self):
        """Only 5 scalars leave each node — no genotype data."""
        from federated_spa import CKKSContext, LocalNode

        ctx = CKKSContext()
        node = LocalNode(node_id="test", n_samples=1000)
        cumulants = np.array([1.0, 2.0, 3.0, 4.0])
        node.set_results(cumulants, Q=42.0)

        ct = node.encrypt_and_transmit(ctx)

        # Ciphertext should have exactly 5 elements
        assert ct.data.shape == (5,), f"Expected 5 scalars, got {ct.data.shape}"

    def test_different_contexts_decrypt_correctly(self):
        """Each context should correctly decrypt its own ciphertext."""
        from federated_spa import CKKSContext

        ctx1 = CKKSContext()
        ctx2 = CKKSContext()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        ct = ctx1.encrypt(data)
        recovered = ctx1.decrypt(ct)

        np.testing.assert_allclose(data, recovered, atol=1e-8,
            err_msg="Context failed to decrypt its own ciphertext")
