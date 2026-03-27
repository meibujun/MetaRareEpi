"""
engine_jax.py — Dual-Space Deflation-Accelerated Cumulant Extractor

MetaRareEpi R2 Framework

Implements the core computational engine described in §2.3 of the paper:
  - Symmetric dual-space reformulation via row-wise Khatri-Rao product (Theorem 1)
  - Deflation-accelerated Hutch++ trace estimation (Algorithm 1)
  - Implicit dual-space MVM without ever forming dense N×N matrices
  - Newton's identities for power-sum traces → cumulant conversion

INVARIANTS:
  - No N×N array is EVER instantiated.
  - All dtypes are float64 (jax_enable_x64 = True).
  - All core loops are @jax.jit compiled for XLA.
  - Dual-space matrix G_dual is symmetric PSD by construction.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# ── CRITICAL: x64 enforcement before ANY JAX computation ─────────────────
jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

class CumulantResult(NamedTuple):
    """Structured output from cumulant extraction."""
    traces: np.ndarray        # (max_power,)  raw tr((P_adj K_epi)^p)
    cumulants: np.ndarray     # (max_power,)  κ_k via Newton's identities
    Q_adj: float | None       # FWL-adjusted test statistic


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PRIMAL-SPACE IMPLICIT MVM  (K_epi @ v without forming K_epi)
# ═══════════════════════════════════════════════════════════════════════════

@jax.jit
def fast_mvm_single(
    Z_A: jnp.ndarray,
    Z_B: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """
    Single-vector MVM:  w = K_epi @ v,  shape (N,).

    K_epi = (Z_A Z_A^T) ⊙ (Z_B Z_B^T)  computed IMPLICITLY:
      Step 1:  C[a,b] = Σ_n Z_A[n,a] · v[n] · Z_B[n,b]
      Step 2:  w[n]   = Σ_{a,b} Z_A[n,a] · C[a,b] · Z_B[n,b]

    Complexity: O(N · m_A · m_B) time, O(m_A · m_B) aux space.
    """
    C = jnp.einsum('ni,n,nj->ij', Z_A, v, Z_B, optimize=True)
    w = jnp.einsum('ni,ij,nj->n',  Z_A, C, Z_B, optimize=True)
    return w


@jax.jit
def fast_mvm_batched(
    Z_A: jnp.ndarray,
    Z_B: jnp.ndarray,
    V: jnp.ndarray,
) -> jnp.ndarray:
    """
    Batched implicit MVM:  W = K_epi @ V,  shape (N, S).

    Parameters
    ----------
    Z_A : (N, m_A)
    Z_B : (N, m_B)
    V   : (N, S) probe matrix.

    Returns
    -------
    W   : (N, S)  K_epi @ V.
    """
    N, S = V.shape
    m_B = Z_B.shape[1]
    T = (V[:, :, None] * Z_B[:, None, :]).reshape(N, S * m_B)
    C_flat = Z_A.T @ T
    C_s = C_flat.reshape(Z_A.shape[1], S, m_B).transpose(1, 0, 2)
    C_2d = C_s.transpose(1, 0, 2).reshape(Z_A.shape[1], S * m_B)
    U = Z_A @ C_2d
    U_3d = U.reshape(N, S, m_B)
    W = jnp.sum(U_3d * Z_B[:, None, :], axis=2)
    return W


# ═══════════════════════════════════════════════════════════════════════════
# 2.  DUAL-SPACE IMPLICIT MVM  (Theorem 1: G_dual = Z_KR^T P_adj Z_KR)
# ═══════════════════════════════════════════════════════════════════════════
#
# The row-wise Khatri-Rao product Z_KR[n, a*m_B + b] = Z_A[n,a] * Z_B[n,b]
# satisfies K_epi = Z_KR @ Z_KR^T.
#
# Theorem 1: eigenvalues of (P_adj K_epi)  ≡  eigenvalues of G_dual
# where G_dual = Z_KR^T P_adj Z_KR  is (m_A*m_B) × (m_A*m_B) SPSD.
#
# This enables Hutch++ in the symmetric dual space.
# ═══════════════════════════════════════════════════════════════════════════

@jax.jit
def khatri_rao_product(
    Z_A: jnp.ndarray,
    Z_B: jnp.ndarray,
) -> jnp.ndarray:
    """
    Row-wise Khatri-Rao product: Z_KR[n, a*m_B + b] = Z_A[n,a] * Z_B[n,b].

    Shape: (N, m_A * m_B).

    Note: This forms an N × d matrix where d = m_A * m_B.
    For typical rare-variant sets (m_A, m_B ≈ 20), d = 400, manageable.
    """
    return (Z_A[:, :, None] * Z_B[:, None, :]).reshape(
        Z_A.shape[0], Z_A.shape[1] * Z_B.shape[1]
    )


@jax.jit
def build_dual_gram(
    Z_KR: jnp.ndarray,
    P_adj_diag: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Build the symmetric dual Gram matrix G_dual = Z_KR^T P_adj Z_KR.

    For the simple case (no explicit P_adj), G_dual = Z_KR^T Z_KR.

    Parameters
    ----------
    Z_KR     : (N, d) Khatri-Rao product, d = m_A * m_B.
    P_adj_diag : (N,) diagonal of P_adj (if using diagonal approximation).

    Returns
    -------
    G_dual : (d, d) symmetric PSD matrix.
    """
    if P_adj_diag is not None:
        # G_dual = Z_KR^T diag(P_adj) Z_KR
        PZ = Z_KR * P_adj_diag[:, None]
        return PZ.T @ Z_KR
    return Z_KR.T @ Z_KR


def dual_space_mvm_factory(Z_KR, P_adj_apply=None):
    """
    Create an implicit MVM function for the dual Gram matrix.

    For full P_adj (not just diagonal), P_adj_apply should be a function
    that computes P_adj @ v for a vector v.

    Parameters
    ----------
    Z_KR       : (N, d) Khatri-Rao product.
    P_adj_apply : callable v -> P_adj @ v (optional).

    Returns
    -------
    dual_mvm : function q -> G_dual @ q, where q is (d,) or (d, S).
    """
    @jax.jit
    def dual_mvm(q):
        """G_dual @ q = Z_KR^T (P_adj (Z_KR @ q))"""
        u = Z_KR @ q                      # (N,) or (N, S)
        if P_adj_apply is not None:
            u = P_adj_apply(u)             # P_adj @ u
        return Z_KR.T @ u                 # (d,) or (d, S)
    return dual_mvm


# ═══════════════════════════════════════════════════════════════════════════
# 3.  FWL ORTHOGONAL DEFLATION (Generalized — Proposition 1)
# ═══════════════════════════════════════════════════════════════════════════
#
# Generalized FWL for combined main-effect + covariate deflation.
# For binary traits, the IRLS weights are ALREADY embedded in V (and hence
# P0), so no explicit weight injection is needed (Remark 1 in paper).
#
# P_adj = P0 - P0 Z_main (Z_main^T P0 Z_main)^{-1} Z_main^T P0
# ═══════════════════════════════════════════════════════════════════════════

@jax.jit
def fwl_cholesky_factor(Z_main: jnp.ndarray) -> tuple:
    """Precompute Cholesky factor of Z_main^T Z_main."""
    gram = Z_main.T @ Z_main
    return jax.scipy.linalg.cho_factor(gram)


@jax.jit
def fwl_project(
    Y: jnp.ndarray,
    Z_main: jnp.ndarray,
    cho: tuple,
) -> jnp.ndarray:
    """Project Y into null-space of Z_main: Y_perp = Y - Z_main (G^{-1} Z_main^T Y)."""
    ZtY = Z_main.T @ Y
    coef = jax.scipy.linalg.cho_solve(cho, ZtY)
    return Y - Z_main @ coef


def build_P_adj_operator(P0_apply, Z_main):
    """
    Build the generalized FWL projection operator P_adj (Eq. 4).

    P_adj = P0 - P0 Z_main (Z_main^T P0 Z_main)^{-1} Z_main^T P0

    For binary traits, P0 already incorporates IRLS weights via V.

    Parameters
    ----------
    P0_apply : callable v -> P0 @ v
    Z_main   : (N, m) concatenation of [Z_A, Z_B].

    Returns
    -------
    P_adj_apply : callable v -> P_adj @ v
    """
    # Precompute: Z_main^T P0 Z_main
    P0_Zm = jnp.column_stack([P0_apply(Z_main[:, j]) for j in range(Z_main.shape[1])])
    gram_p0 = Z_main.T @ P0_Zm
    cho = jax.scipy.linalg.cho_factor(gram_p0)

    @jax.jit
    def P_adj_apply(v):
        p0v = P0_apply(v)
        ZtP0v = Z_main.T @ p0v
        coef = jax.scipy.linalg.cho_solve(cho, ZtP0v)
        return p0v - P0_Zm @ coef

    return P_adj_apply


# ═══════════════════════════════════════════════════════════════════════════
# 4.  RANDOMIZED NYSTRÖM LOW-RANK APPROXIMATION
# ═══════════════════════════════════════════════════════════════════════════

def nystrom_approximation(mvm_fn, d, rank, key):
    """
    Randomized Nyström low-rank approximation of a symmetric PSD matrix.

    Parameters
    ----------
    mvm_fn : callable q -> A @ q, for SPSD matrix A of size (d, d).
    d      : dimension of matrix A.
    rank   : target rank for deflation.
    key    : JAX PRNG key.

    Returns
    -------
    eigvals : (rank,) approximate top eigenvalues.
    eigvecs : (d, rank) approximate top eigenvectors.
    """
    # Generate random sketch
    Omega = jax.random.normal(key, shape=(d, rank), dtype=jnp.float64)
    # Apply MVM to sketch
    Y = jnp.column_stack([mvm_fn(Omega[:, j]) for j in range(rank)])
    # Stabilize: Y = A @ Omega
    # Nyström: A ≈ Y (Omega^T Y)^{-1} Y^T
    # But for eigendecomposition, use the sketch directly
    # QR factorization of Y for numerical stability
    Q, R = jnp.linalg.qr(Y)
    # Small eigenvalue problem: B = Q^T A Q, (rank × rank)
    AQ = jnp.column_stack([mvm_fn(Q[:, j]) for j in range(rank)])
    B = Q.T @ AQ
    # Symmetrize for numerical stability
    B = (B + B.T) / 2.0
    # Eigendecomposition of the small matrix
    evals, evecs_small = jnp.linalg.eigh(B)
    # Map back to original space
    evecs = Q @ evecs_small
    # Sort by descending eigenvalue
    idx = jnp.argsort(evals)[::-1]
    return evals[idx], evecs[:, idx]


# ═══════════════════════════════════════════════════════════════════════════
# 5.  HUTCH++ DEFLATION-ACCELERATED TRACE ESTIMATION (Algorithm 1)
# ═══════════════════════════════════════════════════════════════════════════

def hutchpp_traces(
    mvm_fn,
    d: int,
    *,
    n_probes: int = 100,
    max_power: int = 4,
    deflation_rank: int | None = None,
    seed: int = 0,
) -> jnp.ndarray:
    """
    Deflation-accelerated Hutch++ trace estimation (Algorithm 1 in paper).

    Estimates tr(A^p) for p = 1..max_power using the Hutch++ framework:
      Step 1: Use n_defl probes for Nyström deflation of dominant eigenspace
      Step 2: Use remaining probes for Hutchinson on well-conditioned residual

    Parameters
    ----------
    mvm_fn         : callable q -> A @ q, for SPSD matrix A of size (d, d).
    d              : dimension of matrix A.
    n_probes       : total probe budget S.
    max_power      : highest power trace (default 4).
    deflation_rank : rank for Nyström deflation (default: S // 3).
    seed           : PRNG seed.

    Returns
    -------
    traces : (max_power,) estimated [tr(A), tr(A²), ..., tr(A^max_power)].
    """
    key = jax.random.PRNGKey(seed)
    if deflation_rank is None:
        deflation_rank = max(n_probes // 3, 1)
    # Cap rank at d-1 (cannot deflate more eigenvectors than exist)
    deflation_rank = min(deflation_rank, max(d - 1, 1))
    n_residual = max(n_probes - deflation_rank, 1)

    # Step 1: Eigenspace deflation via Nyström
    key, subkey = jax.random.split(key)
    eigvals, eigvecs = nystrom_approximation(mvm_fn, d, deflation_rank, subkey)

    # Exact trace contribution from deflated eigenspace
    traces_defl = jnp.zeros(max_power, dtype=jnp.float64)
    for p in range(max_power):
        traces_defl = traces_defl.at[p].set(jnp.sum(eigvals ** (p + 1)))

    # Step 2: Residual Hutchinson estimation
    # Deflation projector: v_residual = v - eigvecs eigvecs^T v
    def deflated_mvm(q):
        """Apply A to q, then project out the deflated eigenspace."""
        Aq = mvm_fn(q)
        # Remove deflated component: A_residual @ q ≈ A @ q - Σ λ_i (u_i u_i^T q)
        proj = eigvecs.T @ q    # (rank,)
        correction = eigvecs @ (eigvals * proj)  # (d,)
        return Aq - correction

    key, subkey = jax.random.split(key)
    R = jax.random.rademacher(subkey, shape=(d, n_residual), dtype=jnp.float64)

    traces_resid = jnp.zeros(max_power, dtype=jnp.float64)

    for s in range(n_residual):
        r_s = R[:, s]
        v_state = r_s
        for p in range(max_power):
            v_state = deflated_mvm(v_state)
            traces_resid = traces_resid.at[p].add(jnp.dot(r_s, v_state))

    traces_resid = traces_resid / n_residual

    return traces_defl + traces_resid


# ═══════════════════════════════════════════════════════════════════════════
# 6.  STANDARD HUTCHINSON TRACE ESTIMATION (baseline comparison)
# ═══════════════════════════════════════════════════════════════════════════

@functools.partial(jax.jit, static_argnames=("max_power",))
def hutchinson_traces(
    Z_A: jnp.ndarray,
    Z_B: jnp.ndarray,
    V0: jnp.ndarray,
    max_power: int = 4,
) -> jnp.ndarray:
    """Standard Hutchinson trace estimation (primal space, no deflation)."""
    S = V0.shape[1]
    traces = jnp.zeros(max_power, dtype=jnp.float64)

    def _accumulate_probe(traces, s_idx):
        v0_s = V0[:, s_idx]
        v_state = v0_s
        probe_traces = jnp.zeros(max_power, dtype=jnp.float64)
        for p in range(max_power):
            v_state = fast_mvm_single(Z_A, Z_B, v_state)
            probe_traces = probe_traces.at[p].set(jnp.dot(v0_s, v_state))
        return traces + probe_traces, None

    traces, _ = jax.lax.scan(_accumulate_probe, traces, jnp.arange(S))
    return traces / S


@functools.partial(jax.jit, static_argnames=("max_power",))
def hutchinson_traces_fwl(
    Z_A: jnp.ndarray,
    Z_B: jnp.ndarray,
    Z_main: jnp.ndarray,
    cho: tuple,
    V0: jnp.ndarray,
    max_power: int = 4,
) -> jnp.ndarray:
    """Hutchinson with inline FWL deflation (primal space)."""
    S = V0.shape[1]
    V_state = V0
    traces = jnp.zeros(max_power, dtype=jnp.float64)
    for p in range(max_power):
        W_raw = fast_mvm_batched(Z_A, Z_B, V_state)
        V_state = fwl_project(W_raw, Z_main, cho)
        tr_est = jnp.sum(V0 * V_state) / S
        traces = traces.at[p].set(tr_est)
    return traces


# ═══════════════════════════════════════════════════════════════════════════
# 7.  EXACT MICRO-GRAM TRACE EXTRACTOR (deterministic validation)
# ═══════════════════════════════════════════════════════════════════════════

@functools.partial(jax.jit, static_argnames=("max_power",))
def exact_traces_microgram(
    Z_A: jnp.ndarray,
    Z_B: jnp.ndarray,
    max_power: int = 4,
) -> jnp.ndarray:
    """
    Exact traces via micro-Gram identity: tr(K^p) = tr((Z_KR^T Z_KR)^p).
    Used for validation against stochastic estimators.
    """
    d = Z_A.shape[1] * Z_B.shape[1]
    Z_KR = khatri_rao_product(Z_A, Z_B)
    G = Z_KR.T @ Z_KR  # (d, d) — dual Gram matrix
    traces = jnp.zeros(max_power, dtype=jnp.float64)
    Gp = jnp.eye(d, dtype=jnp.float64)
    for p in range(max_power):
        Gp = Gp @ G
        traces = traces.at[p].set(jnp.trace(Gp))
    return traces


# ═══════════════════════════════════════════════════════════════════════════
# 8.  POWER-SUM TRACES → CUMULANTS (Newton's identities)
# ═══════════════════════════════════════════════════════════════════════════

@jax.jit
def traces_to_cumulants(traces: jnp.ndarray, N: int) -> jnp.ndarray:
    """
    Convert power-sum traces to cumulants via Newton's identities.

    The spectral moments μ_k = tr(A^k) / N relate to cumulants as:
        κ₁ = μ₁
        κ₂ = μ₂ − μ₁²
        κ₃ = μ₃ − 3μ₂μ₁ + 2μ₁³
        κ₄ = μ₄ − 4μ₃μ₁ − 3μ₂² + 12μ₂μ₁² − 6μ₁⁴
    """
    mu = traces / N
    k = jnp.zeros(4, dtype=jnp.float64)
    k = k.at[0].set(mu[0])
    k = k.at[1].set(mu[1] - mu[0] ** 2)
    k = k.at[2].set(mu[2] - 3.0 * mu[1] * mu[0] + 2.0 * mu[0] ** 3)
    k = k.at[3].set(
        mu[3]
        - 4.0 * mu[2] * mu[0]
        - 3.0 * mu[1] ** 2
        + 12.0 * mu[1] * mu[0] ** 2
        - 6.0 * mu[0] ** 4
    )
    return k


# Backward-compatible alias
moments_to_cumulants = traces_to_cumulants


# ═══════════════════════════════════════════════════════════════════════════
# 9.  FWL-ADJUSTED TEST STATISTIC Q_adj
# ═══════════════════════════════════════════════════════════════════════════

@jax.jit
def compute_Q_adj(
    Z_A: jnp.ndarray,
    Z_B: jnp.ndarray,
    y_star: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute Q_adj = (1/2) · ‖Z_A^T diag(ỹ) Z_B‖²_F (Proposition 2).

    This is equivalent to ỹ^T K_epi ỹ / 2 but computed in O(N · m_A · m_B).
    """
    C_obs = jnp.einsum('ni,n,nj->ij', Z_A, y_star, Z_B, optimize=True)
    return 0.5 * jnp.sum(jnp.square(C_obs))


# ═══════════════════════════════════════════════════════════════════════════
# 10.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def extract_local_cumulants(
    Z_A: np.ndarray,
    Z_B: np.ndarray,
    *,
    max_power: int = 4,
    method: str = "hutchpp",
    n_probes: int = 100,
    deflation_rank: int | None = None,
    seed: int = 0,
    y: np.ndarray | None = None,
    apply_fwl: bool = False,
    P0_apply=None,
) -> dict[str, np.ndarray]:
    """
    Extract spectral cumulants κ₁–κ₄ of K_epi = (Z_A Z_A^T) ⊙ (Z_B Z_B^T).

    NEVER instantiates the N×N kernel matrix.

    Parameters
    ----------
    Z_A          : (N, m_A) standardised genotype block A.
    Z_B          : (N, m_B) standardised genotype block B.
    max_power    : highest trace power (default 4).
    method       : "hutchpp"    — dual-space Hutch++ (Algorithm 1, recommended).
                   "hutchinson" — standard Hutchinson (primal space).
                   "exact"      — deterministic micro-Gram traces.
    n_probes     : S, total probe budget.
    deflation_rank : rank for Nyström in Hutch++ (default: S // 3).
    seed         : PRNG seed.
    y            : (N,) phenotype residual (for Q_adj computation).
    apply_fwl    : if True, apply FWL deflation.
    P0_apply     : callable v -> P0 @ v (for generalized FWL with binary traits).

    Returns
    -------
    dict with keys: "traces", "cumulants", "Q_adj".
    """
    Z_A_j = jnp.asarray(Z_A, dtype=jnp.float64)
    Z_B_j = jnp.asarray(Z_B, dtype=jnp.float64)
    N = Z_A_j.shape[0]

    if method == "exact":
        traces = exact_traces_microgram(Z_A_j, Z_B_j, max_power)

    elif method == "hutchinson":
        key = jax.random.PRNGKey(seed)
        V0 = jax.random.rademacher(key, shape=(N, n_probes), dtype=jnp.float64)
        if apply_fwl:
            Z_main = jnp.concatenate([Z_A_j, Z_B_j], axis=1)
            cho = fwl_cholesky_factor(Z_main)
            traces = hutchinson_traces_fwl(Z_A_j, Z_B_j, Z_main, cho, V0, max_power)
        else:
            traces = hutchinson_traces(Z_A_j, Z_B_j, V0, max_power)

    elif method == "hutchpp":
        # Dual-space Hutch++ (Algorithm 1)
        Z_KR = khatri_rao_product(Z_A_j, Z_B_j)
        d = Z_KR.shape[1]

        if apply_fwl and P0_apply is not None:
            # Generalized FWL: build P_adj operator
            Z_main = jnp.concatenate([Z_A_j, Z_B_j], axis=1)
            P_adj_apply = build_P_adj_operator(P0_apply, Z_main)
            dual_mvm = dual_space_mvm_factory(Z_KR, P_adj_apply)
        elif apply_fwl:
            # Simple FWL: project Z_KR through simple main-effect deflation
            Z_main = jnp.concatenate([Z_A_j, Z_B_j], axis=1)
            cho = fwl_cholesky_factor(Z_main)
            def simple_fwl_apply(v):
                return fwl_project(v, Z_main, cho)
            dual_mvm = dual_space_mvm_factory(Z_KR, simple_fwl_apply)
        else:
            dual_mvm = dual_space_mvm_factory(Z_KR)

        traces = hutchpp_traces(
            dual_mvm, d,
            n_probes=n_probes,
            max_power=max_power,
            deflation_rank=deflation_rank,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'exact', 'hutchinson', or 'hutchpp'.")

    cumulants = traces_to_cumulants(traces, N)

    result = {
        "traces": np.asarray(traces),
        "cumulants": np.asarray(cumulants),
    }

    # Optional: FWL-adjusted test statistic
    if y is not None:
        y_j = jnp.asarray(y, dtype=jnp.float64)
        if apply_fwl:
            Z_main = jnp.concatenate([Z_A_j, Z_B_j], axis=1)
            cho = fwl_cholesky_factor(Z_main)
            y_star = fwl_project(y_j, Z_main, cho)
        else:
            y_star = y_j
        result["Q_adj"] = float(compute_Q_adj(Z_A_j, Z_B_j, y_star))

    return result


# ═══════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY ALIASES
# ═══════════════════════════════════════════════════════════════════════════

_epi_mvm_single = fast_mvm_single
_epi_mvm_batched = fast_mvm_batched
_exact_traces_via_microgram = exact_traces_microgram
_hutchinson_traces = hutchinson_traces
_moments_to_cumulants = lambda mu: traces_to_cumulants(mu * mu.shape[0], mu.shape[0])

# Convenience alias
fast_mvm_einsum = fast_mvm_batched
