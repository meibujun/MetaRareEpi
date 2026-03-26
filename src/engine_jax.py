"""
engine_jax.py — XLA-Compiled Implicit Fast-MVM Cumulant Extractor (Local Node)

Nature Genetics 2026 · MetaRareEpi Framework

This is the computational heart of MetaRareEpi.  It guarantees strict O(N)
complexity by exploiting tensor contraction associativity, shattering the
O(N^3) memory/compute wall of dense kernel methods.

Mathematical Specification
--------------------------
Epistatic kernel:   K_epi = (Z_A Z_A^T) ⊙ (Z_B Z_B^T)   [Hadamard of GRMs]

Implicit Fast-MVM identity (NEVER instantiates N×N):

    Step 1:  C_s[a,b] = Σ_n Z_A[n,a] · V[n,s] · Z_B[n,b]      →  (S, m_A, m_B)
    Step 2:  W[n,s]   = Σ_{a,b} Z_A[n,a] · C_s[a,b] · Z_B[n,b] →  (N, S)

Complexity:  O(S · N · m_A · m_B) time,  O(S · m_A · m_B) aux space.
No vmap required — the batched einsum saturates Tensor Cores natively.

Modules
-------
1. fast_mvm_einsum           — Batched implicit MVM via two-step einsum
2. implicit_fwl_deflation    — FWL orthogonalisation via Cholesky micro-inversion
3. hutchinson_trace_estimator — Vectorised Hutchinson with inline FWL deflation
4. exact_traces_microgram     — Deterministic traces via tr(K^p) = tr((H^T H)^p)
5. extract_local_cumulants    — Public API: cumulants + optional test statistic Q_adj

INVARIANTS (enforced at every level):
    - No N×N array is EVER instantiated.
    - All dtypes are float64 (jax_enable_x64 = True).
    - All core loops are @jax.jit compiled for XLA.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# ── CRITICAL: x64 enforcement before ANY JAX computation ─────────────────
# Without this, SPA tail probabilities at P ≈ 10^{-300} underflow to zero,
# producing the pathological horizontal-line artifact in Q-Q plot tails.
jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

class CumulantResult(NamedTuple):
    """Structured output from cumulant extraction."""
    traces: np.ndarray        # (max_power,)  raw tr(K^p) or tr((P·K)^p)
    moments: np.ndarray       # (max_power,)  μ_k = tr/N
    cumulants: np.ndarray     # (max_power,)  κ_k via moment-cumulant relations
    Q_adj: float | None       # FWL-adjusted test statistic (if phenotype provided)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  BATCHED IMPLICIT FAST-MVM  (the mathematical engine)
# ═══════════════════════════════════════════════════════════════════════════
#
# Computes  W = K_epi @ V  for V of shape (N, S)  WITHOUT forming K_epi.
#
# This replaces jax.vmap over single-vector MVMs.  The direct batched einsum
# is superior because:
#   (a) XLA fuses the two contractions into a single kernel launch.
#   (b) No tensor replication across the S dimension (fixes OOM at N=1M).
#   (c) The intermediate C_s is (S, m_A, m_B) ≈ 40K floats — negligible.
#
# ═══════════════════════════════════════════════════════════════════════════

@jax.jit
def fast_mvm_einsum(
    Z_A: jnp.ndarray,
    Z_B: jnp.ndarray,
    V: jnp.ndarray,
) -> jnp.ndarray:
    """
    Batched implicit MVM:  W = K_epi @ V,  shape (N, S).

    Parameters
    ----------
    Z_A : (N, m_A)  standardised genotype block A.
    Z_B : (N, m_B)  standardised genotype block B.
    V   : (N, S)    probe matrix (Rademacher vectors or state).

    Returns
    -------
    W   : (N, S)    K_epi @ V,  computed without forming the N×N kernel.

    Implementation
    --------------
    The contraction is split into explicit 2-operand steps to prevent XLA
    from forming an O(N·S·m_A·m_B) intermediate:

        Step 1a:  T[n,s,b] = V[n,s] * Z_B[n,b]            →  (N, S, m_B)
        Step 1b:  C_s[s,a,b] = Z_A[n,a]^T · T[n,s,b]      →  (S, m_A, m_B)  [sums N]
        Step 2:   W[n,s] = Z_A[n,a] · C_s[s,a,b] · Z_B[n,b]  →  (N, S)

    Peak auxiliary:  O(N·S·m_B) + O(S·m_A·m_B) — negligible at m_B = 20.
    At N=1M, S=100: ~160 MB intermediate vs 320 GB for the naive path.

    Complexity
    ----------
    Time  : O(S · N · m_A · m_B)
    Space : O(S · N · max(m_A, m_B)) transient,  O(S · m_A · m_B) persistent aux.
    """
    # Step 1a: Element-wise broadcast to fold V into Z_B
    # T = V[:, :, None] * Z_B[:, None, :]  — (N, S, m_B)
    # Step 1b: Contract out N via matmul: Z_A^T @ T reshaped
    # This is equivalent to einsum('ni,ns,nj->sij', Z_A, V, Z_B)
    # but prevents XLA from choosing the wrong contraction order.
    N, S = V.shape
    m_B = Z_B.shape[1]

    # T[n, s*m_B] = V[n,s] ⊗ Z_B[n,b]  — explicit Khatri-Rao column
    T = (V[:, :, None] * Z_B[:, None, :]).reshape(N, S * m_B)  # (N, S·m_B)

    # C_flat[a, s*m_B] = Z_A[n,a]^T · T[n, s*m_B]  — sums out N immediately
    C_flat = Z_A.T @ T                                          # (m_A, S·m_B)
    C_s = C_flat.reshape(Z_A.shape[1], S, m_B).transpose(1, 0, 2)  # (S, m_A, m_B)

    # Step 2: Expand back to (N, S) — also explicit 2-operand to block XLA bloat
    # First: Z_A @ C_s_reshaped → intermediate (N, S*m_B)
    # C_s is (S, m_A, m_B) → reshape to (m_A, S*m_B)
    C_2d = C_s.transpose(1, 0, 2).reshape(Z_A.shape[1], S * m_B)  # (m_A, S·m_B)
    U = Z_A @ C_2d                                                  # (N, S·m_B)
    U_3d = U.reshape(N, S, m_B)                                     # (N, S, m_B)

    # Second: element-wise multiply with Z_B and sum over m_B
    W = jnp.sum(U_3d * Z_B[:, None, :], axis=2)                    # (N, S)

    return W


@jax.jit
def fast_mvm_single(
    Z_A: jnp.ndarray,
    Z_B: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """
    Single-vector MVM:  w = K_epi @ v,  shape (N,).

    Retained for unit-testing and single-vector operations.
    For batched probes, prefer fast_mvm_einsum.
    """
    C = jnp.einsum('ni,n,nj->ij', Z_A, v, Z_B, optimize=True)
    w = jnp.einsum('ni,ij,nj->n',  Z_A, C, Z_B, optimize=True)
    return w


# ═══════════════════════════════════════════════════════════════════════════
# 2.  FWL ORTHOGONAL DEFLATION  (inline Cholesky micro-inversion)
# ═══════════════════════════════════════════════════════════════════════════
#
# Projects vectors into the null-space of marginal main effects Z_main,
# optionally after a prior covariate projection P0.
#
# The FWL projector is:
#     P_adj = I - Z_main (Z_main^T Z_main)^{-1} Z_main^T
#
# With prior covariates (P0 precomputed):
#     P_adj = P0 - P0·Z_main (Z_main^T P0 Z_main)^{-1} Z_main^T P0
#
# The Cholesky factorization is on the MICRO-matrix Z_main^T Z_main
# (dimension m_A + m_B ≈ 40×40), so it is O((m_A+m_B)^3) ≈ instant.
#
# ═══════════════════════════════════════════════════════════════════════════

@jax.jit
def fwl_cholesky_factor(Z_main: jnp.ndarray) -> tuple:
    """
    Precompute Cholesky factor of the FWL micro-matrix Z_main^T Z_main.

    Parameters
    ----------
    Z_main : (N, m)  concatenation of [Z_A, Z_B] or covariate matrix.

    Returns
    -------
    cho_factor : tuple (L, lower) for jax.scipy.linalg.cho_solve.
    """
    gram = Z_main.T @ Z_main          # (m, m) — micro-matrix, never N×N
    return jax.scipy.linalg.cho_factor(gram)


@jax.jit
def fwl_project(
    Y: jnp.ndarray,
    Z_main: jnp.ndarray,
    cho: tuple,
) -> jnp.ndarray:
    """
    Project Y into the null-space of Z_main via FWL Cholesky.

        Y_perp = Y - Z_main · (Z_main^T Z_main)^{-1} · Z_main^T · Y

    Parameters
    ----------
    Y      : (N,) or (N, S) — vector(s) to project.
    Z_main : (N, m) — main-effect matrix.
    cho    : Cholesky factor of Z_main^T Z_main.

    Returns
    -------
    Y_perp : same shape as Y — residuals orthogonal to col(Z_main).
    """
    ZtY = Z_main.T @ Y                                  # (m,) or (m, S)
    coef = jax.scipy.linalg.cho_solve(cho, ZtY)          # (m,) or (m, S)
    return Y - Z_main @ coef                             # (N,) or (N, S)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  VECTORISED HUTCHINSON TRACE ESTIMATOR  (with inline FWL deflation)
# ═══════════════════════════════════════════════════════════════════════════
#
# Computes  tr((P_adj · K_epi)^p)  for p = 1…4  using S Rademacher probes.
#
# The critical difference from a naive Hutchinson is that FWL deflation
# is applied INSIDE the iteration loop:
#
#     V_{j+1} = P_adj · K_epi · V_j
#
# This projects out main effects from the kernel's action at every step,
# yielding the correct null distribution for the epistasis-specific test.
#
# ═══════════════════════════════════════════════════════════════════════════

@functools.partial(jax.jit, static_argnames=("max_power",))
def hutchinson_traces(
    Z_A: jnp.ndarray,
    Z_B: jnp.ndarray,
    V0: jnp.ndarray,
    max_power: int = 4,
) -> jnp.ndarray:
    """
    Estimate tr(K^p) for p = 1…max_power using S Rademacher probes.

    MEMORY-SAFE IMPLEMENTATION: probes are processed sequentially via
    fast_mvm_single (O(m_A·m_B) auxiliary per probe), then accumulated.
    This avoids the O(N·S·m_B) intermediate of the batched einsum path.

    At N=1M, S=100: peak memory is ~328 MB (Z_A + Z_B + V0 + 1 state vec).
    The batched einsum would need ~16 GB.

    Parameters
    ----------
    Z_A : (N, m_A)
    Z_B : (N, m_B)
    V0  : (N, S)  Rademacher probe matrix (±1 iid).
    max_power : highest power (default 4).

    Returns
    -------
    traces : (max_power,) — estimated [tr(K), tr(K²), tr(K³), tr(K⁴)].
    """
    S = V0.shape[1]
    traces = jnp.zeros(max_power, dtype=jnp.float64)

    # Process each probe sequentially to bound auxiliary memory at O(m_A·m_B)
    def _accumulate_probe(traces, s_idx):
        """Process probe s and accumulate its trace contributions."""
        v0_s = V0[:, s_idx]          # (N,) — the s-th Rademacher vector
        v_state = v0_s
        probe_traces = jnp.zeros(max_power, dtype=jnp.float64)
        for p in range(max_power):
            v_state = fast_mvm_single(Z_A, Z_B, v_state)  # O(m_A·m_B) aux
            # Contribution: v0_s^T · K^{p+1} · v0_s
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
    """
    Estimate tr((P_adj · K)^p) with inline FWL deflation.

    Each iteration applies:  V_{j+1} = P_adj · K_epi · V_j

    This yields traces of the FWL-projected kernel, which encode the
    null distribution of the epistasis-specific test statistic.
    """
    S = V0.shape[1]
    V_state = V0
    traces = jnp.zeros(max_power, dtype=jnp.float64)

    for p in range(max_power):
        # Step 1: Implicit MVM
        W_raw = fast_mvm_einsum(Z_A, Z_B, V_state)
        # Step 2: FWL deflation (project out main effects)
        V_state = fwl_project(W_raw, Z_main, cho)
        # Step 3: Trace estimate
        tr_est = jnp.sum(V0 * V_state) / S
        traces = traces.at[p].set(tr_est)

    return traces


# ═══════════════════════════════════════════════════════════════════════════
# 4.  EXACT MICRO-GRAM TRACE EXTRACTOR  (deterministic, no probes)
# ═══════════════════════════════════════════════════════════════════════════
#
# Exploits the low-rank identity:
#     K_epi = H H^T,  where  H[n, a·m_B + b] = Z_A[n,a] · Z_B[n,b]
#
# Therefore:  tr(K^p) = tr((HH^T)^p) = tr((H^T H)^p)
#
# H^T H is (m_A·m_B) × (m_A·m_B) — a tiny micro-matrix.
# This gives EXACT traces in O(N·m_A·m_B + (m_A·m_B)³) time.
#
# ═══════════════════════════════════════════════════════════════════════════

@functools.partial(jax.jit, static_argnames=("max_power",))
def exact_traces_microgram(
    Z_A: jnp.ndarray,
    Z_B: jnp.ndarray,
    max_power: int = 4,
) -> jnp.ndarray:
    """
    Exact traces via the micro-gram identity:  tr(K^p) = tr((H^T H)^p).

    Parameters
    ----------
    Z_A : (N, m_A)
    Z_B : (N, m_B)
    max_power : highest power.

    Returns
    -------
    traces : (max_power,) — exact [tr(K), tr(K²), tr(K³), tr(K⁴)].
    """
    m_A, m_B = Z_A.shape[1], Z_B.shape[1]
    d = m_A * m_B

    # H = Z_A ⊗_row Z_B : (N, m_A·m_B)
    H = (Z_A[:, :, None] * Z_B[:, None, :]).reshape(-1, d)

    # Micro-Gram: G = H^T H is (d, d) — tiny!
    G = H.T @ H

    # Iterative trace: tr(G^p)
    traces = jnp.zeros(max_power, dtype=jnp.float64)
    Gp = jnp.eye(d, dtype=jnp.float64)
    for p in range(max_power):
        Gp = Gp @ G
        traces = traces.at[p].set(jnp.trace(Gp))

    return traces


# ═══════════════════════════════════════════════════════════════════════════
# 5.  MOMENT → CUMULANT CONVERSION
# ═══════════════════════════════════════════════════════════════════════════
#
# The spectral moments μ_k = tr(K^k)/N and the cumulants κ_k of the
# eigenvalue distribution satisfy the standard moment-cumulant relations:
#
#     κ₁ = μ₁
#     κ₂ = μ₂ − μ₁²
#     κ₃ = μ₃ − 3μ₂μ₁ + 2μ₁³
#     κ₄ = μ₄ − 4μ₃μ₁ − 3μ₂² + 12μ₂μ₁² − 6μ₁⁴
#
# ═══════════════════════════════════════════════════════════════════════════

@jax.jit
def moments_to_cumulants(mu: jnp.ndarray) -> jnp.ndarray:
    """Convert 4 spectral moments to 4 cumulants."""
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


# ═══════════════════════════════════════════════════════════════════════════
# 6.  FWL-ADJUSTED TEST STATISTIC  Q_adj
# ═══════════════════════════════════════════════════════════════════════════
#
# For the SKAT-type epistasis test:
#
#     Q_adj = (1/2) · || Z_A^T · diag(y*) · Z_B ||_F^2
#
# where y* is the FWL-projected phenotype residual.
# This is equivalent to y*^T K_epi y* but computed in O(N·m_A·m_B).
#
# ═══════════════════════════════════════════════════════════════════════════

@jax.jit
def compute_Q_adj(
    Z_A: jnp.ndarray,
    Z_B: jnp.ndarray,
    y_star: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the FWL-adjusted epistasis test statistic.

    Q_adj = (1/2) · || Z_A^T diag(y*) Z_B ||_F^2

    Parameters
    ----------
    Z_A   : (N, m_A)
    Z_B   : (N, m_B)
    y_star : (N,) FWL-projected phenotype residual.

    Returns
    -------
    Q_adj : scalar test statistic.
    """
    # C_obs = Z_A^T · diag(y*) · Z_B  =  (Z_A * y*)^T · Z_B
    C_obs = jnp.einsum('ni,n,nj->ij', Z_A, y_star, Z_B, optimize=True)
    return 0.5 * jnp.sum(jnp.square(C_obs))


# ═══════════════════════════════════════════════════════════════════════════
# 7.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def extract_local_cumulants(
    Z_A: np.ndarray,
    Z_B: np.ndarray,
    *,
    max_power: int = 4,
    method: str = "exact",
    n_probes: int = 100,
    seed: int = 0,
    y: np.ndarray | None = None,
    apply_fwl: bool = False,
) -> dict[str, np.ndarray]:
    """
    Extract spectral cumulants κ₁–κ₄ of K_epi = (Z_A Z_Aᵀ) ⊙ (Z_B Z_Bᵀ).

    NEVER instantiates the N×N kernel matrix.

    Parameters
    ----------
    Z_A       : (N, m_A)  standardised genotype block A.
    Z_B       : (N, m_B)  standardised genotype block B.
    max_power : highest trace power (default 4).
    method    : "exact" — deterministic micro-gram traces.
                "hutchinson" — stochastic Rademacher trace estimation.
    n_probes  : S, number of Rademacher probes (Hutchinson only).
    seed      : PRNG seed for Rademacher generation.
    y         : (N,) phenotype vector (optional, for Q_adj computation).
    apply_fwl : if True, apply FWL deflation inside the Hutchinson loop
                to project out main effects Z_main = [Z_A, Z_B].

    Returns
    -------
    dict with keys:
        "traces"    — (4,) raw traces.
        "moments"   — (4,) spectral moments μ_k = tr/N.
        "cumulants" — (4,) spectral cumulants κ_k.
        "Q_adj"     — scalar test statistic (only if y is provided).
    """
    Z_A_j = jnp.asarray(Z_A, dtype=jnp.float64)
    Z_B_j = jnp.asarray(Z_B, dtype=jnp.float64)
    N = Z_A_j.shape[0]

    # Trace extraction
    if method == "exact":
        traces = exact_traces_microgram(Z_A_j, Z_B_j, max_power)

    elif method == "hutchinson":
        key = jax.random.PRNGKey(seed)
        V0 = jax.random.rademacher(key, shape=(N, n_probes), dtype=jnp.float64)

        if apply_fwl:
            Z_main = jnp.concatenate([Z_A_j, Z_B_j], axis=1)
            cho = fwl_cholesky_factor(Z_main)
            traces = hutchinson_traces_fwl(
                Z_A_j, Z_B_j, Z_main, cho, V0, max_power,
            )
        else:
            traces = hutchinson_traces(Z_A_j, Z_B_j, V0, max_power)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'exact' or 'hutchinson'.")

    # Moments → cumulants
    moments = traces / N
    cumulants = moments_to_cumulants(moments)

    result = {
        "traces": np.asarray(traces),
        "moments": np.asarray(moments),
        "cumulants": np.asarray(cumulants),
    }

    # Optional: FWL-adjusted test statistic
    if y is not None:
        y_j = jnp.asarray(y, dtype=jnp.float64)
        Z_main = jnp.concatenate([Z_A_j, Z_B_j], axis=1)
        cho = fwl_cholesky_factor(Z_main)
        y_star = fwl_project(y_j, Z_main, cho)
        result["Q_adj"] = float(compute_Q_adj(Z_A_j, Z_B_j, y_star))

    return result


# ═══════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY ALIASES
# ═══════════════════════════════════════════════════════════════════════════
# These maintain compatibility with the Phase 2 test harness.

_epi_mvm_single = fast_mvm_single
_epi_mvm_batched = lambda Z_A, Z_B, V: fast_mvm_einsum(Z_A, Z_B, V)
_exact_traces_via_microgram = exact_traces_microgram
_hutchinson_traces = hutchinson_traces
_moments_to_cumulants = moments_to_cumulants
