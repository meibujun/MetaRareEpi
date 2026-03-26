"""
fast_mvm.py — Implicit epistatic kernel matrix-vector multiplication engine.

Computes K_epi @ v = [(Z_A Z_Aᵀ) ⊙ (Z_B Z_Bᵀ)] @ v  without EVER forming
the N×N kernel.  All operations are O(N · m_A · m_B).

Mathematical specification
--------------------------
K_epi[i,j] = (Σ_a Z_A[i,a] Z_A[j,a]) · (Σ_b Z_B[i,b] Z_B[j,b])

Implicit MVM via einsum:
    Step 1:  C[a,b] = Σ_n Z_A[n,a] · v[n] · Z_B[n,b]         → (m_A, m_B) micro-matrix
    Step 2:  w[n]   = Σ_{a,b} Z_A[n,a] · C[a,b] · Z_B[n,b]   → (N,) result

Complexity invariant:
    Time  : O(N · m_A · m_B)
    Space : O(m_A · m_B)  auxiliary   (input/output are O(N))
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from metararepi._config import JAX_X64_ENABLED as _  # noqa: F401 — ensure x64


# ═══════════════════════════════════════════════════════════════════════════
# Single-vector MVM
# ═══════════════════════════════════════════════════════════════════════════

@jax.jit
def epi_kernel_matvec(
    Z_A: jnp.ndarray,
    Z_B: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute  w = K_epi @ v  for a single vector v ∈ ℝ^N.

    Parameters
    ----------
    Z_A : (N, m_A)  standardised genotype block A.
    Z_B : (N, m_B)  standardised genotype block B.
    v   : (N,)      input vector.

    Returns
    -------
    w   : (N,)      K_epi @ v.
    """
    C = jnp.einsum('ni,n,nj->ij', Z_A, v, Z_B)    # (m_A, m_B)
    w = jnp.einsum('ni,ij,nj->n',  Z_A, C, Z_B)   # (N,)
    return w


# ═══════════════════════════════════════════════════════════════════════════
# Batched MVM via vmap
# ═══════════════════════════════════════════════════════════════════════════

epi_kernel_matvec_batch = jax.vmap(
    epi_kernel_matvec,
    in_axes=(None, None, 1),
    out_axes=1,
)
"""Vectorised MVM:  K_epi @ V  for V of shape (N, S), mapped over columns."""


# ═══════════════════════════════════════════════════════════════════════════
# Exact trace extraction via micro-gram identity
# ═══════════════════════════════════════════════════════════════════════════

@functools.partial(jax.jit, static_argnames=("max_power",))
def extract_traces_exact(
    Z_A: jnp.ndarray,
    Z_B: jnp.ndarray,
    max_power: int = 4,
) -> jnp.ndarray:
    """
    Exact trace computation:  tr(K^p) = tr((H^T H)^p).

    H[n, (a·m_B + b)] = Z_A[n,a] · Z_B[n,b]   →  K_epi = H H^T.
    H^T H is only (m_A·m_B) × (m_A·m_B), a tiny micro-matrix.

    Returns
    -------
    traces : (max_power,) — [tr(K), tr(K²), …, tr(K^max_power)].
    """
    m_A, m_B = Z_A.shape[1], Z_B.shape[1]
    d = m_A * m_B

    H = (Z_A[:, :, None] * Z_B[:, None, :]).reshape(-1, d)
    G = H.T @ H

    traces = jnp.zeros(max_power, dtype=jnp.float64)
    Gp = jnp.eye(d, dtype=jnp.float64)
    for p in range(max_power):
        Gp = Gp @ G
        traces = traces.at[p].set(jnp.trace(Gp))
    return traces


# ═══════════════════════════════════════════════════════════════════════════
# Hutchinson stochastic trace estimation
# ═══════════════════════════════════════════════════════════════════════════

@functools.partial(jax.jit, static_argnames=("max_power",))
def extract_traces_hutchinson(
    Z_A: jnp.ndarray,
    Z_B: jnp.ndarray,
    V0: jnp.ndarray,
    max_power: int = 4,
) -> jnp.ndarray:
    """
    Stochastic trace estimation via S Rademacher probes.

    tr(K^p) ≈ (1/S) Σ_{s=1}^{S}  v_s^T  K^p  v_s

    Parameters
    ----------
    V0 : (N, S) Rademacher probe matrix (±1 iid).

    Returns
    -------
    traces : (max_power,) — estimated traces.
    """
    S = V0.shape[1]
    W = V0
    traces = jnp.zeros(max_power, dtype=jnp.float64)
    for p in range(max_power):
        W = epi_kernel_matvec_batch(Z_A, Z_B, W)
        traces = traces.at[p].set(jnp.sum(V0 * W) / S)
    return traces
