"""
saddlepoint.py — Saddlepoint approximation for ultra-precise p-values.

Given the first K cumulants κ₁…κ_K of the test statistic distribution
(extracted by the kernel engine), compute p-values via the CGF inversion:

    K(t)   = Σ_{k=1}^{K} κ_k t^k / k!
    K'(t̂)  = q   →  solve for saddlepoint t̂
    p(q)  ≈ Φ̄(ŵ) + φ(ŵ)(1/ŵ − 1/û)

where ŵ = sign(t̂)√(2(t̂·q − K(t̂)))  and û = t̂√(K''(t̂)).

Precision: All computations in float64 to reach P ≈ 10^{-300} without
underflow, enforced by metararepi._config.

CRITICAL (2026-03 UPGRADE):
    Uses jax.scipy.stats.norm.sf(w) for Φ̄(ŵ) instead of 1 − Φ(ŵ).
    The survival function sf() computes erfc() internally, which is
    numerically exact down to P ≈ 10^{-300}.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from metararepi._config import JAX_X64_ENABLED as _  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════
# CGF MACLAURIN EXPANSION + jax.grad AUTO-DIFF DERIVATIVES
# ═══════════════════════════════════════════════════════════════════════════

def _make_cgf(kappa: jnp.ndarray):
    """
    Build CGF and its first three AD-derived derivatives.

    K(t) = κ₁t + κ₂t²/2 + κ₃t³/6 + κ₄t⁴/24   (Horner form)

    Returns
    -------
    cgf, cgf_d1, cgf_d2, cgf_d3 : all scalar → scalar JAX functions.
    """
    k1, k2, k3, k4 = kappa[0], kappa[1], kappa[2], kappa[3]

    def cgf(t: jnp.ndarray) -> jnp.ndarray:
        """K(t) — CGF in Horner form for numerical stability."""
        return t * (k1 + t * (k2 / 2.0 + t * (k3 / 6.0 + t * k4 / 24.0)))

    cgf_d1 = jax.grad(cgf)       # K'(t)
    cgf_d2 = jax.grad(cgf_d1)    # K''(t)
    cgf_d3 = jax.grad(cgf_d2)    # K'''(t)

    return cgf, cgf_d1, cgf_d2, cgf_d3


# ═══════════════════════════════════════════════════════════════════════════
# HALLEY'S METHOD (cubic convergence, jax.lax.while_loop)
# ═══════════════════════════════════════════════════════════════════════════

@functools.partial(jax.jit, static_argnames=("max_iter",))
def _solve_saddlepoint(
    kappa: jnp.ndarray,
    Q: jnp.ndarray,
    *,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> jnp.ndarray:
    """
    Solve K'(t̂) = Q via Halley's method with iteration guard.

    Halley update (cubic convergence):
        Δt = 2·f·f' / (2·f'² − f·f'')

    Parameters
    ----------
    kappa    : (4,) global cumulants.
    Q        : scalar test statistic.
    tol      : convergence tolerance on |Δt|.
    max_iter : hard cap on iterations (safety).

    Returns
    -------
    t_hat : scalar saddlepoint.
    """
    _, cgf_d1, cgf_d2, cgf_d3 = _make_cgf(kappa)

    init_state = (jnp.float64(0.0), jnp.int32(0), jnp.float64(1.0))

    def cond_fn(state):
        _, i, abs_dt = state
        return (abs_dt > tol) & (i < max_iter)

    def body_fn(state):
        t, i, _ = state
        fv = cgf_d1(t) - Q
        fp = cgf_d2(t)
        fpp = cgf_d3(t)

        denom = 2.0 * fp * fp - fv * fpp
        denom = jnp.where(jnp.abs(denom) < 1e-30, 1e-30, denom)

        dt = 2.0 * fv * fp / denom
        t_new = t - dt
        return (t_new, i + 1, jnp.abs(dt))

    t_hat, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return t_hat


# ═══════════════════════════════════════════════════════════════════════════
# LUGANNANI-RICE FORMULA + SINGULARITY SAFETY
# ═══════════════════════════════════════════════════════════════════════════

@jax.jit
def _lugannani_rice(
    t_hat: jnp.ndarray,
    Q: jnp.ndarray,
    kappa: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute P(X > Q) via the Lugannani-Rice formula.

    Standard formula:
        ŵ = sign(t̂) · √(2·(t̂Q − K(t̂)))
        û = t̂ · √(K''(t̂))
        P(X > Q) ≈ Φ̄(ŵ) + φ(ŵ)·(1/ŵ − 1/û)

    Uses jax.scipy.stats.norm.sf() for numerically stable extreme-tail
    survival probabilities (exact to P ≈ 10^{-300}).

    Returns
    -------
    pval : scalar tail probability P(X > Q), clamped to [0, 1].
    """
    cgf, _, cgf_d2, _ = _make_cgf(kappa)

    K_at_t = cgf(t_hat)
    K2_at_t = cgf_d2(t_hat)

    # ŵ = sign(t̂) · √(2(t̂Q − K(t̂)))
    exponent = 2.0 * (t_hat * Q - K_at_t)
    exponent = jnp.maximum(exponent, 0.0)
    w_hat = jnp.sign(t_hat) * jnp.sqrt(exponent)

    # û = t̂ · √(K''(t̂))
    K2_safe = jnp.maximum(K2_at_t, 1e-30)
    u_hat = t_hat * jnp.sqrt(K2_safe)

    # Lugannani-Rice tail probability
    phi_w = jax.scipy.stats.norm.pdf(w_hat)
    sf_w = jax.scipy.stats.norm.sf(w_hat)

    w_safe = jnp.where(jnp.abs(w_hat) < 1e-30, 1e-30, w_hat)
    u_safe = jnp.where(jnp.abs(u_hat) < 1e-30, 1e-30, u_hat)

    lr_pval = sf_w + phi_w * (1.0 / w_safe - 1.0 / u_safe)

    # Gaussian fallback when |t̂| < 1e-7 (singularity)
    sigma = jnp.sqrt(jnp.maximum(kappa[1], 1e-30))
    z = (Q - kappa[0]) / sigma
    gauss_pval = jax.scipy.stats.norm.sf(z)

    is_singular = jnp.abs(t_hat) < 1e-7
    pval = jax.lax.select(is_singular, gauss_pval, lr_pval)

    return jnp.clip(pval, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def spa_pvalue(
    q: float | np.ndarray,
    cumulants: np.ndarray,
    *,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> dict[str, np.ndarray]:
    """
    Compute SPA p-value for observed test statistic q.

    Full pipeline: CGF construction → Halley saddlepoint solve →
    Lugannani-Rice tail probability with singularity fallback.

    Parameters
    ----------
    q          : scalar observed test statistic.
    cumulants  : (4,) spectral cumulants κ₁…κ₄ from the kernel engine.
    tol        : Halley convergence tolerance.
    max_iter   : Halley iteration cap.

    Returns
    -------
    dict with keys:
        "saddlepoint" — scalar t̂.
        "pvalue"      — scalar tail probability P(X > q).
        "cumulants"   — (4,) input cumulants (echo).
    """
    kappa = jnp.asarray(cumulants, dtype=jnp.float64)
    Q_j = jnp.float64(q)

    t_hat = _solve_saddlepoint(kappa, Q_j, tol=tol, max_iter=max_iter)
    pval = _lugannani_rice(t_hat, Q_j, kappa)

    return {
        "saddlepoint": float(t_hat),
        "pvalue": float(pval),
        "cumulants": np.asarray(kappa),
    }


def spa_pvalues_batch(
    q_batch: np.ndarray,
    cumulants: np.ndarray,
    *,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> dict[str, np.ndarray]:
    """
    Vectorised SPA over a batch of test statistics.

    Parameters
    ----------
    q_batch    : (B,) array of observed test statistics.
    cumulants  : (4,) spectral cumulants.

    Returns
    -------
    dict with "saddlepoints" (B,), "pvalues" (B,), "cumulants" (4,).
    """
    kappa = jnp.asarray(cumulants, dtype=jnp.float64)
    Q_j = jnp.asarray(q_batch, dtype=jnp.float64)

    solve_one = functools.partial(
        _solve_saddlepoint, kappa, tol=tol, max_iter=max_iter,
    )
    lr_one = functools.partial(_lugannani_rice, kappa=kappa)

    t_hats = jax.vmap(solve_one)(Q_j)
    pvals = jax.vmap(lr_one, in_axes=(0, 0))(t_hats, Q_j)

    return {
        "saddlepoints": np.asarray(t_hats),
        "pvalues": np.asarray(pvals),
        "cumulants": np.asarray(kappa),
    }
