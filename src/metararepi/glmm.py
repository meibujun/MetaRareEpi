"""
glmm.py — Generalized Linear Mixed Model with Binary Trait Support

MetaRareEpi R2 Framework (§2.1, §2.2)

Implements:
  - GLMM null model fitting (continuous + binary traits)
  - AI-REML for variance component estimation
  - IRLS convergence for logistic GLMM
  - P0 projection matrix construction
  - Generalized FWL orthogonalization (Proposition 1)
  - Working weight extraction for binary traits

Key insight (Remark 1): For binary traits, the IRLS working weights are
already embedded within V = (W^{-1} + τ²Φ), so P0 operates in the correct
heteroscedastic metric WITHOUT explicit weight injection.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg as sparse_cg

jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. VARIANCE COMPONENT ESTIMATION (AI-REML)
# ═══════════════════════════════════════════════════════════════════════════

def fit_null_model(
    y: np.ndarray,
    X: np.ndarray,
    GRM: np.ndarray | None = None,
    *,
    trait_type: str = "continuous",
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    """
    Fit the GLMM null model under H0 (no epistatic effect).

    For continuous:  y = Xα + u + ε,  u ~ N(0, τ²Φ)
    For binary:      logit(μ) = Xα + u,  u ~ N(0, τ²Φ)

    Parameters
    ----------
    y          : (N,) phenotype vector.
    X          : (N, p) covariate matrix (intercept, sex, age, PCs).
    GRM        : (N, N) genetic relationship matrix Φ.
    trait_type : "continuous" or "binary".
    max_iter   : maximum iterations for AI-REML / IRLS.
    tol        : convergence tolerance.

    Returns
    -------
    dict with keys:
        "tau2"      — polygenic variance τ².
        "sigma2"    — residual variance σ² (continuous only).
        "P0"        — null projection matrix function (callable v -> P0 @ v).
        "V_inv"     — inverse of phenotypic covariance.
        "y_adj"     — adjusted residual ỹ = P0(y - μ̂).
        "W_diag"    — IRLS working weights (binary only).
        "mu_hat"    — fitted means.
        "converged" — bool.
    """
    N = y.shape[0]
    p = X.shape[1]

    if GRM is None:
        GRM = np.eye(N, dtype=np.float64)

    if trait_type == "continuous":
        return _fit_continuous(y, X, GRM, max_iter, tol)
    elif trait_type == "binary":
        return _fit_binary(y, X, GRM, max_iter, tol)
    else:
        raise ValueError(f"Unknown trait_type: {trait_type!r}")


def _fit_continuous(y, X, GRM, max_iter, tol):
    """Fit continuous trait GLMM via AI-REML."""
    N = y.shape[0]
    p = X.shape[1]

    # Initialize variance components
    sigma2 = float(np.var(y)) * 0.5
    tau2 = float(np.var(y)) * 0.5

    converged = False
    for iteration in range(max_iter):
        # Phenotypic covariance: V = σ²I + τ²Φ
        V = sigma2 * np.eye(N) + tau2 * GRM

        # V inverse (dense for moderate N)
        V_inv = np.linalg.inv(V)

        # P0 = V^{-1} - V^{-1} X (X^T V^{-1} X)^{-1} X^T V^{-1}
        ViX = V_inv @ X
        XtViX = X.T @ ViX
        XtViX_inv = np.linalg.inv(XtViX)

        P0 = V_inv - ViX @ XtViX_inv @ ViX.T

        # AI-REML gradient and Hessian
        Py = P0 @ y
        # Gradient
        g_sigma2 = -0.5 * np.trace(P0) + 0.5 * Py @ Py
        P0_GRM = P0 @ GRM
        g_tau2 = -0.5 * np.trace(P0_GRM) + 0.5 * Py @ GRM @ Py

        # AI Hessian
        H = np.zeros((2, 2))
        H[0, 0] = 0.5 * y.T @ P0 @ P0 @ P0 @ y
        H[0, 1] = 0.5 * y.T @ P0 @ P0 @ GRM @ P0 @ y
        H[1, 0] = H[0, 1]
        H[1, 1] = 0.5 * y.T @ P0 @ GRM @ P0 @ GRM @ P0 @ y

        # Newton update
        grad = np.array([g_sigma2, g_tau2])
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = grad * 0.01  # fallback

        sigma2_new = max(sigma2 + delta[0], 1e-6)
        tau2_new = max(tau2 + delta[1], 1e-6)

        if abs(sigma2_new - sigma2) + abs(tau2_new - tau2) < tol:
            converged = True
            sigma2, tau2 = sigma2_new, tau2_new
            break

        sigma2, tau2 = sigma2_new, tau2_new

    # Final computations
    V = sigma2 * np.eye(N) + tau2 * GRM
    V_inv = np.linalg.inv(V)
    ViX = V_inv @ X
    XtViX_inv = np.linalg.inv(X.T @ ViX)
    P0_mat = V_inv - ViX @ XtViX_inv @ ViX.T

    alpha_hat = XtViX_inv @ (ViX.T @ y)
    mu_hat = X @ alpha_hat
    y_adj = P0_mat @ (y - mu_hat)

    def P0_apply(v):
        return P0_mat @ v

    return {
        "tau2": tau2,
        "sigma2": sigma2,
        "P0": P0_apply,
        "P0_matrix": P0_mat,
        "V_inv": V_inv,
        "y_adj": y_adj,
        "mu_hat": mu_hat,
        "converged": converged,
        "n_iter": iteration + 1,
    }


def _fit_binary(y, X, GRM, max_iter, tol):
    """
    Fit binary trait GLMM via IRLS + AI-REML.

    Key property (Remark 1): The IRLS weights W = diag(μ(1-μ)) are
    absorbed into V = W^{-1} + τ²Φ. The base projection P0 therefore
    operates in the correct heteroscedastic metric.
    """
    N = y.shape[0]
    p = X.shape[1]

    # Initialize
    tau2 = 0.5
    beta = np.zeros(p)
    eta = X @ beta

    converged = False
    for iteration in range(max_iter):
        # Logistic link
        mu = 1.0 / (1.0 + np.exp(-eta))
        mu = np.clip(mu, 1e-10, 1.0 - 1e-10)

        # IRLS working weights: W = μ(1-μ)
        W_diag = mu * (1.0 - mu)

        # Working response
        z = eta + (y - mu) / W_diag

        # Phenotypic covariance: V = W^{-1} + τ²Φ
        # The key insight: W^{-1} naturally embeds the heteroscedastic weights
        W_inv = 1.0 / W_diag
        V = np.diag(W_inv) + tau2 * GRM
        V_inv = np.linalg.inv(V)

        # Update beta
        ViX = V_inv @ X
        XtViX = X.T @ ViX
        XtViX_inv = np.linalg.inv(XtViX)
        beta_new = XtViX_inv @ (ViX.T @ z)

        # P0 projection
        P0_mat = V_inv - ViX @ XtViX_inv @ ViX.T

        # AI-REML update for tau2
        Pz = P0_mat @ z
        P0_GRM = P0_mat @ GRM
        g_tau2 = -0.5 * np.trace(P0_GRM) + 0.5 * Pz @ GRM @ Pz
        H_tau2 = 0.5 * z.T @ P0_mat @ GRM @ P0_mat @ GRM @ P0_mat @ z

        if abs(H_tau2) > 1e-12:
            tau2_new = max(tau2 + g_tau2 / H_tau2, 1e-6)
        else:
            tau2_new = tau2

        eta = X @ beta_new

        # Check convergence
        if np.max(np.abs(beta_new - beta)) < tol and abs(tau2_new - tau2) < tol:
            converged = True
            beta, tau2 = beta_new, tau2_new
            break

        beta, tau2 = beta_new, tau2_new

    # Final computations
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-10, 1.0 - 1e-10)
    W_diag = mu * (1.0 - mu)
    V = np.diag(1.0 / W_diag) + tau2 * GRM
    V_inv = np.linalg.inv(V)
    ViX = V_inv @ X
    XtViX_inv = np.linalg.inv(X.T @ ViX)
    P0_mat = V_inv - ViX @ XtViX_inv @ ViX.T
    y_adj = P0_mat @ (y - mu)

    def P0_apply(v):
        return P0_mat @ v

    return {
        "tau2": tau2,
        "P0": P0_apply,
        "P0_matrix": P0_mat,
        "V_inv": V_inv,
        "y_adj": y_adj,
        "W_diag": W_diag,
        "mu_hat": mu,
        "beta": beta,
        "converged": converged,
        "n_iter": iteration + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. GENERALIZED FWL PROJECTION (Proposition 1)
# ═══════════════════════════════════════════════════════════════════════════

def build_fwl_projection(P0_mat, Z_main):
    """
    Build generalized FWL projection matrix P_adj (Equation 4).

    P_adj = P0 - P0 Z_main (Z_main^T P0 Z_main)^{-1} Z_main^T P0

    For binary traits, P0 already incorporates IRLS weights through V.
    No explicit W injection is needed (Proposition 1, Remark 1).

    Parameters
    ----------
    P0_mat  : (N, N) null projection matrix.
    Z_main  : (N, m) concatenation of main-effect features [Z_A, Z_B].

    Returns
    -------
    P_adj : (N, N) FWL-adjusted projection matrix.
    """
    P0_Zm = P0_mat @ Z_main
    gram = Z_main.T @ P0_Zm
    gram_inv = np.linalg.inv(gram)
    P_adj = P0_mat - P0_Zm @ gram_inv @ P0_Zm.T
    return P_adj


def verify_fwl_properties(P_adj, Z_main, tol=1e-8):
    """
    Verify Proposition 1 properties: annihilation and idempotency.

    Returns True if P_adj Z_main ≈ 0 and P_adj² ≈ P_adj.
    """
    # Annihilation: P_adj @ Z_main should be zero
    annihilation_err = np.max(np.abs(P_adj @ Z_main))
    # Idempotency: P_adj^2 should equal P_adj
    P_adj_sq = P_adj @ P_adj
    idempotency_err = np.max(np.abs(P_adj_sq - P_adj))
    # Symmetry
    symmetry_err = np.max(np.abs(P_adj - P_adj.T))

    return {
        "annihilation_error": float(annihilation_err),
        "idempotency_error": float(idempotency_err),
        "symmetry_error": float(symmetry_err),
        "passed": annihilation_err < tol and idempotency_err < tol and symmetry_err < tol,
    }
