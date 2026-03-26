"""
glmm.py — Generalized Linear Mixed Model (GLMM) Base Engine.

Nature Genetics 2026 · MetaRareEpi Framework (Section 2.2)

Implements the null GLMM under penalized quasi-likelihood (PQL) for
controlling population stratification and cryptic relatedness:

    g(μ) = Xα + u,    u ~ N(0, τ²Φ)

Core operations:
    1. Null projection matrix P₀ computation
    2. Phenotypic covariance V estimation
    3. Whitened residual ỹ = P₀(y − μ̂) computation

All heavy linear algebra is delegated to NumPy/SciPy LAPACK routines
for guaranteed numerical stability.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg


def estimate_grm(G: np.ndarray) -> np.ndarray:
    """
    Compute the Genomic Relationship Matrix (GRM) from genotype matrix.

    GRM = (1/M) Z Z^T, where Z is column-standardised.

    Parameters
    ----------
    G : (N, M) raw genotype matrix (allele counts 0/1/2).

    Returns
    -------
    Phi : (N, N) GRM.
    """
    Z = _standardise_columns(G)
    M = Z.shape[1]
    return Z @ Z.T / M


def compute_V(
    Phi: np.ndarray,
    tau2: float,
    sigma2_e: float = 1.0,
) -> np.ndarray:
    """
    Compute phenotypic covariance matrix V = τ²Φ + σ²ₑI.

    Parameters
    ----------
    Phi     : (N, N) GRM.
    tau2    : polygenic variance component.
    sigma2_e : residual variance (default 1.0 for standardised phenotype).

    Returns
    -------
    V : (N, N) phenotypic covariance matrix.
    """
    N = Phi.shape[0]
    return tau2 * Phi + sigma2_e * np.eye(N, dtype=np.float64)


def compute_P0(
    V: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """
    Compute the null projection matrix P₀.

    P₀ = V⁻¹ − V⁻¹X(X^TV⁻¹X)⁻¹X^TV⁻¹

    This projects into the null space of fixed effects X under the
    V-weighted inner product, as required by REML.

    Parameters
    ----------
    V : (N, N) phenotypic covariance matrix.
    X : (N, p) fixed covariate matrix (intercept + covariates).

    Returns
    -------
    P0 : (N, N) null projection matrix.

    Notes
    -----
    Uses Cholesky factorization for numerical stability.
    """
    N = V.shape[0]

    # V⁻¹ via Cholesky
    cho_V = scipy.linalg.cho_factor(V)
    V_inv_X = scipy.linalg.cho_solve(cho_V, X)     # V⁻¹X, shape (N, p)

    # (X^T V⁻¹ X)⁻¹
    XtVinvX = X.T @ V_inv_X                         # (p, p)
    cho_XtVinvX = scipy.linalg.cho_factor(XtVinvX)

    # P₀ = V⁻¹ − V⁻¹X(X^TV⁻¹X)⁻¹X^TV⁻¹
    V_inv = scipy.linalg.cho_solve(cho_V, np.eye(N))
    correction = V_inv_X @ scipy.linalg.cho_solve(cho_XtVinvX, V_inv_X.T)

    return V_inv - correction


def compute_whitened_residual(
    y: np.ndarray,
    X: np.ndarray,
    P0: np.ndarray,
    mu_hat: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute whitened phenotypic residual ỹ = P₀(y − μ̂).

    Parameters
    ----------
    y      : (N,) phenotype vector.
    X      : (N, p) fixed covariate matrix.
    P0     : (N, N) null projection matrix.
    mu_hat : (N,) estimated mean (if None, uses X @ α̂_OLS).

    Returns
    -------
    y_tilde : (N,) whitened residual vector.
    """
    if mu_hat is None:
        # Simple OLS for fixed effects: α̂ = (X^TX)⁻¹X^Ty
        alpha_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        mu_hat = X @ alpha_hat

    return P0 @ (y - mu_hat)


def estimate_variance_components(
    y: np.ndarray,
    X: np.ndarray,
    Phi: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict[str, float]:
    """
    Estimate variance components (τ², σ²ₑ) via AI-REML.

    Uses the Average Information algorithm for REML estimation
    of the polygenic variance component.

    Parameters
    ----------
    y    : (N,) phenotype.
    X    : (N, p) fixed covariates.
    Phi  : (N, N) GRM.
    max_iter : maximum AI-REML iterations.
    tol  : convergence tolerance.

    Returns
    -------
    dict with "tau2", "sigma2_e", "n_iter", "converged".
    """
    N = y.shape[0]

    # Initialize with moment-based estimates
    y_centered = y - y.mean()
    total_var = float(np.var(y_centered))
    tau2 = total_var * 0.3       # initial h² guess = 0.3
    sigma2_e = total_var * 0.7

    converged = False
    for it in range(max_iter):
        V = compute_V(Phi, tau2, sigma2_e)
        P0 = compute_P0(V, X)

        # Score equations for τ²
        Py = P0 @ y_centered
        PhiPy = Phi @ Py

        # Gradient: ∂ℓ/∂τ² = -0.5[tr(P₀Φ) - y^TP₀Φ P₀y]
        score_tau2 = -0.5 * (np.trace(P0 @ Phi) - Py.T @ PhiPy)

        # Average Information: AI = 0.5 · y^T P Φ P Φ P y
        PPhiPy = P0 @ PhiPy
        ai_tau2 = 0.5 * (Py.T @ Phi @ PPhiPy)
        ai_tau2 = max(ai_tau2, 1e-10)  # guard

        # Update
        delta = float(score_tau2 / ai_tau2)
        tau2_new = max(tau2 + delta, 1e-6)
        sigma2_e = max(total_var - tau2_new, 1e-6)

        if abs(tau2_new - tau2) < tol:
            converged = True
            tau2 = tau2_new
            break
        tau2 = tau2_new

    return {
        "tau2": float(tau2),
        "sigma2_e": float(sigma2_e),
        "h2": float(tau2 / (tau2 + sigma2_e)),
        "n_iter": it + 1,
        "converged": converged,
    }


def null_model_pipeline(
    y: np.ndarray,
    X: np.ndarray | None = None,
    G_common: np.ndarray | None = None,
    Phi: np.ndarray | None = None,
) -> dict:
    """
    Full null model pipeline: estimate variance components → compute P₀ → ỹ.

    Parameters
    ----------
    y        : (N,) phenotype.
    X        : (N, p) fixed covariates (default: intercept only).
    G_common : (N, M) common variant genotypes (for GRM computation).
    Phi      : (N, N) precomputed GRM (overrides G_common).

    Returns
    -------
    dict with "P0", "y_tilde", "V", "variance_components".
    """
    N = y.shape[0]

    # Default: intercept-only model
    if X is None:
        X = np.ones((N, 1), dtype=np.float64)

    # Compute GRM if not provided
    if Phi is None:
        if G_common is not None and G_common.shape[1] > 0:
            Phi = estimate_grm(G_common)
        else:
            # Identity fallback (no relatedness correction)
            Phi = np.eye(N, dtype=np.float64)

    # Estimate variance components
    vc = estimate_variance_components(y, X, Phi)

    # Compute V and P₀
    V = compute_V(Phi, vc["tau2"], vc["sigma2_e"])
    P0 = compute_P0(V, X)

    # Whitened residual
    y_tilde = compute_whitened_residual(y, X, P0)

    return {
        "P0": P0,
        "y_tilde": y_tilde,
        "V": V,
        "variance_components": vc,
    }


def _standardise_columns(G: np.ndarray) -> np.ndarray:
    """Column-wise standardisation: (Z − μ) / σ, with σ floored at 1e-12."""
    Z = G.astype(np.float64)
    mu = Z.mean(axis=0)
    sigma = Z.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (Z - mu) / sigma
