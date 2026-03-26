"""
weighting.py — Deep Prior-Elicited Variant Weighting.

Nature Genetics 2026 · MetaRareEpi Framework (Section 2.2)

Rather than treating all rare variants equally using arbitrary MAF
thresholds, this module deploys a modular annotation scoring system
to weight mutations via multi-omics annotations (e.g., CADD,
AlphaMissense, evolutionary conservation).

For genomic regions A and B, the functionally weighted sparse feature
matrices are:

    Z_A = G_A @ W_A ∈ ℝ^{N × m_A}
    Z_B = G_B @ W_B ∈ ℝ^{N × m_B}

The module provides pluggable backends:
    - CADDScorer: CADD phred-scaled pathogenicity scores
    - AlphaMissenseScorer: AlphaMissense missense pathogenicity
    - BetaWeighter: Beta(MAF; a1, a2) density weighting (Wu et al. 2011)
    - UniformWeighter: equal weights (baseline)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import beta as beta_dist


# ═══════════════════════════════════════════════════════════════════════════
# Base class
# ═══════════════════════════════════════════════════════════════════════════

class AnnotationScorer(ABC):
    """Abstract base for variant annotation scoring backends."""

    @abstractmethod
    def score(self, mafs: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute variant-level pathogenicity/importance scores.

        Parameters
        ----------
        mafs : (m,) minor allele frequencies.
        **kwargs : additional annotation data (e.g., CADD scores).

        Returns
        -------
        weights : (m,) non-negative scores, one per variant.
        """
        ...

    def compute_weight_matrix(
        self, mafs: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Compute the diagonal weight matrix W = diag(scores).

        Parameters
        ----------
        mafs : (m,) minor allele frequencies.

        Returns
        -------
        W : (m, m) diagonal weight matrix.
        """
        scores = self.score(mafs, **kwargs)
        return np.diag(scores)


# ═══════════════════════════════════════════════════════════════════════════
# Concrete implementations
# ═══════════════════════════════════════════════════════════════════════════

class UniformWeighter(AnnotationScorer):
    """Equal weights for all variants (baseline comparator)."""

    def score(self, mafs: np.ndarray, **kwargs) -> np.ndarray:
        return np.ones(len(mafs), dtype=np.float64)


class BetaWeighter(AnnotationScorer):
    """
    Beta(MAF; a₁, a₂) density weighting (Wu et al. 2011, SKAT).

    Default parameters a₁=1, a₂=25 upweight rare variants.
    """

    def __init__(self, a1: float = 1.0, a2: float = 25.0) -> None:
        self.a1 = a1
        self.a2 = a2

    def score(self, mafs: np.ndarray, **kwargs) -> np.ndarray:
        mafs_safe = np.clip(mafs, 1e-10, 1.0 - 1e-10)
        return beta_dist.pdf(mafs_safe, self.a1, self.a2)


class CADDScorer(AnnotationScorer):
    """
    CADD phred-scaled pathogenicity scoring.

    Uses precomputed CADD scores; falls back to Beta(MAF) weighting
    if no scores are provided.

    The score is normalised to [0, 1] via:
        w_j = 1 - 10^(-CADD_j / 10)
    """

    def __init__(self, fallback_a1: float = 1.0, fallback_a2: float = 25.0):
        self._fallback = BetaWeighter(fallback_a1, fallback_a2)

    def score(self, mafs: np.ndarray, **kwargs) -> np.ndarray:
        cadd_scores = kwargs.get("cadd_scores")
        if cadd_scores is not None:
            cadd = np.asarray(cadd_scores, dtype=np.float64)
            return 1.0 - np.power(10.0, -cadd / 10.0)
        return self._fallback.score(mafs)


class AlphaMissenseScorer(AnnotationScorer):
    """
    AlphaMissense missense pathogenicity scoring (Cheng et al. 2023).

    Uses precomputed AlphaMissense probabilities directly as weights;
    falls back to Beta(MAF) if not provided.
    """

    def __init__(self, fallback_a1: float = 1.0, fallback_a2: float = 25.0):
        self._fallback = BetaWeighter(fallback_a1, fallback_a2)

    def score(self, mafs: np.ndarray, **kwargs) -> np.ndarray:
        am_scores = kwargs.get("alphamissense_scores")
        if am_scores is not None:
            return np.asarray(am_scores, dtype=np.float64)
        return self._fallback.score(mafs)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def apply_weights(
    G: np.ndarray,
    weights: np.ndarray,
    standardise: bool = True,
) -> np.ndarray:
    """
    Apply variant weights to a genotype matrix: Z = G @ diag(w).

    Parameters
    ----------
    G       : (N, m) raw genotype matrix (allele counts).
    weights : (m,) variant-level weights.
    standardise : if True, column-standardise Z after weighting.

    Returns
    -------
    Z : (N, m) weighted (and optionally standardised) feature matrix.
    """
    G_f = G.astype(np.float64)
    Z = G_f * weights[np.newaxis, :]

    if standardise:
        mu = Z.mean(axis=0)
        sigma = Z.std(axis=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        Z = (Z - mu) / sigma

    return Z


def compute_weighted_features(
    G_A: np.ndarray,
    G_B: np.ndarray,
    mafs_A: np.ndarray,
    mafs_B: np.ndarray,
    scorer: AnnotationScorer | None = None,
    **annotation_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full weighting pipeline: score → weight → standardise.

    Parameters
    ----------
    G_A, G_B     : (N, m_A), (N, m_B) genotype matrices.
    mafs_A, mafs_B : (m_A,), (m_B,) minor allele frequencies.
    scorer       : AnnotationScorer backend (default: BetaWeighter).
    **annotation_kwargs : passed to scorer.score().

    Returns
    -------
    Z_A, Z_B : (N, m_A), (N, m_B) weighted standardised feature matrices.
    """
    if scorer is None:
        scorer = BetaWeighter()

    w_A = scorer.score(mafs_A, **annotation_kwargs)
    w_B = scorer.score(mafs_B, **annotation_kwargs)

    Z_A = apply_weights(G_A, w_A)
    Z_B = apply_weights(G_B, w_B)

    return Z_A, Z_B
