"""
MetaRareEpi — Federated rare-variant epistasis meta-analysis framework.

Targets Nature Genetics 2026.

Core mathematical object:

    K_epi = (Z_A Z_Aᵀ) ⊙ (Z_B Z_Bᵀ)

All operations on K_epi are performed IMPLICITLY via Fast-MVM to maintain
O(N · m_A · m_B) space-time complexity.  No N×N dense matrix is ever formed.
"""

from metararepi._config import JAX_X64_ENABLED  # noqa: F401  — side-effect import

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "kernel",
    "spa",
    "federated",
    "io",
    "glmm",
    "weighting",
]
