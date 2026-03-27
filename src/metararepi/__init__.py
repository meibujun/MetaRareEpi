"""
MetaRareEpi — Federated Rare-Variant Epistasis Meta-Analysis Framework.

Dual-Space Federated SPA with Deflation-Accelerated Cumulant Extraction
for Biobank-Scale Rare-Variant Epistasis Mapping.

Core innovations (R2):
  1. Symmetric dual-space Khatri-Rao reformulation (Theorem 1)
  2. Hutch++ deflation-accelerated trace estimation (Algorithm 1)
  3. Generalized FWL orthogonalization for binary traits (Proposition 1)
  4. Non-linear genomic control via sparse Hadamard-squared GRM
  5. CKKS homomorphic encryption for privacy-preserving federation
  6. Graph-regularized search space optimization (PPI/TAD/pathway)

All operations on K_epi are performed IMPLICITLY via dual-space MVM to
maintain O(N · m_A · m_B) space-time complexity. No N×N matrix is ever formed.
"""

from metararepi._config import JAX_X64_ENABLED  # noqa: F401  — side-effect import

__version__ = "2.0.0"
__all__ = [
    "__version__",
    "kernel",
    "spa",
    "federated",
    "io",
    "glmm",
    "nlgc",
    "weighting",
    "graph_search",
]
