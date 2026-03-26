"""
metararepi.kernel — Epistatic kernel Fast-MVM engine.

Core operation: K_epi @ v = (Z_A Z_Aᵀ) ⊙ (Z_B Z_Bᵀ) @ v
computed IMPLICITLY via two-step einsum in O(N · m_A · m_B).
"""

from metararepi.kernel.fast_mvm import (  # noqa: F401
    epi_kernel_matvec,
    epi_kernel_matvec_batch,
    extract_traces_exact,
    extract_traces_hutchinson,
)
