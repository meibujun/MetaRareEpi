"""
Global configuration — imported FIRST by metararepi.__init__.

Enforces jax_enable_x64 to prevent underflow at extreme SPA distribution
tails (P ≈ 10^{-300}).  Also defines project-wide numerical constants.
"""

from __future__ import annotations

import jax

# ── CRITICAL: Enable float64 globally before any JAX computation ──────────
jax.config.update("jax_enable_x64", True)
JAX_X64_ENABLED: bool = True

# ── Numerical constants ───────────────────────────────────────────────────
DEFAULT_RTOL: float = 1e-7       # default relative tolerance for assertions
DEFAULT_ATOL: float = 1e-12      # default absolute tolerance
SPA_LOG_P_FLOOR: float = -690.0  # log(10^{-300}), floor for log-p values
DEFAULT_N_PROBES: int = 100      # Hutchinson probe count
DEFAULT_SEED: int = 42           # reproducibility seed
