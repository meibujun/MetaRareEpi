"""
test_config.py — Verify JAX x64 enforcement and package import.
"""

from __future__ import annotations


def test_jax_x64_enabled() -> None:
    """jax_enable_x64 must be True after importing metararepi."""
    import jax.numpy as jnp

    x = jnp.ones(1)
    assert x.dtype == jnp.float64, (
        f"JAX default dtype is {x.dtype}, expected float64. "
        "jax_enable_x64 is not set."
    )


def test_metararepi_import() -> None:
    """metararepi must be importable and expose __version__."""
    import metararepi

    assert hasattr(metararepi, "__version__")
    assert metararepi.__version__ == "0.1.0"
