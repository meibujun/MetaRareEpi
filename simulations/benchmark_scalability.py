#!/usr/bin/env python
"""
benchmark_scalability.py — Computational Scalability Benchmarking

Reproduces Figure 1 and Table 1 from the Nature Genetics manuscript.

Benchmarks:
    1. Standard-EVD (O(N³)) — explicit dense kernel + eigendecomposition
    2. MetaRareEpi Fast-MVM (O(N)) — implicit micro-gram traces

Outputs structured CSV data for Figure 1 generation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("benchmark_scalability")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _standardise(Z: np.ndarray) -> np.ndarray:
    mu = Z.mean(axis=0)
    sigma = Z.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (Z - mu) / sigma


def benchmark_standard_evd(N: int, m_A: int, m_B: int, seed: int) -> dict:
    """O(N³) explicit kernel eigendecomposition."""
    rng = np.random.default_rng(seed)
    Z_A = _standardise(rng.integers(0, 3, size=(N, m_A)).astype(np.float64))
    Z_B = _standardise(rng.integers(0, 3, size=(N, m_B)).astype(np.float64))

    t0 = time.perf_counter()
    try:
        K_epi = (Z_A @ Z_A.T) * (Z_B @ Z_B.T)
        mem_bytes = K_epi.nbytes
        eigenvalues = np.linalg.eigvalsh(K_epi)
        traces = np.array([np.sum(eigenvalues ** p) for p in range(1, 5)])
        elapsed = time.perf_counter() - t0
        return {
            "method": "Standard-EVD", "N": N, "runtime_s": elapsed,
            "peak_memory_gb": mem_bytes / 1e9, "success": True,
            "cumulants_order": 4,
        }
    except MemoryError:
        elapsed = time.perf_counter() - t0
        return {
            "method": "Standard-EVD", "N": N, "runtime_s": elapsed,
            "peak_memory_gb": float("inf"), "success": False,
            "cumulants_order": 0,
        }
    finally:
        gc.collect()


def benchmark_metararepi(N: int, m_A: int, m_B: int, seed: int) -> dict:
    """O(N · m_A · m_B) implicit Fast-MVM traces."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from engine_jax import extract_local_cumulants

    rng = np.random.default_rng(seed)
    Z_A = _standardise(rng.integers(0, 3, size=(N, m_A)).astype(np.float64))
    Z_B = _standardise(rng.integers(0, 3, size=(N, m_B)).astype(np.float64))

    # Warm-up JIT
    _ = extract_local_cumulants(Z_A[:10], Z_B[:10], method="exact")

    t0 = time.perf_counter()
    result = extract_local_cumulants(Z_A, Z_B, method="exact", max_power=4)
    elapsed = time.perf_counter() - t0

    # Estimate memory: Z_A + Z_B + H + G
    d = m_A * m_B
    mem_bytes = (N * m_A + N * m_B + N * d + d * d) * 8

    return {
        "method": "MetaRareEpi", "N": N, "runtime_s": elapsed,
        "peak_memory_gb": mem_bytes / 1e9, "success": True,
        "cumulants_order": 4,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/benchmark_scalability.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m-A", type=int, default=20)
    parser.add_argument("--m-B", type=int, default=20)
    args = parser.parse_args()

    N_range = [1000, 5000, 10000, 20000, 50000, 100000]
    N_range_evd = [1000, 5000, 10000, 20000]  # EVD OOMs beyond this

    results = []
    for N in N_range_evd:
        log.info("Standard-EVD: N=%d", N)
        r = benchmark_standard_evd(N, args.m_A, args.m_B, args.seed)
        results.append(r)
        log.info("  → %.2f s, %.2f GB, success=%s", r["runtime_s"], r["peak_memory_gb"], r["success"])

    for N in N_range:
        log.info("MetaRareEpi: N=%d", N)
        r = benchmark_metararepi(N, args.m_A, args.m_B, args.seed)
        results.append(r)
        log.info("  → %.2f s, %.2f GB", r["runtime_s"], r["peak_memory_gb"])

    # Write CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    log.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
