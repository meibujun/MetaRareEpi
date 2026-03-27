#!/usr/bin/env python
"""
benchmark_type1_error.py �?Type I Error Calibration Benchmark

Reproduces Figure 2 and Table 2 from the R2 manuscript.

Evaluates empirical Type I error rates under the null hypothesis (H₀: τ²_epi = 0)
with injected main effects across multiple significance thresholds.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("benchmark_type1_error")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _standardise(Z: np.ndarray) -> np.ndarray:
    mu = Z.mean(axis=0)
    sigma = Z.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (Z - mu) / sigma


def simulate_null_pvalues(
    N: int = 5000,
    m_A: int = 20,
    m_B: int = 20,
    n_tests: int = 1000,
    h2_main: float = 0.05,
    h2_poly: float = 0.40,
    seed: int = 42,
) -> dict:
    """
    Simulate null epistasis tests with injected main effects.

    Returns Fed-cSPA and asymptotic p-values for Q-Q calibration.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    from engine_jax import extract_local_cumulants
    from federated_spa import federated_spa_pvalue

    rng = np.random.default_rng(seed)
    pvals_spa = []
    pvals_asym = []

    for t in range(n_tests):
        # Generate rare variant genotypes
        G_rare = rng.binomial(2, rng.uniform(0.001, 0.01, size=m_A + m_B), size=(N, m_A + m_B))
        Z_A = _standardise(G_rare[:, :m_A].astype(np.float64))
        Z_B = _standardise(G_rare[:, m_A:].astype(np.float64))

        # Phenotype under H₀ (no epistasis, but with main effects)
        beta_main = rng.standard_normal(m_A) * np.sqrt(h2_main / m_A)
        u_main = G_rare[:, :m_A].astype(np.float64) @ beta_main
        G_common = rng.binomial(2, 0.3, size=(N, 100)).astype(np.float64)
        beta_poly = rng.standard_normal(100) * np.sqrt(h2_poly / 100)
        u_poly = G_common @ beta_poly
        sigma_e = np.sqrt(1.0 - h2_main - h2_poly)
        y = u_poly + u_main + sigma_e * rng.standard_normal(N)

        # Extract cumulants with FWL
        result = extract_local_cumulants(
            Z_A, Z_B, method="exact", y=y, apply_fwl=True,
        )
        kappa = result["cumulants"]
        Q = result.get("Q_adj", float(kappa[0]))

        # Fed-cSPA p-value
        spa_result = federated_spa_pvalue(Q, [kappa])
        pvals_spa.append(spa_result["pvalue"])

        # Asymptotic p-value (Gaussian approximation)
        if kappa[1] > 0:
            z = (Q - kappa[0]) / np.sqrt(kappa[1])
            pvals_asym.append(float(stats.norm.sf(z)))
        else:
            pvals_asym.append(0.5)

        if (t + 1) % 100 == 0:
            log.info("Test %d/%d complete", t + 1, n_tests)

    return {
        "pvals_spa": np.array(pvals_spa),
        "pvals_asym": np.array(pvals_asym),
    }


def compute_type1_error_rates(pvals: np.ndarray, thresholds: list[float]) -> dict:
    """Compute empirical Type I error rates at given significance thresholds."""
    rates = {}
    for alpha in thresholds:
        rate = np.mean(pvals < alpha)
        rates[f"alpha_{alpha:.0e}"] = rate
    return rates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--n-tests", type=int, default=500)
    parser.add_argument("--output", default="results/type1_error.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = simulate_null_pvalues(N=args.N, n_tests=args.n_tests, seed=args.seed)

    thresholds = [1e-2, 1e-3, 1e-4]
    spa_rates = compute_type1_error_rates(result["pvals_spa"], thresholds)
    asym_rates = compute_type1_error_rates(result["pvals_asym"], thresholds)

    # Write results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha in thresholds:
        key = f"alpha_{alpha:.0e}"
        rows.append({
            "threshold": alpha,
            "fed_cspa_rate": spa_rates[key],
            "asymptotic_rate": asym_rates[key],
        })

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    log.info("Type I error results written to %s", args.output)

    # Also save raw p-values for Q-Q plot
    pval_path = args.output.replace(".csv", "_pvalues.npz")
    np.savez(pval_path, pvals_spa=result["pvals_spa"], pvals_asym=result["pvals_asym"])
    log.info("Raw p-values saved to %s", pval_path)


if __name__ == "__main__":
    main()
