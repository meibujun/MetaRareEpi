"""
simulate_biobank.py — Comprehensive Biobank Simulation Suite

MetaRareEpi R2 Framework (§2.7)

Three experimental configurations:
  1. Human semi-empirical (continuous): 1KGP genotypes, h²=0.3, 5000 common variants
  2. Binary trait with case-control imbalance: 1:5 to 1:100 ratios
  3. Bovine WGS with extreme kinship: F_avg=0.06, large LD blocks

Federated partitioning: 5 superpopulations (AFR, AMR, EAS, EUR, SAS)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from engine_jax import extract_local_cumulants
from metararepi.spa.saddlepoint import spa_pvalue
from metararepi.glmm import fit_null_model, build_fwl_projection
from metararepi.nlgc import build_augmented_null, genomic_inflation_factor


# ═══════════════════════════════════════════════════════════════════════════
# 1. GENOTYPE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def simulate_rare_genotypes(
    N: int,
    m_A: int = 20,
    m_B: int = 20,
    maf_max: float = 0.01,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate rare-variant genotype matrices.

    MAF < 0.01 for all variants (rare variant regime).
    Standardized to mean 0, sd 1.
    """
    rng = np.random.default_rng(seed)

    def _make_block(N, m, rng):
        mafs = rng.uniform(0.001, maf_max, size=m)
        G = rng.binomial(2, mafs, size=(N, m)).astype(np.float64)
        # Standardize
        means = G.mean(axis=0)
        stds = G.std(axis=0)
        stds[stds < 1e-8] = 1.0  # avoid division by zero
        return (G - means) / stds

    Z_A = _make_block(N, m_A, rng)
    Z_B = _make_block(N, m_B, rng)
    return Z_A, Z_B


def simulate_grm(N: int, seed: int = 0) -> np.ndarray:
    """Simulate a GRM (positive definite, diagonal ≈ 1)."""
    rng = np.random.default_rng(seed)
    # Low-rank + diagonal
    k = min(50, N)
    L = rng.normal(0, 1 / np.sqrt(k), size=(N, k))
    GRM = L @ L.T + 0.01 * np.eye(N)
    # Normalize diagonal to 1
    d = np.sqrt(np.diag(GRM))
    GRM = GRM / np.outer(d, d)
    return GRM


def simulate_grm_with_inbreeding(
    N: int,
    F_avg: float = 0.06,
    seed: int = 0,
) -> np.ndarray:
    """
    Simulate a GRM with controlled inbreeding coefficient.

    For bovine populations: F_avg ≈ 0.06, extensive off-diagonal kinship.
    """
    rng = np.random.default_rng(seed)
    k = min(100, N)
    L = rng.normal(0, 1 / np.sqrt(k), size=(N, k))

    # Add kinship structure
    n_breeds = max(5, N // 100)
    breed_assign = rng.integers(0, n_breeds, size=N)
    breed_effect = np.zeros((N, n_breeds))
    for i, b in enumerate(breed_assign):
        breed_effect[i, b] = np.sqrt(F_avg * 3)

    GRM = L @ L.T + breed_effect @ breed_effect.T + F_avg * np.eye(N)
    d = np.sqrt(np.diag(GRM))
    GRM = GRM / np.outer(d, d)
    return GRM


# ═══════════════════════════════════════════════════════════════════════════
# 2. PHENOTYPE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def simulate_continuous_phenotype(
    N: int,
    GRM: np.ndarray,
    *,
    h2: float = 0.3,
    n_causal: int = 5000,
    epi_variance: float = 0.0,
    Z_A: np.ndarray | None = None,
    Z_B: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate continuous phenotype under polygenic model.

    y = Xα + u + ε_epi + ε
    u ~ N(0, h²·Φ), ε ~ N(0, (1-h²)·I)

    Returns (y, X) where X includes intercept.
    """
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(N), rng.normal(0, 1, (N, 2))])  # intercept + 2 covariates

    # Polygenic effect
    L = np.linalg.cholesky(GRM + 1e-6 * np.eye(N))
    u = L @ rng.normal(0, np.sqrt(h2), N)

    # Epistatic effect (only if Z_A, Z_B provided and epi_variance > 0)
    epi = np.zeros(N)
    if epi_variance > 0 and Z_A is not None and Z_B is not None:
        m_A, m_B = Z_A.shape[1], Z_B.shape[1]
        # Interaction effects
        beta_epi = rng.normal(0, np.sqrt(epi_variance / (m_A * m_B)), (m_A, m_B))
        for a in range(m_A):
            for b in range(m_B):
                epi += Z_A[:, a] * Z_B[:, b] * beta_epi[a, b]

    # Residual
    residual = rng.normal(0, np.sqrt(1.0 - h2 - epi_variance), N)
    y = u + epi + residual

    return y, X


def simulate_binary_phenotype(
    N: int,
    GRM: np.ndarray,
    *,
    prevalence: float = 0.01,
    h2_liability: float = 0.3,
    epi_variance: float = 0.0,
    Z_A: np.ndarray | None = None,
    Z_B: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Simulate binary phenotype via liability threshold model.

    Prevalence controls the case-control ratio:
      prevalence=0.167 → ~1:5 ratio
      prevalence=0.091 → ~1:10 ratio
      prevalence=0.048 → ~1:20 ratio
      prevalence=0.020 → ~1:50 ratio
      prevalence=0.010 → ~1:100 ratio

    Returns (y_binary, X, achieved_ratio) where y ∈ {0, 1}.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(N), rng.normal(0, 1, (N, 2))])

    # Liability
    L = np.linalg.cholesky(GRM + 1e-6 * np.eye(N))
    u = L @ rng.normal(0, np.sqrt(h2_liability), N)

    epi = np.zeros(N)
    if epi_variance > 0 and Z_A is not None and Z_B is not None:
        m_A, m_B = Z_A.shape[1], Z_B.shape[1]
        beta_epi = rng.normal(0, np.sqrt(epi_variance / (m_A * m_B)), (m_A, m_B))
        for a in range(m_A):
            for b in range(m_B):
                epi += Z_A[:, a] * Z_B[:, b] * beta_epi[a, b]

    residual = rng.normal(0, np.sqrt(1.0 - h2_liability - epi_variance), N)
    liability = u + epi + residual

    # Threshold
    threshold = norm.ppf(1.0 - prevalence)
    y = (liability > threshold).astype(np.float64)

    n_cases = int(np.sum(y))
    n_controls = N - n_cases
    ratio = n_controls / max(n_cases, 1)

    return y, X, ratio


# ═══════════════════════════════════════════════════════════════════════════
# 3. TYPE I ERROR EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def run_type1_error_experiment(
    N: int = 2000,
    n_tests: int = 1000,
    method: str = "hutchpp",
    trait_type: str = "continuous",
    prevalence: float = 0.01,
    seed: int = 0,
) -> dict:
    """
    Evaluate Type I error under the epistatic null.

    Under H0, no epistatic interaction — all signals should be null.
    """
    print(f"Running Type I error: N={N}, tests={n_tests}, trait={trait_type}, method={method}")
    rng = np.random.default_rng(seed)
    GRM = simulate_grm(N, seed=seed)
    pvalues = []

    for test_idx in range(n_tests):
        test_seed = seed + test_idx * 137

        Z_A, Z_B = simulate_rare_genotypes(N, seed=test_seed)

        if trait_type == "continuous":
            y, X = simulate_continuous_phenotype(N, GRM, seed=test_seed)
        else:
            y, X, ratio = simulate_binary_phenotype(
                N, GRM, prevalence=prevalence, seed=test_seed,
            )

        # Fit null model
        null_model = fit_null_model(y, X, GRM, trait_type=trait_type)

        # FWL projection
        Z_main = np.column_stack([Z_A, Z_B])
        P_adj = build_fwl_projection(null_model["P0_matrix"], Z_main)

        # Adjusted residual
        y_adj = P_adj @ (y - null_model["mu_hat"])

        # Extract cumulants
        result = extract_local_cumulants(
            Z_A, Z_B,
            method=method,
            n_probes=100,
            seed=test_seed,
            y=y_adj,
            apply_fwl=True,
        )

        # SPA p-value
        Q = result.get("Q_adj", 0.0)
        spa_result = spa_pvalue(Q, result["cumulants"])
        pvalues.append(spa_result["pvalue"])

        if (test_idx + 1) % 100 == 0:
            print(f"  Completed {test_idx + 1}/{n_tests} tests")

    pvalues = np.array(pvalues)

    # Empirical Type I error at various thresholds
    alphas = [0.05, 0.01, 1e-3, 1e-4, 1e-5, 1e-6]
    empirical_errors = {}
    for alpha in alphas:
        empirical_errors[f"alpha_{alpha}"] = float(np.mean(pvalues < alpha))

    lambda_gc = genomic_inflation_factor(pvalues[pvalues > 0])

    return {
        "pvalues": pvalues,
        "empirical_errors": empirical_errors,
        "lambda_gc": lambda_gc,
        "n_tests": n_tests,
        "N": N,
        "method": method,
        "trait_type": trait_type,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. POWER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def run_power_experiment(
    N: int = 2000,
    n_tests: int = 200,
    epi_variances: list[float] | None = None,
    method: str = "hutchpp",
    alpha: float = 5e-6,
    seed: int = 0,
) -> dict:
    """Evaluate power under varying epistatic effect sizes."""
    if epi_variances is None:
        epi_variances = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05]

    rng = np.random.default_rng(seed)
    GRM = simulate_grm(N, seed=seed)
    results = {}

    for epi_var in epi_variances:
        rejections = 0
        for test_idx in range(n_tests):
            test_seed = seed + test_idx * 137 + int(epi_var * 1e6)
            Z_A, Z_B = simulate_rare_genotypes(N, seed=test_seed)
            y, X = simulate_continuous_phenotype(
                N, GRM, epi_variance=epi_var,
                Z_A=Z_A, Z_B=Z_B, seed=test_seed,
            )

            null_model = fit_null_model(y, X, GRM, trait_type="continuous")
            Z_main = np.column_stack([Z_A, Z_B])
            P_adj = build_fwl_projection(null_model["P0_matrix"], Z_main)
            y_adj = P_adj @ (y - null_model["mu_hat"])

            result = extract_local_cumulants(
                Z_A, Z_B, method=method, n_probes=100,
                seed=test_seed, y=y_adj, apply_fwl=True,
            )
            Q = result.get("Q_adj", 0.0)
            spa_result = spa_pvalue(Q, result["cumulants"])

            if spa_result["pvalue"] < alpha:
                rejections += 1

        power = rejections / n_tests
        results[epi_var] = power
        print(f"  epi_var={epi_var:.4f}: power={power:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 5. SCALABILITY BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════

def run_scalability_benchmark(
    sample_sizes: list[int] | None = None,
    n_gene_pairs: int = 100,
    seed: int = 0,
) -> dict:
    """Benchmark wall-clock time and memory across sample sizes."""
    import tracemalloc

    if sample_sizes is None:
        sample_sizes = [500, 1000, 2000, 5000, 10000, 20000]

    results = {}
    for N in sample_sizes:
        print(f"  Benchmarking N={N}...")
        Z_A, Z_B = simulate_rare_genotypes(N, seed=seed)

        tracemalloc.start()
        t0 = time.time()

        for _ in range(n_gene_pairs):
            extract_local_cumulants(
                Z_A, Z_B, method="hutchpp",
                n_probes=50, seed=seed,
            )

        elapsed = time.time() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results[N] = {
            "wall_clock_sec": elapsed,
            "peak_memory_mb": peak_mem / 1e6,
            "time_per_pair_ms": elapsed / n_gene_pairs * 1000,
        }
        print(f"    Time: {elapsed:.2f}s, Peak mem: {peak_mem/1e6:.1f} MB")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 6. FEDERATED VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def run_federated_validation(
    N: int = 2000,
    n_nodes: int = 5,
    n_tests: int = 200,
    seed: int = 0,
) -> dict:
    """
    Validate federated ≡ centralized by comparing p-values.

    Partitions the cohort into n_nodes and compares Fed-cSPA
    against Mega-SPA on pooled data.
    """
    from federated_spa import federated_spa_plaintext

    rng = np.random.default_rng(seed)
    GRM = simulate_grm(N, seed=seed)

    fed_pvalues = []
    central_pvalues = []

    for test_idx in range(n_tests):
        test_seed = seed + test_idx * 137
        Z_A, Z_B = simulate_rare_genotypes(N, seed=test_seed)
        y, X = simulate_continuous_phenotype(N, GRM, seed=test_seed)

        # Centralized
        null_model = fit_null_model(y, X, GRM)
        Z_main = np.column_stack([Z_A, Z_B])
        P_adj = build_fwl_projection(null_model["P0_matrix"], Z_main)
        y_adj = P_adj @ (y - null_model["mu_hat"])

        central_result = extract_local_cumulants(
            Z_A, Z_B, method="hutchpp", n_probes=100,
            seed=test_seed, y=y_adj, apply_fwl=True,
        )
        central_spa = spa_pvalue(
            central_result.get("Q_adj", 0.0),
            central_result["cumulants"],
        )
        central_pvalues.append(central_spa["pvalue"])

        # Federated
        node_size = N // n_nodes
        node_results = []
        for k in range(n_nodes):
            start = k * node_size
            end = start + node_size if k < n_nodes - 1 else N
            Z_A_k = Z_A[start:end]
            Z_B_k = Z_B[start:end]
            y_k = y[start:end]
            X_k = X[start:end]

            # Simplified local computation (reusing global GRM subset)
            GRM_k = GRM[start:end, start:end]
            local_null = fit_null_model(y_k, X_k, GRM_k)
            Z_main_k = np.column_stack([Z_A_k, Z_B_k])
            P_adj_k = build_fwl_projection(local_null["P0_matrix"], Z_main_k)
            y_adj_k = P_adj_k @ (y_k - local_null["mu_hat"])

            local_result = extract_local_cumulants(
                Z_A_k, Z_B_k, method="hutchpp", n_probes=100,
                seed=test_seed + k, y=y_adj_k, apply_fwl=True,
            )
            node_results.append(local_result)

        fed_result = federated_spa_plaintext(node_results)
        fed_pvalues.append(fed_result["pvalue"])

    fed_pvalues = np.array(fed_pvalues)
    central_pvalues = np.array(central_pvalues)

    # Correlation
    mask = (fed_pvalues > 0) & (central_pvalues > 0)
    log_fed = -np.log10(fed_pvalues[mask] + 1e-300)
    log_central = -np.log10(central_pvalues[mask] + 1e-300)
    r2 = float(np.corrcoef(log_fed, log_central)[0, 1] ** 2) if len(log_fed) > 2 else 0.0

    return {
        "fed_pvalues": fed_pvalues,
        "central_pvalues": central_pvalues,
        "R2": r2,
        "n_tests": n_tests,
        "n_nodes": n_nodes,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. BINARY TRAIT IMBALANCE EXPERIMENT (Fig 5A)
# ═══════════════════════════════════════════════════════════════════════════

def run_binary_imbalance_experiment(
    N: int = 5000,
    n_tests: int = 500,
    prevalences: list[float] | None = None,
    seed: int = 0,
) -> dict:
    """
    Type I error under varying case-control imbalance (Table 3, Figure 5A).

    Prevalences map to case-control ratios:
      0.167 → 1:5, 0.091 → 1:10, 0.048 → 1:20, 0.020 → 1:50, 0.010 → 1:100
    """
    if prevalences is None:
        prevalences = [0.167, 0.091, 0.048, 0.020, 0.010]

    results = {}
    for prev in prevalences:
        print(f"  Prevalence={prev:.3f} ...")
        exp = run_type1_error_experiment(
            N=N, n_tests=n_tests, method="hutchpp",
            trait_type="binary", prevalence=prev, seed=seed,
        )
        ratio = f"1:{int(round((1-prev)/prev))}"
        results[ratio] = exp
        print(f"    λ_GC={exp['lambda_gc']:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("MetaRareEpi R2 — Comprehensive Simulation Suite")
    print("=" * 70)

    results = {}

    # Quick validation runs (reduced parameters for feasibility)
    print("\n[1/5] Type I error (continuous, Hutch++)...")
    results["type1_continuous"] = run_type1_error_experiment(
        N=500, n_tests=200, method="hutchpp", seed=42,
    )

    print(f"\n[2/5] Type I error (binary, 1:10 imbalance)...")
    results["type1_binary"] = run_type1_error_experiment(
        N=500, n_tests=200, method="hutchpp",
        trait_type="binary", prevalence=0.091, seed=42,
    )

    print("\n[3/5] Federated validation...")
    results["federated"] = run_federated_validation(
        N=500, n_nodes=5, n_tests=100, seed=42,
    )

    print("\n[4/5] Scalability benchmark...")
    results["scalability"] = run_scalability_benchmark(
        sample_sizes=[200, 500, 1000, 2000], n_gene_pairs=50, seed=42,
    )

    print("\n[5/5] Power analysis...")
    results["power"] = run_power_experiment(
        N=500, n_tests=100, seed=42,
    )

    # Save summary
    summary = {
        "type1_continuous_lambda_gc": results["type1_continuous"]["lambda_gc"],
        "type1_binary_lambda_gc": results["type1_binary"]["lambda_gc"],
        "federated_R2": results["federated"]["R2"],
        "scalability": {
            str(k): v for k, v in results["scalability"].items()
        },
    }

    output_path = Path(__file__).parent.parent / "results" / "simulation_summary.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Results saved to {output_path}")
    print(f"Continuous λ_GC: {summary['type1_continuous_lambda_gc']:.4f}")
    print(f"Binary λ_GC:     {summary['type1_binary_lambda_gc']:.4f}")
    print(f"Federated R²:    {summary['federated_R2']:.6f}")
    print(f"{'='*70}")
