#!/usr/bin/env python
"""
evaluate_federated.py — Deploy Fed-cSPA across K decentralised assessment centres.

Pipeline:
    1. Spin up K Ray Actors, each loaded with one Zarr v3 genotype shard.
    2. Each actor computes local spectral cumulants κ₁–κ₄ of the epistatic
       kernel K_epi = (Z_A Z_Aᵀ) ⊙ (Z_B Z_Bᵀ) using the JAX engine.
    3. The central aggregator sums cumulants (Cumulant Additivity Theorem)
       and applies the Lugannani-Rice SPA for each variant pair.
    4. An asymptotic χ²-baseline p-value is also computed for comparison.
    5. Results are written to a Parquet file for downstream visualization.

Requirements: ray >= 2.40, jax, zarr >= 3.0, numpy, pyarrow
Note: Requires Python <= 3.13 (Ray has no cp314 wheels as of 2026-03).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate_federated")


# ═══════════════════════════════════════════════════════════════════════════
# Ray Actor — runs on each assessment centre node
# ═══════════════════════════════════════════════════════════════════════════

def _make_actor_class():
    """Lazily define the Ray actor to defer ray import."""
    import ray
    import zarr
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    # Import the engine (add src/ to path)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from engine_jax import extract_local_cumulants

    @ray.remote
    class AssessmentCentreActor:
        """
        Federated node: holds a private genotype shard, exposes only cumulants.
        """

        def __init__(self, centre_id: int, zarr_path: str) -> None:
            self.centre_id = centre_id
            self.zarr_path = zarr_path
            self._G: np.ndarray | None = None
            self._mafs: np.ndarray | None = None

        def load_data(self) -> dict:
            """Load genotype shard from Zarr v3 store."""
            store = zarr.open_group(self.zarr_path, mode="r")
            self._G = np.asarray(store["genotypes"][:], dtype=np.float64)
            self._mafs = np.asarray(store["mafs"][:], dtype=np.float64)
            n, m = self._G.shape
            return {"centre_id": self.centre_id, "n_samples": n, "n_variants": m}

        def compute_cumulants(
            self,
            block_A_idx: np.ndarray,
            block_B_idx: np.ndarray,
        ) -> dict:
            """
            Compute local cumulants for a specific pair of variant blocks.

            Parameters
            ----------
            block_A_idx : variant column indices for locus block A.
            block_B_idx : variant column indices for locus block B.

            Returns
            -------
            dict with "centre_id", "n_local", "cumulants" (4,).
            """
            # Standardise columns
            Z_A = self._standardise(self._G[:, block_A_idx])
            Z_B = self._standardise(self._G[:, block_B_idx])

            result = extract_local_cumulants(
                Z_A, Z_B, max_power=4, method="exact",
            )
            return {
                "centre_id": self.centre_id,
                "n_local": Z_A.shape[0],
                "cumulants": result["cumulants"],
            }

        @staticmethod
        def _standardise(Z: np.ndarray) -> np.ndarray:
            mu = Z.mean(axis=0)
            sigma = Z.std(axis=0)
            sigma = np.where(sigma < 1e-12, 1.0, sigma)
            return (Z - mu) / sigma

    return AssessmentCentreActor


# ═══════════════════════════════════════════════════════════════════════════
# Central aggregator — orchestrates federated computation
# ═══════════════════════════════════════════════════════════════════════════

def run_federated_evaluation(
    zarr_dir: str,
    n_centres: int,
    seed: int,
    n_pairs: int = 200,
    block_size: int = 10,
) -> list[dict]:
    """
    Deploy K Ray actors, compute cumulants for n_pairs random variant-block
    pairs, aggregate, and compute both Fed-cSPA and asymptotic p-values.
    """
    import ray

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from federated_spa import aggregate_cumulants, federated_spa_pvalue

    ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)
    ActorClass = _make_actor_class()

    # ── Spin up actors ────────────────────────────────────────────────────
    actors = []
    for k in range(n_centres):
        zarr_path = str(Path(zarr_dir) / f"centre_{k}.zarr")
        actor = ActorClass.remote(k, zarr_path)
        actors.append(actor)

    # Load data on all nodes
    load_futures = [a.load_data.remote() for a in actors]
    load_results = ray.get(load_futures)
    for r in load_results:
        log.info("Centre %d: %d samples, %d variants", **r)

    n_variants = load_results[0]["n_variants"]
    rng = np.random.default_rng(seed)

    # ── Evaluate random variant-block pairs ───────────────────────────────
    results = []
    t0 = time.perf_counter()

    for pair_idx in range(n_pairs):
        # Random non-overlapping blocks
        all_idx = rng.permutation(n_variants)
        block_A = np.sort(all_idx[:block_size])
        block_B = np.sort(all_idx[block_size : 2 * block_size])

        # Dispatch to all actors
        futures = [
            a.compute_cumulants.remote(block_A, block_B) for a in actors
        ]
        node_results = ray.get(futures)

        # Aggregate cumulants
        node_cumulants = [r["cumulants"] for r in node_results]
        global_kappa = aggregate_cumulants(node_cumulants)

        # ── Fed-cSPA p-value ──────────────────────────────────────────────
        # Test statistic: κ₁ (trace of K_epi) as a simple test statistic
        # Under H₀ (no epistasis), κ₁ ≈ 0 after proper centering.
        # Here we use a standardised test stat: Q = κ₁ / √κ₂
        kappa = np.asarray(global_kappa)
        if kappa[1] > 0:
            Q = float(kappa[0] + 2.0 * np.sqrt(kappa[1]))  # right tail probe
        else:
            Q = float(kappa[0])

        spa_result = federated_spa_pvalue(Q, [kappa])

        # ── Asymptotic baseline (normal approx) ──────────────────────────
        from scipy import stats
        if kappa[1] > 0:
            z = (Q - kappa[0]) / np.sqrt(kappa[1])
            asym_pval = float(stats.norm.sf(z))
        else:
            asym_pval = 0.5

        results.append({
            "pair_idx": pair_idx,
            "block_A": block_A.tolist(),
            "block_B": block_B.tolist(),
            "kappa1": kappa[0],
            "kappa2": kappa[1],
            "kappa3": kappa[2],
            "kappa4": kappa[3],
            "Q": Q,
            "pvalue_fedcspa": spa_result["pvalue"],
            "pvalue_asymptotic": asym_pval,
            "saddlepoint": spa_result["saddlepoint"],
        })

        if (pair_idx + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            log.info(
                "Pair %d/%d  (%.1f s elapsed, %.2f s/pair)",
                pair_idx + 1, n_pairs, elapsed, elapsed / (pair_idx + 1),
            )

    ray.shutdown()
    log.info("✓ Evaluation complete: %d pairs in %.1f s", n_pairs, time.perf_counter() - t0)
    return results


def write_parquet(results: list[dict], output_path: str) -> None:
    """Write results to Parquet via pyarrow (or fallback to CSV)."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Flatten block arrays to strings for Parquet
        for r in results:
            r["block_A"] = str(r["block_A"])
            r["block_B"] = str(r["block_B"])

        table = pa.Table.from_pylist(results)
        pq.write_table(table, output_path)
        log.info("Results written to %s (Parquet)", output_path)
    except ImportError:
        import csv
        # Fallback: CSV
        csv_path = output_path.replace(".parquet", ".csv")
        keys = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        log.info("Results written to %s (CSV fallback)", csv_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr-dir", type=str, default="data/biobank_shards")
    p.add_argument("--n-centres", type=int, default=5)
    p.add_argument("--n-pairs", type=int, default=200,
                    help="Number of variant-block pairs to test")
    p.add_argument("--block-size", type=int, default=10,
                    help="Variants per block (m_A = m_B)")
    p.add_argument("--output", type=str, default="results/fedcspa_results.parquet")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    results = run_federated_evaluation(
        zarr_dir=args.zarr_dir,
        n_centres=args.n_centres,
        seed=args.seed,
        n_pairs=args.n_pairs,
        block_size=args.block_size,
    )
    write_parquet(results, args.output)


if __name__ == "__main__":
    main()
