#!/usr/bin/env python
"""
simulate_biobank.py — High-Fidelity Federated Biobank Simulator

Nature Genetics 2026 · MetaRareEpi Framework

Pipeline:
    1. Simulate N diploid genomes under stdpopsim's OutOfAfrica_3G09
       demographic model via the msprime coalescent engine.
    2. Stream allele frequencies from the tree sequence (O(1) per site,
       never materialises the full N×M haplotype matrix).
    3. Extract BOTH rare (MAF ∈ [10^{-5}, 10^{-2}]) and common (MAF ≥ 0.05)
       variants via single-pass tree-sequence iteration.
    4. Synthesise phenotypes under a strict variance-component model:
           h²_poly  = 0.40  (polygenic background via common variants)
           h²_main  = 0.05  (marginal main effects on 20 rare variants)
           h²_epi   = 0.005 (pure synergistic epistasis between 2 rare variants)
           σ²_e     = 0.545 (environmental noise)
    5. Partition into K geographically-distributed Zarr v3 silos.

Phenotype synthesis provides a COMPLETE GROUND TRUTH for:
    - Verifying FWL deflation removes h²_main correctly.
    - Confirming the Fed-cSPA detects h²_epi epistasis.
    - Calibrating Type-I error under H₀ (non-epistatic variant pairs).

Requirements: msprime >= 1.5, stdpopsim, zarr >= 3.0, numpy
Note: Requires Python <= 3.13 (msprime has no cp314 wheels as of 2026-03).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("simulate_biobank")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulate a federated biobank with ground-truth epistasis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-samples", type=int, default=500_000,
                    help="Total diploid individuals")
    p.add_argument("--n-centres", type=int, default=5,
                    help="Number of federated assessment centres")
    p.add_argument("--maf-lo", type=float, default=1e-5,
                    help="Lower MAF bound for rare variants")
    p.add_argument("--maf-hi", type=float, default=1e-2,
                    help="Upper MAF bound for rare variants")
    p.add_argument("--maf-common", type=float, default=0.05,
                    help="Lower MAF bound for common variants")
    p.add_argument("--seq-length", type=float, default=1e6,
                    help="Simulated sequence length in bp")
    p.add_argument("--recomb-rate", type=float, default=1e-8,
                    help="Recombination rate per bp per generation")
    p.add_argument("--mut-rate", type=float, default=1.29e-8,
                    help="Mutation rate per bp per generation")
    p.add_argument("--h2-poly", type=float, default=0.40,
                    help="Heritability: polygenic background")
    p.add_argument("--h2-main", type=float, default=0.05,
                    help="Heritability: rare-variant main effects")
    p.add_argument("--h2-epi", type=float, default=0.005,
                    help="Heritability: epistatic interaction")
    p.add_argument("--n-main-variants", type=int, default=20,
                    help="Number of rare variants with main effects")
    p.add_argument("--epi-pair", type=int, nargs=2, default=[100, 101],
                    help="Indices of the epistatic variant pair (ground truth)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="data/biobank_shards")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 1. COALESCENT SIMULATION  (stdpopsim + msprime)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_tree_sequence(
    n_samples: int,
    seq_length: float,
    recomb_rate: float,
    mut_rate: float,
    seed: int,
):
    """
    Simulate a tree sequence under the Out-of-Africa 3-population
    Gutenkunst et al. (2009) demographic model via stdpopsim.

    All individuals sampled from YRI; for multi-ancestry, draw from CEU/CHB.
    """
    import msprime
    import stdpopsim

    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("OutOfAfrica_3G09")

    contig = species.get_contig(
        length=seq_length,
        mutation_rate=mut_rate,
        recombination_rate=recomb_rate,
    )

    engine = stdpopsim.get_engine("msprime")
    log.info(
        "Simulating %d diploids, %.0f bp, OutOfAfrica_3G09 ...",
        n_samples, seq_length,
    )
    ts = engine.simulate(model, contig, {"YRI": n_samples}, seed=seed)
    log.info(
        "Tree sequence: %d trees, %d mutations, %d haplotypes",
        ts.num_trees, ts.num_mutations, ts.num_samples,
    )
    return ts


# ═══════════════════════════════════════════════════════════════════════════
# 2. GENOTYPE EXTRACTION  (streaming, O(1) memory per site)
# ═══════════════════════════════════════════════════════════════════════════

def extract_genotypes(
    ts,
    maf_lo: float,
    maf_hi: float,
    maf_common: float,
) -> dict:
    """
    Single-pass extraction of rare AND common genotype matrices.

    Streams variants from the tree sequence — NEVER materialises the
    full N×M haplotype matrix.  Each variant is classified into:
      - rare:    MAF ∈ [maf_lo, maf_hi]
      - common:  MAF ≥ maf_common

    Returns
    -------
    dict with keys:
        "G_rare"     : (N, M_rare) uint8
        "G_common"   : (N, M_common) uint8
        "mafs_rare"  : (M_rare,) float64
        "mafs_common": (M_common,) float64
        "pos_rare"   : (M_rare,) float64
        "pos_common" : (M_common,) float64
    """
    n_haplotypes = ts.num_samples
    n_individuals = n_haplotypes // 2

    log.info("Streaming %d sites, classifying by MAF ...", ts.num_sites)

    # First pass: compute MAFs and classify sites
    site_mafs = np.empty(ts.num_sites, dtype=np.float64)
    site_positions = np.empty(ts.num_sites, dtype=np.float64)

    for i, v in enumerate(ts.variants()):
        af = v.genotypes.mean()           # haploid allele frequency
        site_mafs[i] = min(af, 1.0 - af)  # minor allele frequency
        site_positions[i] = v.site.position

    rare_mask = (site_mafs >= maf_lo) & (site_mafs <= maf_hi)
    common_mask = site_mafs >= maf_common
    n_rare = int(rare_mask.sum())
    n_common = int(common_mask.sum())

    log.info(
        "Sites: %d total | %d rare (MAF [%.1e, %.1e]) | %d common (MAF >= %.2f)",
        ts.num_sites, n_rare, maf_lo, maf_hi, n_common, maf_common,
    )

    if n_rare == 0:
        log.error("No rare variants! Increase --seq-length or widen MAF bounds.")
        sys.exit(1)

    # Second pass: extract diploid genotypes for classified sites
    rare_ids = set(np.where(rare_mask)[0])
    common_ids = set(np.where(common_mask)[0])

    G_rare = np.zeros((n_individuals, n_rare), dtype=np.uint8)
    G_common = np.zeros((n_individuals, n_common), dtype=np.uint8)
    r_col, c_col = 0, 0

    for v in ts.variants():
        sid = v.site.id
        if sid in rare_ids or sid in common_ids:
            diploid = v.genotypes.reshape(n_individuals, 2).sum(axis=1).astype(np.uint8)
            if sid in rare_ids:
                G_rare[:, r_col] = diploid
                r_col += 1
            if sid in common_ids:
                G_common[:, c_col] = diploid
                c_col += 1

    log.info("Genotype matrices: G_rare %s, G_common %s", G_rare.shape, G_common.shape)
    return {
        "G_rare": G_rare,
        "G_common": G_common,
        "mafs_rare": site_mafs[rare_mask],
        "mafs_common": site_mafs[common_mask],
        "pos_rare": site_positions[rare_mask],
        "pos_common": site_positions[common_mask],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. PHENOTYPE SYNTHESIS  (strict variance-component model)
# ═══════════════════════════════════════════════════════════════════════════
#
# The ground-truth phenotype is:
#     Y = u_poly + u_main + u_epi + epsilon
#
# where each component is scaled to its target heritability:
#     Var(u_comp) / Var(Y) = h²_comp
#
# This creates a RIGOROUS calibration target:
#   - Fed-cSPA should detect the epistatic pair with very small P-values.
#   - Non-epistatic variant pairs should produce uniform P-values (Q-Q
#     on the diagonal), confirming correct Type-I error calibration.
#   - FWL deflation should remove the h²_main signal.
#
# ═══════════════════════════════════════════════════════════════════════════

def _scale_to_variance(component: np.ndarray, target_var: float) -> np.ndarray:
    """Scale a mean-zero component to have exactly target_var variance."""
    component = component - component.mean()
    current_var = component.var()
    if current_var < 1e-30:
        return component
    return component * np.sqrt(target_var / current_var)


def synthesise_phenotype(
    G_rare: np.ndarray,
    G_common: np.ndarray,
    *,
    h2_poly: float = 0.40,
    h2_main: float = 0.05,
    h2_epi: float = 0.005,
    n_main_variants: int = 20,
    epi_pair: tuple[int, int] = (100, 101),
    seed: int = 42,
) -> dict:
    """
    Synthesise a phenotype with exact variance partitioning.

    Parameters
    ----------
    G_rare   : (N, M_rare) rare variant genotypes.
    G_common : (N, M_common) common variant genotypes.
    h2_poly  : target heritability for polygenic background.
    h2_main  : target heritability for rare-variant main effects.
    h2_epi   : target heritability for epistatic interaction.
    n_main_variants : number of rare variants with main effects.
    epi_pair : (idx_A, idx_B) column indices in G_rare for epistasis.
    seed     : PRNG seed.

    Returns
    -------
    dict with "Y", "u_poly", "u_main", "u_epi", "epsilon", "ground_truth".
    """
    rng = np.random.default_rng(seed)
    N = G_rare.shape[0]
    h2_epsilon = 1.0 - h2_poly - h2_main - h2_epi

    if h2_epsilon < 0:
        raise ValueError(
            f"Heritabilities sum to {h2_poly + h2_main + h2_epi:.3f} > 1.0"
        )

    log.info(
        "Phenotype model: h2_poly=%.3f, h2_main=%.3f, h2_epi=%.4f, sigma2_e=%.3f",
        h2_poly, h2_main, h2_epi, h2_epsilon,
    )

    # ── 1. Polygenic background (common variants) ─────────────────────────
    M_common = G_common.shape[1]
    if M_common > 0:
        beta_poly = rng.standard_normal(M_common)
        u_poly_raw = G_common.astype(np.float64) @ beta_poly
        u_poly = _scale_to_variance(u_poly_raw, h2_poly)
    else:
        log.warning("No common variants — polygenic background set to zero.")
        u_poly = np.zeros(N, dtype=np.float64)

    # ── 2. Rare-variant main effects (confounding for FWL stress-test) ────
    n_main = min(n_main_variants, G_rare.shape[1])
    if n_main > 0:
        beta_main = rng.standard_normal(n_main)
        u_main_raw = G_rare[:, :n_main].astype(np.float64) @ beta_main
        u_main = _scale_to_variance(u_main_raw, h2_main)
    else:
        u_main = np.zeros(N, dtype=np.float64)

    # ── 3. Pure synergistic epistasis (the target signal) ─────────────────
    idx_A, idx_B = epi_pair
    if idx_A < G_rare.shape[1] and idx_B < G_rare.shape[1]:
        epi_product = (
            G_rare[:, idx_A].astype(np.float64)
            * G_rare[:, idx_B].astype(np.float64)
        )
        u_epi = _scale_to_variance(epi_product, h2_epi)
        log.info(
            "Epistatic pair: variants %d x %d | nonzero carriers: %d / %d",
            idx_A, idx_B, int((epi_product > 0).sum()), N,
        )
    else:
        log.warning("Epistatic pair indices out of range — u_epi set to zero.")
        u_epi = np.zeros(N, dtype=np.float64)

    # ── 4. Environmental noise ────────────────────────────────────────────
    epsilon = _scale_to_variance(rng.standard_normal(N), h2_epsilon)

    # ── Assemble phenotype ────────────────────────────────────────────────
    Y = u_poly + u_main + u_epi + epsilon

    # Verify variance partitioning
    total_var = Y.var()
    log.info(
        "Variance check: poly=%.4f, main=%.4f, epi=%.6f, eps=%.4f, total=%.4f",
        u_poly.var(), u_main.var(), u_epi.var(), epsilon.var(), total_var,
    )

    ground_truth = {
        "h2_poly": h2_poly,
        "h2_main": h2_main,
        "h2_epi": h2_epi,
        "h2_epsilon": h2_epsilon,
        "n_main_variants": n_main,
        "epi_pair": list(epi_pair),
        "epi_carriers": int((epi_product > 0).sum()) if idx_A < G_rare.shape[1] else 0,
        "phenotype_variance": float(total_var),
    }

    return {
        "Y": Y,
        "u_poly": u_poly,
        "u_main": u_main,
        "u_epi": u_epi,
        "epsilon": epsilon,
        "ground_truth": ground_truth,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. FEDERATED PARTITIONING + ZARR v3 OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def partition_and_write_zarr(
    G_rare: np.ndarray,
    G_common: np.ndarray,
    Y: np.ndarray,
    mafs: np.ndarray,
    positions: np.ndarray,
    ground_truth: dict,
    n_centres: int,
    output_dir: str,
    seed: int,
) -> None:
    """
    Randomly partition N individuals into K assessment centres and write
    each shard as a Zarr v3 group.

    Store layout per centre:
        {output_dir}/centre_{k}.zarr/
        ├── genotypes_rare    (N_k, M_rare) uint8
        ├── genotypes_common  (N_k, M_common) uint8
        ├── phenotype         (N_k,) float64
        ├── mafs              (M_rare,) float64
        └── positions         (M_rare,) float64
    """
    import zarr

    rng = np.random.default_rng(seed + 1)  # distinct from phenotype seed
    N = G_rare.shape[0]
    indices = rng.permutation(N)
    splits = np.array_split(indices, n_centres)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for k, idx in enumerate(splits):
        centre_path = out / f"centre_{k}.zarr"
        log.info("Writing centre %d: %d individuals", k, len(idx))

        store = zarr.open_group(str(centre_path), mode="w")

        n_k = len(idx)
        chunk_rows = min(n_k, 10000)

        store.create_array(
            "genotypes", data=G_rare[idx], dtype="uint8",
            chunks=(chunk_rows, G_rare.shape[1]),
        )
        if G_common.shape[1] > 0:
            store.create_array(
                "genotypes_common", data=G_common[idx], dtype="uint8",
                chunks=(chunk_rows, G_common.shape[1]),
            )
        store.create_array("phenotype", data=Y[idx], dtype="float64")
        store.create_array("mafs", data=mafs, dtype="float64")
        store.create_array("positions", data=positions, dtype="float64")

    # ── Metadata ──────────────────────────────────────────────────────────
    meta = {
        "n_total": N,
        "n_centres": n_centres,
        "n_variants_rare": int(G_rare.shape[1]),
        "n_variants_common": int(G_common.shape[1]),
        "centre_sizes": [len(s) for s in splits],
        "ground_truth": ground_truth,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata + ground truth written to %s/metadata.json", output_dir)


# ═══════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # Step 1: Coalescent simulation
    ts = simulate_tree_sequence(
        n_samples=args.n_samples,
        seq_length=args.seq_length,
        recomb_rate=args.recomb_rate,
        mut_rate=args.mut_rate,
        seed=args.seed,
    )

    # Step 2: Genotype extraction (streaming, both rare + common)
    geno = extract_genotypes(
        ts, maf_lo=args.maf_lo, maf_hi=args.maf_hi, maf_common=args.maf_common,
    )

    # Step 3: Phenotype synthesis with ground-truth epistasis
    pheno = synthesise_phenotype(
        geno["G_rare"],
        geno["G_common"],
        h2_poly=args.h2_poly,
        h2_main=args.h2_main,
        h2_epi=args.h2_epi,
        n_main_variants=args.n_main_variants,
        epi_pair=tuple(args.epi_pair),
        seed=args.seed,
    )

    # Step 4: Federated partitioning → Zarr v3 shards
    partition_and_write_zarr(
        G_rare=geno["G_rare"],
        G_common=geno["G_common"],
        Y=pheno["Y"],
        mafs=geno["mafs_rare"],
        positions=geno["pos_rare"],
        ground_truth=pheno["ground_truth"],
        n_centres=args.n_centres,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    log.info("Simulation complete.")


if __name__ == "__main__":
    main()
