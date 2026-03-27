"""
graph_search.py — Graph-Regularized Epistatic Search Space Optimization

MetaRareEpi R2 Framework (§2.6)

Concentrates the testing burden on biologically plausible gene pairs
using three complementary sources:
  (a) PPI interfaces from AlphaFold-Multimer / Predictomes (iPTM ≥ 0.5)
  (b) Co-localization within TADs from Micro-C chromatin maps
  (c) Co-membership in KEGG / Reactome metabolic pathways

The union graph reduces ~2×10⁸ all-by-all hypotheses to ~10⁴ candidates,
yielding a Bonferroni threshold of ~5×10⁻⁶ (vs. ~2.5×10⁻¹⁰).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set, Tuple

import numpy as np


@dataclass
class GeneInfo:
    """Gene metadata for search space construction."""
    gene_id: str
    gene_name: str
    chromosome: str
    start: int
    end: int
    tad_id: str | None = None
    pathways: set = field(default_factory=set)


@dataclass
class GenePairCandidate:
    """A candidate gene pair for epistasis testing."""
    gene_a: str
    gene_b: str
    evidence_sources: set = field(default_factory=set)
    ppi_score: float = 0.0
    same_tad: bool = False
    shared_pathways: set = field(default_factory=set)

    @property
    def priority_score(self) -> float:
        """Combined priority score based on evidence."""
        score = 0.0
        if "ppi" in self.evidence_sources:
            score += self.ppi_score
        if "tad" in self.evidence_sources:
            score += 0.3
        if "pathway" in self.evidence_sources:
            score += 0.2 * len(self.shared_pathways)
        return score


class GraphRegularizedSearch:
    """
    Graph-regularized search space optimizer for epistatic gene pairs.

    Constructs a weighted prior graph from PPI, TAD, and pathway sources,
    then restricts testing to the union of these biologically-informed edges.
    """

    def __init__(self):
        self.genes: dict[str, GeneInfo] = {}
        self.ppi_edges: dict[Tuple[str, str], float] = {}
        self.tad_edges: set[Tuple[str, str]] = set()
        self.pathway_edges: dict[Tuple[str, str], set] = {}
        self.candidates: list[GenePairCandidate] = []

    def add_gene(self, gene: GeneInfo):
        """Register a gene in the search space."""
        self.genes[gene.gene_id] = gene

    def add_ppi_edge(self, gene_a: str, gene_b: str, iptm_score: float):
        """
        Add a PPI edge from AlphaFold-Multimer / Predictomes.

        Only edges with iPTM ≥ 0.5 are retained (§2.6a).
        """
        if iptm_score >= 0.5:
            key = tuple(sorted([gene_a, gene_b]))
            self.ppi_edges[key] = max(self.ppi_edges.get(key, 0), iptm_score)

    def add_tad_colocalization(self, gene_a: str, gene_b: str):
        """Add a TAD co-localization edge (§2.6b)."""
        key = tuple(sorted([gene_a, gene_b]))
        self.tad_edges.add(key)

    def add_pathway_membership(
        self,
        gene_a: str,
        gene_b: str,
        pathway_id: str,
    ):
        """Add a pathway co-membership edge (§2.6c)."""
        key = tuple(sorted([gene_a, gene_b]))
        if key not in self.pathway_edges:
            self.pathway_edges[key] = set()
        self.pathway_edges[key].add(pathway_id)

    def load_ppi_from_file(self, filepath: str, iptm_threshold: float = 0.5):
        """
        Load PPI predictions from a TSV file.

        Expected format: gene_a<tab>gene_b<tab>iptm_score
        """
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    score = float(parts[2])
                    if score >= iptm_threshold:
                        self.add_ppi_edge(parts[0], parts[1], score)

    def load_tad_from_file(self, filepath: str):
        """
        Load TAD annotations from a BED-like file.

        Expected format: chr<tab>start<tab>end<tab>tad_id
        """
        tad_members: dict[str, list[str]] = {}
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    tad_id = parts[3]
                    gene_id = parts[0]  # simplified
                    if tad_id not in tad_members:
                        tad_members[tad_id] = []
                    tad_members[tad_id].append(gene_id)

        # All pairs within the same TAD
        for tad_id, members in tad_members.items():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    self.add_tad_colocalization(members[i], members[j])

    def load_pathway_from_file(self, filepath: str):
        """
        Load pathway memberships from a GMT-like file.

        Expected format: pathway_id<tab>description<tab>gene1<tab>gene2<tab>...
        """
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    pathway_id = parts[0]
                    genes = parts[2:]
                    for i in range(len(genes)):
                        for j in range(i + 1, len(genes)):
                            self.add_pathway_membership(genes[i], genes[j], pathway_id)

    def build_candidate_set(self) -> list[GenePairCandidate]:
        """
        Build the union candidate set from all evidence sources.

        Returns approximately 10⁴ gene pairs — two orders of magnitude
        fewer than the genome-wide ~2×10⁸ set.
        """
        all_pairs: dict[Tuple[str, str], GenePairCandidate] = {}

        # PPI edges
        for (ga, gb), score in self.ppi_edges.items():
            if (ga, gb) not in all_pairs:
                all_pairs[(ga, gb)] = GenePairCandidate(gene_a=ga, gene_b=gb)
            all_pairs[(ga, gb)].evidence_sources.add("ppi")
            all_pairs[(ga, gb)].ppi_score = score

        # TAD edges
        for (ga, gb) in self.tad_edges:
            if (ga, gb) not in all_pairs:
                all_pairs[(ga, gb)] = GenePairCandidate(gene_a=ga, gene_b=gb)
            all_pairs[(ga, gb)].evidence_sources.add("tad")
            all_pairs[(ga, gb)].same_tad = True

        # Pathway edges
        for (ga, gb), pathways in self.pathway_edges.items():
            if (ga, gb) not in all_pairs:
                all_pairs[(ga, gb)] = GenePairCandidate(gene_a=ga, gene_b=gb)
            all_pairs[(ga, gb)].evidence_sources.add("pathway")
            all_pairs[(ga, gb)].shared_pathways = pathways

        self.candidates = sorted(
            all_pairs.values(),
            key=lambda c: c.priority_score,
            reverse=True,
        )
        return self.candidates

    def bonferroni_threshold(self, alpha: float = 0.05) -> float:
        """
        Compute the Bonferroni-corrected significance threshold.

        For ~10⁴ candidates: α / 10⁴ ≈ 5 × 10⁻⁶
        vs. genome-wide: α / 2×10⁸ ≈ 2.5 × 10⁻¹⁰
        """
        n_tests = max(len(self.candidates), 1)
        return alpha / n_tests

    def summary(self) -> dict:
        """Summary statistics of the search space."""
        return {
            "n_genes": len(self.genes),
            "n_ppi_edges": len(self.ppi_edges),
            "n_tad_edges": len(self.tad_edges),
            "n_pathway_edges": len(self.pathway_edges),
            "n_candidates": len(self.candidates),
            "bonferroni_threshold": self.bonferroni_threshold(),
            "reduction_factor": 2e8 / max(len(self.candidates), 1),
        }


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE: Simulated search space for testing / demo
# ═══════════════════════════════════════════════════════════════════════════

def create_demo_search_space(
    n_genes: int = 100,
    ppi_density: float = 0.05,
    tad_groups: int = 10,
    n_pathways: int = 5,
    seed: int = 42,
) -> GraphRegularizedSearch:
    """
    Create a simulated search space for testing.

    Parameters
    ----------
    n_genes      : number of genes.
    ppi_density  : fraction of gene pairs with PPI evidence.
    tad_groups   : number of TAD groups.
    n_pathways   : number of pathways.
    seed         : random seed.
    """
    rng = np.random.default_rng(seed)
    gs = GraphRegularizedSearch()

    # Create genes
    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    for i, name in enumerate(gene_names):
        gs.add_gene(GeneInfo(
            gene_id=name,
            gene_name=name,
            chromosome=f"chr{(i % 22) + 1}",
            start=i * 100000,
            end=i * 100000 + 50000,
            tad_id=f"TAD_{i % tad_groups}",
        ))

    # PPI edges
    n_ppi = int(n_genes * (n_genes - 1) / 2 * ppi_density)
    for _ in range(n_ppi):
        i, j = rng.choice(n_genes, size=2, replace=False)
        score = rng.uniform(0.5, 1.0)
        gs.add_ppi_edge(gene_names[i], gene_names[j], score)

    # TAD co-localization
    genes_per_tad = n_genes // tad_groups
    for t in range(tad_groups):
        members = gene_names[t * genes_per_tad:(t + 1) * genes_per_tad]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                gs.add_tad_colocalization(members[i], members[j])

    # Pathway co-membership
    for p in range(n_pathways):
        pathway_size = rng.integers(5, 20)
        members = rng.choice(gene_names, size=pathway_size, replace=False)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                gs.add_pathway_membership(members[i], members[j], f"KEGG_{p:03d}")

    gs.build_candidate_set()
    return gs
