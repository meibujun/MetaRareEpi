"""
node.py — Federated analysis node (Ray actor).

Each node holds a private shard of genotype data in Zarr format and
exposes a method to compute local spectral cumulants without ever
sharing individual-level data.

Architecture:
    Aggregator  ←──  κ_local  ──  Node_1  (private Z_A, Z_B, y)
                ←──  κ_local  ──  Node_2
                ←──  κ_local  ──  Node_K

The aggregator combines local cumulants into a global test statistic
via meta-analytic summation (Cumulant Additivity Theorem, Supp Note S4).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

try:
    import ray
    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False

log = logging.getLogger(__name__)


def _ray_remote(cls):
    """Apply @ray.remote only when Ray is available."""
    if _RAY_AVAILABLE:
        return ray.remote(cls)
    return cls


def _standardise(Z: np.ndarray) -> np.ndarray:
    """Column-wise standardisation: (Z − μ) / σ, with σ floored at 1e-12."""
    mu = Z.mean(axis=0)
    sigma = Z.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (Z - mu) / sigma


@_ray_remote
class FederatedNode:
    """
    A Ray remote actor representing a single biobank node.

    Each node stores its local genotype shards and computes cumulants
    on demand.  Only the (max_power,)-shaped cumulant vector and scalar
    Q_adj are returned — genotype data never leaves the node.
    """

    def __init__(self, node_id: str, zarr_path: str) -> None:
        """
        Parameters
        ----------
        node_id   : unique identifier for this node.
        zarr_path : path to the local Zarr v3 store containing genotypes.
        """
        self.node_id = node_id
        self.zarr_path = zarr_path
        self._G: np.ndarray | None = None
        self._Y: np.ndarray | None = None
        self._mafs: np.ndarray | None = None

    def load_data(self) -> dict:
        """
        Load genotype shard from Zarr v3 store into memory.

        Returns
        -------
        dict with node metadata (node_id, n_samples, n_variants).
        """
        try:
            import zarr
        except ImportError:
            raise ImportError("zarr >= 3.0 is required for federated nodes")

        store = zarr.open_group(self.zarr_path, mode="r")
        self._G = np.asarray(store["genotypes"][:], dtype=np.float64)

        if "phenotype" in store:
            self._Y = np.asarray(store["phenotype"][:], dtype=np.float64)

        if "mafs" in store:
            self._mafs = np.asarray(store["mafs"][:], dtype=np.float64)

        n, m = self._G.shape
        log.info("Node %s loaded: %d samples × %d variants", self.node_id, n, m)

        return {
            "node_id": self.node_id,
            "n_samples": n,
            "n_variants": m,
        }

    def compute_local_cumulants(
        self,
        block_A_idx: np.ndarray,
        block_B_idx: np.ndarray,
        *,
        max_power: int = 4,
        method: str = "hutchpp",
    ) -> dict:
        """
        Compute spectral cumulants κ₁–κ₄ of the local epistatic kernel.

        Extracts cumulants using the implicit Fast-MVM engine without
        ever forming the N×N dense kernel matrix.

        Parameters
        ----------
        block_A_idx : (m_A,) variant column indices for locus block A.
        block_B_idx : (m_B,) variant column indices for locus block B.
        max_power   : highest trace power (default 4).
        method      : "exact" or "hutchinson".

        Returns
        -------
        dict with "node_id", "n_samples", "cumulants" (4,), "Q_adj" (float|None).
        """
        if self._G is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        # Import engine (deferred to avoid circular imports at module level)
        import sys
        src_path = str(Path(__file__).resolve().parent.parent.parent)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from engine_jax import extract_local_cumulants

        # Standardise genotype blocks
        Z_A = _standardise(self._G[:, block_A_idx])
        Z_B = _standardise(self._G[:, block_B_idx])

        # Compute cumulants and optionally Q_adj
        result = extract_local_cumulants(
            Z_A, Z_B,
            max_power=max_power,
            method=method,
            y=self._Y,
            apply_fwl=self._Y is not None,
        )

        return {
            "node_id": self.node_id,
            "n_samples": Z_A.shape[0],
            "cumulants": result["cumulants"],
            "Q_adj": result.get("Q_adj"),
        }

    def get_info(self) -> dict:
        """Return metadata about the loaded data."""
        if self._G is None:
            return {"node_id": self.node_id, "loaded": False}
        return {
            "node_id": self.node_id,
            "loaded": True,
            "n_samples": self._G.shape[0],
            "n_variants": self._G.shape[1],
            "has_phenotype": self._Y is not None,
            "has_mafs": self._mafs is not None,
        }
