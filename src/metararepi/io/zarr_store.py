"""
zarr_store.py — Zarr v3 genomic data store handler.

Provides a unified interface for reading genotype blocks from Zarr v3
hierarchical stores, with support for:
  - Chunked out-of-core access (memory-mapped)
  - Column-wise standardisation on read
  - Locus-block slicing (Z_A, Z_B extraction by variant indices)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import zarr
    _ZARR_AVAILABLE = True
except ImportError:
    _ZARR_AVAILABLE = False


class GenomicZarrStore:
    """
    Handler for Zarr v3 genomic data stores.

    Expected store layout:
        root.zarr/
        ├── genotypes/      # (N, M) uint8 allele count matrix
        ├── phenotype/      # (N,) float64 phenotype vector
        ├── mafs/           # (M,) float64 minor allele frequencies
        ├── positions/      # (M,) float64 genomic positions
        └── .zmetadata      # consolidated metadata
    """

    def __init__(self, store_path: str | Path) -> None:
        self.store_path = Path(store_path)
        self._root = None
        self._is_open = False

    def open(self) -> "GenomicZarrStore":
        """
        Open the Zarr v3 store for reading.

        Returns self for method chaining.

        Raises
        ------
        ImportError
            If zarr is not installed.
        FileNotFoundError
            If store_path does not exist.
        """
        if not _ZARR_AVAILABLE:
            raise ImportError(
                "zarr >= 3.0 is required. Install via: pip install 'zarr>=3.0'"
            )
        if not self.store_path.exists():
            raise FileNotFoundError(f"Zarr store not found: {self.store_path}")

        self._root = zarr.open_group(str(self.store_path), mode="r")
        self._is_open = True
        return self

    @property
    def n_samples(self) -> int:
        """Number of individuals in the store."""
        self._check_open()
        return self._root["genotypes"].shape[0]

    @property
    def n_variants(self) -> int:
        """Number of variants in the store."""
        self._check_open()
        return self._root["genotypes"].shape[1]

    def read_block(
        self,
        variant_indices: np.ndarray,
        standardise: bool = True,
    ) -> np.ndarray:
        """
        Read a genotype block for the specified variant indices.

        Parameters
        ----------
        variant_indices : (m,) int array of column indices.
        standardise     : if True, column-standardise to zero mean / unit var.

        Returns
        -------
        Z : (N, m) float64 genotype matrix (standardised if requested).
        """
        self._check_open()

        # Validate indices
        variant_indices = np.asarray(variant_indices, dtype=np.intp)
        if variant_indices.ndim != 1:
            raise ValueError("variant_indices must be a 1D array")
        if len(variant_indices) == 0:
            raise ValueError("variant_indices must not be empty")

        n_vars = self.n_variants
        if np.any(variant_indices < 0) or np.any(variant_indices >= n_vars):
            raise IndexError(
                f"variant_indices out of range [0, {n_vars}): "
                f"min={variant_indices.min()}, max={variant_indices.max()}"
            )

        # Read genotype columns
        G = np.asarray(
            self._root["genotypes"][:, variant_indices],
            dtype=np.float64,
        )

        if standardise:
            G = self._standardise_columns(G)

        return G

    def read_phenotype(self) -> np.ndarray:
        """Read the phenotype vector from the store."""
        self._check_open()
        if "phenotype" not in self._root:
            raise KeyError("No 'phenotype' array in the Zarr store")
        return np.asarray(self._root["phenotype"][:], dtype=np.float64)

    def read_mafs(self) -> np.ndarray:
        """Read the minor allele frequency vector."""
        self._check_open()
        if "mafs" not in self._root:
            raise KeyError("No 'mafs' array in the Zarr store")
        return np.asarray(self._root["mafs"][:], dtype=np.float64)

    def close(self) -> None:
        """Close the store and release resources."""
        self._root = None
        self._is_open = False

    def __enter__(self) -> "GenomicZarrStore":
        return self.open()

    def __exit__(self, *args) -> None:
        self.close()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _check_open(self) -> None:
        if not self._is_open or self._root is None:
            raise RuntimeError(
                "Store not open. Call .open() first or use context manager."
            )

    @staticmethod
    def _standardise_columns(Z: np.ndarray) -> np.ndarray:
        """Column-wise standardisation: (Z − μ) / σ, with σ floored at 1e-12."""
        mu = Z.mean(axis=0)
        sigma = Z.std(axis=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        return (Z - mu) / sigma
