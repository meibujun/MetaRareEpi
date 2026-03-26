"""
test_zarr_store.py — Tests for metararepi.io.zarr_store.

Validates Zarr v3 genomic data store handler with temporary store fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import zarr
    _ZARR_AVAILABLE = True
except ImportError:
    _ZARR_AVAILABLE = False

from metararepi.io.zarr_store import GenomicZarrStore  # noqa: E402

SEED = 42
N = 100
M = 50


@pytest.fixture(scope="module")
def zarr_store_path():
    """Create a temporary Zarr v3 store with synthetic data."""
    if not _ZARR_AVAILABLE:
        pytest.skip("zarr not installed")

    rng = np.random.default_rng(SEED)
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "test.zarr"
        root = zarr.open_group(str(store_path), mode="w")

        G = rng.integers(0, 3, size=(N, M)).astype(np.uint8)
        root.create_array("genotypes", data=G, dtype="uint8")
        root.create_array("phenotype", data=rng.standard_normal(N), dtype="float64")
        root.create_array("mafs", data=rng.uniform(0.001, 0.5, size=M), dtype="float64")

        yield str(store_path)


@pytest.mark.skipif(not _ZARR_AVAILABLE, reason="zarr not installed")
class TestGenomicZarrStore:

    def test_open_and_close(self, zarr_store_path):
        store = GenomicZarrStore(zarr_store_path)
        store.open()
        assert store._is_open
        store.close()
        assert not store._is_open

    def test_context_manager(self, zarr_store_path):
        with GenomicZarrStore(zarr_store_path) as store:
            assert store._is_open
            assert store.n_samples == N
            assert store.n_variants == M

    def test_read_block(self, zarr_store_path):
        with GenomicZarrStore(zarr_store_path) as store:
            idx = np.array([0, 5, 10, 20])
            Z = store.read_block(idx, standardise=True)
            assert Z.shape == (N, 4)
            # Standardised: mean ≈ 0
            np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-10)

    def test_read_block_no_standardise(self, zarr_store_path):
        with GenomicZarrStore(zarr_store_path) as store:
            idx = np.array([0, 1])
            Z = store.read_block(idx, standardise=False)
            assert Z.dtype == np.float64
            # Should contain original values (0, 1, 2)
            assert np.all((Z >= 0) & (Z <= 2))

    def test_read_phenotype(self, zarr_store_path):
        with GenomicZarrStore(zarr_store_path) as store:
            y = store.read_phenotype()
            assert y.shape == (N,)
            assert y.dtype == np.float64

    def test_read_mafs(self, zarr_store_path):
        with GenomicZarrStore(zarr_store_path) as store:
            mafs = store.read_mafs()
            assert mafs.shape == (M,)

    def test_invalid_indices(self, zarr_store_path):
        with GenomicZarrStore(zarr_store_path) as store:
            with pytest.raises(IndexError):
                store.read_block(np.array([999]))

    def test_empty_indices(self, zarr_store_path):
        with GenomicZarrStore(zarr_store_path) as store:
            with pytest.raises(ValueError):
                store.read_block(np.array([]))

    def test_not_opened_raises(self, zarr_store_path):
        store = GenomicZarrStore(zarr_store_path)
        with pytest.raises(RuntimeError):
            store.n_samples

    def test_missing_store_raises(self):
        store = GenomicZarrStore("/nonexistent/path.zarr")
        with pytest.raises(FileNotFoundError):
            store.open()
