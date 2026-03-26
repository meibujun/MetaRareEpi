# ═══════════════════════════════════════════════════════════════════════════
# MetaRareEpi — Reproducible Analysis Pipeline
# ═══════════════════════════════════════════════════════════════════════════
#
# Usage:
#   make all          — full pipeline (simulate → evaluate → visualise)
#   make simulate     — generate 500K-genome biobank + 5 Zarr shards
#   make evaluate     — deploy Fed-cSPA across 5 assessment centres
#   make visualise    — render extreme-tail Q-Q plot (PDF)
#   make test         — run full pytest suite
#   make clean        — remove all generated data
#
# Requirements:
#   Python ≥ 3.12 with:  msprime ≥ 1.5, stdpopsim, zarr ≥ 3.0, ray ≥ 2.40
#   R ≥ 4.3 with:        ggplot2, ggrastr, ggsci, extrafont
#   uv (for dependency management)
#
# NOTE: msprime, stdpopsim, and ray require Python ≤ 3.13 (no cp314 wheels
# as of March 2026).  The JAX engine tests run on 3.14; simulation and
# federated evaluation use a 3.13 interpreter.
# ═══════════════════════════════════════════════════════════════════════════

.PHONY: all simulate evaluate visualise test clean help

# ── Configuration ─────────────────────────────────────────────────────────
PYTHON      := python
UV          := uv run
RSCRIPT     := Rscript

DATA_DIR    := data
ZARR_DIR    := $(DATA_DIR)/biobank_shards
RESULTS_DIR := results
VIZ_DIR     := viz/output

N_SAMPLES   := 500000
N_CENTRES   := 5
MAF_LO      := 1e-5
MAF_HI      := 1e-2
SEED        := 42

# ── Targets ───────────────────────────────────────────────────────────────

all: simulate evaluate visualise
	@echo "═══ Pipeline complete. ═══"

help:
	@echo "MetaRareEpi reproducible pipeline"
	@echo ""
	@echo "  make all          full pipeline"
	@echo "  make simulate     generate 500K-genome biobank"
	@echo "  make evaluate     run Fed-cSPA across 5 centres"
	@echo "  make visualise    render Q-Q plot PDF"
	@echo "  make test         pytest suite"
	@echo "  make clean        remove generated data"

# ── 1. Simulate ───────────────────────────────────────────────────────────

$(ZARR_DIR)/.stamp: simulations/simulate_biobank.py
	@echo "══╡ SIMULATE: $(N_SAMPLES) genomes, Out-of-Africa ╞══"
	$(PYTHON) simulations/simulate_biobank.py \
		--n-samples $(N_SAMPLES) \
		--n-centres $(N_CENTRES) \
		--maf-lo $(MAF_LO) \
		--maf-hi $(MAF_HI) \
		--seed $(SEED) \
		--output-dir $(ZARR_DIR)
	@touch $@

simulate: $(ZARR_DIR)/.stamp

# ── 2. Evaluate ───────────────────────────────────────────────────────────

$(RESULTS_DIR)/fedcspa_results.parquet: $(ZARR_DIR)/.stamp simulations/evaluate_federated.py
	@echo "══╡ EVALUATE: Fed-cSPA across $(N_CENTRES) centres ╞══"
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) simulations/evaluate_federated.py \
		--zarr-dir $(ZARR_DIR) \
		--n-centres $(N_CENTRES) \
		--output $(RESULTS_DIR)/fedcspa_results.parquet \
		--seed $(SEED)

evaluate: $(RESULTS_DIR)/fedcspa_results.parquet

# ── 3. Visualise ──────────────────────────────────────────────────────────

$(VIZ_DIR)/Figure_2A_QQ_Calibration_Nature.pdf: $(RESULTS_DIR)/fedcspa_results.parquet viz/viz_qq_nature.R
	@echo "══╡ VISUALISE: Nature Genetics Extreme-Tail Q-Q Plot ╞══"
	@mkdir -p $(VIZ_DIR)
	$(RSCRIPT) viz/viz_qq_nature.R \
		$(RESULTS_DIR)/fedcspa_results.parquet \
		$(VIZ_DIR)/Figure_2A_QQ_Calibration_Nature.pdf

$(VIZ_DIR)/Figure_4B_3D_Surface_Nature.pdf: viz/viz_3d_synergy.py
	@echo "══╡ VISUALISE: Nature 3D Synergy Surface ╞══"
	@mkdir -p $(VIZ_DIR)
	$(PYTHON) viz/viz_3d_synergy.py \
		$(VIZ_DIR)/Figure_4B_3D_Surface_Nature.pdf

visualise: $(VIZ_DIR)/Figure_2A_QQ_Calibration_Nature.pdf $(VIZ_DIR)/Figure_4B_3D_Surface_Nature.pdf

# ── Test ──────────────────────────────────────────────────────────────────

test:
	@echo "══╡ TEST: full pytest suite ╞══"
	$(UV) pytest tests/ -v

# ── Clean ─────────────────────────────────────────────────────────────────

clean:
	rm -rf $(DATA_DIR) $(RESULTS_DIR) $(VIZ_DIR)
	rm -rf .pytest_cache __pycache__ src/__pycache__
	@echo "Cleaned."
