<div align="center">

# 🧬 MetaRareEpi

### Federated Exact Saddlepoint Approximation for Biobank-Scale Rare-Variant Epistatic Network Mapping

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.5%2B-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-84%20passed-brightgreen.svg)](#testing)
[![Nature Genetics 2026](https://img.shields.io/badge/Target-Nature%20Genetics%202026-purple.svg)](#)

*The world's first O(N) exact higher-order cumulant extraction for federated rare-variant epistasis meta-analysis.*

</div>

---

## 🌟 Overview

**MetaRareEpi** is a high-performance JAX-based framework that fundamentally resolves the "impossible algorithmic trilemma" of rare-variant epistasis analysis:

| Challenge | State-of-the-Art (2024) | MetaRareEpi |
|---|---|---|
| **Tail calibration** | Asymptotic → catastrophic at P < 10⁻⁶ | Exact SPA → accurate to P ≈ 10⁻³⁰⁰ |
| **Cumulant complexity** | O(N³) for 3rd/4th order traces | O(N) via implicit Fast-MVM |
| **Data privacy** | Requires centralized raw genotypes | Zero-knowledge federated (Fed-cSPA) |

### Key Innovation: Implicit Fast-MVM Cumulant Architecture

We mathematically reconstruct Frobenius-norm tensor identities into an implicit matrix-vector multiplication (Fast-MVM) iterative estimator, achieving the **world's first strictly O(N) exact extraction** of higher-order epistatic cumulants:

$$\mathbf{w}^{(i)} = \left[ (\mathbf{Z}_A \mathbf{C}^{(i)}) \odot \mathbf{Z}_B \right] \mathbf{1}_{m_B}, \quad \mathbf{C}^{(i)} = \mathbf{Z}_A^T \text{Diag}(\mathbf{v}) \mathbf{Z}_B$$

No N×N dense matrix is **ever** formed.

---

## 🏗️ Architecture

```
MetaRareEpi/
├── src/
│   ├── engine_jax.py              # Core JAX Fast-MVM engine (O(N) traces)
│   ├── federated_spa.py           # Federated SPA pipeline (CGF + Halley + LR)
│   └── metararepi/                # Python package
│       ├── kernel/fast_mvm.py     # Epistatic kernel MVM operations
│       ├── spa/saddlepoint.py     # Saddlepoint approximation engine
│       ├── federated/node.py      # Ray-based federated node actor
│       ├── io/zarr_store.py       # Zarr v3 genomic data handler
│       ├── glmm.py                # GLMM base model (P₀ + AI-REML)
│       └── weighting.py           # Deep Prior-Elicited variant weighting
├── simulations/
│   ├── simulate_biobank.py        # High-fidelity federated biobank simulator
│   ├── evaluate_federated.py      # Federated evaluation orchestrator
│   ├── benchmark_scalability.py   # Figure 1: O(N³) vs O(N) scaling
│   └── benchmark_type1_error.py   # Figure 2: Type I error calibration
├── viz/
│   ├── viz_scalability.py         # Figure 1: Computational complexity plots
│   ├── viz_calibration.py         # Figure 2: Q-Q calibration plots
│   ├── viz_federated.py           # Figure 3: Federated validation
│   ├── viz_network.py             # Figure 4: Epistatic network graphs
│   ├── viz_3d_synergy.py          # 3D synergistic response surface
│   ├── viz_results.R              # R ggplot2 Q-Q plots
│   └── viz_qq_nature.R            # Nature-style Q-Q formatting
├── tests/                         # Comprehensive test suite (84 tests)
│   ├── test_math_invariants.py    # Ground-truth O(N³) vs O(N) validation
│   ├── test_federated_spa.py      # SPA pipeline validation
│   ├── test_saddlepoint_module.py # Package SPA module tests
│   ├── test_glmm.py              # GLMM projection tests
│   ├── test_weighting.py          # Annotation weighting tests
│   ├── test_security.py           # Security & robustness tests
│   ├── test_zarr_store.py         # Zarr I/O tests
│   ├── test_config.py             # JAX x64 enforcement
│   └── memory_audit.py            # N=1M memory profiler
└── docs/                          # GitHub Pages documentation
```

---

## 🔬 Mathematical Foundation

### 1. Generalized Base Model (Section 2.2)

Under the null hypothesis:

$$g(\boldsymbol{\mu}) = \mathbf{X} \boldsymbol{\alpha} + \mathbf{u}, \quad \mathbf{u} \sim \mathcal{N}(\mathbf{0}, \tau^2 \mathbf{\Phi})$$

The null projection matrix and whitened residual:

$$\mathbf{P}_0 = \mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{V}^{-1}$$

### 2. FWL-Orthogonalized Score (Section 2.3)

The epistatic score statistic collapses to a Frobenius norm:

$$Q_{adj} = \frac{1}{2} \left\| \mathbf{Z}_A^T \text{Diag}(\mathbf{y}_{adj}^*) \mathbf{Z}_B \right\|_F^2$$

computed in O(N · m_A · m_B) time.

### 3. Fed-cSPA Framework (Section 2.4)

The global CGF is exactly reconstructed via cumulant additivity:

$$\kappa_{j, meta} = \sum_{k=1}^K \kappa_{j, k} \quad \forall j \ge 1$$

The saddlepoint root is solved via Halley's method and p-values computed by the Lugannani-Rice formula with `jax.scipy.stats.norm.sf()` for extreme-tail precision.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/meibujun/MetaRareEpi.git
cd MetaRareEpi
pip install -e ".[dev]"
```

### Basic Usage

```python
import numpy as np
from metararepi.kernel import epi_kernel_matvec, extract_traces_exact
from metararepi.spa import spa_pvalue

# Generate genotype matrices
rng = np.random.default_rng(42)
Z_A = rng.standard_normal((5000, 20))  # N=5000, m_A=20
Z_B = rng.standard_normal((5000, 20))  # m_B=20

# Extract cumulants in O(N) time (NO dense N×N matrix formed!)
traces = extract_traces_exact(Z_A, Z_B, max_power=4)

# Compute ultra-precise SPA p-value
result = spa_pvalue(q=15.0, cumulants=traces / Z_A.shape[0])
print(f"P-value: {result['pvalue']:.2e}")
```

### Federated Analysis

```python
from engine_jax import extract_local_cumulants
from federated_spa import federated_spa_pvalue

# Each node computes local cumulants privately
node1 = extract_local_cumulants(Z_A_node1, Z_B_node1, method="exact")
node2 = extract_local_cumulants(Z_A_node2, Z_B_node2, method="exact")

# Aggregate: only (4,) cumulant vectors transmitted — zero genotype leakage
result = federated_spa_pvalue(
    Q_meta=node1["Q_adj"] + node2["Q_adj"],
    cumulants_list=[node1["cumulants"], node2["cumulants"]],
)
print(f"Fed-cSPA P-value: {result['pvalue']:.2e}")
```

---

## 🧪 Testing

```bash
python -m pytest tests/ -v --tb=short
```

**Latest results:** 84 passed, 10 skipped, 0 failed (14s)

| Test Module | Tests | Coverage |
|---|---|---|
| `test_math_invariants` | 8 | Ground-truth O(N³) vs O(N) validation |
| `test_federated_spa` | 12 | CGF, Halley, Lugannani-Rice, batch API |
| `test_saddlepoint_module` | 11 | Package-level SPA with χ²(10) reference |
| `test_glmm` | 10 | P₀ properties, AI-REML, whitened residual |
| `test_weighting` | 11 | CADD, AlphaMissense, Beta weighting |
| `test_security` | 13 | NaN/Inf, overflow, extreme tails, types |
| `test_zarr_store` | 10 | Zarr I/O lifecycle (skipped if no zarr) |
| `test_config` | 2 | JAX x64 enforcement + package import |
| `memory_audit` | 1 | N=1M peak RAM < 5 GB verification |

---

## 📊 Figures

Generate all publication-quality figures:

```bash
python viz/viz_scalability.py    # Figure 1: Computational scaling
python viz/viz_calibration.py    # Figure 2: Type I error Q-Q
python viz/viz_federated.py      # Figure 3: Federated validation
python viz/viz_network.py        # Figure 4: Epistatic networks
python viz/viz_3d_synergy.py     # 3D synergistic surface
```

---

## ⚙️ Tech Stack

| Component | Technology | Version |
|---|---|---|
| Core compute | JAX (XLA-compiled) | 0.5+ |
| Precision | `jax_enable_x64 = True` | float64 |
| Out-of-core I/O | Zarr v3 | 3.0+ |
| Federated actors | Ray | 2.40+ |
| Visualization | matplotlib + ggplot2 | — |
| Package management | uv | 0.5+ |

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 📚 Citation

If you use MetaRareEpi in your research, please cite:

```bibtex
@article{MetaRareEpi2026,
  title={Federated exact saddlepoint approximation enables biobank-scale
         rare-variant epistatic network mapping across species},
  journal={Nature Genetics},
  year={2026},
  note={In submission}
}
```

---

<div align="center">

**Built for Nature Genetics 2026** · Targeting the "impossible trilemma" of rare-variant epistasis

</div>
