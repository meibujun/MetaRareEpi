# MetaRareEpi

**Dual-Space Federated Saddlepoint Approximation with Deflation-Accelerated Cumulant Extraction for Biobank-Scale Rare-Variant Epistasis Mapping**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-XLA%20accelerated-green.svg)](https://jax.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-24%2F24%20passed-brightgreen.svg)](#testing)

---

## Overview

MetaRareEpi is a federated statistical framework that resolves the three intertwined bottlenecks obstructing rare-variant epistasis mapping:

1. **Distributional failure**: Asymptotic approximations collapse at extreme significance tails for rare variants (MAF < 0.01)
2. **Cubic complexity**: Higher-order cumulants required for saddlepoint correction scale as O(N³)  
3. **Privacy barriers**: Global regulations prohibit raw genotype pooling across institutions

Through **five integrated innovations**, MetaRareEpi achieves strictly linear-time cumulant extraction, exact SPA-calibrated p-values, and privacy-preserving federation.

## Five Innovations

| # | Innovation | Section | Key Result |
|---|-----------|---------|------------|
| 1 | **Dual-space Khatri-Rao reformulation** | §2.3 | O(N) cumulant extraction via symmetric Gram matrix |
| 2 | **Generalized FWL orthogonalization** | §2.2 | Exact main-effect annihilation for binary traits |
| 3 | **Non-linear genomic control** | §2.4 | Phantom epistasis elimination via sparse Hadamard-squared GRM |
| 4 | **CKKS homomorphic encryption** | §2.5 | Zero-knowledge federated meta-analysis |
| 5 | **Graph-regularized search** | §2.6 | ~100× testing burden reduction via PPI/TAD/pathway priors |

## Architecture

```
MetaRareEpi/
├── src/
│   ├── engine_jax.py              # Core: dual-space Hutch++ cumulant extractor
│   ├── federated_spa.py           # CKKS-encrypted federated protocol
│   └── metararepi/
│       ├── glmm.py                # GLMM: binary IRLS + generalized FWL
│       ├── nlgc.py                # Non-linear genomic control
│       ├── graph_search.py        # Graph-regularized search space
│       ├── spa/saddlepoint.py     # Lugannani-Rice SPA
│       ├── kernel/fast_mvm.py     # Implicit MVM primitives
│       └── weighting.py           # CADD/AlphaMissense weights
├── simulations/
│   └── simulate_biobank.py        # Full paper §2.7 experiments
├── tests/
│   └── test_math_invariants.py    # 24 invariant tests (all pass)
├── viz/                           # Publication-quality figures
└── docs/                          # Project website
```

## Quick Start

### Installation

```bash
git clone https://github.com/meibujun/MetaRareEpi.git
cd MetaRareEpi
pip install -e .
```

### Basic Usage

```python
import numpy as np
from engine_jax import extract_local_cumulants
from metararepi.spa.saddlepoint import spa_pvalue

# Genotype matrices for two variant sets
N, m_A, m_B = 10000, 20, 20
Z_A = np.random.randn(N, m_A)  # standardized genotypes
Z_B = np.random.randn(N, m_B)

# Extract cumulants via dual-space Hutch++ (Algorithm 1)
result = extract_local_cumulants(
    Z_A, Z_B,
    method="hutchpp",      # dual-space deflation-accelerated
    n_probes=100,           # Rademacher probe budget
    y=phenotype_residual,   # FWL-adjusted phenotype
    apply_fwl=True,         # generalized FWL orthogonalization
)

# SPA p-value via Lugannani-Rice formula
pval = spa_pvalue(result["Q_adj"], result["cumulants"])
print(f"P-value: {pval['pvalue']:.2e}")
```

### Binary Trait Analysis

```python
from metararepi.glmm import fit_null_model, build_fwl_projection

# Fit logistic GLMM (IRLS weights absorbed in V — Remark 1)
null_model = fit_null_model(y_binary, X_covariates, GRM, trait_type="binary")

# Generalized FWL projection (Proposition 1)
Z_main = np.column_stack([Z_A, Z_B])
P_adj = build_fwl_projection(null_model["P0_matrix"], Z_main)
y_adj = P_adj @ (y_binary - null_model["mu_hat"])
```

### Federated Analysis with CKKS Encryption

```python
from federated_spa import CKKSContext, LocalNode, FederatedAggregator

# Each node encrypts its 5 summary scalars
ctx = CKKSContext()
nodes_encrypted = [node.encrypt_and_transmit(ctx) for node in local_nodes]

# Aggregator sums ciphertexts (no decryption needed)
aggregator = FederatedAggregator(ctx=ctx)
agg_ct = aggregator.aggregate_encrypted(nodes_encrypted)

# Trusted enclave decrypts and computes global SPA p-value
result = aggregator.decrypt_and_compute_pvalue(agg_ct)
```

---

## 🔬 Mathematical Foundation

### 1. Base Model & Null Projection (§2.1)

Under the null hypothesis (no epistatic effect), the GLMM is:

$$g(\boldsymbol{\mu}) = \mathbf{X}\boldsymbol{\alpha} + \mathbf{u}, \quad \mathbf{u} \sim \mathcal{N}(\mathbf{0}, \tau^2 \boldsymbol{\Phi})$$

For binary traits, the IRLS algorithm yields working weights $W = \mu(1-\mu)$. The phenotypic covariance:

$$\mathbf{V} = \mathbf{W}^{-1} + \tau^2 \boldsymbol{\Phi}$$

**Key insight (Remark 1):** The IRLS weights are *already embedded* in **V**. The base projection $\mathbf{P}\_0 = \mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}^\top \mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top \mathbf{V}^{-1}$ operates in the correct heteroscedastic metric without explicit weight injection.

### 2. Epistatic Score Statistic with Generalized FWL (§2.2)

The epistatic kernel is:

$$\mathbf{K}\_{\mathrm{epi}} = (\mathbf{Z}\_A \mathbf{Z}\_A^\top) \odot (\mathbf{Z}\_B \mathbf{Z}\_B^\top)$$

**Proposition 1 (Generalized main-effect immunity).** The FWL projection:

$$\mathbf{P}\_{\mathrm{adj}} = \mathbf{P}\_0 - \mathbf{P}\_0 \mathbf{Z}\_{\mathrm{main}} (\mathbf{Z}\_{\mathrm{main}}^\top \mathbf{P}\_0 \mathbf{Z}\_{\mathrm{main}})^{-1} \mathbf{Z}\_{\mathrm{main}}^\top \mathbf{P}\_0$$

is symmetric, V-metric idempotent, and satisfies $\mathbf{P}\_{\mathrm{adj}} \mathbf{Z}\_{\mathrm{main}} = \mathbf{0}$ exactly.

**Proposition 2 (Dimensionality collapse):**

$$Q\_{\mathrm{adj}} = \frac{1}{2} \left\lVert \mathbf{Z}\_A^\top \mathrm{Diag}(\tilde{\mathbf{y}}) \mathbf{Z}\_B \right\rVert\_F^2$$

This is an $m\_A \times m\_B$ computation — complexity **O(N · mₐ · m_B)**, no N×N matrix formed.

### 3. Dual-Space Deflation-Accelerated Cumulant Extraction (§2.3, Theorem 1)

**Theorem 1 (Symmetric dual-space reduction).** Let $\mathbf{Z}\_{\mathrm{KR}}$ denote the row-wise Khatri-Rao product of $\mathbf{Z}\_A$ and $\mathbf{Z}\_B$. The non-zero eigenvalues of $\mathbf{P}\_{\mathrm{adj}} \mathbf{K}\_{\mathrm{epi}}$ (N×N, asymmetric) are identical to those of:

$$\mathbf{G}\_{\mathrm{dual}} = \mathbf{Z}\_{\mathrm{KR}}^\top \mathbf{P}\_{\mathrm{adj}} \mathbf{Z}\_{\mathrm{KR}}$$

which is $(m\_A m\_B) \times (m\_A m\_B)$, **symmetric positive semi-definite** by construction.

**Algorithm 1 (Hutch++ deflation):**
1. **Step 1 (Deflation):** Allocate S/3 probes for randomized Nyström low-rank approximation of **G**_dual dominant eigenspace. Compute exact trace contribution.
2. **Step 2 (Residual):** Apply standard Hutchinson with remaining 2S/3 Rademacher probes to the well-conditioned residual.

Variance reduction: from $O(\|\mathbf{A}\|\_F^2)$ to $O(\|\mathbf{A} - \mathbf{A}\_k\|\_F^2)$ — **quadratic improvement**.

### 4. Saddlepoint Approximation (§2.5)

The CGF from cumulants:

$$K(t) = \sum\_{j=1}^{4} \kappa\_j \frac{t^j}{j!}$$

Newton-Raphson for saddlepoint root $\hat{t}$ solving $K'(\hat{t}) = q$. **Lugannani-Rice tail probability:**

$$P(Q > q) \approx \bar{\Phi}(\hat{w}) + \phi(\hat{w}) \left(\frac{1}{\hat{w}} - \frac{1}{\hat{u}}\right)$$

where $\hat{w} = \mathrm{sign}(\hat{t})\sqrt{2(\hat{t}q - K(\hat{t}))}$ and $\hat{u} = \hat{t}\sqrt{K''(\hat{t})}$. Computed via `erfc()` for exact precision to **P ≈ 10⁻³⁰⁰**.

### 5. Non-Linear Genomic Control (§2.4)

Augmented null model conditioning on background epistatic variance:

$$\mathbf{V}\_{\mathrm{aug}} = \sigma\_e^2 \mathbf{I} + \tau^2 \boldsymbol{\Phi} + \tau\_{\mathrm{epi}}^2 (\boldsymbol{\Phi} \odot \boldsymbol{\Phi})$$

Scalable estimation via:
- **Stage 1:** GRM sparsification (threshold kinship < 0.05 → 0)
- **Stage 2:** Randomized Haseman-Elston regression for variance components

### 6. Federated Cumulant Additivity (§2.5, Theorem 2)

**Theorem 2:** For K independent cohorts:

$$\kappa\_{j, \mathrm{meta}} = \sum\_{k=1}^{K} \kappa\_{j,k} \quad \forall j \geq 1$$

**Fed-cSPA-HE Protocol:**
1. Each node k computes local $(\kappa\_{1,k}, \ldots, \kappa\_{4,k}, Q\_k)$ via Algorithm 1
2. Encrypt with CKKS homomorphic encryption: $\mathrm{ct}\_k = \mathrm{Enc}(\kappa\_{1,k}, \ldots, Q\_k)$
3. Aggregator sums ciphertexts: $\mathrm{ct}\_{\mathrm{agg}} = \sum\_k \mathrm{ct}\_k$ (no decryption)
4. Trusted enclave decrypts and computes SPA p-value

Only **5 encrypted scalars** leave each node — **zero raw genotype leakage**.

---

## Testing

```bash
# Run all 24 mathematical invariant tests
pip install -e .
pytest tests/test_math_invariants.py -v
```

### Validated Properties

| Test | Property | Status |
|------|----------|--------|
| Theorem 1 | Dual-space eigenvalue equivalence | ✅ Pass |
| Theorem 1 | Trace equivalence (primal = dual) | ✅ Pass |
| Theorem 1 | G_dual is SPSD | ✅ Pass |
| Proposition 1 | FWL annihilation (continuous) | ✅ Pass |
| Proposition 1 | V-metric idempotency | ✅ Pass |
| Proposition 1 | FWL symmetry | ✅ Pass |
| Proposition 1 | Binary trait FWL annihilation | ✅ Pass |
| Proposition 2 | Frobenius = quadratic form | ✅ Pass |
| Algorithm 1 | Hutch++ accuracy vs exact | ✅ Pass |
| Algorithm 1 | Hutch++ variance reduction | ✅ Pass |
| Newton's identities | Known eigenvalue spectrum | ✅ Pass |
| Theorem 2 | First-order trace additivity | ✅ Pass |
| SPA | Valid p-value range | ✅ Pass |
| SPA | Monotonicity | ✅ Pass |
| SPA | Batch consistency | ✅ Pass |
| NL-GC | GRM sparsification | ✅ Pass |
| NL-GC | Hadamard square | ✅ Pass |
| CKKS | Encrypt-decrypt round-trip | ✅ Pass |
| CKKS | Homomorphic addition | ✅ Pass |
| Graph search | Candidate generation | ✅ Pass |
| Graph search | iPTM threshold | ✅ Pass |
| Memory | No N×N allocation | ✅ Pass |
| Security | 5-scalar transmission only | ✅ Pass |
| Security | Context decryption fidelity | ✅ Pass |

## Simulation Experiments (§2.7)

```bash
python simulations/simulate_biobank.py
```

### Experimental Configurations

1. **Human semi-empirical (continuous):** 1KGP genotypes, h² = 0.3, 5000 common causal variants
2. **Binary traits (1:5 to 1:100 imbalance):** Liability threshold model with varying prevalence
3. **Bovine WGS (extreme kinship):** F_avg = 0.06, 130 breeds (1000 Bull Genomes)
4. **Federated partitioning:** 5 superpopulations (AFR, AMR, EAS, EUR, SAS)

### Competing Methods

- Standard-EVD: explicit 4th-order trace (O(N³))
- QuadKAST + Davies: linear but only 2nd-order moments
- Naive Gaussian: no tail correction
- IVW fixed-effect meta-analysis

---

## Citation

```bibtex
@article{MetaRareEpi2026,
  title   = {Dual-Space Federated Saddlepoint Approximation with
             Deflation-Accelerated Cumulant Extraction for Biobank-Scale
             Rare-Variant Epistasis Mapping},
  author  = {[Authors]},
  year    = {2026},
  note    = {Under peer review}
}
```

## License

[MIT License](LICENSE) — see [LICENSE](LICENSE) for details.

## Links

- 📄 [Project Website](https://meibujun.github.io/MetaRareEpi/)
- 🐛 [Issue Tracker](https://github.com/meibujun/MetaRareEpi/issues)
- 📧 Contact: [meibujun@github](https://github.com/meibujun)
