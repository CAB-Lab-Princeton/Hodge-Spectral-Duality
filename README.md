# Hodge Spectral Duality

Neural operator learning on curved domains with the Hodge decomposition as
an inductive bias. Companion code for **"Topology-Preserving Neural Operator
Learning via Hodge Decomposition"** (ICML 2026,
[arXiv:2605.13834](https://arxiv.org/abs/2605.13834)).

The solution operator of a field equation on a mesh, point cloud, or graph
is learned in the eigenbases of the Hodge Laplacians built on the underlying
simplicial complex. Edge-valued (1-form) signals split into three mutually
orthogonal sectors — gradient, curl, and harmonic — where the harmonic
sector is finite-dimensional, fixed by the topology of the domain (its
dimension is the first Betti number), and carried through the model exactly
rather than learned. A gated dual-branch network combines a spectral branch
operating on basis coefficients with a pointwise ambient branch that
supplies the high-frequency content a finite basis has not yet covered; the
ambient branch's contribution shrinks as the basis grows.

```python
from hodge_spectral import HodgeOperator

model = HodgeOperator.from_mesh(points, faces, task="0to1", k=64)
model.fit(X_train, Y_train)
metrics = model.evaluate(X_test, Y_test)   # {'relative_l2': ..., ...}
```

The same interface accepts point clouds (`from_pointcloud`, automatic
triangulation) and graphs (`from_graph`), and any differential-form task
(`"0to0"`, `"0to1"`, `"1to1"`, ...).

## What is new in v0.2

This release refines the spectral layer and its preprocessing; the network
architecture is unchanged.

- **Scales to large meshes.** The enriched-input construction now keeps the
  incidence matrix sparse: on an 85,000-edge mesh this step drops from
  41.7 s / 20.5 GB to 0.18 s / 1.7 GB.
- **Well-defined mode ordering.** Eigenvalues are cached with their
  eigenvectors and returned in ascending order from every solver path, so
  "the first k modes" always means the k smallest.
- **Geometry-consistent spectrum** (`l0_mode="cotan"`). The cotangent
  Laplacian aligns eigenvalue order with intrinsic surface frequency. On the
  torus benchmark this halves the number of modes needed for 95% signal
  capture (165 → 83).
- **Structured edge basis** (`operators.sector`). The 1-form eigenbasis is
  assembled from the smaller node- and face-level problems: numerically
  identical to a direct solve (eigen residual ~1e-13), every mode carries
  its Hodge-sector label, and the harmonic directions are guaranteed to be
  present — the foundation for topology-aware error auditing.
- **Data-adaptive basis selection** (`operators.adaptive`). When a task's
  spectral content is not concentrated at low frequency, per-sector
  principal directions of the training targets (harmonic block preserved)
  reduce the k = 64 capture gap on the torus benchmark from 0.49 to
  0.055–0.071 relative L2.

With these upgrades the measured behaviour matches the theoretical picture:
capture improves monotonically with basis size, the residual concentrates
at high frequencies and recedes as k grows, and the gate's trust in the
spectral branch rises toward 1 (0.79 → 0.975 on the torus benchmark as k
goes 8 → 256) while the ambient branch's share falls (0.77 → 0.12).

## Installation

```bash
pip install -e .              # core library
pip install -e ".[examples]"  # + mesh generation for the example scripts
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

## Choosing a basis

| Option | When to use | Where |
|---|---|---|
| Combinatorial Laplacian (default) | uniform meshes, drop-in with v0.1 behaviour | `HodgeOperator(...)` |
| Geometry-consistent spectrum | meshes with varying triangle geometry | `HodgeOperator(..., l0_mode="cotan")` |
| Structured edge basis with sector labels | topology-aware analysis, guaranteed harmonic modes | `operators.build_sector_basis_1form` |
| Data-adaptive basis | spectral content away from low frequency; tight mode budgets | `operators.adaptive_basis_1form` / `adaptive_basis_0form` |

Projection capture on the torus benchmark (relative L2 of the best possible
k-mode reconstruction; lower is better):

| k | 8 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| combinatorial, normalized | 0.99 | 0.72 | 0.49 | 0.074 | 0.012 |
| geometry-consistent | 1.00 | 0.84 | 0.25 | 0.021 | 0.009 |
| data-adaptive | 0.43 | 0.16 | 0.071 | 0.029 | 0.015 |

## Performance

One-time preprocessing per mesh (cached and shared by all subsequent
training on that mesh): the three eigendecompositions at k = 64 take
0.5 s / 2.1 s / 14.3 s on meshes with 5k / 21k / 85k edges; building the
data-adaptive basis adds a few seconds on top of the decomposition it
rotates. Per training run on the benchmark meshes: about 2 s of data
preparation plus about 8.5 s for 120 epochs with early stopping, once
training tensors are kept resident on the GPU.

## Repository layout

```
hodge_spectral/
  api.py            HodgeOperator: the one-line user interface
  operators/
    spectral.py     Hodge Laplacians, eigenbases, spectral de Rham operators
    geometry.py     cotangent (geometry-consistent) Laplacian
    sector.py       structured 1-form basis with sector labels
    adaptive.py     data-adaptive basis selection
  models/           dual-branch network, datasets, training loops, baselines
  adapters/         mesh / point-cloud / graph input adapters
  data/             synthetic benchmark generators
examples/           runnable end-to-end examples
tests/              basic correctness tests (python -m pytest tests/)
```

## Examples

```bash
python examples/quickstart.py                 # mesh / point cloud / graph, one task
python examples/example_torus_helmholtz.py    # genus-1 vortex flow
python examples/example_ellipsoid_aero.py     # genus-0 external aerodynamics
```

The experiment scaffolding used for the paper and its rebuttal remains on
the `main` branch; this branch carries the library itself.

## Citation

```bibtex
@inproceedings{zheng2026hodge,
  title     = {Topology-Preserving Neural Operator Learning via Hodge Decomposition},
  author    = {Zheng, Dongzhe and Zhong, Tao and Allen-Blanchette, Christine},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## License

MIT
