# MBOR: Multi-level Bi-objective Routing (Rust + Triton reproduction)

An independent **Rust** reimplementation and **GPU (Triton)** acceleration of the
bi-objective routing method in:

> Mingzhou Yang, Ruolei Zeng, **Arun Sharma**, Shunichi Sawamura, William F. Northrop,
> Shashi Shekhar. *Towards Pareto-optimality with Multi-level Bi-objective Routing:
> A Summary of Results.* ACM SIGSPATIAL IWCTS 2024.
> [DOI: 10.1145/3681772.3698215](https://doi.org/10.1145/3681772.3698215)

The bi-objective routing (BOR) problem finds the complete set of Pareto-optimal paths
between an origin and a destination in a graph whose edges each carry two non-negative
costs (for example travel time and energy). MBOR makes city-scale BOR queries fast with
three ideas: a **boundary multigraph**, a **Multi-level Encoded Pareto Frontier View
(MEPFV)**, and **two-dimensional cost-interval (2DCI) pruning**.

## Why this repo exists

The published implementation is in C++/C. This repo is a from-scratch, reproducible
**Rust** port (memory-safe, one-command build, `criterion` benchmarks) plus **Triton GPU
kernels** for the dense, batched sub-steps (Minkowski-sum + 2D Pareto filter, 2DCI
pruning), benchmarked CPU vs A100. It reproduces the paper's results on the open
9th DIMACS Challenge San Francisco Bay Area road network.

Honest scope, stated plainly:
- **Rust is not faster than good C++** (same LLVM class). The value here is reproducibility,
  safety, clean benchmarks, and a portable artifact that *reproduces* the paper's speedup on
  open data. Where a number is reproduced it is labeled as such; the paper's published "10x
  online speedup on the full BAY network" is cited as the paper result, separate from any
  number this machine reproduces.
- **The core label-setting loop is a poor GPU fit** (irregular, priority-queue). Triton is
  applied only to the dense, batched sub-kernels, where a GPU win is real and measured.

## Layout

```
rust/                 Cargo workspace
  crates/mbor-core/   CSR graph, DIMACS loader, Pareto frontier, label-setting (Alg 2)
triton/               GPU kernels for the dense sub-steps (Phase G4) + CPU-vs-GPU bench
data/                 open DIMACS Bay Area fetch/vendor
docs/                 project hub (arunshar.com/mbor/), EVIDENCE, explainer
```

## Build and test

```
cd rust
cargo test
cargo bench        # criterion micro-benchmarks
```

## Credit

Builds on the upstream reference by the paper authors:
**https://github.com/yang-mingzhou/MBOR** (C++/C). The baseline algorithms there
(BOA*, NAMOA*dr, Bi-Objective Dijkstra) follow Hernández et al., *Simple and efficient
bi-objective search algorithms via fast dominance checks*, Artificial Intelligence 2023,
and the [BOAstar](https://github.com/jorgebaier/BOAstar/) repository.

## License

MIT (this reimplementation). See `LICENSE`. The upstream repository is the authors' own.
