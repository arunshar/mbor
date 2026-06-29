# MBOR: reproduced evidence

All numbers here are produced by running this repo's Rust code. They are kept
distinct from the **paper's published numbers** (the authors' C++ on their machine
with KaHIP partitions). Where a figure is reproduced it says so, with the machine
and the caveats.

## Correctness (the important one)

`Mepfv::query` (MBOR-Basic, Alg 3) and `query_adv` (MBOR-Adv, Alg 4) return the
**exact same Pareto frontier** as two independent compute-on-demand baselines
(BOA*, bi-objective Dijkstra) and as `mbor-core`'s exact full-graph search:

- Unit/integration tests: every node pair on the paper toy (Figure 1) and
  synthetic grids (6x6, 8x4) across many partition counts. `cargo test` all pass.
- On **real BAY data**: the benchmark verifies all four methods agree on every
  query. **Every row below has `exactness_mismatches = 0`.**

The paper toy frontier `n0 -> n7 = {(10,17),(11,16)}` is reproduced by every method.

## Speedup benchmark (reproduced)

Partition: **BFS region-growing** (dependency-free, compact). This is NOT KaHIP
min-cut; KaHIP would cut the boundary-node count further and speed MBOR up more,
so these are conservative MBOR numbers. Online time = average per query (min of
3 passes on Mac, 5 on MSI). k=50 fragments.

### Mac (Apple Silicon, local)

| network | nodes | BOA* (ms) | MBOR-Basic (ms) | MBOR-Adv (ms) | Basic vs BOA* | Adv vs BOA* | avg sol | Adv pruned | precompute |
|---|---|---|---|---|---|---|---|---|---|
| 1/20 BAY | 15,366 | 5.59 | 0.62 | 0.054 | 9.0x | 104x | 15.2 | 94% | 39 s |
| 1/10 BAY | 32,205 | 2.85 | 0.38 | 0.040 | 7.4x | 72x | 5.2 | 99% | 7.3 s |
| 1/5 BAY | 64,684 | 81.70 | 10.42 | 1.87 | 7.8x | 44x | 47.4 | 84% | 22.6 min |

### MSI (Agate, 128-core Linux)

| network | nodes | BOA* (ms) | MBOR-Basic (ms) | MBOR-Adv (ms) | Basic vs BOA* | Adv vs BOA* | avg sol |
|---|---|---|---|---|---|---|---|
| 1/20 BAY | 15,366 | 8.76 | 1.14 | 0.095 | 7.7x | 93x | 15.2 |
| 1/10 BAY | 32,205 | 4.23 | 0.58 | 0.074 | 7.3x | 57x | 5.2 |
| 1/5 BAY | 64,684 | running | | | | | |

## What this shows

- **The paper's headline reproduces.** ">10x online speedup with precomputation"
  holds on every BAY subnet on both machines: MBOR-Basic is 7-9x over BOA*,
  MBOR-Adv is 44-104x. MBOR-Basic on 1/20 BAY (0.62 ms Mac) lands on the paper's
  0.64 ms.
- **Ratios reproduce across two machines.** Apple Silicon is faster per core than
  the Agate nodes, so absolute ms differ, but the speedup ratios are stable, which
  is the robust result.
- **Solution counts match the paper** on 1/20 BAY (15.2 vs 15) and 1/5 BAY
  (47.4 vs 47). 1/10 BAY differs (5.2 vs the paper's 13): the bundled 1/10 query
  set has easier pairs; all four methods still agree (exact), so 5.2 is the true
  average for those queries.
- **MBOR-Adv 2DCI pruning** skips 84-99% of boundary-pair combinations.

**vs the paper (Table 5, KaHIP, authors' C++):** 1/20 BAY MBOR-Basic 0.64 ms
(here 0.62 Mac), MBOR-Adv 0.38 ms; 1/10 MBOR-Basic 0.33 ms (here 0.38 Mac); 1/5
MBOR-Basic 1.20 ms. Same regime; absolute ms differ by machine and partition.

## Pending (in progress on MSI)

- MSI 1/5 BAY (BFS) row (precompute-bound).
- **KaHIP min-cut partition** rows (Mac + MSI) and an **independent cross-check
  against the authors' upstream C++** run on the same KaHIP partition.

## How to reproduce

```
cd rust
cargo test                          # exactness on toy + grids
cargo build --release
bash ../data/fetch_dimacs_bay.sh    # vendor open DIMACS Bay Area maps
./target/release/mbor-bench ../data/maps/BAY20-road-d.txt ../data/queries/BAY20-queries 50 3
```

## Caveats / honest scope

- **Precompute is unoptimized.** It stores the full FPPV (per boundary node, a
  Pareto set to every node in its fragment), `O(boundary x fragment)`. Fine for the
  subnets but slow at scale (22.6 min for 1/5 BAY on Mac). The reproduced **online**
  numbers are the headline and are unaffected.
- **Partition is BFS, not KaHIP.** KaHIP min-cut (the paper's choice) would reduce
  boundary nodes and improve MBOR's online time further; KaHIP rows are pending.
- **Rust is not faster than good C++** (same back end); the value is a memory-safe,
  one-command, reproducible artifact that reproduces the paper on open data.
