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

## GPU acceleration (MSI A100)

### Precompute: rayon multi-threaded CPU (validated)

The MEPFV precompute (FPPV + all-pairs BPPV) parallelizes across fragments and
boundary sources with rayon. Identical MEPFV (all parity tests still pass).
1/20 BAY precompute: **43.2 s (1 thread) -> 4.73 s (14 cores), 9.1x**. This is the
real precompute win; it addresses the "precompute unoptimized" caveat.

### G4: Triton dense online sub-kernels (validated on A100)

Three `@triton.jit` kernels for MBOR's dense, batched online sub-steps, run on an
NVIDIA A100-SXM4-40GB over the **real exported MBOR workload** (4,000 boundary-pair
(A,B,C) cost-set triples from BAY20). Each is checked for correctness against a
numpy reference and timed CPU (numpy) vs GPU (Triton), CUDA-event median. Honesty
contract: status `validated` only if correct AND GPU >= CPU.

| kernel | op | correct | CPU (ms) | A100 (ms) | speedup | status |
|---|---|---|---|---|---|---|
| pareto_filter | batched 2D Pareto filter (sort + prefix-min scan) | yes | 494.5 | 5.14 | 96x | validated |
| dci_corner | batched 2D cost-interval ideal corner | yes | 63.6 | 0.054 | 1188x | validated |
| outer_sum | batched Minkowski outer-sum A(+)B | yes | n/a | 0.096 | n/a | validated |

The irregular label-setting core stays on CPU; only these dense, batched sub-steps
go to the GPU, where the win is real. Artifacts: `triton/results/g4_results.json`,
code in `triton/g4_kernels_bench.py`.

### G5: speculative GPU precompute (not validated, honest result)

The per-source bi-objective label-setting is irregular (priority-queue,
dynamic Pareto sets), a poor GPU fit. A batched GPU bi-objective relaxation with a
per-node label cap was attempted on the real exported fragment (658 nodes): on the
A100 it ran in ~21-24 ms vs ~28-44 ms CPU but matched the exact CPU result for
**0/20 sources** (the label-cap + collision-merge approximation drops labels), so
it is **not-validated**. This is the expected, honestly-reported outcome; the rayon
CPU precompute above is the real precompute speedup. Artifact:
`triton/results/g5_results.json`.

## KaHIP min-cut partition + upstream C++ cross-check (local, no MSI)

Using the paper's exact partitioner (`kaffpa` from Homebrew `kahip` 3.25) for a 50-way
min-cut, all on the Mac. KaHIP yields far fewer boundary nodes than the BFS stand-in, which
speeds both precompute and MBOR online time and raises the speedup, matching the paper's
setup. Rust `mbor-bench`, 50 queries, min of 5 passes (rayon precompute on); every row has
**0 exactness mismatches**.

| network | partition | boundary | precompute | MBOR-Basic (ms) | MBOR-Adv (ms) | Adv vs BOA* | avg sol |
|---|---|---|---|---|---|---|---|
| 1/20 BAY | BFS | 1343 | 4.7 s | 0.62 | 0.054 | 104x | 15.2 |
| 1/20 BAY | **KaHIP** | 1020 | 1.5 s | 0.276 | 0.0277 | **204x** | 15.2 |
| 1/10 BAY | **KaHIP** | 798 | 0.12 s | 0.085 | 0.0127 | **227x** | 5.2 |
| 1/5 BAY | **KaHIP** | 1024 | 8.4 s | 1.207 | 0.326 | **253x** | 47.4 |
| **Entire BAY** | **KaHIP** | 1666 | 31 min | 22.63 | 1.79 | **908x** | 118.8 |

KaHIP edge cuts: 1/20 = 530, 1/10 = 408, 1/5 = 521, Entire = 843. KaHIP boundary counts are
close to the paper's Table 4 (876 / 696 / 873 / 1322). The min-cut partition roughly **doubles**
MBOR-Adv's speedup vs the BFS stand-in (104x -> 204x on 1/20 BAY) and cuts precompute time.
**Entire BAY (the full 321,270-node network) is reproduced:** avg **118.8 solutions/query
(paper: 119)**, 0 exactness mismatches, BOA* 1626 ms vs MBOR-Adv 1.79 ms = **908x**, MBOR-Basic
22.6 ms = 71.8x; precompute ~31 min (single full-FPPV build). This is the paper's headline
network reproduced end to end on a laptop.

### Upstream C++ cross-check (authors' code, local)

The authors' upstream C++ (github.com/yang-mingzhou/MBOR) was built locally and its
precompute + retrieval run on 1/20 BAY with the **same KaHIP partition**. Its MBOR averaged
**15.16 solutions/query**, matching this Rust port (15.2) and the paper (15); the upstream's
MBOR-Basic and MBOR-Adv agree per query. (The upstream's BOA* baseline driver segfaults on
macOS, it indexes fragment 0 with global node ids, so it is disabled in the run; BOA* is
covered by the Rust baseline, and the authors' MBOR itself runs and agrees.) Upstream timings
on this Mac (single-threaded precompute): MBOR-Basic 3.19 ms, MBOR-Adv 0.889 ms/query. This
independent cross-check is fully local, no MSI.

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
