# MBOR: reproduced evidence

All numbers here are produced by running this repo's Rust code. They are kept
distinct from the **paper's published numbers** (which were measured by the
authors' C++ on their machine with KaHIP partitions). Where a figure is
reproduced it says so, with the machine and the caveats.

## Correctness (the important one)

`Mepfv::query` (MBOR-Basic, Alg 3) and `query_adv` (MBOR-Adv, Alg 4) return the
**exact same Pareto frontier** as two independent compute-on-demand baselines
(BOA*, bi-objective Dijkstra) and as `mbor-core`'s exact full-graph search:

- Unit/integration tests: every node pair on the paper toy (Figure 1) and
  synthetic grids (6x6, 8x4) across many partition counts. `cargo test` = all
  pass (mbor-core 8, mbor-precompute parity 10, mbor-baseline 2+all-pairs).
- On **real BAY data**: the benchmark verifies all four methods agree on every
  query. BAY20: `exactness_mismatches = 0` over 50 queries.

The paper toy frontier `n0 -> n7 = {(10,17),(11,16)}` is reproduced by every
method.

## Speedup benchmark (reproduced, this machine)

Machine: Apple Silicon Mac, `cargo build --release`. Partition: **BFS
region-growing** (dependency-free, compact fragments). This is NOT KaHIP min-cut;
KaHIP would cut the boundary-node count further and speed MBOR up more, so these
are conservative MBOR numbers. Online time = average per query, min of 3 passes.

### 1/20 BAY (`BAY20`, 15,366 nodes, 41,180 edges, k=50, 1,343 boundary nodes)

| method | avg per-query online (ms) | speedup vs BOA* |
|---|---|---|
| BOD (bi-objective Dijkstra) | 16.9221 | 0.3x |
| BOA* | 5.5896 | 1.0x |
| MBOR-Basic | 0.6235 | 9.0x |
| MBOR-Adv | 0.0538 | 103.9x |

- avg Pareto solutions/query = **15.2** (paper Table 4 reports **15** for 1/20 BAY).
- MBOR-Adv 2DCI pruning skips **94%** of boundary-pair combinations (28,584 / 30,542).
- precompute_time = 39.1 s (one-time; see caveat).

**vs the paper (Table 5, 1/20 BAY, KaHIP, authors' C++):** MBOR-Basic 0.64 ms,
MBOR-Adv 0.38 ms, BOA* 4.34 ms. Our MBOR-Basic (0.62 ms) lands right on the
paper's 0.64 ms, and the paper's headline ">10x online speedup with
precomputation" is reproduced (MBOR-Adv is 104x over BOA* here; even MBOR-Basic
is 9x). Absolute ms differ by machine and partition.

### 1/10 BAY (`BAY10`) and 1/5 BAY (`BAY5`)

Running (precompute-bound). Appended when the run completes; same harness:
`mbor-bench <map> <queries> 50 3`.

## How to reproduce

```
cd rust
cargo test                       # correctness on toy + grids
cargo build --release
bash ../data/fetch_dimacs_bay.sh # vendor open BAY maps from upstream
./target/release/mbor-bench ../data/maps/BAY20-road-d.txt ../data/queries/BAY20-queries 50 3
```

## Caveats / honest scope

- **Precompute is unoptimized.** It stores the full FPPV (per boundary node, a
  Pareto set to every node in its fragment), which is `O(boundary x fragment)`
  memory/time. This is fine for the BAY subnets but is the bottleneck at full-BAY
  scale; the paper's encoding is more compact. The reproduced **online** numbers
  are the headline and are not affected.
- **Partition is BFS, not KaHIP.** A KaHIP min-cut partition (the paper's choice)
  would reduce boundary nodes and improve MBOR's online time further. A KaHIP run
  on MSI is pending (the MSI module environment breaks `git https`, fixable).
- **MSI timings** of this same Rust harness are pending (a second, larger
  machine), to sit alongside these Mac numbers.
