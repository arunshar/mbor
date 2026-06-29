# Building and running the upstream MBOR (parity oracle)

Upstream: https://github.com/yang-mingzhou/MBOR (C++/C, Makefile). It is the
authors' reference implementation and serves as our parity oracle. It is a Linux
codebase (`gcc -fopenmp -mcmodel=medium`, large static arrays, KaHIP for
partitioning). Clone read-only to `~/code/_mbor_upstream`.

## Data (open, bundled in upstream)
`Maps/{test,BAY5,BAY10,BAY20,BAY}-road-d.txt` and `Queries/*-queries`. Map format:
header `<n> <m>`, then `<from> <to> <c1> <c2>` (1-indexed). `test` = the paper's
Figure 1 toy (n0->n7 Pareto frontier = {(10,17),(11,16)}). The 9th DIMACS
Challenge Bay Area network; no registration.

## macOS arm64 build (what it took)
The stock Makefile does not run on macOS arm64. Two Linux assumptions had to be
worked around (algorithm and results unchanged):
1. **OpenMP / libgomp.** The code calls `omp_get_num_procs` / `omp_set_num_threads`
   / `omp_get_max_threads` and uses `#pragma omp`. Homebrew `libgomp` trips the
   arm64 dyld shared-cache mapping. Fix: build without `-fopenmp` (pragmas become
   no-ops, serial) and link a 5-line `omp_shim.c` providing those calls.
2. **3.2 GB static array.** `heap.c` had `#define HEAPSIZE 400000000;
   snode* heap[HEAPSIZE]` (~3.2 GB `__DATA`), which collides with the arm64 dyld
   shared region (the reason the Makefile uses the x86-only `-mcmodel=medium`).
   Fix: `HEAPSIZE 50000000` (~400 MB), ample for BAY (<= 321,270 nodes).
   (`MAXNODES` arrays were already commented out; `MAXNODES` itself is harmless.)

Build (static C++ runtime so the binary depends only on system `libSystem`):
```
cd ~/code/_mbor_upstream/src
gcc-16 -O2 -std=c++11 -std=c99 -static-libstdc++ -static-libgcc -o mbor_precompute \
  bhepvPrecomputation.cpp bhepv.cpp MultiGraphBOD.cpp bodPathRetrieval.c heap.c bod.c graph.c omp_shim.c -lstdc++ -lm
gcc-16 -O2 -std=c++11 -std=c99 -static-libstdc++ -static-libgcc -o mbor_retrieval \
  bhepvPathRetrieval.cpp hborWithBhepv.cpp pathRetrieval.c heap.c boastar.c graph.c omp_shim.c -lstdc++ -lm
```

## Status on macOS
- **Precompute: works.** Toy `test`/k=3 builds the MEPFV (`fragmentEPV.json`,
  `boundaryEPV.json`, `boundaryGraph.txt`, `boundaryNodes.txt`, ...). Good for
  structural parity.
- **Retrieval: segfaults** (exit 139) inside the `boa()` baseline on the first
  query; large-partition precompute is impractically slow without KaHIP.
- Conclusion: run the **full** oracle (precompute + retrieval + KaHIP min-cut
  partition + real timings) on **MSI/Linux**, where it is designed to run
  (`module load gcc/11.3.0`, KaHIP). On macOS, use the precompute structures only.

## Partition file (`bhepv/<map>/kaffpaIndex.txt`)
One partition id per node in node order 0..n-1. KaHIP (`kaffpa`) produces a min-cut
partition; for a quick deterministic stand-in (same partition fed to both upstream
and the Rust port so structural parity holds) use a contiguous block partition:
`partition[i] = (i * k) / n`. KaHIP-quality partitions (for honest timing) are an
MSI step.

## How we use it for G2 parity
- **Solution correctness:** the Rust port's Pareto sets must equal the exact
  full-graph bi-objective search in `mbor-core` (validated against the paper). This
  is the primary, self-contained oracle and needs no upstream binary.
- **Structural parity:** boundary-node counts and encoded-path counts from the
  upstream precompute (toy now; BAY on MSI) cross-check the Rust precompute on the
  same partition.
