#!/usr/bin/env bash
# Build + run the upstream MBOR (parity oracle) on MSI/Agate (native Linux).
# Produces reference per-query Pareto solution counts + timings on BAY subnets.
# Uses a dependency-free BFS region-growing partition (KaHIP-quality min-cut is a
# later refinement); solution counts are partition-independent (MBOR is exact).
set -uo pipefail

GCC_MOD="gcc/13.1.0-5z64cho"
module load "$GCC_MOD" 2>/dev/null || true
echo "gcc: $(gcc --version 2>/dev/null | head -1)"

WORK="$HOME/mbor_oracle"; RES="$WORK/results"
mkdir -p "$WORK" "$RES"; cd "$WORK"
if [ ! -d MBOR ]; then
  git clone --depth 1 https://github.com/yang-mingzhou/MBOR MBOR 2>&1 | tail -2
fi
cd MBOR/src

# Shrink the 3.2 GB static heap to ~400 MB (ample for BAY <= 321,270 nodes); faster build/load.
sed -i 's/#define HEAPSIZE 400000000/#define HEAPSIZE 50000000/' heap.c

echo "=== build precompute + retrieval (OpenMP, native) ==="
gcc -fopenmp -O3 -std=c++11 -std=c99 -o mbor_precompute \
  bhepvPrecomputation.cpp bhepv.cpp MultiGraphBOD.cpp bodPathRetrieval.c heap.c bod.c graph.c -lstdc++ -lm 2>&1 | grep -iE "error|undefined" | head
gcc -fopenmp -O3 -std=c++11 -std=c99 -o mbor_retrieval \
  bhepvPathRetrieval.cpp hborWithBhepv.cpp pathRetrieval.c heap.c boastar.c graph.c -lstdc++ -lm 2>&1 | grep -iE "error|undefined" | head
ls -la mbor_precompute mbor_retrieval 2>&1 | tail -2

# BFS region-growing partitioner -> kaffpaIndex.txt (one part id per node, 0-based).
cat > /tmp/partition.py <<'PY'
import sys, collections
mapf, outf, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(mapf) as f:
    n, m = map(int, f.readline().split())
    adj = [[] for _ in range(n)]
    for line in f:
        p = line.split()
        if len(p) < 4: continue
        u, v = int(p[0])-1, int(p[1])-1
        adj[u].append(v); adj[v].append(u)
part = [-1]*n
target = (n + k - 1)//k
seeds = [(i*n)//k for i in range(k)]
queues = []
for fid, s in enumerate(seeds):
    if part[s] == -1: part[s] = fid
    queues.append(collections.deque([s]))
counts = [sum(1 for x in part if x==fid) for fid in range(k)]
assigned = sum(1 for x in part if x != -1)
progress = True
while assigned < n and progress:
    progress = False
    for fid in range(k):
        if counts[fid] >= target: continue
        q = queues[fid]
        while q and counts[fid] < target:
            u = q.popleft()
            for v in adj[u]:
                if part[v] == -1:
                    part[v] = fid; counts[fid]+=1; assigned+=1; q.append(v); progress=True
                    if counts[fid] >= target: break
        if q: progress = True
for i in range(n):
    if part[i] == -1: part[i] = 0
open(outf,'w').write('\n'.join(map(str, part))+'\n')
print(f"partition n={n} k={k} sizes(min/max)={min(counts)}/{max(counts)}")
PY

run_map () {
  local M=$1 K=$2
  echo "===== MAP=$M K=$K ====="
  mkdir -p "$WORK/MBOR/bhepv/$M/fragments"
  python3 /tmp/partition.py "../Maps/$M-road-d.txt" "$WORK/MBOR/bhepv/$M/kaffpaIndex.txt" "$K" 2>&1 | tail -2
  echo "--- precompute ---"
  timeout 2400 ./mbor_precompute "$M" "$K" > "$RES/$M.precompute.log" 2>&1 || echo "precompute rc=$?"
  tail -2 "$RES/$M.precompute.log"
  echo "--- retrieval ---"
  timeout 2400 ./mbor_retrieval "$M" "$K" > "$RES/$M.retrieval.log" 2>&1 || echo "retrieval rc=$?"
  grep -E "Query \(|Average (MBOR|HBOR|BOA)" "$RES/$M.retrieval.log" | head -60
}

for M in BAY20 BAY10 BAY5; do run_map "$M" 50; done
echo "=== ALL DONE; logs in $RES ==="
ls -la "$RES"
