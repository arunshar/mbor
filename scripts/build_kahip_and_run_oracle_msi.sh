#!/usr/bin/env bash
# Build KaHIP on MSI and re-run the upstream MBOR oracle with min-cut partitions
# (the synthesized partitions make too many boundary nodes, so the BPPV encode
# OOM/times-out). KaHIP gives compact fragments -> precompute + retrieval finish,
# yielding real per-query Pareto solution counts + MBOR vs BOA* timings.
set -uo pipefail
module load gcc/13.1.0-5z64cho 2>/dev/null || true
module load cmake/3.26.3-gcc-13.1.0-em4tlmo 2>/dev/null || true
echo "gcc: $(gcc --version 2>/dev/null | head -1)"; echo "cmake: $(cmake --version 2>/dev/null | head -1)"

ROOT="$HOME/mbor_oracle"; RES="$ROOT/results"; mkdir -p "$RES"
cd "$ROOT"

# --- Build KaHIP (sequential kaffpa is enough) ---
if [ ! -x "$ROOT/kahip/KaHIP/deploy/kaffpa" ]; then
  mkdir -p "$ROOT/kahip"; cd "$ROOT/kahip"
  [ -d KaHIP ] || git clone --depth 1 https://github.com/KaHIP/KaHIP 2>&1 | tail -2
  cd KaHIP
  echo "=== compiling KaHIP (this takes a while) ==="
  bash compile_withcmake.sh 2>&1 | tail -8
  cd "$ROOT"
fi
echo "=== kaffpa: ==="; ls -la "$ROOT/kahip/KaHIP/deploy/kaffpa" 2>&1 | tail -1
KAFFPA="$ROOT/kahip/KaHIP/deploy/kaffpa"

# --- Build the upstream map->KaHIP converter ---
cd "$ROOT/MBOR/src"
gcc -O3 -std=c++11 -o kahip_convert kahip.cpp -lstdc++ 2>&1 | grep -i error | head

run_map () {
  local M=$1 K=$2
  echo "===== MAP=$M K=$K (KaHIP) ====="
  local BD="$ROOT/MBOR/bhepv/$M"
  mkdir -p "$BD/fragments"
  ./kahip_convert "../Maps/$M-road-d.txt" "$BD/kahip.graph" 2>&1 | tail -1
  echo "--- kaffpa partition ---"
  timeout 1800 "$KAFFPA" "$BD/kahip.graph" --k "$K" --preconfiguration=eco \
     --output_filename="$BD/kaffpaIndex.txt" 2>&1 | grep -iE "cut|time|partition" | tail -4
  echo "--- precompute ---"
  timeout 3000 ./mbor_precompute "$M" "$K" > "$RES/$M.kahip.precompute.log" 2>&1; echo "precompute rc=$?"
  tail -2 "$RES/$M.kahip.precompute.log"
  echo "--- retrieval ---"
  timeout 3000 ./mbor_retrieval "$M" "$K" > "$RES/$M.kahip.retrieval.log" 2>&1; echo "retrieval rc=$?"
  grep -E "Average (MBOR|HBOR|BOA)|Query \(" "$RES/$M.kahip.retrieval.log" | head -40
}

for M in BAY20 BAY10 BAY5; do run_map "$M" 50; done
echo "=== ALL DONE; KaHIP logs in $RES (*.kahip.*) ==="
ls -la "$RES"
