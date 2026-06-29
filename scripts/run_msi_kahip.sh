#!/usr/bin/env bash
# A3 (corrected): build KaHIP on MSI and run the KaHIP-min-cut-partition cross-check.
# Fix vs prior attempt: KaHIP's compile_withcmake.sh does `cd "${0%/*}"`, which
# breaks when invoked as a bare filename -> must run as `bash ./compile_withcmake.sh`.
set -uo pipefail
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
RES="$HOME/mbor_msi_results"; mkdir -p "$RES"
MAPS="$HOME/mbor_oracle/MBOR/Maps"; Q="$HOME/mbor_oracle/MBOR/Queries"
BENCH="$HOME/mbor_rust/target/release/mbor-bench"

if [ ! -x "$HOME/kahip2/KaHIP/deploy/kaffpa" ]; then
  module load gcc/13.1.0-5z64cho cmake/3.26.3-gcc-13.1.0-em4tlmo 2>/dev/null || true
  cd "$HOME/kahip2/KaHIP"
  LOG "building KaHIP (./compile_withcmake.sh) ..."
  bash ./compile_withcmake.sh 2>&1 | tail -10
fi
KAFFPA="$HOME/kahip2/KaHIP/deploy/kaffpa"
LOG "kaffpa: $(ls -la "$KAFFPA" 2>&1 | tail -1)"
[ -x "$KAFFPA" ] || { LOG "KAHIP BUILD FAILED"; exit 1; }

module load gcc/13.1.0-5z64cho 2>/dev/null || true
CONV="$HOME/mbor_oracle/MBOR/src/kahip_convert"
[ -x "$CONV" ] || gcc -O3 -std=c++11 -o "$CONV" "$HOME/mbor_oracle/MBOR/src/kahip.cpp" -lstdc++ 2>&1 | grep -i error

for M in BAY20 BAY10 BAY5; do
  LOG "=== KaHIP $M ==="
  "$CONV" "$MAPS/$M-road-d.txt" "$RES/$M.kahip.graph" 2>&1 | tail -1
  timeout 1800 "$KAFFPA" "$RES/$M.kahip.graph" --k 50 --preconfiguration=eco \
     --output_filename="$RES/$M.kaffpaIndex.txt" 2>&1 | grep -iE "cut|time|imbalance|balance" | tail -4
  # Rust with KaHIP partition (fewer boundary nodes -> less precompute memory + faster MBOR)
  ( source "$HOME/.cargo/env" 2>/dev/null; timeout 5400 "$BENCH" "$MAPS/$M-road-d.txt" "$Q/$M-queries" 50 5 "$RES/$M.kaffpaIndex.txt" > "$RES/$M.msi.kahip.log" 2>&1; LOG "rust-kahip $M rc=$?"; tail -14 "$RES/$M.msi.kahip.log" )
  # Upstream C++ cross-check on the SAME KaHIP partition
  BD="$HOME/mbor_oracle/MBOR/bhepv/$M"; mkdir -p "$BD/fragments"; cp "$RES/$M.kaffpaIndex.txt" "$BD/kaffpaIndex.txt"
  ( cd "$HOME/mbor_oracle/MBOR/src"
    timeout 3000 ./mbor_precompute "$M" 50 > "$RES/$M.upstream.precompute.log" 2>&1; LOG "upstream precompute $M rc=$?"
    timeout 3000 ./mbor_retrieval  "$M" 50 > "$RES/$M.upstream.retrieval.log"  2>&1; LOG "upstream retrieval $M rc=$?"
    grep -E "Average (MBOR|HBOR|BOA)|solutions" "$RES/$M.upstream.retrieval.log" | head -20 )
done
LOG "KAHIP DONE"; ls -la "$RES"
