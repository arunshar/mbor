#!/usr/bin/env bash
# MSI track for MBOR G3: (A2) build the Rust workspace and benchmark on MSI with
# a BFS partition; (A3) build KaHIP (clone BEFORE module load to dodge the
# krb5/OpenSSL git-https break), make min-cut partitions, re-benchmark the Rust
# with them, and run the upstream C++ with the SAME partition as a cross-check.
set -uo pipefail
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
RUST="$HOME/mbor_rust"
MAPS="$HOME/mbor_oracle/MBOR/Maps"
Q="$HOME/mbor_oracle/MBOR/Queries"
RES="$HOME/mbor_msi_results"; mkdir -p "$RES"

# --- rustup userspace (no modules, so crates.io https works) ---
if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  LOG "installing rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal 2>&1 | tail -3
fi
source "$HOME/.cargo/env"
LOG "cargo: $(cargo --version 2>&1)"

cd "$RUST"
LOG "cargo build --release ..."
cargo build --release 2>&1 | tail -6
BENCH="$RUST/target/release/mbor-bench"
[ -x "$BENCH" ] || { LOG "BUILD FAILED"; exit 1; }

# --- A2: BFS-partition benchmark on MSI ---
for M in BAY20 BAY10 BAY5; do
  LOG "=== A2 BFS $M ==="
  timeout 5400 "$BENCH" "$MAPS/$M-road-d.txt" "$Q/$M-queries" 50 5 > "$RES/$M.msi.bfs.log" 2>&1
  LOG "rc=$?"; tail -14 "$RES/$M.msi.bfs.log"
done

# --- A3: KaHIP build (clone before loading modules) ---
if [ ! -x "$HOME/kahip2/KaHIP/deploy/kaffpa" ]; then
  LOG "cloning KaHIP (no modules loaded yet)..."
  mkdir -p "$HOME/kahip2"; cd "$HOME/kahip2"
  [ -d KaHIP ] || git clone --depth 1 https://github.com/KaHIP/KaHIP 2>&1 | tail -3
  module load gcc/13.1.0-5z64cho cmake/3.26.3-gcc-13.1.0-em4tlmo 2>/dev/null || true
  cd KaHIP && LOG "compiling KaHIP ..." && bash compile_withcmake.sh 2>&1 | tail -8
fi
KAFFPA="$HOME/kahip2/KaHIP/deploy/kaffpa"
LOG "kaffpa: $(ls -la "$KAFFPA" 2>&1 | tail -1)"

if [ -x "$KAFFPA" ]; then
  module load gcc/13.1.0-5z64cho 2>/dev/null || true
  CONV="$HOME/mbor_oracle/MBOR/src/kahip_convert"
  [ -x "$CONV" ] || gcc -O3 -std=c++11 -o "$CONV" "$HOME/mbor_oracle/MBOR/src/kahip.cpp" -lstdc++ 2>&1 | grep -i error
  for M in BAY20 BAY10 BAY5; do
    LOG "=== A3 KaHIP $M ==="
    "$CONV" "$MAPS/$M-road-d.txt" "$RES/$M.kahip.graph" 2>&1 | tail -1
    timeout 1800 "$KAFFPA" "$RES/$M.kahip.graph" --k 50 --preconfiguration=eco \
       --output_filename="$RES/$M.kaffpaIndex.txt" 2>&1 | grep -iE "cut|time" | tail -3
    # Rust with KaHIP partition (cargo not needed; binary already built)
    ( source "$HOME/.cargo/env"; timeout 5400 "$BENCH" "$MAPS/$M-road-d.txt" "$Q/$M-queries" 50 5 "$RES/$M.kaffpaIndex.txt" > "$RES/$M.msi.kahip.log" 2>&1; LOG "rust-kahip rc=$?"; tail -14 "$RES/$M.msi.kahip.log" )
    # Upstream C++ cross-check with the SAME KaHIP partition
    BD="$HOME/mbor_oracle/MBOR/bhepv/$M"; mkdir -p "$BD/fragments"; cp "$RES/$M.kaffpaIndex.txt" "$BD/kaffpaIndex.txt"
    ( cd "$HOME/mbor_oracle/MBOR/src"
      timeout 3000 ./mbor_precompute "$M" 50 > "$RES/$M.upstream.precompute.log" 2>&1; LOG "upstream precompute rc=$?"
      timeout 3000 ./mbor_retrieval  "$M" 50 > "$RES/$M.upstream.retrieval.log"  2>&1; LOG "upstream retrieval rc=$?"
      grep -E "Average (MBOR|HBOR|BOA)|Query \(" "$RES/$M.upstream.retrieval.log" | head -20 )
  done
fi
LOG "ALL DONE"; ls -la "$RES"
