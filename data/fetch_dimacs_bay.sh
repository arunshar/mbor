#!/usr/bin/env bash
# Vendor the open San Francisco Bay Area road networks used by the MBOR paper.
#
# The bi-objective maps (each line "<from> <to> <c1> <c2>", 1-indexed) and the
# matching query files ship in the upstream reference repo, which itself derives
# them from the 9th DIMACS Implementation Challenge (Shortest Path) Bay Area
# network. This script copies them from a local upstream clone if present, and
# otherwise clones upstream read-only. All data is open; no registration needed.
#
# Usage:  data/fetch_dimacs_bay.sh [UPSTREAM_DIR]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM="${1:-$HOME/code/_mbor_upstream}"
MAPS_OUT="$HERE/maps"
Q_OUT="$HERE/queries"

if [ ! -d "$UPSTREAM" ]; then
  echo "Cloning upstream MBOR (read-only) to $UPSTREAM ..."
  git clone --depth 1 https://github.com/yang-mingzhou/MBOR "$UPSTREAM"
fi

mkdir -p "$MAPS_OUT" "$Q_OUT"
# Map sizes: test (toy = paper Fig 1), BAY5/BAY10/BAY20 subnetworks, full BAY.
for m in test BAY5 BAY10 BAY20 BAY; do
  src="$UPSTREAM/Maps/${m}-road-d.txt"
  [ -f "$src" ] && cp "$src" "$MAPS_OUT/" && echo "map:   ${m}-road-d.txt"
done
for q in test BAY5 BAY10 BAY20 BAY; do
  src="$UPSTREAM/Queries/${q}-queries"
  [ -f "$src" ] && cp "$src" "$Q_OUT/" && echo "query: ${q}-queries"
done

echo "Done. Maps in $MAPS_OUT, queries in $Q_OUT"
echo "Reference: 9th DIMACS Implementation Challenge (Shortest Path), Bay Area network."
