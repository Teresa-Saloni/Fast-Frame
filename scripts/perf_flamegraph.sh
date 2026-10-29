#!/usr/bin/env bash
# Profile the release binary with perf and emit a flame graph SVG into docs/.
#   scripts/perf_flamegraph.sh [fastframe args...]
# Needs: linux-tools (perf), https://github.com/brendangregg/FlameGraph on PATH or in ~/FlameGraph
set -euo pipefail
cd "$(dirname "$0")/.."
ARGS=("$@"); [ ${#ARGS[@]} -eq 0 ] && ARGS=(--synthetic 600 --size 1920x1080 --fake-cost 15000 --bench)
FG=${FLAMEGRAPH_DIR:-$HOME/FlameGraph}
cmake --preset release -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer" >/dev/null && cmake --build --preset release >/dev/null
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
mkdir -p docs
perf record -F 999 -g --call-graph dwarf -o /tmp/ff_perf.data ./build/release/fastframe "${ARGS[@]}"
perf report -i /tmp/ff_perf.data --stdio --no-children 2>/dev/null | head -60 > docs/perf_top_${SHA}.txt
perf script -i /tmp/ff_perf.data | "$FG/stackcollapse-perf.pl" | "$FG/flamegraph.pl" --title "fastframe ${SHA}" > docs/flame_${SHA}.svg
perf stat -e cycles,instructions,cache-misses,L1-dcache-load-misses,context-switches,page-faults \
  ./build/release/fastframe "${ARGS[@]}" 2> docs/perf_stat_${SHA}.txt
echo "wrote docs/flame_${SHA}.svg docs/perf_top_${SHA}.txt docs/perf_stat_${SHA}.txt"
