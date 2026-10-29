#!/usr/bin/env bash
# amx_clk.c on a GPU node's host: AMX TDPBSSD a core with the core's clock read on its sibling thread, on one core and
# on every core. Writes $OUT/cpu.txt, amx_clk.json and load.txt (the host's load before and after).
#   research run --on vy-dc-h200-5 --project verity --campaign pouw-night --source <tree> --cwd source \
#     -- bash benchmarks/pouw/np_floor/amx_clk.sh [SECONDS]
set -euo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SECS=${1:-3}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
lscpu >"$OUT/cpu.txt"
gcc -O3 -march=sapphirerapids -pthread "$HERE/amx_clk.c" -o "$WORK/amx_clk"
load() { echo "== $1"; cat /proc/loadavg; ps -eo pcpu,pid,comm --sort=-pcpu | head -12; } >>"$OUT/load.txt"
load before
"$WORK/amx_clk" "$SECS" | tee "$OUT/amx_clk.json"
load after
