#!/usr/bin/env bash
# The relay's compute route on node 2's host (`cpu_e4m3.c`): exact E4M3 group sums on AMX-BF16 and AVX-512 BF16, gated
# against the exact sum, on verity's 48 CPUs (no GPU):
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting \
#     --declared-output 'out/*' -- taskset -c 48-95 bash benchmarks/link_pricing/run_cpu.sh
#
# Outputs under $RESEARCH_RUN_DIR/out: build.txt, cpu.jsonl.
set -euo pipefail
cd "$(dirname "$0")"
O=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$O"
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
{ gcc --version | head -1; grep -o -w -m1 'amx_bf16\|amx_int8\|amx_fp16\|avx512_bf16\|avx512_fp16' /proc/cpuinfo | sort -u | tr '\n' ' '; echo;
  taskset -pc $$; } > "$O/build.txt" 2>&1
gcc -O2 -march=sapphirerapids -mamx-tile -mamx-bf16 -mavx512bf16 -pthread cpu_e4m3.c -o "$S/cpu_e4m3" -lm >> "$O/build.txt" 2>&1
sha256sum cpu_e4m3.c run_cpu.sh "$S/cpu_e4m3" >> "$O/build.txt"
N=$(nproc)
for t in 1 "$N"; do "$S/cpu_e4m3" --threads "$t" --seconds 3 | tee -a "$O/cpu.jsonl"; done
