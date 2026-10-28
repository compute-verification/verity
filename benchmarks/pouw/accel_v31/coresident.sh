#!/usr/bin/env bash
# coresident.cu on one leased H200, cwd = the source tree's root under `research run --cwd source`: the build outside the
# lease (ptxas's registers and spills to build.log), then under it the GPU's clock and power sampled beside
# `coresident REPS KTILES HREPS ITEM_MS KDIM` (coresident.jsonl; defaults 5 4096 2048 2000 4096), and the tables
# (table.md). CFGS (optional): the configurations to run, comma-separated name prefixes (coresident.cu's CFGS). OPS
# (optional, default s8): the operands' kinds, comma-separated, s8 (random int8) and u2 (the 2-bit values 0..3). RASTER
# (optional, default 0, gemm_r5's order): the loaded tiles in bands of RASTER rows.
#   research run --on vy-dc-h200-3 --project verity --campaign C --source . --cwd source --env GPU_LEASE_WHO=LANE -- \
#     bash benchmarks/pouw/accel_v31/coresident.sh
# PROBE (a directory, optional): another probe's sources, built (nvcc -O3 PROBE/PROBE_MAIN) and run first under the same
# lease, its output to probe.jsonl. LEASE (default gpu-lease 1 --wait --max-min 15 --). ITEM_MS: each arm's energy item
# (default 2000).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/accel_v31
G="-O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v"
git rev-parse HEAD > "$O/commit.txt" 2>&1
: > "$O/rc.txt"
# shellcheck disable=SC2086
nvcc $G -o "$O/coresident" $D/coresident.cu -lcuda > "$O/build.log" 2>&1
echo "build rc $?" >> "$O/rc.txt"
[ -x "$O/coresident" ] || { tail -30 "$O/build.log"; exit 1; }
if [ -n "${PROBE:-}" ]; then
  # shellcheck disable=SC2086
  (cd "$PROBE" && nvcc $G -o "$O/probe" "${PROBE_MAIN:-coissue.cu}") > "$O/build-probe.log" 2>&1
  echo "build probe rc $?" >> "$O/rc.txt"
fi
export REPS=${REPS:-5} KTILES=${KTILES:-4096} HREPS=${HREPS:-2048} ITEM_MS=${ITEM_MS:-2000} KDIM=${KDIM:-4096}
${LEASE-gpu-lease 1 --wait --max-min 15 --} bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv > "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv -lms 200 > "$O/smi.csv" 2>&1 &
  smi=$!
  if [ -x "$O/probe" ]; then "$O/probe" ${PROBE_ARGS:-1 2048} > "$O/probe.jsonl" 2> "$O/probe.err"; echo "probe rc $?" >> "$O/rc.txt"; fi
  "$O/coresident" "$REPS" "$KTILES" "$HREPS" "$ITEM_MS" "$KDIM" > "$O/coresident.jsonl" 2> "$O/coresident.err"
  echo "coresident rc $?" >> "$O/rc.txt"
  kill $smi
'
echo "lease rc $?" >> "$O/rc.txt"
python3 - "$O/coresident.jsonl" > "$O/table.md" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
cfg = {r["cfg"]: r for r in rows if "regs" in r}
arms = [r for r in rows if "arm" in r]
alone = {(r["cfg"], r.get("ops")): r for r in arms if r["arm"] == "gemm"}
solo = {(r["cfg"].rsplit("_", 1)[0], r["nhw"]): r for r in arms if r["arm"] == "hash"}
f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
print("theta: of 989.4 T MACs a second; a clock: of 4,096 MACs a SM-clock (clock64). hash / bar: of 4.355 G compressions a "
      "second; a clock: of 0.018029 a SM-clock. Watts: the board's energy over the item (mean) and the most of its power "
      "sampled every 20 ms after the item's first second (max). Operands: s8, random int8; u2, the 2-bit values 0..3 stored as int8 (the hash alone "
      "reads none).\n")
print("| configuration | Y | tile | threads | CTAs | HR | spill B | operands | arm | hash warps | theta | theta a clock "
      "| clock64 MHz | hash / bar | hash / bar a clock | W mean | W max | pJ / MAC | nJ / compression | item s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in arms:
    c = cfg.get(r["cfg"], {})
    print(f'| {r["cfg"]} | {c.get("y", "")} | {64 * c.get("nm", 0)} x {c.get("bn", "")} | {c.get("threads", "")} | '
          f'{c.get("ctas", "")} | {c.get("hr", "")} | {c.get("local", "")} | {r.get("ops", "s8") if r["arm"] != "hash" else "-"} | '
          f'{r["arm"]} | {r["nhw"]} | {f(r["theta"])} | {f(r["theta_clk"])} | '
          f'{r["mhz"]:.0f} | {f(r["hash_bar"])} | {f(r["hash_bar_clk"])} | {f(r.get("w"), 0)} | {f(r.get("wmax"), 0)} | '
          f'{f(r.get("pj_mac"))} | {f(r.get("nj_comp"), 2)} | {r["item_ms"] / 1e3:.1f} |')
print("\nEach GEMM with the hash against the GEMM alone and the hash alone (the same configuration and hash warps). The "
      "hash's rate over its rate alone is its clock's ratio times its rate a clock's ratio (co-issue).\n")
print("| configuration | operands | hash warps | theta alone | theta | theta / alone | a clock / alone | hash alone / bar "
      "| hash / bar | clock / alone (hash) | a clock / alone (hash) | pJ / MAC alone | pJ / MAC | W alone | W |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in arms:
    if r["arm"] != "gemm+hash":
        continue
    a, h = alone.get((r["cfg"], r.get("ops"))), solo.get((r["cfg"].rsplit("_", 1)[0], r["nhw"]))
    q = lambda x, y: f"{x / y:.3f}" if x is not None and y else "-"
    print(f'| {r["cfg"]} | {r.get("ops", "s8")} | {r["nhw"]} | {f(a and a["theta"])} | {f(r["theta"])} | '
          f'{q(r["theta"], a and a["theta"])} | '
          f'{q(r["theta_clk"], a and a["theta_clk"])} | {f(h and h["hash_bar"])} | {f(r["hash_bar"])} | '
          f'{q(r["mhz"], h and h["mhz"])} | {q(r["hash_bar_clk"], h and h["hash_bar_clk"])} | '
          f'{f(a and a.get("pj_mac"))} | {f(r.get("pj_mac"))} | {f(a and a.get("w"), 0)} | {f(r.get("w"), 0)} |')
EOF
cat "$O/table.md"
! grep -qv ' rc 0$' "$O/rc.txt"
