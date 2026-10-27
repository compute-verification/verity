#!/usr/bin/env bash
# coresident.cu on one leased H200, cwd = the source tree's root under `research run --cwd source`: the build outside the
# lease (ptxas's registers and spills to build.log), then under it the GPU's clock and power sampled beside
# `coresident REPS KTILES HREPS` (coresident.jsonl; defaults 5 4096 2048), and the table (table.md).
#   research run --on vy-dc-h200-3 --project verity --campaign C --source . --cwd source --env GPU_LEASE_WHO=LANE -- \
#     bash benchmarks/pouw/accel_v31/coresident.sh
# PROBE (a directory, optional): another probe's sources, built (nvcc -O3 PROBE/PROBE_MAIN) and run first under the same
# lease, its output to probe.jsonl. LEASE (default gpu-lease 1 --wait --max-min 15 --).
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
export REPS=${REPS:-5} KTILES=${KTILES:-4096} HREPS=${HREPS:-2048}
${LEASE-gpu-lease 1 --wait --max-min 15 --} bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv > "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv -lms 200 > "$O/smi.csv" 2>&1 &
  smi=$!
  if [ -x "$O/probe" ]; then "$O/probe" ${PROBE_ARGS:-1 2048} > "$O/probe.jsonl" 2> "$O/probe.err"; echo "probe rc $?" >> "$O/rc.txt"; fi
  "$O/coresident" "$REPS" "$KTILES" "$HREPS" > "$O/coresident.jsonl" 2> "$O/coresident.err"
  echo "coresident rc $?" >> "$O/rc.txt"
  kill $smi
'
echo "lease rc $?" >> "$O/rc.txt"
python3 - "$O/coresident.jsonl" > "$O/table.md" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
cfg = {r["cfg"]: r for r in rows if "regs" in r}
alone = {r["cfg"]: r for r in rows if r.get("arm") == "gemm"}
solo = {(r["cfg"].rsplit("_", 1)[0], r["nhw"]): r for r in rows if r.get("arm") == "hash"}
print("| configuration | Y | tile | threads | HR | regs | spill B | hash warps | theta alone | theta | theta / alone | hash / bar "
      "| hash alone / bar | SM MHz | ms |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    if r.get("arm") != "gemm+hash":
        continue
    c, a = cfg.get(r["cfg"], {}), alone.get(r["cfg"])
    h = solo.get((r["cfg"].rsplit("_", 1)[0], r["nhw"]))
    rel = f'{r["theta"] / a["theta"]:.3f}' if a and a["theta"] else ""
    print(f'| {r["cfg"]} | {c.get("y", "")} | {64 * c.get("nm", 0)} x {c.get("bn", "")} | {c.get("threads", "")} | '
          f'{c.get("hr", "")} | {c.get("regs", "")} | {c.get("local", "")} | {r["nhw"]} | {a["theta"] if a else 0:.4f} | '
          f'{r["theta"]:.4f} | {rel} | {r["hash_bar"]:.4f} | {h["hash_bar"] if h else float("nan"):.4f} | {r["mhz"]:.0f} | '
          f'{r["ms"]:.3f} |')
EOF
cat "$O/table.md"
