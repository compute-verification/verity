#!/usr/bin/env bash
# pouw_hash on one RTX PRO 6000, from the shipped tree (build.sh): the tree against MANIFEST.sha256, the device's name and UUID,
# the clock lock on this GPU only, an nvidia-smi trace of this GPU, then the gates, then the timing:
#   bash run.sh          the device's vectors against host_check.jsonl (check.py --against), then `rates` and `epi` (whose own
#                        gate recomputes warp 0's digests on the host before it times anything)
#   bash run.sh check    the gates only
# Env: POUW_EXPECT_UUID (required): the GPU's UUID from internal/pouw/rtx-pro/server.md, or "pod" on a one-GPU pod (refused
# if more than one GPU is visible); LOCK_MHZ: lock this GPU's SM clock there (nvidia-smi -i UUID -lgc, reset on exit), else
# the clocks are recorded and the rows labelled unlocked; MS: each row's target milliseconds (default 100).
# Exit: 3 the wrong device; 5 the tree does not match its manifest; 4 a gate failed (nothing timed); else the bench's.
# Outputs under $RESEARCH_RUN_DIR/out (or ./out): device.txt, env.txt, inputs.check, clocks.txt, smi_trace.csv,
# check_device.jsonl, check_summary.jsonl, rates.jsonl, epi.jsonl, summary.txt.
set -uo pipefail
cd "$(dirname "$0")"
MODE=${1:-run}
OUT=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$OUT"
if ! sha256sum -c MANIFEST.sha256 > "$OUT/inputs.check" 2>&1; then echo "REFUSED: the tree does not match MANIFEST.sha256"; exit 5; fi
WANT=${POUW_EXPECT_UUID:-}
if [ "$WANT" = pod ]; then
  [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" = 1 ] || { echo "REFUSED: POUW_EXPECT_UUID=pod with more than one GPU"; exit 3; }
  WANT=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
fi
[ -n "$WANT" ] || { echo "REFUSED: POUW_EXPECT_UUID is not set"; exit 3; }
export POUW_EXPECT_UUID=$WANT
NAME=$(nvidia-smi -i "$WANT" --query-gpu=name --format=csv,noheader 2> /dev/null | head -1)
echo "$NAME $WANT CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" > "$OUT/device.txt"
case "$NAME" in *"RTX PRO 6000"*) ;; *) echo "REFUSED: GPU $WANT is '$NAME', not an RTX PRO 6000"; exit 3 ;; esac
{ nvidia-smi -i "$WANT"; nvidia-smi -i "$WANT" -q -d CLOCK,PERFORMANCE,POWER; cat toolchain.txt source.txt; lscpu | grep -E "Model name|^CPU\(s\)"; grep PRETTY /etc/os-release; } > "$OUT/env.txt" 2>&1
LOCKED=unlocked
if [ -n "${LOCK_MHZ:-}" ] && nvidia-smi -i "$WANT" -lgc "$LOCK_MHZ,$LOCK_MHZ" >> "$OUT/env.txt" 2>&1; then
  LOCKED="locked $LOCK_MHZ MHz"
  trap 'nvidia-smi -i "$WANT" -rgc > /dev/null 2>&1' EXIT
fi
echo "$LOCKED" > "$OUT/clocks.txt"
nvidia-smi -i "$WANT" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active,utilization.gpu \
  --format=csv,nounits -lms 100 > "$OUT/smi_trace.csv" &
SMI=$!
finish() { kill $SMI 2> /dev/null; cat "$OUT/summary.txt"; exit "$1"; }
echo "clocks: $LOCKED" > "$OUT/summary.txt"
./pouw_hash_bench check > "$OUT/check_device.jsonl" 2>> "$OUT/summary.txt"; RC=$?
[ $RC = 0 ] || { echo "check: pouw_hash_bench exited $RC" >> "$OUT/summary.txt"; finish $RC; }
python3 check.py --against host_check.jsonl "$OUT/check_device.jsonl" > "$OUT/check_summary.jsonl"; RC=$?
cat "$OUT/check_summary.jsonl" >> "$OUT/summary.txt"
[ $RC = 0 ] || { echo "GATE FAILED: the device's vectors; nothing timed" >> "$OUT/summary.txt"; finish 4; }
[ "$MODE" = check ] && finish 0
./pouw_hash_bench rates "${MS:-100}" > "$OUT/rates.jsonl" 2>> "$OUT/summary.txt"; RC=$?
[ $RC = 0 ] || { echo "rates: exited $RC" >> "$OUT/summary.txt"; finish $RC; }
./pouw_hash_bench epi "${MS:-100}" > "$OUT/epi.jsonl" 2>> "$OUT/summary.txt"; RC=$?
python3 - "$OUT" "$LOCKED" >> "$OUT/summary.txt" <<'EOF'
import json, sys
out, locked = sys.argv[1], sys.argv[2]
for f in ("rates", "epi"):
    try:
        rows = [json.loads(l) for l in open(f"{out}/{f}.jsonl") if l.startswith("{")]
    except FileNotFoundError:
        continue
    for r in rows:
        if r["group"] in ("device", "epi_gate"):
            print(json.dumps(r))
            continue
        extra = f" warps/SM {r['warps_per_sm']}" + (f" sweep {r['sweep_warps']}" if "sweep_warps" in r else "")
        unit = "ops" if r["group"] == "op" else "B"
        mac = f" MACs/SM/clk {r['macs_per_sm_clk']:.0f}" if r["group"] == "epi" else ""
        print(f"{r['group']:8} {r['name'][:60]:60} {r['per_sm_clk']:9.3f} {unit}/SM/clk {r['gb_s']:10.1f} GB/s"
              f" {r['cyc_per_iter']:10.1f} cyc/iter{mac} regs {r['regs']}{extra} f {r['f_eff_ghz']:.3f} GHz ({locked})")
EOF
[ $RC = 4 ] && echo "GATE FAILED: the epilogue's digests; nothing timed in epi" >> "$OUT/summary.txt"
finish $RC
