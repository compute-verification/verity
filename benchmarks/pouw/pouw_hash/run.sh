#!/usr/bin/env bash
# pouw_hash on one RTX PRO 6000, from the shipped tree (build.sh): the tree against MANIFEST.sha256, the device's name and UUID,
# the clock lock on this GPU only, an nvidia-smi trace of this GPU, then the gates, then the timing:
#   bash run.sh          the device's vectors against host_check.jsonl (check.py --against), then `rates`, `epi` and `acommit`
#                        (the last two gate their own outputs against the host before they time anything)
#   bash run.sh check    the gates only
# Env: POUW_EXPECT_UUID (required): the GPU's UUID from internal/pouw/rtx-pro/server.md; "pod" on a one-GPU pod (refused
# if more than one GPU is visible); or "lease" under `gpu-lease 1` (its GPU_LEASE_UUID, or else CUDA_VISIBLE_DEVICES one index
# in PCI bus order, whose UUID the bench then requires of its CUDA device); LOCK_MHZ: lock this GPU's SM clock there
# (nvidia-smi -i UUID -lgc, reset on exit); else CLOCK_LABEL labels the rows (default locked-2100 under "lease", node 2's
# node-wide lock, else unlabelled), and every row records its clocks, the summary flagging a row more than 2% off a locked-N
# label; MS: each row's target milliseconds (default 100); PARTS: which timed modes run (default "rates epi acommit").
# Exit: 3 the wrong device; 5 the tree does not match its manifest; 4 a gate failed (nothing timed); else the bench's.
# Outputs under $RESEARCH_RUN_DIR/out (or ./out): device.txt, env.txt, inputs.check, clocks.txt, smi_trace.csv,
# check_device.jsonl, check_summary.jsonl, rates.jsonl, epi.jsonl, acommit.jsonl, summary.txt.
set -uo pipefail
cd "$(dirname "$0")"
MODE=${1:-run}
OUT=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$OUT"
if ! sha256sum -c MANIFEST.sha256 > "$OUT/inputs.check" 2>&1; then echo "REFUSED: the tree does not match MANIFEST.sha256"; exit 5; fi
chmod +x pouw_hash_bench   # a shipped repo (ship.sh) checks every file out 100644
WANT=${POUW_EXPECT_UUID:-}
DEFAULT_LABEL=unlabelled
if [ "$WANT" = pod ]; then
  [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" = 1 ] || { echo "REFUSED: POUW_EXPECT_UUID=pod with more than one GPU"; exit 3; }
  WANT=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
elif [ "$WANT" = lease ] && [ -n "${GPU_LEASE_UUID:-}" ]; then
  WANT=$GPU_LEASE_UUID
  DEFAULT_LABEL=locked-2100
elif [ "$WANT" = lease ]; then
  case "${CUDA_VISIBLE_DEVICES:-}" in ''|*[!0-9]*) echo "REFUSED: POUW_EXPECT_UUID=lease needs CUDA_VISIBLE_DEVICES to be one index"; exit 3 ;; esac
  [ "${CUDA_DEVICE_ORDER:-}" = PCI_BUS_ID ] || { echo "REFUSED: POUW_EXPECT_UUID=lease needs CUDA_DEVICE_ORDER=PCI_BUS_ID"; exit 3; }
  WANT=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid --format=csv,noheader | head -1)
  DEFAULT_LABEL=locked-2100
fi
[ -n "$WANT" ] || { echo "REFUSED: POUW_EXPECT_UUID is not set"; exit 3; }
export POUW_EXPECT_UUID=$WANT
NAME=$(nvidia-smi -i "$WANT" --query-gpu=name --format=csv,noheader 2> /dev/null | head -1)
echo "$NAME $WANT CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" > "$OUT/device.txt"
case "$NAME" in *"RTX PRO 6000"*) ;; *) echo "REFUSED: GPU $WANT is '$NAME', not an RTX PRO 6000"; exit 3 ;; esac
{ nvidia-smi -i "$WANT"; nvidia-smi -i "$WANT" -q -d CLOCK,PERFORMANCE,POWER; cat toolchain.txt source.txt; lscpu | grep -E "Model name|^CPU\(s\)"; grep PRETTY /etc/os-release; } > "$OUT/env.txt" 2>&1
LOCKED=${CLOCK_LABEL:-$DEFAULT_LABEL}
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
PARTS=" ${PARTS:-rates epi acommit} "
RC=0 RCA=0
if [[ $PARTS == *" rates "* ]]; then
  ./pouw_hash_bench rates "${MS:-100}" > "$OUT/rates.jsonl" 2>> "$OUT/summary.txt"; RC=$?
  [ $RC = 0 ] || { echo "rates: exited $RC" >> "$OUT/summary.txt"; finish $RC; }
fi
[[ $PARTS == *" epi "* ]] && { ./pouw_hash_bench epi "${MS:-100}" > "$OUT/epi.jsonl" 2>> "$OUT/summary.txt"; RC=$?; }
[[ $PARTS == *" acommit "* ]] && { ./pouw_hash_bench acommit > "$OUT/acommit.jsonl" 2>> "$OUT/summary.txt"; RCA=$?; }
python3 - "$OUT" "$LOCKED" >> "$OUT/summary.txt" <<'EOF'
import json, re, sys
out, locked = sys.argv[1], sys.argv[2]
lock = re.match(r"locked[- ](\d+)", locked)
lock_ghz = int(lock.group(1)) / 1e3 if lock else None
# the W1 unit is one dense FP8 mma.sync MAC: a rate row's price per byte is the mma row's MACs per SM per clock over its bytes
w1 = None
for f in ("rates", "epi", "acommit"):
    try:
        rows = [json.loads(l) for l in open(f"{out}/{f}.jsonl") if l.startswith("{")]
    except FileNotFoundError:
        continue
    w1 = w1 or next((r["per_sm_clk"] for r in rows if r["group"] == "mma" and "skipped" not in r), None)
    for r in rows:
        if r["group"] in ("device", "epi_gate", "acommit_gate") or "skipped" in r:
            print(json.dumps(r))
            continue
        if r["group"] == "acommit":
            # a pass over m rows: its price in W1 is the whole card's MACs over the pass (ms x SMs x the SM clock x w1)
            ghz = r["clk_before"]["sm_mhz"] / 1e3
            mac = f" W1/B {r['ms'] * 1e6 * ghz * r['sms'] * w1 / r['bytes']:.1f}" if w1 else ""
            print(f"acommit  {r['name']:28} {r['ms'] * 1e3:10.2f} us (min {r['ms_min'] * 1e3:.2f}, nop {r['nop_ms'] * 1e3:.2f})"
                  f" {r['gb_s']:9.1f} GB/s{mac} regs {r['regs']} grid {r['blocks']}x{r['threads']} {ghz:.3f} GHz ({locked})")
            if lock_ghz and abs(ghz / lock_ghz - 1) > 0.02:
                print(f"CLOCK MISMATCH: {r['name']} read {ghz:.3f} GHz, labelled {locked}")
            continue
        extra = f" warps/SM {r['warps_per_sm']}" + (f" sweep {r['sweep_warps']}" if "sweep_warps" in r else "")
        unit = {"op": "ops", "mma": "MACs"}.get(r["group"], "B")
        mac = f" MACs/SM/clk {r['macs_per_sm_clk']:.0f}" if r["group"] == "epi" else ""
        if w1 and r["group"] in ("rate", "sweep"):
            mac = f" W1/B {w1 / r['per_sm_clk']:.1f}"
        elif r["group"] == "latency":
            mac = f" {r['cyc_per_iter'] / r['f_eff_ghz'] / 1e3:.2f} us/iter"
        print(f"{r['group']:8} {r['name'][:60]:60} {r['per_sm_clk']:9.3f} {unit}/SM/clk {r['gb_s']:10.1f} GB/s"
              f" {r['cyc_per_iter']:10.1f} cyc/iter{mac} regs {r['regs']}{extra} f {r['f_eff_ghz']:.3f} GHz ({locked})")
        if lock_ghz and abs(r["f_eff_ghz"] / lock_ghz - 1) > 0.02:
            print(f"CLOCK MISMATCH: {r['name']} ran at {r['f_eff_ghz']:.3f} GHz, labelled {locked}")
EOF
[ $RC = 4 ] && echo "GATE FAILED: the epilogue's digests; nothing timed in epi" >> "$OUT/summary.txt"
[ $RCA = 4 ] && echo "GATE FAILED: A's commitment forms; nothing timed in acommit" >> "$OUT/summary.txt"
[ $RC = 0 ] && RC=$RCA
finish $RC
