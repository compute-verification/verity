#!/usr/bin/env bash
# The bandwidth audit's GPU tagger on one RTX PRO 6000 of vy-nebius-2, as one Attempt (cwd = the source tree):
#
#   research run --on vy-nebius-2 --project verity --source . --cwd source --campaign resource-accounting -- \
#     bash benchmarks/bandwidth/run.sh
#   (a timed run: `--no-sampler` on research run, and TIMED=1 here, which takes the lease with --timed)
#
# Outside the lease: build `bwtag` and `libbwtag.so` (refused on any register spill), and compute with the core reference
# the expected roots over the full store and over the pinned vectors' 37 blocks, and the store's PoUS leaves.  Under
# `gpu-lease 1 --wait --max-min 20`: `bwtag`'s gates (SHA-512's and keyed BLAKE3's known answers; every tag's roots
# against the reference with the output buffers poisoned; a control without the tag kernel that must not match), then its
# timing; a failed gate times nothing (exit 3).  Then `gpu.py windows`: honest windows through `audit.Verifier` at
# DEADLINE_NS (default the largest it allows), and windows that tag only part of the store.  After the lease:
# `summarize.py`.
# Env: BLOCKS (default 524,288, the certificate's store), SEED (hex, default 00..1f), REPS (default 100), WINDOWS (20),
# DEADLINE_NS, BUSY_READS (the windows' `--busy-reads`, default 0), BUSY_COMPARE (if set, ro-blake3/v1's honest windows
# again at that `--busy-reads`, into windows_busy.json), WORKERS (the reference's processes, default 32), TIMED.
# Outputs under $RESEARCH_RUN_DIR/out: build.txt, expect.json, gpu.txt, env.txt, smi.csv, device.json, validation.json,
# timing.json, windows.json, summary.json.
set -euo pipefail
cd "$(dirname "$0")"
export O=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$O"
export PYTHONPATH=$(cd ../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-python3}

if [ "${1:-}" = leased ]; then
  S=$2
  UUID=${GPU_LEASE_UUID:?run under gpu-lease}
  NAME=$(nvidia-smi -i "$UUID" --query-gpu=name --format=csv,noheader | head -1)
  nvidia-smi -i "$UUID" --query-gpu=index,uuid,name,pci.bus_id,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu \
    --format=csv > "$O/gpu.txt"
  case "$NAME" in *"RTX PRO 6000"*) ;; *) echo "REFUSED: GPU $UUID is '$NAME', not an RTX PRO 6000"; exit 3 ;; esac
  { nvidia-smi -i "$UUID" -q -d CLOCK,PERFORMANCE,POWER; lscpu | grep -E "Model name|^CPU\(s\)|NUMA"; } > "$O/env.txt" 2>&1
  nvidia-smi -i "$UUID" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,utilization.memory \
    --format=csv,nounits -lms 100 > "$O/smi.csv" &
  SMI=$!
  trap 'kill $SMI 2> /dev/null || true' EXIT
  mapfile -t A < <("$PY" - "$O/expect.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
f, s = d["full"], d["small"]
print(f["blocks"], f["seed"], s["seed"], sep="\n")
for short, name in (("gf31", "gf31-poly/v1"), ("ro", "ro-sha512/v1"), ("b3", "ro-blake3/v1")):
    print(f"--expect-{short}", f["roots"][name], f"--small-expect-{short}", s["roots"][name], sep="\n")
EOF
  )
  rc=0
  "$S/bwtag" --out "$O" --blocks "${A[0]}" --reps "${REPS:-100}" --seed "${A[1]}" --small-seed "${A[2]}" \
    "${A[@]:3}" || rc=$?
  [ $rc = 0 ] || { echo "bwtag exited $rc: nothing more runs"; exit $rc; }
  W=(gpu.py windows --lib "$S/libbwtag.so" --leaves "$S/store_leaves.bin" --blocks "${A[0]}" --windows "${WINDOWS:-20}"
     ${DEADLINE_NS:+--deadline-ns "$DEADLINE_NS"})
  "$PY" "${W[@]}" --busy-reads "${BUSY_READS:-0}" --out "$O/windows.json" > "$O/windows.jsonl"
  if [ -n "${BUSY_COMPARE:-}" ]; then
    "$PY" "${W[@]}" --tags ro-blake3/v1 --skip --busy-reads "$BUSY_COMPARE" --out "$O/windows_busy.json" > /dev/null
  fi
  exit 0
fi

NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
FLAGS=(-O3 -std=c++17 -arch=sm_120 -Xptxas -v)
{ "$NVCC" --version; echo "flags: ${FLAGS[*]}"; } > "$O/build.txt"
"$NVCC" "${FLAGS[@]}" -DBW_MAIN tagger.cu -o "$S/bwtag" >> "$O/build.txt" 2>&1 &
B1=$!
"$NVCC" "${FLAGS[@]}" -Xcompiler -fPIC -shared tagger.cu -o "$S/libbwtag.so" >> "$O/build.txt" 2>&1 &
B2=$!
wait $B1 && wait $B2 || { tail -30 "$O/build.txt"; exit 3; }
if grep -E "[1-9][0-9]* bytes spill" "$O/build.txt"; then echo "REFUSED: a kernel spills"; exit 3; fi
sha256sum tagger.cu "$S/bwtag" "$S/libbwtag.so" >> "$O/build.txt"

"$PY" reference.py --blocks "${BLOCKS:-524288}" --seed "${SEED:-000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f}" \
  --workers "${WORKERS:-32}" --out "$O/expect.json" --leaves "$S/store_leaves.bin" > /dev/null

rc=0
GPU_LEASE_WHO=ra-bandwidth-protocol gpu-lease 1 --wait ${TIMED:+--timed} --max-min 20 -- bash "$PWD/run.sh" leased "$S" || rc=$?
"$PY" summarize.py "$O" > "$O/summary.json" || true
cat "$O/summary.json"
exit $rc
