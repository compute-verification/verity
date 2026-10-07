#!/usr/bin/env bash
# Pearl-C on the pod, from the shipped tree (build.sh): the H100 SXM name check, env.txt, the tree against MANIFEST.sha256,
# an nvidia-smi trace, then
#   bash run.sh          run.py (the check at the fixture's shape, then the timing at m = n = k), and vendor.py at run.py's shapes;
#   bash run.sh bench    bench.py (the check, then each shape of bench.json: its sampled tiles against the CPU twin, then its
#                        timing), and vendor.py at bench.json's shapes (real rows);
# vendor.py (cuBLASLt FP8 through torch._scaled_mm) only when the image has torch.  Nothing is fetched or built here.
# Exit: 3 not an H100 SXM; 5 the tree does not match its manifest; 4 a check or a sampled tile failed (rows still written);
# 124 the runner timed out; else the runner's.  Outputs under $RESEARCH_RUN_DIR/out (or ./out): gpu_name.txt, env.txt,
# inputs.check, smi_trace.csv, pearl_c.jsonl or bench.jsonl, vendor.jsonl, summary.txt.
set -uo pipefail
cd "$(dirname "$0")"
MODE=${1:-run}
OUT=${RESEARCH_RUN_DIR:-$PWD}/out; mkdir -p "$OUT"
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "$NAME" > "$OUT/gpu_name.txt"
if [ "$NAME" != "NVIDIA H100 80GB HBM3" ]; then echo "REFUSED: GPU is '$NAME', not an H100 SXM (NVIDIA H100 80GB HBM3)"; exit 3; fi
{ nvidia-smi; nvidia-smi -q -d CLOCK,PERFORMANCE,POWER; python3 --version; lscpu | grep -E "Model name|^CPU\(s\)"; grep PRETTY /etc/os-release; } > "$OUT/env.txt" 2>&1
if ! sha256sum -c MANIFEST.sha256 > "$OUT/inputs.check" 2>&1; then echo "REFUSED: the tree does not match MANIFEST.sha256"; exit 5; fi
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active,utilization.gpu \
  --format=csv,nounits -lms 100 > "$OUT/smi_trace.csv" &
SMI=$!
if [ "$MODE" = bench ]; then
  timeout "${RUN_TIMEOUT:-1200}" python3 bench.py bench.json check/check.json > "$OUT/bench.jsonl" 2> "$OUT/summary.txt"; RC=$?
  VENDOR_SHAPES=$(python3 -c 'import json; print(" ".join("%dx%dx%d" % (s["m_real"], s["n"], s["k"]) for s in json.load(open("bench.json"))["shapes"]))')
else
  timeout "${RUN_TIMEOUT:-150}" python3 run.py check/check.json > "$OUT/pearl_c.jsonl" 2> "$OUT/summary.txt"; RC=$?
  VENDOR_SHAPES=${SHAPES:-8192 16384}
fi
if python3 -c "import torch" 2> /dev/null; then
  SHAPES="$VENDOR_SHAPES" timeout "${VENDOR_TIMEOUT:-300}" python3 vendor.py > "$OUT/vendor.jsonl" 2>> "$OUT/summary.txt"
else
  echo "vendor: no torch on the image, cuBLASLt not timed" >> "$OUT/summary.txt"
fi
kill $SMI 2> /dev/null
cat "$OUT/summary.txt"
exit $RC
