#!/usr/bin/env bash
# The skeleton at coverage f = 1 on one H200: run.py at the frozen shape with `e2e_h200.percall:$H1U`, every
# served linear call's J10-R units through kernels' sm_90a build H1U (kraqwxsc, or kraq), receipts and z committed under
# the step's salt key, sampled units held to the exact fallback. cwd = the source tree's root; the builds are sent, as
# the SHA-512 engine's check held them:
#
#   research run --on H200_NODE --project verity --campaign e2e-demo --source . --cwd source --timeout 60m \
#     --send RUN/out/lib/sm_90a/h1u_kraqwxsc.so --send RUN/inputs/j10r_ref.so [--send PIN/sass_pins.json] -- \
#     bash examples/h200/percall_job.sh
#
# Before the lease: uv sync, attn_ref's cubin, the SHA-512 engine's build, and the SASS gate on the h1u build under this
# tree's pins and (sass_pins.json sent) the sent pins. Under one GPU lease (gpu-lease, shared with the node's other runs):
# run.py into $OUT/f1. Env: H1U (kraqwxsc; the sent h1u_$H1U.so), BATCH (4096), STEPS (3), MB (1024), SAMPLES (2,
# PERCALL_SAMPLES), LEASE_MIN (40).
set -euo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
IN=${RESEARCH_RUN_DIR:-.}/inputs
E=examples/h200
H1U=${H1U:-kraqwxsc}
mkdir -p "$OUT"
uv sync -q --all-packages --extra torch
for c in /usr/local/cuda-13.0 /usr/local/cuda; do [ -x "$c/bin/nvcc" ] && export CUDA=$c && break; done
export PATH="$CUDA/bin:$PATH"
CUBIN_ONLY=1 bash benchmarks/torch_gpu/attn_ref/build.sh "$OUT/attn_ship" 2>&1 | tail -2
export ATTN_REF_CUBIN="$OUT/attn_ship/attn_ref.sm_90a.cubin"
uv run --no-sync python -c "import sys; sys.path.insert(0, 'benchmarks/torch_gpu/commit_slots'); import slots_gpu; print(slots_gpu.build('9.0'))"
sha256sum "$IN/h1u_$H1U.so" "$IN/j10r_ref.so" | tee "$OUT/libs.sha256"
POUW_CUDA_BIN=$CUDA/bin POUW_SASS_GATE_CACHE=none uv run --no-sync python benchmarks/pouw/harness/sass_gate.py "$IN/h1u_$H1U.so" \
  --json "$OUT/sass-gate-main.json" > "$OUT/sass-gate-main.txt" 2>&1 || true
tail -2 "$OUT/sass-gate-main.txt"
if [ -f "$IN/sass_pins.json" ]; then
  POUW_CUDA_BIN=$CUDA/bin uv run --no-sync python - "$IN/h1u_$H1U.so" "$IN/sass_pins.json" "$OUT/sass-gate-pin.json" <<'EOF' || true
import json, sys
from pathlib import Path
sys.path.insert(0, "benchmarks/pouw/harness")
import sass_gate as G
rec = G.scan(sys.argv[1], G.find_tools(), None, Path(sys.argv[2]))
Path(sys.argv[3]).write_text(json.dumps(rec, indent=1))
print("sass gate under the sent pins:", [{k: c.get(k) for k in ("toolkit", "unpinned", "flagged", "offending_n")}
                                         for c in rec["cubins"] if c.get("functions")])
EOF
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PERCALL_H1U=$IN/h1u_$H1U.so PERCALL_REF=$IN/j10r_ref.so
export PERCALL_SAMPLES=${SAMPLES:-2}
export ARGS="--out $OUT/f1 --shape h200 --seed demo --device cuda --batch ${BATCH:-4096} --steps ${STEPS:-3} --prompt-bytes 8
  --micro-batch ${MB:-1024} --workers $(nproc) --open 1 --draw 1 --subcircuit e2e_h200.percall:$H1U --run-id e2e-h200-percall-f1"
# shellcheck disable=SC2016
gpu-lease 1 --wait --mem-gb 0 --max-min "${LEASE_MIN:-40}" -- bash -c '
  nvidia-smi --query-gpu=index,name,clocks.sm,memory.used,memory.total --format=csv -i "$GPU_LEASE_UUID"
  uv run --no-sync python '"$E"'/run.py $ARGS'
