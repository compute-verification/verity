#!/usr/bin/env bash
# Nsight Compute on one prefill attention call per (ARM, PROMPT), attn_knobs.py's Step and Arms, under `research run`
# (cwd: the root of a source tree):
#
#   research run --on MACHINE --project verity --source . --cwd source --timeout 30m \
#     --env GPU_LEASE_WHO=LANE --env VENV=DIR [--env FA3_SO="NS=PATH ..."] [--env ARMS=served] [--env PROMPTS="16x1024 16384"] \
#     -- bash benchmarks/pouw/gemm_rate/attn_ncu.sh
#
# VENV is attn_knobs.sh's (the served vLLM wheel). Under one gpu-lease (MAX_MIN, default 15): for each arm and prompt,
# ncu --set full on the launch after WARM calls of KERNEL (a base function name: FA3's is cutlass's device_kernel). $OUT gets ARM-PROMPT.ncu-rep, its details page as
# ARM-PROMPT.csv, and rc.txt.
set -uo pipefail
export O=${OUT:?}
TREE=$PWD
CUDA=${CUDA:-/usr/local/cuda-13.0}
NCU=${NCU:-$CUDA/bin/ncu}
VENV=${VENV:?}
ARMS=${ARMS:-served}
PROMPTS=${PROMPTS:-"16x1024 16384"}
KERNEL=${KERNEL:-device_kernel}
WARM=${WARM:-3}
git rev-parse HEAD > "$O/commit.txt" 2>&1
: > "$O/rc.txt"
"$NCU" --version > "$O/ncu.txt" 2>&1
echo "ncu rc $?" >> "$O/rc.txt"
grep -qv ' rc 0$' "$O/rc.txt" && { cat "$O/rc.txt" "$O/ncu.txt"; exit 1; }
cat > "$O/call.py" << 'EOF'
import sys

import torch

sys.path.insert(0, "benchmarks/pouw/gemm_rate")
import attn_knobs as ak  # noqa: E402
import attn_rate as ar  # noqa: E402

arm, prompt, warm, sos = sys.argv[1], sys.argv[2], int(sys.argv[3]), dict(x.split("=", 1) for x in sys.argv[4:])
s = ak.Step(ar.model(ar.CONFIG), ar.parse_prompt(prompt), 1)
f = ak.Arms(sos).call(arm, s)
for _ in range(warm + 1):
    f()
torch.cuda.synchronize()
EOF
export O TREE NCU KERNEL WARM ARMS PROMPTS PY=$VENV/bin/python PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/config
export FA3_SO=${FA3_SO:-}
LEASE=${LEASE-gpu-lease 1 --wait --max-min ${MAX_MIN:-15} --}
# shellcheck disable=SC2016
$LEASE bash -c '
  for a in ${ARMS//,/ }; do
    for p in $PROMPTS; do
      n=${a//:/_}-$p
      "$NCU" --set full --kernel-name "$KERNEL" --launch-skip "$WARM" --launch-count 1 --target-processes all \
        -f -o "$O/$n" "$PY" "$O/call.py" "$a" "$p" "$WARM" $FA3_SO > "$O/$n.log" 2>&1
      echo "ncu $n rc $?" >> "$O/rc.txt"
      "$NCU" --import "$O/$n.ncu-rep" --page details --csv > "$O/$n.csv" 2>> "$O/$n.log"
    done
  done'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
! grep -qv ' rc 0$' "$O/rc.txt"
