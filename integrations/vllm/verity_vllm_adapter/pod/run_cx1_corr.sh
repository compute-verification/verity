#!/bin/bash
# [COMP, CX1] correspondence of the compiled TP=2 commit capture against ServeTP2 through the port's compiled partition
# (verity_vllm.correspondence.tp2 --port profiles/b1-tp2-compiled), the TP control's sampling exactly (tp2-profile.md §3): request
# 917b9faa (LP=512, 16 outputs) at positions {0,1,2,63,64,127,255,511,512,520,526}, layers {0,1,13,27}, full GEMMs at (64,13) and
# (512,27); request 147eed15 (LP=32) at {0,1,31,32}, layers {0,27}. Then the manifest scorecard beside the two controls.
# usage: run_cx1_corr.sh <TAG> [--only 917b9faa|147eed15]
set -u
TAG=$1; shift
ONLY=""
while [ $# -gt 0 ]; do case "$1" in --only) ONLY=$2; shift;; esac; shift; done
REPO=/workspace/veritor
R=/workspace/results/comp/$TAG
# the IR (veritor.core) needs Python 3.12; the cos/sin table is vLLM's GPU derivation, computed once with the vLLM venv (held-out-cell.md §5)
PY=${PY:-/workspace/py312/bin/python}
VPY=${VPY:-/workspace/venv/bin/python}
CK=$(ls -d /workspace/hf/hub/models--Qwen--Qwen2.5-1.5B/snapshots/*/model.safetensors | head -1)
PORT=$REPO/vllm-poc/profiles/b1-tp2-compiled
mkdir -p $R/correspondence $R/analysis
cd $REPO && export PYTHONPATH=.
log() { echo "[$(date -u +%H:%M:%SZ)] cx1-corr/$TAG: $*" | tee -a $R/pipeline.log; }
CS=$R/correspondence/cos_sin_cuda.npy
[ -f $CS ] || $VPY - <<EOF
import torch, numpy as np
D=128; theta=1_000_000.0; max_pos=131072
inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float, device="cuda") / D))
freqs = torch.einsum("i,j -> ij", torch.arange(max_pos, dtype=torch.float, device="cuda"), inv_freq)
np.save("$CS", torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16).cpu().contiguous().view(torch.int16).numpy().view(np.uint16))
EOF
run_one() {
  local rid=$1; shift
  log "correspondence $rid start"
  $PY -m verity_vllm.correspondence.tp2 --port $PORT --commit $R/commit --checkpoint $CK --request $rid "$@" --cos-sin-npy $CS \
      --out $R/correspondence/corr-tp2-compiled-$rid.json > $R/correspondence/corr-tp2-compiled-$rid.log 2>&1 \
      || { log "correspondence $rid FAILED: $(grep -E 'Error|error' $R/correspondence/corr-tp2-compiled-$rid.log | tail -1 | cut -c1-300)"; return 1; }
  log "correspondence $rid done: $(grep -E '^\[corr-tp2\] summary|summary' $R/correspondence/corr-tp2-compiled-$rid.log | tail -1 | cut -c1-400)"
}
if [ -z "$ONLY" ] || [ "$ONLY" = 917b9faa ]; then
  run_one 917b9faa-f356-5233-b140-e757b2c5de99 --positions 0 1 2 63 64 127 255 511 512 520 526 --layers 0 1 13 27 --full 64,13 512,27
fi
if [ -z "$ONLY" ] || [ "$ONLY" = 147eed15 ]; then
  run_one 147eed15-8c24-5c92-887d-7a2a7abb795c --positions 0 1 31 32 --layers 0 27
fi
CTRL_C=$REPO/fixtures/results/B1-0fc5b2ce24f2-20260908T0642Z/control-ampere-compiled/control-ampere-147eed15.json
CTRL_T=$REPO/fixtures/results/B1-7f26cdcedcd0-20260908T0229Z/correspondence
FILES=$(ls $R/correspondence/corr-tp2-compiled-*.json 2>/dev/null | tr '\n' ',' | sed 's/,$//')
[ -n "$FILES" ] && $PY $PORT/scorecard.py --cell tp2-compiled=$FILES --cell compiled=$CTRL_C --cell tp2=$CTRL_T/corr-tp2-917b9faa.json,$CTRL_T/corr-tp2-147eed15.json \
    --out $R/analysis/scorecard.json --md $R/analysis/scorecard.md | tail -40 | tee -a $R/pipeline.log
log "CORR_DONE"
