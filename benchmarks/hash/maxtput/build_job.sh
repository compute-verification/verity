#!/usr/bin/env bash
# A Build of record for a model that has none (Llama-3.1-8B on the RTX PRO 6000): the served match alone, as a Build
# job's served_job.sh (MODE=match+commit) runs it, without the commit stage. Run from the source tree's root
# on one GPU, with the match workload in inputs/match.json (a llama_match.json). Writes out/build.json (a pin
# names it as this run's art, path out/build.json), match.json and the compile records.
#
#   RESEARCH_RUN_DIR=DIR VERITY_BENCH_VENV=VENV CASE=LLAMA31_8B CKPT_REPO=unsloth--Llama-3.1-8B CKPT_REV=REV \
#     bash benchmarks/hash/maxtput/build_job.sh      (on one RTX PRO 6000)
#
# Env: CASE (LLAMA31_8B), CKPT_REPO (unsloth--Llama-3.1-8B), CKPT_REV, VERITY_BENCH_VENV, NATIVE_COLLECT_BUILD, HF_HOME, MATCH_EXTRA.
set -uo pipefail
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
ROOT=$PWD
I="$RESEARCH_RUN_DIR/inputs"; O="$RESEARCH_RUN_DIR/out"; mkdir -p "$O" "$S/bin"
export HF_HOME=${HF_HOME:-/workspace/hf}
export PYTHONPATH=$ROOT/integrations/vllm:$ROOT/tools/config:$ROOT/tools/research/src:$ROOT HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TMPDIR=$S OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export VERITY_RETAIN=host VERITY_WINDOW_MB=256 VERITY_WINDOW_SLOTS=8
VENV=${VERITY_BENCH_VENV:?}
printf '#!/bin/sh\nexec %s/bin/python "$@"\n' "$VENV" > "$S/bin/python"; chmod +x "$S/bin/python"
printf '#!/bin/sh\nexec %s -u -m verity_vllm.pipeline.cli "$@"\n' "$S/bin/python" > "$S/bin/verity-vllm"; chmod +x "$S/bin/verity-vllm"
export PATH=$S/bin:$VENV/bin:$CUDA_HOME/bin:$PATH
echo "gpu ${CUDA_VISIBLE_DEVICES:-?} ${GPU_LEASE_UUID:-}"
ls -d "$HF_HOME/hub/models--${CKPT_REPO:?}/snapshots/${CKPT_REV:?}" || exit 8
verity-vllm served-match --case "${CASE:?}" --workload "$I/match.json" --out "$S/m" ${MATCH_EXTRA:-} 2>&1 | grep --line-buffered -vE "^\s*$"
rcm=${PIPESTATUS[0]}
for f in match build facts launch_configs routes moe; do test -f "$S/m/$f.json" && cp "$S/m/$f.json" "$O/"; done
(cd "$S/m" && find inductor \( -name '*.best_config' -o -name '*.ptx' \) -print0 | xargs -0 -r tar czf "$O/triton_ptx.tgz") 2>/dev/null
echo "MATCH rc=$rcm"
[ -f "$O/build.json" ] && python -c "import json,sys; a=json.load(open(sys.argv[1])).get('artifact') or {}; print('artifact', a.get('digest'), len(a.get('winners') or {}), 'winners')" "$O/build.json"
exit $rcm
