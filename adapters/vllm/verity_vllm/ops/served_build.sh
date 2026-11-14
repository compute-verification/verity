#!/usr/bin/env bash
# A Build of record for a model that has none (Llama-3.1-8B on the RTX PRO 6000): the vLLM integration's served match alone
# (`verity-vllm served-match`, adapters/vllm/verity_vllm/pipeline/served_match.py), with no Commit stage after it. vLLM serves
# the match workload with every value of every engine step committed; the Build is derived from the serving process's own
# compile artifact, and its Program is checked word for word against the committed values. Its build.json is what
# adapters/vllm/verity_vllm/ops/served_commit.sh takes as the required manifest and benchmarks/served_commit/served_arm.sh as
# PIN_BUILD.
#
# Run from the source tree's root, on one RTX PRO 6000, with the match workload in $RESEARCH_RUN_DIR/inputs/match.json:
#
#   RESEARCH_RUN_DIR=DIR VERITY_BENCH_VENV=VENV CASE=LLAMA31_8B CKPT_REPO=unsloth--Llama-3.1-8B CKPT_REV=REV \
#     bash adapters/vllm/verity_vllm/ops/served_build.sh
#
# Env:
#   CASE                  the checkpoint role (verity_vllm.config.CHECKPOINTS), LLAMA31_8B
#   CKPT_REPO, CKPT_REV   the checkpoint's directory and revision under $HF_HOME/hub, read offline
#   VERITY_BENCH_VENV     the Python environment with CUDA torch and vLLM
#   HF_HOME               the Hugging Face cache, default /workspace/hf
#   NATIVE_COLLECT_BUILD  the parent directory of the native collector's builds (the integration's settings)
#   MATCH_EXTRA           further served-match arguments, such as --build-of-record BUILD.json or --max-num-seqs N
#
# Writes into $RESEARCH_RUN_DIR/out: build.json (the Build, its compile artifact pinned), match.json (the Match's verdict),
# facts.json, launch_configs.json, routes.json and, with fused-MoE layers, moe.json, as served-match writes them, and
# triton_ptx.tgz (the Inductor directory's autotune configs and PTX).
#
# Exit: served-match's: 0 when the Match passes, 1 when it fails, 3 when the Build refuses the artifact, the device or the
# served switches; 8 when the checkpoint's snapshot is absent. What served-match runs, the served Build's derivation and its
# check, is tested in adapters/vllm/tests/acquire/ and adapters/vllm/tests/check/.
set -uo pipefail
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT
ROOT=$PWD
I="$RESEARCH_RUN_DIR/inputs"; O="$RESEARCH_RUN_DIR/out"; mkdir -p "$O" "$S/bin"
export HF_HOME=${HF_HOME:-/workspace/hf}
export PYTHONPATH=$ROOT/adapters/vllm:$ROOT/tools/config:$ROOT/tools/research/src:$ROOT HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
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
