#!/usr/bin/env bash
# A CPU node readied for a row's v2 CPU jobs (trace_compare.sh, build_record.sh, match_cpu.sh) under W: the venv (pod_bootstrap.sh
# --cpu, no checkpoint), the H200 hosts' libcuda.so.1 at W/cuda-driver where the node has none (CUDA_DRIVER_ART, a cuda-driver/v1
# art holding libcuda.so.<version>: the pinned vLLM imports it on the meta device and opens no device), and the row's checkpoint
# metadata at SNAP (model.sh stage_meta: the pin's non-shard files from the hub, each checkpoints.json's).
#   research run --on vy-nebius-cpu-3 --mem-gb 16 --timeout 1h --cwd source --project verity -- env ROW=... CUDA_DRIVER_ART=art:... bash integrations/vllm/manifests/glm_v2_tree/jobs/stage_cpu.sh
# env: ROW, [W=/workspace/cp/circuits], [CUDA_DRIVER_ART].
set -uo pipefail
: "${ROW:?ROW}"
TREE=$PWD
source "$TREE/integrations/vllm/manifests/glm_v2_tree/jobs/model.sh"
W=${W:-/workspace/cp/circuits}
mkdir -p "$W" || exit 3
(cd integrations/vllm && WORK=$W VENV=$W/venv312 HF_HOME=$W/hf bash verity_vllm/ops/pod_bootstrap.sh --cpu --cases "" --out "$OUT/bootstrap") \
  >"$OUT/bootstrap.log" 2>&1 || { tail -30 "$OUT/bootstrap.log"; exit 5; }
tail -1 "$OUT/bootstrap.log"
PY=$W/venv312/bin/python
if ldconfig -p | grep -q 'libcuda\.so\.1 '; then
  echo "libcuda.so.1: the node's own"
elif [ -e "$W/cuda-driver/libcuda.so.1" ]; then
  echo "libcuda.so.1: $W/cuda-driver ($(readlink "$W/cuda-driver/libcuda.so.1"))"
else
  : "${CUDA_DRIVER_ART:?no libcuda.so.1 on the node: CUDA_DRIVER_ART}"
  C=$(dirname "$(dirname "$RESEARCH_RUN_DIR")")/requests/$RESEARCH_RUN_ID/custody
  [ -f "$C/cred.json" ] || { echo "no custody key at $C"; exit 4; }
  export PYTHONPATH=$TREE:$TREE/tools/research/src:$TREE/integrations/vllm RESEARCH_STORE_CONFIG=$C/store.toml
  eval "$($PY -m verity_vllm.pipeline.cli epoch custody-env "$C")"
  T=$(mktemp -d "$W/.cuda-driver.XXXXXX")
  "$PY" -m research data fetch "$CUDA_DRIVER_ART" --to "$T/x" >/dev/null || { echo "research data fetch $CUDA_DRIVER_ART failed"; exit 6; }
  lib=$(cd "$T/x" && ls libcuda.so.[0-9]* 2>/dev/null | head -1)
  [ -n "$lib" ] || { echo "$CUDA_DRIVER_ART holds no libcuda.so.<version>"; exit 6; }
  mv "$T/x" "$W/cuda-driver" && ln -s "$lib" "$W/cuda-driver/libcuda.so.1" && rmdir "$T" || exit 6
  echo "libcuda.so.1: $W/cuda-driver ($lib, $CUDA_DRIVER_ART)"
fi
stage_meta "$PY" || { echo "no $WHAT metadata at $SNAP"; exit 3; }
ls -la "$SNAP" | tee "$OUT/meta.txt"
