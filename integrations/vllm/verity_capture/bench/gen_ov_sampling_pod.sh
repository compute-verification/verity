#!/usr/bin/env bash
# [ov-sampling] pod-side driver: GPU phase of every B1 sampling configuration in sequence (the GPU is exclusive), then the CPU
# phases three at a time (32 cores, each run_config uses --parallel 8).  Run dirs live on the pod's local disk (wave-1 hit the
# volume's inode quota with the ~13k snapshot files a run writes) and are checkpointed to /vol as single tarballs after each phase.
#
#   bash verity_capture/bench/gen_ov_sampling_pod.sh gpu  [cfg ...]     # default: all nine
#   bash verity_capture/bench/gen_ov_sampling_pod.sh cpu  [cfg ...]
#   bash verity_capture/bench/gen_ov_sampling_pod.sh tar  [cfg ...]     # checkpoint run dirs to /vol/cp/ov-sampling/runs/<cfg>.tgz
set -uo pipefail
PHASE=${1:?gpu|cpu|tar}; shift
CFGS=("$@"); [ ${#CFGS[@]} -gt 0 ] || CFGS=(temp07 topk50 topp09 topk50_topp09 n2 2req_seeds noseed combo mixed)
SRC=/workspace/cp/veritor; RUNS=${RUNS:-/workspace/cp/runs/ov-sampling}; VOL=/vol/cp/ov-sampling; mkdir -p "$RUNS" "$VOL/runs" "$VOL/cards" 2>/dev/null
CKPT=/workspace/hf/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323
PYG=/workspace/venv-cu129/bin/python; PYC=/workspace/venv-py312/bin/python
log() { echo "[ov-sampling $(date -u +%FT%TZ)] $*"; }

run_cfg() {   # $1 = phase, $2 = cfg
  local ph=$1 cfg=$2
  cd "$SRC" && PYTHONPATH=. HF_HOME=/workspace/hf $PYC -m verity_capture.bench.run_config \
    --config-id "ov-sampling-$cfg" --case B1 --workload "workloads/workload_b1_sampling_$cfg.json" \
    --out "$RUNS/$cfg" --cards-dir "$RUNS/cards" --python-gpu $PYG --python-cpu $PYC --checkpoint-dir "$CKPT" \
    --phase "$ph" --parallel 8 > "$RUNS/$cfg.$ph.out" 2>&1
  log "$cfg $ph rc=$?"
}
tar_cfg() {   # single large write to the volume (many small files hit the inode quota)
  local cfg=$1
  tar czf "$VOL/runs/$cfg.tgz" -C "$RUNS" --exclude "$cfg/capture/values" "$cfg" 2>/dev/null && log "$cfg -> $VOL/runs/$cfg.tgz ($(du -h "$VOL/runs/$cfg.tgz" | cut -f1))"
}
ovs_cfg() {   # $1 = cfg: the CPU phase again under the lane's experimental profile, on a hardlinked copy of the GPU artifacts
  local cfg=$1
  local src="$RUNS/$cfg" dst="$RUNS/${cfg}_ovs"     # (a separate `local`: bash expands every word of one `local` before assigning)
  rm -rf "$dst"; mkdir -p "$dst"
  # hardlinks only for the read-only GPU artifacts (a stage that rewrote a hardlinked file in place would clobber the original)
  for d in capture control census; do [ -e "$src/$d" ] && cp -al "$src/$d" "$dst/$d"; done
  [ -e "$src/nonint" ] && cp -a "$src/nonint" "$dst/nonint"
  cp "$src/run.json" "$dst/run.json"; [ -e "$src/gpu_done.json" ] && cp "$src/gpu_done.json" "$dst/"
  cd "$SRC" && PYTHONPATH=. HF_HOME=/workspace/hf $PYC -m verity_capture.bench.run_config --run-dir "$dst" \
    --profile derived_B1_ovs --config-id "ov-sampling-$cfg-ovs" --cards-dir "$RUNS/cards" --python-gpu $PYG --python-cpu $PYC \
    --phase cpu --parallel 8 > "$RUNS/${cfg}_ovs.cpu.out" 2>&1
  log "${cfg}_ovs cpu rc=$?"
}

case $PHASE in
  gpu) for c in "${CFGS[@]}"; do log "GPU $c start"; run_cfg gpu "$c"; tar_cfg "$c"; done; log "GPU phase done" ;;
  cpu) printf '%s\n' "${CFGS[@]}" | xargs -P 3 -I{} bash -c "$(declare -f log run_cfg tar_cfg); SRC=$SRC; RUNS=$RUNS; VOL=$VOL; CKPT=$CKPT; PYG=$PYG; PYC=$PYC; log 'CPU {} start'; run_cfg cpu {}; tar_cfg {}"; log "CPU phase done" ;;
  ovs) printf '%s\n' "${CFGS[@]}" | xargs -P 3 -I{} bash -c "$(declare -f log ovs_cfg tar_cfg); SRC=$SRC; RUNS=$RUNS; VOL=$VOL; CKPT=$CKPT; PYG=$PYG; PYC=$PYC; log 'OVS {} start'; ovs_cfg {}; tar_cfg {}_ovs"; log "OVS phase done" ;;
  tar) for c in "${CFGS[@]}"; do tar_cfg "$c"; done ;;
esac
