#!/usr/bin/env bash
# GEN ov-batch lane (overnight swarm 2026-09-17) — pod-side runs of the sealed driver under the FORK POINT (swarm-base-20260917), pod
# cp-ov-batch (1x RTX 4090).  Differences from gen_batch_pod.sh (wave 1): run dirs live on the container disk for speed (the network
# volume takes ~13 ms per small file: a Qwen conc4 capture is ~16 k snapshot files), and every run is made durable on /vol right after
# its GPU phase as ONE tarball (`/vol/cp/ov-batch/runs/<name>.gpu.tgz`, snapshots included, so the sealed verifier can re-run the CPU
# phase from it if the deadman takes the pod); the small files are synced to `/vol/cp/ov-batch/runs/<name>/` after the CPU phase.
# The CPU phase here is a PREVIEW (`--k 3 --seed 0`); the sealed card comes from verify_lane.sh (its own seed, default k).
#
#   cp verity_capture/bench/gen_ovbatch_pod.sh /workspace/cp/ && nohup bash /workspace/cp/gen_ovbatch_pod.sh run <names...> > /workspace/cp/ov-batch/logs/pod.log 2>&1 &
#   bash /workspace/cp/gen_ovbatch_pod.sh cpu <names...>        # CPU preview phase again over existing captures
#   bash /workspace/cp/gen_ovbatch_pod.sh sync <names...>       # small files -> /vol/cp/ov-batch/runs/<name>/
#   bash /workspace/cp/gen_ovbatch_pod.sh restore <name>        # /vol tarball -> container disk (after a pod loss)
#
# `b0_chunked_200p` is captured with `--snapshot-steps 0..14`: the run takes 15 engine steps (wave-1 §4.1) and the driver's default
# `all` = range(max(max_tokens)) = 0..11 would leave steps 12-14 without evidence (tier A: 3 uncovered attention specializations).  More
# evidence, not a changed criterion: the gates that fail (G1 step count / discarded samples, G2 step count) fail the same way.
set -uo pipefail
cd /workspace/cp/veritor
export PYTHONPATH=. HF_HOME=/workspace/hf
PY312=/workspace/venv-py312/bin/python; PY311=/workspace/venv-cu129/bin/python
OUT=/workspace/cp/ov-batch/runs; CARDS=/workspace/cp/ov-batch/cards; LOGS=/workspace/cp/ov-batch/logs; VOL=/vol/cp/ov-batch
mkdir -p "$OUT" "$CARDS" "$LOGS" "$VOL/runs" "$VOL/logs" "$VOL/cards"
SMOL=/workspace/hf/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2
QWEN=$(ls -d /workspace/hf/hub/models--Qwen--Qwen2.5-1.5B/snapshots/* 2>/dev/null | head -1)
say() { echo "[gen_ovbatch_pod] $(date -u +%FT%TZ) $*"; }
drv() { $PY312 -m verity_capture.bench.run_config --python-gpu $PY311 --python-cpu $PY312 --cards-dir "$CARDS" "$@"; }

sync_small() {  # <name>: everything but the snapshots / census traces / nonint values -> /vol (extracted, for the record)
  local n=$1; mkdir -p "$VOL/runs/$n"
  (cd "$OUT" && tar cf - --exclude="$n/capture/values" --exclude="$n/census/traces" --exclude="$n/nonint/*/values" "$n") | (cd "$VOL/runs" && tar xf -) \
    && say "synced small files of $n -> $VOL/runs/$n" || say "sync $n FAILED"
  cp "$LOGS"/$n.*.log "$VOL/logs/" 2>/dev/null; cp -r "$CARDS"/. "$VOL/cards/" 2>/dev/null
}

durable_gpu() {  # <name>: the full run dir (snapshots included) as one tarball on /vol, right after the GPU phase
  local n=$1
  (cd "$OUT" && tar czf "$VOL/runs/$n.gpu.tgz" "$n") && say "durable: $VOL/runs/$n.gpu.tgz ($(du -h "$VOL/runs/$n.gpu.tgz" | cut -f1))" || say "durable tar $n FAILED"
}

cpu_preview() {  # <name>: CPU phase (--k 3, seed 0) + the lane's torch-free step attribution, then the small files to /vol
  local n=$1
  drv --run-dir "$OUT/$n" --phase cpu --parallel 8 --k 3 --seed 0 > "$LOGS/$n.cpu.log" 2>&1; say "cpu $n rc=$?"
  $PY312 -m verity_capture.bench.gen_batch_attribution --log "$OUT/$n/capture/log.jsonl.gz" \
      --workload "workloads/workload_$n.json" --run-dir "$OUT/$n" >> "$LOGS/$n.cpu.log" 2>&1; say "attribution $n rc=$?"
  sync_small "$n"
}

run() {  # run <workload-name> <case> <checkpoint-dir> [driver args...]
  local name=$1 case=$2 ck=$3; shift 3
  say "gpu $name start ($*)"
  drv --case "$case" --workload "workloads/workload_$name.json" --out "$OUT/$name" --checkpoint-dir "$ck" --phase gpu \
      --max-num-seqs 8 "$@" > "$LOGS/$name.gpu.log" 2>&1
  say "gpu $name rc=$?"
  durable_gpu "$name"
  ( cpu_preview "$name" ) &
}

restore() {  # <name>: the /vol tarball back onto the container disk
  local n=$1; mkdir -p "$OUT"; (cd "$OUT" && tar xzf "$VOL/runs/$n.gpu.tgz") && say "restored $n from $VOL/runs/$n.gpu.tgz"
}

# NOTE: run from a COPY outside the shipped tree (cp to /workspace/cp/): re-shipping over a running bash script corrupts it.
MODE=${1:-run}; shift || true
case "$MODE" in
  cpu) for n in "$@"; do ( cpu_preview "$n" ) & done; wait; exit 0 ;;
  sync) for n in "$@"; do sync_small "$n"; done; exit 0 ;;
  restore) restore "$1"; exit 0 ;;
  run) ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
say "start ($*); QWEN=$QWEN"
for n in "$@"; do
  case $n in
    b0_prefix64_2req) run $n B0 "$SMOL" --engine-arg enable_prefix_caching=true ;;
    b0_chunked_200p)  run $n B0 "$SMOL" --engine-arg max_num_batched_tokens=64 --snapshot-steps 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 ;;
    qwen15_*)         run $n B1 "$QWEN" ;;
    *)                run $n B0 "$SMOL" ;;
  esac
done
say "all gpu phases done; waiting for the cpu previews"
wait
say "ALL DONE"
