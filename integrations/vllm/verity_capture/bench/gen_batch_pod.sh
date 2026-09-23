#!/usr/bin/env bash
# GEN batch lane — pod-side runs of the sealed driver for the memory-model / request-attribution configurations (pod cp-batch, 1x RTX 4090).
# GPU phases run one after another (one GPU); each run's CPU phase starts in the background as soon as its GPU phase is done, with the
# wave-1 replay budget (--replay-budget-leaves 8000000: instances above it are budget-skipped, tier b `partial`, G5 `skipped`; the
# integrator re-runs tier b with the vectorized evaluator).  The two engine-arg configurations use the SHARED-CODE `--engine-arg`
# passthrough (commit "GEN batch SHARED-CODE"), which forwards the knob to m1_capture / census / noninterference -> build_engine.
#
#   cp verity_capture/bench/gen_batch_pod.sh /workspace/cp/ && nohup bash /workspace/cp/gen_batch_pod.sh [all | run <names...>] > /vol/cp/batch/logs/gen_batch_pod.log 2>&1 &
#   bash /workspace/cp/gen_batch_pod.sh decode-replays          # afterwards: budget-free tier b of the 2- and 4-row decode steps
#   bash /workspace/cp/gen_batch_pod.sh fix-nonint <name>       # re-run only the nonint stage (+ the CPU stages reading it) of an existing run
#   bash /workspace/cp/gen_batch_pod.sh gates [names...]        # nonint_compare + gates again (after tier-b replays were re-run by hand)
set -uo pipefail
cd /workspace/cp/veritor
export PYTHONPATH=. HF_HOME=/workspace/hf
PY312=/workspace/venv-py312/bin/python; PY311=/workspace/venv-cu129/bin/python
# Runs are written to the container disk: the shared network volume /vol was OVER QUOTA in session 1 (writes failed with EDQUOT mid-capture,
# killing every stage that touched it); `sync-vol` copies the small files (no capture/values, no census traces) to /vol/cp/batch afterwards.
OUT=/workspace/cp/batch/runs; CARDS=/workspace/cp/batch/cards; LOGS=/workspace/cp/batch/logs; mkdir -p "$OUT" "$CARDS" "$LOGS"
SMOL=/workspace/hf/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2
QWEN=$(ls -d /workspace/hf/hub/models--Qwen--Qwen2.5-1.5B/snapshots/* 2>/dev/null | head -1)
say() { echo "[gen_batch_pod] $(date -u +%FT%TZ) $*"; }
drv() { $PY312 -m verity_capture.bench.run_config --python-gpu $PY311 --python-cpu $PY312 --cards-dir "$CARDS" "$@"; }

run() {  # run <workload-name> <case> <checkpoint-dir> [driver args...]
  local name=$1 case=$2 ck=$3; shift 3
  say "gpu $name start ($*)"
  drv --case "$case" --workload "verity_capture/profiles/workload_$name.json" --out "$OUT/$name" --checkpoint-dir "$ck" --phase gpu \
      --max-num-seqs 8 "$@" > "$LOGS/$name.gpu.log" 2>&1
  say "gpu $name rc=$?"
  ( drv --run-dir "$OUT/$name" --phase cpu --parallel 4 --replay-budget-leaves 8000000 > "$LOGS/$name.cpu.log" 2>&1
    say "cpu $name rc=$?"
    $PY312 -m verity_capture.bench.gen_batch_attribution --log "$OUT/$name/capture/log.jsonl.gz" \
        --workload "verity_capture/profiles/workload_$name.json" --run-dir "$OUT/$name" >> "$LOGS/$name.cpu.log" 2>&1
    say "attribution $name rc=$?" ) &
}

decode_replays() {  # budget-free tier b of the small decode steps (2 rows: ~20 min each, 4 rows: ~40 min; pure Python) -> then the gates again
  for name in b0_conc2_small b0_conc4_mixed; do
    for s in 1 2 3; do
      ( $PY312 -m verity_capture.replay --log "$OUT/$name/capture/log.jsonl.gz" --snapshots "$OUT/$name/capture" \
            --profile vllm_d9105ea80_sm89_eager --tier b --step $s --out "$OUT/$name/replay_b_step$s.json" > "$LOGS/$name.replay_b_step$s.nobudget.log" 2>&1
        say "nobudget tier b $name step $s rc=$?" ) &
    done
  done
  wait
  for name in b0_conc2_small b0_conc4_mixed; do
    drv --run-dir "$OUT/$name" --phase cpu --skip replay,census,nonint,fold,resolve,compare,root_policy,golden > "$LOGS/$name.gates2.log" 2>&1
    say "gates again $name rc=$?"
  done
}

sync_vol() {  # small files -> the durable volume (and a tarball for the laptop under /workspace/cp/batch/)
  mkdir -p /vol/cp/batch/runs /vol/cp/batch/cards
  for d in "$OUT"/*/; do n=$(basename "$d"); mkdir -p "/vol/cp/batch/runs/$n"
    (cd "$OUT" && tar cf - --exclude="$n/capture/values" --exclude="$n/census/traces" --exclude="$n/nonint/*/values" "$n") | (cd /vol/cp/batch/runs && tar xf -) || say "sync $n FAILED (quota?)"
  done
  cp -r "$CARDS"/. /vol/cp/batch/cards/ 2>/dev/null; cp "$LOGS"/*.log /vol/cp/batch/logs/ 2>/dev/null
  (cd /workspace/cp/batch && tar czf runs_small.tgz --exclude="runs/*/capture/values" --exclude="runs/*/census/traces" --exclude="runs/*/nonint/*/values" runs cards logs) && ls -la /workspace/cp/batch/runs_small.tgz
}

gates_only() {  # nonint_compare + gates of existing runs (after replays finished / were re-run by hand)
  for name in "$@"; do
    drv --run-dir "$OUT/$name" --phase cpu --skip replay,census,fold,resolve,compare,root_policy,golden > "$LOGS/$name.gates3.log" 2>&1; say "gates $name rc=$?"
  done
}

cpu_only() {  # CPU phase of an existing run (background)
  local name=$1
  ( drv --run-dir "$OUT/$name" --phase cpu --parallel 4 --replay-budget-leaves 8000000 > "$LOGS/$name.cpu.log" 2>&1; say "cpu $name rc=$?"
    $PY312 -m verity_capture.bench.gen_batch_attribution --log "$OUT/$name/capture/log.jsonl.gz" \
        --workload "verity_capture/profiles/workload_$name.json" --run-dir "$OUT/$name" >> "$LOGS/$name.cpu.log" 2>&1; say "attribution $name rc=$?" ) &
}

fix_nonint() {  # re-run only the nonint GPU stage of an existing run (after the HashOnlyCapture fix landed), then the CPU stages that read it
  local name=$1; shift
  drv --run-dir "$OUT/$name" --phase gpu --skip capture,control,census "$@" > "$LOGS/$name.nonint2.log" 2>&1; say "nonint again $name rc=$?"
  drv --run-dir "$OUT/$name" --phase cpu --skip replay,fold,resolve,census,compare,root_policy,golden > "$LOGS/$name.gates2.log" 2>&1; say "gates again $name rc=$?"
}

# NOTE: run this script from a COPY outside the shipped tree (cp to /workspace/cp/); re-shipping the worktree over a running bash script
# corrupts it (bash reads the file incrementally) -- that is how the first session's sequence broke at 03:29Z.
case "${1:-all}" in
  decode-replays) decode_replays; exit 0 ;;
  fix-nonint) shift; fix_nonint "$@"; exit 0 ;;
  sync-vol) sync_vol; exit 0 ;;
  gates) shift; gates_only "${@:-b0_conc2_small b0_conc4_mixed b0_conc8_mixed b0_long512_1req b0_prefix64_2req b0_chunked_200p qwen15_conc4_mixed}"; exit 0 ;;
  cpu) shift; for n in "$@"; do cpu_only "$n"; done; wait; exit 0 ;;
  redo-conc2) # session 1: conc2's GPU stages are valid but nonint crashed (HashOnlyCapture) and its CPU phase wrote to the full volume
    drv --run-dir "$OUT/b0_conc2_small" --phase gpu --skip capture,control,census > "$LOGS/b0_conc2_small.nonint2.log" 2>&1; say "nonint again b0_conc2_small rc=$?"
    cpu_only b0_conc2_small; wait; exit 0 ;;
  run) shift; NAMES="$*" ;;
  all) NAMES="b0_conc2_small b0_conc4_mixed b0_conc8_mixed b0_long512_1req b0_prefix64_2req b0_chunked_200p qwen15_conc4_mixed" ;;
esac
say "start ($NAMES); QWEN=$QWEN"
for n in $NAMES; do
  case $n in
    b0_prefix64_2req) run $n B0 "$SMOL" --engine-arg enable_prefix_caching=true ;;
    b0_chunked_200p)  run $n B0 "$SMOL" --engine-arg max_num_batched_tokens=64 ;;
    qwen15_*)         run $n B1 "$QWEN" ;;
    *)                run $n B0 "$SMOL" ;;
  esac
done
say "all gpu phases done; waiting for the cpu phases"
wait
say "ALL DONE"
