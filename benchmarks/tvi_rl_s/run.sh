#!/usr/bin/env bash
# One leased GPU, a queue of GRPO runs (crux 1, training-vs-inference program). QUEUE="task:method:seed:steps[:extra args] ...",
# extra args comma-separated (e.g. gsm8k:base:0:0:--passk,64). Checkpoints go to $CKPT_ROOT, results and logs to $OUT.
set -uo pipefail
IN="$RESEARCH_RUN_DIR/inputs"
S=$(mktemp -d "${SCRATCH_ROOT:-/tmp}/tvi-a2.XXXXXX")
trap 'rm -rf "$S"' EXIT
trap 'exit 143' TERM INT
cp "$IN"/grpo.py "$S"/
mkdir -p "$S/data"
cp "$IN"/*.jsonl "$S/data/"
cd "$S"

PY=/workspace/jobs/venv312/bin/python
if [ "${PY_PREF:-auto}" != venv312 ] && [ -f /workspace/jobs/tvi/venv/READY ] \
   && /workspace/jobs/tvi/venv/bin/python -c 'import vllm, transformers, torch' 2>/dev/null; then
  PY=/workspace/jobs/tvi/venv/bin/python
fi
if [ ! -x "$PY" ]; then
  # No node env: build one for this worker under the program's tree, once per node (the second job waits on the lock).
  V=/workspace/jobs/tvi/venv-A2
  mkdir -p /workspace/jobs/tvi
  (
    flock 9
    if [ ! -f "$V/READY" ]; then
      rm -rf "$V"
      uv venv -q -p 3.12 "$V" && uv pip install -q --python "$V/bin/python" vllm math-verify ninja && touch "$V/READY"
    fi
  ) 9>/workspace/jobs/tvi/.venv-A2.lock
  command -v research >/dev/null && research keep "$V" --owner @training --expires 2026-10-27 --priority normal --why 'training-vs-inference program' || true
  PY="$V/bin/python"
fi
echo "python: $PY"
export PATH="$(dirname "$PY"):$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
if ! "$PY" -c 'import math_verify' 2>/dev/null; then
  uv pip install -q --python "$PY" --target "$S/extra" --no-deps math-verify latex2sympy2_extended antlr4-python3-runtime==4.13.2 \
    && export PYTHONPATH="$S/extra"
fi
if [ -d /workspace/hf/hub ] && [ -w /workspace/hf/hub ]; then export HF_HUB_CACHE=/workspace/hf/hub; else export HF_HOME="$S/hf"; fi
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export MODEL DATA="$S/data"
mkdir -p /workspace/jobs/tvi/cache/A2
export SVD_CACHE=/workspace/jobs/tvi/cache/A2
CKPT_ROOT="${CKPT_ROOT:-/workspace/jobs/tvi/ckpt/A2}"
mkdir -p "$CKPT_ROOT"
if command -v research >/dev/null; then
  research keep /workspace/jobs/tvi/ckpt/A2 --owner @training --expires 2026-10-27 --priority normal --why 'training-vs-inference program' || true
  research keep /workspace/jobs/tvi/cache/A2 --owner @training --expires 2026-10-27 --priority normal --why 'training-vs-inference program' || true
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
"$PY" -c 'import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__, torch.cuda.get_device_name(0))'

if [ -n "${EVAL_ARGS:-}" ]; then
  # Checkpoint evaluation instead of training: EVAL_ARGS are evalckpts.py's flags, comma-separated; several invocations
  # (e.g. one per model, a later --model overriding MODEL) are separated by ';' and run in turn under the one lease.
  cp "$IN"/evalckpts.py "$S"/
  set -f
  IFS=';' read -ra EVALS <<< "$EVAL_ARGS"
  i=0
  for ea in "${EVALS[@]}"; do
    i=$((i + 1))
    # shellcheck disable=SC2086
    "$PY" evalckpts.py --root "$CKPT_ROOT" --model "$MODEL" --data "$S/data" --out "$OUT" ${ea//,/ } > "$OUT/log-evalckpts-$i.txt" 2>&1
    echo "evalckpts $i rc=$?"
  done
  set +f
  grep -h '^{' "$OUT"/log-evalckpts-*.txt | tail -80
  exit 0
fi

run_item() {  # task:method:seed:steps[:est_min[:extra]]
  IFS=: read -r task method seed steps est extra <<< "$1"
  # shellcheck disable=SC2086
  "$PY" grpo.py --task "$task" --method "$method" --seed "$seed" --steps "$steps" --model "$MODEL" --data "$S/data" \
    --out "$OUT" --ckpt-root "$CKPT_ROOT" ${extra//,/ } ${EXTRA:-} > "$OUT/log-$task-$method-seed$seed.txt" 2>&1
  echo "$(date -u +%H:%MZ) $1 rc=$?"
  grep -h '^{' "$OUT/log-$task-$method-seed$seed.txt" | tail -2
}

if [ -n "${ITEMS_FILE:-}" ]; then
  # Worker over a shared items file, re-read before every claim so items can be added while jobs hold their leases. A line
  # is `NAME EST_MIN grpo|eval ARGS...` (ARGS go to grpo.py or evalckpts.py verbatim); claims are atomic mkdirs in QDIR, an
  # item whose estimate would run past DEADLINE is skipped, and every item is killed at DEADLINE. When CODE_FROM/READY
  # exists the scripts are refreshed from CODE_FROM before each item, and the copy an item ran is kept in $OUT/code/NAME.
  mkdir -p "$QDIR"
  cp "$IN"/evalckpts.py "$S"/
  idle_since=$(date +%s)
  while :; do
    if [ -f "${CODE_FROM:-/nonexistent}/READY" ]; then cp "$CODE_FROM"/grpo.py "$CODE_FROM"/evalckpts.py "$S"/; fi
    picked=""
    while read -r name est kind rest; do
      case "$name" in ""|"#"*) continue ;; esac
      [ -e "$QDIR/$name" ] && continue
      [ $(( $(date +%s) + est * 60 )) -gt "$DEADLINE" ] && continue
      if mkdir "$QDIR/$name" 2>/dev/null; then
        echo "${RESEARCH_RUN_ID:-} $(date -u +%H:%MZ)" > "$QDIR/$name/claimed"
        picked="$name"; pkind="$kind"; pargs="$rest"
        break
      fi
    done < "$ITEMS_FILE"
    if [ -z "$picked" ]; then
      [ $(( $(date +%s) - idle_since )) -gt $(( ${IDLE_MAX_MIN:-12} * 60 )) ] && break
      [ $(( $(date +%s) + 300 )) -gt "$DEADLINE" ] && break
      sleep 30
      continue
    fi
    mkdir -p "$OUT/code/$picked" && cp "$S"/grpo.py "$S"/evalckpts.py "$OUT/code/$picked/"
    left=$(( DEADLINE - $(date +%s) ))
    script=grpo.py; [ "$pkind" = eval ] && script=evalckpts.py
    set -f
    # shellcheck disable=SC2086
    if [ "$pkind" = eval ]; then
      timeout "$left" "$PY" evalckpts.py --root "$CKPT_ROOT" --data "$S/data" --out "$OUT" $pargs > "$OUT/log-$picked.txt" 2>&1
    else
      timeout "$left" "$PY" grpo.py --data "$S/data" --out "$OUT" --ckpt-root "$CKPT_ROOT" --deadline "$DEADLINE" $pargs \
        > "$OUT/log-$picked.txt" 2>&1
    fi
    rc=$?
    set +f
    echo "$(date -u +%H:%MZ) $picked ($script) rc=$rc"
    grep -h '^{' "$OUT/log-$picked.txt" | tail -2
    echo "$rc $(date -u +%H:%MZ)" > "$QDIR/$picked/finished"
    idle_since=$(date +%s)
  done
elif [ -n "${QDIR:-}" ]; then
  # Shared queue: every GPU job walks ITEMS in order and claims the first unclaimed one (mkdir is atomic), skipping an item
  # whose estimate would run past DEADLINE (epoch seconds).
  mkdir -p "$QDIR"
  while :; do
    picked=""
    for item in $ITEMS; do
      IFS=: read -r task method seed steps est extra <<< "$item"
      name="$task-$method-seed$seed${extra:+-x}"
      [ -e "$QDIR/$name" ] && continue
      if [ -n "${DEADLINE:-}" ] && [ $(( $(date +%s) + ${est:-20} * 60 )) -gt "$DEADLINE" ]; then continue; fi
      if mkdir "$QDIR/$name" 2>/dev/null; then
        echo "${RESEARCH_RUN_ID:-} $(date -u +%H:%MZ)" > "$QDIR/$name/claimed"
        picked="$item"
        break
      fi
    done
    [ -z "$picked" ] && break
    run_item "$picked"
    touch "$QDIR/$name/finished"
  done
else
  for item in $QUEUE; do run_item "$item"; done
fi

NODE="${NODE_NAME:-$(hostname)}" "$PY" - <<'EOF'
import glob, json, os
out = os.environ["OUT"]
rows = []
for p in sorted(glob.glob(f"{out}/result-*.json")):
    r = json.load(open(p))
    rows.append({"node": os.environ.get("NODE", ""), "run_id": os.environ.get("RESEARCH_RUN_ID", ""), "task": r["task"],
                 "method": r["method"], "seed": r.get("seed", 0), "S_bits": r["S_bits"], "n_params": r.get("n_params", 0),
                 "steps": r["steps"], "score": r["acc"], "ckpts": r.get("ckpts", {}), "model": r.get("model")})
json.dump(rows, open(f"{out}/manifest.json", "w"), indent=1)
print(json.dumps(rows, indent=1))
EOF
