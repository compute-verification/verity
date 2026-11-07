#!/usr/bin/env bash
# The capture of sampled proof blocks' values from a replay of a served GLM-5.3 window, under research run from a tree with
# benchmarks/serve_tax/replay.py at its root (--cwd source), with capture.py, stage.py and this file sent (--send):
#   ARM=RECORD_ARM_DIR LOAD=NAME [OPEN=dense:1,moe:1] [PER_TYPE=8] [SMALL=1,31] [BIG=32,64] [STEPS=s,...] [VENV=...]
#     bash "$RESEARCH_RUN_DIR/inputs/spot_job.sh"
# 1. The first-stage coin: one step of the window's with SMALL tokens (the FP8 linears' FlashInfer branch) and one with BIG
#    (their DeepGEMM branch and its group quants), each uniform over those steps, drawn here before the replay starts
#    (OUT/first_stage.json); STEPS overrides it.
# 2. replay.py replays the window from an empty engine, served as recorded, with capture.py loaded in every worker: those
#    steps' interiors are recorded and committed, the coins drawn after the root open layers (/workspace/spot-capture/<run>,
#    outside OUT, which custody uploads).
# 3. stage.py stages each step's sampled proof blocks into OUT/values-step<s> (a11's saved values: a few MB) and the
#    commitment's small files are copied to OUT/commit; the capture's blobs are then deleted unless KEEP=1.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out; IN=$D/inputs; mkdir -p "$OUT"
TREE=$(pwd)
[ -f "$TREE/benchmarks/serve_tax/replay.py" ] || { echo "spot: run from a tree with benchmarks/serve_tax/replay.py (--cwd source)"; exit 2; }
for f in capture.py stage.py; do [ -f "$IN/$f" ] || { echo "spot: --send $f"; exit 2; }; done
: "${ARM:?ARM: the record arm directory}" "${LOAD:?LOAD: the load to replay}"
if [ -z "${VENV:-}" ]; then for v in /workspace/cp/venv312 /workspace/venv312; do [ -x "$v/bin/python" ] && { VENV=$v; break; }; done; fi
PY=$VENV/bin/python; export PATH="$VENV/bin:$PATH"
command -v nvcc >/dev/null || for d in ${CUDA_HOME:+"$CUDA_HOME/bin"} $(ls -d /usr/local/cuda*/bin 2>/dev/null | sort -rV); do
  [ -x "$d/nvcc" ] && { export PATH="$PATH:$d"; break; }
done
REV=aca966e4e02791568aa6a4ced368624b3d897f42
if [ -e "/workspace/cp/glm53-ckpt/$REV/.verified" ]; then CK=/workspace/cp/glm53-ckpt/$REV; else CK=/workspace/fs/glm53/$REV; fi
CK=${CHECKPOINT:-$CK}
CAP=/workspace/spot-capture/$(basename "$D"); mkdir -p "$CAP/site"
free=$(df -BG --output=avail "$CAP" | tail -1 | tr -dc 0-9)
[ "${free:-0}" -ge 100 ] || { echo "spot: ${free}G free under $CAP, want 100G"; exit 2; }
unset NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM
export HF_HOME=${HF_HOME:-/workspace/cp/hf} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
echo "spot: tree $(git -C "$TREE" rev-parse HEAD 2>/dev/null) venv $VENV checkpoint $CK capture $CAP ($free G free)"

if [ -z "${STEPS:-}" ]; then
  "$PY" - "$TREE" "$ARM" "$LOAD" "${SMALL:-1,31}" "${BIG:-32,64}" "$OUT/first_stage.json" <<'EOF' || exit 2
import json, os, sys, time
from pathlib import Path
tree, arm, load, small, big, out = sys.argv[1:]
sys.path.insert(0, str(Path(tree) / "benchmarks" / "serve_tax"))
import replay
w = replay.window(Path(arm), load, 1)
tokens = {s["step"]: sum(int(r[2]) for r in s["rows"]) for s in w["steps"]}
coins = os.urandom(64)
pick, classes = {}, {}
for name, rng in (("small", small), ("big", big)):
    lo, hi = map(int, rng.split(","))
    steps = sorted(s for s, n in tokens.items() if lo <= n <= hi)
    classes[name] = {"tokens": [lo, hi], "steps": len(steps)}
    if steps:
        pick[name] = steps[int.from_bytes(coins[:32] if name == "small" else coins[32:], "big") % len(steps)]
json.dump({"window_steps": len(tokens), "classes": classes, "coins": coins.hex(), "drawn_at": time.time(), "steps": pick,
           "tokens": {k: tokens[v] for k, v in pick.items()}, "rows": {k: next(s["rows"] for s in w["steps"] if s["step"] == v)
                                                                       for k, v in pick.items()}}, open(out, "w"), indent=1)
print("spot first stage:", pick, {k: tokens[v] for k, v in pick.items()}, classes)
EOF
  STEPS=$("$PY" -c "import json; print(','.join(str(s) for s in sorted(json.load(open('$OUT/first_stage.json'))['steps'].values())))")
fi
[ -n "$STEPS" ] || { echo "spot: no step drawn"; exit 2; }
"$PY" "$IN/capture.py" site "$CAP/site" >/dev/null || exit 2
export PYTHONPATH="$CAP/site${PYTHONPATH:+:$PYTHONPATH}"
export VERITY_SPOT_CAPTURE=$CAP VERITY_SPOT_STEPS=$STEPS VERITY_SPOT_OPEN=${OPEN:-dense:1,moe:1}
export TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
echo "== replay with capture of steps $STEPS $(date -u +%FT%TZ)"
GPU_LEASE_WHO=${GPU_LEASE_WHO:-spot-capture} gpu-lease 8 --wait --max-min "${MAX_MIN:-120}" -- \
  "$PY" "$TREE/benchmarks/serve_tax/replay.py" --arm-dir "$ARM" --load "$LOAD" --checkpoint "$CK" --out "$OUT/replay" \
  > "$OUT/replay.log" 2>&1
rc=$?
grep -a "spot-capture" "$OUT/replay.log" | tail -n 40
tail -n 3 "$OUT/replay.log"
echo "== stage $(date -u +%FT%TZ)"
mkdir -p "$OUT/commit"
staged=1
for s in ${STEPS//,/ }; do
  for r in "$CAP"/rank*; do
    [ -d "$r/step$s" ] || continue
    mkdir -p "$OUT/commit/$(basename "$r")/step$s"
    cp "$r/step$s/meta.json" "$r/step$s/commit.json" "$OUT/commit/$(basename "$r")/step$s/" 2>/dev/null
    gzip -c "$r/step$s/ops.jsonl" > "$OUT/commit/$(basename "$r")/step$s/ops.jsonl.gz"
    cp "$r/params.json" "$OUT/commit/$(basename "$r")/" 2>/dev/null
  done
  PYTHONPATH="$TREE:$TREE/verity/ml/flock/python:$TREE/integrations/vllm" "$PY" "$IN/stage.py" --capture "$CAP" --step "$s" \
    --out "$OUT/values-step$s" --per-type "${PER_TYPE:-8}" --row "glm53-spot-${LOAD}-step$s" \
    --of "GLM-5.3 FP8 TP8 on 8x H200: step $s of load $LOAD's served window ($ARM), replayed from an empty engine on the same machine with CUDA graphs as served; its interior recorded by a graph-off forward of the same step checked bit for bit against the graph's" \
    --link "the replay's tokens against the served window's (OUT/replay/replay.json); the weights are the served engine's parameters, committed by SHA-512" \
    > "$OUT/stage-step$s.log" 2>&1 || { echo "spot: stage step $s failed"; staged=0; }
  tail -n 40 "$OUT/stage-step$s.log"
done
du -sh "$CAP" "$OUT"/* 2>/dev/null
if [ "${KEEP:-0}" = 1 ] || [ $staged = 0 ]; then echo "spot: capture kept at $CAP"; else rm -rf "$CAP"; fi
exit $rc
