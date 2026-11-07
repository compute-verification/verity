#!/usr/bin/env bash
# The capture of sampled proof blocks' values from a replay of a served GLM-5.3 window, under research run from a tree with
# benchmarks/serve_tax/replay.py at its root (--cwd source), with capture.py, stage.py and this file sent (--send):
#   ARM=RECORD_ARM_DIR LOAD=NAME [OPEN=dense:1,moe:1] [PER_TYPE=8] [SMALL=1,31] [BIG=256,384] [STEPS=s,...] [VENV=...]
#     bash "$RESEARCH_RUN_DIR/inputs/spot_job.sh"
#   ARM=... LOAD=... CAPTURE=DIR STEPS=s,... bash .../spot_job.sh   # only stage an earlier run's capture again (CPU only)
# The engine's environment is the served one's: ENGINE_ENV=FILE (KEY=VALUE lines, a server's engine_env.txt) is exported,
# and FLASHINFER=1 (the default, as production) puts $CUDA_HOME/bin (default /usr/local/cuda-12.9) on PATH for FlashInfer's
# JIT; OUT/engine_env.txt.
# 1. The first-stage coin: one step of the window's with SMALL tokens (the FP8 linears' FlashInfer branch, the fused MNNVL
#    all-reduce + norm) and one with BIG (DeepGEMM and its group quants, the multimem all-reduce, the fused add + RMSNorm at
#    its 256-thread block, FA3 still split), each uniform over those steps, drawn here before the replay starts
#    (OUT/first_stage.json; a class with no step takes the window's nearest one above it up to BIG_MAX=1024 tokens, else
#    the big class its largest one over 42 tokens); STEPS overrides it.
# 2. replay.py replays the window from an empty engine, served as recorded, with capture.py loaded in every worker: those
#    steps' interiors are recorded and committed, the coins drawn after the root open layers, and only the opened layers'
#    tensors are written (/workspace/spot-capture/<run>, outside OUT, which custody uploads). The graph-off forward holds the
#    step's tensors in host memory until the coins: about 10 GB a rank at 384 tokens.
# 3. stage.py stages each step's sampled proof blocks into OUT/values-step<s> (a11's saved values: a few MB) and the
#    commitment's small files are copied to OUT/commit. The capture (the opened layers only) stays for a restage; KEEP=0
#    deletes it once every step staged.
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
[ -d "$CK" ] || { echo "spot: no checkpoint at $CK"; exit 2; }
v=$("$PY" -c 'import vllm; print(vllm.__version__)') || exit 2
case $v in *+gd9105ea8*) ;; *) echo "spot: vLLM $v is not d9105ea80, the integration's pin"; exit 2 ;; esac
CAP=${CAPTURE:-/workspace/spot-capture/$(basename "$D")}; mkdir -p "$CAP/site"
restage=0; [ -n "${CAPTURE:-}" ] && { restage=1; : "${STEPS:?STEPS: the captured steps to stage again}"; }
free=$(df -BG --output=avail "$CAP" | tail -1 | tr -dc 0-9)
[ "${free:-0}" -ge 40 ] || { echo "spot: ${free}G free under $CAP, want 40G"; exit 2; }
unset NCCL_P2P_DISABLE VLLM_USE_DEEP_GEMM
export HF_HOME=${HF_HOME:-/workspace/cp/hf} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
# the served engine's environment: replay.py takes its engine arguments from the trace, and its environment from here
if [ -n "${ENGINE_ENV:-}" ]; then
  while IFS='=' read -r k v; do
    [[ $k =~ ^[A-Z_][A-Z0-9_]*$ ]] && [ "$k" != CUDA_VISIBLE_DEVICES ] && [ "$v" != unset ] && export "$k=$v"
  done < "$ENGINE_ENV"
fi
if [ "${FLASHINFER:-1}" = 1 ]; then
  [ -n "${CUDA_HOME:-}" ] || for c in /usr/local/cuda-12.9 /usr/local/cuda; do [ -x "$c/bin/nvcc" ] && { CUDA_HOME=$c; break; }; done
  export CUDA_HOME PATH="${CUDA_HOME:?FLASHINFER=1 needs nvcc: set CUDA_HOME}/bin:$VENV/bin:$PATH"
fi
{ for k in VLLM_USE_DEEP_GEMM VLLM_USE_FLASHINFER_SAMPLER CUDA_HOME CUDA_VISIBLE_DEVICES; do echo "$k=${!k-unset}"; done
  echo "nvcc=$(command -v nvcc || echo none)"
  "$PY" -c 'import importlib.util as u; print("has_flashinfer=" + str(u.find_spec("flashinfer") is not None))'
} > "$OUT/engine_env.txt"
cat "$OUT/engine_env.txt"
echo "spot: tree $(git -C "$TREE" rev-parse HEAD 2>/dev/null) venv $VENV checkpoint $CK capture $CAP ($free G free)"

rc=0
if [ $restage = 0 ] && [ -z "${STEPS:-}" ]; then
  "$PY" - "$TREE" "$ARM" "$LOAD" "${SMALL:-1,31}" "${BIG:-256,384}" "$OUT/first_stage.json" <<'EOF' || exit 2
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
    if not steps:
        # the graph-off forward holds about 26 MB a token a rank until the coins: no step past BIG_MAX tokens
        near = sorted((n, s) for s, n in tokens.items() if hi < n <= int(os.environ.get("BIG_MAX", "1024")))
        near = near or sorted(((-n, s) for s, n in tokens.items() if 42 < n < lo and name == "big"))
        if near:
            steps = [s for n, s in near if n == near[0][0]]
            classes[name]["nearest"] = {"tokens": abs(near[0][0]), "steps": len(steps)}
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
if [ $restage = 0 ]; then
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
fi
"$PY" - "$TREE" "$ARM" "$LOAD" "$CAP" "$STEPS" "$OUT/alignment.json" <<'EOF'
import json, sys
from pathlib import Path
tree, arm, load, cap, steps, out = sys.argv[1:]
sys.path.insert(0, str(Path(tree) / "benchmarks" / "serve_tax"))
import replay
served = {s["step"]: {r[0]: int(r[2]) for r in s["rows"]} for s in replay.window(Path(arm), load, 1)["steps"]}
res = {}
for s in map(int, steps.split(",")):
    m = Path(cap) / "rank0" / f"step{s}" / "meta.json"
    got = json.loads(m.read_text())["scheduled"] if m.exists() else None
    want = served.get(s)
    same = got is not None and want is not None and len(got) == len(want) and all(
        any(g == w or g.startswith(w + "-") for g in got if got[g] == n) for w, n in want.items())
    res[s] = {"same_rows": same, "served": want, "captured": got}
    print(f"spot: step {s}: the captured step's rows {'are' if same else 'are NOT'} the served window's step {s}'s")
json.dump(res, open(out, "w"), indent=1)
EOF
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
  (cd "$TREE" && "$PY" "$IN/stage.py" --capture "$CAP" --step "$s" \
    --out "$OUT/values-step$s" --per-type "${PER_TYPE:-8}" --row "glm53-spot-${LOAD}-step$s" \
    --of "GLM-5.3 FP8 TP8 on 8x H200: step $s of load $LOAD's served window ($ARM), replayed from an empty engine on the same machine with CUDA graphs as served; its interior recorded by a graph-off forward of the same step checked bit for bit against the graph's" \
    --link "the replay's tokens against the served window's (OUT/replay/replay.json); the weights are the served engine's parameters, committed by SHA-512" \
    > "$OUT/stage-step$s.log" 2>&1) || { echo "spot: stage step $s failed"; staged=0; }
  tail -n 40 "$OUT/stage-step$s.log"
done
du -sh "$CAP" "$OUT"/* 2>/dev/null
if [ "${KEEP:-1}" = 1 ] || [ $staged = 0 ]; then echo "spot: capture kept at $CAP"; else rm -rf "$CAP"; fi
exit $rc
