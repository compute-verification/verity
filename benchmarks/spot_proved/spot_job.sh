#!/usr/bin/env bash
# The capture of sampled proof blocks' values from a replay of a served GLM-5.3 window, under research run from a tree with
# benchmarks/serve_tax/replay.py at its root (--cwd source), with capture.py, stage.py and this file sent (--send):
#   ARM=RECORD_ARM_DIR LOAD=NAME [OPEN=dense:1,moe:1] [PER_TYPE=8] [SMALL=1,31] [BIG=256,384] [LONG=1] [STEPS=s,...]
#     [KEEP_TENSORS=opened|all] [KEEP_STEPS=s,...] [KEEP_MAX_MIB=64] [MLA_TOKENS=8] [GATE_EVERY=N] [STAGE=1]
#     [DEADLINE=UNIX_TIME] [REPLAY_ARGS=...] [VENV=...]
#     bash "$RESEARCH_RUN_DIR/inputs/spot_job.sh"
#   ARM=... LOAD=... CAPTURE=DIR STEPS=s,... bash .../spot_job.sh   # only stage an earlier run's capture again (CPU only)
# The engine's environment is the served one's: ENGINE_ENV=FILE (KEY=VALUE lines, a server's engine_env.txt) is exported,
# and FLASHINFER=1 (the default, as production) puts $CUDA_HOME/bin (default /usr/local/cuda-12.9) on PATH for FlashInfer's
# JIT; OUT/engine_env.txt.
# 1. The first-stage coin: one step of the window's with SMALL tokens (the FP8 linears' FlashInfer branch, the fused MNNVL
#    all-reduce + norm) and one with BIG (DeepGEMM and its group quants, the multimem all-reduce, the fused add + RMSNorm at
#    its 256-thread block, FA3 still split), each uniform over those steps, drawn here before the replay starts
#    (OUT/first_stage.json; a class with no step takes the window's nearest one above it up to BIG_MAX=1024 tokens, else
#    the big class its largest one over 42 tokens); LONG=1 (the default) adds one uniform over the steps with a prefill row
#    and a row past LONG_KEYS=2048 keys, of LONG_TOKENS=256,384 tokens (else the largest such step under them), other
#    than the big class's step when another qualifies; STEPS overrides it.
# 2. replay.py replays the window from an empty engine, served as recorded, with capture.py loaded in every worker
#    (REPLAY_ARGS are added to its arguments): those steps' interiors are recorded and committed, the coins drawn after the
#    root open layers, and the opened layers' tensors are written, or every recorded tensor with KEEP_TENSORS=all at the
#    KEEP_STEPS (default: the drawn steps, long first, while they fit the disk at GB_PER_TOKEN)
#    (/workspace/spot-capture/<run>, outside OUT, which custody uploads; MIN_FREE_G=40 is the free space it wants). The
#    graph-off forward holds the step's tensors in host memory until the coins: about 10 GB a rank at 384 tokens.
# 3. GATE_EVERY=N runs stage.py's match gate on N instances of every call of every type at each step, the long step first,
#    in GATE_JOBS processes, into OUT/gate-step<s>/gate.json, stopping at DEADLINE.
# 4. STAGE=1 (the default) stages each step's sampled proof blocks into OUT/values-step<s> (a11's saved values: a few MB),
#    unless DEADLINE has passed, and the commitment's small files are copied to OUT/commit. The capture stays for a restage;
#    KEEP=0 deletes it once every step staged.
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
[ "${free:-0}" -ge "${MIN_FREE_G:-40}" ] || { echo "spot: ${free}G free under $CAP, want ${MIN_FREE_G:-40}G"; exit 2; }
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
import hashlib, json, os, sys, time
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
if os.environ.get("LONG", "1") == "1":
    keys = int(os.environ.get("LONG_KEYS", "2048"))
    lo, hi = map(int, os.environ.get("LONG_TOKENS", "256,384").split(","))
    longs = {s["step"]: tokens[s["step"]] for s in w["steps"]
             if any(int(r[2]) > 1 for r in s["rows"]) and any(int(r[1]) + int(r[2]) > keys for r in s["rows"])}
    steps = sorted(s for s, n in longs.items() if lo <= n <= hi)
    classes["long"] = {"prefill_and_keys_over": keys, "tokens": [lo, hi], "steps": len(steps), "any_tokens": len(longs)}
    if not steps:
        near = sorted((-n, s) for s, n in longs.items() if n < lo)
        steps = [s for n, s in near if n == near[0][0]] if near else []
        if steps:
            classes["long"]["nearest"] = {"tokens": -near[0][0], "steps": len(steps)}
    steps = [s for s in steps if s != pick.get("big")] or steps
    if steps:
        pick["long"] = steps[int.from_bytes(hashlib.sha512(coins + b"long").digest(), "big") % len(steps)]
json.dump({"window_steps": len(tokens), "classes": classes, "coins": coins.hex(), "drawn_at": time.time(), "steps": pick,
           "tokens": {k: tokens[v] for k, v in pick.items()}, "rows": {k: next(s["rows"] for s in w["steps"] if s["step"] == v)
                                                                       for k, v in pick.items()}}, open(out, "w"), indent=1)
print("spot first stage:", pick, {k: tokens[v] for k, v in pick.items()}, classes)
EOF
  STEPS=$("$PY" -c "import json; p = json.load(open('$OUT/first_stage.json'))['steps']; print(','.join(dict.fromkeys(str(p[k]) for k in ('long', 'big', 'small') if k in p)))")
fi
[ -n "$STEPS" ] || { echo "spot: no step drawn"; exit 2; }
if [ "${KEEP_TENSORS:-opened}" = all ] && [ -z "${KEEP_STEPS:-}" ] && [ -f "$OUT/first_stage.json" ]; then
  # whole steps in the order drawn (long, big, small) while about GB_PER_TOKEN (0.5: about 64 MB a token on each of 8
  # ranks, the tensors up to capture.py's keep cap)
  # a token fits the free space less 40 GB; the others keep the opened layers'
  KEEP_STEPS=$("$PY" - "$OUT/first_stage.json" "$free" "$STEPS" <<'EOF'
import json, os, sys
fs, free, steps = json.load(open(sys.argv[1])), float(sys.argv[2]), sys.argv[3].split(",")
tok = {str(fs["steps"][k]): fs["tokens"][k] for k in fs["steps"]}
per, left, keep = float(os.environ.get("GB_PER_TOKEN", "0.5")), free - 40, []
for s in steps:
    if tok.get(s, 0) * per <= left:
        keep.append(s)
        left -= tok.get(s, 0) * per
print(",".join(keep))
EOF
)
  echo "spot: steps kept whole: ${KEEP_STEPS:-none} (${free}G free)"
  [ -n "$KEEP_STEPS" ] || KEEP_TENSORS=opened
fi
if [ $restage = 0 ]; then
"$PY" "$IN/capture.py" site "$CAP/site" --capture "$CAP" --steps "$STEPS" --open "${OPEN:-dense:1,moe:1}" \
  --keep "${KEEP_TENSORS:-opened}" ${KEEP_STEPS:+--keep-steps "$KEEP_STEPS"} --keep-max-mib "${KEEP_MAX_MIB:-64}" \
  --min-free-gib "${MIN_FREE_G:-40}" --mla-tokens "${MLA_TOKENS:-8}" >/dev/null || exit 2
export PYTHONPATH="$CAP/site${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
echo "== replay with capture of steps $STEPS $(date -u +%FT%TZ)"
GPU_LEASE_WHO=${GPU_LEASE_WHO:-spot-capture} gpu-lease 8 --wait --max-min "${MAX_MIN:-120}" -- \
  "$PY" "$TREE/benchmarks/serve_tax/replay.py" --arm-dir "$ARM" --load "$LOAD" --checkpoint "$CK" --out "$OUT/replay" \
  ${REPLAY_ARGS:-} > "$OUT/replay.log" 2>&1
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
mkdir -p "$OUT/commit"
for s in ${STEPS//,/ }; do
  for r in "$CAP"/rank*; do
    [ -d "$r/step$s" ] || continue
    mkdir -p "$OUT/commit/$(basename "$r")/step$s"
    cp "$r/step$s/meta.json" "$r/step$s/commit.json" "$r/step$s/params.json" "$OUT/commit/$(basename "$r")/step$s/" 2>/dev/null
    gzip -c "$r/step$s/ops.jsonl" > "$OUT/commit/$(basename "$r")/step$s/ops.jsonl.gz"
    cp "$r/params.json" "$OUT/commit/$(basename "$r")/" 2>/dev/null
  done
done
if [ "${GATE_EVERY:-0}" -gt 0 ]; then
  for s in ${STEPS//,/ }; do
    echo "== gate every call of step $s $(date -u +%FT%TZ)"
    (cd "$TREE" && "$PY" "$IN/stage.py" --capture "$CAP" --step "$s" --out "$OUT/gate-step$s" --every-call "$GATE_EVERY" \
      --jobs "${GATE_JOBS:-$(( $(nproc) * 7 / 8 ))}" --deadline "${DEADLINE:-0}" > "$OUT/gate-step$s.log" 2>&1) \
      || echo "spot: gate step $s failed"
    grep -a "^gate" "$OUT/gate-step$s.log" | tail -n 40
  done
fi
staged=1
[ "${STAGE:-1}" = 1 ] || STEPS=
for s in ${STEPS//,/ }; do
  if [ "${DEADLINE:-0}" != 0 ] && [ "$(date +%s)" -ge "${DEADLINE%.*}" ]; then echo "spot: past DEADLINE, step $s not staged"; staged=0; continue; fi
  echo "== stage step $s $(date -u +%FT%TZ)"
  (cd "$TREE" && "$PY" "$IN/stage.py" --capture "$CAP" --step "$s" \
    --out "$OUT/values-step$s" --per-type "${PER_TYPE:-8}" --row "glm53-spot-${LOAD}-step$s" \
    --of "GLM-5.3 FP8 TP8 on 8x H200: step $s of load $LOAD's served window ($ARM), replayed from an empty engine on the same machine with CUDA graphs as served; its interior recorded by a graph-off forward of the same step whose outputs are compared bit for bit with the graph's (the outcome: stage.json's checks)" \
    --link "the replay's tokens against the served window's (OUT/replay/replay.json); the weights are the served engine's parameters, committed by SHA-512" \
    > "$OUT/stage-step$s.log" 2>&1) || { echo "spot: stage step $s failed"; staged=0; }
  tail -n 40 "$OUT/stage-step$s.log"
done
du -sh "$CAP" "$OUT"/* 2>/dev/null
if [ "${KEEP:-1}" = 1 ] || [ $staged = 0 ]; then echo "spot: capture kept at $CAP"; else rm -rf "$CAP"; fi
exit $rc
