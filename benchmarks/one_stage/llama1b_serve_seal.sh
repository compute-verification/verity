#!/usr/bin/env bash
# A LLAMA32_1B committing OpenAI endpoint whose Commit windows are sealed as deferred Commits' row dirs, for PROVE-PATH's served
# windows (`--serve-run DIR --window K`): `verity-vllm serve-openai --scheme rows-sha512 --seal` (the declared Commit's committer,
# hm96-sha512 rows in frame-v3-sha512 trees) on one leased RTX PRO 6000, a short varied load in bursts (each burst one window,
# closed at the engine's idle), `verity-vllm advice` per closed window under a registered cap (--cap), then each window's request
# Programs and manifest.json into its row dir (`tools/advice_compile.py RUN --window K`). The endpoint serves until UNTIL (UTC
# HH:MM) or `touch $D/stop`.
#   research run --on vy-nebius-1 --project verity --source TREE --cwd source --gpus 1 --mem-gb 64 --timeout 200m \
#     --env PORT=18084 --env UNTIL=13:30 -- bash benchmarks/one_stage/llama1b_serve_seal.sh
# READVICE=DIR (WINDOW=K, default 5) serves nothing: DIR's advice again under this tree's registered compiler, and window K's
# request Programs and manifest.json.
# Needs on the node: vLLM's venv (/workspace/jobs/venv312), the checkpoint in /workspace/hf and CUDA (the collector's JIT build).
# The serve dir D (row dirs, bundles) stays on the node under `research keep`; $OUT keeps the records and logs.
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
PORT=${PORT:-18084} UNTIL=${UNTIL:-13:30} BURSTS=${BURSTS:-8} PER_BURST=${PER_BURST:-4} CAP=${CAP:-8388608}
REPO=unsloth/Llama-3.2-1B REV=9535bd9b1d1dea6acafbdc4813b728796aeb28da
TREE=$PWD
ROOT=${SERVE_ROOT:-/workspace/research/vllm-serve-seal}
D=$ROOT/${RESEARCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$D" "$ROOT/cache"
export PY=/workspace/jobs/venv312/bin/python HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
export CUDA_HOME=/workspace/jobs/cuda-12.9 TORCH_EXTENSIONS_DIR=$ROOT/cache/torch-extensions
export PATH=/workspace/jobs/venv312/bin:$CUDA_HOME/bin:$PATH
export PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/research/src
export TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm VLLM_BATCH_INVARIANT=1 VERITY_RETAIN=host
export HIDDEN_GPU_BUILD=$ROOT/cache/jit/hidden_gpu NATIVE_COLLECT_BUILD=$ROOT/cache/jit/nc
rs() { "$PY" -m research "$@" 2>&1 || research "$@" 2>&1; }
{ id; hostname; git -C "$TREE" rev-parse HEAD 2>/dev/null; echo "serve dir $D port $PORT until $UNTIL"; nvidia-smi -L; } | tee "$OUT/machine.txt"
rs keep "$ROOT" --owner @vllm --expires 7d --priority high \
  --why "G-A (network accounting): LLAMA32_1B serve-openai windows sealed as rows-sha512 row dirs for PROVE-PATH" | tail -2
cd "$TREE/integrations/vllm"
vv() { "$PY" -m verity_vllm.pipeline.cli "$@"; }

if [ -n "${READVICE:-}" ]; then     # a served dir's advice again (every closed window), then WINDOW's Programs and manifest.json
  K=${WINDOW:-5}
  vv advice "$READVICE" --cap "$CAP" 2>&1 | tee "$OUT/advice.log"
  "$PY" tools/advice_compile.py "$READVICE" --window "$K" > "$OUT/compile_w$K.log" 2>&1
  echo "w$K compile rc=$?" | tee "$OUT/steps.txt"
  cp "$READVICE/advice/w$K/registration.json" "$READVICE/windows/w$K/manifest.json" "$OUT/"
  exit 0
fi

vv serve-openai --model "$REPO" --revision "$REV" --out "$D" --port "$PORT" --precision pin --run-id "${RESEARCH_RUN_ID:-local}" \
  --scheme rows-sha512 --seal --gpu-tree --window-steps 256 > "$OUT/serve.log" 2>&1 &
SRV=$!
echo "$SRV" > "$D/serve.pid"
for _ in $(seq 1 600); do
  grep -q "SERVE-OPENAI listening" "$OUT/serve.log" && break
  kill -0 "$SRV" 2>/dev/null || { tail -40 "$OUT/serve.log"; echo "SERVER EXITED before listening"; exit 10; }
  sleep 2
done
grep "SERVE-OPENAI listening" "$OUT/serve.log" || { tail -40 "$OUT/serve.log"; echo "SERVER NOT LISTENING"; kill "$SRV"; exit 11; }
cp "$D/serve.json" "$OUT/" 2>/dev/null

"$PY" tools/serve_load.py --port "$PORT" --model "$REPO" --bursts "$BURSTS" --per-burst "$PER_BURST" --out "$D/load.jsonl" \
  2>&1 | tee "$OUT/load.log"
sleep 5
vv advice "$D" --cap "$CAP" 2>&1 | tee "$OUT/advice.log"
cp "$D/windows.jsonl" "$D/requests.jsonl" "$D/advice/windows.json" "$OUT/" 2>/dev/null
WS=$("$PY" -c "import json,sys; print(' '.join(sorted(json.load(open('$D/advice/windows.json'))['windows'], key=int)))" 2>/dev/null)
FIRST=${WINDOW:-5}
for k in $FIRST $WS; do
  [ -d "$D/windows/w$k" ] || continue
  [ -f "$D/windows/w$k/manifest.json" ] && continue
  "$PY" tools/advice_compile.py "$D" --window "$k" > "$OUT/compile_w$k.log" 2>&1
  rc=$?
  if [ "$rc" = 2 ]; then      # compile_steps refused the window's advice: request by request instead, the refusals named
    "$PY" tools/advice_compile.py "$D" --window "$k" --partial > "$OUT/compile_w${k}_partial.log" 2>&1
    rc="2, --partial rc=$?"
  fi
  echo "$(date -u +%H:%M:%SZ) w$k compile rc=$rc" | tee -a "$OUT/steps.txt"
  "$PY" - "$D" "$k" <<'EOF' | tee -a "$OUT/row_dirs.txt"
import json, sys
from pathlib import Path
d, k = Path(sys.argv[1]), sys.argv[2]
w = d / "windows" / f"w{k}"
v = json.loads((w / "commit" / "verdict.json").read_text())
m = json.loads((w / "manifest.json").read_text()) if (w / "manifest.json").exists() else {}
print(json.dumps({"window": int(k), "row_dir": str(w), "bundle": str(w / "commit" / "replay_bundle_p0"), "run_root": v.get("run_root"),
                  "commit_scheme": v.get("commit_scheme"), "manifest_sha256": v["replay_deferred"]["bundles"][0]["manifest_sha256"],
                  "registration": str(d / "advice" / f"w{k}" / "registration.json"), "programs": len(m.get("requests") or {}),
                  "builds": sorted(p.name for p in w.glob("build_request*"))}))
EOF
done
echo "READY $D" | tee -a "$OUT/steps.txt"

end=$(date -u -d "today $UNTIL" +%s)
while [ "$(date -u +%s)" -lt "$end" ] && [ ! -e "$D/stop" ] && kill -0 "$SRV" 2>/dev/null; do sleep 30; done
kill -TERM "$SRV" 2>/dev/null; wait "$SRV"; rc=$?
cp "$D/summary.json" "$D/windows.jsonl" "$D/requests.jsonl" "$OUT/" 2>/dev/null
tail -3 "$OUT/serve.log"
exit $rc
