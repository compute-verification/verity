#!/usr/bin/env bash
# The Wed 28 run (VLLM-33): one window of Llama-3.2-1B int8 served on an H200 through every subprotocol, each step's own
# verdict line in $OUT/verdicts.txt, and one record of every verdict. The chain, in order:
#  1. serve: `verity-vllm serve-openai` of LLAMA32_1B_INT8 (verity-int8/Llama-3.2-1B-INT8, --precision verity_int8) on one GPU
#     under `gpu-lease`, `--scheme rows-sha512 --seal --admit release-order --window-requests SLOTS`, its committer built from
#     the H200 int8 row's manifest (ROW/manifest.json) and taps (HIDDEN_SO, NORM_TAP_SO). Each closed window is sealed as a
#     rows-sha512 row dir, and SERVED-SEAL opens each one's replay bundle (`deferred_rows.of_record`).
#  2. requests: REQUESTS greedy requests (default SLOTS) from the certifier's traffic generator (`tools/serve_load.py --varied`:
#     prompts of PROMPTS tokens, at most MAX_TOKENS outputs), numbered under one release run and sent together (RATE a second),
#     so the engine's idle closes them into one window. The lease ends once every closed window is sealed.
#  3. sampled proofs, on the window `served_verdicts.py window` picks (the most requests, none carried across windows): every
#     closed window's advice under PROVE-PATH's registered compiler, then this window's request Programs
#     (`tools/advice_compile.py`); V3b's window circuit at each class of CLASSES (`prove_path.py window-circuit --pin PIN`,
#     default `weights`; #4317 with #4757; `record` is the class of record); then the window audited from its records alone
#     (`prove_path/run.sh` with PP_RECORDS=1, `records_audit.py`): C-Flock `--zk` on the drawn proof blocks behind the Lean
#     firewall, and the Lean verifier's verdict. Those proof blocks are the sealed window's step units (PP_PARTITION, default
#     layer 0's SiluMul_v2 at step 1), not the window circuit's: proving the window circuit's own units is V3c (VLLM-103),
#     named missing until it lands.
#  4. PoUW (H1 credit), the bandwidth audit, PoUS and the network certifier: `served_verdicts.py pieces` runs each one's piece
#     where POUW_CMD, BANDWIDTH_CMD, POUS_CMD or CERTIFIER_CMD names it, else names it MISSING with its issue. No verdict is
#     made for a piece that didn't run. A subprotocol that runs on the serving engine (H1's `--protocols pouw:...`, the
#     bandwidth session's `--remote-verifier HOST:PORT`) goes in SERVE_ARGS, serve-openai's further arguments, and its piece
#     reads what it wrote in the serve dir.
#  5. the record: `served_verdicts.py record`: the all-protocols window report ($OUT/record/verdicts.json) and CMP-153's block
#     record ($OUT/record/block-record.json), composed `pending` while any subprotocol is missing.
# Coins: sampled proofs' draw is on the verifier's own live coins (Lean's IO.getRandomBytes, in-process) unless PP_CHALLENGER
# names the live challenger (run.sh's gate:USER@HOST with PP_PEER, PP_CHALLENGER_KEY and PP_GATE_KEY); PoUW H1's are
# in-process until vy-challenger-1's gate takes this node's key (~/.ssh/vllm_gate_ed25519; INF-263). The COINS line says which.
# A step that can't run says why, and the chain stops there. C-Flock, Lean's flock-verify and sampled-proofs-verifier and the
# tree's venv are built while the endpoint serves, C-Flock in FLOCK_WORK (kept on the node, so a rerun reuses its build).
#   research run --on vy-dc-h200 --project verity --source TREE --cwd clone --timeout 4h --env CLASSES='4,32,4 record' -- \
#     bash benchmarks/one_stage/served_verdicts.sh
# Defaults are vy-dc-h200's: the int8 B4 row and taps of r20261026-185144-3ccb, vLLM's venv /workspace/cp/venv312, CUDA 13.0.
# The serve dir D (row dirs, advice, the window circuits' records) stays on the node under `research keep`; $OUT keeps the
# public records and every log.
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
TREE=$PWD
H=/workspace/research/vllm-composed-h200int8
ROW=${ROW:-$H/r20261026-185144-3ccb/llama32-1b__int8__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager}
HIDDEN_SO=${HIDDEN_SO:-$H/fa2/build/fa3_matReqG/verity_fa3_matReqG.so} NORM_TAP_SO=${NORM_TAP_SO:-$H/norm_tap/build/verity_norm_tap.so}
VENV=${VENV:-/workspace/cp/venv312}
REPO=verity-int8/Llama-3.2-1B-INT8 REV=15b0976d400d796ee589e7737d03a919526bf800
PORT=${PORT:-18190} SLOTS=${SLOTS:-4} PROMPTS=${PROMPTS:-16:24} MAX_TOKENS=${MAX_TOKENS:-4} RATE=${RATE:-1000} SEED=${SEED:-1}
REQUESTS=${REQUESTS:-$SLOTS} CLASSES=${CLASSES:-record} CAP=${CAP:-8388608} MAX_MODEL_LEN=${MAX_MODEL_LEN:-512}
DISK_FLOOR_GB=${DISK_FLOOR_GB:-100}
ROOT=${SERVE_ROOT:-/workspace/research/vllm-served-verdicts}
D=${D:-$ROOT/${RESEARCH_RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}}
FW=${FLOCK_WORK:-$ROOT/flock-work}
export PY=$VENV/bin/python HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/research/src
export TORCH_EXTENSIONS_DIR=$ROOT/cache/torch-extensions TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm
export VLLM_BATCH_INVARIANT=1 VERITY_RETAIN=host HIDDEN_GPU_BUILD=$ROOT/cache/jit/hidden_gpu NATIVE_COLLECT_BUILD=$ROOT/cache/jit/nc
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
say() { echo "$*" | tee -a "$OUT/verdicts.txt"; }
count() { cat "$1" 2>/dev/null | grep -c .; }
vv() { "$PY" -m verity_vllm.pipeline.cli "$@"; }

serve() {  # under the lease: the endpoint, the requests, every closed window sealed, the endpoint stopped
  export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
  mkdir -p "$D" "$OUT/serve"
  cd "$TREE/integrations/vllm" || return 9
  # shellcheck disable=SC2086
  "$PY" -m verity_vllm.pipeline.cli serve-openai --model "$REPO" --revision "$REV" --out "$D" --port "$PORT" \
    --precision verity_int8 --run-id "${RESEARCH_RUN_ID:-local}" --scheme rows-sha512 --seal --gpu-tree --window-steps 256 \
    --required-manifest "$ROW/manifest.json" --hidden-so "$HIDDEN_SO" --norm-tap-so "$NORM_TAP_SO" --tap-cap-mb 2048 \
    --admit release-order --window-requests "$SLOTS" --max-model-len "$MAX_MODEL_LEN" --window-gb 32 \
    --disk-floor-gb "$DISK_FLOOR_GB" --seal-hold-gb 32 --seal-warm-gb 0 --copy-records-to "$OUT/serve" ${SERVE_ARGS:-} \
    > "$OUT/serve.log" 2>&1 &
  local srv=$! up=0 c=0 s=0 rc release
  trap 'kill -TERM "$srv" 2>/dev/null; wait "$srv"; exit 143' TERM INT
  for _ in $(seq 1 900); do
    grep -q "SERVE-OPENAI listening" "$OUT/serve.log" && { up=1; break; }
    kill -0 "$srv" 2>/dev/null || break
    sleep 2
  done
  if [ "$up" != 1 ]; then
    tail -40 "$OUT/serve.log"
    say "SERVE: not listening: $(grep -m1 -E 'UsageError|Error|refus' "$OUT/serve.log" | cut -c1-400)"
    kill -TERM "$srv" 2>/dev/null; wait "$srv"; return 10
  fi
  say "SERVE $(grep -m1 'SERVE-OPENAI listening' "$OUT/serve.log" | cut -c1-300)"
  release=$("$PY" -c 'import uuid; print(uuid.uuid4().hex)')
  "$PY" tools/serve_load.py --port "$PORT" --model "$REPO" --varied "$REQUESTS" --rate "$RATE" --prompt-range "$PROMPTS" \
    --max-tokens "$MAX_TOKENS" --release "$release" --seed "$SEED" --out "$D/load.jsonl" > "$OUT/load.log" 2>&1
  rc=$?
  say "REQUESTS (exit $rc, release run $release): $(tail -1 "$OUT/load.log" | cut -c1-300)"
  for _ in $(seq 1 450); do     # every closed window's seal line
    c=$(count "$D/windows.jsonl"); s=$(count "$D/seals.jsonl")
    [ "$c" -gt 0 ] && [ "$s" -ge "$c" ] && break
    sleep 2
  done
  trap - TERM INT
  kill -TERM "$srv" 2>/dev/null; wait "$srv"
  say "SERVE stopped (exit $?): $c window(s) closed, $s sealed"
  [ "$rc" = 0 ] && [ "$c" -gt 0 ] && [ "$s" -ge "$c" ]
}

if [ "${1:-}" = serve ]; then serve; exit $?; fi

stop() { say "STOP at $1 (exit $2)"; exit "$2"; }
{ id -un; hostname; git -C "$TREE" rev-parse HEAD 2>/dev/null; nproc; free -g | head -2; nvidia-smi -L; echo "serve dir $D"; } \
  | tee "$OUT/machine.txt"
for f in "$ROW/manifest.json" "$HIDDEN_SO" "$NORM_TAP_SO" "$PY"; do [ -e "$f" ] || { say "PREFLIGHT: no $f"; exit 3; }; done
command -v gpu-lease >/dev/null || { say "PREFLIGHT: no gpu-lease on this node"; exit 3; }
mkdir -p "$D" "$ROOT/cache" "$FW"
KEEP_WHY="VLLM-33 Wed 28: served_verdicts.sh's serve dirs (sealed row dirs, advice, window circuits) and C-Flock's build"
{ "$PY" -m research keep "$ROOT" --owner @vllm --expires 7d --priority high --why "$KEEP_WHY" \
  || research keep "$ROOT" --owner @vllm --expires 7d --priority high --why "$KEEP_WHY"; } 2>&1 | tail -2

build() {  # what prove_path/run.sh builds, here while the endpoint serves; run.sh's own builds then find them done
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
  uv sync --all-packages --extra torch-cpu || return 5
  (cd verity/core/service/flock && lake build flock-verify flock-firewall) || return 7
  (cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) || return 7
  mkdir -p "$OUT/m0-build"
  RESEARCH_RUN_DIR=$OUT/m0-build FLOCK_WORK=$FW MODE=build GPU=0 bash verity/ml/flock/pod/60-circuit.sh || return 6
}
( unset PYTHONPATH; build ) > "$OUT/build.log" 2>&1 &
BUILD=$!

H1_COINS="PoUW H1: in-process until vy-challenger-1's gate takes this node's key (INF-263; Daniel's card, due Tue 27 Oct 15:00Z)"
if [ -n "${PP_CHALLENGER:-}" ]; then
  say "COINS sampled proofs: the live challenger $PP_CHALLENGER (peer ${PP_PEER:-?}); $H1_COINS"
else
  say "COINS sampled proofs: the verifier's own live coins (Lean's IO.getRandomBytes, in-process); $H1_COINS; the bandwidth" \
      "verifier on vy-auditor-2 (10.96.0.36:7952) is INF-263, due Tue 27 Oct 18:00Z"
fi

# 1-2. the endpoint and its requests, under the GPU lease
export D OUT
GPU_LEASE_WHO=${GPU_LEASE_WHO:-vllm-composed} gpu-lease 1 --wait --max-min "${LEASE_MIN:-60}" --mem-gb "${LEASE_MEM_GB:-128}" \
  -- bash "$TREE/benchmarks/one_stage/served_verdicts.sh" serve
rc=$?
[ "$rc" = 0 ] || stop serve "$rc"
"$PY" - "$D" <<'EOF' 2>&1 | tee -a "$OUT/verdicts.txt"
import json, sys
from pathlib import Path
from verity_vllm.pipeline import deferred_rows as DR
d = Path(sys.argv[1])
serve = json.loads((d / "serve.json").read_text())
seals = [json.loads(x) for x in (d / "seals.jsonl").read_text().splitlines() if x.strip()] if (d / "seals.jsonl").exists() else []
ok = bool(seals) and serve.get("scheme") == "rows-sha512"
for s in seals:
    k = s["window"]
    if "error" in s:
        print(f"WINDOW w{k}: SEAL FAILED {s['error']}"[:600])
        ok = False
        continue
    w = d / "windows" / f"w{k}"
    v = json.loads((w / "commit" / "verdict.json").read_text())
    c = json.loads((w / "commit" / "runs.jsonl").read_text().splitlines()[0])["commit"]
    try:
        r = DR.of_record(v["replay_deferred"]["bundles"][0]["path"], str(w))
        opened, store, good = f"VERIFIED (manifest sha256 {r.bundle['manifest_sha256'][:16]}, {len(r.bundle['manifest']['files'])} files)", r.commitment, True
    except Exception as e:  # noqa: BLE001
        opened, store, good = f"REFUSED ({type(e).__name__}: {e})"[:400], None, False
    rr = v.get("run_root") or ""
    good = good and v.get("commit_scheme") == c.get("commit_scheme") == store == "rows-sha512" and len(rr) == 128
    ok &= good
    print(f"WINDOW w{k}: commit_scheme {v.get('commit_scheme')}, committer {c.get('committer')}, run root {rr[:16]}… "
          f"({len(rr) // 2} bytes), {len(s.get('requests') or [])} requests; bundle {opened}: {'PASS' if good else 'FAIL'}")
print(f"SERVED-SEAL VERDICT: {'PASS' if ok else 'FAIL'} ({len(seals)} sealed window(s), scheme {serve.get('scheme')}, "
      f"precision {serve.get('precision')})")
sys.exit(0 if ok else 1)
EOF
[ "${PIPESTATUS[0]}" = 0 ] || stop seal 11
for f in serve.json windows.jsonl seals.jsonl requests.jsonl load.jsonl; do cp "$D/$f" "$OUT/serve/" 2>/dev/null; done

# 3. sampled proofs: the window, its advice and Programs, V3b's window circuit, then C-Flock --zk and the Lean verifier
PICK=$("$PY" benchmarks/one_stage/served_verdicts.py window "$D" --slots "$SLOTS" ${WINDOW:+--window "$WINDOW"} 2> "$OUT/window.err")
[ -n "$PICK" ] || { say "WINDOW refused: $(cut -c1-600 "$OUT/window.err")"; stop window 12; }
K=$(echo "$PICK" | "$PY" -c 'import json, sys; print(json.load(sys.stdin)["window"])')
say "WINDOW $PICK"
cd "$TREE/integrations/vllm" || exit 9
CA=$(cd "$TREE" && "$PY" examples/prove_path/prove_path.py compiler --serve-run "$D" --cap-bits "$CAP" \
  | "$PY" -c "import json, sys; c = json.load(sys.stdin)['inner']; print('--compiler-name', c['name'], '--compiler-digest', c['digest'])")
[ -n "$CA" ] || stop compiler 13
echo "registered compiler: $CA" > "$OUT/compiler.txt"
for k in $("$PY" -c "import json,sys; [print(json.loads(l)['window']) for l in open(sys.argv[1]) if l.strip()]" "$D/windows.jsonl"); do
  # shellcheck disable=SC2086
  vv advice "$D" --window "$k" --cap "$CAP" $CA --out "$D/advice/.w$k" >> "$OUT/advice.log" 2>&1
  [ -d "$D/advice/.w$k/w$k" ] && mv "$D/advice/.w$k/w$k" "$D/advice/w$k"
  [ "$k" = "$K" ] && break
done
[ -f "$D/advice/w$K/registration.json" ] || { tail -20 "$OUT/advice.log"; say "ADVICE w$K: none written"; stop advice 14; }
cp "$D/advice/w$K/registration.json" "$OUT/registration-w$K.json"
"$PY" tools/advice_compile.py "$D" --window "$K" > "$OUT/compile_w$K.log" 2>&1
rc=$?
say "ADVICE w$K: $("$PY" -c 'import json, sys; w = json.load(open(sys.argv[1]))["windows"][sys.argv[2]]; print("root", w["root"][:16] + "…", "bits", w["bits"])' \
  "$D/advice/.w$K/windows.json" "$K" 2>&1 | cut -c1-200); compile (advice_compile.py) exit $rc: $(tail -1 "$OUT/compile_w$K.log" | cut -c1-300)"
[ "$rc" = 0 ] || stop compile 15
cd "$TREE" || exit 9
wc_ok=0
for c in $CLASSES; do
  if ! grep -q '"window-circuit"' examples/prove_path/prove_path.py; then
    say "MISSING V3b's window circuit (#4317 with #4757, VLLM-103): this tree's prove_path.py has no window-circuit"; break
  fi
  cls=(); tag=record; [ "$c" = record ] || { cls=(--class "$c"); tag=${c//,/-}; }
  f=$D/window-circuit-w$K-$tag.json
  timeout "${WC_TIMEOUT:-30m}" "$PY" examples/prove_path/prove_path.py window-circuit --serve-run "$D" --window "$K" \
    --row-dir "$D/windows/w$K" --advice "$D/advice/w$K/private/advice.bin" --out "$f" --pin "${PIN:-weights}" "${cls[@]}" \
    > "$OUT/window-circuit-$tag.json" 2> "$OUT/window-circuit-$tag.err"
  rc=$?
  if [ "$rc" = 0 ]; then
    wc_ok=1
    say "WINDOW-CIRCUIT class $c: $(cat "$OUT/window-circuit-$tag.json"); record $(sha256sum "$f" | cut -c1-16)… $(stat -c %s "$f") bytes"
  else
    say "WINDOW-CIRCUIT class $c refused (exit $rc): $(grep -m1 'window-circuit' "$OUT/window-circuit-$tag.err" | cut -c1-600)"
  fi
done
say "MISSING V3c (VLLM-103, vllm-stack-neutral): the committer writing the window circuit's units as the record's members," \
    "so its own units are drawn and proved; the proof blocks below are the sealed window's step units"
wait "$BUILD"; rc=$?
[ "$rc" = 0 ] || { tail -30 "$OUT/build.log"; stop build "$rc"; }
env -u PYTHONPATH PP_SERVE_RUN="$D" PP_WINDOW="$K" PP_SERVE_ROWS="$D/windows/w$K" PP_RECORDS=1 PP_ONLY="${PP_ONLY:-honest}" \
  PP_VLLM_PYTHON="$PY" PP_FLOCK_WORK="$FW" OUT="$OUT/sampled-proofs" bash examples/prove_path/run.sh > "$OUT/sampled-proofs.log" 2>&1
rc=$?
say "SAMPLED-PROOFS (run.sh exit $rc): $(grep -m1 '^RECORDS' "$OUT/sampled-proofs.log" | cut -c1-3000)"
[ -n "$(grep -m1 '^RECORDS' "$OUT/sampled-proofs.log")" ] || say "SAMPLED-PROOFS: no RECORDS line; $(grep -v '^+' "$OUT/sampled-proofs.log" | tail -3 | tr '\n' ' ' | cut -c1-600)"

# 4. the other subprotocols, each its piece's verdict or MISSING with its issue
TPY=$TREE/.venv/bin/python
"$TPY" benchmarks/one_stage/served_verdicts.py pieces --serve-run "$D" --window "$K" --out "$OUT/pieces" 2>&1 | tee -a "$OUT/verdicts.txt"

# 5. one record of every verdict
"$TPY" benchmarks/one_stage/served_verdicts.py record --serve-run "$D" --window "$K" --sampled-proofs "$OUT/sampled-proofs/served" \
  --pieces "$OUT/pieces" --out "$OUT/record" 2>&1 | tee -a "$OUT/verdicts.txt"
rec=${PIPESTATUS[0]}
echo "---- verdicts"; cat "$OUT/verdicts.txt"
[ "$wc_ok" = 1 ] && [ "$rec" = 0 ] && grep -q '"composed": "\(pending\|accept\)"' "$OUT/verdicts.txt"
