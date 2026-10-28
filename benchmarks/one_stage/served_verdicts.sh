#!/usr/bin/env bash
# One window of Llama-3.2-1B int8 served on an H200 through every subprotocol, each step's own
# verdict line in $OUT/verdicts.txt, and one record of every verdict. The chain, in order:
#  1. serve: `verity-vllm serve-openai` of LLAMA32_1B_INT8 (REPO at REV, --precision PRECISION: verity-int8/Llama-3.2-1B-INT8, verity_int8) on one GPU
#     under `gpu-lease`, `--scheme rows-sha512 --seal --admit release-order --window-requests SLOTS`, its committer built from
#     the H200 int8 row's manifest (ROW/manifest.json) and taps (HIDDEN_SO, NORM_TAP_SO), T tokens a step (--max-num-batched-tokens;
#     T defaults to H1_T, else 512, with or without H1). Each closed window is sealed as a
#     rows-sha512 row dir, and the seal opens each one's replay bundle (`deferred_rows.of_record`). Once it listens, before
#     the first request, the committer registers its weights (`prove_path.py weights-registration`: each weight parameter of
#     the checkpoint CKPT, REPO's snapshot at REV under HF_HOME, and the rotary table at the words the engine serves,
#     $D/private/cos_sin_cache.npy, once within the analytic table's bound, committed under salts drawn from the committer's
#     key $D/private/weights.key, 0600, neither of which a record copies); WEIGHTS says how many it roots and which it leaves unrooted, and
#     why, or NO WEIGHTS REGISTRATION why not, and then every weight read counts free.
#  2. requests: REQUESTS greedy requests (default SLOTS) from the certifier's traffic generator (`examples/vllm/serve_load.py --varied`:
#     prompts of PROMPTS tokens, at most MAX_TOKENS outputs), numbered under one release run and sent together (RATE a second),
#     so the engine's idle closes them into one window. With LOAD=fixed (default varied) they are one full window of fixed
#     length instead (`serve_load.py --fixed SLOTS --window-steps WINDOW_STEPS`): SLOTS requests of PROMPTS tokens, each
#     with ignore_eos and max_tokens WINDOW_STEPS, all prefilled at the window's first step and run through its last. The
#     lease ends once every closed window is sealed. Each request's
#     send, first-token and done times (the load's timings file, `serve_load.py`'s docstring) go to $OUT/timings/; each
#     committed step's end is $OUT/serve/step_trace.jsonl's t_end, on the clock of the timings' start_unix.
#  3. sampled proofs, on the window `served_verdicts.py window` picks (the most requests, none carried across windows): every
#     closed window's advice under the prove path's registered compiler and the weights registration (`--weights-of-record`),
#     then this window's request Programs (`tools/advice_compile.py`); V3b's window circuit (`prove_path.py window-circuit
#     --pins` the registration, and PIN's NAME=ROOT if set), its registered and free input bits at those
#     pins: at each class of CLASSES (default 4,32,4, V2's), each within WC_TIMEOUT (default 30m), all of which the run needs;
#     and at the class of record, a measurement within WC_RECORD_TIMEOUT (default 20m), run after the record (step 5),
#     since nothing reads it, whose timeout or refusal is said and never fails the run (unless CLASSES names `record`, which
#     then gates here instead); then the window audited from its records alone
#     (`prove_path/run.sh` with PP_RECORDS=1, `records_audit.py`): C-Flock `--zk` on the drawn proof blocks behind the Lean
#     firewall, and the Lean verifier's verdict. Those proof blocks are the sealed window's step units (PP_PARTITION, default
#     layer 0's SiluMul_v2 at step 1), not the window circuit's.
#  4. PoUW (H1 credit), the bandwidth audit, PoUS and the network certifier: `served_verdicts.py pieces` runs each one's piece
#     where POUW_CMD, BANDWIDTH_CMD, POUS_CMD or CERTIFIER_CMD names it (its public answer.json; its case in D/private),
#     one at a time, each within <COMPONENT>_TIMEOUT (else PIECE_TIMEOUT, default 60m), else answers it undecided, "not
#     run", and says NOT RUN with the command it lacks and its owner; a piece that refused to run (refused.txt) is "not
#     run" too, since only the protocol's answer rejects, and one that ran and wrote no answer errs, undecided. A piece
#     that needs a GPU takes its own `gpu-lease`, the serve's being over. POUS_CMD defaults to
#     'bash benchmarks/pous/served_piece.sh "$@"' where the tree has it and this node holds the memory challenger's key
#     (~/.ssh/vy-auditor2-pous), and BANDWIDTH_CMD to 'bash benchmarks/one_stage/served_bandwidth.sh "$1" "$2" "$3"' where
#     the tree has it and this node holds the bandwidth verifier's key (~/.ssh/vy-auditor2-bw) and its registered store
#     (BW_STORE's setup.json); set empty, not run. A subprotocol that runs on the serving engine (H1's
#     `--protocols pouw:...`) goes in SERVE_ARGS, serve-openai's further arguments, and its piece reads what it wrote in
#     the serve dir. The wipe's piece runs in step 1 instead: with WIPE_CMD set (by default
#     'bash benchmarks/pous/streamed_wipe/served_wipe.sh "$1" "$2" "$3"' where the tree has it; empty, none), serve()
#     calls it in TREE, under the serve's lease and before serve-openai starts, within WIPE_TIMEOUT (default 20m), with
#     (the serve dir D, `all`, $OUT/pieces/wipe), since the window isn't picked yet, and says its last line as WIPE. The
#     record reads its pieces/wipe/answer.json, and takes the wipe into the block record only when that answer accepts
#     and pieces/wipe/served.json names this node as the server it ran on. A wipe that wrote no answer (refused, or
#     failed before its judge) is undecided, one served elsewhere stays out of the block record, and one whose answer
#     rejects fails the run with no block record: a failed wipe is never accepted.
#  5. the record: `served_verdicts.py record`: the all-protocols window report ($OUT/record/verdicts.json, every row a
#     core answer) whose verdict is onsite's Lean verdict (`composite.py onsite`, the `onsite` executable) when the pieces'
#     cases (each in its private directory under D, else beside its public answer) make up onsite, else remote's
#     (`composite.py remote`), else undecided naming what each lacks; and the block record ($OUT/record/block-record.json)
#     unless the window or a subprotocol rejects; any other server's answer is carried as elsewhere. composite.py's
#     private output (the composed case, its evidence, every answer) stays in D/private/composite/wK; only its public
#     record, held to its form, goes to $OUT/record/composite. The run passes on a window that accepts or is undecided
#     with no subprotocol rejecting or erring.
#  6. V3c, after the record since V2 doesn't read it: it proves the window circuit's own units at V3C_CLASS (default 4,32,4,
#     which CLASSES must hold). `prove_path.py window-commit`, within V3C_TIMEOUT (default 15m), commits the window as that
#     class circuit's units, the weights it serves under the committer's key, so each served as registered has the
#     registration's root; its circuit.json less its run fields is held to the verifier's own window-circuit record at the
#     class; then `records_audit.py audit --circuit --registered`, within V3C_AUDIT_TIMEOUT (default 30m), draws the
#     registration's members (V3C_LAW, default subset:1) and proves them. `served_verdicts.sh v3c` runs only that step on an
#     existing serve dir (D and its window K set), the build first, its limits 120m and 180m.
# Coins: sampled proofs' draw is on the verifier's own live coins (Lean's IO.getRandomBytes, in-process) unless PP_CHALLENGER
# names the live challenger (run.sh's gate:USER@HOST with PP_PEER, PP_CHALLENGER_KEY and PP_GATE_KEY). With H1_GATE (the
# lane's gate command, by default "ssh -i $HOME/.ssh/vllm_gate_ed25519 -o BatchMode=yes research@204.12.178.84"; empty, no
# H1) the endpoint serves PoUW H1 (--protocols pouw:eps1-sm120-h1fx1 --challenger-gate H1_GATE --challenger-peer H1_PEER,
# default vy-dc-h200, at the engine's T and S WINDOW_STEPS 16, the Program build H1 affords), each window's coins from a session of
# its own on vy-challenger-1, once `served_verdicts.py gate` finds no period of the lane open (a reserve would join it).
# PoUW's piece (POUW_CMD's default then, `served_verdicts.py pouw`) checks the window's coins against the gate's session,
# ledger and pins under H1_KEY_SHA512 (vy-challenger-1's key) and main's first-parent line from MAIN_SHA (default the tree's
# HEAD). That key is a development key, so the run is a development run and credits nothing. The COINS line says which.
# With H1_CIRCUITS too (a directory), once the endpoint listens and before the requests,
# `pouw_h1_circuits --program D/h1 --out H1_CIRCUITS --live H1_T,WINDOW_STEPS --strata H1_STRATA` builds the circuit files of the Program serve-openai wrote (the live
# file, and only the strata H1_STRATA lists, default none; a file already there under the same bind is reused), then
# `served_verdicts.py period open` opens period H1_PERIOD (default served-RUN_ID) on the Program digest
# serve-openai wrote, its draw K H1_DRAW_K (default `pouw_h1_challenger.draw_k`'s), its weights H1_MODEL's registration
# (default none, which credit refuses), and its fixed length SLOTS requests of MAX_TOKENS; every window joins it, and
# `period close` closes it once they're sealed. The PERIOD lines say what it committed. H1_STATEMENTS and
# H1_LIVE_EVIDENCE (files) go to PoUW's piece as its --statements and --live-evidence, the evidence core's credit reads.
# A step that can't run says why, and the chain stops there. C-Flock, Lean's flock-verify and sampled-proofs-verifier and the
# tree's venv are built while the endpoint serves, C-Flock in FLOCK_WORK (kept on the node, so a rerun reuses its build).
#   research run --on vy-dc-h200 --project verity --source TREE --cwd clone --timeout 6h -- \
#     bash benchmarks/one_stage/served_verdicts.sh
# `served_verdicts.sh circuits K`, in TREE with D and OUT, counts serve dir D's window K's circuits alone (its advice written).
# Defaults are vy-dc-h200's: the int8 B4 row and taps under $H, vLLM's venv /workspace/cp/venv312, CUDA 13.0.
# The serve dir D (row dirs, advice, the window circuits' records) stays on the node under `research keep`; $OUT keeps the
# public records and every log.
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
TREE=$PWD
H=/workspace/research/vllm-composed-h200int8
ROW=${ROW:-$H/r20261026-185144-3ccb/llama32-1b__int8__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager}
HIDDEN_SO=${HIDDEN_SO:-$H/fa2/build/fa3_matReqG/verity_fa3_matReqG.so} NORM_TAP_SO=${NORM_TAP_SO:-$H/norm_tap/build/verity_norm_tap.so}
VENV=${VENV:-/workspace/cp/venv312}
REPO=${REPO:-verity-int8/Llama-3.2-1B-INT8} REV=${REV:-15b0976d400d796ee589e7737d03a919526bf800} PRECISION=${PRECISION:-verity_int8}
PORT=${PORT:-18190} SLOTS=${SLOTS:-4} PROMPTS=${PROMPTS:-16:24} MAX_TOKENS=${MAX_TOKENS:-4} RATE=${RATE:-1000} SEED=${SEED:-1}
REQUESTS=${REQUESTS:-$SLOTS} CLASSES=${CLASSES:-4,32,4} CAP=${CAP:-8388608} MAX_MODEL_LEN=${MAX_MODEL_LEN:-512}
T=${T:-${H1_T:-512}}
DISK_FLOOR_GB=${DISK_FLOOR_GB:-100}
H1_KEY_SHA512=${H1_KEY_SHA512:-27271eaf917d1a817198cbcfe7684debd9bdb11b9e386460948bb981642fab3ae065c0a877330a0da68ae76969550eb786dc172319876aaca21fd116e5a193bc}
H1_GATE=${H1_GATE-ssh -i $HOME/.ssh/vllm_gate_ed25519 -o BatchMode=yes research@204.12.178.84} H1_PEER=${H1_PEER:-vy-dc-h200}
if [ -n "$H1_GATE" ]; then WINDOW_STEPS=${WINDOW_STEPS:-16}; else WINDOW_STEPS=${WINDOW_STEPS:-256}; fi
MAIN_SHA=${MAIN_SHA-$(git -C "$TREE" rev-parse HEAD 2>/dev/null)}
# shellcheck disable=SC2016
[ -f "$TREE/benchmarks/pous/streamed_wipe/served_wipe.sh" ] \
  && WIPE_CMD=${WIPE_CMD-'bash benchmarks/pous/streamed_wipe/served_wipe.sh "$1" "$2" "$3"'}
# shellcheck disable=SC2016
[ -f "$TREE/benchmarks/pous/served_piece.sh" ] && [ -f "$HOME/.ssh/vy-auditor2-pous" ] \
  && POUS_CMD=${POUS_CMD-'bash benchmarks/pous/served_piece.sh "$@"'}
# shellcheck disable=SC2016
[ -f "$TREE/benchmarks/one_stage/served_bandwidth.sh" ] && [ -f "$HOME/.ssh/vy-auditor2-bw" ] \
  && [ -f "${BW_STORE:-/workspace/research/pous-store/mem69-v2}/setup.json" ] \
  && BANDWIDTH_CMD=${BANDWIDTH_CMD-'bash benchmarks/one_stage/served_bandwidth.sh "$1" "$2" "$3"'}
export H1_KEY_SHA512 H1_GATE H1_PEER WINDOW_STEPS MAIN_SHA WIPE_CMD POUS_CMD BANDWIDTH_CMD
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
case ${LOAD:=varied} in varied | fixed) ;; *) say "PREFLIGHT: LOAD is varied or fixed, not $LOAD"; exit 3 ;; esac
CKPT=${CKPT:-$HF_HOME/hub/models--${REPO//\//--}/snapshots/$REV} WREG=$D/weights_registration.json
NO_WEIGHTS="NO WEIGHTS REGISTRATION: no window registers a weight's root, so every weight read counts free"
weights() {  # under the lease (its commit hashes on the leased GPU), before the first request: the committer's registration
  (cd "$TREE" && "$PY" benchmarks/prove_path/prove_path.py weights-registration --serve-run "$D" --checkpoint-dir "$CKPT" \
    --key "$D/private/weights.key" --out "$WREG") > "$OUT/weights_registration.txt" 2>&1
  local rc=$? log=$OUT/weights_registration.txt
  if [ "$rc" != 0 ] || [ ! -f "$WREG" ]; then
    say "$NO_WEIGHTS (weights-registration exit $rc: $({ grep -m1 'prove_path.py weights-registration:' "$log" || tail -1 "$log"; } | cut -c1-400))"
    return
  fi
  cp "$WREG" "$OUT/"
  say "WEIGHTS $("$PY" - "$WREG" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
v, u = r["values"], r["unrooted"]
print(f"registered before the first request: {len(v)} weight parameters rooted ({sum(e['words'] * e['leaves'] for e in v.values()) * 16} "
      f"bits) of {r['model']['repo']}@{r['model']['revision'][:10]} under the committer's key (kept off every record); "
      f"{len(u)} unrooted" + "".join(f"; {p}: {why}" for p, why in sorted(u.items())))
EOF
)"
}
V3C=${V3C_CLASS:-4,32,4} v3c_ok=0; vtag=${V3C//,/-}
if [ "${1:-}" = v3c ]; then V3C_TIMEOUT=${V3C_TIMEOUT:-120m} V3C_AUDIT_TIMEOUT=${V3C_AUDIT_TIMEOUT:-180m}
else V3C_TIMEOUT=${V3C_TIMEOUT:-15m} V3C_AUDIT_TIMEOUT=${V3C_AUDIT_TIMEOUT:-30m}; fi
v3c_commit() {  # window K of $D committed as V3C's class circuit's units, its circuit held to the verifier's window-circuit record
  VW=$D/v3c-w$K-$vtag
  [ -f "$D/window-circuit-w$K-$vtag.json" ] || { say "WINDOW-COMMIT class $V3C: no window-circuit record at it (CLASSES '$CLASSES')"; return 1; }
  [ -f "$WREG" ] || { say "WINDOW-COMMIT class $V3C: $NO_WEIGHTS"; return 1; }
  # off the GPU (it isn't leased here): the weights hashed by h1.registered's NumPy path, the reference
  (cd "$TREE" && CUDA_VISIBLE_DEVICES='' timeout -k 1m "$V3C_TIMEOUT" "$PY" benchmarks/prove_path/prove_path.py window-commit --serve-run "$D" \
    --window "$K" --row-dir "$D/windows/w$K" --advice "$D/advice/w$K/private/advice.bin" --class "$V3C" \
    --key "$D/private/weights.key" --registration "$WREG" --checkpoint-dir "$CKPT" --out "$VW/window") \
    > "$OUT/window-commit-$vtag.json" 2> "$OUT/window-commit-$vtag.err"
  local rc=$?
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
    say "WINDOW-COMMIT class $V3C timed out after V3C_TIMEOUT=$V3C_TIMEOUT (exit $rc)"
    return 1
  elif [ "$rc" != 0 ]; then
    say "WINDOW-COMMIT class $V3C refused (exit $rc): $(grep -v '^ ' "$OUT/window-commit-$vtag.err" | tail -2 | cut -c1-800)"
    return 1
  fi
  say "WINDOW-COMMIT class $V3C: $(cut -c1-1500 "$OUT/window-commit-$vtag.json")"
  "$PY" - "$D/window-circuit-w$K-$vtag.json" "$VW/window/circuit.json" "$V3C" <<'EOF' 2>&1 | tee -a "$OUT/verdicts.txt"
import json, sys
from verity.ml.windows import interface as IF
mine, theirs = (json.load(open(f)) for f in sys.argv[1:3])
ok = IF.circuit_of(theirs) == mine
print(f"WINDOW-CIRCUIT-HELD class {sys.argv[3]}: the committed circuit.json less its run fields is the verifier's own "
      f"window-circuit record: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
EOF
  [ "${PIPESTATUS[0]}" = 0 ] && v3c_ok=1
}
v3c_audit() {  # the committed window audited from its records alone: its circuit and run root the verifier's, the weights pinned
  local run leanb m0 rc
  run=$("$PY" -c 'import json, sys; print(next(json.loads(l)["run_root"] for l in open(sys.argv[1]) if l.strip() and json.loads(l)["window"] == int(sys.argv[2])))' \
    "$D/windows.jsonl" "$K") || { say "V3C-AUDIT: no run root for window $K in windows.jsonl"; return 1; }
  leanb=$TREE/verity/core/service/flock/.lake/build/bin
  m0=$TREE:$TREE/benchmarks/numerical/python:$TREE/verity/ml/flock/python:$TREE/integrations/vllm
  mkdir -p "$OUT/v3c-audit"
  (cd "$TREE" && env -u PYTHONPATH RAYON_NUM_THREADS="$(nproc)" RUST_MIN_STACK=$((64 << 20)) timeout -k 1m "$V3C_AUDIT_TIMEOUT" \
    "$TREE/.venv/bin/python" benchmarks/prove_path/records_audit.py audit --window "$VW/window" --law "${V3C_LAW:-subset:1}" \
    --circuit "$(sha512sum "$VW/window/circuit.json" | cut -c1-128)" --run "$run" --registered "$WREG" \
    --m0-pythonpath "$m0" --bin "$FW/flock-gpu-link/flock/target/release/flock-circuit" --lean "$leanb/flock-verify" \
    --firewall "$leanb/flock-firewall" --out "$VW/audit") > "$OUT/v3c-audit/audit.log" 2> "$OUT/v3c-audit/audit.err"
  rc=$?
  cp "$VW/audit/summary.json" "$OUT/v3c-audit/" 2>/dev/null
  say "V3C-AUDIT class $V3C (records_audit.py exit $rc): $(grep -m1 '^RECORDS' "$OUT/v3c-audit/audit.log" | cut -c1-3000)"
  [ -n "$(grep -m1 '^RECORDS' "$OUT/v3c-audit/audit.log")" ] \
    || say "V3C-AUDIT: no RECORDS line; $(tail -3 "$OUT/v3c-audit/audit.err" | tr '\n' ' ' | cut -c1-800)"
  return $rc
}

serve() {  # under the lease: the endpoint, the requests, every closed window sealed, the endpoint stopped
  export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
  mkdir -p "$D" "$OUT/serve"
  cd "$TREE/integrations/vllm" || return 9
  local h1=()
  [ -n "$H1_GATE" ] && h1=(--protocols pouw:eps1-sm120-h1fx1 --challenger-gate "$H1_GATE"
                           --challenger-peer "$H1_PEER")
  [ -n "$H1_GATE" ] && [ -n "${H1_CIRCUITS:-}" ] && h1+=(--h1-prover "$D/private/h1")
  if [ -n "${WIPE_CMD:-}" ]; then  # the wipe of this card, before its engine loads
    mkdir -p "$OUT/pieces/wipe"
    (cd "$TREE" && timeout -k 1m "${WIPE_TIMEOUT:-20m}" bash -c "$WIPE_CMD" _ "$D" all "$OUT/pieces/wipe") \
      > "$OUT/pieces/wipe/piece.log" 2>&1
    say "WIPE (exit $?, within WIPE_TIMEOUT=${WIPE_TIMEOUT:-20m}): $(tail -1 "$OUT/pieces/wipe/piece.log" | cut -c1-600)"
  fi
  # shellcheck disable=SC2086
  "$PY" -m verity_vllm.pipeline.cli serve-openai --model "$REPO" --revision "$REV" --out "$D" --port "$PORT" \
    --precision "$PRECISION" --run-id "${RESEARCH_RUN_ID:-local}" --scheme rows-sha512 --seal --gpu-tree \
    --max-num-batched-tokens "$T" --window-steps "$WINDOW_STEPS" "${h1[@]}" \
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
  weights
  local period=0
  if [ -n "${H1_GATE:-}" ] && [ -n "${H1_CIRCUITS:-}" ]; then  # before the first window reserves, at its first engine step
    (cd "$TREE/integrations/vllm" && PYTHONPATH=$PYTHONPATH:$TREE/verity/ml/flock/python "$PY" -m tests.protocols.pouw_h1_circuits \
      --program "$D/h1" --out "$H1_CIRCUITS" --live "$T,$WINDOW_STEPS" --strata "${H1_STRATA:-none}") \
      > "$OUT/h1-circuits.log" 2>&1 || say "H1-CIRCUITS refused (exit $?): $(tail -1 "$OUT/h1-circuits.log" | cut -c1-400)"
    # shellcheck disable=SC2086
    say "$("$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" period open "$D" --gate "$H1_GATE" \
      --name "${H1_PERIOD:-served-${RESEARCH_RUN_ID:-local}}" --circuits "$H1_CIRCUITS" --slots "$SLOTS" \
      --max-tokens "$MAX_TOKENS" ${H1_DRAW_K:+--draw-k "$H1_DRAW_K"} ${H1_MODEL:+--model "$H1_MODEL"} 2>&1 | tail -1)"
    [ -f "$D/period.json" ] && period=1
  fi
  if [ -n "${CERTIFIER_FRONT:-}" ]; then  # the network certifier's warden in front of the endpoint, before the requests
    # below: its group closes into the run's earliest window, which `served_verdicts.py window` picks on a tie in requests
    (cd "$TREE" && timeout -k 1m 20m bash -c "$CERTIFIER_FRONT" _ "$PORT" "$D" "$OUT/certifier-front") \
      > "$OUT/certifier-front.log" 2>&1
    say "CERTIFIER-FRONT (exit $?): $(tail -1 "$OUT/certifier-front.log" | cut -c1-600)"
  fi
  release=$("$PY" -c 'import uuid; print(uuid.uuid4().hex)')
  local load=(--varied "$REQUESTS" --rate "$RATE" --max-tokens "$MAX_TOKENS")
  [ "$LOAD" = fixed ] && load=(--fixed "$SLOTS" --window-steps "$WINDOW_STEPS")
  "$PY" "$TREE/examples/vllm/serve_load.py" --port "$PORT" --model "$REPO" "${load[@]}" --prompt-range "$PROMPTS" \
    --release "$release" --seed "$SEED" --out "$D/load.jsonl" > "$OUT/load.log" 2>&1
  rc=$?
  say "REQUESTS (exit $rc, release run $release): $(tail -1 "$OUT/load.log" | cut -c1-300)"
  mkdir -p "$OUT/timings" && cp "$D/load.timings.json" "$OUT/timings/" || say "TIMINGS: no $D/load.timings.json"
  for _ in $(seq 1 450); do     # every closed window's seal line
    c=$(count "$D/windows.jsonl"); s=$(count "$D/seals.jsonl")
    [ "$c" -gt 0 ] && [ "$s" -ge "$c" ] && break
    sleep 2
  done
  trap - TERM INT
  kill -TERM "$srv" 2>/dev/null; wait "$srv"
  say "SERVE stopped (exit $?): $c window(s) closed, $s sealed"
  [ "$period" = 1 ] && say "$("$PY" "$TREE/benchmarks/one_stage/served_verdicts.py" period close "$D" --gate "$H1_GATE" 2>&1 | tail -1)"
  [ "$rc" = 0 ] && [ "$c" -gt 0 ] && [ "$s" -ge "$c" ]
}

circuit() {  # window K's circuit at class $1 within $2 (NAME=LIMIT), its line headed $3: 0 iff it counted
  local c=$1 head=$3 cls=() tag=record pins=() wp=$NO_WEIGHTS f rc
  [ "$c" = record ] || { cls=(--class "$c"); tag=${c//,/-}; }
  [ -f "$WREG" ] && pins=(--pins "$WREG") wp="at the weights registration's pins"
  f=$D/window-circuit-w$K-$tag.json
  timeout "${2#*=}" "$PY" benchmarks/prove_path/prove_path.py window-circuit --serve-run "$D" --window "$K" \
    --row-dir "$D/windows/w$K" --advice "$D/advice/w$K/private/advice.bin" --out "$f" "${pins[@]}" ${PIN:+--pin "$PIN"} \
    "${cls[@]}" > "$OUT/window-circuit-$tag.json" 2> "$OUT/window-circuit-$tag.err"
  rc=$?
  if [ "$rc" = 0 ]; then
    say "$head: $("$PY" -c 'import json, sys; c = json.load(open(sys.argv[1]))["count"]; print("registered", c["registered"], "bits, free", c["free"], "bits of", c["inputs"], "input bits")' \
      "$OUT/window-circuit-$tag.json" 2>&1 | cut -c1-200) $wp; $(cat "$OUT/window-circuit-$tag.json"); record $(sha256sum "$f" | cut -c1-16)… $(stat -c %s "$f") bytes"
  elif [ "$rc" = 124 ]; then
    say "$head timed out after $2 (exit 124): no count"
  else
    say "$head refused (exit $rc): $({ grep -m1 'window-circuit' "$OUT/window-circuit-$tag.err" \
      || tail -1 "$OUT/window-circuit-$tag.err"; } | cut -c1-600)"
  fi
  return "$rc"
}
circuits() {  # in TREE: every class of CLASSES gates (0 iff each counted)
  local bad=0 c
  grep -q '"window-circuit"' benchmarks/prove_path/prove_path.py \
    || { say "MISSING V3b's window circuit: this tree's prove_path.py has no window-circuit"; return 1; }
  for c in $CLASSES; do circuit "$c" "WC_TIMEOUT=${WC_TIMEOUT:-30m}" "WINDOW-CIRCUIT class $c" || bad=1; done
  return "$bad"
}
measure() {  # in TREE: the class of record, unless CLASSES gates on it; it gates nothing, and nothing after it reads it
  grep -q '"window-circuit"' benchmarks/prove_path/prove_path.py || return 0
  case " $CLASSES " in
    *" record "*) ;;
    *) circuit record "WC_RECORD_TIMEOUT=${WC_RECORD_TIMEOUT:-20m}" "WINDOW-CIRCUIT class record (measurement)";;
  esac
  return 0
}

if [ "${1:-}" = serve ]; then serve; exit $?; fi
if [ "${1:-}" = circuits ]; then K=${2:?served_verdicts.sh circuits WINDOW}; circuits; rc=$?; measure; exit "$rc"; fi

stop() { say "STOP at $1 (exit $2)"; exit "$2"; }
{ id -un; hostname; git -C "$TREE" rev-parse HEAD 2>/dev/null; nproc; free -g | head -2; nvidia-smi -L; echo "serve dir $D"; } \
  | tee "$OUT/machine.txt"
for f in "$ROW/manifest.json" "$HIDDEN_SO" "$NORM_TAP_SO" "$PY"; do [ -e "$f" ] || { say "PREFLIGHT: no $f"; exit 3; }; done
command -v gpu-lease >/dev/null || { say "PREFLIGHT: no gpu-lease on this node"; exit 3; }
mkdir -p "$D" "$ROOT/cache" "$FW"
KEEP_WHY="served_verdicts.sh's serve dirs (sealed row dirs, advice, window circuits) and C-Flock's build"
{ "$PY" -m research keep "$ROOT" --owner @vllm --expires 7d --priority high --why "$KEEP_WHY" \
  || research keep "$ROOT" --owner @vllm --expires 7d --priority high --why "$KEEP_WHY"; } 2>&1 | tail -2

build() {  # what prove_path/run.sh builds, here while the endpoint serves; run.sh's own builds then find them done
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
  uv sync --all-packages --extra torch-cpu || return 5
  (cd verity/core/service/flock && lake build flock-verify flock-firewall) || return 7
  (cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) || return 7
  (cd verity/core/protocols/onsite && lake build onsite) || return 7
  (cd verity/core/protocols/remote && lake build remote) || return 7
  mkdir -p "$OUT/m0-build"
  # FW is the node's, every run's: two builds in it at once break each other's crates
  RESEARCH_RUN_DIR=$OUT/m0-build FLOCK_WORK=$FW MODE=build GPU=0 flock "$FW/.build.lock" bash verity/ml/flock/pod/60-circuit.sh || return 6
}
( unset PYTHONPATH; build ) > "$OUT/build.log" 2>&1 &
BUILD=$!

if [ "${1:-}" = v3c ]; then  # an existing serve dir's window: D and K set, its advice written (step 3); V3c alone
  [ -n "${K:-}" ] && [ -f "$D/advice/w$K/private/advice.bin" ] || { say "V3C: set D and K, a serve dir and its window with advice"; exit 3; }
  [ -f "$WREG" ] || CUDA_VISIBLE_DEVICES='' weights
  [ -f "$WREG" ] || exit 13
  # the verifier's record at the class from this tree, whatever an earlier run left in D
  CUDA_VISIBLE_DEVICES='' circuit "$V3C" "WC_TIMEOUT=${WC_TIMEOUT:-30m}" "WINDOW-CIRCUIT class $V3C" || exit 12
  v3c_commit
  wait "$BUILD"; rc=$?
  [ "$rc" = 0 ] || { tail -30 "$OUT/build.log"; stop build "$rc"; }
  [ "$v3c_ok" = 1 ] || exit 16
  v3c_audit; exit $?
fi

H1_COINS="PoUW H1: not served (H1_GATE empty)"
if [ -n "$H1_GATE" ]; then
  if [ -n "$MAIN_SHA" ]; then
    git -C "$TREE" rev-list --first-parent "$MAIN_SHA" > "$OUT/main.txt" || { say "PREFLIGHT: MAIN_SHA $MAIN_SHA not in this tree"; exit 3; }
    export MAIN_FILE=$OUT/main.txt
  fi
  "$PY" benchmarks/one_stage/served_verdicts.py gate --gate "$H1_GATE" ${MAIN_FILE:+--main "$MAIN_FILE"} 2>&1 \
    | tee -a "$OUT/verdicts.txt"
  [ "${PIPESTATUS[0]}" = 0 ] || stop gate 3
  H1_COINS="PoUW H1: each window's coins from a session of its own on vy-challenger-1's gate (peer $H1_PEER, T $T, S $WINDOW_STEPS), a development run under the lane's development key: it credits nothing"
  export POUW_CMD=${POUW_CMD:-"$TREE/.venv/bin/python benchmarks/one_stage/served_verdicts.py pouw \$1 \$2 \$3${H1_STATEMENTS:+ --statements $H1_STATEMENTS}${H1_LIVE_EVIDENCE:+ --live-evidence $H1_LIVE_EVIDENCE}"}
fi
if [ -n "${PP_CHALLENGER:-}" ]; then
  say "COINS sampled proofs: the live challenger $PP_CHALLENGER (peer ${PP_PEER:-?}); $H1_COINS"
else
  say "COINS sampled proofs: the verifier's own live coins (Lean's IO.getRandomBytes, in-process); $H1_COINS"
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
[ -d "$D/h1" ] && (cd "$D" && find h1 -name pouw-h1-calls.json -exec cp --parents {} "$OUT/serve/" \;)

# 3. sampled proofs: the window, its advice and Programs, V3b's window circuit, then C-Flock --zk and the Lean verifier
PICK=$("$PY" benchmarks/one_stage/served_verdicts.py window "$D" --slots "$SLOTS" ${WINDOW:+--window "$WINDOW"} 2> "$OUT/window.err")
[ -n "$PICK" ] || { say "WINDOW refused: $(cut -c1-600 "$OUT/window.err")"; stop window 12; }
K=$(echo "$PICK" | "$PY" -c 'import json, sys; print(json.load(sys.stdin)["window"])')
say "WINDOW $PICK"
cd "$TREE/integrations/vllm" || exit 9
CA=$(cd "$TREE" && "$PY" benchmarks/prove_path/prove_path.py compiler --serve-run "$D" --cap-bits "$CAP" \
  | "$PY" -c "import json, sys; c = json.load(sys.stdin)['inner']; print('--compiler-name', c['name'], '--compiler-digest', c['digest'])")
[ -n "$CA" ] || stop compiler 13
echo "registered compiler: $CA" > "$OUT/compiler.txt"
WA=()
[ -f "$WREG" ] && WA=(--weights-of-record "$WREG")
for k in $("$PY" -c "import json,sys; [print(json.loads(l)['window']) for l in open(sys.argv[1]) if l.strip()]" "$D/windows.jsonl"); do
  # shellcheck disable=SC2086
  vv advice "$D" --window "$k" --cap "$CAP" $CA "${WA[@]}" --out "$D/advice/.w$k" >> "$OUT/advice.log" 2>&1
  [ -d "$D/advice/.w$k/w$k" ] && mv "$D/advice/.w$k/w$k" "$D/advice/w$k"
  [ "$k" = "$K" ] && break
done
[ -f "$D/advice/w$K/registration.json" ] || { tail -20 "$OUT/advice.log"; say "ADVICE w$K: none written"; stop advice 14; }
cp "$D/advice/w$K/registration.json" "$OUT/registration-w$K.json"
"$PY" tools/advice_compile.py "$D" --window "$K" > "$OUT/compile_w$K.log" 2>&1
rc=$?
say "ADVICE w$K: $("$PY" -c 'import json, sys; w = json.load(open(sys.argv[1]))["windows"][sys.argv[2]]; print("root", w["root"][:16] + "…", "bits", w["bits"])' \
  "$D/advice/.w$K/windows.json" "$K" 2>&1 | cut -c1-200); $("$PY" -c 'import json, sys; print(len(json.load(open(sys.argv[1])).get("registered_roots") or {}))' \
  "$D/advice/w$K/registration.json" 2>&1 | cut -c1-200) weight roots recorded (${WA[*]:-$NO_WEIGHTS}); compile (advice_compile.py) exit $rc: $(tail -1 "$OUT/compile_w$K.log" | cut -c1-300)"
[ "$rc" = 0 ] || stop compile 15
cd "$TREE" || exit 9
wc_ok=0
circuits && wc_ok=1
wait "$BUILD"; rc=$?
[ "$rc" = 0 ] || { tail -30 "$OUT/build.log"; stop build "$rc"; }
env -u PYTHONPATH PP_SERVE_RUN="$D" PP_WINDOW="$K" PP_SERVE_ROWS="$D/windows/w$K" PP_RECORDS=1 PP_ONLY="${PP_ONLY:-honest}" \
  PP_VLLM_PYTHON="$PY" PP_FLOCK_WORK="$FW" OUT="$OUT/sampled-proofs" bash benchmarks/prove_path/run.sh > "$OUT/sampled-proofs.log" 2>&1
rc=$?
say "SAMPLED-PROOFS (run.sh exit $rc): $(grep -m1 '^RECORDS' "$OUT/sampled-proofs.log" | cut -c1-3000)"
[ -n "$(grep -m1 '^RECORDS' "$OUT/sampled-proofs.log")" ] || say "SAMPLED-PROOFS: no RECORDS line; $(grep -v '^+' "$OUT/sampled-proofs.log" | tail -3 | tr '\n' ' ' | cut -c1-600)"

# 4. the other subprotocols, each its piece's verdict, or undecided and NOT RUN with the command it lacks
TPY=$TREE/.venv/bin/python
"$TPY" benchmarks/one_stage/served_verdicts.py pieces --serve-run "$D" --window "$K" --out "$OUT/pieces" 2>&1 | tee -a "$OUT/verdicts.txt"

# 5. one record of every verdict, and the window's verdict: onsite's or remote's Lean answer
"$TPY" benchmarks/one_stage/served_verdicts.py record --serve-run "$D" --window "$K" --sampled-proofs "$OUT/sampled-proofs/served" \
  --pieces "$OUT/pieces" --out "$OUT/record" 2>&1 | tee -a "$OUT/verdicts.txt"
rec=${PIPESTATUS[0]}
measure
# V3c after the record, each step within its limit: it gates nothing
v3c_commit && v3c_audit
echo "---- verdicts"; cat "$OUT/verdicts.txt"
[ "$wc_ok" = 1 ] && [ "$rec" = 0 ] && grep '^RECORD ' "$OUT/verdicts.txt" | grep -q '"verdict": "\(undecided\|accept\)"'
