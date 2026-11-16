#!/usr/bin/env bash
# The serving half of `verity bench compute`: Llama 3.2 1B served under Pearl+ on one H100 or H200, every pass retained,
# then Pearl+'s verdict on the passes in Python and in Lean, written as the record `run.py` reads:
#
#   bash benchmarks/compute/serve.sh                      on a machine with the card, from any directory
#   research run --on NODE --project verity --cwd clone --timeout 2h -- bash benchmarks/compute/serve.sh
#
# Steps, each one's log in $OUT/logs/STEP.log and its exit status a line of $OUT/steps.txt:
#   1. cubin: Pearl-C's sm_90a cubin built from this tree (benchmarks/harness/setup/pearl_c_sm90.sh: nvcc, its SASS gates);
#   2. serve: `served_passes.py serve` of the spec's Pearl-C arm (SPEC) on one card: the caller's gpu-lease's where serve.sh
#      runs under one, else under `gpu-lease 1` where the machine has it (GPU picks the card, default 0), else directly: with no warm-up, CLASS's fixed load, SLOTS requests of
#      PROMPT seeded random tokens and OUTPUTS outputs each, every engine step one pass retained under PASSES;
#   3. retain: Pearl+'s window record W over the timed call's passes (`served_passes.py retain --window-record`), which
#      are the class's rounds alone (a warm-up's would be rounds before them, which the class's fixed load refuses);
#   4. python-verify: Pearl+'s verdict in Python (`served_passes.py verify --tie W --class CLASS`), the passes held to the class's
#      fixed load, a sample of DRAWS work-weighted tiles at DELTA, on fast_tile's batch forms (FAST, default on);
#   5. lean: the Lean verdict (`served_passes.py lean`, pouw-pearlplus built here unless PEARLPLUS_LEAN_EXE names one),
#      written only where it is the Python's;
#   6. policy: the inference-only policy at CLASS, from this card's serve-phi cut at Pearl+'s linears
#      (`benchmarks/one_stage/inference_only.py policy`), which names the class and the model `run.py` counts MACs at;
#   7. circuit: the class's circuit record cut at Pearl+'s linears (`prove_path.py class-circuit`), into out/record/.
#   8. verdicts: record/verdicts.json, PoUW's verdict: `missing`, as the window's compose (which composes it of record)
#      isn't run here, so the credit is the Pearl+ piece's alone;
#   9. publish and piece, after the verdicts: the verifier's coin into pieces/pouw/ (coin.json, coin-put.json: put in
#      no store) and PoUW's development piece (`served_verdicts.py pearlplus`: development.json, its coin_source);
#  10. registered: out/registered/circuit.json and log.jsonl (W alone), what training reads; no program.json, as the
#      tied path writes no Program;
#  11. share: `benchmarks/compute/run.py --from RUN_DIR`, which prints the share from the record this wrote.
# Coins (COINS): `local` (default), a local challenger: infra's gate code (AUDITOR) on a scratch auditor host under
# PASSES with an LMS test key (W 4, H 6), each session's challenger its own serve.py on 127.0.0.1, whose coins are the
# Lean spot-verifier's (`benchmarks/one_stage/pearlplus_local_e2e.py init|gate|stop`): the salt coin a salt-only session's
# and the sample a withheld sample session's on the run's commitment, so the Lean verdict is taken of record. `in-process`:
# the salt coin the in-process challenger's and the sample the node's own (`verify_run.node_draw`), the Lean's verdict
# a development one. Both are development parameters, which $OUT/compute-run.json and the COINS line say.
# Writes under OUT, RUN_DIR/out: pieces/pouw/pearlplus-answer.json,
# pearlplus-lean.json, pearlplus-lean-check.json, verify.json, served.json and coin.json; inference-only-policy.json;
# record/circuit-record.json, record/window-record.json and record/verdicts.json; registered/; auditor/ (the local gate's ledger,
# pins, public key and sessions' public files); compute-run.json and coins.txt. The record writes no Program: the tied
# path's window record W stands in its place. The passes (about 2 GB) stay outside OUT, as they hold the run's activations
# and salt key: in a directory under the run's own scratch under research run, else under SCRATCH or $TMPDIR, removed
# on exit with the local gate's key; PASSES names a directory to keep them in instead.
# Env: VERITY_BENCH_VENV (the pinned-vLLM Python's venv, else benchmarks/harness/env.sh's search), HF_HOME, CUDA_HOME,
# RUN_DIR (the run directory, default $RESEARCH_RUN_DIR, else ./compute-run), COINS (local), GPU (0), DRAWS (459),
# DELTA (1/100), FAST (1; empty: the scalar reference alone), JOBS (nproc), LEASE_MIN (30), LEAN_TIMEOUT (the seconds
# the Lean verdict has: 3,600 on 96 or more cores, 3,600 x 96 / nproc on fewer). The load is
# benchmarks/compute/load.toml's, which run.py holds the record to. Before any step it exits 3 on a RUN_DIR already
# holding a record, a missing verity_torch or lake, or a GPU that isn't an idle H100 or H200. A failed step exits 1 with
# a sentence naming it and compute-run.json written, and run.py isn't run on what it left. Exit 0 only where Python and
# Lean both accept and run.py prints the share.
set -euo pipefail
# the one path under verity/ this names, which a layout move may change
AUDITOR=verity/protocols/compose/verifier/live_challenger/auditor.py
cd "$(dirname "$0")/../.." || exit 3
TREE=$PWD
RUN=${RUN_DIR:-${RESEARCH_RUN_DIR:-$TREE/compute-run}}
OUT=$RUN/out
COINS=${COINS:-local} GPU=${GPU-0} DRAWS=${DRAWS:-459} DELTA=${DELTA:-1/100} FAST=${FAST-1} JOBS=${JOBS:-$(nproc)}
# the Lean verdict runs on every core it may: 1,917 s on 96, 3,451 s on 36, and more than 3,600 s on 20, so its limit grows
# as the cores fall
CORES=$(nproc) && LEAN_TIMEOUT=${LEAN_TIMEOUT:-$(( CORES >= 96 ? 3600 : 3600 * 96 / CORES ))}
# under a caller's gpu-lease (which exports GPU_LEASE_UUID, the cards it holds) the card is the lease's, and serve.sh takes
# no lease of its own: gpu-lease isn't re-entrant, so a second lease of the held card would wait forever
[ -z "${GPU_LEASE_UUID:-}" ] || GPU=${GPU_LEASE_UUID%%,*}
SCHEME=pearl-c-h100-v1-h3
SPEC=benchmarks/harness/specs/llama32-1b-1gpu-pouw-protocol-served-h200.toml
SP=benchmarks/pouw/pearl_c_vllm/served_passes.py
PIECE=$OUT/pieces/pouw REC=$OUT/record
# a run's share is read from the files it wrote alone, never an earlier run's left in the same place
for f in steps.txt compute-run.json logs pieces record registered; do
  [ ! -e "$OUT/$f" ] || { echo "$OUT already holds a run's record ($OUT/$f): give RUN_DIR a new directory, or remove" \
    "$RUN, so this run's share can't be read from an earlier run's files" >&2; exit 3; }
done
mkdir -p "$OUT/logs" "$PIECE" "$REC" || exit 3
# shellcheck source=/dev/null
. benchmarks/harness/env.sh
PY=$VENV/bin/python
# the benchmark's load (benchmarks/compute/load.toml): the model, its revision, and the class, SLOTS requests of PROMPT
# prompt tokens and OUTPUTS output tokens each
read -r REPO REV SLOTS PROMPT OUTPUTS < <("$PY" -c 'import sys, tomllib
d = tomllib.load(open(sys.argv[1], "rb"))
print(d["model"], d["revision"], d["slots"], d["prompt"], d["outputs"])' benchmarks/compute/load.toml) && [ -n "$OUTPUTS" ] \
  || { echo "benchmarks/compute/load.toml doesn't read as the benchmark's load" >&2; exit 3; }
CLASS=$SLOTS,$PROMPT,$OUTPUTS
export PATH=$HOME/.elan/bin:$PATH HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
# what the serve and the Lean steps need from the machine, checked before any step
"$PY" -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("verity_torch") is None)' || {
  echo "verity_torch isn't importable by $PY, and the served model's PoUW hooks import it: put this tree's adapters/torch" \
       "on PYTHONPATH (export PYTHONPATH=$TREE/adapters/torch:\$PYTHONPATH), or run under research run, which does" >&2
  exit 3; }
command -v lake > /dev/null || {
  echo "lake isn't on PATH (nor in $HOME/.elan/bin), and the Lean verifiers build with it: install elan" \
       "(https://github.com/leanprover/elan), which fetches the toolchain each Lean package's lean-toolchain file names," \
       "or put its bin directory on PATH" >&2
  exit 3; }
# the card: Pearl-C's cubin is sm_90a's, so an H100 or H200 (compute capability 9.0), and idle, before the build
command -v nvidia-smi > /dev/null || { echo "nvidia-smi isn't on PATH: serve.sh serves on one H100 or H200 (GPU, default 0)" \
  "and needs the NVIDIA driver" >&2; exit 3; }
card=$(nvidia-smi ${GPU:+-i "$GPU"} --query-gpu=index,name,compute_cap --format=csv,noheader 2> /dev/null | grep ', 9\.0$' \
  | head -1) || card=
[ -n "$card" ] || { echo "no H100 or H200 (compute capability 9.0) at GPU=${GPU:-any} (nvidia-smi -L lists the cards):" \
  "Pearl-C's cubin is sm_90a's; set GPU to such a card's index" >&2; exit 3; }
busy=$(nvidia-smi -i "${card%%,*}" --query-compute-apps=pid --format=csv,noheader 2> /dev/null | grep -c .) || busy=0
[ "$busy" = 0 ] || { echo "card ${card%%,*} ($card) runs $busy compute process(es) (nvidia-smi -i ${card%%,*}): serve.sh" \
  "serves alone on its card; wait for them to end or set GPU to an idle H100 or H200" >&2; exit 3; }
# the passes' directory: under research run the run's own scratch (`--cwd clone`'s, the tree's parent), else SCRATCH or
# TMPDIR; removed on exit with the challengers stopped, unless PASSES names it
here=$(dirname "$TREE")
[ -n "${RESEARCH_RUN_ID:-}" ] && [ "$(basename "$here")" = "$RESEARCH_RUN_ID" ] || here=${SCRATCH:-${TMPDIR:-/tmp}}
W=${PASSES:-$(mktemp -d -p "$here" pearlplus-compute.XXXXXX)}
P=$W/passes WR=$W/window-record.json
mkdir -p "$W/tmp" && chmod 700 "$W" || exit 3
# every temporary file of every step under W too, so the run leaves nothing in /tmp
export TMPDIR=$W/tmp
umask 077
failed=0
#: each long step's minutes in the measured run (one H200, 96 cores), printed as it starts
declare -A ABOUT=([cubin]=2 [gate]=5 [serve]=1 [python-verify]=15 [lean]=32 [policy]=2 [circuit]=2)
step() {  # NAME CMD...: CMD's output to its log, its status a line of steps.txt
  local name=$1 t rc
  shift
  [ -z "${ABOUT[$name]:-}" ] || echo "$name: started $(date -u +%H:%MZ), about ${ABOUT[$name]} min in the measured run on" \
    "96 cores (longer on fewer); its log is $OUT/logs/$name.log"
  t=$(date +%s)
  "$@" > "$OUT/logs/$name.log" 2>&1
  rc=$?
  printf '%s rc=%s %ss: %s\n' "$name" "$rc" "$(( $(date +%s) - t ))" "$(tail -1 "$OUT/logs/$name.log" | cut -c1-400)" \
    | tee -a "$OUT/steps.txt"
  return "$rc"
}
lease() {  # CMD...: on one card: the caller's lease's, else under gpu-lease where the machine has it, else GPU
  if [ -n "${GPU_LEASE_UUID:-}" ]; then
    "$@"
  elif command -v gpu-lease > /dev/null; then
    GPU_LEASE_WHO=${GPU_LEASE_WHO:-compute-serve} gpu-lease 1 --wait ${GPU:+--on "$GPU"} --max-min "${LEASE_MIN:-30}" -- "$@"
  else
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU} "$@"
  fi
}
cubin() (  # a subshell: the setup script exits on a failed gate, which must end this step, not serve.sh
  # shellcheck source=/dev/null
  . benchmarks/harness/setup/pearl_c_sm90.sh && printf '%s\n' "$PEARLC_KERNEL" "$PEARLC_CUBIN" > "$W/setup.paths"
)
fail() {  # NAME: a step serve.sh can't go on without failed; the run's statement written, no share read
  echo "serve.sh: $1 failed, so this run has no share (its log is $OUT/logs/$1.log; $OUT/steps.txt has each step)" >&2
  coins > "$OUT/coins.txt" || true
  exit 1
}
coins() {  # the coins and parameters the run used, as a line and as compute-run.json
  "$PY" - "$OUT/compute-run.json" "$COINS" "$CLASS" "$DRAWS" "$DELTA" "$SCHEME" "$REPO" "$REV" \
    "$(git -C "$TREE" rev-parse HEAD 2> /dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}")" "$OUT/auditor-gate.json" \
    "$AUDITOR" "$PIECE/development.json" <<'EOF'
import json, os, sys
out, mode, cls, draws, delta, scheme, repo, rev, commit, gate, auditor, piece = sys.argv[1:13]
if mode == "local":
    g = json.load(open(gate)) if os.path.exists(gate) else {}
    w, h = g.get("params") or (None, None)
    held = {"challenger": "local", "gate": f"{auditor} on a scratch host "
            "(benchmarks/one_stage/pearlplus_local_e2e.py), each session's challenger serve.py on 127.0.0.1",
            "key": f"LMS test key (W {w}, H {h}), made for this run", "key_sha512": g.get("key_sha512"),
            "lane": g.get("lane"), "salt_coin_from": "a salt-only session's (live)",
            "sample_coin": "a withheld sample session's on the run's commitment (live)", "lean": "of record (--of-record)"}
    line = f"local challenger (scratch auditor gate, serve.py on 127.0.0.1, LMS test key {w},{h}, Lean spot-verifier's coins)"
else:
    held = {"challenger": "in-process", "salt_coin_from": "in-process challenger",
            "sample_coin": "node draw (verify_run.node_draw)", "lean": "development (no --of-record)"}
    line = "in-process challenger (salt) and the node itself (sample)"
# the salt coin as the piece states it (development.json's coin_source), null until the piece is written
held["salt_coin"] = json.load(open(piece)).get("coin_source") if os.path.exists(piece) else None
steps = {}
if os.path.exists(os.path.join(os.path.dirname(out), "steps.txt")):
    for x in open(os.path.join(os.path.dirname(out), "steps.txt")):
        name, _, rest = x.partition(" rc=")
        rc, _, sec = rest.partition(" ")
        if sec.split("s:")[0].isdigit():
            steps[name] = {"rc": int(rc), "seconds": int(sec.split("s:")[0])}
doc = {"schema": "verity-bench/compute-run/v1", **held, "development": True, "draws": int(draws), "delta": delta,
       "over_cap": "uncredited", "class": cls, "scheme": scheme, "model": repo, "revision": rev, "commit": commit,
       "steps": steps, "seconds": sum(v["seconds"] for v in steps.values())}
open(out, "w").write(json.dumps(doc, indent=1) + "\n")
print(f"COINS {line}; parameters: development (sample size {draws} at delta {delta}, tiles over the inert-work bound "
      f"uncredited, {scheme}, class "
      f"{cls}), {repo}@{rev[:10]}, commit {commit[:12]}")
EOF
}
gate_up() {  # the local challenger's scratch auditor host, its gate command and key into auditor-gate.json
  "$PY" benchmarks/one_stage/pearlplus_local_e2e.py init --root "$W/auditor" && tail -1 > "$OUT/auditor-gate.json" \
    < "$OUT/logs/gate.log"
}
gate_down() {
  [ -d "$W/auditor" ] && "$PY" benchmarks/one_stage/pearlplus_local_e2e.py stop --root "$W/auditor" --out "$OUT" \
    > "$OUT/logs/gate-stop.log" 2>&1
  [ -s "$P/coin.json" ] && cp "$P/coin.json" "$PIECE/coin.json"
  return 0
}
verdicts() {  # PoUW's verdict as run.py reads it: missing, the window's compose not run here
  "$PY" - "$REC/verdicts.json" "$PIECE/pearlplus-answer.json" "$OUT/compute-run.json" <<'EOF'
import json, os, sys
out, answer, run = sys.argv[1:4]
a = json.load(open(answer)) if os.path.exists(answer) else {}
row = {"verdict": "missing", "owner": "compute-accounting",
       "why": "PoUW's row of record is the window's compose (V3c), not run here: the credit is the Pearl+ piece's alone "
              f"(its verdict {a.get('verdict')}, development coins and parameters)"}
doc = {"format": "verity-bench/compute-verdicts/v1", "development": True, "run": json.load(open(run)),
       "protocols": {"work": row}}
open(out, "w").write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
print(f"work: {row['verdict']}: {row['why']}")
EOF
}
publish() {  # the verifier's coin published in OUT alone, put in no store: coin.json and coin-put.json
  cp "$P/coin.json" "$PIECE/coin.json" && "$PY" - "$PIECE/coin.json" "$PIECE/coin-put.json" "${RESEARCH_RUN_ID:-}" <<'EOF'
import hashlib, json, sys
path, out, run = sys.argv[1:4]
raw = open(path, "rb").read()
rec = json.loads(raw)
put = {"sha512": hashlib.sha512(raw).hexdigest(), "commitment": rec["commitment"], "source": rec["source"],
       "draw": (rec.get("draw") or {}).get("from"), "run": run or None, "art": None, "preserved": False,
       "error": "not put in the evidence store: a local run's coin, published in its out/ alone", "custody": None}
open(out, "w").write(json.dumps(put, indent=1, sort_keys=True) + "\n")
print(f"coin {put['sha512'][:16]}… on {put['commitment'][:16]}…, its sample {put['draw']}, in out/ alone")
EOF
}
piece() {  # PoUW's development piece (served_verdicts.py pearlplus): development.json, its coin_source among it
  local g=() key
  if [ "$COINS" = local ]; then
    "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["pin"])' "$OUT/auditor-gate.json" > "$OUT/main.txt" \
      || return 1
    key=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["key_sha512"])' "$OUT/auditor-gate.json") || return 1
    g=(--gate "$PEARLC_CHALLENGER_GATE" --main "$OUT/main.txt" --key-sha512 "$key")
  fi
  "$PY" benchmarks/one_stage/served_verdicts.py pearlplus "$PIECE" --served "$PIECE/served.json" \
    --verify "$PIECE/verify.json" --coin "$PIECE/coin-put.json" --tie "$WR" ${g[@]+"${g[@]}"}
}
registered() {  # what training reads the run registered: the circuit record and the registration log, W alone
  mkdir -p "$OUT/registered" && cp "$REC/circuit-record.json" "$OUT/registered/circuit.json" \
    && "$PY" -c 'import json, sys; print(json.dumps(json.load(open(sys.argv[1])), separators=(",", ":"), sort_keys=True))' \
      "$WR" > "$OUT/registered/log.jsonl" \
    && echo "registered: circuit.json and log.jsonl (W alone); no program.json: the tied path writes no Program"
}
cleanup() {
  gate_down
  [ -n "${PASSES:-}" ] || rm -rf "$W"
}
trap cleanup EXIT
echo "passes: $P" | tee -a "$OUT/steps.txt"

step cubin cubin || fail cubin
{ read -r PEARLC_KERNEL; read -r PEARLC_CUBIN; } < "$W/setup.paths"
export PEARLC_KERNEL PEARLC_CUBIN PEARLC_BF16_ROWS=${PEARLC_BF16_ROWS:-0}
unset PEARLC_CHALLENGER_GATE PEARLC_CHALLENGER_PEER PEARLC_CHALLENGER_LANE VLLM_BATCH_INVARIANT
case $COINS in
  local)
    [ -f "$AUDITOR" ] || { echo "no $AUDITOR in this tree: set AUDITOR at the top of serve.sh to the gate's path" >&2; exit 3; }
    step gate gate_up || fail gate
    { read -r PEARLC_CHALLENGER_GATE; read -r PEARLC_CHALLENGER_PEER; read -r PEARLC_CHALLENGER_LANE; } < <("$PY" -c '
import json, sys
g = json.load(open(sys.argv[1]))
print(g["gate"], g["peer"], g["lane"], sep="\n")' "$OUT/auditor-gate.json") || fail gate
    export PEARLC_CHALLENGER_GATE PEARLC_CHALLENGER_PEER PEARLC_CHALLENGER_LANE ;;
  in-process) ;;
  *) echo "COINS is local or in-process, not $COINS" >&2; exit 3 ;;
esac
coins | tee "$OUT/coins.txt"
step serve lease "$PY" "$SP" serve --spec "$SPEC" --requests "$SLOTS" --prompt-tokens "$PROMPT" \
  --output-tokens "$OUTPUTS" --max-num-batched-tokens "${T:-8192}" --no-warm-up --passes "$P" --out "$PIECE/served.json" || failed=1
if [ "$failed" = 0 ]; then
  step volunteer-needed "$PY" "$SP" volunteer-needed || fail volunteer-needed
  grep -q '^no' "$OUT/logs/volunteer-needed.log" || {
    echo "serve.sh: the verdict needs a give-up before its coin, and this runs none" >&2; fail volunteer-needed; }
  step retain "$PY" "$SP" retain --passes "$P" --window-record "$WR" || fail retain
  cp "$WR" "$REC/window-record.json"
  step python-verify "$PY" "$SP" verify --passes "$P" --tie "$WR" --class "$CLASS" --out "$PIECE/verify.json" \
    --answer "$PIECE/pearlplus-answer.json" --draws "$DRAWS" --delta "$DELTA" --jobs "$JOBS" ${FAST:+--fast} || failed=1
  [ -s "$PIECE/verify.json" ] && step lean "$PY" "$SP" lean --passes "$P" --verify "$PIECE/verify.json" \
    --exe "${PEARLPLUS_LEAN_EXE:-build}" --out "$PIECE/pearlplus-lean.json" --timeout "$LEAN_TIMEOUT" || failed=1
  if [ -s "$P/coin.json" ]; then step publish publish && step piece piece || failed=1
  else echo "serve.sh: the python-verify step left no verifier's coin ($P/coin.json), so no piece" >&2; failed=1; fi
fi
phi() {
  CUDA_VISIBLE_DEVICES='' "$PY" -c '
import json, sys
from verity.device import load_card, number
from verity_vllm import TargetProfile
from verity_vllm.engine.served_seed import DEFAULT_SEED
sms = int(number(load_card("h200-sxm-141gb")["sm"]["count"]))
doc = {"model": sys.argv[1], "revision": sys.argv[2], "default_seed": DEFAULT_SEED,
       "target": TargetProfile.default().with_overrides(compute_capability=(9, 0), num_sms=sms).to_json(),
       "execution": "bi-eager"}
open(sys.argv[3], "w").write(json.dumps(doc, indent=1) + "\n")
print(json.dumps(doc, sort_keys=True))
' "$REPO" "$REV" "$REC/serve-phi.json"
}
step serve-phi phi \
  && step policy env CUDA_VISIBLE_DEVICES='' "$PY" benchmarks/one_stage/inference_only.py policy --class "$CLASS" \
    --model "$REPO" --revision "$REV" --lease-min "${LEASE_MIN:-30}" --tree "$(git rev-parse HEAD 2> /dev/null)" \
    --serve "$REC/serve-phi.json" --pearl-tables "$SCHEME" --out "$OUT/inference-only-policy.json" || failed=1
step circuit env CUDA_VISIBLE_DEVICES='' "$PY" benchmarks/prove_path/prove_path.py class-circuit --serve "$REC/serve-phi.json" \
  --class "$CLASS" --pearl-tables "$SCHEME" --out "$REC/circuit-record.json" || failed=1
step verdicts verdicts || failed=1
step registered registered || failed=1
coins
if [ "$failed" != 0 ]; then
  coins > "$OUT/coins.txt"
  echo "serve.sh: no share: these steps failed, so run.py isn't run on what they left:" \
    "$(grep -v ' rc=0 ' "$OUT/steps.txt" | grep -v '^passes: ' | cut -d' ' -f1 | tr '\n' ' ')(each one's log is" \
    "$OUT/logs/STEP.log)" >&2
  exit 1
fi
step share "$PY" benchmarks/compute/run.py --from "$RUN" || failed=1
coins > "$OUT/coins.txt"
cat "$OUT/logs/share.log"
exit "$failed"
