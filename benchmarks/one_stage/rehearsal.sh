#!/usr/bin/env bash
# The remote demo's served path as one command a node: each verb is the harness (`served_verdicts.sh`) at fixed
# settings, its output through a filter that withholds a session secret after its key, so no run's log keeps one. On a
# TERM to the run's process group the harness cleans up as it does alone: this shell waits for it, the filter outlives
# the signal so the cleanup still has somewhere to write, and the exit status is the harness's.
#
#   rehearsal.sh check    any CPU node, in the tree: what the verbs below read that needs no card and no gate, at the
#                         serve's settings. Each step's log is $OUT/logs/STEP.log, its exit status and seconds a line of
#                         $OUT/steps.txt, and the exit status is 0 only where every step's is. The steps: this tree's
#                         environment (uv sync); the filter around a stand-in harness (a made-up secret, an exit status
#                         carried, a TERM waited out); the settings file as `serve` holds it; `serve` around a stand-in
#                         harness (the deferral's copy asked once, only after a serve that exited 0); the class's
#                         circuit cut at Pearl+'s linears from the serve's phi (a stand-in for `serve-phi`, which
#                         reads the card: the H200's compute capability and SMs); the harness's own policy step,
#                         which fixes the limits and states what they prove; the policy's Lean check on the class's
#                         own Program (`rehearsal_limits.py`), and the settings' ALPHA, GAMMA and DAY_PROGRAM held to
#                         what it answers ($OUT/record.env.add has the lines where they differ); the day's declaration
#                         at that Program in a scratch log, checked as the follow-on checks it; the Lean the serve,
#                         the audit and the compose run (Flock's verifier and ZK server, Pearl+'s check, remote's
#                         composite); Pearl+'s period, salt session, draw, coin and piece against this tree's own
#                         gate code (`pearlplus_local_e2e.py`), whose piece then holds the execution phase and each
#                         check's cost.
#                         The whole verb runs inside one memory scope (MEM_GB, default 64) where the node makes one.
#   rehearsal.sh serve    the serve node, one card: the harness at SETTINGS (default the run's sent record.env), after
#                         the settings are held to what the rehearsal serves (`settings_ok`). Where the serve deferred
#                         its audit and COPY_TO names HOST[:DIR], the deferral's files go there before the run ends
#                         (`served_verdicts.sh copy deferred`), since the serve dir is gone with the node.
#   rehearsal.sh audit    the audit node, D the deferral's copy at the serve dir's own path: `served_verdicts.sh v3c`,
#                         the follow-on, at the deferral's settings. PP_PEER is this node's name in the gate's peers
#                         file. MAIN_FILE and PP_CHALLENGER_KEY default to the run's sent main.txt and
#                         challenger-key.json, V3C_AUDIT_TIMEOUT to 46h, and V3C_ARCHIVE_LINK to 1: each member's circuit
#                         file is linked into its archive, not copied (`a0.lean_verify`). PREFLIGHT_ONLY=1 stops after
#                         the harness's preflight.
#   rehearsal.sh compose  a node that holds the compose hand-off alone, D its copy: `served_verdicts.sh compose` inside
#                         one memory scope of its own (COMPOSE_MEM_GB, default 64, no swap). Each Lean check it starts
#                         runs in a scope of its own and is not started past its blob limit (`served_cases.py`:
#                         CHECK_MEM_GB, CHECK_BLOBS_GB); on a node that makes no user scope the verb is refused.
#   rehearsal.sh filter CMD...   CMD through the filter alone, with its exit status (`filter-test DIR`, `settings FILE`
#                         and `serve-test DIR` are `check`'s three steps of those names).
#
# A follow-on's settings come from the serve dir (deferred.env, compose.env), which carry no POUW: `audit` and `compose`
# set POUW=pearlplus, the rehearsal's.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SELF=$HERE/rehearsal.sh HARNESS=${HARNESS:-$HERE/served_verdicts.sh}
PATTERN="([\"']secret[\"'] *: *[\"'])[0-9a-f]{16,}"
filtered() {  # CMD...: its output through the filter; the exit status is CMD's
  trap ':' TERM INT HUP
  "$@" 2>&1 | (trap '' TERM INT HUP; exec sed -u -E "s/$PATTERN/\1(withheld)/g")
  return "${PIPESTATUS[0]}"
}
refuse() { echo "REHEARSAL refused: $*" >&2; exit 3; }
user_scope() { command -v systemd-run > /dev/null && systemd-run --user --scope --quiet --collect true > /dev/null 2>&1; }

settings_ok() {  # $1, the settings file: what the rehearsal's serve is, held before the harness's own preflight
  [ -f "$1" ] && [ -r "$1" ] || { echo "SETTINGS refused: $1 is not a readable file"; return 3; }
  (
    set -a
    # shellcheck source=/dev/null
    . "$1"
    set +a
    bad=0
    no() { echo "SETTINGS refused: $*"; bad=3; }
    for want in POUW=pearlplus PEARLPLUS_TIE=1 PRODUCT=remote; do
      v=${want%%=*}
      [ "${!v-}" = "${want#*=}" ] || no "$v is ${!v-unset}, and the rehearsal serves at $want"
    done
    for v in ROW PP_CHALLENGER PP_PEER MAIN_FILE PP_CHALLENGER_KEY ALPHA GAMMA LIMITS_UNIT DAY DAY_PROGRAM DAY_LOG DAY_PARAMS; do
      [ -n "${!v-}" ] || no "$v is unset"
    done
    # the record states the class's own two numbers with what they prove
    [ -n "${LIMITS_PROVE-}" ] || no "LIMITS_PROVE is unset: the limits are the class's own two numbers, and the record" \
      "says in a sentence what they prove (the slowdown they prove is 1, that is none)"
    for v in MAIN_FILE PP_CHALLENGER_KEY DAY_PARAMS; do
      [ -z "${!v-}" ] || [ -s "${!v}" ] || no "$v names ${!v}, which is missing or empty: send it with the run"
    done
    [ "$bad" != 0 ] || echo "SETTINGS held: Pearl+ tied to the window, product remote, limits $ALPHA and $GAMMA with their" \
      "unit and what they prove, day $DAY at program ${DAY_PROGRAM:0:16}…"
    exit "$bad"
  )
}

serve() {
  export SETTINGS=${SETTINGS:-${RESEARCH_RUN_DIR:?no SETTINGS and no run dir}/inputs/record.env}
  settings_ok "$SETTINGS" || exit 3
  local rc d out=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
  filtered bash "$HARNESS" "$@"
  rc=$?
  [ "$rc" = 0 ] && [ -n "${COPY_TO:-}" ] || return "$rc"
  d=$(sed -n 's/^serve dir //p' "$out/machine.txt" 2> /dev/null | tail -1)
  if [ -n "$d" ] && [ -f "$d/private/deferred/deferred.env" ]; then
    filtered env D="$d" bash "$HARNESS" copy deferred "$COPY_TO"
    rc=$?
  else
    echo "REHEARSAL: COPY_TO is $COPY_TO, and the serve dir ${d:-(not named in $out/machine.txt)} holds no deferral: nothing copied"
  fi
  return "$rc"
}

audit() {
  [ -n "${D:-}" ] && [ -f "$D/private/deferred/deferred.env" ] \
    || refuse "audit runs on the deferral's copy: D=${D:-unset} holds no private/deferred/deferred.env"
  [ -n "${PP_PEER:-}" ] || refuse "PP_PEER is unset: this node's name in the gate's peers file, never the served node's"
  export D PP_PEER POUW=pearlplus PEARLC_CHALLENGER_LANE=${PEARLC_CHALLENGER_LANE:-vllm} \
    PP_GATE_KEY=${PP_GATE_KEY:-$HOME/.ssh/vllm_lane_ed25519} \
    MAIN_FILE=${MAIN_FILE:-${RESEARCH_RUN_DIR:-.}/inputs/main.txt} \
    PP_CHALLENGER_KEY=${PP_CHALLENGER_KEY:-${RESEARCH_RUN_DIR:-.}/inputs/challenger-key.json} \
    V3C_AUDIT_TIMEOUT=${V3C_AUDIT_TIMEOUT:-46h} V3C_ARCHIVE_LINK=${V3C_ARCHIVE_LINK:-1}
  filtered bash "$HARNESS" v3c
}

compose() {
  [ -n "${D:-}" ] && [ -f "$D/private/compose/compose.env" ] \
    || refuse "compose runs on the hand-off's copy: D=${D:-unset} holds no private/compose/compose.env"
  user_scope || refuse "this node makes no systemd user scope, so the compose and its Lean checks would run under no" \
    "memory limit of their own: run it from a login session (loginctl enable-linger), on a node that makes one"
  export D POUW=pearlplus
  filtered systemd-run --user --scope --quiet --collect -p "MemoryMax=${COMPOSE_MEM_GB:-64}G" -p MemorySwapMax=0 -- \
    bash "$HARNESS" compose
}

# ---- check: the CPU-only steps
filter_test() {  # the filter around a stand-in harness that quotes a made-up secret; prints counts, never the secret
  local scr=$1 rc pid n=0 h
  export SCR=$scr
  cat > "$scr/harness.sh" <<'EOF'
#!/usr/bin/env bash
hex=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
echo "$hex" >> "$SCR/hex"
echo "CoinError: the gate refused the open: {\"commitment\": null, \"secret\": \"$hex\", \"session\": \"s1\"}"
echo "on stderr: {'secret': '$hex', 'seq': 1}" >&2
if [ "${1:-}" = wait ]; then
  trap 'echo "harness: TERM received, the answer was {\"secret\": \"'"$hex"'\"}"; sleep 1; touch "$SCR/cleaned"; exit 143' TERM
  touch "$SCR/ready"
  while :; do sleep 0.2; done
fi
exit 7
EOF
  count() { n=0; while read -r h; do [ -z "$h" ] || n=$((n + $(grep -c -F "$h" "$1" || true))); done < "$scr/hex"; echo "$n"; }
  bash "$SELF" filter bash "$scr/harness.sh" > "$scr/log1" 2>&1; rc=$?
  echo "a harness that exits 7: the wrapper exits $rc; made-up secrets in its log $(count "$scr/log1"); lines with (withheld) $(grep -c -F '(withheld)' "$scr/log1")"
  [ "$rc" = 7 ] && [ "$(count "$scr/log1")" = 0 ] && [ "$(grep -c -F '(withheld)' "$scr/log1")" = 2 ] || return 1
  setsid bash "$SELF" filter bash "$scr/harness.sh" wait > "$scr/log2" 2>&1 &
  pid=$!
  for _ in $(seq 100); do [ -e "$scr/ready" ] && break; sleep 0.1; done
  kill -TERM -- "-$pid"; wait "$pid"; rc=$?
  echo "a harness ended by TERM to its process group: the wrapper exits $rc; the harness's trap ran: $([ -e "$scr/cleaned" ] && echo yes || echo no); its trap's line is in the log: $(grep -c 'TERM received' "$scr/log2"); made-up secrets in its log $(count "$scr/log2")"
  [ "$rc" = 143 ] && [ -e "$scr/cleaned" ] && [ "$(grep -c 'TERM received' "$scr/log2")" = 1 ] && [ "$(count "$scr/log2")" = 0 ] || return 1
  bash "$scr/harness.sh" > "$scr/log3" 2>&1
  echo "the same harness without the wrapper: made-up secrets in its log $(count "$scr/log3")"
  [ "$(count "$scr/log3")" = 2 ]
}
serve_test() {  # `serve` around a stand-in harness, at SETTINGS: the deferral's copy is asked once, after a serve that
  # exited 0 and deferred, of the serve dir the harness named; the exit status is the serve's, then the copy's
  local scr=$1 rc n=0 ok=0
  mkdir -p "$scr/out" "$scr/d/private/deferred"
  export SCR=$scr
  cat > "$scr/harness.sh" <<'EOF'
#!/usr/bin/env bash
if [ "${1:-}" = copy ]; then echo "D=$D $*" >> "$SCR/copies"; exit "${COPY_RC:-0}"; fi
echo "serve dir $SCR/d" > "$OUT/machine.txt"
rm -f "$SCR/d/private/deferred/deferred.env"
[ -z "${DEFERS:-}" ] || : > "$SCR/d/private/deferred/deferred.env"
exit "${SERVE_RC:-0}"
EOF
  : > "$scr/copies"
  try() {  # WHAT WANT_RC WANT_COPIES ENV...: one serve under ENV
    local what=$1 want=$2 copies=$3
    shift 3
    env OUT="$scr/out" HARNESS="$scr/harness.sh" "$@" bash "$SELF" serve > "$scr/log" 2>&1; rc=$?
    n=$(grep -c . "$scr/copies")
    echo "$what: the wrapper exits $rc (want $want); copies asked so far $n (want $copies)"
    [ "$rc" = "$want" ] && [ "$n" = "$copies" ] || ok=1
  }
  try "a serve that deferred, with COPY_TO" 0 1 DEFERS=1 COPY_TO=host:dir
  [ "$(tail -1 "$scr/copies")" = "D=$scr/d copy deferred host:dir" ] \
    || { echo "the copy was asked as: $(tail -1 "$scr/copies")"; ok=1; }
  try "a serve that deferred, with no COPY_TO" 0 1 DEFERS=1
  try "a serve that exits 7, with COPY_TO" 7 1 DEFERS=1 SERVE_RC=7 COPY_TO=host:dir
  try "a serve that deferred nothing, with COPY_TO" 0 1 COPY_TO=host:dir
  grep -q 'holds no deferral: nothing copied' "$scr/log" || { echo "no line says nothing was copied"; ok=1; }
  try "a copy that exits 5" 5 2 DEFERS=1 COPY_RC=5 COPY_TO=host:dir
  return "$ok"
}

check() {
  local OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out} MEM_GB=${MEM_GB:-64}
  mkdir -p "$OUT/logs"
  if [ -z "${REHEARSAL_SCOPED:-}" ]; then
    if user_scope; then
      echo "memory scope: MemoryMax=${MEM_GB}G, no swap" > "$OUT/scope.txt"
      REHEARSAL_SCOPED=1 exec systemd-run --user --scope --quiet --collect -p "MemoryMax=${MEM_GB}G" -p MemorySwapMax=0 -- \
        bash "$SELF" check
    fi
    echo "memory scope: NONE (this node makes no user scope): the audit's and the compose's node must" > "$OUT/scope.txt"
  fi
  local TREE S=$OUT/steps.txt failed=0 scratch PY D CLASS=${CLASS:-4,32,4} SCHEME=${SCHEME:-pearl-c-h100-v1-h3}
  TREE=$(cd "$HERE/../.." && pwd)
  cd "$TREE" || exit 3
  SETTINGS=${SETTINGS:-${RESEARCH_RUN_DIR:-.}/inputs/record.env}
  step() {  # NAME LIMIT CMD...
    local name=$1 limit=$2 t rc
    shift 2
    t=$(date +%s)
    timeout "$limit" "$@" > "$OUT/logs/$name.log" 2>&1
    rc=$?
    printf '%s rc=%s %ss\n' "$name" "$rc" "$(( $(date +%s) - t ))" | tee -a "$S"
    [ "$rc" = 0 ] || failed=1
    return "$rc"
  }
  scratch=$(mktemp -d)
  # shellcheck disable=SC2064
  trap "rm -rf '$scratch'" EXIT
  { git -C "$TREE" rev-parse HEAD 2> /dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}"; hostname -s; nproc; grep MemTotal /proc/meminfo; } > "$OUT/tree.txt"
  step sync 20m uv sync -q --all-packages --extra torch-cpu || { echo "no environment: nothing else ran" | tee -a "$S"; exit 1; }
  PY=$TREE/.venv/bin/python
  export PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/research/src

  mkdir -p "$scratch/filter"
  step filter 2m bash "$SELF" filter-test "$scratch/filter"
  cp "$OUT/logs/filter.log" "$OUT/filter.txt"
  step settings 1m bash "$SELF" settings "$SETTINGS"
  step serve-copy 1m env SETTINGS="$SETTINGS" bash "$SELF" serve-test "$scratch/serve-copy"

  # the class's circuit from phi, as PEARLPLUS-CIRCUIT's command makes it on the serve node
  D=$scratch/serve
  mkdir -p "$D/pearlplus"
  step serve-phi 5m "$PY" -c '
import json, sys
from verity_vllm import TargetProfile
from verity_vllm.engine.served_seed import DEFAULT_SEED
doc = {"model": sys.argv[1], "revision": sys.argv[2], "default_seed": DEFAULT_SEED,
       "target": TargetProfile.default().with_overrides(compute_capability=(9, 0), num_sms=132).to_json(),
       "execution": "bi-eager"}
open(sys.argv[3], "w").write(json.dumps(doc, indent=1) + "\n")
print(json.dumps(doc, sort_keys=True))
' "${REPO:-unsloth/Llama-3.2-1B}" "${REV:-9535bd9b1d1dea6acafbdc4813b728796aeb28da}" "$D/pearlplus/serve-phi.json"
  cp "$D/pearlplus/serve-phi.json" "$OUT/serve-phi.json" 2> /dev/null
  step class-circuit 30m "$PY" benchmarks/prove_path/prove_path.py class-circuit --serve "$D/pearlplus/serve-phi.json" \
    --class "$CLASS" --pearl-tables "$SCHEME" --out "$D/pearlplus/circuit-record.json"

  # the harness's own policy step at the settings' limits: the limits file and its line, with what they prove
  step inference-policy 30m env SETTINGS="$SETTINGS" PEARLC_CHALLENGER_GATE= D="$D" OUT="$OUT" VENV="$TREE/.venv" \
    RESEARCH_SOURCE_SHA="$(head -1 "$OUT/tree.txt")" bash "$HARNESS" inference-policy
  step limits 60m "$PY" benchmarks/one_stage/rehearsal_limits.py "$D/pearlplus/serve-phi.json" "$CLASS" "$SCHEME" \
    "$OUT/inference-only-policy.json" "$OUT/limits.json" ${REJECTS:+--rejects}
  step limits-held 2m "$PY" - "$OUT/limits.json" "$OUT/inference-only-limits.json" "$SETTINGS" "$OUT/record.env.add" <<'EOF'
import json, os, subprocess, sys
from fractions import Fraction
prof, said, settings, add = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3], sys.argv[4]
p = prof["profile"]
want = {"ALPHA": f"{p['alpha'][0]}/{p['alpha'][1]}", "GAMMA": str(p["gamma"]), "DAY_PROGRAM": prof["program"]}
open(add, "w").write("".join(f"{k}={v}\n" for k, v in want.items()))
got = subprocess.run(["bash", "-c", 'set -a; . "$0"; set +a; printf "%s\\n%s\\n%s\\n" "$ALPHA" "$GAMMA" "$DAY_PROGRAM"', settings],
                     capture_output=True, text=True, check=True).stdout.split("\n")[:3]
bad = [f"{k} is {g or 'unset'}, and the class's own is {want[k]}" for k, g in zip(want, got)
       if not g or (Fraction(g) != Fraction(want[k]) if k == "ALPHA" else g != want[k])]
if prof.get("at_profile", {}).get("verdict") != "accept":
    bad.append(f"the policy's Lean check at the class's own numbers: {prof.get('at_profile')}")
if (Fraction(*said["alpha"]), said["gamma"]) != (Fraction(want["ALPHA"]), p["gamma"]) or not said.get("proves"):
    bad.append(f"the harness's limits file states {said['alpha']}, {said['gamma']}, proves {said.get('proves')!r}")
if bad:
    sys.exit("LIMITS not held (record.env.add has the settings' three lines): " + "; ".join(bad))
print(f"LIMITS held: the settings' ALPHA, GAMMA and DAY_PROGRAM are the class's own ({prof['templates']} templates, "
      f"program {prof['program'][:16]}…), the policy's Lean check accepts at them, and the limits file says: {said['proves']}")
EOF

  # the day: serve-openai's own declaration of the settings' day at that Program in a scratch log, then the follow-on's
  # check of it, the same declaration read again, and another circuit that day refused
  step day 5m "$PY" - "$OUT/limits.json" "$scratch/day" "$SETTINGS" "$OUT/day.json" <<'EOF'
import json, subprocess, sys
from pathlib import Path
from verity.core import site as S
from verity.core.protocols import registration as G
from verity_vllm.pipeline import serve_day as SD
prof, d, settings, out = json.load(open(sys.argv[1])), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
day, params = subprocess.run(["bash", "-c", 'set -a; . "$0"; set +a; printf "%s\\n%s\\n" "$DAY" "$DAY_PARAMS"', settings],
                             capture_output=True, text=True, check=True).stdout.split("\n")[:2]
day = int(day)
d.mkdir(parents=True, exist_ok=True)
made = json.loads(json.dumps(G.params("remote", S.Auditor())))
sent = json.loads(Path(params).read_text())
assert sent == made, f"DAY_PARAMS holds {sent}, and remote's parameters with no auditor's device are {made}"
p = G.params_of(sent)
got = SD.append(d / "day-log.json", G.Declaration(day, prof["program"], ()), p)
composite, log = G.log_of(json.load(open(d / "day-log.json")))
res = G.check_log(p, (composite, log))
assert res[0] == "accept", res
want = got.describe()["declaration"]
assert any(isinstance(e, G.Declaration) and e.day == day and G.Declaration(e.day, e.program, e.blocks).digest() == want
           for e in log), want
again = SD.append(d / "day-log.json", G.Declaration(day, prof["program"], ()), p)
assert (again.declared, again.digest) == ("earlier", got.digest), again
try:
    SD.append(d / "day-log.json", G.Declaration(day, "00" * 64, ()), p)
    sys.exit("a second circuit was declared that day")
except G.Refused as e:
    second = str(e)
doc = {"day": day, "program": prof["program"], "declaration": want, "declared": got.declared, "check": res[0],
       "again": again.declared, "another_circuit": second, "params": sent}
open(out, "w").write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
print(json.dumps(doc, sort_keys=True))
EOF

  # the Lean the three verbs run
  step lean-flock 50m bash -c 'cd verity/core/service/flock && lake build Flock flock-verify flock-firewall'
  step lean-pearlplus 40m bash -c 'cd verity/core/protocols/pouw/lean/pearlplus && lake build pouw-pearlplus'
  step lean-remote-v1 40m bash -c 'cd verity/core/protocols/remote && lake build remote-v1'

  # Pearl+ end to end against this tree's own gate code; the piece holds the phase's times and each check's cost
  step pearlplus-e2e 30m "$PY" benchmarks/one_stage/pearlplus_local_e2e.py run --out "$scratch/e2e" --lean build
  cp "$scratch/e2e/e2e.json" "$OUT/pearlplus-e2e.json" 2> /dev/null
  step e2e-held 1m "$PY" - "$scratch/e2e" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
e2e = json.loads((d / "e2e.json").read_text())
piece, phase, checks = e2e["piece"], e2e["piece"]["execution_phase"], e2e["piece"]["checks"]
lean, row = e2e["lean"], e2e["row"]
assert piece["verdict"] == "accept", piece
assert lean["rc"] == 0 and lean["lean"]["verdict"] == "accept" and not lean["differ"], lean
# a test instance credits nothing, so on the toy every condition of a row of record holds up to its last, the credit
assert row["verdict"] == "missing" and "the Lean verdict credits 0 MACs" in row["why"], row
period = phase["period"]
assert phase["start_utc"] and phase["end_utc"] and period["opened_utc"] and period["closed_utc"], phase
assert phase["prover"]["generate_s"] and "prover's" in phase["prover"]["clock"], phase
assert all((checks.get(k) or {}).get("seconds") is not None for k in ("python", "lean")), checks
print(json.dumps({"verdict": piece["verdict"], "execution_phase": phase, "checks": checks, "row": e2e["row"]}, sort_keys=True))
EOF
  cp "$OUT/logs/e2e-held.log" "$OUT/e2e-held.json" 2> /dev/null
  echo "done: $([ "$failed" = 0 ] && echo "every step passed" || echo "a step failed")" | tee -a "$S"
  return "$failed"
}

case ${1:-} in
  check) check ;;
  serve) shift; serve "$@" ;;
  audit) audit ;;
  compose) compose ;;
  filter) shift; filtered "$@" ;;
  filter-test) filter_test "${2:?a scratch directory}" ;;
  settings) settings_ok "${2:?a settings file}" ;;
  serve-test) serve_test "${2:?a scratch directory}" ;;
  *) refuse "rehearsal.sh check|serve|audit|compose|filter CMD..., not ${1:-nothing}" ;;
esac
