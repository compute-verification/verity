#!/usr/bin/env bash
# `serve --tree-from` against the real challenger, on a CPU machine, on hidden_zk's builds (benchmarks/pouw/hidden_zk/job.sh,
# sourced: the proving build of flock-circuit and the Lean flock-verify). From a clone:
#   research run --on auto --source . --as LANE --project verity --cwd clone --cpus 16 --mem-gb 32 \
#     -- bash verity/protocols/flock/pod/93-zk-tree-from.sh
# The challengers are challenger.sh's (verity/composite/device/live_challenger/serve.py on the Lean verifier's coins, a test LMS key and
# a session secret outside $OUT). The statement is 91-plain-hidden.sh's (tests/plain_hidden.py: RoPE d64 at 32 instances,
# hidden outputs, the verifier's own program and partition), each session sampling all 32 (--draw subset:32, as every
# soundness claim is about a sampled session):
#   wrong-secret  serve --tree-from with another secret: the challenger refuses its hello and serve never serves
#   tree-from     serve/prove --zk, the session's coin tree sampled and opened by the challenger (--tree-from with its
#                 secret, --sessions 1); the challenger exits 0 once serve closes its session
#   own-tree      serve/prove --zk without --tree-from: a coin tree sampled by the server, which no challenger released
#   plain-tree    serve/prove without --zk on the keyed coin tree (FC_COINS=tree), sampled and opened by a second
#                 challenger
# and the audit of each session's record against its challenger's release log (challenger.sh's `challenger_audit` and
# `challenger_cases`: `challenger.sessions_tree_reason`, and the Lean's `challenger_session_trees`, which
# `sampled-proofs-one-stage-zk` runs): the tree-from and plain-tree sessions'
# pass; the own-tree session's, both --zk sessions at once, and the tree-from record with one round's coin or one link
# point changed are refused, the Lean's reason the Python's. Lean's flock-verify accepts every session (the verdict reads
# no challenger). Kept in $OUT: challenger/ and challenger-plain/ (record, key, releases.jsonl, hellos.jsonl,
# challenger.json, audit.json), sessions/, audit.json, lean.log, verdict.json (every check, `ok` when all hold; the job
# exits 1 otherwise) and the logs.
set -uo pipefail
export HZ_CONFIGS=zk-tree-from HZ_MEM_GB=${TF_MEM_GB:-32}
[ -n "${TF_CACHE:-}" ] && export HZ_CACHE=$TF_CACHE
source benchmarks/pouw/hidden_zk/job.sh
export PYTHONPATH=$SRC:$SRC/verity/protocols/flock/prover/python:$SRC/adapters/vllm:$SRC/benchmarks/numerical/python
CHALLENGER_REPO=$SRC CHALLENGER_WORK=$W/challengers
source verity/protocols/flock/prover/pod/challenger.sh
challenger_build
setup
CHALLENGER_PY=$PY
challenger_built || exit 1
S=$OUT/sessions CH=$OUT/challenger CHP=$OUT/challenger-plain
mkdir -p "$S"

"$PY" "$SRC/verity/protocols/flock/prover/tests/plain_hidden.py" --out "$W/stage" > "$OUT/stage.json" 2> "$OUT/stage.err" \
  || { log "stage FAILED: $(tail -2 "$OUT/stage.err")"; exit 1; }
PIN=$(sha512sum "$W/stage/circuit.txt" | cut -d' ' -f1)
log "staged: circuit_sha512 ${PIN:0:16}"
ST=(--circuit "$W/stage/circuit.txt" --tables "$W/stage/tables")

challenger_start "$CH" || exit 1
log "challenger listening on $CHALLENGER"
"$PY" -c 'import secrets; print(secrets.token_hex(32))' > "$W/wrong"

served() {   # served NAME zk|plain SERVE_ARGS...: one loopback session sampling all 32 instances under $S/NAME, its prove's exit
  local name=$1 mode=$2 srv port coins=${FC_COINS:-}; shift 2
  local -a zk=(--zk)
  [ "$mode" = plain ] && zk=() coins=tree
  FC_COINS=$coins "$BIN" serve "${zk[@]}" --listen 127.0.0.1:0 --out "$S/$name" --instances "$W/stage/pub-32.bin" "${ST[@]}" \
    --pin "$PIN" --operator loopback --draw subset:32 --sessions 1 "$@" > "$OUT/serve-$name.log" 2>&1 &
  srv=$!
  until grep -q '^SERVING ' "$OUT/serve-$name.log" 2>/dev/null; do kill -0 $srv 2>/dev/null || break; sleep 1; done
  port=$(sed -n 's/^SERVING 127.0.0.1://p' "$OUT/serve-$name.log")
  if [ -z "$port" ]; then wait $srv; echo "serve exit $? before SERVING: $(tail -1 "$OUT/serve-$name.log")"; return; fi
  FC_COINS=$coins "$BIN" prove "${zk[@]}" --verifier 127.0.0.1:$port --instances "$W/stage/inst-32.bin" "${ST[@]}" \
    > "$OUT/prove-$name.out" 2> "$OUT/prove-$name.err"
  local rc=$?
  wait $srv
  echo "prove exit $rc, serve exit $?: $(grep -h '^SESSION' "$OUT/serve-$name.log" | cut -c1-200)"
}
log "wrong-secret: $(served wrong-secret zk --tree-from "$CHALLENGER" --tree-secret-file "$W/wrong")"
log "tree-from: $(served tree-from zk "${TREE_FROM[@]}")"
challenger_end "$CH"; CH_RC=$?
log "challenger exit $CH_RC: $(tail -1 "$CH/challenger.err")"
log "own-tree: $(served own-tree zk)"
challenger_start "$CHP" || exit 1
log "plain-tree: $(served plain-tree plain "${TREE_FROM[@]}")"
challenger_end "$CHP"; CHP_RC=$?
log "plain challenger exit $CHP_RC: $(tail -1 "$CHP/challenger.err")"

one() { ls -d "$S/$1"/*/ 2>/dev/null | head -1; }
challenger_audit "$CHP" "$(one plain-tree)" | tee -a "$OUT/job.log"
challenger_cases "$CH" "$(one tree-from)" "$(one own-tree)" "$OUT/audit.json"
[ -s "$OUT/audit.json" ] || log "audit FAILED"

verify() {   # verify SESSION zk|plain: Lean flock-verify's verdict, accepted and why
  local coins=(--zk --session-tables 1); [ "$2" = plain ] && coins=(--coins tree)
  "$LEAN" verify "${coins[@]}" --statement verity/flock-circuit --circuit "$W/stage/circuit.txt" --public "$W/stage/pub-32.bin" \
    --partition "$W/stage/partition.json" --program "$W/stage/program.json" --session "$1" 2>&1 \
    | grep '^VERDICT ' \
    | "$PY" -c "import json, sys; [print(v['accepted'], v['why']) for v in (json.loads(l[8:]) for l in sys.stdin)]"
}
{ for n in tree-from own-tree; do echo "$n: $(verify "$(one $n)" zk)"; done
  echo "plain-tree: $(verify "$(one plain-tree)" plain)"; } | tee "$OUT/lean.log" | tee -a "$OUT/job.log"

"$PY" - "$OUT" "$CH_RC" "$CHP_RC" <<'EOF' | tee -a "$OUT/job.log"
import json, sys
from pathlib import Path
out, ch_rc, chp_rc = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
audit = json.loads((out / "audit.json").read_text()) if (out / "audit.json").exists() else {"cases": {}}
plain = out / "challenger-plain" / "audit.json"
plain = json.loads(plain.read_text()) if plain.exists() else {}
lean = dict(line.split(": ", 1) for line in (out / "lean.log").read_text().splitlines() if ": " in line)
cases = audit["cases"]
checks = {
    "wrong secret refused at the hello, never served": audit.get("hellos_refused") == 1
        and not (out / "sessions" / "wrong-secret").exists(),
    "challenger exit 0 (serve closed its session)": ch_rc == 0,
    "one tree sampled, opened": audit.get("trees") == 1 and all(n > 0 for n in audit.get("opened", {}).values()),
    "tree-from audit passes, Python and Lean": cases.get("tree-from") == {"python": None, "lean": None},
    **{f"{n} refused, Lean's reason the Python's": bool(cases.get(n, {}).get("python"))
       and cases[n]["python"] == cases[n]["lean"] for n in ("own-tree", "both", "coin-changed", "point-changed")},
    "plain challenger exit 0 (serve closed its session)": chp_rc == 0,
    "plain-tree audit passes, Python and Lean": plain.get("ok") is True,
    **{f"flock-verify accepts {n}": lean.get(n, "").startswith("True") for n in ("tree-from", "own-tree", "plain-tree")},
}
verdict = {"ok": all(checks.values()), "checks": checks}
(out / "verdict.json").write_text(json.dumps(verdict, indent=1) + "\n")
print(json.dumps(verdict))
sys.exit(0 if verdict["ok"] else 1)
EOF
rc=$?
log "zk-tree-from: done, verdict $([ $rc = 0 ] && echo ok || echo FAILED)"
exit $rc
