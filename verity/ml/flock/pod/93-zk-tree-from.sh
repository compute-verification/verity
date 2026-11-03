#!/usr/bin/env bash
# `serve --tree-from` against the real challenger, on a CPU machine, on hidden_zk's builds (benchmarks/pouw/hidden_zk/job.sh,
# sourced: the proving build of flock-circuit and the Lean flock-verify). From a clone:
#   research run --on auto --source . --as LANE --project verity --cwd clone --cpus 16 --mem-gb 32 \
#     -- bash verity/ml/flock/pod/93-zk-tree-from.sh
# The challenger is benchmarks/live_challenger/serve.py, its coins the Lean verifier's `IO.getRandomBytes`
# (`one_stage.verifier.coins`, built here with lake first), a test LMS key outside $OUT and a session secret. The statement
# is 91-plain-hidden.sh's (tests/plain_hidden.py: RoPE d64 at 32 instances, hidden outputs, the verifier's own program and
# partition), each session drawing all 32 (--draw subset:32, as every soundness claim is about a drawn session):
#   wrong-secret  serve --tree-from with another secret: the challenger refuses its hello and serve never serves
#   tree-from     serve/prove --zk, the session's coin tree sampled and opened by the challenger (--tree-from with its
#                 secret, --sessions 1); the challenger exits 0 once serve closes its session
#   own-tree      serve/prove --zk without --tree-from: a coin tree drawn by the server, which no challenger released
# and the audit of each session's record against the challenger's release log (`challenger.sessions_tree_reason`, and
# the Lean's `challenger_session_trees`, which `sampled-proofs-one-stage-zk` runs): the tree-from session's passes; the
# own-tree session's, both sessions at once, and the tree-from record with one round's coin or one link point changed are
# refused, the Lean's reason the Python's. Lean's flock-verify accepts both sessions (the verdict reads no challenger).
# Kept in $OUT: challenger/ (record, key, releases.jsonl, hellos.jsonl, challenger.json), sessions/, audit.json,
# lean.log, verdict.json (every check, `ok` when all hold; the job exits 1 otherwise) and the logs.
set -uo pipefail
export HZ_CONFIGS=zk-tree-from HZ_MEM_GB=${TF_MEM_GB:-32}
[ -n "${TF_CACHE:-}" ] && export HZ_CACHE=$TF_CACHE
source benchmarks/pouw/hidden_zk/job.sh
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/benchmarks/numerical/python
command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
SP=$SRC/verity/core/service/sampled_proofs
( cd "$SP" && lake build sampled-proofs-verifier sampled-proofs-difftest > "$OUT/build-sampled-proofs.log" 2>&1; echo $? > "$W/sp.rc" ) &
setup
[ "$(cat "$W/sp.rc" 2>/dev/null)" = 0 ] || { log "sampled-proofs build FAILED: $(tail -5 "$OUT/build-sampled-proofs.log")"; exit 1; }
DIFFTEST=$SP/.lake/build/bin/sampled-proofs-difftest
S=$OUT/sessions CH=$OUT/challenger
mkdir -p "$S"

"$PY" "$SRC/verity/ml/flock/tests/plain_hidden.py" --out "$W/stage" > "$OUT/stage.json" 2> "$OUT/stage.err" \
  || { log "stage FAILED: $(tail -2 "$OUT/stage.err")"; exit 1; }
PIN=$(sha512sum "$W/stage/circuit.txt" | cut -d' ' -f1)
log "staged: circuit_sha512 ${PIN:0:16}"
ST=(--circuit "$W/stage/circuit.txt" --tables "$W/stage/tables")

"$PY" -c 'import secrets; print(secrets.token_hex(32)); print(secrets.token_hex(32), file=open(__import__("sys").argv[1], "w"))' \
  "$W/wrong" > "$W/secret"
"$PY" "$SRC/benchmarks/live_challenger/serve.py" --out "$CH" --port 0 --scheme verity/flock/tree-from/v1 \
  --families '[["positions", 1]]' --session tree-from-1 --key "$W/key" --params 4,3 --keygen-workers 1 \
  --secret-file "$W/secret" > "$OUT/challenger.out" 2> "$OUT/challenger.err" &
ch=$!
until grep -q '^listening ' "$OUT/challenger.out" 2>/dev/null; do kill -0 $ch 2>/dev/null || break; sleep 1; done
ADDR=$(sed -n 's/^listening //p' "$OUT/challenger.out")
[ -n "$ADDR" ] || { log "challenger FAILED to listen: $(tail -3 "$OUT/challenger.err")"; exit 1; }
log "challenger listening on $ADDR"

served() {   # served NAME SERVE_ARGS...: one loopback --zk session drawing all 32 instances under $S/NAME, its prove's exit
  local name=$1 srv port; shift
  "$BIN" serve --zk --listen 127.0.0.1:0 --out "$S/$name" --instances "$W/stage/pub-32.bin" "${ST[@]}" --pin "$PIN" \
    --operator loopback --draw subset:32 --sessions 1 "$@" > "$OUT/serve-$name.log" 2>&1 &
  srv=$!
  until grep -q '^SERVING ' "$OUT/serve-$name.log" 2>/dev/null; do kill -0 $srv 2>/dev/null || break; sleep 1; done
  port=$(sed -n 's/^SERVING 127.0.0.1://p' "$OUT/serve-$name.log")
  if [ -z "$port" ]; then wait $srv; echo "serve exit $? before SERVING: $(tail -1 "$OUT/serve-$name.log")"; return; fi
  "$BIN" prove --zk --verifier 127.0.0.1:$port --instances "$W/stage/inst-32.bin" "${ST[@]}" \
    > "$OUT/prove-$name.out" 2> "$OUT/prove-$name.err"
  local rc=$?
  wait $srv
  echo "prove exit $rc, serve exit $?: $(grep -h '^SESSION' "$OUT/serve-$name.log" | cut -c1-200)"
}
log "wrong-secret: $(served wrong-secret --tree-from "$ADDR" --tree-secret-file "$W/wrong")"
log "tree-from: $(served tree-from --tree-from "$ADDR" --tree-secret-file "$W/secret")"
wait $ch; CH_RC=$?
log "challenger exit $CH_RC: $(tail -1 "$OUT/challenger.err")"
log "own-tree: $(served own-tree)"

one() { ls -d "$S/$1"/*/ 2>/dev/null | head -1; }
"$PY" - "$CH" "$(one tree-from)" "$(one own-tree)" "$DIFFTEST" "$OUT/audit.json" <<'EOF'
import copy, json, subprocess, sys
from pathlib import Path
from verity.core.protocols.sampled_proofs import challenger as C, remote as R

ch, tree, own, difftest, out = sys.argv[1:]
log = R.log_lines(Path(ch) / R.RELEASES)
rec = {n: json.loads((Path(d) / "session.json").read_text()) for n, d in (("tree-from", tree), ("own-tree", own))}
coin, point = copy.deepcopy(rec["tree-from"]), copy.deepcopy(rec["tree-from"])
s = next(s for s in coin["streams"] if s["rounds"])
c = s["rounds"][0]["coins"][0]
s["rounds"][0]["coins"][0] = c[:-1] + "0123456789abcdef"[("0123456789abcdef".index(c[-1]) + 1) % 16]
p = point["link"]["points"][0][0]
point["link"]["points"][0][0] = p[:-1] + "0123456789abcdef"[("0123456789abcdef".index(p[-1]) + 1) % 16]
cases = {"tree-from": [("tree-from", rec["tree-from"])], "own-tree": [("own-tree", rec["own-tree"])],
         "both": [("tree-from", rec["tree-from"]), ("own-tree", rec["own-tree"])],
         "coin-changed": [("tree-from", coin)], "point-changed": [("tree-from", point)]}
lean = subprocess.run([difftest], input="".join(json.dumps({"fn": "challenger_session_trees", "input": {
    "log": log, "records": [{"name": n, "record": r} for n, r in rs]}}) + "\n" for rs in cases.values()),
    capture_output=True, text=True, check=True).stdout.splitlines()
trees = C.coin_trees_of(log)
audit = {"trees": len(trees), "opened": {root: len(t.opened) for root, t in trees.items()},
         "hellos_refused": len(R.hello_lines(Path(ch) / R.HELLOS) or []), "cases": {}}
for (name, rs), answer in zip(cases.items(), lean):
    py = C.sessions_tree_reason(trees, rs)
    audit["cases"][name] = {"python": py, "lean": json.loads(answer).get("output", {}).get("reason", answer)}
json.dump(audit, open(out, "w"), indent=1)
print(json.dumps(audit))
EOF
[ -s "$OUT/audit.json" ] || log "audit FAILED"

verify() {   # verify SESSION: Lean flock-verify's verdict, accepted and why
  "$LEAN" verify --zk --statement verity/flock-circuit --circuit "$W/stage/circuit.txt" --public "$W/stage/pub-32.bin" \
    --partition "$W/stage/partition.json" --program "$W/stage/program.json" --session-tables 1 --session "$1" 2>&1 \
    | grep '^VERDICT ' \
    | "$PY" -c "import json, sys; [print(v['accepted'], v['why']) for v in (json.loads(l[8:]) for l in sys.stdin)]"
}
{ for n in tree-from own-tree; do echo "$n: $(verify "$(one $n)")"; done; } | tee "$OUT/lean.log" | tee -a "$OUT/job.log"

"$PY" - "$OUT" "$CH_RC" <<'EOF' | tee -a "$OUT/job.log"
import json, sys
from pathlib import Path
out, ch_rc = Path(sys.argv[1]), int(sys.argv[2])
audit = json.loads((out / "audit.json").read_text()) if (out / "audit.json").exists() else {"cases": {}}
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
    **{f"flock-verify accepts {n}": lean.get(n, "").startswith("True") for n in ("tree-from", "own-tree")},
}
verdict = {"ok": all(checks.values()), "checks": checks}
(out / "verdict.json").write_text(json.dumps(verdict, indent=1) + "\n")
print(json.dumps(verdict))
sys.exit(0 if verdict["ok"] else 1)
EOF
rc=$?
log "zk-tree-from: done, verdict $([ $rc = 0 ] && echo ok || echo FAILED)"
exit $rc
