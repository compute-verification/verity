# Sourced by a pod whose `flock-circuit serve` takes its sessions' coin trees from the auditor's challenger
# (`serve --tree-from`): verity/composite/device/live_challenger/serve.py, its coins the Lean verifier's `IO.getRandomBytes`
# (`one_stage.verifier.coins`, the sampled-proofs-verifier built here with lake), a test LMS key and a fresh session secret
# in the pod's scratch, none of it the prover's. One challenger serves one `serve` process: its session closes after that
# server's last session, and the challenger exits 0 once it closes.
#   challenger_build                 start building the Lean the challenger and its audit run (sampled-proofs-verifier,
#                                    sampled-proofs-difftest), in the background
#   challenger_built                 wait for that build; 0 when it built
#   challenger_start DIR             a challenger whose record, key, releases.jsonl and hellos.jsonl go under DIR; sets
#                                    CHALLENGER (its HOST:PORT), CHALLENGER_SECRET (the secret's file) and TREE_FROM, the
#                                    flags `serve` takes (--tree-from HOST:PORT --tree-secret-file FILE)
#   challenger_end DIR               wait for DIR's challenger to exit (its server closed the session): its exit status
#   challenger_audit DIR SESSION...  each SESSION directory's session.json held to DIR's release log
#                                    (`challenger.sessions_tree_reason`, and the Lean's `challenger_session_trees`): DIR/audit.json,
#                                    0 when both pass every record
#   challenger_cases DIR TREE OWN OUT  93-zk-tree-from.sh's audit, the same two checks per case: TREE's record alone, OWN's
#                                    alone, both at once, and TREE's with one round's coin or one link point changed; OUT
#                                    (JSON) holds each case's Python reason beside the Lean's
# Both hold each record at the tables its link's root_b keys, with no root_B length: neither sets up the statement that
# fixes one.
# env: CHALLENGER_REPO (default the working directory: a verity tree), CHALLENGER_PY (default python3, with the tree's
#      Python importable), CHALLENGER_WORK (default `mktemp -d`: the key and the secret), CHALLENGER_PARAMS (the LMS key's
#      height and Winternitz parameter, default 4,3: a test key)
CHALLENGER_REPO=${CHALLENGER_REPO:-$PWD}
CHALLENGER_PY=${CHALLENGER_PY:-python3}
CHALLENGER_WORK=${CHALLENGER_WORK:-$(mktemp -d)}
CHALLENGER_SP=$CHALLENGER_REPO/verity/protocols/spot/verifier/lean

challenger_build() {
  command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
  local lock=$CHALLENGER_REPO/verity/difftest/lean/.lake/harness-build.lock
  mkdir -p "$CHALLENGER_WORK" "$(dirname "$lock")"
  # under one_stage.verifier's own build lock: in a tree other runs share (--cwd source), they build here too
  ( cd "$CHALLENGER_SP" && flock "$lock" lake build sampled-proofs-verifier sampled-proofs-difftest \
      > "$CHALLENGER_WORK/build.log" 2>&1
    echo $? > "$CHALLENGER_WORK/build.rc" ) &
  CHALLENGER_BUILD=$!
}

challenger_built() {
  # a bare `wait` in the pod (hidden_zk's setup) may have reaped it already; its status is in build.rc either way
  [ -n "${CHALLENGER_BUILD:-}" ] && wait "$CHALLENGER_BUILD" 2>/dev/null
  [ "$(cat "$CHALLENGER_WORK/build.rc" 2>/dev/null)" = 0 ] && return 0
  echo "challenger: the sampled-proofs build failed: $(tail -5 "$CHALLENGER_WORK/build.log" 2>/dev/null)" >&2
  return 1
}

challenger_start() {
  local dir=$1 name; name=$(basename "$1")
  mkdir -p "$dir" "$CHALLENGER_WORK/$name"
  CHALLENGER_SECRET=$CHALLENGER_WORK/$name/secret
  "$CHALLENGER_PY" -c 'import secrets; print(secrets.token_hex(32))' > "$CHALLENGER_SECRET" || return 1
  "$CHALLENGER_PY" "$CHALLENGER_REPO/verity/composite/device/live_challenger/serve.py" --out "$dir" --port 0 \
    --scheme verity/flock/tree-from/v1 --families '[["positions", 1]]' --session "$name" --key "$CHALLENGER_WORK/$name/key" \
    --params "${CHALLENGER_PARAMS:-4,3}" --keygen-workers 1 --secret-file "$CHALLENGER_SECRET" \
    > "$dir/challenger.out" 2> "$dir/challenger.err" &
  echo $! > "$CHALLENGER_WORK/$name/pid"
  until grep -q '^listening ' "$dir/challenger.out" 2>/dev/null; do
    kill -0 "$(cat "$CHALLENGER_WORK/$name/pid")" 2>/dev/null || break
    sleep 1
  done
  CHALLENGER=$(sed -n 's/^listening //p' "$dir/challenger.out")
  if [ -z "$CHALLENGER" ]; then
    echo "challenger: $name never listened: $(tail -3 "$dir/challenger.err")" >&2
    return 1
  fi
  TREE_FROM=(--tree-from "$CHALLENGER" --tree-secret-file "$CHALLENGER_SECRET")
}

challenger_end() {
  local pid; pid=$(cat "$CHALLENGER_WORK/$(basename "$1")/pid")
  wait "$pid"
}

challenger_audit() {
  _challenger_trees audit "$@"
}

challenger_cases() {
  _challenger_trees cases "$@"
}

_challenger_trees() {
  "$CHALLENGER_PY" - "$CHALLENGER_SP/.lake/build/bin/sampled-proofs-difftest" "$@" <<'EOF'
import copy, json, subprocess, sys
from pathlib import Path
from verity.protocols.spot.device import challenger as C, remote as R

difftest, mode, ch, *rest = sys.argv[1:]
log = R.log_lines(Path(ch) / R.RELEASES)
trees = C.coin_trees_of(log)


def record(d):
    return json.loads((Path(d) / "session.json").read_text())


def tables(rec):
    link = rec.get("link") if isinstance(rec, dict) else None
    return dict.fromkeys(link["root_b"]) if isinstance(link, dict) and isinstance(link.get("root_b"), dict) else {}


def reasons(cases):
    asks = [json.dumps({"fn": "challenger_session_trees", "input": {
        "log": log, "records": [{"name": n, "record": r, "tables": tables(r)} for n, r in rs]}}) for rs in cases.values()]
    answers = subprocess.run([difftest], input="".join(a + "\n" for a in asks), capture_output=True, text=True,
                             check=True).stdout.splitlines()
    if len(answers) != len(asks):
        sys.exit(f"challenger: {difftest} answered {len(answers)} lines to {len(asks)} requests")
    return {c: {"python": C.sessions_tree_reason(trees, [(n, tables(r), r) for n, r in rs]),
                "lean": json.loads(a).get("output", {}).get("reason", a)} for (c, rs), a in zip(cases.items(), answers)}


def bumped(h):
    return h[:-1] + "0123456789abcdef"[("0123456789abcdef".index(h[-1]) + 1) % 16]


if mode == "audit":
    names = [Path(s).name for s in rest]
    got = reasons({"audit": [(n, record(s)) for n, s in zip(names, rest)]})["audit"]
    audit = {"sessions": names, **got, "ok": got["python"] is None and got["lean"] is None}
    (Path(ch) / "audit.json").write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps(audit))
    sys.exit(0 if audit["ok"] else 1)
if mode != "cases":
    sys.exit(f"challenger: no audit {mode!r}")
tree, own, out = rest
rec = {"tree-from": record(tree), "own-tree": record(own)}
coin, point = copy.deepcopy(rec["tree-from"]), copy.deepcopy(rec["tree-from"])
s = next(s for s in coin["streams"] if s["rounds"])
s["rounds"][0]["coins"][0] = bumped(s["rounds"][0]["coins"][0])
point["link"]["points"][0][0] = bumped(point["link"]["points"][0][0])
cases = {"tree-from": [("tree-from", rec["tree-from"])], "own-tree": [("own-tree", rec["own-tree"])],
         "both": [("tree-from", rec["tree-from"]), ("own-tree", rec["own-tree"])],
         "coin-changed": [("tree-from", coin)], "point-changed": [("tree-from", point)]}
audit = {"trees": len(trees), "opened": {root: len(t.opened) for root, t in trees.items()},
         "hellos_refused": len(R.hello_lines(Path(ch) / R.HELLOS) or []), "cases": reasons(cases)}
Path(out).write_text(json.dumps(audit, indent=1) + "\n")
print(json.dumps(audit))
EOF
}
