# Sourced by a pod whose `flock-circuit serve` takes its sessions' coin trees from the auditor's challenger
# (`serve --tree-from`): benchmarks/live_challenger/serve.py, its coins the Lean verifier's `IO.getRandomBytes`
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
# env: CHALLENGER_REPO (default the working directory: a verity tree), CHALLENGER_PY (default python3, with the tree's
#      Python importable), CHALLENGER_WORK (default `mktemp -d`: the key and the secret), CHALLENGER_PARAMS (the LMS key's
#      height and Winternitz parameter, default 4,3: a test key)
CHALLENGER_REPO=${CHALLENGER_REPO:-$PWD}
CHALLENGER_PY=${CHALLENGER_PY:-python3}
CHALLENGER_WORK=${CHALLENGER_WORK:-$(mktemp -d)}
CHALLENGER_SP=$CHALLENGER_REPO/verity/core/service/sampled_proofs

challenger_build() {
  command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
  local lock=$CHALLENGER_REPO/verity/core/difftest/lean/.lake/harness-build.lock
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
  "$CHALLENGER_PY" "$CHALLENGER_REPO/benchmarks/live_challenger/serve.py" --out "$dir" --port 0 \
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
  local dir=$1; shift
  "$CHALLENGER_PY" - "$dir" "$CHALLENGER_SP/.lake/build/bin/sampled-proofs-difftest" "$@" <<'EOF'
import json, subprocess, sys
from pathlib import Path
from verity.core.protocols.sampled_proofs import challenger as C, remote as R

ch, difftest, *sessions = sys.argv[1:]
log = R.log_lines(Path(ch) / R.RELEASES)
records = [(Path(s).name, json.loads((Path(s) / "session.json").read_text())) for s in sessions]
py = C.sessions_tree_reason(C.coin_trees_of(log), records)
line = json.dumps({"fn": "challenger_session_trees",
                   "input": {"log": log, "records": [{"name": n, "record": r} for n, r in records]}})
answer = subprocess.run([difftest], input=line + "\n", capture_output=True, text=True, check=True).stdout
lean = json.loads(answer).get("output", {}).get("reason", answer)
audit = {"sessions": [n for n, _ in records], "python": py, "lean": lean, "ok": py is None and lean is None}
(Path(ch) / "audit.json").write_text(json.dumps(audit, indent=1) + "\n")
print(json.dumps(audit))
sys.exit(0 if audit["ok"] else 1)
EOF
}
