#!/usr/bin/env bash
# Publish the auditor host's ledger and every ended session to the store, from infra's machine (auditor.py's "The ledger"):
#
#   benchmarks/live_challenger/publish.sh [DEST]     # DEST: vy-admin@204.12.178.84 (vy-challenger-1)
#
# It reads both through the gate (`sudo vy-challenger-gate ledger`, `fetch SESSION`) over vy-admin's ssh ($VY_CHALLENGER_SSH_KEY,
# default ~/.ssh/vy-challenger-admin). The ledger is put (`research data put --kind challenger-ledger/v1 --preserve`) each time it
# has grown since the last put; each session with an end entry is put once, `--kind challenger-session/v1 --ref ledger=art:...`,
# its tree the session's request.json and out/. What was put is kept in ~/.local/state/vy-challenger/ (ledger.sha512, ledger.art,
# sessions), so a rerun puts only what is new. A verifier reads the ledger's art, checks it (auditor.verify_ledger) and takes a
# window's one session from it.
set -euo pipefail
DEST=${1:-vy-admin@204.12.178.84}
KEY=${VY_CHALLENGER_SSH_KEY:-$HOME/.ssh/vy-challenger-admin}
ST=$HOME/.local/state/vy-challenger
mkdir -p "$ST"
touch "$ST/sessions"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
gate() { ssh -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes "$DEST" sudo -n /usr/local/sbin/vy-challenger-gate "$@"; }
put() { local a; a=$(research data put "$@" --preserve); echo "art:${a#art:}"; }

gate ledger > "$W/windows.jsonl"
digest=$(sha512sum "$W/windows.jsonl" | cut -d' ' -f1)
if [ -s "$W/windows.jsonl" ] && [ "$digest" != "$(cat "$ST/ledger.sha512" 2> /dev/null)" ]; then
  meta=$(python3 -c 'import json, sys
e = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(json.dumps({"host": "vy-challenger-1", "entries": len(e), "sha512": sys.argv[2],
                  "key_sha512": sorted({x["body"]["key_sha512"] for x in e if x["kind"] == "open"})}))' "$W/windows.jsonl" "$digest")
  put --kind challenger-ledger/v1 --file "$W/windows.jsonl" --meta "$meta" > "$ST/ledger.art"
  echo "$digest" > "$ST/ledger.sha512"
  echo "ledger: $(cat "$ST/ledger.art") ($digest)"
fi
[ -s "$ST/ledger.art" ] || exit 0

python3 -c 'import json, sys
for l in open(sys.argv[1]):
    if l.strip() and json.loads(l)["kind"] == "end":
        print(json.loads(l)["session"])' "$W/windows.jsonl" | while read -r s; do
  grep -qx "$s" "$ST/sessions" && continue
  mkdir -p "$W/$s"
  gate fetch "$s" | tar -xz -C "$W/$s" --no-same-owner
  art=$(put --kind challenger-session/v1 --tree "$W/$s/$s" --ref "ledger=$(cat "$ST/ledger.art")" \
    --meta "{\"session\": \"$s\", \"host\": \"vy-challenger-1\"}")
  echo "$s" >> "$ST/sessions"
  echo "session $s: $art"
done
