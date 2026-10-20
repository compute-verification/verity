#!/usr/bin/env bash
# Publish the auditor host's ledger and every ended session to the store (auditor.py's "The ledger"). The control pod runs it as
# the watchers loop `challenger-publish` (infra/control/watchers.toml), so no agent VM has to be up for it:
#
#   benchmarks/live_challenger/publish.sh [--every SECONDS] [DEST]     # DEST: research@204.12.178.84 (vy-challenger-1)
#
# It reads both through the gate's `ledger` and `fetch SESSION` verbs: as research, whose every key the host forces to the gate (the
# pod's RunPod key is the one there), or as vy-admin, through `sudo vy-challenger-gate`. $VY_CHALLENGER_SSH_KEY is the ssh key when
# set, else ssh's config picks it (the pod's bin/ssh). The ledger is put (`research data put --kind challenger-ledger/v1
# --preserve`) each time it has grown since the last put; each session with an end entry is put once, `--kind
# challenger-session/v1 --ref ledger=art:...`, its tree the session's request.json and out/. What was put is kept in
# $VY_CHALLENGER_STATE (default ~/.local/state/vy-challenger: ledger.sha512, ledger.art, sessions), so a rerun puts only what is
# new; an artifact's id is its manifest's hash, so two publishers putting the same ledger get the same art. With --every, each
# pass is its own run of this script, started SECONDS after the last one ended; a failed pass is printed and tried again at the
# next. A verifier reads the ledger's art, checks it (auditor.verify_ledger) and takes a window's one session from it.
set -euo pipefail
if [ "${1:-}" = --every ]; then
  every=$2
  shift 2
  while :; do
    bash "$0" "$@" || echo "$(date -u +%FT%TZ) pass failed (rc=$?); next in ${every}s"
    sleep "$every"
  done
fi
DEST=${1:-research@204.12.178.84}
ST=${VY_CHALLENGER_STATE:-$HOME/.local/state/vy-challenger}
SSH=(ssh -o BatchMode=yes)
[ -z "${VY_CHALLENGER_SSH_KEY:-}" ] || SSH+=(-i "$VY_CHALLENGER_SSH_KEY" -o IdentitiesOnly=yes)
case $DEST in research@*) GATE=() ;; *) GATE=(sudo -n /usr/local/sbin/vy-challenger-gate) ;; esac
mkdir -p "$ST"
touch "$ST/sessions"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
gate() { "${SSH[@]}" "$DEST" "${GATE[@]}" "$@"; }
put() { local a; a=$(research data put "$@" --preserve); echo "art:${a#art:}"; }

gate ledger > "$W/windows.jsonl"
digest=$(sha512sum "$W/windows.jsonl" | cut -d' ' -f1)
if [ -s "$W/windows.jsonl" ] && [ "$digest" != "$(cat "$ST/ledger.sha512" 2> /dev/null)" ]; then
  meta=$(python3 -c 'import json, sys
e = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(json.dumps({"host": "vy-challenger-1", "entries": len(e), "sha512": sys.argv[2],
                  "key_sha512": sorted({x["body"]["key_sha512"] for x in e if x["kind"] == "open"})}))' "$W/windows.jsonl" "$digest")
  art=$(put --kind challenger-ledger/v1 --file "$W/windows.jsonl" --meta "$meta")
  echo "$art" > "$ST/ledger.art"
  echo "$digest" > "$ST/ledger.sha512"
  echo "$(date -u +%FT%TZ) ledger: $(cat "$ST/ledger.art") ($digest)"
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
  echo "$(date -u +%FT%TZ) session $s: $art"
done
