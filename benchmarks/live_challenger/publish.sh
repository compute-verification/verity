#!/usr/bin/env bash
# Publish the auditor host's pins log, its ledger and every ended session to the store (auditor.py's "The ledger"). The control pod runs it as
# the watchers loop `challenger-publish` (infra/control/watchers.toml), so no agent VM has to be up for it:
#
#   benchmarks/live_challenger/publish.sh [--every SECONDS] [DEST]     # DEST: the auditor host's gate, research@204.12.178.84
#
# It reads both through the gate's `ledger` and `fetch SESSION` verbs: as research, with a key the host forces to the gate (the
# pod's RunPod key, which reads only), or as vy-admin, through `sudo vy-challenger-gate`. $VY_CHALLENGER_SSH_KEY is the ssh key
# when set, else ssh's config picks it (the pod's bin/ssh). The ledger is put in the store (kind challenger-ledger/v2,
# preserved, v1 while it holds only v1 entries) each time it has grown since the last put, and labelled `challenger-ledger
# head`; a ledger that isn't the last put one with entries appended (a fork, a cut or an edit) is refused, exit 4, and
# nothing is put until it is resolved by hand. Each session with an end entry is put once, `--kind challenger-session/v1 --ref
# ledger=art:...`, its tree the session's request.json and out/. What was put is kept in $VY_CHALLENGER_STATE (default
# ~/.local/state/vy-challenger: ledger.jsonl, ledger.sha512, ledger.art, sessions, and pins.* as ledger.*), so a rerun puts only what is new; an
# artifact's id is its manifest's hash, so two publishers putting the same ledger get the same art. With --every, each pass is
# its own run of this script, started SECONDS after the last one ended; a failed pass is printed and tried again at the next,
# and after each pass the store's copies already verified on its remote are evicted until $VY_CHALLENGER_FREE_GB (8) GB is free. A
# verifier reads the ledger's art, checks it (auditor.verify_ledger) and takes a window's one session from it. The pins log
# (the gate's `pins`) is put first, the same way (`--kind challenger-pins/v1`, label `challenger-pins head`, a log that isn't
# the last put one extended refused, exit 4), so no published open seals lines the store lacks; `pinned_session` reads it.
# The model registry (the gate's `models`) follows it the same way (`--kind challenger-models/v1`, `challenger-models head`).
set -euo pipefail
if [ "${1:-}" = --every ]; then
  every=$2
  shift 2
  while :; do
    bash "$0" "$@" || { rc=$?; echo "$(date -u +%FT%TZ) pass failed (rc=$rc); next in ${every}s"; }
    # every session's tree stays in the local store after its put, and the control pod's root is a 20 GiB overlay
    research data evict --target-free-gb "${VY_CHALLENGER_FREE_GB:-8}" > /dev/null \
      || echo "$(date -u +%FT%TZ) evict failed (rc=$?); the store keeps its copies until the next pass"
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
gate() { "${SSH[@]}" "$DEST" "${GATE[@]}" "$@" < /dev/null; }  # ssh would eat the session loop's stdin
put() { local a; a=$(research data put "$@" --preserve); echo "art:${a#art:}"; }

# an append-only log the gate reads with verb $1, put as challenger-$2/v1 and labelled challenger-$2 head each time it grew
log() {
  local verb=$1 name=$2 digest art
  gate "$verb" > "$W/$name.jsonl" 2> /dev/null && [ -s "$W/$name.jsonl" ] || return 0
  digest=$(sha512sum "$W/$name.jsonl" | cut -d' ' -f1)
  [ "$digest" != "$(cat "$ST/$name.sha512" 2> /dev/null)" ] || return 0
  if [ -e "$ST/$name.jsonl" ] && ! cmp -s -n "$(stat -c %s "$ST/$name.jsonl")" "$ST/$name.jsonl" "$W/$name.jsonl"; then
    echo "$(date -u +%FT%TZ) refused: the host's $name log ($digest) doesn't extend the last published head" \
      "$(cat "$ST/$name.art" 2> /dev/null) (a cut or an edit): nothing put; infra must look" >&2
    exit 4
  fi
  art=$(put --kind "challenger-$name/v1" --file "$W/$name.jsonl" \
    --meta "{\"host\": \"vy-challenger-1\", \"lines\": $(wc -l < "$W/$name.jsonl"), \"sha512\": \"$digest\"}")
  research data label "$art" "challenger-$name" head --by infra --off-vocab > /dev/null
  cp "$W/$name.jsonl" "$ST/$name.jsonl.tmp"
  mv "$ST/$name.jsonl.tmp" "$ST/$name.jsonl"
  echo "$art" > "$ST/$name.art"
  echo "$digest" > "$ST/$name.sha512"
  echo "$(date -u +%FT%TZ) $name: $art ($digest)"
}
log pins pins
log models models

gate ledger > "$W/windows.jsonl"
digest=$(sha512sum "$W/windows.jsonl" | cut -d' ' -f1)
if [ -s "$W/windows.jsonl" ] && [ "$digest" != "$(cat "$ST/ledger.sha512" 2> /dev/null)" ]; then
  if [ -e "$ST/ledger.jsonl" ] && ! cmp -s -n "$(stat -c %s "$ST/ledger.jsonl")" "$ST/ledger.jsonl" "$W/windows.jsonl"; then
    echo "$(date -u +%FT%TZ) refused: the host's ledger ($digest) doesn't extend the last published head" \
      "$(cat "$ST/ledger.art" 2> /dev/null) (a fork, a cut or an edit): nothing put; infra must look" >&2
    exit 4
  fi
  read -r kind meta < <(python3 -c 'import json, sys
e = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print("challenger-ledger/" + e[-1]["schema"].rsplit("/", 1)[1], json.dumps({
    "host": "vy-challenger-1", "entries": len(e), "sha512": sys.argv[2],
    "key_sha512": sorted({x["body"]["key_sha512"] for x in e if x["kind"] == "open"})}))' "$W/windows.jsonl" "$digest")
  art=$(put --kind "$kind" --file "$W/windows.jsonl" --meta "$meta")
  research data label "$art" challenger-ledger head --by infra --off-vocab > /dev/null
  cp "$W/windows.jsonl" "$ST/ledger.jsonl.tmp"
  mv "$ST/ledger.jsonl.tmp" "$ST/ledger.jsonl"
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
