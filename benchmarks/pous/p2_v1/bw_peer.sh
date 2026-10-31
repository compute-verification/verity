#!/usr/bin/env bash
# bw_session.sh's two sessions at fixture scale with the verifier on a cluster peer, in the shape of a party of
# the auditor host's gate: each session starts `p2_party --listen` on PEER over ssh (`bandwidth.py session --party`; PEER's
# line in the node's peers file, the cluster's ssh key) as the gate starts a party (the system python3 -s -E with a cleared
# environment, from the tree, its `--device` directory), whose stdout says its port and then its own record of the
# answer, and the prover reaches that port over TCP. The cluster's nodes pass only ssh to each other, so here the TCP goes through an
# `ssh -L` forward of the same port (the card reaches the auditor host's party port directly). The prover serves the toy's
# C* (`bandwidth.py toy`) through kernels' staging decode with P2's reference decode on the CPU; the peer builds the
# toy's registration itself and holds that, never the prover's. PEER is the lane's machine, not the auditor's, so the
# run certifies nothing. cwd = a clone:
#
#   OUT=DIR bash benchmarks/pous/p2_v1/bw_peer.sh PEER [PEER ...]
#
# PEER is the first of those named that ssh reaches and that isn't this host. $OUT/<session>/session.json is each
# session's record (the verifier's own answer in `verifier_record`), $OUT/verdicts.txt each verdict line, which passes
# only if the verifier's record is the answer the session ended with, and $OUT/party-answers.jsonl the device's own
# log of them. The peer's scratch is removed on exit.
set -euo pipefail
cd "$(dirname "$0")/../../.."
[ $# -gt 0 ] || { echo "usage: bw_peer.sh PEER [PEER ...]"; exit 2; }
O=${OUT:?}
SSHO=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
PEER= TO=()
for p in "$@"; do
  line=$(grep -E "^$p " /etc/vy/peers) || { echo "bw_peer.sh: $p has no line in /etc/vy/peers" >> "$O/peers.txt"; continue; }
  read -r -a to <<< "${line#"$p "}"
  there=$("${SSHO[@]}" "${to[@]}" hostname 2>> "$O/peers.txt") || { echo "$p: unreachable" >> "$O/peers.txt"; continue; }
  [ "$there" = "$(hostname)" ] && { echo "$p: this host" >> "$O/peers.txt"; continue; }
  PEER=$p TO=("${to[@]}"); break
done
[ -n "$PEER" ] || { cat "$O/peers.txt"; echo "bw_peer.sh: no peer reachable"; exit 2; }
SSH=("${SSHO[@]}" "${TO[@]}")
S=$(mktemp -d)
R=$("${SSH[@]}" mktemp -d /tmp/bw-peer-XXXXXX)
trap '[ -S "$S/ctl" ] && "${SSHO[@]}" -S "$S/ctl" -O exit "${TO[@]}" 2>/dev/null; "${SSH[@]}" rm -rf "$R" || true; rm -rf "$S"' EXIT
tar -c --exclude=./.git --exclude=./.venv --exclude=.lake . | "${SSH[@]}" tar -x -C "$R"
uv sync -q --all-packages --extra torch-cpu
"${SSH[@]}" "bash -lc 'cd $R && uv sync -q --all-packages --extra torch-cpu && .venv/bin/python benchmarks/pous/p2_v1/bandwidth.py toy $R/toy'" \
  > "$O/peer-toy.json"
uv run --no-sync python benchmarks/pous/p2_v1/bandwidth.py toy "$S/toy" > "$O/prover-toy.json"
"${SSH[@]}" cat "$R/toy/registration.json" > "$O/verifier-registration.json"
PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"${SSHO[@]}" -f -N -M -S "$S/ctl" -o ExitOnForwardFailure=yes -L "127.0.0.1:$PORT:127.0.0.1:$PORT" "${TO[@]}"
{ hostname; echo "$PEER ${TO[-1]}, its 127.0.0.1:$PORT forwarded to this host's"; git rev-parse HEAD; } > "$O/hosts.txt"
PARTY="${SSH[*]} cd $R '&&' exec /usr/bin/env -i PYTHONDONTWRITEBYTECODE=1 PATH=/usr/bin:/bin /usr/bin/python3 -s -E"
PARTY+=" -m verity.core.protocols.poub.p2_party --device $R/party --listen 127.0.0.1:$PORT"
PARTY+=" --registration $R/toy/registration.json"
: > "$O/verdicts.txt"

session() {  # NAME EXPECT [bandwidth.py session options]
  local name=$1 expect=$2 rc=0
  shift 2
  uv run --no-sync python benchmarks/pous/p2_v1/bandwidth.py session --store "$S/toy/store" --layout "$S/toy/layout.json" \
    --node p2-toy-node --registration "$S/toy/registration.json" --party "$PARTY" --party-host 127.0.0.1 --device cpu \
    --ref-decode --workers 4 --out "$O/$name" --expect "$expect" "$@" > "$O/$name.out" 2> "$O/$name.err" || rc=$?
  tail -n 1 "$O/$name.out" >> "$O/verdicts.txt"
  return $rc
}

rc=0
session honest accept --windows 2 --passes 2 --samples 16 --window-s 1 --deadline-s 1 || rc=1
session cheat reject --windows 1 --passes 1 --samples 8 --window-s 1 --deadline-s 1 --skip-layer 1 || rc=1
"${SSH[@]}" cat "$R/party/answers.jsonl" > "$O/party-answers.jsonl" || rc=1
cat "$O/verdicts.txt"
exit $rc
