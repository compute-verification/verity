#!/usr/bin/env bash
# bw_session.sh's two sessions at fixture scale with the verifier on a cluster peer, in the shape vy-auditor-2's gate will
# run it: `p2_party --listen 127.0.0.1:0 --sessions 2 --answer` on PEER (its line in /etc/vy/peers, the vy-cluster key),
# the prover here reaching it through `ssh -L`, the honest session and then the cheat. The prover serves the toy's C*
# (`bandwidth.py toy`) through kernels' staging decode with P2's reference decode on the CPU; the peer builds the toy's
# registration itself and holds that, never the prover's. PEER is the lane's machine, not the auditor's, so the run
# certifies nothing. cwd = a clone:
#
#   research run --on auto --project verity --campaign e2e-demo --source . --cwd clone --timeout 30m -- \
#     bash benchmarks/pous/p2_v1/bw_peer.sh PEER [PEER ...]
#
# PEER is the first of those named that ssh reaches and that isn't this host. $OUT/<session>/ has the prover's
# session.json, $OUT/verifier-answers.jsonl the verifier's own record of both answers, and $OUT/verdicts.txt each
# session's verdict line with the verifier's record agreeing with it. The peer's scratch is removed on exit.
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
{ hostname; echo "$PEER ${TO[-1]}"; git rev-parse HEAD; } > "$O/hosts.txt"
: > "$O/verdicts.txt"

coproc V { "${SSH[@]}" "cd $R && exec .venv/bin/python -m verity.core.protocols.remote_bandwidth.p2_party --registration $R/toy/registration.json --listen 127.0.0.1:0 --sessions 2 --answer $R/answers.jsonl" 2> "$O/verifier.err"; }
read -r line <&"${V[0]}"
port=$(python3 -c 'import json, sys; print(json.loads(sys.argv[1])["listening"])' "$line")
lport=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"${SSHO[@]}" -f -N -M -S "$S/ctl" -o ExitOnForwardFailure=yes -L "127.0.0.1:$lport:127.0.0.1:$port" "${TO[@]}"
echo "verifier on $PEER at its 127.0.0.1:$port, here at 127.0.0.1:$lport" >> "$O/hosts.txt"

session() {  # NAME EXPECT [bandwidth.py session options]
  local name=$1 expect=$2
  shift 2
  uv run --no-sync python benchmarks/pous/p2_v1/bandwidth.py session --store "$S/toy/store" --layout "$S/toy/layout.json" \
    --node p2-toy-node --registration "$S/toy/registration.json" --connect "127.0.0.1:$lport" --device cpu --ref-decode \
    --workers 4 --out "$O/$name" --expect "$expect" "$@" > "$O/$name.out" 2> "$O/$name.err"
}

rc=0
session honest accept --windows 2 --passes 2 --samples 16 --window-s 1 --deadline-s 1 || rc=1
session cheat reject --windows 1 --passes 1 --samples 8 --window-s 1 --deadline-s 1 --skip-layer 1 || rc=1
wait "$V_PID" || rc=1
"${SSH[@]}" cat "$R/answers.jsonl" > "$O/verifier-answers.jsonl"
python3 - "$O" >> "$O/verdicts.txt" <<'EOF' || rc=1
import json, sys
o = sys.argv[1]
mine = [json.loads(x) for x in open(f"{o}/verifier-answers.jsonl")]
bad = len(mine) != 2
for k, (name, expect) in enumerate((("honest", "accept"), ("cheat", "reject"))):
    theirs = json.load(open(f"{o}/{name}/session.json"))
    same = k < len(mine) and mine[k] == theirs["answer"]
    print(theirs["verdict_line"], f"verifier_record_agrees={same}")
    bad |= not same or mine[k]["verdict"] != expect
sys.exit(1 if bad else 0)
EOF
cat "$O/verdicts.txt"
exit $rc
