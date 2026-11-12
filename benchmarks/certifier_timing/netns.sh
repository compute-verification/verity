#!/usr/bin/env bash
# certifier_timing's cases with the observer in a network namespace of its own, on one host and without root.
#
#   netns.sh OUT [serve's flags]
#
# A user namespace holds two network namespaces joined by a veth pair: the server and the certifier in one, the observer
# in the other, as `pair.sh`'s two sides on two hosts would be (its cases, and its INTERVALS, ATTACKS and OBSERVERS),
# but on one clock and one host's cores. The certifier's bytes leave by a device of their own, `cert0`, which a launch
# gate can sit on (`--launch cert0,AFTER,OPEN[,MODE]`); loopback takes none. It leaves the cases in OUT/out, their
# observers' files in OUT/seen and the score in OUT/result.json, and nothing outside its namespaces.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
if [ "${1:-}" != --inside ]; then
    exec unshare -rn bash "$0" --inside "$@"
fi
out=$2
shift 2
address=10.77.0.1
port=47812

ip link set lo up
unshare -n sleep infinity &
peer=$!
trap 'kill "$peer" 2>/dev/null || true' EXIT
until [ "$(readlink "/proc/$peer/ns/net")" != "$(readlink /proc/self/ns/net)" ]; do :; done
ip link add cert0 numtxqueues 2 numrxqueues 2 type veth peer name obs0 numtxqueues 2 numrxqueues 2 netns "$peer"
ip addr add "$address/24" dev cert0
ip link set cert0 up
nsenter -t "$peer" -n ip addr add 10.77.0.2/24 dev obs0
nsenter -t "$peer" -n ip link set obs0 up
nsenter -t "$peer" -n ip link set lo up

mkdir -p "$out"
start=$(($(date +%s) + 3))
bash "$here/pair.sh" serve "$address" "$port" "$start" "$out" "$@" &
serve=$!
nsenter -t "$peer" -n bash "$here/pair.sh" observe "$address" "$port" "$start" "$out"
wait "$serve"
"${PYTHON:-python3}" "$here/run.py" score "$out"/out/* --observer "$out/seen" | tee "$out/result.json"
