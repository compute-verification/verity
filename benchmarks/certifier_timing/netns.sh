#!/usr/bin/env bash
# One case of certifier_timing with the observer in a network namespace of its own, on one host and without root.
#
#   [INTERVALS=3000] [OBSERVER=spin] [CERTIFIER=warden] netns.sh OUT [serve's other flags]
#
# A user namespace holds two network namespaces joined by a veth pair: the server and the certifier in one, the observer
# in the other, as `serve` and `observe` on two hosts would be, but on one clock and one host's cores. The certifier's
# bytes leave by a device of their own, `cert0`, which a launch gate can sit on (`--launch cert0,AFTER,OPEN`); loopback
# takes none. It leaves the case in OUT/case and its score in OUT/result.json, and nothing outside its namespaces.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
if [ "${1:-}" != --inside ]; then
    exec unshare -rn bash "$0" --inside "$@"
fi
out=$2
shift 2
py=${PYTHON:-python3}
intervals=${INTERVALS:-3000}
certifier=${CERTIFIER:-warden}
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
"$py" "$here/run.py" serve --listen "$address" --port "$port" --wait-s 60 --intervals "$intervals" \
    --certifier "$certifier" --out "$out/case" "$@" &
serve=$!
nsenter -t "$peer" -n "$py" "$here/run.py" observe "$address:$port" --wait-s 60 --intervals "$intervals" \
    --certifier "$certifier" --observer "${OBSERVER:-spin}" --out "$out/case"
wait "$serve"
"$py" "$here/run.py" score "$out/case" | tee "$out/result.json"
