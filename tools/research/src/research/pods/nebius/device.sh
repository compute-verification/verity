#!/bin/bash
# vy-device ROLE CMD...: run one of the auditor's devices on vy-nebius-2, on cores nothing else uses, in a scope of devices.slice:
# the network certifier on 120-123 and the memory challenger, which sits next to it inside the node's isolation boundary, on
# 124-127 (infra item I3 of note:verity-root/20261006T0550Z-report-proof-service-implementation). user.slice and system.slice, so
# every other process on the host, are confined to 0-119 by vy-device-cores (device_cores.sh, which also moves IRQs, unbound
# workqueues and kernel threads off 120-127); provers.slice (vy-provers) holds 128-191. 124-127 are vy-pous-quiet's cores: the
# memory challenger replaces PoUS's timed verifier there.
# A timed run takes its GPU under the lease: gpu-lease 1 --on 7 --max-min M --timed -- vy-device memory-challenger CMD...
set -euo pipefail
case "${1:-}" in
  certifier) cpus="${VY_CERTIFIER_CPUS:-120-123}" ;;
  memory-challenger) cpus="${VY_MEMORY_CHALLENGER_CPUS:-124-127}" ;;
  *) echo "usage: vy-device certifier|memory-challenger CMD..." >&2; exit 2 ;;
esac
shift
[ $# -gt 0 ] || { echo "usage: vy-device certifier|memory-challenger CMD..." >&2; exit 2; }
exec sudo -n --preserve-env env PATH="$PATH" systemd-run --quiet --scope --collect --slice=devices.slice \
  -p AllowedCPUs="$cpus" --uid="$(id -u)" --gid="$(id -g)" -- "$@"
