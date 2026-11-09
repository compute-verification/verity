#!/bin/bash
# vy-pous-quiet CMD...: run CMD on vy-nebius-2's cores 124-127, in a scope of pous-quiet.slice, for memory accounting's PoUS timed
# runs and its untimed jobs on GPU 7. user.slice and system.slice, so
# every other process on the host, are confined to 0-123; provers.slice (vy-provers) holds 128-191.
# Use it under the GPU lease: gpu-lease 1 --on 7 --max-min M [--timed] -- vy-pous-quiet CMD...
set -euo pipefail
[ $# -gt 0 ] || { echo "usage: vy-pous-quiet CMD..." >&2; exit 2; }
exec sudo -n --preserve-env env PATH="$PATH" systemd-run --quiet --scope --collect --slice=pous-quiet.slice \
  -p AllowedCPUs="${VY_POUS_QUIET_CPUS:-124-127}" --uid="$(id -u)" --gid="$(id -g)" -- "$@"
