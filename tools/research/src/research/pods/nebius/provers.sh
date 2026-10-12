#!/bin/bash
# vy-provers CMD...: run CMD on vy-nebius-2's prover cores, 128-191, in a scope of provers.slice (infra, verity-top's ruling of 1 Oct
# 07:00Z, until 17:00Z: extended by top-level at 07:40Z). user.slice and system.slice, so every other process on the host, are confined to 0-127.
# CMD keeps this user, working directory and environment (CUDA_VISIBLE_DEVICES from gpu-lease included). hold_slices
# (verity/ml/flock/pod/cpu-slices.sh) sees the range as four 16-CPU slices: 128-143, 144-159, 160-175, 176-191.
set -euo pipefail
[ $# -gt 0 ] || { echo "usage: vy-provers CMD..." >&2; exit 2; }
exec sudo -n --preserve-env env PATH="$PATH" systemd-run --quiet --scope --collect --slice=provers.slice \
  -p AllowedCPUs="${VY_PROVERS_CPUS:-128-191}" --uid="$(id -u)" --gid="$(id -g)" -- "$@"
