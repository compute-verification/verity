#!/usr/bin/env bash
# Configurations of served_profile.sh one after another in one run, each in a lease of its own, alternating (ABAB) so that
# host drift spreads over both; each writes into $RESEARCH_RUN_DIR/<label>-<round>.  A configuration's TREE runs another tree on
# the node (a shipped /workspace/research/src/<sha>), with its own SHIP_TAR:
#   research run ... --send served_ab.sh --send served_profile.sh --send served_profile.py -- \
#     taskset -c 160-175 bash "$RESEARCH_RUN_DIR"/inputs/served_ab.sh ROUNDS "label:K=V,K=V" "label:K=V" ...
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROUNDS=$1; shift
for r in $(seq 1 "$ROUNDS"); do
  for spec in "$@"; do
    label=${spec%%:*}; envs=${spec#*:}
    ( IFS=,; for kv in $envs; do [ -n "$kv" ] && export "$kv"; done
      cd "${TREE:-$PWD}" && OUT=$RESEARCH_RUN_DIR/$label-$r bash "$HERE/served_profile.sh" ) || echo "served_ab: $label-$r failed ($?)"
  done
done
