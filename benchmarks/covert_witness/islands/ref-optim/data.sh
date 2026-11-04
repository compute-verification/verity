#!/usr/bin/env bash
# data.sh FILES: the ladder's stream (ladder_data.py, its first FILES training files) at $LADDER_DATA on this node, one
# worker for every two of the slot's cores. A rerun keeps the parts already built.
#
#   research run --on NODE --project verity --source . --pool POOL --as training --timeout 4h --declared-output 'out/*' \
#     -- bash benchmarks/covert_witness/islands/ref-optim/data.sh 57
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$(mktemp -d)}
. "$here/pyenv.sh"
DATA=${LADDER_DATA:-/workspace/research/scratch/te-ladder/stream}
workers=$(( $(nproc) / 2 ))
[ "$workers" -ge 1 ] || workers=1
mkdir -p "$DATA"
df -h "$DATA" | tail -n 1
exec "$P" "$here/ladder_data.py" --dest "$DATA" --files "$1" --workers "$workers" --threads 2 --manifest-out "$OUT"
