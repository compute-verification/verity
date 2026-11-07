#!/usr/bin/env bash
# The composite-only rehearsal on 591e's kept inputs on cpu-11 (rehearse.sh, mem_series.py sent beside it) at the tree
# SRC, with no new coins: the drawn member m50611 (RoPEHead_v1{D=64}/c2436) with its completed public file pinned and
# its staged entry relaid false, the members' layout written into the pin and the case (LAYOUT=n), and the onsite record
# after the composite (RECORD=1). Exit 0 when v2 accepts, the public composite line is written and the record's
# registered-files check holds; rehearse.txt in the run's out has every line.
#   SRC=CLEAN_WORKTREE [AS=LANE] [COMPOSITE_TIMEOUT=60m] bash launch.sh
# COMPOSITE_TIMEOUT: 60m where the tree has Record.matchesPublic's fast form or v2 no longer calls it, else about 11h.
set -uo pipefail
: "${SRC:?a clean worktree at the head to rehearse}"
git -C "$SRC" diff --quiet HEAD || { echo "REFUSED: $SRC isn't clean"; exit 3; }
echo "SRC $SRC at $(git -C "$SRC" rev-parse HEAD)"
cd "$(dirname "$0")" && exec research run --as "${AS:-network-accounting}" --on vy-nebius-cpu-11 --project verity \
  --source "$SRC" --cwd clone --timeout 3h --custody-ttl 24h --mem-gb 64 \
  --env KEEP=/workspace/incoming/netacct/r20261106-152723-4a4c/keep-591e --env V=v3c-w0-8-32-4-4-16 --env M=m50611 \
  --env PIN=f0faf66c5f3dfae0aa41aa8bfb6267df77129124f3020a8ed0cba772dab5a04c05e32ecff01305a427531e5abfcbeb4d03f9987805f626b1ef66e3dc8f8ac155 \
  --env COMPLETED=aa59233a894d4c2afc895c422b3d96df2c7e4df5d8d2f357a40b62188df91aa4806e211560b632d68d59b68f641b75e38a76ba230c915f45d8689f4db777b5f0:41f820bc49b74d21460addeaefe5be95d3e79cb4d0e7472e3664984b5f6b29b7d21dff9556c16d732b7e4a597572054ab6302da7064805ab80948f1117aeb5db \
  --env RELAID=false --env LAYOUT=n --env RECORD=1 --env COMPOSITE_TIMEOUT="${COMPOSITE_TIMEOUT:-60m}" \
  --send rehearse.sh --send mem_series.py -- bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/rehearse.sh"'
