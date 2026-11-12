#!/usr/bin/env bash
# Every off-card rehearsal of the sampled accept's capture in one job, from the root of the tree that holds the capture:
#   research run --on auto --project verity --source TREE --cwd source --timeout 20m --mem-gb 8 --send rehearse_all.sh \
#     --send capture_dry.sh --send capture_dry.py --send ptx_check.py --send ptx_sim.py --send accept_ptx.tgz \
#     --send accept_capture.sh --send accept_capture_dry.sh --send accept_replay.sh --send capture_rehearsed.tgz \
#     -- bash -c 'bash $RESEARCH_RUN_DIR/inputs/rehearse_all.sh'
# 1. capture_dry.sh: the capture's Python with no card (its launches bind and compile the read PTX and machine code; its
#    record and --replay on an interpreter's words).  It ends REHEARSAL OK.
# 2. accept_capture_dry.sh: the card script's shell path (the checks before the card, the lease with its cores, the quota's
#    refusal) with a fake nvidia-smi and a fake venv.  It ends SHELL REHEARSAL OK.
# 3. accept_replay.sh on capture_rehearsed.tgz, the three cases an earlier rehearsal kept (an interpreter's words): the
#    off-card replay as the capture's run will be replayed.  It ends SAMPLED-ACCEPT GO.
# Exit 0 when all three did.
set -uo pipefail
I="$RESEARCH_RUN_DIR/inputs"
( cd "$I" && sha256sum -- * )
echo "=== 1. capture_dry.sh"; bash "$I/capture_dry.sh"; a=$?
echo "=== 2. accept_capture_dry.sh"; bash "$I/accept_capture_dry.sh"; b=$?
echo "=== 3. accept_replay.sh"; bash "$I/accept_replay.sh" "$I/capture_rehearsed.tgz"; c=$?
echo "=== steps: capture_dry $a, accept_capture_dry $b, accept_replay $c"
[ "$a" = 0 ] && [ "$b" = 0 ] && [ "$c" = 0 ]
