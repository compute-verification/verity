#!/usr/bin/env bash
# served_zk's served side on an RTX PRO 6000 node, untimed: GPU 1's ship built from this tree (CPU, build.sh), then
# window.sh's smoke (its identity, ship and device checks, both engines resident, the arm's gates) with RETAIN=1, under
# -h2: the arm serves e2e.py's SERVED requests, "request" and "other", each a retained pass under $WORK/passes/<run id>,
# salted by epoch coins drawn from the challenger whose entropy is $WORK/coin-state/<run id>.  The passes stay on the
# node for commit.py and job.sh under window.sh's retention record (priority keep): only `research retention rm`, with
# the owner's yes, deletes them.
#   research run --on MACHINE --project verity --source <tree> --cwd source --declared-output 'out/*' \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/served_zk/serve.sh
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; ID=${RESEARCH_RUN_ID:?}; OUT=$D/out; mkdir -p "$OUT"
WORK=/workspace/pouw/mvp-e2e
S=$(mktemp -d /workspace/pouw/tmp-served-zk.XXXXXX); trap 'rm -rf "$S"' EXIT
HEAD=$(git rev-parse HEAD); echo "tree $HEAD" | tee "$OUT/tree.txt"
stamp() { echo "served-zk: $* at $(date -u +%H:%M:%SZ)"; }
df -h /workspace | tail -1

stamp "ship build"
CUDA=/usr/local/cuda-13.0 bash benchmarks/pouw/pearl_c_sm120/build.sh "$S/ship-${HEAD:0:8}" > "$OUT/build.log" 2>&1 \
  || { stamp "BUILD_FAIL rc=$?"; tail -20 "$OUT/build.log"; exit 11; }
tar -cf "$S/ship.tar" -C "$S" "ship-${HEAD:0:8}"

# the challenger's coins are the Lean verifier's (pouw.verifier.coins): built before the lease, not inside it
command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
stamp "pouw-check-difftest build"
(cd verity/core/protocols/pouw/lean/check && lake build pouw-check-difftest) > "$OUT/lean-coins-build.log" 2>&1 \
  || { stamp "LEAN_BUILD_FAIL rc=$?"; tail -20 "$OUT/lean-coins-build.log"; exit 12; }

export SCHEME=pearl-c-sm120-v1-h3 MODE=smoke RETAIN=1
# the epoch's challenger entropy, which job.sh's challenger (--state) holds too: the pass's salt coin is the one it opens
export POUW_COIN_STATE=$WORK/coin-state/$ID
stamp "asking for one preemptible GPU"
SHIP_TAR=$S/ship.tar OUT=$OUT/window gpu-lease 1 --wait --preemptible --max-min 25 -- \
  timeout 1440 bash benchmarks/pouw/pearl_c_vllm/window.sh
rc=$?; stamp "window.sh exited $rc"
rm -rf "$OUT/window/ship"
du -sh "$WORK/passes/$ID"/* 2>/dev/null
exit "$rc"
