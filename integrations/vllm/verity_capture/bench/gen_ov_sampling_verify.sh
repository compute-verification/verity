#!/usr/bin/env bash
# [ov-sampling] sealed verification of one configuration: the CORE checkout's verify_lane.sh judges `lane/ov-sampling` in a fresh
# sparse worktree and re-runs the CPU phase on the pod over a FRESH copy of the configuration's GPU artifacts
# (/workspace/cp/runs/ov-sampling/verify_<cfg>: hardlinked capture/control/census, copied nonint, the lane profile's run.json), so the
# lane's own `<cfg>_ovs` outputs stay untouched.  The local run dir receives the sealed card + reports.
#
#   bash verity_capture/bench/gen_ov_sampling_verify.sh <cfg> [<cfg> ...]        # sequential; run two shells for two at a time
set -uo pipefail
CORE=${CORE:-/private/tmp/lane-capture}   # the merged core (lane/capture): verify_lane.sh + protected.py are taken from here, never from the lane
LANE_WT=${LANE_WT:-/private/tmp/lane-sampling}
SSHF=$LANE_WT/out/capture/pods/cp-ov-sampling.ssh
RUNS=/workspace/cp/runs/ov-sampling
BASE=${BASE:-swarm-base-20260917}
# verify_lane.sh step 2 runs the lane's pytest under coreutils `timeout`, which macOS lacks: without it the tests "fail" and the
# quarantine admission is stamped negative_matcher_tests=fail.  Supply a minimal shim (same semantics: exit 124 on expiry).
if ! command -v timeout >/dev/null 2>&1; then
  SHIM=$(mktemp -d /tmp/ovs_bin.XXXX)
  printf '%s\n' '#!/usr/bin/env python3' 'import subprocess, sys' 'try:' '    sys.exit(subprocess.run(sys.argv[2:], timeout=float(sys.argv[1])).returncode)' \
    'except subprocess.TimeoutExpired:' '    sys.exit(124)' > "$SHIM/timeout"; chmod +x "$SHIM/timeout"; export PATH=$SHIM:$PATH
fi
for cfg in "$@"; do
  local_dir=$LANE_WT/out/gen/lanes/ov-sampling/verify/$cfg
  mkdir -p "$local_dir"
  cp "$LANE_WT/out/gen/lanes/ov-sampling/runs/${cfg}_ovs/quarantine.json" "$local_dir/quarantine.json" 2>/dev/null || true
  bash "$SSHF" "src=$RUNS/$cfg; dst=$RUNS/verify_$cfg; rm -rf \$dst; mkdir -p \$dst; for d in capture control census; do [ -e \$src/\$d ] && cp -al \$src/\$d \$dst/\$d; done; [ -e \$src/nonint ] && cp -a \$src/nonint \$dst/nonint; cp $RUNS/${cfg}_ovs/run.json \$dst/run.json; [ -e \$src/gpu_done.json ] && cp \$src/gpu_done.json \$dst/; ls \$dst | tr '\n' ' '" 2>&1 | grep -v Warning
  echo "[verify $(date -u +%FT%TZ)] $cfg: verify_lane.sh from $CORE (base $BASE) over pod dir $RUNS/verify_$cfg"
  ( cd "$CORE" && bash verity_capture/bench/verify_lane.sh lane/ov-sampling "$local_dir" --base "$BASE" --pod "$SSHF" --pod-run-dir "$RUNS/verify_$cfg" --pod-parallel 8 ) > "$local_dir/verify_driver.log" 2>&1
  echo "[verify $(date -u +%FT%TZ)] $cfg: rc=$? verdict=$(python3 -c "import json;print(json.load(open('$local_dir/card.json')).get('verdict'))" 2>/dev/null)"
done
