#!/usr/bin/env bash
# The settings job on vy-dc-h200-2 for Monday's run, from the run's tree (env.sh's TREE, tree.sh's):
#   bash settings/settings_job_launch.sh
# It sends FILES (default settings-monday.env, beside this script), and settings_job.sh checks and installs each under
# its name in ONSITE (0600 in a 0700 directory), printing no count. GATE_PIN is the auditor gate's pinned commit, COPY_TO
# the V3c follow-ons' nodes (the copy key is write-only there). It waits for the job (10 minutes at most) and prints its
# lines; its run id and log are in LOGS/settings.txt.
set -euo pipefail
K=$(cd "$(dirname "$0")" && pwd)
. "$K/../env.sh"
tree_ok
GATE_PIN=${GATE_PIN:-$CERTIFIER_GATE_PIN} COPY_TO=${COPY_TO:-research@vy-nebius-cpu-11 research@vy-nebius-cpu-12}
FILES=${FILES:-$SETTINGS_NAME}
send=()
for f in $FILES; do
  [ -f "$K/$f" ] || { echo "REFUSED: no settings file $K/$f"; exit 3; }
  send+=(--send "$K/$f")
  echo "FILE $f $(sha256sum "$K/$f" | cut -c1-16)"
done
echo "TREE $(git -C "$TREE" rev-parse HEAD) (= $REF), GATE_PIN $GATE_PIN, COPY_TO $COPY_TO, FILES $FILES"
cmd=(research run --as network-accounting --on "$SERVED" --project verity --source "$TREE" --cwd clone --timeout 10m --wait
  --env GATE_PIN="$GATE_PIN" --env COPY_TO="$COPY_TO" --env NAMES="$FILES"
  "${send[@]}" --send "$K/settings_job.sh" --send "$K/front_check.py" --
  bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/settings_job.sh"')
[ -z "${DRY:-}" ] || { printf '%q ' "${cmd[@]}"; echo; exit 0; }
cd "$TREE"
rc=0
"${cmd[@]}" > "$LOGS/settings.txt" 2>&1 || rc=$?
rid=$(grep -o -E 'run r[0-9]{8}-[0-9]{6}-[0-9a-f]{4} launched' "$LOGS/settings.txt" | head -1 | cut -d' ' -f2 || true)
echo "SETTINGS JOB ${rid:-not launched} exit $rc (its launch in $LOGS/settings.txt)"
[ -n "$rid" ] && bash "$K/../runlog.sh" h200-2 "$rid" all | tee -a "$LOGS/settings.txt"
exit "$rc"
