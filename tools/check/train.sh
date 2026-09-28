#!/usr/bin/env bash
# `tools/check/train.sh COMMIT MACHINE [--agreement] [--verdicts-in PACK]`: the lander's recorded check of a train.
#
# COMMIT (a train-prep branch or sha) is checked out in a worktree of its own, so the checkout it is run from stays on main, and
# `check.py --record --on MACHINE --priority --keep-going` runs there: an Attempt of exactly COMMIT, in the next check slot of
# MACHINE to free, ahead of lane checks still waiting and never stopping a running one (`slot.py --priority`), with the
# pinned upstream build when COMMIT changes backends/flock/ against origin/main. --agreement sends that build whatever the
# train changes: the lander's rule for a train that adds or changes a pinned guarantee (`lean-audit.json`) anywhere. It
# prints the run id; `research merge COMMIT` lands it on a pass.
#
# Run it where the parent store key is (the Cursor secrets), from a Verity checkout with `machines.d` for MACHINE. It refuses a
# temporary credential, which cannot mint the run's custody key, and a shell that skips the store-backed tests.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
commit=$1 machine=$2; shift 2
[ -z "${VERITY_SKIP_STORE:-}" ] || { echo "train: VERITY_SKIP_STORE is set; a recorded train runs the store-backed tests" >&2; exit 1; }
[ -z "${AWS_SESSION_TOKEN:-}" ] || { echo "train: AWS_SESSION_TOKEN is set; a temporary credential cannot mint the custody key" >&2; exit 1; }
[ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${R2_ENDPOINT:-}" ] || { echo "train: no parent store key (AWS_ACCESS_KEY_ID, R2_ENDPOINT)" >&2; exit 1; }
root=$(git rev-parse --show-toplevel)
sha=$(git -C "$root" rev-parse --verify "$commit^{commit}")
git -C "$root" fetch -q origin main
tree="${TMPDIR:-/tmp}/train-${sha:0:12}"
git -C "$root" worktree remove --force "$tree" 2>/dev/null || true
git -C "$root" worktree add -q --detach "$tree" "$sha"
log="$tree.log"
cd "$tree"
echo "train: ${sha:0:12} on $machine; log $log"
# `research run` stays attached to the run until it ends; the run doesn't need it (custody publishes from the machine)
setsid nohup uv run -q python tools/check/check.py --record --on "$machine" --priority --keep-going "$@" > "$log" 2>&1 < /dev/null &
for _ in $(seq 120); do
  id=$(grep -m1 -oE 'run r20[0-9]{6}-[0-9]{6}-[0-9a-f]{4} launched' "$log" | cut -d' ' -f2) && break
  grep -qE "refused|Traceback" "$log" && break
  sleep 5
done
grep -E "^check:|refused|rror" "$log" | head -5 || true
[ -n "${id:-}" ] || { echo "train: no run launched; see $log" >&2; exit 1; }
echo "RUN $id"
