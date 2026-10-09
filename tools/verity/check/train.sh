#!/usr/bin/env bash
# `tools/verity/check/train.sh COMMIT MACHINE [--agreement]`: the lander's recorded check of a train, and the one way the lander
# launches a check. It runs the merge queue's command (`research.jobs.dispatch.command`: `check.py --record --on MACHINE`, against
# origin/main, on the passes it gathers from the newest recorded checks) in a priority slot and past a failed step, and refuses
# any other flag, so no flag of the lander's turns reuse off or changes what is checked.
#
# COMMIT (a train-prep branch or sha) is checked out in a worktree of its own, so the checkout it is run from stays on main, and
# `check.py --record --on MACHINE --priority --keep-going` runs there: an Attempt of exactly COMMIT, in the next check slot of
# MACHINE to free, ahead of lane checks still waiting and never stopping a running one (`slot.py --priority`), with the pinned
# upstream build when COMMIT changes verity/ml/flock/ or verity/core/service/flock/ against origin/main. --agreement sends that
# build whatever the train changes: the lander's rule for a train that adds or changes a guarantee (`lean-audit.json`) anywhere.
# It prints the run ids (`RUN ID...`); `research merge COMMIT` lands it on a pass. `research merge --train --prepare` builds
# COMMIT's branch. With MACHINE `auto` and no free slot, check.py launches twins, one run per machine that can take it (`slot.py
# --twin`): the first to take a slot runs the check and the others exit 75, not started, so RUN names one id per twin. When every
# twin exits 75 (check.py's own exit 75), none of them ran: rerun the train.
#
# Run it where the parent store key is (the Cursor secrets), from a Verity checkout with `machines.d` for MACHINE. It refuses a
# temporary credential, which cannot mint the run's custody key, and a shell that skips the store-backed tests.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
commit=$1 machine=$2; shift 2
for a in "$@"; do
  [ "$a" = --agreement ] || { echo "train: $a refused: train.sh runs the merge queue's check, which takes --agreement alone" >&2; exit 2; }
done
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
setsid nohup uv run -q python tools/verity/check/check.py --record --on "$machine" --priority --keep-going "$@" > "$log" 2>&1 < /dev/null &
# with twins (`check: --on: twins PAIR on M1, M2: ...`) one run launches per machine, and only one of them runs the check
want=1 ids=
for _ in $(seq 120); do
  ids=$(grep -oE 'run r20[0-9]{6}-[0-9]{6}-[0-9a-f]{4} launched' "$log" | cut -d' ' -f2 | awk '!seen[$0]++' | tr '\n' ' ') || true
  on=$(grep -m1 -oE '^check: --on: twins [^ ]+ on [^:]+' "$log" | sed 's/^.* on //') || true
  [ -z "$on" ] || want=$(( $(tr -cd , <<<"$on" | wc -c) + 1 ))
  [ "$(wc -w <<<"$ids")" -ge "$want" ] && break
  grep -qE "refused|Traceback" "$log" && break
  sleep 5
done
grep -E "^check:|refused|rror" "$log" | head -5 || true
[ -n "${ids// /}" ] || { echo "train: no run launched; see $log" >&2; exit 1; }
[ "$(wc -w <<<"$ids")" -ge "$want" ] || echo "train: $want twins, but only these launched; see $log" >&2
echo "RUN ${ids% }"
