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
# `--against BASE`: the touched-suites run in place of the check, every suite the change from BASE (an ancestor of COMMIT, such
# as the main a passed train was checked on) touches, slow tests included (`suites.py --changed BASE`, through the quick tier's
# job `research/quick_tier.py`, with its Lean build first), in a priority slot of MACHINE (`auto` or a list: the machine
# `research queue`'s `place` picks). It is a `research run --cwd source` of COMMIT's tree with a `quick-tier/v1` result, not
# a merge check: `research merge` still wants the check. It takes no --agreement.
#
# Run it where the parent store key is (the Cursor secrets), from a Verity checkout with `machines.d` for MACHINE. It refuses a
# temporary credential, which cannot mint the run's custody key, a shell that skips the store-backed tests, a COMMIT that is
# empty (its tree is its base's: BASE, or else where it leaves origin/main, or its parent for a commit on main) or that origin
# doesn't hold, and a BASE that isn't an ancestor of COMMIT.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
commit=$1 machine=$2; shift 2
pass=() against=
while [ $# -gt 0 ]; do
  case $1 in
    --agreement) pass+=(--agreement) ;;
    --against) [ $# -ge 2 ] || { echo "train: --against needs BASE" >&2; exit 2; }; against=$2; shift ;;
    *) echo "train: $1 refused: train.sh runs the merge queue's check, which takes --agreement and --against BASE alone" >&2; exit 2 ;;
  esac
  shift
done
[ -z "$against" ] || [ ${#pass[@]} -eq 0 ] || { echo "train: --agreement refused with --against: a touched-suites run sends no build" >&2; exit 2; }
[ -z "${VERITY_SKIP_STORE:-}" ] || { echo "train: VERITY_SKIP_STORE is set; a recorded train runs the store-backed tests" >&2; exit 1; }
[ -z "${AWS_SESSION_TOKEN:-}" ] || { echo "train: AWS_SESSION_TOKEN is set; a temporary credential cannot mint the custody key" >&2; exit 1; }
[ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${R2_ENDPOINT:-}" ] || { echo "train: no parent store key (AWS_ACCESS_KEY_ID, R2_ENDPOINT)" >&2; exit 1; }
# From here a launch that registers no run says so in the merge thread, with the reason and the log's last lines, as well as
# exiting nonzero: two launches of 9-10 Oct registered nothing and told nobody for hours.
why=$(mktemp "${TMPDIR:-/tmp}/train-why.XXXXXX") log=
die() { echo "train: $*" | tee -a "$why" >&2; exit 1; }
tell() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    { echo "train: ${sha:-$commit} on $machine${against:+ --against $against} registered no run (exit $rc):"
      tail -n 5 "$why"; [ -z "$log" ] || { echo "$log:"; tail -n 12 "$log" 2>/dev/null; }; } > "$why.post"
    uv run -q research slack reply --as "${TRAIN_AS:-old-circuits-and-proofs}" --thread "${MERGE_THREAD:-1790957906.278529}" \
      --file "$why.post" > /dev/null 2>&1 || echo "train: and could not post that to the merge thread" >&2
  fi
  rm -f "$why" "$why.post"
}
trap tell EXIT
root=$(git rev-parse --show-toplevel)
sha=$(git -C "$root" rev-parse --verify "$commit^{commit}" 2>>"$why") || die "$commit: no such commit"
# one fetch and worktree at a time per checkout: two launches racing on the fetch both failed (22:51Z, 9 Oct)
exec 9> "$(git -C "$root" rev-parse --path-format=absolute --git-common-dir)/train.lock"
flock 9
git -C "$root" fetch -q origin 2>>"$why" || die "git fetch origin failed"
[ -n "$(git -C "$root" for-each-ref --contains "$sha" refs/remotes/origin)" ] \
  || die "${sha:0:12} is on no branch of origin: push it first, the run ships what origin holds"
if [ -n "$against" ]; then
  base=$(git -C "$root" rev-parse --verify "$against^{commit}" 2>>"$why") || die "--against $against: no such commit"
  git -C "$root" merge-base --is-ancestor "$base" "$sha" || die "--against ${base:0:12} is not an ancestor of ${sha:0:12}"
elif git -C "$root" merge-base --is-ancestor "$sha" refs/remotes/origin/main; then
  base=$(git -C "$root" rev-parse -q --verify "$sha^1" || true)
else
  base=$(git -C "$root" merge-base "$sha" refs/remotes/origin/main) || die "${sha:0:12} shares no history with origin/main"
fi
[ -z "$base" ] || [ "$(git -C "$root" rev-parse "$sha^{tree}")" != "$(git -C "$root" rev-parse "$base^{tree}")" ] \
  || die "${sha:0:12} is empty (its tree is its base ${base:0:12}'s): nothing to check"
tree="${TMPDIR:-/tmp}/train-${sha:0:12}"
git -C "$root" worktree remove --force "$tree" 2>/dev/null || true
git -C "$root" worktree add -q --detach "$tree" "$sha" 2>>"$why" || die "git worktree add $tree failed"
exec 9>&-
log="$tree.log"
cd "$tree"
echo "train: ${sha:0:12} on $machine; log $log"
# `research run` stays attached to the run until it ends; the run doesn't need it (custody publishes from the machine)
if [ -z "$against" ]; then
  setsid nohup uv run -q python tools/verity/check/check.py --record --on "$machine" --priority --keep-going ${pass[@]+"${pass[@]}"} > "$log" 2>&1 < /dev/null &
else
  case $machine in
    auto|*,*) machine=$(uv run -q python -c 'import sys; from pathlib import Path; from research.queue import place
got, why = place(sys.argv[1], Path.cwd()); print("train: --on", sys.argv[1] + ":", got or "no machine", why, file=sys.stderr); print(got or "")' "$machine")
      [ -n "$machine" ] || exit 1 ;;
  esac
  # research.queue's REMOTE_WAIT_S and REMOTE_TIMEOUT_S, the quick tier's slot wait and run time, and LEAN_BUILDS
  wait_s=21600 run_s=10800
  # the run's command is the shipped tree's slot.py, the one form a machine with /etc/vy/runs-via-slot launches (slot_only.py)
  job="exec python3 \"\$RESEARCH_RUN_DIR/inputs/quick_tier.py\""
  job+=" --lean verity/core/service/flock --head $sha --base $base -- uv run --locked --extra torch-cpu python tools/verity/check/suites.py"
  setsid nohup uv run -q research run --on "$machine" --project verity --source "$tree" --cwd source \
    --send tools/research/src/research/quick_tier.py --require-result \
    --timeout $((wait_s + run_s)) --custody-ttl $((wait_s + run_s + 1800)) --declared-output 'out/*' \
    -- python3 tools/verity/check/slot.py --priority --wait-max $wait_s --run-max $run_s -- sh -c "$job" \
    > "$log" 2>&1 < /dev/null &
fi
pid=$!
# with twins (`check: --on: twins PAIR on M1, M2: ...`) one run launches per machine, and only one of them runs the check
want=1 ids=
for _ in $(seq 120); do
  kill -0 "$pid" 2>/dev/null || break
  ids=$(grep -oE 'run r20[0-9]{6}-[0-9]{6}-[0-9a-f]{4} launched' "$log" | cut -d' ' -f2 | awk '!seen[$0]++' | tr '\n' ' ') || true
  on=$(grep -m1 -oE '^check: --on: twins [^ ]+ on [^:]+' "$log" | sed 's/^.* on //') || true
  [ -z "$on" ] || want=$(( $(tr -cd , <<<"$on" | wc -c) + 1 ))
  [ "$(wc -w <<<"$ids")" -ge "$want" ] && break
  grep -qE "refused|Traceback" "$log" && break
  sleep 5
done
ids=$(grep -oE 'run r20[0-9]{6}-[0-9]{6}-[0-9a-f]{4} launched' "$log" | cut -d' ' -f2 | awk '!seen[$0]++' | tr '\n' ' ') || true
grep -E "^check:|refused|rror" "$log" | head -5 || true
if [ -z "${ids// /}" ] && ! kill -0 "$pid" 2>/dev/null; then
  rc=0; wait "$pid" || rc=$?
  echo "train: $([ -z "$against" ] && echo check.py || echo 'research run') exited $rc before launching a run$([ "$rc" = 137 ] && echo ' (killed: SIGKILL, the OOM killer when memory runs out)'); see $log" | tee -a "$why" >&2
  exit $(( rc ? rc : 1 ))
fi
[ -n "${ids// /}" ] || die "no run launched; see $log"
[ "$(wc -w <<<"$ids")" -ge "$want" ] || echo "train: $want twins, but only these launched; see $log" >&2
echo "RUN ${ids% }"
