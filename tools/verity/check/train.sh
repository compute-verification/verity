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
# `--deferred STEP,...`: drive's after-landing run and its Lean checks in place of the check: only those ci.toml jobs and what
# they need (`check.py --record --deferred`), in a priority slot as a check runs, recorded without `--tool check`, so no gate
# reads it; drive reads its steps and suites. With it alone, `--agreement` sends the upstream build (a run of `main` has no
# change to send it for), `--no-cache` reuses no pass (`--circuit-cache` beside it: but circuit-check's per-target ones), and
# `--no-regenerate` fails lean-audit on a lock record difference rather than passing it on a regenerated record: drive's
# after-landing run (`after_landing_no_cache`), where nothing commits regenerated records.
#
# Run it where the parent store key is (the Cursor secrets), from a Verity checkout with `machines.d` for MACHINE. It refuses a
# temporary credential, which cannot mint the run's custody key, a shell that skips the store-backed tests, and a COMMIT that is
# empty (its tree is its base's: where it leaves origin/main, or its parent for a commit on main) or that origin doesn't hold.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
commit=$1 machine=$2; shift 2
pass=() deferred= nocache= ccache= noregen=
while [ $# -gt 0 ]; do
  case $1 in
    --agreement) pass+=(--agreement) ;;
    --no-cache) nocache=1 ;;
    --circuit-cache) ccache=1 ;;
    --no-regenerate) noregen=1 ;;
    --deferred) [[ $# -ge 2 && $2 =~ ^[a-z0-9-]+(,[a-z0-9-]+)*$ ]] || { echo "train: --deferred needs STEP,..." >&2; exit 2; }
      deferred=$2; shift ;;
    *) echo "train: $1 refused: train.sh runs the merge queue's check, which takes --agreement and --deferred STEPS alone" >&2; exit 2 ;;
  esac
  shift
done
[ -z "$nocache" ] || [ -n "$deferred" ] \
  || { echo "train: --no-cache refused without --deferred: a train's check reuses the passes its inputs earned" >&2; exit 2; }
[ -z "$ccache" ] || [ -n "$nocache" ] \
  || { echo "train: --circuit-cache refused without --no-cache: it keeps circuit-check's reuse on under --no-cache" >&2; exit 2; }
[ -z "$noregen" ] || [ -n "$deferred" ] \
  || { echo "train: --no-regenerate refused without --deferred: a train's check regenerates the locks drive commits" >&2; exit 2; }
[ -z "$deferred" ] || pass+=(--deferred "$deferred")
[ -z "$nocache" ] || pass+=(--no-cache)
[ -z "$ccache" ] || pass+=(--circuit-cache)
[ -z "$noregen" ] || pass+=(--no-regenerate)
[ -z "${VERITY_SKIP_STORE:-}" ] || { echo "train: VERITY_SKIP_STORE is set; a recorded train runs the store-backed tests" >&2; exit 1; }
[ -z "${AWS_SESSION_TOKEN:-}" ] || { echo "train: AWS_SESSION_TOKEN is set; a temporary credential cannot mint the custody key" >&2; exit 1; }
[ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${R2_ENDPOINT:-}" ] || { echo "train: no parent store key (AWS_ACCESS_KEY_ID, R2_ENDPOINT)" >&2; exit 1; }
# From here a launch that registers no run says so in the merge thread, with the reason and the log's last lines, as well as
# exiting nonzero: two launches of 9-10 Oct registered nothing and told nobody for hours. Quiet, ringing only the poster's own
# handle: three refused resends of one commit rang every addressee of the merge thread (16 Oct, 14:02Z).
why=$(mktemp "${TMPDIR:-/tmp}/train-why.XXXXXX") log=
die() { echo "train: $*" | tee -a "$why" >&2; exit 1; }
tell() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    { echo "@${TRAIN_AS:-ci} train: ${sha:-$commit} on $machine${deferred:+ --deferred $deferred} registered no run (exit $rc):"
      tail -n 5 "$why"; [ -z "$log" ] || { echo "$log:"; tail -n 12 "$log" 2>/dev/null; }; } > "$why.post"
    uv run -q research msg reply --as "${TRAIN_AS:-ci}" --thread "${MERGE_THREAD:-1790957906.278529}" --quiet --report \
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
# a fetch of drive's tick in the same clone can hold a ref's lock (`cannot lock ref`): two more tries
for backoff in 5 15 ''; do
  git -C "$root" fetch -q origin 2>>"$why" && break
  [ -n "$backoff" ] || die "git fetch origin failed"
  sleep "$backoff"
done
[ -n "$(git -C "$root" for-each-ref --contains "$sha" refs/remotes/origin)" ] \
  || die "${sha:0:12} is on no branch of origin: push it first, the run ships what origin holds"
if git -C "$root" merge-base --is-ancestor "$sha" refs/remotes/origin/main; then
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
setsid nohup uv run -q python tools/verity/check/check.py --record --on "$machine" --priority --keep-going ${pass[@]+"${pass[@]}"} \
  > "$log" 2>&1 < /dev/null &
pid=$!
# with twins (`check: --on: twins PAIR on M1, M2: ...`) one run launches per machine, and only one of them runs the check
# 30 min: drive launches several runs at once, each gathering ~1 GiB of verdicts into TMPDIR on the control pod's network
# volume (21 Oct 14:11-14:25Z: three such launches gathered past a 10 min wait and were stopped)
launch_wait_s=1800
want=1 ids=
for _ in $(seq $((launch_wait_s / 5))); do
  kill -0 "$pid" 2>/dev/null || break
  ids=$(grep -oE 'run r20[0-9]{6}-[0-9]{6}-[0-9a-f]{4} launched' "$log" | cut -d' ' -f2 | awk '!seen[$0]++' | tr '\n' ' ') || true
  on=$(grep -m1 -oE '^check: --on: twins [^ ]+ on [^:]+' "$log" | sed 's/^.* on //') || true
  [ -z "$on" ] || want=$(( $(tr -cd , <<<"$on" | wc -c) + 1 ))
  [ "$(wc -w <<<"$ids")" -ge "$want" ] && break
  grep -qE "refused|Traceback" "$log" && break
  sleep 5
done
ids=$(grep -oE 'run r20[0-9]{6}-[0-9]{6}-[0-9a-f]{4} launched' "$log" | cut -d' ' -f2 | awk '!seen[$0]++' | tr '\n' ' ') || true
# a launcher still running with no run after the wait is stopped, its whole session (setsid): left, it launched runs nobody
# followed, which held check slots for hours on trees no longer on `next` (18 Oct: r20261018-173205-b114, 3 h, launched past
# the wait while check.py gathered verdicts)
if [ -z "${ids// /}" ] && kill -0 "$pid" 2>/dev/null; then
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  ids=$(grep -oE 'run r20[0-9]{6}-[0-9]{6}-[0-9a-f]{4} launched' "$log" | cut -d' ' -f2 | awk '!seen[$0]++' | tr '\n' ' ') || true
  [ -n "${ids// /}" ] || die "no run launched in the wait; stopped its launcher (pid $pid), so none launches later; see $log"
fi
grep -E "^check:|refused|rror" "$log" | head -5 || true
if [ -z "${ids// /}" ] && ! kill -0 "$pid" 2>/dev/null; then
  rc=0; wait "$pid" || rc=$?
  echo "train: check.py exited $rc before launching a run$([ "$rc" = 137 ] && echo ' (killed: SIGKILL, the OOM killer when memory runs out)'); see $log" | tee -a "$why" >&2
  exit $(( rc ? rc : 1 ))
fi
[ -n "${ids// /}" ] || die "no run launched; see $log"
[ "$(wc -w <<<"$ids")" -ge "$want" ] || echo "train: $want twins, but only these launched; see $log" >&2
echo "RUN ${ids% }"
