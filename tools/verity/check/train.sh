#!/usr/bin/env bash
# `tools/verity/check/train.sh COMMIT MACHINE [--agreement]`: the lander's recorded check of a train, and the one way the lander
# launches a check. It runs the merge queue's command (`research.jobs.dispatch.command`: `check.py --record --on MACHINE`, against
# origin/main, on the passes of the newest recorded checks, which the run fetches) in a priority slot and past a failed step, and refuses
# any other flag, so no flag of the lander's turns reuse off or changes what is checked.
#
# COMMIT (a train-prep branch or sha) is checked out in a worktree of its own, so the checkout it is run from stays on main, and
# `check.py --record --on MACHINE --priority --timeout 36000 --keep-going` runs there: an Attempt of exactly COMMIT, in the next check slot of
# MACHINE to free, ahead of lane checks still waiting and never stopping a running one (`slot.py --priority`), with presigned
# URLs of the pinned upstream build when COMMIT changes verity/protocols/flock/ or a Lean package its verifier requires by path
# (but sigma/, a README.md or PROTOCOL.md) against origin/main, which the run fetches. --agreement sends them whatever the train
# changes: the lander's rule for a train that adds or changes a guarantee (`lean-audit.json`) anywhere.
# It prints the run ids (`RUN ID...`); `research merge COMMIT` lands it on a pass. `research merge --train --prepare` builds
# COMMIT's branch. With MACHINE `auto` and no free slot, check.py launches twins, one run per machine that can take it (`slot.py
# --twin`): the first to take a slot runs the check and the others exit 75, not started, so RUN names one id per twin. When every
# twin exits 75 (check.py's own exit 75), none of them ran: rerun the train.
#
# `--deferred STEP,...`: drive's after-landing run and its lock run in place of the check: only those ci.toml jobs and what
# they need (`check.py --record --deferred`), in a priority slot as a check runs, recorded without `--tool check`, so no gate
# reads it; drive reads its steps and suites. With it alone, `--agreement` sends the upstream build's URLs (a run of `main` has
# no change to send them for), `--no-cache` reuses no pass (`--circuit-cache` beside it: but circuit-check's per-target ones), and
# `--no-regenerate` fails lean-audit on a lock record difference rather than passing it on a regenerated record: drive's
# after-landing run, of a commit whose records drive's lock run regenerated and drive committed. `--base REF` (with --deferred alone)
# is the run's base (`check.py --record --base`), above which ratchet-gate judges every landing: drive sends the last verified
# `main`, below the tip it checks, since against origin/main a run of `main`'s tip would judge nothing. `--ratchet-base REF` (with
# --deferred alone) is ratchet-gate's base in place of it (`check.py --record --ratchet-base`): drive sends the tip of its last
# run whose ratchet-gate gave a verdict, so a step red elsewhere doesn't make each run re-judge every landing since.
# `--bookkeeping-warn` (with --deferred, unsharded, without --no-regenerate: drive's lock run) has lean-audit repair a lock's
# bookkeeping and the caps it breaks and warn, rather than fail (`check.py --bookkeeping-warn`); on a COMMIT whose check.py
# predates it, the run goes without it.
#
# `--shard K/N`: shard K of ci.toml's plan with N shards (`check.py --shard`): with `--deferred`, only those steps' parts in
# it (drive's `after_landing_shards`, one launch per shard); alone, a shard Attempt of the check, which `check.py --record
# --combine` joins with the other N-1 into the commit's check. `--shard auto:K/N --shard-plan-file PLAN`: shard K of the auto
# plan PLAN (`check.py --shard-plan --out PLAN`, made once for all N; drive's `after_landing_shards = "auto"`), sent as the run's
# input.
#
# `--deselect FILE` (with --deferred; an absolute path): the run's pytest and lean-suites steps leave out the tests FILE names, one
# node id a line (`check.py --record --deselect`: the vqueue loop's list); on a COMMIT whose check.py predates it, the run goes
# without it.
#
# `--cc-records`: circuit-check's records step alone (`check.py --record --cc-records`), which drive's sharded after-landing run
# runs first; `--cc-records-from ART@SHA256` (with `--shard`) gives a shard that step's artifact, so its circuit-check builds
# no binding.
#
# Run it where the parent store key is (the Cursor secrets), from a Verity checkout with `machines.d` for MACHINE. It refuses a
# temporary credential, which cannot mint the run's custody key, a shell with a debug or unknown check setting (`settings.py
# --of-record`: one that skips the store-backed tests), and a COMMIT that is empty (its tree is its base's: where it leaves
# origin/main, or its parent for a commit on main) or that origin doesn't hold.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,48p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
commit=$1 machine=$2; shift 2
pass=() deferred= nocache= ccache= noregen= shard= below= rbelow= records= recfrom= plan= bookkeep= deselect=
while [ $# -gt 0 ]; do
  case $1 in
    --agreement) pass+=(--agreement) ;;
    --no-cache) nocache=1 ;;
    --circuit-cache) ccache=1 ;;
    --no-regenerate) noregen=1 ;;
    --bookkeeping-warn) bookkeep=1 ;;
    --deferred) [[ $# -ge 2 && $2 =~ ^[a-z0-9-]+(,[a-z0-9-]+)*$ ]] || { echo "train: --deferred needs STEP,..." >&2; exit 2; }
      deferred=$2; shift ;;
    --shard) [[ $# -ge 2 && $2 =~ ^(auto:)?[1-9][0-9]*/[1-9][0-9]*$ ]] || { echo "train: --shard needs K/N or auto:K/N" >&2; exit 2; }
      shard=$2; shift ;;
    --shard-plan-file) [[ $# -ge 2 && $2 == /* && -f $2 ]] || { echo "train: --shard-plan-file needs a plan's absolute path" >&2; exit 2; }
      plan=$2; shift ;;
    --base) [ $# -ge 2 ] || { echo "train: --base needs REF" >&2; exit 2; }
      below=$2; shift ;;
    --ratchet-base) [ $# -ge 2 ] || { echo "train: --ratchet-base needs REF" >&2; exit 2; }
      rbelow=$2; shift ;;
    --cc-records) records=1 ;;
    --cc-records-from) [[ $# -ge 2 && $2 =~ ^art:[0-9a-f]+@[0-9a-f]{64}$ ]] || { echo "train: --cc-records-from needs ART@SHA256" >&2; exit 2; }
      recfrom=$2; shift ;;
    --deselect) [[ $# -ge 2 && $2 == /* && -f $2 ]] || { echo "train: --deselect needs a list's absolute path" >&2; exit 2; }
      deselect=$2; shift ;;
    *) echo "train: $1 refused: train.sh runs the merge queue's check, which takes --agreement, --deferred STEPS, --base REF, --ratchet-base REF, --shard K/N, --shard auto:K/N --shard-plan-file PLAN, --bookkeeping-warn, --deselect FILE and --cc-records[-from] alone" >&2; exit 2 ;;
  esac
  shift
done
[ -z "$nocache" ] || [ -n "$deferred" ] \
  || { echo "train: --no-cache refused without --deferred: a train's check reuses the passes its inputs earned" >&2; exit 2; }
[ -z "$ccache" ] || [ -n "$nocache" ] \
  || { echo "train: --circuit-cache refused without --no-cache: it keeps circuit-check's reuse on under --no-cache" >&2; exit 2; }
[ -z "$noregen" ] || [ -n "$deferred" ] \
  || { echo "train: --no-regenerate refused without --deferred: a train's check regenerates stale lock records" >&2; exit 2; }
[ -z "$bookkeep" ] || { [ -n "$deferred" ] && [ -z "$noregen$shard" ]; } \
  || { echo "train: --bookkeeping-warn refused but with --deferred, unsharded and without --no-regenerate: drive's lock run" >&2; exit 2; }
[ -z "$below" ] || [ -n "$deferred" ] \
  || { echo "train: --base refused without --deferred: a train's check judges its merges above origin/main" >&2; exit 2; }
[ -z "$rbelow" ] || [ -n "$deferred" ] \
  || { echo "train: --ratchet-base refused without --deferred: a train's check judges its merges above origin/main" >&2; exit 2; }
[ -z "$deselect" ] || [ -n "$deferred" ] \
  || { echo "train: --deselect refused without --deferred: a check that leaves tests out is no check a gate takes" >&2; exit 2; }
[ -z "$records" ] || [ -z "$deferred$nocache$ccache$noregen$shard$below$rbelow$recfrom$deselect${pass[*]+x}" ] \
  || { echo "train: --cc-records refused with another flag: it builds circuit-check's records alone" >&2; exit 2; }
[ -z "$recfrom" ] || [ -n "$shard" ] \
  || { echo "train: --cc-records-from refused without --shard: only a shard's circuit-check takes the records" >&2; exit 2; }
if [[ $shard == auto:* ]]; then [ -n "$plan" ]; else [ -z "$plan" ]; fi \
  || { echo "train: --shard auto:K/N takes --shard-plan-file PLAN, and only it: every shard of one auto plan" >&2; exit 2; }
[ -z "$records" ] || pass+=(--cc-records)
[ -z "$recfrom" ] || pass+=(--cc-records-from "$recfrom")
[ -z "$deferred" ] || pass+=(--deferred "$deferred")
[ -z "$nocache" ] || pass+=(--no-cache)
[ -z "$ccache" ] || pass+=(--circuit-cache)
[ -z "$noregen" ] || pass+=(--no-regenerate)
[ -z "$shard" ] || pass+=(--shard "$shard")
[ -z "$plan" ] || pass+=(--shard-plan-file "$plan")
# the tree's settings.py first: drive runs a copy of this script from outside the tree (`drive.detach`)
settings=tools/verity/check/settings.py; [ -f "$settings" ] || settings="$(dirname "$0")/settings.py"
python3 "$settings" --of-record || { echo "train: a recorded train takes no debug or unknown check setting" >&2; exit 1; }
[ -z "${AWS_SESSION_TOKEN:-}" ] || { echo "train: AWS_SESSION_TOKEN is set; a temporary credential cannot mint the custody key" >&2; exit 1; }
[ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${R2_ENDPOINT:-}" ] || { echo "train: no parent store key (AWS_ACCESS_KEY_ID, R2_ENDPOINT)" >&2; exit 1; }
# From here a launch that registers no run says so in the merge thread, with the reason and the log's last lines, as well as
# exiting nonzero, so a launch that registers nothing never goes unnoticed. Quiet, ringing only the poster's own handle, so
# refused resends of one commit don't ring every addressee of the merge thread.
why=$(mktemp "${TMPDIR:-/tmp}/train-why.XXXXXX") log=
die() { echo "train: $*" | tee -a "$why" >&2; exit 1; }
tell() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    { echo "@${TRAIN_AS:-ci} train: ${sha:-$commit} on $machine${records:+ --cc-records}${deferred:+ --deferred $deferred}${shard:+ --shard $shard} registered no run (exit $rc):"
      tail -n 5 "$why"; [ -z "$log" ] || { echo "$log:"; tail -n 12 "$log" 2>/dev/null; }; } > "$why.post"
    # from the queue's own checkout, never the checked commit's tree: its code would run with the queue's Slack token
    (cd "${root:-.}" && uv run -q research msg reply --as "${TRAIN_AS:-ci}" --thread "${MERGE_THREAD:-1790957906.278529}" \
      --quiet --report --file "$why.post") > /dev/null 2>&1 || echo "train: and could not post that to the merge thread" >&2
  fi
  rm -f "$why" "$why.post"
}
trap tell EXIT
root=$(git rev-parse --show-toplevel)
sha=$(git -C "$root" rev-parse --verify "$commit^{commit}" 2>>"$why") || die "$commit: no such commit"
common=$(git -C "$root" rev-parse --path-format=absolute --git-common-dir)
# origin holds the commit: a branch of origin points at it (the queue pushed it from this clone) or origin's main has it. Both
# are milliseconds, where asking every branch of origin (`for-each-ref --contains`) walks the whole history
held() {
  [ -n "$(git -C "$root" for-each-ref --count=1 --points-at "$sha" refs/remotes/origin)" ] \
    || git -C "$root" merge-base --is-ancestor "$sha" refs/remotes/origin/main 2>/dev/null
}
# one fetch and worktree at a time per checkout: two launches racing on the fetch both fail
exec 9> "$common/train.lock"
flock 9
if ! held; then
  # a fetch of drive's tick in the same clone can hold a ref's lock (`cannot lock ref`): two more tries
  for backoff in 5 15 ''; do
    git -C "$root" fetch -q origin 2>>"$why" && break
    [ -n "$backoff" ] || die "git fetch origin failed"
    sleep "$backoff"
  done
  held || [ -n "$(git -C "$root" for-each-ref --contains "$sha" refs/remotes/origin)" ] \
    || die "${sha:0:12} is on no branch of origin: push it first, the run ships what origin holds"
fi
if [ -n "$below" ]; then
  b=$(git -C "$root" rev-parse -q --verify "$below^{commit}") || die "--base $below: no such commit"
  pass+=(--base "$b")
fi
if [ -n "$rbelow" ]; then
  b=$(git -C "$root" rev-parse -q --verify "$rbelow^{commit}") || die "--ratchet-base $rbelow: no such commit"
  pass+=(--ratchet-base "$b")
fi
if git -C "$root" merge-base --is-ancestor "$sha" refs/remotes/origin/main; then
  base=$(git -C "$root" rev-parse -q --verify "$sha^1" || true)
else
  base=$(git -C "$root" merge-base "$sha" refs/remotes/origin/main) || die "${sha:0:12} shares no history with origin/main"
fi
[ -z "$base" ] || [ "$(git -C "$root" rev-parse "$sha^{tree}")" != "$(git -C "$root" rev-parse "$base^{tree}")" ] \
  || die "${sha:0:12} is empty (its tree is its base ${base:0:12}'s): nothing to check"
# TRAIN_TREES: where the checkout goes, apart from TMPDIR (drive keeps it on the control pod's container disk; its TMPDIR is on
# the network volume, where uv and imports from the checkout stalled launches for minutes)
trees="${TRAIN_TREES:-${TMPDIR:-/tmp}}"
mkdir -p "$trees" || die "can't make $trees"
tag=${shard/\//of}
tree="$trees/train-${sha:0:12}${shard:+-${tag/:/-}}${records:+-records}"
git -C "$root" worktree remove --force "$tree" 2>/dev/null || true
# a tree git no longer lists as a worktree (half removed on the control pod's NFS TMPDIR) would refuse `worktree add`
rm -rf "$tree" 2>/dev/null || rm -rf "$tree" || true
git -C "$root" worktree prune
git -C "$root" worktree add -q --detach "$tree" "$sha" 2>>"$why" || die "git worktree add $tree failed"
exec 9>&-
log="$tree.log"
cd "$tree"
if [ -n "$bookkeep" ]; then
  if grep -q -- "--bookkeeping-warn" tools/verity/check/check.py; then pass+=(--bookkeeping-warn)
  else echo "train: ${sha:0:12}'s check.py predates --bookkeeping-warn: its lock run fails on bookkeeping"; fi
fi
if [ -n "$deselect" ]; then
  if grep -q DESELECT_FILE tools/verity/check/check.py; then pass+=(--deselect "$deselect")
  else echo "train: ${sha:0:12}'s check.py predates --deselect: its run leaves no test out"; fi
fi
echo "train: ${sha:0:12} on $machine${shard:+ shard $shard}; log $log"
# `research run` stays attached to the run until it ends; the run doesn't need it (custody publishes from the machine). The
# commit's own code runs from here, so without the queue's GitHub and Slack credentials: any PR in it could push `main`.
# --timeout: only a run with one extends its node's lease (`remote.lease_extend_s`), so a node that runs nothing but checks
# would otherwise stop two hours after its boot; 10 h is past any check's wall time, as the nightly's
setsid nohup env -u GH_TOKEN -u GITHUB_TOKEN -u GH_CONFIG_DIR -u SLACK_BOT_TOKEN \
  uv run -q python tools/verity/check/check.py --record --on "$machine" --priority --timeout 36000 --keep-going \
  ${pass[@]+"${pass[@]}"} \
  > "$log" 2>&1 < /dev/null &
pid=$!
# with twins (`check: --on: twins PAIR on M1, M2: ...`) one run launches per machine, and only one of them runs the check
# 60 min: the queue launches a check's shards at once, each placing itself and shipping its tree to its machine
launch_wait_s=3600
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
# followed, which held check slots for hours on trees no longer on `next`
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
