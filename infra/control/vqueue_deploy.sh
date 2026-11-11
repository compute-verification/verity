#!/bin/bash
# ci deploys a ci/vqueue-live pin into the live loop, by the run-book: stop, wait for the tick, dry-run the pin from its own
# worktree, repin vqueue-src and record it in live/pin (where vqueue_boot.sh reads it after a reset), add ARG (a new flag the
# pin needs, if any) to live/args, restart. The stop is not optional: a dry run takes no lock, and beside a live tick the two
# outgrow the pod's 16 GB cgroup, whose OOM killer then kills the tick. Deployed as /data/vqueue/bin/vqueue_deploy.sh, linked
# as /root/ci-deploy.sh.
# Usage: ci-deploy.sh PIN [ARG], PIN being origin's ci/vqueue-live tip in full.
set -u
PIN=${1:?usage: ci-deploy.sh PIN [ARG]: PIN is origin ci/vqueue-live tip, in full}
ARG=${2:-}
VQ=/data/vqueue/live
SRC=/root/fast/vqueue-src
DRY=/root/fast/vqueue-dry
touch $VQ/stopped
flock -w 1500 $VQ/vqueue.lock true || { echo "lock still held after 1500 s: nothing deployed"; rm -f $VQ/stopped; exit 1; }
echo "stopped at $(date -u +%T)Z; current pin $(git -C $SRC rev-parse --short=10 HEAD)"
git -C $SRC fetch -q origin ci/vqueue-live && [ "$(git -C $SRC rev-parse FETCH_HEAD)" = "$PIN" ] || { echo "origin's ci/vqueue-live isn't $PIN"; rm -f $VQ/stopped; exit 1; }
git -C $SRC worktree remove --force $DRY 2>/dev/null; rm -rf $DRY
git -C $SRC worktree add -q --detach $DRY $PIN
sed "s#^CODE=.*#CODE=$DRY#" /data/vqueue/live.sh > /root/fast/live-dry.sh && chmod +x /root/fast/live-dry.sh
echo "--- dry run of $PIN ${ARG:+with $ARG}"
/root/fast/live-dry.sh once --dry-run $(grep -v '^#' $VQ/args) $ARG 2>&1 | grep -v "^ready " | tail -n 8 | cut -c1-400
rc=${PIPESTATUS[0]}
echo "--- dry run rc=$rc"
if [ "$rc" != 0 ]; then echo "dry run failed: pin NOT changed, loop left stopped"; exit 1; fi
git -C $SRC checkout -q --detach $PIN && echo "vqueue-src now $(git -C $SRC rev-parse --short=10 HEAD)"
git -C $SRC rev-parse HEAD > $VQ/pin
git -C $SRC worktree remove --force $DRY
# the last occurrence of a flag wins, so ARG is appended unless it is already the last line
if [ -n "$ARG" ] && [ "$(grep -v '^#' $VQ/args | tail -n 1)" != "$ARG" ]; then cp -p $VQ/args $VQ/args.bak-ci-$(date -u +%H%MZ); echo "$ARG" >> $VQ/args; fi
grep -v '^#' $VQ/args | tr '\n' ' '; echo
rm -f $VQ/stopped
echo "loop resumed at $(date -u +%T)Z"
