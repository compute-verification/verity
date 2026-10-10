#!/bin/sh
# disk_guard.sh check | status | release: vy-nebius-1's /workspace guard (a systemd timer every minute, from cluster_up.sh).
# At VY_DISK_HOLD_PCT (80, infra's mark at 4:38 PM PDT Sep 30) it sets stopPolicy: Hold on every queue whose jobs write to /workspace (VY_DISK_QUEUES), so Kueue
# admits nothing new; admitted workloads keep running, and it deletes nothing. While the disk stays at or above the mark it
# re-asserts the hold every tick, so the quiet hour's release can't undo it. Below VY_DISK_RELEASE_PCT (75) it releases only the
# queues it held itself (VY_DISK_STATE), and never inside the quiet hour (12:30-13:30Z), whose own release follows. Each hold and
# release is a line on stdout (the journal) and an alert note in the alert sink's outbox, which reaches lanes/infra/.
# It also holds `research run` launches on the node: while /workspace is at its disk mark or its inodes at VY_INODE_HOLD_PCT (65),
# VY_LAUNCH_HOLD (/etc/vy/launch-hold) names why, and research.remote's launcher refuses every new run there. A line
# `except: TOKEN...` (a run id, or a piece of the launcher's agent URL), added by hand, admits those runs; the guard keeps it.
# The file goes once the disk is under its release mark and the inodes under VY_INODE_RELEASE_PCT (63).
set -eu
K=${VY_KUBECTL:-k3s kubectl}
QUEUES=${VY_DISK_QUEUES:-deployments-cpu deployments-gpu provers backfill circuits}
MOUNT=${VY_DISK_MOUNT:-/workspace}
HOLD=${VY_DISK_HOLD_PCT:-80}
RELEASE=${VY_DISK_RELEASE_PCT:-75}
STATE=${VY_DISK_STATE:-/var/lib/vy-disk-guard/held}
OUTBOX=${VY_DISK_OUTBOX:-/workspace/monitoring/alerts/outbox}
IHOLD=${VY_INODE_HOLD_PCT:-65}
IRELEASE=${VY_INODE_RELEASE_PCT:-63}
LAUNCH_HOLD=${VY_LAUNCH_HOLD:-/etc/vy/launch-hold}

pct=$(df -P "$MOUNT" | awk 'NR == 2 {u = $3; a = $4; printf "%d", (u * 100 + u + a - 1) / (u + a)}')   # df's rounding up
ipct=$(df -iP "$MOUNT" | awk 'NR == 2 {u = $3; a = $4; if (u + a == 0) print 0; else printf "%d", (u * 100 + u + a - 1) / (u + a)}')
case $pct$ipct in '' | *[!0-9]*) echo "disk_guard: can't read $MOUNT's use" >&2; exit 1 ;; esac

policy() { $K get clusterqueue "$1" -o jsonpath='{.spec.stopPolicy}'; }
set_policy() { $K patch clusterqueue "$1" --type=merge -p "{\"spec\":{\"stopPolicy\":\"$2\"}}" > /dev/null; }
held() { tr '\n' ' ' < "$STATE" 2> /dev/null | sed 's/ *$//'; }

note() {   # note hold|release "QUEUES": an alert note for lanes/infra/ (the alert sink's outbox)
  stamp=$(date -u +%Y%m%dT%H%MZ)
  mkdir -p "$OUTBOX"
  f="$OUTBOX/$stamp-alert-node1-disk-guard-$1.md"
  if [ "$1" = hold ]; then
    title="Disk guard: $MOUNT is $pct% full, so Kueue admits nothing new on vy-nebius-1"
    body="- Held at $pct% (the mark is $HOLD%): $2. Admitted workloads keep running; nothing was deleted.
- The guard releases these queues itself once $MOUNT is under $RELEASE%. To release them by hand: vy-disk-guard release."
  elif [ "$1" = launch-hold ]; then
    title="Disk guard: vy-nebius-1 refuses new research runs"
    body="- $2. Running work keeps running; nothing was deleted.
- The hold lifts itself under $RELEASE% disk and $IRELEASE% inodes. To admit one run, add a line \`except: RUN-ID\` to $LAUNCH_HOLD."
  elif [ "$1" = launch-release ]; then
    title="Disk guard: vy-nebius-1 takes new research runs again"
    body="- $MOUNT at $pct% and its inodes at $ipct%."
  else
    title="Disk guard: $MOUNT is down to $pct%, so the queues it held admit again"
    body="- Released: $2."
  fi
  cat > "$f.tmp" << EOF
---
id: $(basename "$f" .md)
campaign: nebius-monitoring
lane: infra
kind: finding
status: open
repo: danielreuter/verity
origin: vy-disk-guard on vy-nebius-1 (pods/nebius/sky/disk_guard.sh)
---

# $title

$body
EOF
  mv "$f.tmp" "$f"
}

release() {
  h=$(held)
  [ -n "$h" ] || return 0
  for q in $h; do set_policy "$q" None; done
  : > "$STATE"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $MOUNT at $pct%: released $h"
  note release "$h"
}

hold_launches() {   # hold_launches WHY: write or refresh LAUNCH_HOLD, keeping its `except:` lines and its first `since:`
  [ -f "$LAUNCH_HOLD" ] && new="" || new=1
  keep=$(grep '^except:' "$LAUNCH_HOLD" 2> /dev/null || true)
  since=$(sed -n 's/^since: //p' "$LAUNCH_HOLD" 2> /dev/null | head -1)
  mkdir -p "$(dirname "$LAUNCH_HOLD")"
  { echo "reason: $1"; echo "since: ${since:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"; echo "by: vy-disk-guard"
    [ -z "$keep" ] || echo "$keep"; } > "$LAUNCH_HOLD.tmp"
  mv "$LAUNCH_HOLD.tmp" "$LAUNCH_HOLD"
  if [ -n "$new" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launches held: $1"
    note launch-hold "$1"
  fi
}

case ${1:-status} in
  status)
    echo "$MOUNT at $pct% (hold at $HOLD%, release under $RELEASE%); held by the guard: $(held)"
    echo "inodes at $ipct% (launch hold at $IHOLD%, release under $IRELEASE%); launch hold: $(sed -n 's/^reason: //p' "$LAUNCH_HOLD" 2> /dev/null || true)"
    for q in $QUEUES; do echo "$q $(policy "$q")"; done
    exit 0 ;;
  release | check) mkdir -p "$(dirname "$STATE")"; touch "$STATE" ;;
  *) echo "usage: disk_guard.sh check | status | release" >&2; exit 2 ;;
esac
[ "$1" = check ] || { release; exit 0; }

if [ "$pct" -ge "$HOLD" ]; then
  new=""
  for q in $QUEUES; do
    p=$(policy "$q")
    if grep -qx "$q" "$STATE"; then
      [ "$p" = Hold ] || set_policy "$q" Hold      # released by someone else while the disk is still full
    elif [ "$p" != Hold ] && [ "$p" != HoldAndDrain ]; then
      set_policy "$q" Hold
      echo "$q" >> "$STATE"
      new="$new $q"
    fi
  done
  if [ -n "$new" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $MOUNT at $pct%: held$new"
    note hold "${new# }"
  fi
elif [ "$pct" -lt "$RELEASE" ]; then
  now=$(date -u +%H%M)
  if [ "$now" -lt 1230 ] || [ "$now" -ge 1330 ]; then release; fi
fi

why=""
[ "$pct" -lt "$HOLD" ] || why="$MOUNT is $pct% full (hold at $HOLD%, release under $RELEASE%)"
[ "$ipct" -lt "$IHOLD" ] || why="${why:+$why; }$MOUNT's inodes are $ipct% used (hold at $IHOLD%, release under $IRELEASE%)"
if [ -n "$why" ]; then
  hold_launches "$why"
elif [ -f "$LAUNCH_HOLD" ] && [ "$pct" -lt "$RELEASE" ] && [ "$ipct" -lt "$IRELEASE" ]; then
  rm -f "$LAUNCH_HOLD"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launches released: $MOUNT at $pct%, inodes at $ipct%"
  note launch-release ""
fi
