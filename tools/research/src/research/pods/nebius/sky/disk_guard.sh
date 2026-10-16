#!/bin/sh
# disk_guard.sh check | status | release: vy-nebius-1's /workspace guard (a systemd timer every minute, from cluster_up.sh).
# At VY_DISK_HOLD_PCT (80, infra's mark at 4:38 PM PDT Sep 30) it sets stopPolicy: Hold on every queue whose jobs write to /workspace (VY_DISK_QUEUES), so Kueue
# admits nothing new; admitted workloads keep running, and it deletes nothing. While the disk stays at or above the mark it
# re-asserts the hold every tick, so the quiet hour's release can't undo it. Below VY_DISK_RELEASE_PCT (75) it releases only the
# queues it held itself (VY_DISK_STATE), and never inside the quiet hour (12:30-13:30Z), whose own release follows. Each hold and
# release is a line on stdout (the journal) and an alert note in the alert sink's outbox, which reaches lanes/infra/.
set -eu
K=${VY_KUBECTL:-k3s kubectl}
QUEUES=${VY_DISK_QUEUES:-deployments-cpu deployments-gpu provers backfill circuits}
MOUNT=${VY_DISK_MOUNT:-/workspace}
HOLD=${VY_DISK_HOLD_PCT:-80}
RELEASE=${VY_DISK_RELEASE_PCT:-75}
STATE=${VY_DISK_STATE:-/var/lib/vy-disk-guard/held}
OUTBOX=${VY_DISK_OUTBOX:-/workspace/monitoring/alerts/outbox}

pct=$(df -P "$MOUNT" | awk 'NR == 2 {u = $3; a = $4; printf "%d", (u * 100 + u + a - 1) / (u + a)}')   # df's rounding up
case $pct in '' | *[!0-9]*) echo "disk_guard: can't read $MOUNT's use" >&2; exit 1 ;; esac

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
repo: compute-verification/verity
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

case ${1:-status} in
  status)
    echo "$MOUNT at $pct% (hold at $HOLD%, release under $RELEASE%); held by the guard: $(held)"
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
