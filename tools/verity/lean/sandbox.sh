#!/usr/bin/env bash
# The build sandbox, run as
#   unshare --user --map-root-user --mount --net --pid --fork --kill-child -- bash sandbox.sh RW_DIR... -- CMD...
# Inside new user, mount, network and PID namespaces: every mount read-only except the RW_DIRs (the build's .lake
# directories), a private /tmp, /dev/shm and /var/tmp unless an RW_DIR lives under one, no network, and a fresh /proc.
# CMD is PID 1, so every process a build starts (compile-time code included) ends with it.
set -euo pipefail
rw=()
while [ "$1" != "--" ]; do rw+=("$1"); shift; done
shift
mount --make-rprivate /
for d in "${rw[@]}"; do mkdir -p "$d"; mount --bind "$d" "$d"; done
keep=" ${rw[*]} "
awk '{print $5}' /proc/self/mountinfo | sort -u | while read -r m; do
  case "$keep" in *" $m "*) continue;; esac
  mount -o remount,bind,ro "$m" 2>/dev/null || true
done
for t in /tmp /dev/shm /var/tmp; do
  [ -d "$t" ] || continue
  case "$keep" in *" $t/"*) continue;; esac
  mount -t tmpfs -o mode=1777 tmpfs "$t"
done
mount -t proc proc /proc
exec "$@"
