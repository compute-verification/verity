#!/bin/bash
# vy-lean-build-setup: what vy-lean-sandbox needs on vy-nebius-1, as root; safe to run again (a second run changes nothing).
#
#   - the system user lean-build: no home, no login shell, a locked password, no sudo, in no group but its own (never research,
#     sudo, docker or adm), so it reads neither /home/research (0750) nor /etc/verity's keys. It never runs outside a unit
#     vy-lean-sandbox starts: nothing else runs as it, and sshd lets only research in;
#   - /build (root, 0755), where `vy-lean-sandbox checkout` makes /build/verity, and /opt/elan (root, 0755, empty), the toolchain's
#     mount point inside the unit, which systemd would otherwise make on the host's disk;
#   - DATA (/workspace/lean-build, root, 0755) with build/, bind-mounted on /build by /etc/fstab, and deps/, run-daily's copies of
#     each package's dependencies: ~40 GB that would otherwise land on the root disk, which filled on 6 Oct. The line has nofail
#     (a missing /workspace must not stop the boot); vy-lean-cache-daily.service requires the mount, so no build runs without it.
# Then it checks all of that from lean-build's side, and exits 1 when something doesn't hold.
set -euo pipefail
U=${VY_LEAN_BUILD_USER:-lean-build}
DATA=${VY_LEAN_BUILD_DATA:-/workspace/lean-build}
BIND="$DATA/build /build none bind,nofail,x-systemd.requires-mounts-for=/workspace 0 0"
FORBIDDEN="root research sudo wheel docker adm"
log() { echo "vy-lean-build-setup: $*"; }
[ "$(id -u)" = 0 ] || { log "run as root"; exit 2; }
export LC_ALL=C

if ! getent passwd "$U" >/dev/null; then
  useradd --system --user-group --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin \
    --comment "Lean builds, only in vy-lean-sandbox" "$U"
  log "created $U"
fi
[ "$(id -gn "$U")" = "$U" ] || { log "$U's primary group is $(id -gn "$U"), not $U: fix it by hand"; exit 1; }
for g in $(id -nG "$U"); do
  [ "$g" = "$U" ] || { gpasswd -d "$U" "$g" >/dev/null; log "removed $U from group $g"; }
done
[ "$(getent passwd "$U" | cut -d: -f7)" = /usr/sbin/nologin ] || usermod --shell /usr/sbin/nologin "$U"
passwd -S "$U" | awk '{exit $2 == "L" ? 0 : 1}' || usermod --lock "$U"
install -d -o root -g root -m 0755 /build /opt/elan "$DATA" "$DATA/build" "$DATA/deps"
if ! grep -qxF "$BIND" /etc/fstab; then
  echo "$BIND" >> /etc/fstab
  systemctl daemon-reload
  log "added /build's bind from $DATA/build to /etc/fstab"
fi
if ! mountpoint -q /build; then
  [ -z "$(ls -A /build)" ] || { log "/build isn't mounted and isn't empty: move what's in it to $DATA/build, then run again"; exit 1; }
  mount /build
  log "mounted $DATA/build on /build"
fi

bad=0
fail() { log "FAIL: $*"; bad=1; }
as() { setpriv --reuid="$U" --regid="$U" --init-groups --no-new-privs -- "$@"; }
[ "$(id -u "$U")" != 0 ] || fail "$U is uid 0"
for g in $FORBIDDEN; do
  ! id -nG "$U" | tr ' ' '\n' | grep -qx "$g" || fail "$U is in $g"
done
[ "$(getent passwd "$U" | cut -d: -f7)" = /usr/sbin/nologin ] || fail "$U has a login shell"
passwd -S "$U" | awk '{exit $2 == "L" ? 0 : 1}' || fail "$U's password isn't locked"
sudo -n -l -U "$U" 2>&1 | grep -q "is not allowed to run sudo" || fail "$U may run sudo: $(sudo -n -l -U "$U" 2>&1 | tail -1)"
[ ! -e /home/research ] || ! as test -r /home/research -o -x /home/research || fail "$U can open /home/research"
[ ! -e /root ] || ! as test -r /root || fail "$U can list /root"
for p in /etc/verity/* /etc/shadow; do
  [ -f "$p" ] || continue
  ! as test -r "$p" || fail "$U can read $p"
done
mountpoint -q /build && [ "$(stat -c %d /build)" = "$(stat -c %d "$DATA")" ] || fail "/build isn't $DATA/build mounted"
for p in / /build /opt/elan /usr/local/bin "$DATA" "$DATA/deps"; do
  ! as test -w "$p" || fail "$U can write $p"
done
[ "$bad" = 0 ] || exit 1
log "$U (uid $(id -u "$U")): no login, no sudo, in no group but its own; /build ($DATA/build), $DATA/deps and /opt/elan root's"
