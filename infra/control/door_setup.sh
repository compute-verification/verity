#!/bin/bash
# vy-control-setup [--lock|--unlock]: the control pod's door, as root, safe to rerun; post_start.sh runs it after every reset,
# since all it makes but $DOOR is on the container disk.
#
#   - `research`: a system account whose keys reach only vy-control-gate. Its authorized_keys is the public keys of
#     $DOOR/gate-keys (`LANE TYPE BASE64 [COMMENT]` a line: LANE `-` for the key every agent holds, `ci` for ci's own, which the
#     gate lets run drive), each forced to the gate (restrict,command=). The file and its directories are root's, so the account
#     can't edit them. /etc/sudoers.d/60-vy-control lets the account run that gate as root and nothing else. The gate is copied
#     from /data/infra/vy-control-gate to /usr/local/sbin, where every directory above it is root's.
#   - root: each key of $DOOR/admin-keys (`TYPE BASE64 [COMMENT]` a line, no options), which no agent holds, is added to root's
#     authorized_keys. With --lock, refused until a run without it has added them, that file becomes exactly those keys, the
#     file it replaces is kept at $DOOR/root-keys.before-lock, and $DOOR/locked records the lock, so the run after a reset,
#     when the image's /start.sh has given root the pod's PUBLIC_KEY again, locks again. --unlock puts the kept file back, with
#     the admin keys, and removes the mark.
#   A key in both files is refused: a key agents hold is not root's.
# $DOOR is /data/boot/door, on the volume. `vy-control-setup --authorized-keys GATE_KEYS [ADMIN_KEYS]` only prints the two
# authorized_keys files it would write (research's, then root's after `--`), as any user.
# Undo: `vy-control-setup --unlock`, then `userdel research; rm /etc/sudoers.d/60-vy-control /usr/local/sbin/vy-control-gate`.
set -euo pipefail
die() { echo "vy-control-setup: $1" >&2; exit "${2:-1}"; }
usage="usage: vy-control-setup [--lock|--unlock] | --authorized-keys GATE_KEYS [ADMIN_KEYS]"
DOOR=/data/boot/door
GATE=/usr/local/sbin/vy-control-gate
KEY='(ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp(256|384|521)) ([A-Za-z0-9+/]+=*)( [^[:cntrl:]]*)?'

# GATE_KEYS -> $research, the account's authorized_keys, and $blobs, its keys' base64
gate_keys() {
  local ln lane key blob
  research= blobs=
  [ -f "$1" ] && ! grep -q 'PRIVATE KEY' "$1" || die "$1: a file of public keys, LANE TYPE BASE64 [COMMENT] a line"
  while IFS= read -r ln || [ -n "$ln" ]; do
    case "$ln" in ''|'#'*) continue ;; esac
    [[ $ln =~ ^(-|[A-Za-z0-9][A-Za-z0-9._:-]{0,63})\ ($KEY)$ ]] \
      || die "$1: a line that is not LANE TYPE BASE64 [COMMENT] (LANE - for the key every agent holds; no options)"
    lane=${BASH_REMATCH[1]} key=${BASH_REMATCH[2]} blob=${BASH_REMATCH[5]}
    case "$lane" in root|research) die "$1: $lane is an account here, not a lane" ;; esac
    case " $blobs " in *" $blob "*) die "$1: one key on two lines (sshd would take the first's lane)" ;; esac
    blobs="$blobs $blob"
    [ "$lane" = - ] && lane= || lane=" $lane"
    research+="restrict,command=\"$GATE$lane\" $key"$'\n'
  done < "$1"
  [ -n "$research" ] || die "$1 holds no key"
}

# ADMIN_KEYS -> $admin, root's keys, none of them one of $blobs
admin_keys() {
  local ln
  admin=
  [ -f "$1" ] && ! grep -q 'PRIVATE KEY' "$1" || die "$1: a file of public keys, TYPE BASE64 [COMMENT] a line"
  while IFS= read -r ln || [ -n "$ln" ]; do
    case "$ln" in ''|'#'*) continue ;; esac
    [[ $ln =~ ^$KEY$ ]] || die "$1: a line that is not TYPE BASE64 [COMMENT] (no options)"
    case " $blobs " in *" ${BASH_REMATCH[3]} "*) die "$1: a key that is also in gate-keys: a key agents hold is not root's" ;; esac
    admin+="$ln"$'\n'
  done < "$1"
  [ -n "$admin" ] || die "$1 holds no key: root would have no way in"
}

if [ "${1:-}" = --authorized-keys ]; then
  [ $# = 2 ] || [ $# = 3 ] || die "$usage" 2
  gate_keys "$2"
  printf '%s' "$research"
  [ $# = 2 ] || { admin_keys "$3"; printf '%s\n%s' -- "$admin"; }
  exit 0
fi
mode=${1:-}
[ $# -le 1 ] || die "$usage" 2
case "$mode" in ''|--lock|--unlock) ;; *) die "$usage" 2 ;; esac
[ "$(id -u)" = 0 ] || die "run as root" 3
command -v sudo > /dev/null && command -v visudo > /dev/null && [ -d /etc/sudoers.d ] \
  || die "no sudo here: apt-get install -y sudo (post_start.sh's packages installs it after a reset)"
[ -f /data/infra/vy-control-gate ] || die "no /data/infra/vy-control-gate to install"
gate_keys "$DOOR/gate-keys"
admin_keys "$DOOR/admin-keys"
ak=/root/.ssh/authorized_keys
install -d -m 0700 /root/.ssh
[ -e "$ak" ] || install -m 0600 /dev/null "$ak"
if [ "$mode" = --lock ] && [ ! -e "$DOOR/locked" ]; then
  while IFS= read -r ln; do
    grep -qxF -- "$ln" "$ak" || die "--lock: an admin key is not root's yet: run vy-control-setup, log in as root with each admin key, then --lock"
  done <<< "${admin%$'\n'}"
fi

A=research H=/var/lib/vy-control-door
id "$A" > /dev/null 2>&1 \
  || useradd --system --user-group --home-dir "$H" --no-create-home --shell /bin/sh --comment "the control pod's door" "$A"
usermod -p '*' "$A"       # no password, and not locked: sshd takes its keys
install -d -m 0755 -o root -g root "$H" "$H/.ssh"
printf '%s' "$research" > "$H/.ssh/authorized_keys.tmp"
chmod 0644 "$H/.ssh/authorized_keys.tmp"
mv -f "$H/.ssh/authorized_keys.tmp" "$H/.ssh/authorized_keys"
install -m 0755 -o root -g root /data/infra/vy-control-gate "$GATE.tmp"
mv -f "$GATE.tmp" "$GATE"
echo "research's authorized_keys: $(grep -c . "$H/.ssh/authorized_keys") keys, each forced to $GATE"

s=/etc/sudoers.d/60-vy-control
printf '%s\n' "# vy-control-setup: research runs the control pod's gate as root, and nothing else (infra/control/vy-control-gate)" \
  "research ALL=(root) NOPASSWD: $GATE --ssh *" > "$s.tmp"
visudo -cqf "$s.tmp" || { rm -f "$s.tmp"; die "visudo refuses $s"; }
chmod 0440 "$s.tmp"
mv -f "$s.tmp" "$s"
visudo -cq || die "visudo -c refuses the configuration"

if [ "$mode" = --lock ] && [ ! -e "$DOOR/locked" ]; then
  cp -p "$ak" "$DOOR/root-keys.before-lock"
  : > "$DOOR/locked"
fi
cp -p "$ak" "$ak.tmp"
if [ "$mode" = --unlock ]; then
  rm -f "$DOOR/locked"
  [ ! -e "$DOOR/root-keys.before-lock" ] || cp -p "$DOOR/root-keys.before-lock" "$ak.tmp"
fi
if [ -e "$DOOR/locked" ]; then
  printf '%s' "$admin" > "$ak.tmp"
else
  while IFS= read -r ln; do grep -qxF -- "$ln" "$ak.tmp" || printf '%s\n' "$ln" >> "$ak.tmp"; done <<< "${admin%$'\n'}"
fi
chmod 0600 "$ak.tmp"
mv -f "$ak.tmp" "$ak"
echo "root's authorized_keys: $(grep -c . "$ak") keys, $([ -e "$DOOR/locked" ] && echo "the admin keys only (locked)" || echo "the admin keys among them (not locked)")"
[ -z "$(find /data -maxdepth 0 -perm /022)" ] \
  || echo "vy-control-setup: WARNING: /data is writable by other accounts, which could replace /data/infra: chmod 755 /data" >&2
