#!/bin/bash
# vy-admin-setup KEYS_FILE [--check]: the node's admin account, vy-admin, root through sudo for the people whose public keys
# KEYS_FILE holds (Dan's and infra's lead's; never a worker's). Run as root; rerun it to change the keys: authorized_keys becomes
# exactly KEYS_FILE's keys. The keys come only from this argument, never from a secret store or the repository.
#
#   - refuses an empty KEYS_FILE, a private key, a line with options (command=, from=, ...) and any line ssh-keygen can't read
#   - creates vy-admin (its own group, no password: `passwd -l`) when it doesn't exist, and installs ~vy-admin/.ssh/authorized_keys
#     (0600) by rename, so a failed run leaves the previous one
#   - installs LIB/10-vy-admin as /etc/sudoers.d/10-vy-admin (0440) once `visudo -cf` accepts it, and takes it out again unless
#     `visudo -c` then accepts the whole configuration: vy-admin's sudo is logged to the journal, each command a shell of its
#     runs too (log_subcmds), and its sessions' output under /var/log/sudo-io (root's, 0700)
#   - prints each key's fingerprint, never the key
#
# --check validates KEYS_FILE and prints the fingerprints, as root or not, and changes nothing.
set -euo pipefail
NAME=vy-admin
LIB=${VY_LOCKOUT_LIB:-/usr/local/lib/vy-lockout}
SUDOERS=/etc/sudoers.d/10-vy-admin
IOLOG=/var/log/sudo-io
TYPES='ssh-ed25519|sk-ssh-ed25519@openssh\.com|ecdsa-sha2-nistp(256|384|521)|sk-ecdsa-sha2-nistp256@openssh\.com|ssh-rsa'
die() { echo "vy-admin-setup: $*" >&2; exit "${2:-1}"; }

[ $# -ge 1 ] && [ $# -le 2 ] || die "usage: vy-admin-setup KEYS_FILE [--check]" 2
keys=$1 check=0
[ $# = 1 ] || { [ "$2" = --check ] || die "usage: vy-admin-setup KEYS_FILE [--check]" 2; check=1; }
[ -f "$keys" ] && [ ! -L "$keys" ] || die "$keys: not a regular file"
! grep -q 'PRIVATE KEY' "$keys" || die "$keys holds a private key: give public keys only (*.pub), and treat that key as exposed"

lines=()
n=0
while IFS= read -r ln || [ -n "$ln" ]; do
  n=$((n + 1))
  ln=${ln%$'\r'}
  case "$ln" in ''|'#'*) continue ;; esac
  [[ $ln =~ ^($TYPES)\ [A-Za-z0-9+/]+=*(\ [^[:cntrl:]]*)?$ ]] || die "$keys line $n: not one public key (TYPE BASE64 [COMMENT], no options)"
  fp=$(ssh-keygen -lf /dev/stdin <<< "$ln" 2> /dev/null) || die "$keys line $n: ssh-keygen can't read it as a public key"
  echo "key: $fp"
  lines+=("$ln")
done < "$keys"
[ "${#lines[@]}" -gt 0 ] || die "$keys holds no public key: refusing to make an account nobody can log in to"
[ "$check" = 0 ] || { echo "vy-admin-setup: ${#lines[@]} keys ok (--check: nothing changed)"; exit 0; }

[ "$(id -u)" = 0 ] || die "run as root" 3
[ -f "$LIB/10-vy-admin" ] || die "$LIB/10-vy-admin missing: research deploy install NODE first"
visudo -cqf "$LIB/10-vy-admin" || die "visudo -cf refuses $LIB/10-vy-admin"

if ! id "$NAME" > /dev/null 2>&1; then
  useradd --create-home --user-group --shell /bin/bash --comment "admin: root through logged sudo" "$NAME"
  echo "created $NAME"
fi
passwd -l "$NAME" > /dev/null
home=$(getent passwd "$NAME" | cut -d: -f6)
[ -n "$home" ] && [ -d "$home" ] && [ ! -L "$home" ] || die "$NAME's home $home: not a directory"
install -d -m 0700 -o "$NAME" -g "$NAME" "$home/.ssh"
tmp=$(mktemp "$home/.ssh/.authorized_keys.XXXXXX")
printf '%s\n' "${lines[@]}" > "$tmp"
chown "$NAME:$NAME" "$tmp"
chmod 0600 "$tmp"
mv -f "$tmp" "$home/.ssh/authorized_keys"
echo "installed $home/.ssh/authorized_keys: ${#lines[@]} keys"

install -d -m 0700 -o root -g root "$IOLOG"
install -m 0440 -o root -g root "$LIB/10-vy-admin" "$SUDOERS.tmp"
mv -f "$SUDOERS.tmp" "$SUDOERS"
if ! visudo -cq; then
  rm -f "$SUDOERS"
  die "visudo -c refuses the configuration with $SUDOERS: taken out again"
fi
echo "installed $SUDOERS; next: log in as $NAME and run \`sudo -n true\` (vy-lockout preflight checks for that login)"
