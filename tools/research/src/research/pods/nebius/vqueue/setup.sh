#!/bin/bash
# vy-vqueue-setup: the account and the directories of the verified queue's loop, as root, safe to rerun. It starts nothing and
# copies no credential: it says which are missing, by file name.
#
#   - `vy-vqueue`: a system user with no login, no sudo and no group but its own, home /var/lib/vy-vqueue (0750: the loop's
#     state, clone, pinned tree and scratch, which loop.sh lays out), closed to other accounts; whoever runs the loop's commands
#     by hand does so as this user (`sudo -u vy-vqueue`).
#   - its ssh key, made here once under the name research reads (~/.runpod/ssh/runpodctl-ssh-key) and never copied anywhere:
#     the public half is printed, for research's authorized_keys on each machine the loop's args name with --on, the host of
#     its --graph2 command and the merge mirror's. The key every agent holds never comes to this node.
#   - /etc/vy/vqueue (root's, 0750, the loop's group): the credentials root puts there by hand, never from git. `*.env` files
#     are the loop's to read (0640): the store's key (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, R2_ACCOUNT_ID, R2_BUCKET,
#     R2_ENDPOINT) and the Slack token (SLACK_BOT_TOKEN). verity-drive.pem, the App's key, is root's alone (0400): only
#     vy-vqueue-token.service reads it, and the loop gets the one-hour token it mints.
set -euo pipefail
die() { echo "vy-vqueue-setup: $*" >&2; exit "${2:-1}"; }
[ $# = 0 ] || die "usage: vy-vqueue-setup" 2
[ "$(id -u)" = 0 ] || die "run as root" 3

U=vy-vqueue H=/var/lib/vy-vqueue C=/etc/vy/vqueue
if ! id "$U" > /dev/null 2>&1; then
  useradd --system --user-group --home-dir "$H" --no-create-home --shell /usr/sbin/nologin --comment "the verified queue's loop" "$U"
fi
[ "$(id -nG "$U")" = "$U" ] || die "$U is in other groups ($(id -nG "$U")): it holds its own group and nothing else"
install -d -m 0750 -o "$U" -g "$U" "$H" "$H/live" "$H/shadow" "$H/tmp" "$H/trains"
install -d -m 0700 -o "$U" -g "$U" "$H/.runpod" "$H/.runpod/ssh"
k=$H/.runpod/ssh/runpodctl-ssh-key
[ -e "$k" ] || runuser -u "$U" -- ssh-keygen -q -t ed25519 -N '' -C "$U@$(hostname)" -f "$k"
echo "the loop's ssh key, for research's authorized_keys on its machines: $(cat "$k.pub")"

install -d -m 0750 -o root -g "$U" "$C"
for f in "$C"/*.env; do
  [ -f "$f" ] || continue
  chown root:"$U" "$f"
  chmod 0640 "$f"
done
if [ -f "$C/verity-drive.pem" ]; then
  chown root:root "$C/verity-drive.pem"
  chmod 0400 "$C/verity-drive.pem"
fi
missing=
for f in r2.env slack.env verity-drive.pem; do
  [ -s "$C/$f" ] || missing="$missing $f"
done
echo "credentials in $C: ${missing:+missing$missing}${missing:-all three files are there}"
