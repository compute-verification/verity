#!/bin/sh
# vy-vqueue-token.service, as root with the loop's group: mint the App's one-hour token (gh_app_token.sh, the control pod's
# helper, with the key where this node keeps it) and put it where the loop reads it, written beside and renamed so a reader
# never sees half of one. The key stays root's alone: the loop holds a token, never what mints them.
#   token.sh OUT    GH_APP_KEY names the key (default /etc/vy/vqueue/verity-drive.pem)
set -eu
out=${1:?usage: token.sh OUT}
export GH_APP_KEY="${GH_APP_KEY:-/etc/vy/vqueue/verity-drive.pem}"
[ -r "$GH_APP_KEY" ] || { echo "vy-vqueue-token: no key at $GH_APP_KEY: copy the App's key there, root's and 0400" >&2; exit 1; }
t=$("$(dirname "$0")/gh_app_token.sh")
[ -n "$t" ] || { echo "vy-vqueue-token: gh_app_token.sh printed no token" >&2; exit 1; }
umask 027
printf '%s\n' "$t" > "$out.new"
mv -f "$out.new" "$out"
