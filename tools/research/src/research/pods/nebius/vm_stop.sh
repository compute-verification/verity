#!/bin/sh
# vy-nebius-stop INSTANCE_ID: stop this VM through the Nebius API (lease.sh's LEASE_TERMINATE_CMD on the Nebius server).
# `shutdown` inside the VM is not a stop: Nebius treats it as a failure, restarts the VM and keeps billing it.  The token is the
# attached service account's, from the metadata service; it lives in a private directory only for this call.
# vy-nebius-stop --probe INSTANCE_ID reads the instance with that token instead, and fails when the account can't: an account
# with no rights in the VM's project (vy-vm-self-stop in us-central1, 13 Oct) refuses every stop, so lease and deadline fail open.
set -eu
probe=
[ "${1:-}" = --probe ] && { probe=1; shift; }
id=${1:?instance id}
tok=$(curl -sf -H 'Metadata: true' http://metadata.nebius.internal/v1/iam/sa/token/access_token) || {
  echo "no service-account token from the metadata service (the VM has no service account attached): cannot stop $id"
  exit 3
}
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
(umask 077 && printf '%s' "$tok" > "$d/token")
nebius --no-check-update -c "$d/config.yaml" profile create vm --endpoint api.nebius.cloud --token-file "$d/token" --skip-auth > /dev/null
if [ -n "$probe" ]; then
  nebius --no-check-update --no-progress -c "$d/config.yaml" compute instance get --id "$id" > /dev/null 2>&1 || {
    echo "the attached service account cannot read $id, so it cannot stop it: grant it the compute editor role in the VM's project"
    exit 4
  }
  echo "the attached service account can read $id"
  exit 0
fi
nebius --no-check-update --no-progress -c "$d/config.yaml" compute instance stop --id "$id" --async > /dev/null
echo "TERMINATED $id (nebius compute instance stop)"
