#!/bin/sh
# vy-nebius-stop INSTANCE_ID: stop this VM through the Nebius API (lease.sh's LEASE_TERMINATE_CMD on the Nebius server).
# `shutdown` inside the VM is not a stop: Nebius treats it as a failure, restarts the VM and keeps billing it.  The token is the
# attached service account's, from the metadata service; it lives in a private directory only for this call.
set -eu
id=${1:?instance id}
tok=$(curl -sf -H 'Metadata: true' http://metadata.nebius.internal/v1/iam/sa/token/access_token) || {
  echo "no service-account token from the metadata service (the VM has no service account attached): cannot stop $id"
  exit 3
}
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
(umask 077 && printf '%s' "$tok" > "$d/token")
nebius --no-check-update -c "$d/config.yaml" profile create vm --endpoint api.nebius.cloud --token-file "$d/token" --skip-auth > /dev/null
nebius --no-check-update --no-progress -c "$d/config.yaml" compute instance stop --id "$id" --async > /dev/null
echo "TERMINATED $id (nebius compute instance stop)"
