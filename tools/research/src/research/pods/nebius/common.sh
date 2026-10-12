# shellcheck shell=bash disable=SC2034
# Sourced by launch.sh and teardown.sh: the one Nebius server's names and sizes, and service-account auth.
#
# Auth reads the Cursor secrets NEBIUS_SA_ID, NEBIUS_SA_PUBLIC_KEY_ID and NEBIUS_SA_PRIVATE_KEY_B64 (the PEM as one line of base64,
# `base64 -w0 key.pem`).  The key must be one line: a multi-line secret's body lines pass through `env | cut -d= -f1` and escape
# Cursor's redaction, which is how the old multi-line NEBIUS_SA_PRIVATE_KEY leaked six times by 2 Oct and again on 3 Oct. It is
# refused, so that nobody keeps it as a Cursor secret.
# The key is written to a mode-600 file in a mode-700 directory made for this process and removed when it exits; the CLI's profile
# lives there too (`nebius -c`), so nothing is written to ~/.nebius or the repository and no value is ever printed.  Never run these
# with `bash -x`.

NEBIUS_REGION=${NEBIUS_REGION:-uk-south2}
NEBIUS_PROJECT_ID=${NEBIUS_PROJECT_ID-project-e05mvq7kln00dvsa5xj2mg}   # default-project-uk-south2; the service account is editor there only
NEBIUS_PROJECT_NAME=${NEBIUS_PROJECT_NAME:-vy-nebius}                    # only when NEBIUS_PROJECT_ID is set empty
VM_NAME=${NEBIUS_VM_NAME:-vy-nebius-1}
PRESET=${NEBIUS_PRESET:-8gpu-192vcpu-1744gb}
PLATFORM_PREFIX=${NEBIUS_PLATFORM_PREFIX:-gpu-rtx6000}   # uk-south2 lists it as gpu-rtx6000-a; NEBIUS_PLATFORM pins one
IMAGE_FAMILY=${NEBIUS_IMAGE_FAMILY:-ubuntu24.04-cuda13.0}   # driver preset cuda13.0: NVIDIA 580.x
BOOT_GIB=${NEBIUS_BOOT_GIB:-256}
BOOT_TYPE=network_ssd
DATA_GIB=${NEBIUS_DATA_GIB:-5022}                  # 54 x 93 GiB (non-replicated disks come in 93 GiB units) = 4.90 TiB
DATA_TYPE=${NEBIUS_DATA_TYPE:-network_ssd_non_replicated}
DATA_DEVICE_ID=data                                # the guest sees /dev/disk/by-id/virtio-data
SG_NAME=${VM_NAME}-ssh
SSH_CIDRS=${NEBIUS_SSH_CIDRS:-0.0.0.0/0}           # comma-separated, at most 8; port 22 only either way
SSH_USER=research                                  # Nebius reserves root and admin: neither may log in over ssh
RATE_USD_PER_HOUR=${NEBIUS_RATE_USD_PER_HOUR:-14.40}   # the RTX PRO 6000 preset's; any other shape states its own
FS_ID=${NEBIUS_FILESYSTEM_ID:-}                    # a shared filesystem to attach read-write, mounted at /workspace/fs
PREEMPTIBLE=${NEBIUS_PREEMPTIBLE:-}                # 1: a preemptible VM, stopped on preemption (its disks are kept)
MACHINE=${VM_NAME}
PROJECT=verity

die() { echo "nebius: $*" >&2; exit 1; }
say() { echo "nebius: $*" >&2; }

NEBIUS_BIN=${NEBIUS_BIN:-$(command -v nebius || echo "$HOME/.nebius/bin/nebius")}

nebius_auth() {
  [ -x "$NEBIUS_BIN" ] || die "no nebius CLI (curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash)"
  local v
  for v in NEBIUS_SA_ID NEBIUS_SA_PUBLIC_KEY_ID; do
    [ -n "${!v:-}" ] && continue
    case ",${CLOUD_AGENT_INJECTED_SECRET_NAMES:-}," in
      *",$v,"*) die "$v not in this shell though this VM injects it: rerun from a fresh tmux login shell" ;;
      *) die "$v is not set (a Cursor secret: Dashboard -> Cloud Agents -> Secrets)" ;;
    esac
  done
  if [ -z "${NEBIUS_SA_PRIVATE_KEY_B64:-}" ]; then
    [ -z "${NEBIUS_SA_PRIVATE_KEY:-}" ] || die "NEBIUS_SA_PRIVATE_KEY (the multi-line key) is no longer read: store the key as \
NEBIUS_SA_PRIVATE_KEY_B64 (\`base64 -w0 key.pem\`) and delete NEBIUS_SA_PRIVATE_KEY (Dashboard -> Cloud Agents -> Secrets)"
    case ",${CLOUD_AGENT_INJECTED_SECRET_NAMES:-}," in
      *",NEBIUS_SA_PRIVATE_KEY_B64,"*)
        die "the Nebius key is not in this shell though this VM injects it: rerun from a fresh tmux login shell" ;;
      *) die "NEBIUS_SA_PRIVATE_KEY_B64 is not set (a Cursor secret, \`base64 -w0 key.pem\`: Dashboard -> Cloud Agents -> Secrets)" ;;
    esac
  fi
  NB_DIR=$(mktemp -d "${XDG_RUNTIME_DIR:-/tmp}/nebius-auth.XXXXXX") || die "mktemp failed"
  chmod 700 "$NB_DIR"
  trap 'rm -rf "$NB_DIR"' EXIT
  (umask 077 && printf '%s' "$NEBIUS_SA_PRIVATE_KEY_B64" | base64 -d > "$NB_DIR/key.pem" 2>/dev/null) \
    || die "NEBIUS_SA_PRIVATE_KEY_B64 is not base64"
  "$NEBIUS_BIN" --no-check-update -c "$NB_DIR/config.yaml" profile create sa --endpoint api.nebius.cloud \
    --service-account-id "$NEBIUS_SA_ID" --public-key-id "$NEBIUS_SA_PUBLIC_KEY_ID" \
    --private-key-file-path "$NB_DIR/key.pem" --skip-auth > /dev/null 2>&1 || die "could not write the CLI profile"
}

nb() { "$NEBIUS_BIN" --no-check-update --no-progress -c "$NB_DIR/config.yaml" --format json "$@"; }

# jf PATH [DEFAULT]: one field of the JSON on stdin (`items.0.metadata.id`); DEFAULT (or nothing) when absent
jf() {
  python3 -c '
import json, sys
d = json.load(sys.stdin)
for k in sys.argv[1].split("."):
    if isinstance(d, list):
        d = d[int(k)] if k.isdigit() and int(k) < len(d) else None
    else:
        d = d.get(k) if isinstance(d, dict) else None
print(sys.argv[2] if d in (None, "") else (json.dumps(d) if isinstance(d, (dict, list)) else d))' "$1" "${2:-}"
}

# by_name NAME: the id of the item named NAME in the JSON list on stdin (`{"items": [...]}`), or nothing
by_name() {
  python3 -c '
import json, sys
hits = [i["metadata"]["id"] for i in (json.load(sys.stdin).get("items") or []) if i.get("metadata", {}).get("name") == sys.argv[1]]
if len(hits) > 1:
    sys.exit(f"{len(hits)} resources are named {sys.argv[1]!r}: {hits}")
print(hits[0] if hits else "")' "$1"
}

# resolve_project: TENANT_ID and PROJECT_ID (NEBIUS_TENANT_ID / NEBIUS_PROJECT_ID, else looked up by name); CREATE=1 makes the
# project when it is missing
resolve_project() {
  TENANT_ID=${NEBIUS_TENANT_ID:-}
  PROJECT_ID=${NEBIUS_PROJECT_ID:-}
  if [ -n "$PROJECT_ID" ]; then
    nb compute instance list --parent-id "$PROJECT_ID" > /dev/null 2>&1 \
      || die "cannot list VMs in $PROJECT_ID: check that NEBIUS_SA_ID, NEBIUS_SA_PUBLIC_KEY_ID and NEBIUS_SA_PRIVATE_KEY_B64 are one current key pair"
    return 0
  fi
  if [ -z "$TENANT_ID" ]; then
    local tenants
    tenants=$(nb iam v2 tenant list --all) || die "cannot list tenants (is the service account's key right?)"
    TENANT_ID=$(printf '%s' "$tenants" | python3 -c '
import json, sys
ids = [i["metadata"]["id"] for i in json.load(sys.stdin).get("items") or []]
if len(ids) != 1:
    sys.exit(f"the service account sees {len(ids)} tenants {ids}: set NEBIUS_TENANT_ID")
print(ids[0])') || die "tenant lookup failed"
  fi
  PROJECT_ID=$(nb iam v2 project list --parent-id "$TENANT_ID" --all | by_name "$NEBIUS_PROJECT_NAME") || die "cannot list projects"
  if [ -z "$PROJECT_ID" ] && [ "${CREATE:-0}" = 1 ]; then
    say "creating project $NEBIUS_PROJECT_NAME in $NEBIUS_REGION"
    PROJECT_ID=$(nb iam v2 project create --parent-id "$TENANT_ID" --name "$NEBIUS_PROJECT_NAME" --region "$NEBIUS_REGION" | jf metadata.id) \
      || die "cannot create project $NEBIUS_PROJECT_NAME (create it in the console, or set NEBIUS_PROJECT_ID)"
  fi
  [ -n "$PROJECT_ID" ] || die "no project named $NEBIUS_PROJECT_NAME in tenant $TENANT_ID (set NEBIUS_PROJECT_ID)"
}

vm_id() { nb compute instance list --parent-id "$PROJECT_ID" --all | by_name "$VM_NAME"; }
disk_id() { nb compute disk list --parent-id "$PROJECT_ID" --all | by_name "$1"; }

# wait_state ID STATE [TRIES]: poll the VM every 10 s until status.state is STATE
wait_state() {
  local id=$1 want=$2 tries=${3:-180} st=""
  while [ "$tries" -gt 0 ]; do
    st=$(nb compute instance get --id "$id" | jf status.state | tr '[:lower:]' '[:upper:]')
    [ "$st" = "$want" ] && return 0
    tries=$((tries - 1))
    sleep 10
  done
  die "VM $id is $st, not $want, after waiting"
}
