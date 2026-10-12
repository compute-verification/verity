#!/usr/bin/env bash
# Bring up (or start again) the one Nebius server, vy-nebius-1: 8x RTX PRO 6000 (sm_120), 192 vCPU, 1,744 GiB, $14.40/h, in uk-south2.
#
#   launch.sh --dry-run                          print the cloud-init and the VM spec; needs no credentials, touches nothing
#   launch.sh --max-hours H --yes [--register] [--deadline UTC]
#                                                find or create each piece below, start the VM, arm its lease for H hours; --deadline
#                                                (e.g. 2026-10-07T15:00:00Z, the end of the spend authorization) writes the VM's
#                                                /etc/research/deadline, a hard cap no lease extension or restart passes (vm_setup.sh)
#
# Idempotent: every resource is looked up by name first, so a rerun after a failure (or on a stopped VM) resumes; nothing is
# ever deleted here (teardown.sh).  In order, in project NEBIUS_PROJECT_ID (default-project-uk-south2; set it empty to look up or
# create NEBIUS_PROJECT_NAME, which needs tenant rights the launching service account doesn't have):
#   network + subnet   the project's first subnet, else vy-net / vy-subnet
#   security group     <vm>-ssh: ingress TCP 22 from NEBIUS_SSH_CIDRS (default any), egress any.  The VM has a static public IP:
#                      the research tooling reaches machines over ssh from agent VMs, the laptop and the control pod, and with no
#                      CPU quota there is no bastion, so the lock-down is the security group plus key-only sshd (no root, no
#                      passwords, one user) with the research key (RUNPOD_SSH_KEY_B64, or NEBIUS_SSH_PUBKEY)
#   disks              <vm>-boot (network SSD, ubuntu24.04-cuda13.0: driver 580.x) and <vm>-data (SSD non-replicated, 5,022 GiB,
#                      the weight cache: formatted once, mounted at /workspace; kept by stop and by teardown's delete-vm)
#   VM                 platform gpu-rtx6000* with preset 8gpu-192vcpu-1744gb; the service account NEBIUS_VM_SA_ID (default the
#                      launching one; `none` for no lease) is attached so the VM's lease can stop the VM itself
# Other shapes and regions come from the environment: NEBIUS_VM_NAME, NEBIUS_REGION, NEBIUS_PROJECT_ID, NEBIUS_PLATFORM_PREFIX (cpu-d3,
# gpu-h200-sxm), NEBIUS_PRESET, NEBIUS_IMAGE_FAMILY, NEBIUS_DATA_GIB and NEBIUS_RATE_USD_PER_HOUR; NEBIUS_FILESYSTEM_ID attaches an
# existing shared filesystem at /workspace/fs, and NEBIUS_PREEMPTIBLE=1 makes the VM preemptible.  GPU clocks are locked only on
# gpu-rtx6000.
# Billing starts when the VM is Running and is per second; --yes is required for anything that starts it.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
src=$(cd "$here/../../.." && pwd)            # tools/research/src: the research package, stdlib only
. "$here/common.sh"

max_hours="" yes=0 dry=0 register=0 deadline=""
while [ $# -gt 0 ]; do
  case $1 in
    --max-hours) max_hours=${2:?}; shift ;;
    --yes) yes=1 ;;
    --dry-run) dry=1 ;;
    --register) register=1 ;;
    --deadline) deadline=${2:?}; shift ;;
    *) die "unknown argument $1 (launch.sh --dry-run | --max-hours H --yes [--register] [--deadline UTC])" ;;
  esac
  shift
done
[ $((DATA_GIB % 93)) -eq 0 ] || [ "$DATA_TYPE" != network_ssd_non_replicated ] || die "NEBIUS_DATA_GIB=$DATA_GIB is not a multiple of 93"
if [ "$dry" = 0 ]; then
  [ "$yes" = 1 ] || die "this starts billing at \$$RATE_USD_PER_HOUR/h: pass --yes (or --dry-run)"
  [[ $max_hours =~ ^[0-9]+$ ]] && [ "$max_hours" -ge 1 ] || die "--max-hours H (whole hours >= 1): the VM stops itself after it"
fi
deadline_epoch=""
if [ -n "$deadline" ]; then
  deadline_epoch=$(date -u -d "$deadline" +%s 2> /dev/null) || die "--deadline $deadline is not a UTC time (2026-10-07T15:00:00Z)"
  [ "$deadline_epoch" -gt "$(date +%s)" ] || die "--deadline $deadline has passed"
fi

ssh_pubkey() {
  if [ -n "${NEBIUS_SSH_PUBKEY:-}" ]; then echo "$NEBIUS_SSH_PUBKEY"; return; fi
  PYTHONPATH=$src python3 -c 'from research.pods import runpod; print(runpod.public_key())' 2> /dev/null \
    || die "no research ssh key: set RUNPOD_SSH_KEY_B64 (the Cursor secret) or NEBIUS_SSH_PUBKEY"
}

# render_user_data PUBKEY LEASE_EPOCH: the cloud-init document (scripts gzip+base64 so no YAML quoting can bend them)
render_user_data() {
  VY_PUBKEY=$1 VY_LEASE_EPOCH=$2 VY_USER=$SSH_USER VY_DEV=$DATA_DEVICE_ID VY_DEADLINE=$deadline_epoch VY_FS=${FS_ID:+vy-fs} \
    VY_CLOCKS=$([ "$PLATFORM_PREFIX" = gpu-rtx6000 ] && echo 1 || echo 0) python3 - "$here" "$src/research/pods/sh" <<'PY'
import base64, gzip, os, sys
nebius, sh = sys.argv[1], sys.argv[2]
def blob(path):
    return base64.b64encode(gzip.compress(open(path, "rb").read(), mtime=0)).decode()
env = (f"VY_USER={os.environ['VY_USER']}\nVY_DATA_DEVICE_ID={os.environ['VY_DEV']}\nVY_BOOT_LEASE_S=7200\n"
       f"VY_LEASE_EPOCH={os.environ['VY_LEASE_EPOCH']}\nVY_DEADLINE_EPOCH={os.environ['VY_DEADLINE']}\n"
       f"VY_FS_TAG={os.environ['VY_FS']}\nVY_LOCK_CLOCKS={os.environ['VY_CLOCKS']}\n")
files = [("/etc/vy/vy.env", "0644", base64.b64encode(gzip.compress(env.encode(), mtime=0)).decode()),
         ("/usr/local/sbin/vy-setup", "0755", blob(f"{nebius}/vm_setup.sh")),
         ("/usr/local/sbin/vy-nebius-stop", "0755", blob(f"{nebius}/vm_stop.sh")),
         ("/etc/systemd/system/user@.service.d/vy-cpuset.conf", "0644", blob(f"{nebius}/user-cpuset.conf")),
         ("/usr/local/share/vy/lease.sh", "0755", blob(f"{sh}/lease.sh")),
         ("/usr/local/bin/gpu-lease", "0755", blob(f"{sh}/gpu_lease.sh"))]
out = ["#cloud-config", "users:", f"  - name: {os.environ['VY_USER']}", "    sudo: ALL=(ALL) NOPASSWD:ALL", "    shell: /bin/bash",
       "    ssh_authorized_keys:", f"      - {os.environ['VY_PUBKEY']}", "write_files:"]
for path, mode, data in files:
    out += [f"  - path: {path}", f"    permissions: '{mode}'", "    encoding: gz+b64", f"    content: {data}"]
out += ["runcmd:", "  - [/usr/local/sbin/vy-setup]", ""]
print("\n".join(out), end="")
PY
}

# vm_body ... : the instance create request
vm_body() {
  VB_NAME=$VM_NAME VB_PARENT=$1 VB_PLATFORM=$2 VB_PRESET=$PRESET VB_BOOT=$3 VB_DATA=$4 VB_DEV=$DATA_DEVICE_ID VB_SUBNET=$5 VB_SG=$6 \
    VB_SA=$7 VB_UD=$8 VB_FS=$FS_ID VB_PRE=$PREEMPTIBLE python3 -c '
import json, os
e = os.environ
spec = {"resources": {"platform": e["VB_PLATFORM"], "preset": e["VB_PRESET"]},
        "boot_disk": {"attach_mode": "READ_WRITE", "existing_disk": {"id": e["VB_BOOT"]}},
        "secondary_disks": [{"attach_mode": "READ_WRITE", "device_id": e["VB_DEV"], "existing_disk": {"id": e["VB_DATA"]}}],
        "network_interfaces": [{"name": "eth0", "subnet_id": e["VB_SUBNET"], "ip_address": {}, "public_ip_address": {"static": True},
                                "security_groups": [{"id": e["VB_SG"]}]}],
        "cloud_init_user_data": e["VB_UD"], "recovery_policy": "RECOVER"}
if e["VB_FS"]:
    spec["filesystems"] = [{"attach_mode": "READ_WRITE", "mount_tag": "vy-fs", "existing_filesystem": {"id": e["VB_FS"]}}]
if e["VB_PRE"] == "1":
    spec["preemptible"] = {"on_preemption": "STOP", "priority": 3}
if e["VB_SA"] not in ("", "none"):
    spec["service_account_id"] = e["VB_SA"]
print(json.dumps({"metadata": {"name": e["VB_NAME"], "parent_id": e["VB_PARENT"], "labels": {"machine": e["VB_NAME"], "project": "verity"}},
                  "spec": spec}))'
}

if [ "$dry" = 1 ]; then
  pub=${NEBIUS_SSH_PUBKEY:-$( (PYTHONPATH=$src python3 -c 'from research.pods import runpod; print(runpod.public_key())' 2> /dev/null) || echo "ssh-ed25519 AAAA-dry-run-placeholder")}
  ud=$(render_user_data "$pub" "$(( $(date +%s) + 3600 ))")
  echo "# cloud-init ($(printf '%s' "$ud" | wc -c) bytes)"
  printf '%s\n' "$ud"
  echo "# instance create request (ids are placeholders)"
  vm_body project-DRYRUN "${NEBIUS_PLATFORM:-$PLATFORM_PREFIX-a}" computedisk-BOOT computedisk-DATA vpcsubnet-DRYRUN vpcsecuritygroup-DRYRUN \
    "${NEBIUS_VM_SA_ID:-serviceaccount-LAUNCHER}" "<cloud-init above>" | python3 -m json.tool
  exit 0
fi

pub=$(ssh_pubkey)
nebius_auth
CREATE=1 resolve_project
say "project $PROJECT_ID ($NEBIUS_REGION)"

subnets=$(nb vpc subnet list --parent-id "$PROJECT_ID" --all)
SUBNET_ID=${NEBIUS_SUBNET_ID:-$(printf '%s' "$subnets" | jf items.0.metadata.id)}
if [ -z "$SUBNET_ID" ]; then
  say "creating network vy-net and subnet vy-subnet"
  net=$(nb vpc network create --parent-id "$PROJECT_ID" --name vy-net | jf metadata.id)
  SUBNET_ID=$(nb vpc subnet create --parent-id "$PROJECT_ID" --name vy-subnet --network-id "$net" | jf metadata.id)
fi
NETWORK_ID=$(nb vpc subnet get --id "$SUBNET_ID" | jf spec.network_id)
say "subnet $SUBNET_ID (network $NETWORK_ID)"

SG_ID=$(nb vpc security-group list --parent-id "$PROJECT_ID" --all | by_name "$SG_NAME")
if [ -z "$SG_ID" ]; then
  SG_ID=$(nb vpc security-group create --parent-id "$PROJECT_ID" --name "$SG_NAME" --network-id "$NETWORK_ID" | jf metadata.id)
fi
rules=$(nb vpc security-rule list --parent-id "$SG_ID" --all)
want_cidrs=$(python3 -c 'import json,sys; c=[x.strip() for x in sys.argv[1].split(",") if x.strip()]; assert 0 < len(c) <= 8, "1-8 CIDRs"; print(json.dumps(sorted(c)))' "$SSH_CIDRS")
ssh_rule=$(printf '%s' "$rules" | by_name ssh-in)
if [ -n "$ssh_rule" ]; then
  have=$(nb vpc security-rule get --id "$ssh_rule" | python3 -c 'import json,sys; print(json.dumps(sorted(json.load(sys.stdin)["spec"]["ingress"].get("source_cidrs") or ["0.0.0.0/0"])))')
  if [ "$have" != "$want_cidrs" ]; then
    say "ssh-in allows $have, want $want_cidrs: replacing it"
    nb vpc security-rule delete --id "$ssh_rule" > /dev/null
    ssh_rule=""
  fi
fi
if [ -z "$ssh_rule" ]; then
  nb vpc security-rule create "$(python3 -c 'import json,sys; print(json.dumps({"metadata": {"name": "ssh-in", "parent_id": sys.argv[1]},
    "spec": {"access": "ALLOW", "protocol": "TCP", "type": "STATEFUL", "ingress": {"source_cidrs": json.loads(sys.argv[2]), "destination_ports": [22]}}}))' \
    "$SG_ID" "$want_cidrs")" > /dev/null
fi
if [ -z "$(printf '%s' "$rules" | by_name egress-all)" ]; then
  nb vpc security-rule create "$(python3 -c 'import json,sys; print(json.dumps({"metadata": {"name": "egress-all", "parent_id": sys.argv[1]},
    "spec": {"access": "ALLOW", "protocol": "ANY", "type": "STATEFUL", "egress": {}}}))' "$SG_ID")" > /dev/null
fi
say "security group $SG_ID: ssh from $want_cidrs, egress any"

PLATFORM=${NEBIUS_PLATFORM:-$(nb compute platform list --parent-id "$PROJECT_ID" | python3 -c '
import json, sys
hits = sorted(p["metadata"]["name"] for p in json.load(sys.stdin).get("items") or []
              if p["metadata"]["name"].startswith(sys.argv[1]) and any(x.get("name") == sys.argv[2] for x in p.get("spec", {}).get("presets") or []))
if not hits:
    sys.exit(f"no {sys.argv[1]}* platform with preset {sys.argv[2]} in this project")
print(hits[0])' "$PLATFORM_PREFIX" "$PRESET")}
say "platform $PLATFORM, preset $PRESET"

BOOT_ID=$(disk_id "$VM_NAME-boot")
if [ -z "$BOOT_ID" ]; then
  BOOT_ID=$(nb compute disk create --parent-id "$PROJECT_ID" --name "$VM_NAME-boot" --type "$BOOT_TYPE" --size-gibibytes "$BOOT_GIB" \
    --block-size-bytes 4096 --source-image-family-image-family "$IMAGE_FAMILY" | jf metadata.id)
fi
DATA_ID=$(disk_id "$VM_NAME-data")
if [ -z "$DATA_ID" ]; then
  DATA_ID=$(nb compute disk create --parent-id "$PROJECT_ID" --name "$VM_NAME-data" --type "$DATA_TYPE" --size-gibibytes "$DATA_GIB" \
    --block-size-bytes 4096 | jf metadata.id)
fi
say "disks: boot $BOOT_ID ($BOOT_GIB GiB $IMAGE_FAMILY), data $DATA_ID ($DATA_GIB GiB $DATA_TYPE)"

VM_ID=$(vm_id)
if [ -z "$VM_ID" ]; then
  ud=$(render_user_data "$pub" "$(( $(date +%s) + max_hours * 3600 ))")
  say "creating VM $VM_NAME: billing starts once it is Running"
  VM_ID=$(nb compute instance create "$(vm_body "$PROJECT_ID" "$PLATFORM" "$BOOT_ID" "$DATA_ID" "$SUBNET_ID" "$SG_ID" \
    "${NEBIUS_VM_SA_ID:-$NEBIUS_SA_ID}" "$ud")" | jf metadata.id)
else
  st=$(nb compute instance get --id "$VM_ID" | jf status.state | tr '[:lower:]' '[:upper:]')
  say "VM $VM_ID exists ($st)"
  if [ "$st" = STOPPED ]; then nb compute instance start --id "$VM_ID" --async > /dev/null; fi
fi
wait_state "$VM_ID" RUNNING
IP=$(nb compute instance get --id "$VM_ID" | jf status.network_interfaces.0.public_ip_address.address)
IP=${IP%%/*}
[ -n "$IP" ] || die "VM $VM_ID has no public address"
say "VM $VM_ID running at $IP"

eval "SSH=($(PYTHONPATH=$src python3 -c 'import shlex, sys; from research import remote; print(shlex.join(remote.ssh_argv(sys.argv[1], 22, user=sys.argv[2])))' "$IP" "$SSH_USER"))"
for _ in $(seq 90); do "${SSH[@]}" true 2> /dev/null && break; sleep 10; done
"${SSH[@]}" true || die "no ssh to $SSH_USER@$IP"
"${SSH[@]}" 'sudo cloud-init status --wait > /dev/null; test -f /var/lib/vy-setup.done' || die "first-boot setup did not finish: ssh in and read /var/log/cloud-init-output.log"
if [ -n "$deadline_epoch" ]; then
  "${SSH[@]}" "sudo install -d -m 755 /etc/research && printf '%s %s %s\\n' $deadline_epoch $(date -u -d "@$deadline_epoch" +%Y-%m-%dT%H:%M:%SZ) launch \
    | sudo tee /etc/research/deadline.tmp > /dev/null && sudo mv -f /etc/research/deadline.tmp /etc/research/deadline && cat /etc/research/deadline" \
    | sed 's/^/nebius: deadline /' >&2
fi
"${SSH[@]}" "sh ~/.research/lease.sh extend $((max_hours * 3600)) launch" | sed 's/^/nebius: lease /' >&2

if [ "$register" = 1 ]; then
  PYTHONPATH=$src python3 -m research pods register "$MACHINE" --host "$IP" --user "$SSH_USER" --project "$PROJECT" \
    --rate "$RATE_USD_PER_HOUR" --root /workspace/research --replace
fi
cat <<EOF
READY $MACHINE $IP vm=$VM_ID project=$PROJECT_ID data=$DATA_ID sg=$SG_ID
next: research run --on $MACHINE --project $PROJECT --source . --cwd source -- python3 tools/research/src/research/pods/nebius/check.py
EOF
