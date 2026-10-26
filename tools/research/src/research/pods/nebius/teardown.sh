#!/usr/bin/env bash
# Stop, inspect or delete the Nebius server (launch.sh brings it up or starts it again).
#
#   teardown.sh status               the VM's state and address, its disks, and what is still billed
#   teardown.sh stop                 stop the VM: compute billing ($14.40/h) ends; the disks, the static IP and the weight cache stay
#   teardown.sh delete-vm --yes      delete the VM and its boot disk: the GPU quota and the static IP are released; the data disk stays
#   teardown.sh delete-all --yes     delete-vm, then the data disk (the weight cache, gone for good) and the security group
#
# Billing (Nebius, per second): the VM from Running until stopped or deleted; each disk from creation to deletion whether or not
# the VM runs (data 5,022 GiB non-replicated at $0.053/GiB-month, about $0.36/h; boot 256 GiB SSD at $0.071, about $0.03/h).
# `shutdown` inside the VM is not a stop: Nebius restarts it and keeps billing.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$here/common.sh"

cmd=${1:-}
yes=0
[ "${2:-}" = --yes ] && yes=1
case $cmd in
  status|stop) ;;
  delete-vm|delete-all) [ "$yes" = 1 ] || die "$cmd deletes resources: pass --yes" ;;
  *) die "usage: teardown.sh status | stop | delete-vm --yes | delete-all --yes" ;;
esac

nebius_auth
resolve_project
VM_ID=$(vm_id)
BOOT_ID=$(disk_id "$VM_NAME-boot")
DATA_ID=$(disk_id "$DATA_NAME")

case $cmd in
  status)
    if [ -n "$VM_ID" ]; then
      vm=$(nb compute instance get --id "$VM_ID")
      echo "vm   $VM_ID $(printf '%s' "$vm" | jf status.state) $(printf '%s' "$vm" | jf status.network_interfaces.0.public_ip_address.address -)"
    else
      echo "vm   none"
    fi
    echo "boot ${BOOT_ID:-none}"
    echo "data ${DATA_ID:-none}"
    ;;
  stop)
    [ -n "$VM_ID" ] || die "no VM named $VM_NAME"
    nb compute instance stop --id "$VM_ID" > /dev/null
    wait_state "$VM_ID" STOPPED
    say "stopped $VM_ID; disks $BOOT_ID $DATA_ID are still billed (delete-vm / delete-all)"
    ;;
  delete-vm|delete-all)
    if [ -n "$VM_ID" ]; then
      nb compute instance delete --id "$VM_ID" > /dev/null
      say "deleted VM $VM_ID"
    fi
    if [ -n "$BOOT_ID" ]; then nb compute disk delete --id "$BOOT_ID" > /dev/null && say "deleted boot disk $BOOT_ID"; fi
    if [ "$cmd" = delete-all ]; then
      if [ -n "$DATA_ID" ]; then nb compute disk delete --id "$DATA_ID" > /dev/null && say "deleted data disk $DATA_ID (the weight cache)"; fi
      SG_ID=$(nb vpc security-group list --parent-id "$PROJECT_ID" --all | by_name "$SG_NAME")
      if [ -n "$SG_ID" ]; then nb vpc security-group delete --id "$SG_ID" > /dev/null && say "deleted security group $SG_ID"; fi
    fi
    say "then: research pods unregister $MACHINE"
    ;;
esac
