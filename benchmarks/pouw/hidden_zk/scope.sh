#!/usr/bin/env bash
# scope.sh CPUS MEM: job.sh in a top-level systemd scope of its own (`sudo systemd-run --scope --slice=vy-lend.slice`) on CPUS,
# under MemoryMax=MEM without swap, as this user and with this environment, for a node whose sessions carry a CPU mask narrower
# than CPUS (the RTX PRO 6000 node's are 0-91): the scope's AllowedCPUs widen the cgroup and the inner `taskset -c CPUS`
# the mask. Run it as job.sh is run, from the source tree's root; a SIGTERM stops the whole scope.  HZ_JOB names another
# job script to run so (served_zk's).
#   research run --on MACHINE --cwd source ... -- bash benchmarks/pouw/hidden_zk/scope.sh 48-79 420G
set -euo pipefail
cpus=${1:?CPUS} mem=${2:?MEM}
unit=hidden-zk-${RESEARCH_RUN_ID:-$$}
mkdir -p "${TMPDIR:-/tmp}"
envf=$(mktemp "${TMPDIR:-/tmp}/hidden-zk-env.XXXXXX")      # mode 600; read and removed inside the scope
export -p > "$envf"
echo "scope.sh: $unit.scope, CPUs $cpus, MemoryMax $mem, as $(id -un)"
sudo -n systemd-run --scope --quiet --unit="$unit" --slice=vy-lend.slice -p AllowedCPUs="$cpus" -p MemoryMax="$mem" \
  -p MemorySwapMax=0 -- setpriv --reuid="$(id -u)" --regid="$(id -g)" --init-groups \
  bash -c 'source "$1"; rm -f "$1"; cd "$2" && exec taskset -c "$3" bash "$4"' _ "$envf" "$PWD" "$cpus" "${HZ_JOB:-benchmarks/pouw/hidden_zk/job.sh}" &
pid=$!
trap 'echo "scope.sh: stopping $unit.scope"; sudo -n systemctl stop "$unit.scope"' TERM INT HUP
rc=0
wait $pid || rc=$?
while kill -0 $pid 2>/dev/null; do wait $pid || rc=$?; done
rm -f "$envf"
exit $rc
