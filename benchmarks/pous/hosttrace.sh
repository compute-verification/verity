#!/bin/bash
# hosttrace.sh VCPU RCPU -- CMD...: run CMD with hosttrace.bt tracing the memory challenger's and the responder's cores into
# $RESEARCH_RUN_DIR/hosttrace.txt (HOSTTRACE_OUT overrides; HOSTTRACE_OVER_NS, default 5000, is its threshold).  Start it
# outside the node's quiet-slice wrapper so bpftrace's own process stays off the quiet cores:
#   gpu-lease 1 --on N --timed -- bash benchmarks/pous/hosttrace.sh 124 126 -- QUIET env ... bash benchmarks/pous/pod.sh ...
# Without passwordless sudo or bpftrace, CMD runs untraced and the file says why.  Exits with CMD's status.
set -u
[ $# -ge 4 ] && [ "$3" = -- ] || { echo "usage: hosttrace.sh VCPU RCPU -- CMD..." >&2; exit 2; }
vcpu=$1 rcpu=$2; shift 3
out=${HOSTTRACE_OUT:-${RESEARCH_RUN_DIR:-.}/hosttrace.txt}
here=$(cd "$(dirname "$0")" && pwd)
pid=
# bpftrace runs as root, so nothing this script's caller can signal reaches it: a root guard stops it once this script is
# gone, however it ended (a SIGKILLed lease included), so a tracer never outlives its run and holds the timed lease.
guard='p=$1; shift; "$@" & b=$!
trap "kill -INT $b" TERM HUP
while kill -0 "$p" 2>/dev/null && kill -0 "$b" 2>/dev/null; do sleep 0.2; done
kill -INT "$b" 2>/dev/null; wait "$b"'
bpftrace=$(command -v "${HOSTTRACE_BPFTRACE:-bpftrace}")
if sudo -n true 2>/dev/null && [ -n "$bpftrace" ]; then
  sudo -n sh -c "$guard" hosttrace-guard $$ "$bpftrace" "$here/hosttrace.bt" "$vcpu" "$rcpu" "${HOSTTRACE_OVER_NS:-5000}" > "$out" 2>&1 &
  pid=$!
  for _ in $(seq 300); do grep -q '^hosttrace' "$out" 2>/dev/null && break; kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
  grep -q '^hosttrace' "$out" 2>/dev/null || { echo "hosttrace: bpftrace did not attach" >> "$out"; pid=; }
else
  echo "hosttrace: no passwordless sudo or no bpftrace; untraced" > "$out"
fi
steal() { awk -v a="cpu$vcpu" -v b="cpu$rcpu" -v w="$1" '$1 == a || $1 == b {print "steal_ticks", w, $1, $9}' /proc/stat; }
before=$(steal before)
"$@"
rc=$?
if [ -n "$pid" ]; then
  sudo -n kill -TERM "$pid" 2>/dev/null
  wait "$pid"
fi
{ echo "$before"; steal after; echo "steal_ticks_hz $(getconf CLK_TCK)"; } >> "$out"
exit $rc
