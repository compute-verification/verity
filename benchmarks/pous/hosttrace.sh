#!/bin/bash
# hosttrace.sh VCPU RCPU -- CMD...: run CMD with hosttrace.bt tracing the verifier's and the responder's cores into
# $RESEARCH_RUN_DIR/hosttrace.txt (HOSTTRACE_OUT overrides; HOSTTRACE_OVER_NS, default 5000, is its threshold).  Start it
# outside vy-pous-quiet so bpftrace's own process stays off the quiet cores:
#   gpu-lease 1 --on N --timed -- bash benchmarks/pous/hosttrace.sh 124 126 -- vy-pous-quiet env ... bash benchmarks/pous/pod.sh ...
# Without passwordless sudo or bpftrace, CMD runs untraced and the file says why.  Exits with CMD's status.
set -u
[ $# -ge 4 ] && [ "$3" = -- ] || { echo "usage: hosttrace.sh VCPU RCPU -- CMD..." >&2; exit 2; }
vcpu=$1 rcpu=$2; shift 3
out=${HOSTTRACE_OUT:-${RESEARCH_RUN_DIR:-.}/hosttrace.txt}
here=$(cd "$(dirname "$0")" && pwd)
pid=
if sudo -n true 2>/dev/null && command -v bpftrace >/dev/null; then
  sudo -n bpftrace "$here/hosttrace.bt" "$vcpu" "$rcpu" "${HOSTTRACE_OVER_NS:-5000}" > "$out" 2>&1 &
  pid=$!
  for _ in $(seq 300); do grep -q '^hosttrace' "$out" 2>/dev/null && break; kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
  grep -q '^hosttrace' "$out" 2>/dev/null || { echo "hosttrace: bpftrace did not attach" >> "$out"; pid=; }
else
  echo "hosttrace: no passwordless sudo or no bpftrace; untraced" > "$out"
fi
"$@"
rc=$?
if [ -n "$pid" ]; then
  sudo -n kill -INT "$(pgrep -P "$pid" bpftrace || echo "$pid")" 2>/dev/null
  wait "$pid"
fi
exit $rc
