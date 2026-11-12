#!/usr/bin/env bash
# A rehearsal of accept_capture.sh's shell path with no card and no node: a fake nvidia-smi, a fake venv whose python
# reports the pin's vLLM and prints what it was started with, and the tree's gpu_lease.sh on a lease directory and a CPU
# quota of its own (the lease script's test hooks: GPU_LEASE_COUNT, GPU_LEASE_DIR, GPU_LEASE_CPU_MAX).  From the root of a
# tree holding the capture and gpu_lease.sh:
#   bash accept_capture_dry.sh [DIR holding accept_capture.sh; this file's by default]
# Each case runs accept_capture.sh and reads its exit status and what it printed.  Exit 0 when every case did what the
# card job needs of it.  It waits on events (a FIFO), never on the clock.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
S=${1:-$here}/accept_capture.sh
L=tools/research/src/research/pods/sh/gpu_lease.sh
[ -f "$S" ] || { echo "no $S"; exit 2; }
[ -f "$L" ] || { echo "run from the root of a tree holding $L"; exit 2; }
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
mkdir -p "$d/bin" "$d/venv/bin" "$d/lease" "$d/run"
cat > "$d/bin/nvidia-smi" << 'EOF'
#!/usr/bin/env bash
# the three questions accept_capture.sh asks, answered from FAKE_USED and FAKE_APPS
case "$*" in
  *--query-gpu=memory.used*) echo "${FAKE_USED:-4}" ;;
  *--query-gpu=uuid*) echo "GPU-fake-$2" ;;
  *--query-compute-apps=gpu_uuid*) [ -z "${FAKE_APPS:-}" ] || echo "$FAKE_APPS" ;;
  *) echo "fake nvidia-smi: unexpected $*" >&2; exit 9 ;;
esac
EOF
cat > "$d/venv/bin/python" << 'EOF'
#!/usr/bin/env bash
if [ "${1:-}" = -c ]; then echo "0.28.1rc1.dev472+gd9105ea80"; exit 0; fi
echo "fake python: args: $*"
echo "fake python: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} GPU_LEASE_UUID=${GPU_LEASE_UUID:-}" \
  "cpus=$(sed -n 's/^Cpus_allowed_list:[[:space:]]*//p' /proc/self/status)"
echo "fake python: owner: $(cut -c1-300 "$GPU_LEASE_DIR/${CUDA_VISIBLE_DEVICES:-x}.owner" 2> /dev/null)"
exit "${FAKE_RC:-0}"
EOF
chmod +x "$d/bin/nvidia-smi" "$d/venv/bin/python"

allowed=$(sed -n 's/^Cpus_allowed_list:[[:space:]]*//p' /proc/self/status); part=${allowed%%,*}; lo=${part%-*}; hi=${part#*-}
[ "$hi" -gt "$lo" ] && cores=$lo-$((lo + 1)) || cores=$lo
ncpu=$(nproc)
echo "200000 100000" > "$d/cpu.max"   # a quota of two CPUs: the capture's two fit alone, and not beside a lease that declared none
export PATH="$d/bin:$PATH" VENV="$d/venv" GPU_LEASE_DIR="$d/lease" GPU_LEASE_COUNT=8 GPU_LEASE_CPU_MAX="$d/cpu.max" \
  GPU_LEASE_UUID_TABLE=GPU-fake-0,GPU-fake-1,GPU-fake-2,GPU-fake-3,GPU-fake-4,GPU-fake-5,GPU-fake-6,GPU-fake-7 \
  GPU_LEASE_WINDOWS="$d/no-windows" GPU_LEASE_ALLOWED_FILE="$d/no-allowed" GPU_LEASE_USAGE_LOG="$d/usage.jsonl" \
  GPU_LEASE_UTIL_LOG="$d/no-util/*.jsonl" RESEARCH_RUN_DIR="$d/run" OUT="$d/run/out" CARD=0 CORES="$cores"
unset GPU_LEASE_SH GPU_LEASE_WHO MAX_MIN ARGS NOT_BEFORE END_UTC FAKE_USED FAKE_APPS FAKE_RC
at() { date -u -d "@$(( $(date -u +%s) + $1 ))" +%FT%TZ; }
echo "start $(date -u +%FT%TZ) on $(hostname): $ncpu CPUs, allowed $allowed, the capture's cores $cores; $S ($(sha256sum < "$S" | cut -c1-12))"

fails=0 cases=0
run() {
  # run NAME WANT_RC "PATTERN|PATTERN..." [K=V...]: each pattern (a fixed string; a leading ! means absent) against the output
  local name=$1 want=$2 pats=$3 rc p ok=1; shift 3
  ( for kv in "$@"; do export "${kv?}"; done; exec bash "$S" ) > "$d/log" 2>&1; rc=$?
  [ "$rc" = "$want" ] || ok=0
  local IFS='|'
  for p in $pats; do
    case $p in
      '!'*) ! grep -qF -- "${p#!}" "$d/log" || { ok=0; echo "  $name: unexpected: ${p#!}"; } ;;
      *) grep -qF -- "$p" "$d/log" || { ok=0; echo "  $name: missing: $p"; } ;;
    esac
  done
  cases=$((cases + 1))
  if [ "$ok" = 1 ]; then echo "ok   $name: exit $rc"; [ -z "${SHOW:-}" ] || sed 's/^/    /' "$d/log" | cut -c1-330; else fails=$((fails + 1)); echo "FAIL $name: exit $rc, wanted $want"; sed 's/^/    /' "$d/log" | cut -c1-260; fi
}
hold() {
  # hold GPU $1's lease (other arguments go to gpu_lease.sh) until `release`; returns once the lease says it holds it
  local gpu=$1; shift
  rm -f "$d/hold" "$d/said"; mkfifo "$d/hold" "$d/said"
  GPU_LEASE_WHO=another-lane bash "$L" 1 --on "$gpu" --mem-gb 0 "$@" -- bash -c 'read -r _ < "$0"' "$d/hold" 2> "$d/said" &
  holder=$!
  exec 7< "$d/said"
  read -r said <&7
  case $said in *"GPUs $gpu "*) ;; *) echo "the holder of GPU $gpu did not start: $said"; fails=$((fails + 1)) ;; esac
}
release() { echo > "$d/hold"; wait "$holder"; exec 7<&-; }

T=adapters/vllm/tests/properties/sampled_accept_capture_gpu.py
SHOW=1 run "the launch as planned" 0 \
  "fake python: args: $T --out $d/run/out/capture --scratch |--random 400|CUDA_VISIBLE_DEVICES=0 GPU_LEASE_UUID=GPU-fake-0 cpus=$cores|who=circuits-accept| cores=$cores | until=|card 0 idle (4 MiB, no compute process, lease free)|CMD pinned to CPUs $cores|accept-capture: exit 0" \
  "NOT_BEFORE=$(at -60)" "END_UTC=$(at 2400)" MAX_MIN=25 "ARGS=--random 400"
case $(cat "$d/lease/0.owner" 2> /dev/null) in
  released=*" cores=$cores "*) echo "ok   the lease's line is marked released and keeps cores=$cores" ;;
  *) fails=$((fails + 1)); echo "FAIL the lease's line after the run: $(cut -c1-200 "$d/lease/0.owner" 2> /dev/null)" ;;
esac
run "the capture's NO-GO is the job's exit" 1 "fake python: args|accept-capture: exit 1" FAKE_RC=1
run "before NOT_BEFORE" 75 "nothing done|!fake python: args|!gpu-lease:" "NOT_BEFORE=$(at 3600)" "END_UTC=$(at 6000)"
run "fewer than MAX_MIN minutes left" 75 "fewer than 25 minutes|!fake python: args|!gpu-lease:" "END_UTC=$(at 600)" MAX_MIN=25
run "the shorter lease near the end" 0 "fake python: args| until=" "END_UTC=$(at 960)" MAX_MIN=15
run "a lease over 30 minutes" 2 "at most 30 minutes|!fake python: args" MAX_MIN=31
run "a compute process on the card" 76 "a compute process is on card 0|!fake python: args|!gpu-lease:" FAKE_APPS=GPU-fake-0
run "a compute process on another card" 0 "fake python: args" FAKE_APPS=GPU-fake-3
run "the card holds memory" 76 "card 0 holds 2048 MiB|!fake python: args|!gpu-lease:" FAKE_USED=2048
run "not the pin's vLLM" 2 "is not the build asked for|!fake python: args" VLLM_WANT=+g0000000
hold 0 --cores "$cores"
run "the card's lease is held" 76 "card 0's lease is not free|0 held|who=another-lane|!fake python: args"
release
run "after the holder left" 0 "fake python: args"
if [ "$ncpu" -ge 3 ] && [ "$hi" -ge $((lo + 2)) ]; then
  hold 3
  SHOW=1 run "refused at the quota beside a lease that declared no cores" 75 \
    "gpu-lease: refused, holding nothing|over its CPU quota of 2|this request: --cores $cores|who another-lane: none declared|who holds what|3 held|0 free|!fake python: args"
  release
  hold 3 --cores "$((lo + 2))"
  run "refused at the quota beside a lease that declared one more core" 75 \
    "gpu-lease: refused, holding nothing|who another-lane: cores=$((lo + 2))|who holds what|!fake python: args"
  release
  run "after the quota's holder left" 0 "fake python: args"
else
  echo "skip the quota cases: this machine allows fewer than three CPUs ($allowed)"
fi
echo "$cases cases, $fails failed"
echo "SHELL REHEARSAL $([ "$fails" = 0 ] && echo OK || echo NOT OK)"
[ "$fails" = 0 ]
