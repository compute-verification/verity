#!/usr/bin/env bash
# A host segment's part of a pass (`host_read.py`), gated and then timed on CARDS of one node (cwd = the tree,
# `research run --cwd clone`). The whole script runs under `taskset -c CORES` when CORES is given.
#
# Holding no card, it builds session.cu and kv.cu (p1024, masked) for ARCH, and BASE_REV's session.cu beside them
# (default: the commit this one's branch left `origin/main` at), and writes which of the base's kernels the new build
# holds instruction for instruction and what the kernels only the new build has were built to (`same_sass.py`,
# out/same_sass.json, their listings in out/sass): the run ends there (exit 5) unless the build holds every kernel of the base's as it was and
# none of its own spills, and also when there is no base, the base does not compile or cuobjdump fails. BUILD_ONLY=1
# ends after the build, so a machine with no GPU checks that it compiles (TOOLKIT=1 assembles CUDA 12.9's nvcc first,
# `benchmarks/pouw/harness/jobs/toolkit.sh`). The libraries are built in a scratch directory that goes at exit, or in
# BUILD_DIR, which stays (`host_read_tiers.sh` builds with BUILD_ONLY=1 into its own).
#
# Then host_read.py under one lease on CARDS, of at most LEASE_MIN minutes and taken without waiting: a card whose
# lock is held, or with a compute process or more than a GiB in use on it, ends the run (exit 75) with nothing
# started there, and so does a start after NOT_AFTER (`date -d` syntax; exit 11), so a run stays inside its booked
# hours. The store is GIB GiB on the cards and a host tier of HOST_GIB GiB, in chunks of CHUNK_BLOCKS blocks (the
# tier's step); CTAS are the rows after `pass`, each a host segment's receipt CTAs (none: `pass` alone); BUDGET_S is
# host_read.py's --budget-s and ARGS its other arguments.
# out/host_read.node.txt lists the compute processes on every card of the machine as the lease starts and as it ends.
#
#   research run --on NODE --project verity --source TREE --cwd clone -- \
#     env CARDS=2 CORES=24-35 ARCH=90a LEASE_MIN=25 NOT_AFTER=2026-11-12T16:00Z bash benchmarks/remote_session/host_read.sh
#   research run --on CPU_NODE ... -- env TOOLKIT=1 BUILD_ONLY=1 ARCH=120a bash benchmarks/remote_session/host_read.sh
set -uo pipefail
TREE=$PWD
CARDS=${CARDS:-0} CORES=${CORES:-}
if [ -n "$CORES" ] && [ -z "${HOST_READ_PINNED:-}" ]; then
  exec env HOST_READ_PINNED=1 taskset -c "$CORES" bash "$TREE/benchmarks/remote_session/host_read.sh" "$@"
fi
RS=benchmarks/remote_session O=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$O"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/host_read.txt"; }
first() { local p; for p in "$@"; do [ -x "$p" ] && { echo "$p"; return; }; done; }

GIB=${GIB:-32} HOST_GIB=${HOST_GIB:-1} CHUNK_BLOCKS=${CHUNK_BLOCKS:-65536} BUDGET_S=${BUDGET_S:-1080}
CTAS=${CTAS:-}
LEASE_MIN=${LEASE_MIN:-25} LEASE_MEM_GB=${LEASE_MEM_GB:-64}
N=$(($(tr -cd , <<< "$CARDS" | wc -c) + 1))
if [ -n "${TOOLKIT:-}" ]; then
  # shellcheck disable=SC1091
  . benchmarks/pouw/harness/jobs/toolkit.sh || { say "PREFLIGHT: the toolkit did not assemble"; exit 3; }
fi
NVCC=$(first "${CUDA_HOME:-/nonexistent}/bin/nvcc" /usr/local/cuda-12.9/bin/nvcc /usr/local/cuda-13.0/bin/nvcc \
  /usr/local/cuda/bin/nvcc)
[ -n "$NVCC" ] || { say "PREFLIGHT: no nvcc (CUDA_HOME, /usr/local/cuda-12.9, -13.0)"; exit 3; }
export CUDA_HOME=${NVCC%/bin/nvcc} PATH=${NVCC%/nvcc}:$PATH
# `cuobjdump -sass` runs nvdisasm. Where NVCC's toolkit has none, another toolkit's goes last on the PATH: same_sass.py
# compares two listings the same nvdisasm made, each instruction with its encoding.
command -v nvdisasm > /dev/null || for d in /usr/local/cuda*/bin; do
  [ -x "$d/nvdisasm" ] && { PATH=$PATH:$d; break; }
done
if [ -z "${ARCH:-}" ]; then
  case $(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CARDS%%,*}" 2> /dev/null) in
    *H200* | *H100*) ARCH=90a ;;
    *"RTX PRO 6000"*) ARCH=120a ;;
    *) say "PREFLIGHT: ARCH (90a, 120a) is not given and the card's name does not tell it"; exit 3 ;;
  esac
fi
B=${BUILD_DIR:-}
[ -n "$B" ] || { B=$(mktemp -d /tmp/host-read.XXXXXX); trap 'rm -rf "$B"' EXIT; }
{ id -un; hostname; git -C "$TREE" rev-parse HEAD; echo "cores ${CORES:-not pinned} ($(nproc)) cards $CARDS arch $ARCH"
  "$NVCC" --version | tail -n 2; cat /sys/fs/cgroup/cpu.max; nvidia-smi -L; command -v cuobjdump nvdisasm
} > "$O/machine.txt" 2>&1

FLAGS=(-O3 -gencode "arch=compute_$ARCH,code=sm_$ARCH" -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v)
BASE_REV=${BASE_REV:-$(git -C "$TREE" merge-base HEAD origin/main 2> /dev/null)}
mkdir -p "$B/base/benchmarks/remote_session"
ln -s "$TREE/benchmarks/codec_tensor" "$B/base/benchmarks/codec_tensor"
ln -s "$TREE/verity" "$B/base/verity"
BASE=$B/base/$RS/session.cu
{ [ -n "$BASE_REV" ] && git -C "$TREE" show "$BASE_REV:$RS/session.cu" > "$BASE" 2> /dev/null; } || rm -f "$BASE"
"$NVCC" "${FLAGS[@]}" -o "$B/session.so" $RS/session.cu > "$O/session.ptxas" 2>&1 & S=$!
"$NVCC" "${FLAGS[@]}" -DKV_FIELD=1024 -DKV_MASK=1 -o "$B/kv.so" benchmarks/word_codec/kv.cu > "$O/kv.ptxas" 2>&1 & K=$!
M=
[ -f "$BASE" ] && { "$NVCC" "${FLAGS[@]}" -o "$B/base.so" "$BASE" > "$O/base.ptxas" 2>&1 & M=$!; }
wait $S || { grep -n -m 40 -A 3 'error' "$O/session.ptxas"; say "BUILD: session.cu does not compile for sm_$ARCH"; exit 4; }
wait $K || { tail -n 40 "$O/kv.ptxas"; say "BUILD: kv.cu does not compile for sm_$ARCH"; exit 4; }
[ -n "$M" ] || {
  say "BUILD: no base to compare the kernels with, so nothing runs: give BASE_REV (a commit whose $RS/session.cu" \
    "holds the kernels as they were)"
  exit 5
}
wait "$M" || {
  tail -n 20 "$O/base.ptxas"
  say "BUILD: the base's session.cu ($BASE_REV) does not compile for sm_$ARCH, so nothing runs"
  exit 5
}
sha256sum "$B"/*.so > "$O/libs.sha256"
same=$(python3 $RS/same_sass.py "$B/base.so" "$B/session.so" --out "$O/same_sass.json" --listings "$O/sass" \
  --ptxas "$O/session.ptxas" 2>&1)
rc=$?
say "build: sm_$ARCH; against $BASE_REV: $same"
[ $rc = 0 ] || {
  say "BUILD: same_sass.py exits $rc: this build is not shown to hold every kernel of the base's as it was, or a" \
    "kernel of its own spills, so nothing runs (out/same_sass.json, out/sass)"
  exit 5
}
[ -z "${BUILD_ONLY:-}" ] || exit 0

PY=${PY:-$(first /workspace/venv312/bin/python \
  /workspace/pouw/fill-out/pous/work/pous-bench/gpu312-torch2.14.0-cu130/bin/python)}
[ -x "$PY" ] || { say "PREFLIGHT: no torch venv (PY)"; exit 3; }
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
LEASE=$TREE/tools/research/src/research/pods/sh/gpu_lease.sh
command -v gpu-lease > /dev/null && LEASE=$(command -v gpu-lease)

others() {  # into out/host_read.node.txt: the compute processes on every card of the machine, at the lease's $1
  { echo "$(date -u +%H:%M:%SZ) $1: card, memory used; then each compute process (card's uuid, pid, memory, name)"
    nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader
  } >> "$O/host_read.node.txt" 2>&1
}

if [ -n "${NOT_AFTER:-}" ] && [ "$(date -u +%s)" -gt "$(date -u -d "$NOT_AFTER" +%s)" ]; then
  say "not started, past NOT_AFTER $NOT_AFTER"; exit 11
fi
uuids=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$CARDS")
[ -n "$uuids" ] || { say "not started, nvidia-smi names no card $CARDS"; exit 3; }
busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader | grep -F "$uuids")
used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$CARDS" | awk -F', ' '$2 > 1024')
if [ -n "$busy$used" ]; then
  say "not started, cards $CARDS are in use: ${busy:-no process listed}; MiB used: ${used:-under 1024}"
  exit 75
fi
others start
# shellcheck disable=SC2086
GPU_LEASE_WHO=${GPU_LEASE_WHO:-memory-accounting-rows64} bash "$LEASE" "$N" --on "$CARDS" --max-min "$LEASE_MIN" \
  --mem-gb "$LEASE_MEM_GB" -- "$PY" $RS/host_read.py --lib "$B/session.so" --kv "$B/kv.so" --out "$O" --gib "$GIB" \
  --host-gib "$HOST_GIB" --gpus "$N" --chunk-blocks "$CHUNK_BLOCKS" --budget-s "$BUDGET_S" --ctas $CTAS \
  ${ARGS:-} > "$O/host_read.log" 2>&1
rc=$?
others end
tail -n 60 "$O/host_read.log"
say "host_read.py: exit $rc$([ $rc = 75 ] && echo ": the lease on cards $CARDS was not free, so nothing started")"
say "end (exit $rc); nothing is left on this machine outside the run's own directory ($B goes at exit)"
exit "$rc"
