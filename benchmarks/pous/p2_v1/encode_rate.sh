#!/usr/bin/env bash
# P2's encode on the decode's squaring (p2enc.cuh, p2v1_encode_sq) beside the library's default encoder, on one card
# of one node. cwd = the tree, into which it writes nothing (`research run --cwd source`). The whole script runs under
# `taskset -c CORES` when CORES is given.
#
# First, unless BUILD_ONLY: CARD must be free (free()): its lease not held, no compute process on it and under a GiB
# of its memory in use, and the time not past NOT_AFTER (`date -d` syntax). Else it exits 75 (11 past NOT_AFTER)
# with nothing started. The same check runs again just before the lease.
#
# Holding no card, in a directory of its own that goes at exit, it builds libp2v1.so for ARCH (90a, 120a; the card's
# when not given) with the Makefile's defaults (`default`: both encoders), with REGS=R for each R of A_REGS (`regsR`:
# the default encoder with at most R registers a thread), and from BASE_REV's sources (`base`; default: the commit this
# branch left origin/main at). `encode_bench.py build` then writes out/build.json: ptxas's registers and spills of
# every kernel of every build, each kernel's disassembly's SHA-256, and which of base's kernels default holds
# instruction for instruction. out/ptxas/ holds ptxas's own reports. CGBN at ../p2_gpu/CGBN.commit is the run's
# inputs/cgbn.tar (`git archive` of that commit) when the run was sent one, else GitHub's; gmp.h is the system's, else
# the run's inputs/gmp.h, else gmpy2's wheel's. BUILD_ONLY=1 ends here, so a machine with no GPU checks the build
# (TOOLKIT=1 assembles CUDA 12.9's nvcc first, `benchmarks/pouw/harness/jobs/toolkit.sh`).
#
# Then, still holding no card, the reference's rows on the cores (`encode_bench.py reference`), and under one lease of
# CARD, of at most LEASE_MIN minutes (30 at most) and taken without waiting, `encode_bench.py run`: the gates, the
# timing of both encoders on one segment, `decode_bench.py` and each encoder alone on the card, then the A_REGS
# builds, each held to the default build's bytes before it is timed. No pass starts that would end in the lease's last
# four minutes. SEGMENTS is the store's, for the minutes a whole encode takes; SAY_AT is a rate in blocks a second at
# or above which the run says so on a line of its own.
#
#   research run --on NODE --project verity --source TREE --cwd source [--send cgbn.tar --send gmp.h] -- \
#     env CARD=0 CORES=0-11 NOT_AFTER=TIME SEGMENTS=N bash benchmarks/pous/p2_v1/encode_rate.sh
#   research run --on CPU-NODE ... -- env TOOLKIT=1 BUILD_ONLY=1 ARCH=90a bash benchmarks/pous/p2_v1/encode_rate.sh
set -uo pipefail
TREE=$PWD
CARD=${CARD:-0} CORES=${CORES:-}
if [ -n "$CORES" ] && [ -z "${ENCODE_RATE_PINNED:-}" ]; then
  exec env ENCODE_RATE_PINNED=1 taskset -c "$CORES" bash "$TREE/benchmarks/pous/p2_v1/encode_rate.sh" "$@"
fi
D=benchmarks/pous/p2_v1 O=${OUT:-${RESEARCH_RUN_DIR:?}/out} IN=${RESEARCH_RUN_DIR:-.}/inputs
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$O/ptxas"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/encode-rate.txt"; }
first() { local p; for p in "$@"; do [ -x "$p" ] && { echo "$p"; return; }; done; }
A_REGS=${A_REGS-200 168} LEASE_MIN=${LEASE_MIN:-30} LEASE_MEM_GB=${LEASE_MEM_GB:-32} SEGMENTS=${SEGMENTS:-1}
[ "$LEASE_MIN" -le 30 ] || { say "PREFLIGHT: one lease is at most 30 minutes, not $LEASE_MIN"; exit 3; }
LEASE=$TREE/tools/research/src/research/pods/sh/gpu_lease.sh
command -v gpu-lease > /dev/null && LEASE=$(command -v gpu-lease)

free() {  # CARD may be taken now; else says why not as step $1's, and returns 75 (11 past NOT_AFTER)
  local uuid busy used held
  if [ -n "${NOT_AFTER:-}" ] && [ "$(date -u +%s)" -gt "$(date -u -d "$NOT_AFTER" +%s)" ]; then
    say "$1: not started, past NOT_AFTER $NOT_AFTER"; return 11
  fi
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$CARD")
  [ -n "$uuid" ] || { say "$1: not started, nvidia-smi names no card $CARD"; return 3; }
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader | grep -F "$uuid")
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$CARD" | awk -F', ' '$2 > 1024')
  held=$(bash "$LEASE" status 2> /dev/null | awk -v card="$CARD" '$1 == card && $2 != "free"')
  if [ -n "$busy$used$held" ]; then
    say "$1: not started, card $CARD is not free: lease: ${held:-free}; processes: ${busy:-none}; MiB used: ${used:-under 1024}"
    return 75
  fi
}
if [ -z "${BUILD_ONLY:-}" ]; then free start || exit $?; fi

B=$(mktemp -d "${TMPDIR:-/tmp}/p2enc.XXXXXX"); trap 'rm -rf "$B"' EXIT
if [ -n "${TOOLKIT:-}" ]; then
  # the node's assembled toolkit when it has one, else one of this run's own
  [ -n "${POUW_HARNESS_TOOLKIT:-}" ] || [ -f /workspace/cache/pouw-cuda-12.9.1/ready ] || export POUW_HARNESS_TOOLKIT=$B/toolkit
  # shellcheck disable=SC1091
  . benchmarks/pouw/harness/jobs/toolkit.sh || { say "PREFLIGHT: the toolkit did not assemble"; exit 3; }
fi
NVCC=$(first "${CUDA_HOME:-/nonexistent}/bin/nvcc" /usr/local/cuda-12.9/bin/nvcc /usr/local/cuda-13.0/bin/nvcc \
  /usr/local/cuda/bin/nvcc)
[ -n "$NVCC" ] || { say "PREFLIGHT: no nvcc (CUDA_HOME, /usr/local/cuda-12.9, -13.0)"; exit 3; }
export CUDA_HOME=${NVCC%/bin/nvcc} PATH=${NVCC%/nvcc}:$PATH
# `cuobjdump -sass` runs nvdisasm. Where NVCC's toolkit has none, another toolkit's goes last on the PATH: the builds
# compared are listed by the same one.
command -v nvdisasm > /dev/null || for d in /usr/local/cuda*/bin; do
  [ -x "$d/nvdisasm" ] && { PATH=$PATH:$d; break; }
done
if [ -z "${ARCH:-}" ]; then
  case $(nvidia-smi --query-gpu=name --format=csv,noheader -i "$CARD" 2> /dev/null) in
    *H200* | *H100*) ARCH=90a ;;
    *"RTX PRO 6000"*) ARCH=120a ;;
    *) say "PREFLIGHT: ARCH (90a, 120a) is not given and the card's name does not tell it"; exit 3 ;;
  esac
fi
{ id -un; hostname; git -C "$TREE" rev-parse HEAD; echo "cores ${CORES:-not pinned} ($(nproc)) card $CARD arch $ARCH"
  "$NVCC" --version | tail -n 2; cat /sys/fs/cgroup/cpu.max; nvidia-smi -L; command -v cuobjdump nvdisasm
} > "$O/machine.txt" 2>&1

PIN=$(cat benchmarks/pous/p2_gpu/CGBN.commit)
if [ -f "$IN/cgbn.tar" ]; then
  [ "$(git get-tar-commit-id < "$IN/cgbn.tar")" = "$PIN" ] && mkdir "$B/CGBN" && tar -xf "$IN/cgbn.tar" -C "$B/CGBN" \
    && CGBN_FROM="inputs/cgbn.tar, sha256 $(sha256sum < "$IN/cgbn.tar" | cut -c1-64)"
else
  git clone -q https://github.com/NVlabs/CGBN "$B/CGBN" && git -C "$B/CGBN" checkout -q "$PIN" && CGBN_FROM=GitHub
fi || { say "PREFLIGHT: no CGBN at $PIN (inputs/cgbn.tar is not that commit's git archive, or GitHub did not give it)"; exit 3; }
INC=
if ! echo '#include <gmp.h>' | cpp > /dev/null 2>&1; then
  if [ -f "$IN/gmp.h" ]; then
    mkdir "$B/gmp" && cp "$IN/gmp.h" "$B/gmp/" && GMP=$B/gmp
  else
    if command -v uv > /dev/null; then uv pip install -q --target "$B/py" gmpy2; else python3 -m pip install -q --target "$B/py" gmpy2; fi
    GMP=$(find "$B/py" -name gmp.h -printf '%h\n' 2> /dev/null | head -1)
  fi
  [ -f "${GMP:-/nonexistent}/gmp.h" ] || { say "PREFLIGHT: no gmp.h (the system's, inputs/gmp.h, gmpy2's wheel's)"; exit 3; }
  INC="-I $GMP"
fi
BASE_REV=${BASE_REV:-$(git -C "$TREE" merge-base HEAD origin/main 2> /dev/null)}
mkdir "$B/base-src"
[ -n "$BASE_REV" ] && git -C "$TREE" archive "$BASE_REV" $D benchmarks/pous/p2_gpu verity/definitions/kernels/sha512.cuh \
  2> /dev/null | tar -x -C "$B/base-src" || BASE_REV=

mk() {  # mk NAME ROOT [VARIABLE=VALUE ...]: ROOT's libp2v1.so and ptxas's report into $B/NAME
  local name=$1 root=$2; shift 2
  make -C "$root/$D" NVCC="$NVCC" CGBN="$B/CGBN" BIN="$B/$name" ARCH="sm_$ARCH" ${INC:+"INC=$INC"} "$@" > "$B/$name.make" 2>&1
}
builds=(default); pids=()
mk default "$TREE" & pids+=($!)
for r in $A_REGS; do builds+=("regs$r"); mk "regs$r" "$TREE" REGS="$r" & pids+=($!); done
[ -z "$BASE_REV" ] || { builds+=(base); mk base "$B/base-src" & pids+=($!); }
libs=()
for i in "${!builds[@]}"; do
  b=${builds[$i]}
  wait "${pids[$i]}"; rc=$?
  cp "$B/$b/libp2v1.so.ptxas" "$O/ptxas/$b.txt" 2> /dev/null
  if [ $rc != 0 ]; then
    tail -n 5 "$B/$b.make"; grep -n -m 20 -A 3 -i 'error' "$O/ptxas/$b.txt"
    say "BUILD: $b does not build for sm_$ARCH"; exit 4
  fi
  libs+=(--lib "$b=$B/$b")
done
(cd "$B" && sha256sum -- */libp2v1.so) > "$O/libs.sha256"
{ "$NVCC" --version | tail -n 2; echo "CGBN $PIN from $CGBN_FROM; gmp.h: ${INC:-of the system}; base ${BASE_REV:-none}"; } > "$O/toolchain.txt"
python3 $D/encode_bench.py build "${libs[@]}" --out "$O/build.json" > "$O/build.txt" 2>&1 \
  || { cat "$O/build.txt"; say "BUILD: no record of the builds (out/build.txt)"; exit 4; }
cat "$O/build.txt"
say "build: sm_$ARCH, ${builds[*]}; ptxas's registers and spills in out/build.json"
[ -z "${BUILD_ONLY:-}" ] || exit 0

PY=${PY:-$(first /workspace/venv312/bin/python /workspace/cp/venv312/bin/python)}
[ -x "$PY" ] || { say "PREFLIGHT: no torch venv (PY)"; exit 3; }
export PYTHONPATH=$TREE:$TREE/benchmarks/pous${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
"$PY" $D/encode_bench.py reference --out "$O/reference.json" > "$O/reference.txt" 2>&1 \
  || { tail -n 20 "$O/reference.txt"; say "REFERENCE: the reference's rows did not compute (out/reference.txt)"; exit 5; }
say "$(tail -n 1 "$O/reference.txt")"

free lease || exit $?
a=(); for r in $A_REGS; do a+=(--a "regs$r=$B/regs$r"); done
GPU_LEASE_WHO=${GPU_LEASE_WHO:-p2-encode-rate} bash "$LEASE" 1 --on "$CARD" --max-min "$LEASE_MIN" --mem-gb "$LEASE_MEM_GB" -- \
  "$PY" $D/encode_bench.py run --lib "$B/default" --reference "$O/reference.json" --out "$O" "${a[@]}" --decode-bench \
  --segments "$SEGMENTS" --minutes "$((LEASE_MIN - 4))" ${SAY_AT:+--say-at "$SAY_AT"} > "$O/run.txt" 2>&1
rc=$?
tail -n 60 "$O/run.txt"
say "run: exit $rc$([ $rc = 75 ] && echo ": the lease on card $CARD was not free, so nothing started")"
say "end (exit $rc); nothing is left on this machine outside the run's own directory ($B goes at exit)"
exit "$rc"
