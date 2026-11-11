#!/usr/bin/env bash
# The session's pass at 64 and at 16 sketch rows on CARDS of one node (cwd = the tree; `research run --cwd clone`
# when STEPS has `session`, whose verifier builds its Lean inside the tree). The whole script runs under
# `taskset -c CORES` when CORES is given.
#
# Holding no card, it builds session.cu and kv.cu (p1024, masked) for ARCH, and writes which kernels of BASE_REV's
# session.cu (default: the commit this one's branch left `origin/main` at; else the file sent as
# inputs/main-session.cu) the new build holds instruction for instruction (`same_sass.py`, out/same_sass.json): at 16
# rows the pass runs those. BUILD_ONLY=1 ends there, so a machine with no GPU checks that it compiles (TOOLKIT=1
# assembles CUDA 12.9's nvcc first, `benchmarks/pouw/harness/jobs/toolkit.sh`). DEFS are session.cu's own nvcc
# arguments (-DSK64_PER_SM=2: the 64-row receipt kernels at 128 registers a thread).
#
# Then each of STEPS (`rows session` unless given) under a lease of its own on CARDS, of at most LEASE_MIN minutes and
# taken without waiting: a card whose lock is held, or with a compute process or more than a GiB in use on it, ends
# the run (exit 75) with nothing started there. No step starts after NOT_AFTER (`date -d` syntax), so a run stays
# inside its booked hours.
#
#   rows      rows.py: at each of ROWS in turn on one store of GIB GiB across CARDS, the device's gates, the store's
#             openings and the calibration (CAL chained passes each, over the plain read); out/rows.json
#   session   PLAN's sessions at SESSION_ROWS rows over SESSION_GIB GiB on the first card, the verifier's process
#             (on VERIFIER_CORES) beside the prover (on PROVER_CORES) on this machine, which audits nothing: a check
#             of the kernels against the verifier; out/session/
#   look      rows.py at LOOK_ROWS for each of LOOK_CTAS receipt CTAs in turn, a lease each: which count the pass over
#             the read is least at; out/look-CTAS/rows.json
#   inplace   inplace_gate.py --no-timing at 64 and at 16 rows: the in-place pass against the reference
#
#   research run --on NODE --project verity --source TREE --cwd clone --send main-session.cu -- \
#     env CARDS=0 CORES=0-7 ARCH=90a NOT_AFTER=2026-11-11T11:30Z bash benchmarks/remote_session/rows.sh
#   ... env CARDS=0,1,2,3,4,5,6,7 CORES=0-15 ARCH=90a STEPS=rows ROWS='16 64 16 64' GIB=496 CAL=4000 LEASE_MIN=60 ...
set -uo pipefail
TREE=$PWD
CARDS=${CARDS:-0} CORES=${CORES:-}
if [ -n "$CORES" ] && [ -z "${ROWS_PINNED:-}" ]; then
  exec env ROWS_PINNED=1 taskset -c "$CORES" bash "$TREE/benchmarks/remote_session/rows.sh" "$@"
fi
export RS=benchmarks/remote_session O=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$O"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/rows.txt"; }
first() { local p; for p in "$@"; do [ -x "$p" ] && { echo "$p"; return; }; done; }
pinned() { local cores=$1; shift; if [ -n "$cores" ]; then taskset -c "$cores" "$@"; else "$@"; fi; }

session() {  # under the lease: PLAN's sessions, the verifier's process beside the prover
  local port=$((7700 + RANDOM % 200)) sessions v p w
  sessions=$(($(tr -cd , <<< "$PLAN" | wc -c) + 1))
  mkdir -p "$O/session"
  pinned "${VERIFIER_CORES:-}" "$PY" verity/protocols/poub/verifier/remote_session/verifier_process.py \
    --port "$port" --out "$O/session" --windows "$WINDOWS" --window-ms "$WINDOW_MS" --samples "$SAMPLES" \
    --jitter-ms "$JITTER_MS" --sessions "$sessions" --lean-exe "$LEAN_EXE" > "$O/session/verifier.log" 2>&1 & v=$!
  pinned "${PROVER_CORES:-}" "$PY" $RS/prover.py --connect "127.0.0.1:$port" --out "$O/session" --plan "$PLAN" \
    --rows "$SESSION_ROWS" --lib "$B/session.so" --kv "$B/kv.so" --gib "$SESSION_GIB" --gpus 1 \
    --calibration-passes "$CAL" --sketch-ctas "$SKETCH_CTAS" > "$O/session/prover.log" 2>&1
  p=$?
  [ $p = 0 ] || kill $v 2> /dev/null
  wait $v; w=$?
  tail -n 8 "$O/session/verifier.log" "$O/session/prover.log"
  [ $p = 0 ] && [ $w = 0 ]
}
if [ "${1:-}" = session ]; then session; exit $?; fi

STEPS=${STEPS-rows session} ROWS=${ROWS:-64 16} GIB=${GIB:-16}
LOOK_ROWS=${LOOK_ROWS:-64} LOOK_CTAS=${LOOK_CTAS:-64 96 128}
LEASE_MIN=${LEASE_MIN:-30} LEASE_MEM_GB=${LEASE_MEM_GB:-64}
export CAL=${CAL:-400} SKETCH_CTAS=${SKETCH_CTAS:-128} PLAN=${PLAN:-honest,skip:0.0625}
export SESSION_ROWS=${SESSION_ROWS:-64} SESSION_GIB=${SESSION_GIB:-16}
export WINDOWS=${WINDOWS:-7} WINDOW_MS=${WINDOW_MS:-1000} SAMPLES=${SAMPLES:-2773} JITTER_MS=${JITTER_MS:-10}
N=$(($(tr -cd , <<< "$CARDS" | wc -c) + 1))
if [ -n "${TOOLKIT:-}" ]; then
  # shellcheck disable=SC1091
  . benchmarks/pouw/harness/jobs/toolkit.sh || { say "PREFLIGHT: the toolkit did not assemble"; exit 3; }
fi
NVCC=$(first "${CUDA_HOME:-/nonexistent}/bin/nvcc" /usr/local/cuda-12.9/bin/nvcc /usr/local/cuda-13.0/bin/nvcc \
  /usr/local/cuda/bin/nvcc)
[ -n "$NVCC" ] || { say "PREFLIGHT: no nvcc (CUDA_HOME, /usr/local/cuda-12.9, -13.0)"; exit 3; }
export CUDA_HOME=${NVCC%/bin/nvcc} PATH=${NVCC%/nvcc}:$PATH
[ -x "$HOME/.elan/bin/lake" ] && export PATH=$HOME/.elan/bin:$PATH
if [ -z "${ARCH:-}" ]; then
  case $(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CARDS%%,*}" 2> /dev/null) in
    *H200* | *H100*) ARCH=90a ;;
    *"RTX PRO 6000"*) ARCH=120a ;;
    *) say "PREFLIGHT: ARCH (90a, 120a) is not given and the card's name does not tell it"; exit 3 ;;
  esac
fi
B=$(mktemp -d /tmp/rows64.XXXXXX); trap 'rm -rf "$B"' EXIT
export B
{ id -un; hostname; git -C "$TREE" rev-parse HEAD; echo "cores ${CORES:-not pinned} ($(nproc)) cards $CARDS arch $ARCH"
  "$NVCC" --version | tail -n 2; cat /sys/fs/cgroup/cpu.max; nvidia-smi -L; command -v lake cuobjdump
} > "$O/machine.txt" 2>&1

FLAGS=(-O3 -gencode "arch=compute_$ARCH,code=sm_$ARCH" -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v)
BASE_REV=${BASE_REV:-$(git -C "$TREE" merge-base HEAD origin/main 2> /dev/null)}
mkdir -p "$B/base/benchmarks/remote_session"
ln -s "$TREE/benchmarks/codec_tensor" "$B/base/benchmarks/codec_tensor"
ln -s "$TREE/verity" "$B/base/verity"
BASE=$B/base/$RS/session.cu
{ [ -n "$BASE_REV" ] && git -C "$TREE" show "$BASE_REV:$RS/session.cu" > "$BASE" 2> /dev/null; } \
  || cp "${RESEARCH_RUN_DIR:-.}/inputs/main-session.cu" "$BASE" 2> /dev/null || rm -f "$BASE"
# shellcheck disable=SC2086
"$NVCC" "${FLAGS[@]}" ${DEFS:-} -o "$B/session.so" $RS/session.cu > "$O/session.ptxas" 2>&1 & S=$!
"$NVCC" "${FLAGS[@]}" -DKV_FIELD=1024 -DKV_MASK=1 -o "$B/kv.so" benchmarks/word_codec/kv.cu > "$O/kv.ptxas" 2>&1 & K=$!
M=
[ -f "$BASE" ] && { "$NVCC" "${FLAGS[@]}" -o "$B/base.so" "$BASE" > "$O/base.ptxas" 2>&1 & M=$!; }
wait $S || { grep -n -m 40 -A 3 'error' "$O/session.ptxas"; say "BUILD: session.cu does not compile for sm_$ARCH"; exit 4; }
wait $K || { tail -n 40 "$O/kv.ptxas"; say "BUILD: kv.cu does not compile for sm_$ARCH"; exit 4; }
sha256sum "$B"/*.so > "$O/libs.sha256"
if [ -n "$M" ] && wait "$M"; then
  say "build: sm_$ARCH; against ${BASE_REV:-the sent file}:" \
    "$(python3 $RS/same_sass.py "$B/base.so" "$B/session.so" --out "$O/same_sass.json" --listings "$O/sass" 2>&1)"
else
  say "build: sm_$ARCH; no base build to compare the kernels with (BASE_REV, or inputs/main-session.cu)"
fi
[ -z "${BUILD_ONLY:-}" ] || exit 0

export PY=${PY:-$(first /workspace/venv312/bin/python \
  /workspace/pouw/fill-out/pous/work/pous-bench/gpu312-torch2.14.0-cu130/bin/python)}
[ -x "$PY" ] || { say "PREFLIGHT: no torch venv (PY)"; exit 3; }
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
LEASE=$TREE/tools/research/src/research/pods/sh/gpu_lease.sh
command -v gpu-lease > /dev/null && LEASE=$(command -v gpu-lease)

leased() {  # step $1 under one lease of CARDS: the rest is its command, its output out/$1.log
  local name=$1 uuids busy used rc; shift
  if [ -n "${NOT_AFTER:-}" ] && [ "$(date -u +%s)" -gt "$(date -u -d "$NOT_AFTER" +%s)" ]; then
    say "$name: not started, past NOT_AFTER $NOT_AFTER"; return 11
  fi
  uuids=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$CARDS")
  [ -n "$uuids" ] || { say "$name: not started, nvidia-smi names no card $CARDS"; return 3; }
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader | grep -F "$uuids")
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$CARDS" | awk -F', ' '$2 > 1024')
  if [ -n "$busy$used" ]; then
    say "$name: not started, cards $CARDS are in use: ${busy:-no process listed}; MiB used: ${used:-under 1024}"
    return 75
  fi
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-memory-accounting-rows64} bash "$LEASE" "$N" --on "$CARDS" --max-min "$LEASE_MIN" \
    --mem-gb "$LEASE_MEM_GB" -- "$@" > "$O/$name.log" 2>&1
  rc=$?
  tail -n 12 "$O/$name.log"
  say "$name: exit $rc$([ $rc = 75 ] && echo ": the lease on cards $CARDS was not free, so nothing started")"
  return $rc
}

bad=0
for step in $STEPS; do
  case $step in
    rows)
      # shellcheck disable=SC2086
      leased rows "$PY" $RS/rows.py --lib "$B/session.so" --kv "$B/kv.so" --out "$O" --rows $ROWS --gib "$GIB" \
        --gpus "$N" --calibration-passes "$CAL" --sketch-ctas "$SKETCH_CTAS" ${ROWS_ARGS:-} ;;
    look)
      rc=0
      for c in $LOOK_CTAS; do
        mkdir -p "$O/look-$c"
        # shellcheck disable=SC2086
        leased "look-$c" "$PY" $RS/rows.py --lib "$B/session.so" --kv "$B/kv.so" --out "$O/look-$c" --rows $LOOK_ROWS \
          --gib "$GIB" --gpus "$N" --calibration-passes "$CAL" --sketch-ctas "$c" ${ROWS_ARGS:-} || { rc=$?; break; }
      done
      (exit $rc) ;;
    inplace)
      # shellcheck disable=SC2016
      leased inplace bash -c 'rc=0; for k in 64 16; do mkdir -p "$O/inplace-$k"
        "$PY" $RS/inplace_gate.py --lib "$B/session.so" --out "$O/inplace-$k" --rows $k --no-timing || rc=$?; done
        exit $rc' ;;
    session)
      # The verifier's Lean, built on the cores before the card is taken.
      if (cd verity/protocols/poub/verifier/lean && lake build remote-audit-verifier) > "$O/lean-build.log" 2>&1; then
        export LEAN_EXE=$TREE/verity/protocols/poub/verifier/lean/.lake/build/bin/remote-audit-verifier
        leased session bash "$TREE/$RS/rows.sh" session
      else
        say "session: not started, the Lean verifier did not build (out/lean-build.log)"; false
      fi ;;
    *) say "a step is rows, look, inplace or session, not $step"; exit 3 ;;
  esac
  rc=$?
  [ $rc = 0 ] || bad=$rc
  [ $rc = 75 ] || [ $rc = 11 ] && break
done
say "end (exit $bad); nothing is left on this machine outside the run's own directory ($B goes at exit)"
exit "$bad"
