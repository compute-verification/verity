#!/usr/bin/env bash
# The session's pass at 64 and at 16 sketch rows on CARDS of one node (cwd = the tree; `research run --cwd clone`,
# which gives BASE_REV's session.cu to `git show`, and whose verifier builds its Lean inside the tree when STEPS has
# `session`). The whole script runs under `taskset -c CORES` when CORES is given.
#
# Holding no card, it builds session.cu and kv.cu (p1024, masked) for ARCH, and writes which kernels of BASE_REV's
# session.cu (default: the commit this one's branch left `origin/main` at) the new build holds instruction for
# instruction (`same_sass.py`, out/same_sass.json): at 16 rows the pass runs those, so the run ends there (exit 5)
# unless the build holds every one, and also when there is no base, the base does not compile or cuobjdump fails.
# The kernels the base has none of, the 64-row pass's, get a record and their listings (`sass_gate.py`,
# out/sass_gate.json, out/sass64), and the run ends there too (exit 5) if one holds what that gate refuses.
# BUILD_ONLY=1 ends after the build, so a machine with no GPU checks that it compiles (TOOLKIT=1 assembles CUDA
# 12.9's nvcc first, `benchmarks/pouw/harness/jobs/toolkit.sh`; CUDA_HOME names a toolkit, which goes first).
# DEFS are session.cu's own nvcc arguments (-DSK64_PER_SM=2: the 64-row receipt kernels at 128 registers a thread).
#
# Then each of STEPS (`rows session` unless given) under a lease of its own on CARDS, of at most LEASE_MIN minutes and
# taken without waiting: a card whose lock is held, or with a compute process or more than a GiB in use on it, ends
# the run (exit 75) with nothing started there. No step starts after NOT_AFTER (`date -d` syntax), so a run stays
# inside its booked hours. out/STEP.node.txt lists the compute processes on every card of the machine as the step
# starts and as it ends: whether anything shared the node with a timed row. With them, each card's link, and this
# cgroup's CPU counters, its pressure and the machine's load: whether the cgroup was throttled between the two.
#
#   rows      rows.py: at each of ROWS in turn on one store of GIB GiB across CARDS, the device's gates, the store's
#             openings and the calibration (CAL chained passes each, over the plain read); out/rows.json. CTAS_AT
#             (rows.py's --ctas-at words, `64=96,128,264`) gives a row count its receipt CTA counts, each in turn on
#             that store: which count the pass over the read is least at, and the count a timed run takes. ROWS_ARGS
#             are rows.py's own (--whole-store)
#   session   PLAN's sessions at SESSION_ROWS rows over SESSION_GIB GiB on the first card, the verifier's process
#             (on VERIFIER_CORES) beside the prover (on PROVER_CORES) on this machine, which audits nothing: a check
#             of the kernels against the verifier; out/session/. It registers as SESSION_NODE (default: this
#             machine's name, the card's index and its model)
#
#   research run --on NODE --project verity --source TREE --cwd clone -- \
#     env CARDS=0 CORES=0-7 ARCH=120a NOT_AFTER=2026-11-20T18:00Z bash benchmarks/remote_session/rows.sh
#   ... env BUILD_ONLY=1 ARCH=120a CUDA_HOME=/usr/local/cuda-13.0 BASE_REV=REV bash benchmarks/remote_session/rows.sh
#   ... env CARDS=0 CORES=0-7 ARCH=120a STEPS=rows ROWS='64 16' GIB=62 CTAS_AT='64=96,128,188' ...
set -uo pipefail
TREE=$PWD
CARDS=${CARDS:-0} CORES=${CORES:-}
if [ -n "$CORES" ] && [ -z "${ROWS_PINNED:-}" ]; then
  exec env ROWS_PINNED=1 taskset -c "$CORES" bash "$TREE/benchmarks/remote_session/rows.sh" "$@"
fi
export RS=benchmarks/remote_session PR=verity/protocols/accounting/poub/prover O=${OUT:-${RESEARCH_RUN_DIR:?}/out}
mkdir -p "$O"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/rows.txt"; }
first() { local p; for p in "$@"; do [ -x "$p" ] && { echo "$p"; return; }; done; }
pinned() { local cores=$1; shift; if [ -n "$cores" ]; then taskset -c "$cores" "$@"; else "$@"; fi; }

session() {  # under the lease: PLAN's sessions, the verifier's process beside the prover
  local port=$((7700 + RANDOM % 200)) sessions v p w
  sessions=$(($(tr -cd , <<< "$PLAN" | wc -c) + 1))
  mkdir -p "$O/session"
  pinned "${VERIFIER_CORES:-}" "$PY" verity/protocols/accounting/poub/verifier/remote_session/verifier_process.py \
    --port "$port" --out "$O/session" --windows "$WINDOWS" --window-ms "$WINDOW_MS" --samples "$SAMPLES" \
    --jitter-ms "$JITTER_MS" --sessions "$sessions" --lean-exe "$LEAN_EXE" > "$O/session/verifier.log" 2>&1 & v=$!
  pinned "${PROVER_CORES:-}" "$PY" $PR/remote_session/prover.py --connect "127.0.0.1:$port" --out "$O/session" \
    --plan "$PLAN" --rows "$SESSION_ROWS" --lib "$B/session.so" --kv "$B/kv.so" --gib "$SESSION_GIB" --gpus 1 \
    --calibration-passes "$CAL" --sketch-ctas "$SKETCH_CTAS" --node "$SESSION_NODE" > "$O/session/prover.log" 2>&1
  p=$?
  [ $p = 0 ] || kill $v 2> /dev/null
  wait $v; w=$?
  tail -n 8 "$O/session/verifier.log" "$O/session/prover.log"
  [ $p = 0 ] && [ $w = 0 ]
}
if [ "${1:-}" = session ]; then session; exit $?; fi

STEPS=${STEPS-rows session} ROWS=${ROWS:-64 16} GIB=${GIB:-16}
LEASE_MIN=${LEASE_MIN:-30} LEASE_MEM_GB=${LEASE_MEM_GB:-64}
export CAL=${CAL:-400} SKETCH_CTAS=${SKETCH_CTAS:-128} PLAN=${PLAN:-honest,skip:0.0625}
export SESSION_ROWS=${SESSION_ROWS:-64} SESSION_GIB=${SESSION_GIB:-16}
# The node a session registers: this machine, its first card and the card's model (prover.py's default names another
# machine), in the 64 bytes the verifier takes.
SESSION_NODE=${SESSION_NODE:-$(echo "$(hostname)/${CARDS%%,*}/$(nvidia-smi --query-gpu=name --format=csv,noheader \
  -i "${CARDS%%,*}" 2> /dev/null | tr 'A-Z ' 'a-z-')" | cut -c1-64)}
export SESSION_NODE
export WINDOWS=${WINDOWS:-7} WINDOW_MS=${WINDOW_MS:-1000} SAMPLES=${SAMPLES:-2773} JITTER_MS=${JITTER_MS:-10}
N=$(($(tr -cd , <<< "$CARDS" | wc -c) + 1))
if [ -n "${TOOLKIT:-}" ]; then
  # shellcheck disable=SC1091
  . benchmarks/pouw/harness/jobs/toolkit.sh || { say "PREFLIGHT: the toolkit did not assemble"; exit 3; }
fi
NVCC=$(first "${CUDA_HOME:-/nonexistent}/bin/nvcc" /usr/local/cuda-12.9/bin/nvcc /usr/local/cuda-13.0/bin/nvcc \
  /usr/local/cuda/bin/nvcc)
[ -n "$NVCC" ] || { say "PREFLIGHT: no nvcc (CUDA_HOME, /usr/local/cuda-12.9, -13.0)"; exit 3; }
if [ -n "${CUDA_HOME:-}" ] && [ "$NVCC" != "$CUDA_HOME/bin/nvcc" ]; then
  say "PREFLIGHT: CUDA_HOME=$CUDA_HOME has no bin/nvcc, so the build would be another toolkit's"; exit 3
fi
export CUDA_HOME=${NVCC%/bin/nvcc} PATH=${NVCC%/nvcc}:$PATH
# `cuobjdump -sass` runs nvdisasm. Where NVCC's toolkit has none, another toolkit's goes last on the PATH: same_sass.py
# compares two listings the same nvdisasm made, each instruction with its encoding.
command -v nvdisasm > /dev/null || for d in /usr/local/cuda*/bin; do
  [ -x "$d/nvdisasm" ] && { PATH=$PATH:$d; break; }
done
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
  "$NVCC" --version | tail -n 2; cat /sys/fs/cgroup/cpu.max; nvidia-smi -L; command -v lake cuobjdump nvdisasm
  nvidia-smi topo -m
} > "$O/machine.txt" 2>&1

FLAGS=(-O3 -gencode "arch=compute_$ARCH,code=sm_$ARCH" -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v)
BASE_REV=${BASE_REV:-$(git -C "$TREE" merge-base HEAD origin/main 2> /dev/null)}
# The base's session.cu where the tree's is, beside the tree's own includes (the circuits' kernels and the reader).
mkdir -p "$B/base/$PR/remote_session"
ln -s "$TREE/verity/circuits" "$B/base/verity/circuits"
ln -s "$TREE/$PR/codec_tensor" "$B/base/$PR/codec_tensor"
BASE=$B/base/$PR/remote_session/session.cu
{ [ -n "$BASE_REV" ] && git -C "$TREE" show "$BASE_REV:$PR/remote_session/session.cu" > "$BASE" 2> /dev/null; } \
  || rm -f "$BASE"
# shellcheck disable=SC2086
"$NVCC" "${FLAGS[@]}" ${DEFS:-} -o "$B/session.so" $PR/remote_session/session.cu > "$O/session.ptxas" 2>&1 & S=$!
"$NVCC" "${FLAGS[@]}" -DKV_FIELD=1024 -DKV_MASK=1 -o "$B/kv.so" $PR/word_codec/kv.cu > "$O/kv.ptxas" 2>&1 & K=$!
M=
[ -f "$BASE" ] && { "$NVCC" "${FLAGS[@]}" -o "$B/base.so" "$BASE" > "$O/base.ptxas" 2>&1 & M=$!; }
wait $S || { grep -n -m 40 -A 3 'error' "$O/session.ptxas"; say "BUILD: session.cu does not compile for sm_$ARCH"; exit 4; }
wait $K || { tail -n 40 "$O/kv.ptxas"; say "BUILD: kv.cu does not compile for sm_$ARCH"; exit 4; }
[ -n "$M" ] || {
  say "BUILD: no base to compare the kernels with, so nothing runs: give BASE_REV (a commit whose" \
    "$PR/remote_session/session.cu is main's)"
  exit 5
}
wait "$M" || {
  tail -n 20 "$O/base.ptxas"
  say "BUILD: the base's session.cu ($BASE_REV) does not compile for sm_$ARCH, so nothing runs"
  exit 5
}
sha256sum "$B"/*.so > "$O/libs.sha256"
same=$(python3 $RS/same_sass.py "$B/base.so" "$B/session.so" --out "$O/same_sass.json" --listings "$O/sass" 2>&1)
rc=$?
say "build: sm_$ARCH; against $BASE_REV: $same"
[ $rc = 0 ] || {
  say "BUILD: same_sass.py exits $rc: this build is not shown to hold every kernel of the base's as it was, so" \
    "nothing runs (out/same_sass.json, out/sass)"
  exit 5
}
# The kernels the base has none of: their record and listings, and what the 64-row kernels may not hold.
cuobjdump -sass "$B/session.so" > "$B/session.sass" 2> "$O/cuobjdump.err" || {
  say "BUILD: cuobjdump -sass exits $? on the build ($(tr '\n' ' ' < "$O/cuobjdump.err" | cut -c1-300)), so nothing runs"
  exit 5
}
gate=$(python3 $RS/sass_gate.py "$O/session.ptxas" "$B/session.sass" "$O/sass_gate.json" --arch "sm_$ARCH" \
  --listings "$O/sass64" 2>&1)
rc=$?
say "build: the 64-row kernels: $gate"
[ $rc = 0 ] || {
  say "BUILD: sass_gate.py exits $rc: a 64-row kernel is not what its gate allows, so nothing runs" \
    "(out/sass_gate.json, out/sass64)"
  exit 5
}
[ -z "${BUILD_ONLY:-}" ] || exit 0

export PY=${PY:-$(first /workspace/venv312/bin/python \
  /workspace/pouw/fill-out/pous/work/pous-bench/gpu312-torch2.14.0-cu130/bin/python)}
[ -x "$PY" ] || { say "PREFLIGHT: no torch venv (PY)"; exit 3; }
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
LEASE=$TREE/tools/research/src/research/pods/sh/gpu_lease.sh
command -v gpu-lease > /dev/null && LEASE=$(command -v gpu-lease)

others() {  # into out/$1.node.txt: the compute processes on every card of the machine, at the step's $2
  { echo "$(date -u +%H:%M:%SZ) $2: card, memory used, its link's generation and width; then each compute process" \
      "(card's uuid, pid, memory, name); then the cgroup's cpu.stat and cpu.pressure, and the load"
    nvidia-smi --query-gpu=index,uuid,memory.used,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader
    cat /sys/fs/cgroup/cpu.stat /sys/fs/cgroup/cpu.pressure /proc/loadavg
  } >> "$O/$1.node.txt" 2>&1
}

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
  others "$name" start
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-memory-accounting-rows64} bash "$LEASE" "$N" --on "$CARDS" --max-min "$LEASE_MIN" \
    --mem-gb "$LEASE_MEM_GB" -- "$@" > "$O/$name.log" 2>&1
  rc=$?
  others "$name" end
  tail -n 40 "$O/$name.log"
  say "$name: exit $rc$([ $rc = 75 ] && echo ": the lease on cards $CARDS was not free, so nothing started")"
  return $rc
}

bad=0
for step in $STEPS; do
  case $step in
    rows)
      # shellcheck disable=SC2086
      leased rows "$PY" $RS/rows.py --lib "$B/session.so" --kv "$B/kv.so" --out "$O" --rows $ROWS --gib "$GIB" \
        --gpus "$N" --calibration-passes "$CAL" --sketch-ctas "$SKETCH_CTAS" ${CTAS_AT:+--ctas-at $CTAS_AT} \
        ${ROWS_ARGS:-} ;;
    session)
      # The verifier's Lean, built on the cores before the card is taken.
      LEAN=verity/protocols/accounting/poub/verifier/lean
      if (cd "$LEAN" && lake build remote-audit-verifier) > "$O/lean-build.log" 2>&1; then
        export LEAN_EXE=$TREE/$LEAN/.lake/build/bin/remote-audit-verifier
        leased session bash "$TREE/$RS/rows.sh" session
      else
        say "session: not started, the Lean verifier did not build (out/lean-build.log)"; false
      fi ;;
    *) say "a step is rows or session, not $step"; exit 3 ;;
  esac
  rc=$?
  [ $rc = 0 ] || bad=$rc
  [ $rc = 75 ] || [ $rc = 11 ] && break
done
say "end (exit $bad); nothing is left on this machine outside the run's own directory ($B goes at exit)"
exit "$bad"
