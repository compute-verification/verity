#!/usr/bin/env bash
# stage_prove.sh: window.py's stage and prove steps for a few statements, in one job or in two (STEPS), on the CPU or on named
# cards of a GPU node, every session kept under out/kept/NAME for lean_shard.sh. Run from a clone of the tree, whose Flock
# package builds flock-read-tmpl in place for stage.py:
#   research run --on NODE --cwd clone --source TREE --send UNITS_FILE --env UNITS=UNITS_FILE --env DEV=gpu --env CARDS=4,5 \
#     --env SM=90 --env NOT_BEFORE=EPOCH --env KILL_AT=EPOCH -- taskset -c CORES bash benchmarks/served_units/stage_prove.sh
# STEPS="stage prove" (the default) runs both in one job. STEPS=stage builds and stages into FLOCK_WORK and keeps it, touching
# no card (CUDA_VISIBLE_DEVICES empty); STEPS=prove, from the same commit, proves what it kept (`--cwd source` will do), so
# that a short window on the cards holds the proving alone.
# Steps, each logged to out/steps.txt:
#   0. the start: nothing before NOT_BEFORE (UTC epoch seconds; it sleeps holding nothing); then KILL_AT at least MIN_LEFT_S
#      away; for STEPS=prove, a FLOCK_WORK this commit staged (its .staged); and to prove on DEV=gpu, every card of CARDS
#      free in `gpu-lease status` and holding no compute process; else exit 9 having started nothing
#   1. (stage) flock-read-tmpl built in the clone's Flock package (lake build), which stage.py runs for a table read's template
#   2. (stage) window.py stage UNITS into FLOCK_WORK on disk, a fresh directory under WORK_ROOT unless FLOCK_WORK is given:
#      prove_time.sh's stage step builds flock-tmpl-unit under it and places each templated statement's unit with it, so the
#      job refuses a FLOCK_WORK under /dev/shm or on a noexec mount; window.py builds the prover beside the stagings. Under
#      RAM_WORK=1 a FLOCK_WORK under /dev/shm holds the statements alone, its py, flock-gpu-link, flock-circuit and tmpl-unit
#      symlinks into DISK_WORK (a fresh directory under /workspace/tmp unless given, refused if noexec), named in its
#      .disk_work and removed with it; window.py counts it as host memory against WIN_MEM_GB. In RAM the node's
#      MemAvailable is read before staging and every MEM_POLL_S during it: under MEM_FLOOR_GB the job stages nothing more,
#      removes FLOCK_WORK and DISK_WORK and exits 5; a job that proves removes both at any exit, and STEPS=stage keeps them
#      once window.py stage has ended. Under
#      STEPS=stage the job ends here, FLOCK_WORK kept and, if every statement staged, .staged naming the commit
#   3. (prove) window.py prove: one group per card of CARDS (one group on the CPU), each group's `gpu-lease 1` taken only among
#      CARDS (GPU_LEASE_ALLOWED_FILE) and ending by KILL_AT less 3 minutes, the whole step under `timeout` to KILL_AT less 90 s;
#      every session kept in out/kept/NAME; FLOCK_WORK removed at its end
# env: UNITS (a path, or a name sent with the run) DEV (gpu|cpu) CARDS (gpu, to prove) SM (gpu) NOT_BEFORE KILL_AT; STEPS;
#      FLOCK_WORK (under /workspace/tmp: STEPS=stage's choice, STEPS=prove's input); RAM_WORK=1 DISK_WORK (a FLOCK_WORK in
#      /dev/shm) MEM_FLOOR_GB=500 MEM_POLL_S=5 MEMINFO=/proc/meminfo; MIN_LEFT_S=1800 WORK_ROOT=/workspace/tmp
#      WARM=1 RUNS=4 VCPUS=2 LEASE_MIN=60 WIN_MEM_GB=64 WIN_DISK_GB=40 WIN_FREE_GB=20 WIN_DEV_GIB=135
set -uo pipefail
IN=$RESEARCH_RUN_DIR/inputs; OUT=$RESEARCH_RUN_DIR/out; mkdir -p "$OUT"; L=$OUT/steps.txt
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$L"; }
UNITS=${UNITS:?the units file: a path, or a name sent with the run}; [ -f "$UNITS" ] || UNITS=$IN/$UNITS
DEV=${DEV:-gpu}; KILL_AT=${KILL_AT:?UTC epoch seconds the job is stopped by}; STEPS=${STEPS:-stage prove}
case $DEV in gpu | cpu) ;; *) say "DEV is gpu or cpu, not $DEV"; exit 2 ;; esac
case $STEPS in "stage prove" | stage | prove) ;; *) say "STEPS is \"stage prove\", stage or prove, not $STEPS"; exit 2 ;; esac
has() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
export PYTHONDONTWRITEBYTECODE=1 PATH=$HOME/.elan/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH
NCPU=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))')
FW=; DROP=
drop() { local ex; [ -n "$FW" ] || return 0; ex=$(cat "$FW/.disk_work" 2>/dev/null); [ -z "$ex" ] || rm -rf "$ex"; rm -rf "$FW"; }
S=$(mktemp -d); trap 'rm -rf "$S"; [ -z "$DROP" ] || drop' EXIT; trap 'exit 143' TERM INT
avail() { awk '/^MemAvailable:/ { print int($2 / 1048576) }' "${MEMINFO:-/proc/meminfo}"; }
FLOOR=${MEM_FLOOR_GB:-500}
TREE=$(git rev-parse HEAD 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}")
say "tree $TREE, host $(hostname), $NCPU cpus, device $DEV${CARDS:+, cards $CARDS}, steps $STEPS"

# 0. the start
now=$(date +%s)
if [ -n "${NOT_BEFORE:-}" ] && [ "$now" -lt "$NOT_BEFORE" ]; then
  say "sleeping $((NOT_BEFORE - now)) s, holding nothing"; sleep $((NOT_BEFORE - now))
fi
left=$((KILL_AT - $(date +%s)))
[ "$left" -ge "${MIN_LEFT_S:-1800}" ] || { say "REFUSED: $left s to KILL_AT, under MIN_LEFT_S=${MIN_LEFT_S:-1800}: nothing started"; exit 9; }
if ! has stage; then
  FW=${FLOCK_WORK:?the FLOCK_WORK a STEPS=stage job of this commit kept}; st=$(head -1 "$FW/.staged" 2>/dev/null)
  [ "${st%% *}" = "$TREE" ] || { say "REFUSED: FLOCK_WORK $FW was not staged by this commit: its .staged reads '${st:-nothing}': nothing started"; exit 9; }
  say "FLOCK_WORK $FW staged by this commit: $st"
  case $FW in /dev/shm*) DROP=1 ;; esac
fi
ngroups=1
if [ "$DEV" = gpu ] && has prove; then
  CARDS=${CARDS:?the cards, comma-separated indices}; SM=${SM:?the cards compute capability}
  command -v gpu-lease > /dev/null || { ln -s "$PWD/tools/research/src/research/pods/sh/gpu_lease.sh" "$S/gpu-lease"; export PATH=$S:$PATH; }
  echo "$CARDS" > "$S/cards"; export GPU_LEASE_ALLOWED_FILE=$S/cards
  st=$(gpu-lease status 2>&1); echo "$st" > "$OUT/lease-at-start.txt"
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader 2>&1); echo "$apps" > "$OUT/cards-at-start.txt"
  for c in ${CARDS//,/ }; do
    u=$(nvidia-smi -i "$c" --query-gpu=uuid --format=csv,noheader 2>/dev/null) || { say "REFUSED: no card $c: nothing started"; exit 9; }
    grep -q "^$c free" <<< "$st" || { say "REFUSED: card $c's lease is not free: $(grep "^$c " <<< "$st"): nothing started"; exit 9; }
    ! grep -q "$u" <<< "$apps" || { say "REFUSED: card $c holds a compute process: $(grep "$u" <<< "$apps"): nothing started"; exit 9; }
  done
  ngroups=$(tr ',' '\n' <<< "$CARDS" | grep -c .)
  say "start check passed: cards $CARDS free in gpu-lease status and holding no compute process (out/lease-at-start.txt, out/cards-at-start.txt)"
elif [ "$DEV" = gpu ]; then
  export CUDA_VISIBLE_DEVICES=; say "no card: STEPS=stage builds and stages on the CPU with CUDA_VISIBLE_DEVICES empty"
fi
export WIN_DEVICE=$DEV WIN_CPUS=$NCPU WIN_MEM_GB=${WIN_MEM_GB:-64} ${SM:+SM=$SM}

if has stage; then
  # 1. flock-read-tmpl
  command -v lake > /dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain none >> "$OUT/lean-build.log" 2>&1
  t=$(date +%s)
  (cd verity/protocols/proofs/flock/verifier && lake build flock-read-tmpl) >> "$OUT/lean-build.log" 2>&1 \
    || { tail -20 "$OUT/lean-build.log" | tee -a "$L"; say "flock-read-tmpl BUILD FAILED: a table read can't stage"; exit 4; }
  say "flock-read-tmpl built in $(( $(date +%s) - t )) s"

  # 2. stage, on disk, or under RAM_WORK=1 the statements in RAM and the executables on disk
  FW=${FLOCK_WORK:-}; WR=${WORK_ROOT:-/workspace/tmp}; what="WORK_ROOT $WR"; [ -z "$FW" ] || what="FLOCK_WORK $FW"
  ram=; case ${FW:-$WR} in /dev/shm*) ram=1 ;; esac
  if [ -n "$ram" ] && [ "${RAM_WORK:-}" != 1 ]; then
    say "REFUSED: $what is in RAM; flock-tmpl-unit is built and run under FLOCK_WORK, on disk (RAM_WORK=1 puts it on DISK_WORK)"; exit 4
  fi
  if [ -n "$FW" ]; then mkdir -p "$FW" || { say "no FLOCK_WORK $FW"; exit 4; }
  else mkdir -p "$WR" && FW=$(mktemp -d "$WR/stage-prove.XXXXXX") || { say "no FLOCK_WORK under $WR"; exit 4; }; fi
  EX=$FW
  if [ -n "$ram" ]; then
    EX=${DISK_WORK:-}; [ -n "$EX" ] || EX=$(mkdir -p /workspace/tmp && mktemp -d /workspace/tmp/stage-prove-exec.XXXXXX) \
      || { say "no DISK_WORK under /workspace/tmp"; exit 4; }
    case $EX in /dev/shm*) say "REFUSED: DISK_WORK $EX is in RAM"; exit 4 ;; esac
    for d in py flock-gpu-link flock-circuit tmpl-unit; do
      mkdir -p "$EX/$d" && { [ -L "$FW/$d" ] || ln -s "$EX/$d" "$FW/$d"; } || { say "no $d of DISK_WORK $EX linked into $FW"; exit 4; }
    done
    echo "$EX" > "$FW/.disk_work"; DROP=1
    a=$(avail); [ "${a:-0}" -ge "$FLOOR" ] \
      || { say "REFUSED: the node's MemAvailable is ${a:-unread} GB, under MEM_FLOOR_GB=$FLOOR: nothing staged"; exit 5; }
  fi
  if findmnt -no OPTIONS -T "$EX" | tr ',' '\n' | grep -qx noexec; then
    exw=FLOCK_WORK; [ -z "$ram" ] || exw=DISK_WORK
    say "REFUSED: $exw $EX is on a noexec mount, where flock-tmpl-unit can't run"; has prove && rm -rf "$FW"; exit 4
  fi
  [ -z "$ram" ] || say "RAM_WORK=1: FLOCK_WORK $FW holds the statements in RAM; its py, flock-gpu-link, flock-circuit and tmpl-unit are DISK_WORK $EX's"
  say "flock-tmpl-unit is built and run under $EX, on $(findmnt -no TARGET -T "$EX") (exec allowed)"
  export FLOCK_WORK=$FW WIN_DISK_GB=${WIN_DISK_GB:-40} WIN_FREE_GB=${WIN_FREE_GB:-20} \
    WIN_UNTIL=$(date -u -d @$((KILL_AT - 1200)) +%Y-%m-%dT%H:%MZ)
  n=$(grep -vc '^#' "$UNITS")
  say "window.py stage $n lines of $(basename "$UNITS"), until $WIN_UNTIL"
  python3 benchmarks/served_units/window.py stage "$UNITS" & wp=$!; mw=
  if [ -n "$ram" ]; then
    (while kill -0 "$wp" 2> /dev/null; do
      a=$(avail); [ "${a:-0}" -ge "$FLOOR" ] || { echo "${a:-unread}" > "$S/floor"; kill -TERM "$wp"; exit; }
      sleep "${MEM_POLL_S:-5}"
    done) & mw=$!
  fi
  wait "$wp"; rc=$?; [ -z "$mw" ] || { kill "$mw" 2> /dev/null; wait "$mw" 2> /dev/null; }
  if [ -s "$S/floor" ]; then
    say "STOPPED: the node's MemAvailable fell to $(cat "$S/floor") GB, under MEM_FLOOR_GB=$FLOOR: staging stopped, FLOCK_WORK and DISK_WORK removed"
    exit 5
  fi
  staged=$(ls -d "$FW"/data/*/.done 2>/dev/null | wc -l)
  say "window.py stage exit $rc; staged: $staged; placed units: $(ls "$FW"/data/*/unit.placed 2>/dev/null | wc -l)"
  ls "$FW"/tmpl-unit/lean-*/flock-tmpl-unit 2>/dev/null | while read -r b; do say "flock-tmpl-unit $b, sha256 $(sha256sum "$b" | cut -c1-16)"; done
  if ! has prove; then
    DROP=
    [ "$rc" = 0 ] && [ "$staged" -ge "$n" ] \
      && echo "$TREE $(date -u +%Y-%m-%dT%H:%M:%SZ) units-sha256 $(sha256sum < "$UNITS" | cut -c1-16) staged $staged" > "$FW/.staged"
    say "STEPS=stage: FLOCK_WORK $FW kept ($(du -sh "$FW" 2>/dev/null | cut -f1)); .staged: $(cat "$FW/.staged" 2>/dev/null || echo 'none, so no STEPS=prove takes it')"
    exit $rc
  fi
fi

# 3. prove
export FLOCK_WORK=$FW
left=$(( KILL_AT - 90 - $(date +%s) )); lease=$(( (KILL_AT - $(date +%s)) / 60 - 3 ))
[ "$left" -gt 300 ] || { say "REFUSED: $left s to KILL_AT: nothing proved"; drop; exit 9; }
export WIN_GPUS=$ngroups WIN_DEV_GIB=${WIN_DEV_GIB:-135} WIN_KEEP=0 WARM=${WARM:-1} RUNS=${RUNS:-4} VCPUS=${VCPUS:-2} \
  LEASE_MIN=$(( ${LEASE_MIN:-60} < lease ? ${LEASE_MIN:-60} : lease ))
say "window.py prove: $ngroups groups, lease $LEASE_MIN min, stopped in $left s"
timeout --signal=TERM --kill-after=60 "$left" python3 benchmarks/served_units/window.py prove "$UNITS"; rc=$?
say "window.py prove exit $rc"
[ "$DEV" = gpu ] && nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader > "$OUT/cards-at-end.txt" 2>&1
fin=$(awk -F'\t' 'NR > 1 && $4 == 1' "$OUT/seconds.tsv" 2>/dev/null | wc -l)
say "statements: $(awk 'NR > 1' "$OUT/seconds.tsv" 2>/dev/null | wc -l) started, $fin finished with exit 0; kept: $(ls "$OUT/kept" 2>/dev/null | wc -l)"
drop
exit $rc
