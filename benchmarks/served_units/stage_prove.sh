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
#   3. (prove) on DEV=gpu, window.py route first: a statement past the GPU prover's nonzero bound goes to the CPU prover
#      (out/gpu.units, out/cpu.units). window.py prove: one group per card of CARDS (one group on the CPU), each group's
#      `gpu-lease 1` taken on its own card alone and on that card's NUMA node's CPUs (numa.py pin: out/topology.json, the
#      same count a card), ending by KILL_AT less 3 minutes, the whole step under `timeout` to KILL_AT less 90 s; every
#      session kept in out/kept/NAME. numa.py check then reads each group's card and CPUs into out/pinning.tsv: a group
#      off its card's node is NUMA OFF NODE in steps.txt and the job exits 8. Then the CPU prover's statements, one
#      group on the job's cores with no card, into out/cpu/out (its kept/ there); under CPU_PROVE=0 they are only listed
#      (out/cpu.units), for CPU nodes to prove
#   4. (prove, LEAN=1, the default) Flock's Lean verifier on each proved statement's kept sessions, in this job and on
#      the statement FLOCK_WORK holds, before it is removed (lean_check.sh): the last session first, then, while each
#      check takes under LEAN_ALL_S (120) seconds, every other kept one. out/lean.tsv a row a session checked,
#      out/lean-statements.tsv a row a statement (sessions kept and checked, the verdict, the most seconds and GiB of one
#      check), out/lean/NAME/SESSION/. A statement with a session it doesn't accept, or a proved one it can't check, is
#      LEAN REFUSED in steps.txt and the job exits 7; nothing is retried. FLOCK_WORK removed at the job's end
# env: UNITS (a path, or a name sent with the run) DEV (gpu|cpu) CARDS (gpu, to prove) SM (gpu) NOT_BEFORE KILL_AT; STEPS;
#      FLOCK_WORK (under /workspace/tmp: STEPS=stage's choice, STEPS=prove's input); RAM_WORK=1 DISK_WORK (a FLOCK_WORK in
#      /dev/shm) MEM_FLOOR_GB=500 MEM_POLL_S=5 MEMINFO=/proc/meminfo; MIN_LEFT_S=1800 WORK_ROOT=/workspace/tmp
#      WARM=1 RUNS=4 VCPUS=2 LEASE_MIN=60 WIN_MEM_GB=64 WIN_DISK_GB=40 WIN_FREE_GB=20 WIN_DEV_GIB=135; LEAN=1
#      LEAN_ALL_S=120 CPU_PROVE=1
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
  nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv,noheader > "$S/gpus.csv" 2>&1
  pin=$(python3 benchmarks/served_units/numa.py pin "$CARDS" "$S/gpus.csv" "$OUT/topology.json" 2>&1) \
    || { say "REFUSED: $pin: nothing started"; exit 9; }
  export WIN_CARD_CPUS=$pin
  say "NUMA: each card's prover and verifier on its own node's CPUs, card=CPUs: $pin (out/topology.json)"
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
LEAN=${LEAN:-1}; DROP=1   # FLOCK_WORK outlives each window.py prove, for the CPU prover's and the Lean verifier's steps
export WIN_GPUS=$ngroups WIN_DEV_GIB=${WIN_DEV_GIB:-135} WIN_KEEP=1 WARM=${WARM:-1} RUNS=${RUNS:-4} VCPUS=${VCPUS:-2} \
  LEASE_MIN=$(( ${LEASE_MIN:-60} < lease ? ${LEASE_MIN:-60} : lease ))
GU=$UNITS; CU=$S/none; : > "$CU"
if [ "$DEV" = gpu ]; then
  GU=$OUT/gpu.units CU=$OUT/cpu.units
  python3 benchmarks/served_units/window.py route "$UNITS" "$GU" "$CU" >> "$OUT/route.log" 2>&1 \
    || { say "window.py route FAILED: $(tail -1 "$OUT/route.log"): nothing proved"; drop; exit 9; }
  say "route: $(grep -c . "$GU") statements to the cards, $(grep -c . "$CU") past the GPU prover's nonzero bound to the CPU prover (out/gpu.units, out/cpu.units)"
fi
rc=0
if [ -s "$GU" ]; then
  say "window.py prove: $ngroups groups, lease $LEASE_MIN min, stopped in $left s"
  timeout --signal=TERM --kill-after=60 "$left" python3 benchmarks/served_units/window.py prove "$GU"; rc=$?
  say "window.py prove exit $rc"
  if [ "$DEV" = gpu ]; then
    if off=$(python3 benchmarks/served_units/numa.py check "$OUT/topology.json" "$OUT" 2>&1); then
      say "NUMA check passed: every group's prover and verifier on its card's node (out/pinning.tsv)"
    else
      say "NUMA OFF NODE: ${off//$'\n'/; }: its times are not on its card's node (out/pinning.tsv)"; [ "$rc" != 0 ] || rc=8
    fi
  fi
  [ "$DEV" = gpu ] && nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader > "$OUT/cards-at-end.txt" 2>&1
  fin=$(awk -F'\t' 'NR > 1 && $4 == 1' "$OUT/seconds.tsv" 2>/dev/null | wc -l)
  say "statements: $(awk 'NR > 1' "$OUT/seconds.tsv" 2>/dev/null | wc -l) started, $fin finished with exit 0; kept: $(ls "$OUT/kept" 2>/dev/null | wc -l)"
fi
if [ -s "$CU" ] && [ "${CPU_PROVE:-1}" = 0 ]; then
  say "CPU_PROVE=0: $(grep -c . "$CU") statements past the GPU prover's bound not proved here: they are the CPU nodes' (out/cpu.units)"
elif [ -s "$CU" ]; then
  left=$(( KILL_AT - 90 - $(date +%s) ))
  say "window.py prove on the CPU prover: $(grep -c . "$CU") statements, one group on $NCPU cores, no card, stopped in $left s (out/cpu/out)"
  RESEARCH_RUN_DIR=$OUT/cpu WIN_DEVICE=cpu WIN_GPUS=1 WIN_CARD_CPUS= CUDA_VISIBLE_DEVICES= \
    timeout --signal=TERM --kill-after=60 "$left" python3 benchmarks/served_units/window.py prove "$CU"; c=$?
  say "window.py prove on the CPU prover exit $c"; [ "$rc" != 0 ] || rc=$c
fi

# 4. the Lean verifier on each proved statement's kept sessions, before FLOCK_WORK is removed: the last first, then, while
#    one check took under LEAN_ALL_S seconds, every other one; a statement's row in out/lean-statements.tsv
if [ "$LEAN" = 1 ]; then
  proved=(); for K in "$OUT"/kept/*/ "$OUT"/cpu/out/kept/*/; do
    [ -d "$K" ] || continue; K=${K%/}; x=$(cat "$K/exit" 2>/dev/null)
    if [ "$x" = 0 ]; then proved+=("$K"); else say "LEAN: $(basename "$K"): its prover exited ${x:-unread}: no proof to check"; fi
  done
  ok=0; bad=0
  if [ ${#proved[@]} -gt 0 ]; then
    . benchmarks/served_units/lean_check.sh
    LEAN_PY=(uv run --no-sync python); LEAN_TMP=$S; V=verity/protocols/proofs/flock/verifier
    mkdir -p "$OUT/lean"; printf 'name\tsession\taccepted\tlabel\twall_s\tpeak_gib\tcircuit_bytes\n' > "$OUT/lean.tsv"
    printf 'name\tkept\tchecked\taccepted\tverdict\tmax_wall_s\tmax_peak_gib\tcircuit_bytes\n' > "$OUT/lean-statements.tsv"
    exe=$(lean_exes "$V" 2>> "$OUT/lean-build.log" | cut -d' ' -f1)
    if [ -z "$exe" ] || ! (cd "$V" && lake build "$exe") >> "$OUT/lean-build.log" 2>&1; then
      say "LEAN REFUSED: the verifier ${exe:-named in $V/lakefile.toml} did not build: none of ${#proved[@]} proved statements checked (out/lean-build.log)"
      bad=${#proved[@]}; proved=()
    else
      EXE=$PWD/$V/.lake/build/bin/$exe; say "the Lean verifier $exe built, sha256 $(sha256sum "$EXE" | cut -c1-16)"
    fi
    for K in "${proved[@]}"; do
      nm=$(basename "$K"); mapfile -t sds < <(for d in "$K"/sessions/*/; do [ -s "$d/session.json" ] && echo "${d%/}"; done | sort -r)
      sd=${sds[0]:-}
      ds=("$FW"/data/"$nm"-*/); D=${ds[0]%/}
      if [ -z "$sd" ] || [ ${#ds[@]} != 1 ] || [ ! -s "$D/circuit.txt" ]; then
        say "LEAN REFUSED: $nm: ${sd:+its statement is not one directory of FLOCK_WORK}${sd:-no kept session}: not checked"; bad=$((bad + 1)); continue
      fi
      lean_word "$D" > "$OUT/lean/$nm.word.txt" 2>&1 \
        || { say "LEAN REFUSED: $nm: program-word.json not written: $(tail -1 "$OUT/lean/$nm.word.txt" | cut -c1-200)"; bad=$((bad + 1)); continue; }
      n_ok=0; n_ck=0; refused=; wmax=0; mmax=0
      for sd in "${sds[@]}"; do
        LEAN_BENCH=(timeout --signal=TERM --kill-after=30 "$(( KILL_AT - 60 - $(date +%s) ))" uv run --no-sync verity bench)
        lean_check "$nm" "$sd" "$D" "$K" "$OUT/lean/$nm/$(basename "$sd")" "$EXE" "$OUT/lean.tsv"; a=$?; n_ck=$((n_ck + 1))
        row=$(tail -1 "$OUT/lean.tsv"); w=$(cut -f5 <<< "$row"); m=$(cut -f6 <<< "$row")
        wmax=$(python3 -c 'import sys; print(max(float(x or 0) for x in sys.argv[1:]))' "$wmax" "$w")
        mmax=$(python3 -c 'import sys; print(max(float(x or 0) for x in sys.argv[1:]))' "$mmax" "$m")
        [ "$a" = 0 ] || { refused="$(basename "$sd"): $(cut -f3-4 <<< "$row")"; break; }
        n_ok=$((n_ok + 1))
        python3 -c 'import sys; sys.exit(float(sys.argv[1] or 1e9) >= float(sys.argv[2]))' "$w" "${LEAN_ALL_S:-120}" || break
      done
      cb=$(stat -c %s "$D/circuit.txt")
      if [ -z "$refused" ]; then
        ok=$((ok + 1)); v=accepted; say "LEAN accepted: $nm: $n_ck of ${#sds[@]} kept sessions checked, each accepted; at most $wmax s, $mmax GiB"
      else
        bad=$((bad + 1)); v=refused; say "LEAN REFUSED: $nm $refused ($n_ok accepted before it, $n_ck of ${#sds[@]} checked)"
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$nm" "${#sds[@]}" "$n_ck" "$n_ok" "$v" "$wmax" "$mmax" "$cb" >> "$OUT/lean-statements.tsv"
    done
  fi
  say "the Lean verifier: $ok proved statements accepted, $bad refused or not checked (out/lean.tsv)"
  [ "$bad" = 0 ] || [ "$rc" != 0 ] || rc=7
fi
drop
exit $rc
