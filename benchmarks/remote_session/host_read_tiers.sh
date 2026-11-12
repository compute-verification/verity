#!/usr/bin/env bash
# The host tier at several sizes on a whole node, in one timed lease: host_read.py (its rows, each gated and then
# timed) once for each size of SIZES, in that order, every size at the same step, beside the same store and on the same
# cards, so what differs from one size's record to the next is the tier's size alone. Three stages, the first argument:
#
#   launch  from a clean, pushed checkout (cwd = the tree): sends `job` to MACHINE as a `research run` of the checkout's
#           commit in a slot of that machine (SLOTS, taken as BY, waited for at most SLOT_WAIT_S; empty SLOTS: no slot).
#           It sends nothing (exit 2) more than five minutes before NOT_BEFORE, with too little time before END for the
#           build and one size, or from a checkout that is not clean and pushed. DRY=1 prints the command, sends
#           nothing, and runs `job` with DRY=1 here.
#   job     (the default; cwd = the tree, `research run --cwd source`) holding no card: it waits for NOT_BEFORE if that
#           is at most ten minutes away, then host_read.sh's build and its gate on the kernels (BUILD_ONLY=1, for ARCH
#           with CUDA_HOME's nvcc against BASE_REV), then one lease of the machine's own gpu-lease (GPU_LEASE) on GPUS
#           cards: waited for, so a card another lease holds does not end the run; timed, so the machine is kept quiet
#           under it; of at most LEASE_MIN minutes, and asked to end GRANT_S seconds before END (later, up to END, where
#           one size needs it): a request that waits that long for its grant still asks for nothing past END, which a
#           machine that grants only what ends by its own deadline needs. Whatever waits or runs is ended at END.
#   lease   inside the lease: the lease holds GPUS distinct cards (exit 7); then each size in turn in its own process,
#           pinned to DRIVER_CPUS, its record in out/host-SIZE/host_read.json (it states the step and the tier's
#           size). A size starts on clear cards, no compute process and at most CLEAR_MIB MiB in use on any, waited for
#           up to CLEAR_WAIT_S seconds (exit 11, the sizes left not run). The time left, to a minute before the
#           lease's end or END, is shared equally among the sizes still to run; a size's process starts no step in
#           the last TAIL_S seconds of its share and is ended at it. A size whose share would be under SIZE_MIN_S
#           seconds is not run, the last of SIZES first.
#
# A card the driver holds below its clock makes every card's pass longer, so the lease keeps what tells that from the
# code under test. out/cards.start.txt, cards.after-SIZE.txt and cards.end.txt are the lease's cards at those moments:
# temperature, power, clock, the reasons the driver gives for holding the clock down and, where the driver counts
# them, how long it has held each card's clocks down by reason since it loaded. A size's line names the cards whose
# count grew during it, and one line before the verdict those over the whole lease. out/cards.during.csv is each
# card's temperature, clock, power and use every SAMPLE_MS ms (0: not kept), from one nvidia-smi on SAMPLE_CPUS, or on
# no core of its own where that pin does not run.
#
# NOT_BEFORE and END (`date -d` syntax, UTC) are the hours: nothing starts before the one, and no lease is held at
# the other. NAMES are host_read.py's rows after `pass` (its own default without it), GIB the store on the cards,
# CHUNK_BLOCKS the tier's step in blocks, ARGS more arguments of host_read.py. DRY=1 builds nothing and touches no
# card: it prints the lease's line and each size's command, on a clock that jumps a size's share at each. NOW (epoch
# seconds) stands in for the clock, for a dry run of other hours and for the tests.
#
# Exit: 0 every size ran and passed; 10 those that ran passed and some were not run for time; otherwise the first
# failing size's own code (6 a row's gate or equality failed or a control was not refused, 124 ended at its share);
# 9 outside the hours, nothing started; 8 the tree is not EXPECT_COMMIT; 3, 4, 5 host_read.sh's build; 13 the pin does
# not run here; 75 no lease by END; 143 ended by a signal. out/tiers.txt is the run's account, its last line the verdict.
#
#   END=2031-05-01T23:30Z NOT_BEFORE=2031-05-01T22:15Z MACHINE=node BY=lane SIZES='8 12 16' NAMES=ahead2 \
#     bash benchmarks/remote_session/host_read_tiers.sh launch
set -uo pipefail
STAGE=${1:-job}
TREE=$PWD RS=benchmarks/remote_session
SELF=$TREE/$RS/host_read_tiers.sh
now() { echo "${NOW:-$(date +%s)}"; }
utc() { date -u -d "@$1" +%FT%H:%M:%SZ; }

set -a
SIZES=${SIZES:-8 12 16} GIB=${GIB:-700} CHUNK_BLOCKS=${CHUNK_BLOCKS:-65536} GPUS=${GPUS:-8}
DRIVER_CPUS=${DRIVER_CPUS:-48-87} LEASE_MIN=${LEASE_MIN:-30} GRANT_S=${GRANT_S:-300} MEM_GB=${MEM_GB:-300}
CLEAR_MIB=${CLEAR_MIB:-1024} CLEAR_WAIT_S=${CLEAR_WAIT_S:-120} SIZE_MIN_S=${SIZE_MIN_S:-240} TAIL_S=${TAIL_S:-90}
SAMPLE_MS=${SAMPLE_MS:-2000} SAMPLE_CPUS=${SAMPLE_CPUS:-88-95}
ARCH=${ARCH:-120a} CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
PY=${PY:-/workspace/pouw/fill-out/pous/work/pous-bench/gpu312-torch2.14.0-cu130/bin/python}
GPU_LEASE=${GPU_LEASE:-/workspace/pouw/infra/bin/gpu-lease} SMI=${SMI:-nvidia-smi}
END=${END:?the time no lease is held at (date -d syntax, UTC)}
set +a
BUILD_S=${BUILD_S:-240}   # what the build is given before the lease is asked for
END_S=$(date -u -d "$END" +%s) && FROM_S=$(date -u -d "${NOT_BEFORE:-@0}" +%s) || exit 2
[ "$SIZE_MIN_S" -gt "$TAIL_S" ] || { echo "SIZE_MIN_S ($SIZE_MIN_S) is not over TAIL_S ($TAIL_S)" >&2; exit 2; }

if [ "$STAGE" = launch ]; then
  refuse() { echo "REFUSED: $*" >&2; exit 2; }
  t=$(now)
  [ "$t" -ge $((FROM_S - 300)) ] || refuse "$(utc "$t") is more than 5 min before NOT_BEFORE ($(utc "$FROM_S"))"
  [ $((END_S - t)) -ge $((BUILD_S + SIZE_MIN_S + 60)) ] || refuse "$(((END_S - t) / 60)) min from $(utc "$t") to END" \
    "($(utc "$END_S")): under the $(((BUILD_S + SIZE_MIN_S + 60) / 60)) min the build and one size need"
  [ -z "$(git status --short)" ] || refuse "the checkout is not clean"
  EXPECT_COMMIT=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)
  [ "$(git rev-parse -q --verify "origin/$branch")" = "$EXPECT_COMMIT" ] || refuse "$EXPECT_COMMIT is not origin/$branch"
  BASE_REV=${BASE_REV:-$(git merge-base HEAD origin/main)} || refuse "no BASE_REV, and no merge base with origin/main"
  E=()
  for v in END NOT_BEFORE EXPECT_COMMIT BASE_REV SIZES NAMES GIB CHUNK_BLOCKS GPUS DRIVER_CPUS LEASE_MIN GRANT_S MEM_GB \
    CLEAR_MIB CLEAR_WAIT_S SIZE_MIN_S TAIL_S SAMPLE_MS SAMPLE_CPUS ARCH CUDA_HOME PY GPU_LEASE SMI ARGS GPU_LEASE_WHO; do
    [ -z "${!v+x}" ] || { E+=(--env "$v=${!v}"); export "${v?}"; }
  done
  run_s=$((END_S - t + 300))   # from taking the slot: the run outlives its lease by the end's records and its upload
  SLOTS=${SLOTS-/workspace/research/locks/gpu/slots} SLOT_WAIT_S=${SLOT_WAIT_S:-300}
  CMD=(uv run research run --on "${MACHINE:?the machine}" --project verity --source . --cwd source --no-sampler
    --timeout "$((SLOT_WAIT_S + run_s))" "${E[@]}" --)
  [ -z "$SLOTS" ] || CMD+=(env "CHECK_SLOTS=$SLOTS" python3 tools/verity/check/slot.py --by "${BY:?who holds the slot}"
    --wait-max "$SLOT_WAIT_S" --run-max "$run_s" --)
  CMD+=(bash "$RS/host_read_tiers.sh")
  [ -n "${DRY:-}" ] || exec "${CMD[@]}"
  echo "DRY launch at $(utc "$t"), nothing is sent: $(printf '%q ' "${CMD[@]}")"
  exec bash "$SELF" job
fi

O=${OUT:-${RESEARCH_RUN_DIR:-\$RESEARCH_RUN_DIR}/out}
if [ -n "${DRY:-}" ]; then
  say() { echo "DRY $(date -u -d "@$(now)" +%H:%M:%SZ) $*"; }
else
  [ -n "${OUT:-${RESEARCH_RUN_DIR:-}}" ] || { echo "no RESEARCH_RUN_DIR (or OUT)" >&2; exit 2; }
  mkdir -p "$O"
  say() { echo "$(date -u -d "@$(now)" +%H:%M:%SZ) $*" | tee -a "$O/tiers.txt"; }
fi
export OUT=$O
verdict() {  # verdict CODE WORD TEXT...: the stage's last line, kept for `job` to end on, and its exit
  local line="VERDICT host-read-tiers $2 (exit $1): ${*:3}"
  [ -n "${DRY:-}" ] || echo "$line" > "$O/verdict.txt"
  say "$line"
  exit "$1"
}
hours() {  # hours WHERE SECONDS: exit 9 before NOT_BEFORE, or with fewer than SECONDS left before END
  local t; t=$(now)
  [ "$t" -ge "$FROM_S" ] || verdict 9 refused "$1: before NOT_BEFORE ($(utc "$FROM_S")); nothing started"
  [ $((END_S - t)) -ge "$2" ] || verdict 9 refused "$1: $(((END_S - t) / 60)) min to END ($(utc "$END_S")), under the" \
    "$(($2 / 60)) min needed; nothing started"
}
ended=
watch() {  # watch CMD...: CMD, ended with this stage when a signal ends the stage; `ended` then says so
  "$@" &
  local child=$! r
  trap 'ended=1; kill -TERM "$child" 2> /dev/null' TERM INT HUP
  wait "$child"; r=$?
  [ -z "$ended" ] || wait "$child"
  trap - TERM INT HUP
  return "$r"
}

if [ "$STAGE" = job ]; then
  COMMIT=${RESEARCH_SOURCE_SHA:-$(git -C "$TREE" rev-parse HEAD 2> /dev/null || echo unknown)}
  say "host tiers of $SIZES GiB in chunks of $CHUNK_BLOCKS blocks, beside $GIB GiB on $GPUS cards; rows: pass" \
    "${NAMES:-and the default of host_read.py}; tree $COMMIT on $(hostname); not before $(utc "$FROM_S"), no lease at $(utc "$END_S")"
  t=$(now)
  if [ "$t" -lt "$FROM_S" ] && [ $((FROM_S - t)) -le 600 ] && [ -z "${DRY:-}${NOW:-}" ]; then
    say "waiting $((FROM_S - t)) s for NOT_BEFORE, holding no card"
    sleep $((FROM_S - t))
  fi
  hours "before the build" $((BUILD_S + SIZE_MIN_S))
  [ -z "${EXPECT_COMMIT:-}" ] || [ "$COMMIT" = "$EXPECT_COMMIT" ] \
    || verdict 8 refused "the tree is at $COMMIT, not EXPECT_COMMIT ($EXPECT_COMMIT); nothing started"
  if [ -n "${DRY:-}" ]; then
    export B=/tmp/host-read-tiers.DRY
    say "not done here: the build (BUILD_ONLY=1 BUILD_DIR=$B ARCH=$ARCH CUDA_HOME=$CUDA_HOME" \
      "${BASE_REV:+BASE_REV=$BASE_REV }bash $RS/host_read.sh), and that $PY, $GPU_LEASE and taskset -c $DRIVER_CPUS run"
  else
    for f in "$PY" "$GPU_LEASE" "$CUDA_HOME/bin/nvcc"; do
      [ -x "$f" ] || verdict 3 refused "no $f on this machine; nothing started"
    done
    pin=$(taskset -c "$DRIVER_CPUS" grep Cpus_allowed_list /proc/self/status 2>&1) \
      || verdict 13 refused "the driver's pin does not run here (taskset -c $DRIVER_CPUS): $(tail -n 1 <<< "$pin")"
    B=$(mktemp -d /tmp/host-read-tiers.XXXXXX); trap 'rm -rf "$B"' EXIT; export B
    env BUILD_ONLY=1 BUILD_DIR="$B" ARCH="$ARCH" CUDA_HOME="$CUDA_HOME" ${BASE_REV:+BASE_REV="$BASE_REV"} \
      bash "$TREE/$RS/host_read.sh" || verdict $? failed "host_read.sh's build or its gate on the kernels (out/host_read.txt)"
    say "built: $(awk '{ print substr($1, 1, 12), $2 }' "$O/libs.sha256" | sed "s|$B/||" | paste -sd' '); the driver's" \
      "cores: $(tr -d '\t ' <<< "${pin#*:}")"
  fi
  one=$(((SIZE_MIN_S + 119) / 60))   # the whole minutes one size needs: its SIZE_MIN_S and the lease's last minute
  hours "before the lease" $((one * 60))
  t=$(now); left=$((END_S - t)); m=$(((left - GRANT_S) / 60))
  [ "$m" -ge "$one" ] || m=$one
  [ "$m" -le "$LEASE_MIN" ] || m=$LEASE_MIN
  export LEASE_MIN=$m
  LEASE=("$GPU_LEASE" "$GPUS" --wait --timed --max-min "$m" --mem-gb "$MEM_GB" -- bash "$SELF" lease)
  say "the lease, $GPUS cards for at most $m min (granted within $((left - m * 60)) s, it asks for nothing past END), as" \
    "${GPU_LEASE_WHO:-the name gpu-lease gives this user}: ${LEASE[*]}"
  [ -z "${DRY:-}" ] || exec bash "$SELF" lease
  { echo "# $(utc "$t") before the lease"; "$GPU_LEASE" status 2>&1 | cut -c1-220; echo "# loadavg $(cat /proc/loadavg)"
  } > "$O/beside.before.txt"
  watch timeout -k 30 $((END_S - t)) "${LEASE[@]}"
  rc=$?
  if [ ! -e "$O/leased" ]; then
    [ $rc != 124 ] || rc=75
    verdict "$rc" refused "no lease of $GPUS cards (gpu-lease ended with $rc; 75: none by END); nothing started"
  fi
  [ -s "$O/verdict.txt" ] || verdict "$rc" failed "the lease's time, END or a signal ended it (exit $rc); out/host-*/" \
    "hold what was measured"
  tail -n 1 "$O/tiers.txt"
  exit "$rc"
fi

# ---- inside the lease ------------------------------------------------------------------------------------------------
[ "$STAGE" = lease ] || { echo "host_read_tiers.sh [launch|job|lease], not '$STAGE'" >&2; exit 2; }
t=$(now)
stop=$((t + LEASE_MIN * 60)); [ "$stop" -le "$END_S" ] || stop=$END_S
stop=$((stop - 60))   # the last minute is for the processes to leave the cards and for the end's records
on_cards() {  # what a size does not start beside: compute processes on the lease's cards, and cards with memory in use
  local apps used
  apps=$("$SMI" --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader 2>&1) \
    && used=$("$SMI" --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPU_LEASE_UUID" 2>&1) \
    || { echo "nvidia-smi did not answer: $(head -n 1 <<< "${used:-$apps}")"; return; }
  grep -F "$(tr ',' '\n' <<< "$GPU_LEASE_UUID")" <<< "$apps"
  awk -F', *' -v m="$CLEAR_MIB" '$2 + 0 > m { print "card " $1 ": " $2 " MiB in use" }' <<< "$used"
}
clear() {  # waits up to CLEAR_WAIT_S, and no later than `stop`, for the lease's cards to be clear; 1 if they are not
  local found waited=0
  while found=$(on_cards); [ -n "$found" ]; do
    if [ "$waited" -ge "$CLEAR_WAIT_S" ] || [ "$(now)" -ge "$stop" ]; then
      echo "$found" > "$O/cards.not-clear.txt"
      say "the lease's cards are not clear after $waited s (out/cards.not-clear.txt): $(head -n 1 <<< "$found")"
      return 1
    fi
    sleep 5; waited=$((waited + 5))
  done
}
HELD=clocks_event_reasons_counters   # microseconds the driver has held a card's clocks down, by reason, since it loaded
cards() {  # cards TAG: the lease's cards now, into out/cards.TAG.txt; prints the cards with a throttle reason active
  local q=index,uuid,name,memory.used,temperature.gpu,power.draw,clocks.sm reason out line r bad=
  local counted=$HELD.sw_power_cap,$HELD.sw_thermal_slowdown,$HELD.hw_thermal_slowdown,$HELD.hw_power_brake_slowdown
  # the fields' names, by driver: the counters where it has them, and the reasons active now last
  for reason in "$counted,clocks_event_reasons.active" clocks_event_reasons.active clocks_throttle_reasons.active ""; do
    out=$("$SMI" --query-gpu="$q${reason:+,$reason}" --format=csv,noheader,nounits -i "$GPU_LEASE_UUID" 2>&1) && break
  done
  { echo "# $(utc "$(now)") $q${reason:+,$reason}"; echo "$out"; } > "$O/cards.$1.txt"
  [ -n "$reason" ] || { echo unread; return; }
  # 0x4 software power cap, 0x8 hardware slowdown, 0x20 software thermal, 0x40 hardware thermal, 0x80 power brake
  while IFS= read -r line; do
    r=${line##*, }
    [[ $r =~ ^0x[0-9a-fA-F]+$ ]] && ((r & 0xEC)) && bad+=" ${line%%,*}"
  done <<< "$out"
  echo "${bad# }"
}
held_down() {  # held_down A B: the cards whose clocks the driver held down between out/cards.A.txt and out/cards.B.txt,
  # each with the seconds by reason; nothing where no count grew, and nothing where the driver counts none
  awk -F', *' '
    FNR == 1 { for (i = 2; i <= NF; i++) if ($i ~ /_counters\./) { name[i] = $i; sub(/.*\./, "", name[i]) } next }
    NR == FNR { for (i in name) was[$1, i] = $i; next }
    { s = ""
      for (i = 2; i <= NF; i++) if (i in name && $i - was[$1, i] > 0)
        s = s sprintf(" %s %.1f s", name[i], ($i - was[$1, i]) / 1e6)
      if (s != "") all = all sprintf("%scard %s%s", all == "" ? "" : "; ", $1, s) }
    END { print all }' "$O/cards.$1.txt" "$O/cards.$2.txt" 2> /dev/null
}

sampler=
if [ -n "${DRY:-}" ]; then
  say "not done here: that the lease holds $GPUS distinct cards, that they are clear before each size ($CLEAR_MIB MiB," \
    "waited for up to $CLEAR_WAIT_S s), the cards' records at the start, after each size and at the end, and each" \
    "card's temperature and clock every $SAMPLE_MS ms between them"
else
  utc "$t" > "$O/leased"
  held=$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES:-}" | grep . | sort -u | wc -l)
  [ "$held" = "$GPUS" ] && [ "$(tr ',' '\n' <<< "${GPU_LEASE_UUID:-}" | grep . | sort -u | wc -l)" = "$GPUS" ] \
    || verdict 7 refused "the lease holds cards '${CUDA_VISIBLE_DEVICES:-}', not $GPUS distinct ones; nothing started"
  throttled=$(cards start)
  say "leased: cards $CUDA_VISIBLE_DEVICES, the sizes over by $(utc "$stop")${throttled:+; throttle reasons, cards: $throttled}"
  grep -q "$HELD" "$O/cards.start.txt" || say "this driver does not count how long it holds a card's clocks down: a" \
    "card held down under a size shows only in out/cards.during.csv"
  if [ "$SAMPLE_MS" -gt 0 ]; then
    reason=$(head -n 1 "$O/cards.start.txt"); reason=${reason##*,}; [[ $reason == clocks_*_reasons.active ]] || reason=
    pin=(taskset -c "$SAMPLE_CPUS") on="cores $SAMPLE_CPUS"
    "${pin[@]}" true 2> /dev/null || pin=() on="no core of its own (taskset -c $SAMPLE_CPUS does not run here)"
    "${pin[@]}" timeout $((stop + 60 - t)) "$SMI" -i "$GPU_LEASE_UUID" -lms "$SAMPLE_MS" --format=csv,nounits \
      --query-gpu="timestamp,index,temperature.gpu,clocks.sm,power.draw,utilization.gpu${reason:+,$reason}" \
      > "$O/cards.during.csv" 2>&1 &
    sampler=$!
    trap 'kill "$sampler" 2> /dev/null' EXIT
    say "each card's temperature, clock, power and use every $SAMPLE_MS ms, in out/cards.during.csv: one nvidia-smi" \
      "on $on"
  fi
fi
read -ra S <<< "$SIZES"
cut=() ran=()
while [ ${#S[@]} -gt 0 ] && [ $(((stop - t) / ${#S[@]})) -lt "$SIZE_MIN_S" ]; do
  cut=("${S[-1]}" "${cut[@]}"); unset 'S[-1]'
done
[ ${#S[@]} -gt 0 ] || verdict 9 refused "$(((stop - t) / 60)) min of lease are left before $(utc "$stop"), under the" \
  "$SIZE_MIN_S s a size needs; nothing started"

rc=0 k=0 last=start
for size in "${S[@]}"; do
  if [ -z "${DRY:-}" ] && ! clear; then
    [ $rc != 0 ] || rc=11
    cut+=("${S[@]:k}"); break
  fi
  t=$(now); share=$(((stop - t) / (${#S[@]} - k))); k=$((k + 1))
  if [ "$share" -lt "$SIZE_MIN_S" ]; then
    cut+=("$size"); say "$size GiB: not run, $share s are left for it"
    continue
  fi
  D=$O/host-$size
  RUN=(timeout -k 20 "$share" taskset -c "$DRIVER_CPUS" "$PY" "$RS/host_read.py" --lib "$B/session.so" --kv "$B/kv.so"
    --out "$D" --gib "$GIB" --host-gib "$size" --gpus "$GPUS" --chunk-blocks "$CHUNK_BLOCKS"
    --budget-s "$((share - TAIL_S))")
  [ -z "${NAMES:-}" ] || { read -ra W <<< "$NAMES"; RUN+=(--names "${W[@]}"); }
  [ -z "${ARGS:-}" ] || { read -ra W <<< "$ARGS"; RUN+=("${W[@]}"); }
  say "$size GiB, over by $(utc $((t + share))): ${RUN[*]}"
  ran+=("$size")
  if [ -n "${DRY:-}" ]; then
    export NOW=$((t + share))
    continue
  fi
  mkdir -p "$D"
  watch "${RUN[@]}" > "$D/host_read.log" 2>&1; r=$?
  [ -z "$ended" ] || verdict 143 failed "a signal ended the lease during $size GiB; out/host-*/ hold what was measured"
  throttled=$(cards "after-$size"); slowed=$(held_down "$last" "after-$size"); last=after-$size
  line="$size GiB: exit $r after $(($(now) - t)) s${throttled:+; throttle reasons after it, cards: $throttled}"
  say "$line${slowed:+; clocks held down during it: $slowed}"
  grep -E 's (store:|[^ ]+: gate |control under |alone: |beside: |chain: )' "$D/host_read.log" | cut -c1-400 \
    | tee -a "$O/tiers.txt"
  [ $rc != 0 ] || rc=$r
done
[ $rc != 0 ] || [ ${#cut[@]} = 0 ] || rc=10
if [ -z "${DRY:-}" ]; then
  cards end > /dev/null
  ! grep -q "$HELD" "$O/cards.end.txt" \
    || { slowed=$(held_down start end); say "clocks held down over the lease: ${slowed:-on no card}"; }
fi
case $rc in 0) word=ok ;; 10) word=partial ;; *) word=failed ;; esac
verdict "$rc" "$word" "sizes run: ${ran[*]:-none} GiB in chunks of $CHUNK_BLOCKS blocks${cut:+; not run: ${cut[*]} GiB};" \
  "each size's record is out/host-SIZE/host_read.json"
