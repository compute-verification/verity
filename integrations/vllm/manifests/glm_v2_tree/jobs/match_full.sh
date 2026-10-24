#!/usr/bin/env bash
# A split Match CPU phase's full run on a CPU node, for a Commit on another node, every input and output carried through the store
# (store_tar.sh), so the node's memory, admitted once, serves its own ranks' legs and the full run.  OWN ("r=RUN:fold-r ...")
# are this run's ranks: each rank's compare and GM-01 over its fold (match_rank.sh with FOLD_PARTS, one rank at a time, in its own
# sweep $SW-r<r>, removed once its set ranks-<r> is put and labelled on this run).  FULL=1 (the default) is the full run: the
# Build of record (BUILD_PARTS) and the legs' small files (LEGS_PARTS) into the row dir first; OWN's documents copied in; then
# each other rank's (RANK_SETS, match_rank.sh's sets, waited for); the legs checked against match_gpu.sh's list (LEGS) less its
# values, which the CPU phase doesn't read; and match_cpu.sh's full run with MATCH_REUSE=legs, which keeps every rank's documents
# and runs the check and the cross-rank leg.  When the Match passes, the files it holds (match_cpu.sha256) are put as the set
# match-cpu and this run is labelled glm_v2.match_run with its vllm.match run, the label the Commit's start waits for; a failed
# step labels match-cpu failed instead.  FULL=0 runs OWN alone.
#   research run --on NODE --mem-gb 410 --mem-wait 43200 --timeout 18h --custody-ttl 24h --source . --cwd source --project verity --send LEGS.sha256.gz -- env ROW=... BUILD_RUN=... BUILD_PARTS=RUN:build LEGS_PARTS=RUN:legs-small LEGS=LEGS.sha256.gz OWN="2=RUN:fold-2" RANK_SETS="RUN:ranks-0 RUN:ranks-1 ..." SW=... VERITY_QWORD_MAX_GATES=... VERITY_QWORD_MAX_GATES_ALLOWED=... bash integrations/vllm/manifests/glm_v2_tree/jobs/match_full.sh
#   research run --on NODE --mem-gb 380 ... -- env ROW=... BUILD_RUN=... BUILD_PARTS=... LEGS_PARTS=... OWN="0=RUN:fold-0 1=RUN:fold-1" FULL=0 SW=... ... bash .../match_full.sh
# env: ROW, BUILD_RUN, BUILD_PARTS, LEGS_PARTS, [OWN], [FULL=1]; FULL=1: LEGS (a path, else $RESEARCH_RUN_DIR/inputs/LEGS; gzipped
# or not), RANK_SETS; [SW], [FOLD_WORKERS=1], [FOLD_RANK_WORKERS=1], store_tar.sh's [WAIT_MIN=900], [EVICT_FREE_GB], [FETCH_*];
# [CUDA_DRIVER_ART] (README "GPU batch"): a node without the venv, libcuda.so.1 or the model's metadata is readied by stage_cpu.sh.
set -uo pipefail
: "${ROW:?ROW}" "${BUILD_RUN:?BUILD_RUN}" "${BUILD_PARTS:?BUILD_PARTS}" "${LEGS_PARTS:?LEGS_PARTS}"
FULL=${FULL:-1}
[ "$FULL" = 1 ] || [ -n "${OWN:-}" ] || { echo "FULL=0 without OWN: nothing to run"; exit 3; }
for own in ${OWN:-}; do
  [[ $own =~ ^[0-9]=r[0-9]{8}-[0-9]{6}-[0-9a-f]+:fold-[0-9]$ ]] && [ "${own%%=*}" = "${own##*-}" ] ||
    { echo "OWN: $own is not r=RUN:fold-r"; exit 3; }
done
TREE=$PWD J=$PWD/integrations/vllm/manifests/glm_v2_tree/jobs
source "$J/model.sh"
D=$SW/$ROW O=$OUT
export FOLD_WORKERS=${FOLD_WORKERS:-1} FOLD_RANK_WORKERS=${FOLD_RANK_WORKERS:-1} WAIT_MIN=${WAIT_MIN:-900}
step() { echo "$(date -u +%FT%TZ) [full] $*"; }
fail() { step "$1"; [ "$FULL" != 1 ] || OUT=$O NAME=match-cpu MSG="$1" SW=$SW bash "$J/store_tar.sh" fail; exit "$2"; }
W0=/workspace/cp/circuits
# store_tar.sh needs the venv before anything is fetched; a node that has it is left alone (its other jobs share the venv). Jobs
# started together on a fresh node take turns: the first readies it and the others find it ready.
mkdir -p "$W0" && exec 9>"$W0/.stage_cpu.lock" && flock 9 || fail "no lock $W0/.stage_cpu.lock" 3
if [ ! -x "$W0/venv312/bin/python" ] || [ ! -d "$SNAP" ] || ! { ldconfig -p | grep -q 'libcuda\.so\.1 ' || [ -e "$W0/cuda-driver/libcuda.so.1" ]; }; then
  step "readying the node (stage_cpu.sh, CUDA_DRIVER_ART=${CUDA_DRIVER_ART:-})"
  mkdir -p "$O/stage" && OUT=$O/stage W=$W0 bash "$J/stage_cpu.sh" || fail "stage_cpu.sh failed (rc $?)" 5
fi
exec 9>&-

if [ "$FULL" = 1 ]; then
  : "${LEGS:?LEGS}" "${RANK_SETS:?RANK_SETS}"
  [ -f "$LEGS" ] || LEGS=$RESEARCH_RUN_DIR/inputs/$LEGS
  [ -s "$LEGS" ] || fail "no legs list $LEGS (match_gpu.sh's match.sha256)" 3
  case $LEGS in *.gz) gzip -dc "$LEGS" ;; *) cat "$LEGS" ;; esac |
    awk 'substr($0, 67) !~ /\.tree$/ && substr($0, 67) !~ /\/values\//' >"$O/legs-small.sha256"
  [ -s "$O/legs-small.sha256" ] || fail "$LEGS: no legs but values" 3
  step "the Build of record from $BUILD_PARTS into $D"
  [ ! -e "$D" ] || { mv "$D" "$D.prior-$RESEARCH_RUN_ID" && echo "$D moved aside to $D.prior-$RESEARCH_RUN_ID"; } || exit 3
  [ ! -e "$D.tree" ] || mv "$D.tree" "$D.prior-$RESEARCH_RUN_ID.tree" || exit 3
  mkdir -p "$O/build" && OUT=$O/build PARTS=$BUILD_PARTS SW=$SW bash "$J/store_tar.sh" get || fail "the Build's get ($BUILD_PARTS) failed (rc $?)" 5
  [ -f "$D/build_summary.json" ] && [ -s "$D/manifest.json" ] || fail "no Build of record in $D" 3
  step "the legs' small files from $LEGS_PARTS"
  mkdir -p "$O/legs" && OUT=$O/legs PARTS=$LEGS_PARTS SW=$SW bash "$J/store_tar.sh" get || fail "the legs' get ($LEGS_PARTS) failed (rc $?)" 5
  step "the legs against $LEGS, less its values ($(wc -l <"$O/legs-small.sha256") files)"
  (cd "$SW" && sha256sum --quiet -c -) <"$O/legs-small.sha256" || fail "the legs in $D/match are not the ones match_gpu.sh hashed" 3
fi

for own in ${OWN:-}; do
  r=${own%%=*} S=$SW-r${own%%=*}
  step "rank $r's compare and GM-01 over ${own#*=} in $S"
  mkdir -p "$O/rank$r" && OUT=$O/rank$r SW=$S RANKS=$r FOLD_PARTS=${own#*=} LABEL_ALSO= bash "$J/match_rank.sh" ||
    fail "rank $r's compare and GM-01 (match_rank.sh) failed (rc $?)" 5
  if [ "$FULL" = 1 ]; then
    (cd "$S" && cut -c67- "$O/rank$r/match_ranks.sha256" | tar -cf - -T -) | (cd "$SW" && tar -xf -) &&
      (cd "$SW" && sha256sum --quiet -c -) <"$O/rank$r/match_ranks.sha256" || fail "rank $r's documents differ in $D" 3
  fi
  rm -rf "$S/$ROW" "$S/$ROW.tree"
done
[ "$FULL" = 1 ] || { step "OWN done: $(for own in $OWN; do printf '%s:ranks-%s ' "$RESEARCH_RUN_ID" "${own%%=*}"; done)"; exit 0; }

for s in $RANK_SETS; do
  step "the documents of $s"
  mkdir -p "$O/${s#*:}" && OUT=$O/${s#*:} PARTS=$s SW=$SW bash "$J/store_tar.sh" get || fail "$s: not there (rc $?)" 5
done
tp=${ROW#*__tp} tp=${tp%%__*}
for r in $(seq 0 $((tp - 1))); do
  [ -n "$(ls -A "$D/match/fold_match/rank$r" 2>/dev/null)" ] || fail "no documents of rank $r in $D/match/fold_match (OWN and RANK_SETS name every rank)" 3
done
step "the full run, MATCH_REUSE=legs"
mkdir -p "$O/match" && OUT=$O/match SW=$SW MATCH_REUSE=legs bash "$J/match_cpu.sh"; rc=$?
MATCH_RUN=$(grep -oE -- '--id r[0-9]{8}-[0-9]{6}-[0-9a-f]+ --tool vllm\.match' "$O/match/chain.out" | tail -1 | cut -d' ' -f2)
echo "{\"match_run\": \"$MATCH_RUN\", \"rc\": $rc}" >"$O/match_run.json"
[ "$rc" = 0 ] || fail "the Match ($MATCH_RUN) failed (rc $rc)" "$rc"
[ -n "$MATCH_RUN" ] || fail "no vllm.match run in $O/match/chain.out" 3
step "the Match passed ($MATCH_RUN)"
OUT=$O LIST=$O/match/match_cpu.sha256 NAME=match-cpu SW=$SW bash "$J/store_tar.sh" put || fail "the put of match-cpu failed (rc $?)" 5
OUT=$O KEY=glm_v2.match_run VALUE=$MATCH_RUN SW=$SW bash "$J/store_tar.sh" label || { step "labelling glm_v2.match_run failed"; exit 5; }
step "match-cpu put and glm_v2.match_run=$MATCH_RUN labelled on $RESEARCH_RUN_ID"
