#!/usr/bin/env bash
# A split Match CPU phase's primary, on the GPU node that holds the legs, chained to the Commit so that neither waits on an agent:
# the Build of record into the row dir (BUILD_PARTS, store_tar.sh); this node's own RANKS run in a second sweep SIDE holding the
# Build and the legs' small files with only those ranks' observer logs (the row dir can't take it: a RANKS run writes the
# check, and the full run refuses a row dir that ran it); those ranks' documents, then each other node's (RANK_SETS, the sets
# match_rank.sh puts, waited for), into the row dir's match/fold_match; the legs checked against match_gpu.sh's list (LEGS);
# match_cpu.sh's full run with MATCH_REUSE=legs, which keeps every rank's documents and runs the cross-rank leg; and, when the
# Match passes, commit.sh over its vllm.match run.  Each step's records are under $OUT/<step>.
#   research run --on vy-glm53-h200-1 --mem-gb 1300 --mem-wait 28800 --timeout 30h --custody-ttl 36h --source . --cwd source --project verity -- env ROW=... BUILD_RUN=... BUILD_PARTS=RUN:build LEGS=/workspace/research/runs/RUN/out/match.sha256 RANKS=0,1,2,3 RANK_SETS="RUN:ranks-4 RUN:ranks-5 ..." NODE=vy-glm53-h200-1 VERITY_QWORD_MAX_GATES=... VERITY_QWORD_MAX_GATES_ALLOWED=... bash integrations/vllm/manifests/glm_v2_tree/jobs/pod_chain.sh
# env: ROW, BUILD_RUN, BUILD_PARTS, LEGS (a path, else $RESEARCH_RUN_DIR/inputs/LEGS), RANKS, RANK_SETS, [NODE], [SIDE=$SW-side],
# [FOLD_WORKERS], [COMMIT=1], store_tar.sh's [WAIT_MIN], [EVICT_FREE_GB].
set -uo pipefail
: "${ROW:?ROW}" "${BUILD_RUN:?BUILD_RUN}" "${BUILD_PARTS:?BUILD_PARTS}" "${LEGS:?LEGS}" "${RANKS:?RANKS}" "${RANK_SETS:?RANK_SETS}"
[[ $RANKS =~ ^[0-9](,[0-9])*$ ]] || { echo "RANKS=$RANKS: single-digit ranks separated by commas"; exit 3; }
TREE=$PWD J=$PWD/integrations/vllm/manifests/glm_v2_tree/jobs
source "$J/model.sh"
D=$SW/$ROW SIDE=${SIDE:-$SW-side} O=$OUT
[ -f "$LEGS" ] || LEGS=$RESEARCH_RUN_DIR/inputs/$LEGS
[ -s "$LEGS" ] || { echo "no legs list $LEGS (match_gpu.sh's match.sha256)"; exit 3; }
step() { echo "$(date -u +%FT%TZ) [chain] $*"; }
[ -d "$D/match/capture" ] || { echo "no legs in $D/match"; exit 3; }
[ ! -e "$D/match/tp2_match.json" ] || { echo "$D/match already ran its check"; exit 3; }
cp "$LEGS" "$O/legs.sha256"

step "the Build of record from $BUILD_PARTS"
mkdir -p "$O/build" && OUT=$O/build PARTS=$BUILD_PARTS bash "$J/store_tar.sh" get || exit
B=$(ls "$O"/build/got-*.sha256 | head -1)

step "ranks $RANKS in $SIDE"
[ ! -e "$SIDE/$ROW" ] || mv "$SIDE/$ROW" "$SIDE/$ROW.prior-$RESEARCH_RUN_ID" || exit 3
[ ! -e "$SIDE/$ROW.tree" ] || mv "$SIDE/$ROW.tree" "$SIDE/$ROW.prior-$RESEARCH_RUN_ID.tree" || exit 3
x="/match/capture/log[.]rank[^${RANKS//,/}][.]jsonl[.]gz\$"
{ cat "$B"; awk -v x="$x" 'substr($0, 67) !~ /\.tree$/ && substr($0, 67) !~ /\/values\// && substr($0, 67) !~ x' "$LEGS"; } >"$O/side.sha256"
mkdir -p "$SIDE" && cp "$SW/$ROW.tree" "$SIDE/$ROW.tree" || exit 3
(cd "$SW" && cut -c67- "$O/side.sha256" | tar -cf - -T -) | (cd "$SIDE" && tar -xf -) &&
  (cd "$SIDE" && sha256sum --quiet -c -) <"$O/side.sha256" || { echo "$SIDE differs from $O/side.sha256"; exit 3; }
echo "$SIDE: $(wc -l <"$O/side.sha256") files"
mkdir -p "$O/side" && OUT=$O/side SW=$SIDE FOLD_WORKERS=${FOLD_WORKERS:-4} bash "$J/match_cpu.sh" ||
  { rc=$?; step "match_cpu.sh RANKS=$RANKS in $SIDE failed (rc $rc): no full run, no Commit"; exit "$rc"; }
(cd "$SIDE" && cut -c67- "$O/side/match_ranks.sha256" | tar -cf - -T -) | (cd "$SW" && tar -xf -) &&
  (cd "$SW" && sha256sum --quiet -c -) <"$O/side/match_ranks.sha256" || { echo "ranks $RANKS' documents differ in $D"; exit 3; }

for s in $RANK_SETS; do
  step "the documents of $s"
  mkdir -p "$O/${s#*:}" && OUT=$O/${s#*:} PARTS=$s bash "$J/store_tar.sh" get ||
    { rc=$?; step "$s: not there (rc $rc): no full run, no Commit"; exit "$rc"; }
done

step "the legs against $LEGS"
grep -vE '  [^ ]+\.tree$' "$LEGS" | (cd "$SW" && sha256sum --quiet -c -) || { echo "the legs in $D/match are not the ones match_gpu.sh hashed"; exit 3; }
step "the full run, MATCH_REUSE=legs"
mkdir -p "$O/match" && OUT=$O/match MATCH_REUSE=legs FOLD_RANK_WORKERS=${FOLD_RANK_WORKERS:-1} bash "$J/match_cpu.sh"; rc=$?
MATCH_RUN=$(grep -oE -- '--id r[0-9]{8}-[0-9]{6}-[0-9a-f]+ --tool vllm\.match' "$O/match/chain.out" | tail -1 | cut -d' ' -f2)
echo "{\"match_run\": \"$MATCH_RUN\", \"rc\": $rc}" >"$O/match_run.json"
[ "$rc" = 0 ] || { step "the Match ($MATCH_RUN) failed (rc $rc): no Commit"; exit "$rc"; }
[ -n "$MATCH_RUN" ] || { step "no vllm.match run in $O/match/chain.out"; exit 3; }
step "the Match passed ($MATCH_RUN)"
[ "${COMMIT:-1}" = 1 ] || exit 0
step "the Commit over $MATCH_RUN"
mkdir -p "$O/commit" && OUT=$O/commit MATCH_RUN=$MATCH_RUN NODE=${NODE:-$(hostname)} bash "$J/commit.sh"; rc=$?
step "the Commit ended (rc $rc); bundles: $O/commit/bundles.json"
exit "$rc"
