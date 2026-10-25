#!/usr/bin/env bash
# One CPU node's share of a split Match CPU phase (match_cpu.sh's RANKS), its inputs and outputs carried through the store by
# store_tar.sh: the Build of record (BUILD_PARTS; without it the row dir must hold the Build already, as the Build's own node
# does), the legs' small files with only RANKS' observer logs (LEGS_PARTS), then the RANKS run, then those ranks' documents
# (match_ranks.sha256) put as the set ranks-<RANKS> and labelled on this run, which pod_chain.sh waits for.  A failed step labels
# that set failed instead, so pod_chain.sh stops.  With BUILD_PARTS, a row dir already there (an earlier attempt's) is moved aside
# to <row>.prior-<run id>, its .tree with it.
#   research run --on NODE --mem-gb 450 --mem-wait 28800 --timeout 12h --custody-ttl 24h --source . --cwd source --project verity -- env ROW=... BUILD_RUN=... RANKS=5 [BUILD_PARTS=RUN:build] LEGS_PARTS=RUN:legs-small VERITY_QWORD_MAX_GATES=... VERITY_QWORD_MAX_GATES_ALLOWED=... bash integrations/vllm/manifests/glm_v2_tree/jobs/match_rank.sh
# FOLD_PARTS (RUN:fold-<rank>, fold_rank.sh's set, of a single RANKS) takes that rank's fold, waited for once the Build and
# the legs are in, and keeps it (MATCH_REUSE=fold, --reuse-fold: only a fold under the rank Build's construction), so the run is
# the rank's compare and GM-01.  CLEAN=1, with a sweep of this job alone (SW), records SW as @circuits's (`research keep`) before
# anything lands in it and frees it with `research retention rm` once the set is put (the full Match gets the set from the store).
# env: ROW, BUILD_RUN, RANKS, LEGS_PARTS, [BUILD_PARTS], [FOLD_PARTS], [SW], [CLEAN], match_cpu.sh's (FOLD_WORKERS, FOLD_RANK_WORKERS: at
# --mem-gb 380 a rank needs FOLD_RANK_WORKERS=1 FOLD_WORKERS=1), store_tar.sh's [WAIT_MIN], [EVICT_FREE_GB], [FETCH_MIN_MBPS],
# [FETCH_TRIES] and [LABEL_ALSO] (the run id this one replaces, which pod_chain.sh's RANK_SETS names: its set and a failure are
# labelled there too).
set -uo pipefail
: "${ROW:?ROW}" "${BUILD_RUN:?BUILD_RUN}" "${RANKS:?RANKS}" "${LEGS_PARTS:?LEGS_PARTS}"
[[ $RANKS =~ ^[0-9](,[0-9])*$ ]] || { echo "RANKS=$RANKS: single-digit ranks separated by commas"; exit 3; }
[ -z "${FOLD_PARTS:-}" ] || [[ $RANKS =~ ^[0-9]$ ]] || { echo "FOLD_PARTS is one rank's fold: RANKS=$RANKS"; exit 3; }
[ "${CLEAN:-0}" != 1 ] || [ -n "${SW:-}" ] || { echo "CLEAN=1 frees SW whole: give this job a sweep of its own"; exit 3; }
TREE=$PWD J=$PWD/integrations/vllm/manifests/glm_v2_tree/jobs
source "$J/model.sh"
D=$SW/$ROW SET=ranks-${RANKS//,/} O=$OUT
fail() { echo "$1"; OUT=$O NAME=$SET MSG="$1" bash "$J/store_tar.sh" fail; exit "$2"; }
rsch() { PYTHONPATH=$TREE/tools/research/src python3 -m research "$@"; }
[ "${CLEAN:-0}" != 1 ] || { mkdir -p "$SW" && rsch keep "$SW" --owner @circuits --expires 2d --priority normal \
  --why "B8 ranks $RANKS ($RESEARCH_RUN_ID): freed once $SET is in the store"; } || fail "research keep of $SW failed" 3
if [ -n "${BUILD_PARTS:-}" ]; then
  [ ! -e "$D" ] || { mv "$D" "$D.prior-$RESEARCH_RUN_ID" && echo "$D moved aside to $D.prior-$RESEARCH_RUN_ID"; } || exit 3
  [ ! -e "$D.tree" ] || mv "$D.tree" "$D.prior-$RESEARCH_RUN_ID.tree" || exit 3
  OUT=$O PARTS=$BUILD_PARTS bash "$J/store_tar.sh" get || fail "the Build's get ($BUILD_PARTS) failed (rc $?)" 5
fi
[ -f "$D/build_summary.json" ] && [ -s "$D/manifest.json" ] || fail "no Build of record in $D" 3
[ -z "$(find "$D/match" -type f 2>/dev/null | head -1)" ] || fail "$D/match holds files already: this run takes an empty one" 3
OUT=$O PARTS=$LEGS_PARTS EXCLUDE="/match/capture/log[.]rank[^${RANKS//,/}][.]jsonl[.]gz\$" bash "$J/store_tar.sh" get ||
  fail "the legs' get ($LEGS_PARTS) failed (rc $?)" 5
if [ -n "${FOLD_PARTS:-}" ]; then
  OUT=$O PARTS=$FOLD_PARTS bash "$J/store_tar.sh" get || fail "rank $RANKS's fold's get ($FOLD_PARTS) failed (rc $?)" 5
  export MATCH_REUSE=fold
fi
OUT=$O bash "$J/match_cpu.sh"; rc=$?
[ "$rc" = 0 ] || fail "match_cpu.sh RANKS=$RANKS failed (rc $rc)" "$rc"
OUT=$O LIST=$O/match_ranks.sha256 NAME=$SET bash "$J/store_tar.sh" put || fail "the put of $SET failed (rc $?)" 5
[ "${CLEAN:-0}" != 1 ] || rsch retention rm "$SW" --approved-by @circuits --ref "$RESEARCH_RUN_ID: $SET put" \
  --by "${RESEARCH_AGENT_URL:-research run $RESEARCH_RUN_ID}" --apply || echo "CLEAN=1: retention rm of $SW failed; it stays recorded as @circuits's"
