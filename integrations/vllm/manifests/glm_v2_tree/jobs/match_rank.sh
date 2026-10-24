#!/usr/bin/env bash
# One CPU node's share of a split Match CPU phase (match_cpu.sh's RANKS), its inputs and outputs carried through the store by
# store_tar.sh: the Build of record (BUILD_PARTS; without it the row dir must hold the Build already, as the Build's own node
# does), the legs' small files with only RANKS' observer logs (LEGS_PARTS), then the RANKS run, then those ranks' documents
# (match_ranks.sha256) put as the set ranks-<RANKS> and labelled on this run, which pod_chain.sh waits for.  A failed step labels
# that set failed instead, so pod_chain.sh stops.  With BUILD_PARTS, a row dir already there (an earlier attempt's) is moved aside
# to <row>.prior-<run id>, its .tree with it.
#   research run --on NODE --mem-gb 450 --mem-wait 28800 --timeout 12h --custody-ttl 24h --source . --cwd source --project verity -- env ROW=... BUILD_RUN=... RANKS=5 [BUILD_PARTS=RUN:build] LEGS_PARTS=RUN:legs-small VERITY_QWORD_MAX_GATES=... VERITY_QWORD_MAX_GATES_ALLOWED=... bash integrations/vllm/manifests/glm_v2_tree/jobs/match_rank.sh
# env: ROW, BUILD_RUN, RANKS, LEGS_PARTS, [BUILD_PARTS], [SW], match_cpu.sh's (FOLD_WORKERS, FOLD_RANK_WORKERS: at --mem-gb 380
# a rank needs FOLD_RANK_WORKERS=1 FOLD_WORKERS=1), store_tar.sh's [WAIT_MIN], [EVICT_FREE_GB].
set -uo pipefail
: "${ROW:?ROW}" "${BUILD_RUN:?BUILD_RUN}" "${RANKS:?RANKS}" "${LEGS_PARTS:?LEGS_PARTS}"
[[ $RANKS =~ ^[0-9](,[0-9])*$ ]] || { echo "RANKS=$RANKS: single-digit ranks separated by commas"; exit 3; }
TREE=$PWD J=$PWD/integrations/vllm/manifests/glm_v2_tree/jobs
source "$J/model.sh"
D=$SW/$ROW SET=ranks-${RANKS//,/} O=$OUT
fail() { echo "$1"; OUT=$O NAME=$SET MSG="$1" bash "$J/store_tar.sh" fail; exit "$2"; }
if [ -n "${BUILD_PARTS:-}" ]; then
  [ ! -e "$D" ] || { mv "$D" "$D.prior-$RESEARCH_RUN_ID" && echo "$D moved aside to $D.prior-$RESEARCH_RUN_ID"; } || exit 3
  [ ! -e "$D.tree" ] || mv "$D.tree" "$D.prior-$RESEARCH_RUN_ID.tree" || exit 3
  OUT=$O PARTS=$BUILD_PARTS bash "$J/store_tar.sh" get || fail "the Build's get ($BUILD_PARTS) failed (rc $?)" 5
fi
[ -f "$D/build_summary.json" ] && [ -s "$D/manifest.json" ] || fail "no Build of record in $D" 3
[ -z "$(find "$D/match" -type f 2>/dev/null | head -1)" ] || fail "$D/match holds files already: this run takes an empty one" 3
OUT=$O PARTS=$LEGS_PARTS EXCLUDE="/match/capture/log[.]rank[^${RANKS//,/}][.]jsonl[.]gz\$" bash "$J/store_tar.sh" get ||
  fail "the legs' get ($LEGS_PARTS) failed (rc $?)" 5
OUT=$O bash "$J/match_cpu.sh"; rc=$?
[ "$rc" = 0 ] || fail "match_cpu.sh RANKS=$RANKS failed (rc $rc)" "$rc"
OUT=$O LIST=$O/match_ranks.sha256 NAME=$SET bash "$J/store_tar.sh" put || fail "the put of $SET failed (rc $?)" 5
