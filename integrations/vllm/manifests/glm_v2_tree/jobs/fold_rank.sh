#!/usr/bin/env bash
# One rank's fold, the fold Match's first leg (match_cpu.sh), alone: a GLM-5.3 B8 rank's peaks near 44 GB for about 2 h, against
# its GM-01's 370 GB, so folds run where memory is short and GM-01 later (match_rank.sh's FOLD_PARTS).  Over the Build of record
# (BUILD_PARTS) and the legs' small files with only RANK's observer log (LEGS_PARTS), both carried by store_tar.sh into the row
# dir (a row dir already there moved aside to <row>.prior-<run id>, as match_rank.sh does), it folds the rank as tp.fold_match
# does: the profile derived_<ROLE>_tp<W>, under the construction the rank's Build Programs declare (match.fold_construction_args).
# Its match/fold_match/rank<RANK> is put as the set fold-<RANK>, labelled on this run; a failed step labels that set failed
# instead.  match_rank.sh with MATCH_REUSE=fold keeps that fold (--reuse-fold: only one folded under the same flags).  Two jobs on
# one node take two sweeps (SW); CLEAN=1 removes this job's row dir once the set is put.
#   research run --on NODE --mem-gb 48 --mem-wait 28800 --timeout 8h --custody-ttl 24h --source . --cwd source --project verity -- env ROW=... RANK=0 SW=/workspace/cp/circuits/glm-v2/rows-fold0 BUILD_PARTS=RUN:build LEGS_PARTS=RUN:legs-small bash integrations/vllm/manifests/glm_v2_tree/jobs/fold_rank.sh
# env: ROW, RANK, BUILD_PARTS, LEGS_PARTS, [SW], [REF], [FOLD_TIMEOUT=86400], [CLEAN], store_tar.sh's [WAIT_MIN], [EVICT_FREE_GB],
# [FETCH_MIN_MBPS], [FETCH_TRIES] and [LABEL_ALSO].
set -uo pipefail
: "${ROW:?ROW}" "${RANK:?RANK}" "${BUILD_PARTS:?BUILD_PARTS}" "${LEGS_PARTS:?LEGS_PARTS}"
[[ $RANK =~ ^[0-9]$ ]] || { echo "RANK=$RANK: one single-digit rank"; exit 3; }
TREE=$PWD J=$PWD/integrations/vllm/manifests/glm_v2_tree/jobs W0=/workspace/cp/circuits
source "$J/model.sh"
D=$SW/$ROW SET=fold-$RANK O=$OUT
fail() { echo "$1"; OUT=$O NAME=$SET MSG="$1" bash "$J/store_tar.sh" fail; exit "$2"; }
[ ! -e "$D" ] || { mv "$D" "$D.prior-$RESEARCH_RUN_ID" && echo "$D moved aside to $D.prior-$RESEARCH_RUN_ID"; } || exit 3
[ ! -e "$D.tree" ] || mv "$D.tree" "$D.prior-$RESEARCH_RUN_ID.tree" || exit 3
OUT=$O PARTS=$BUILD_PARTS bash "$J/store_tar.sh" get || fail "the Build's get ($BUILD_PARTS) failed (rc $?)" 5
[ -f "$D/build_summary.json" ] && [ -s "$D/manifest.json" ] || fail "no Build of record in $D" 3
OUT=$O PARTS=$LEGS_PARTS EXCLUDE="/match/capture/log[.]rank[^$RANK][.]jsonl[.]gz\$" bash "$J/store_tar.sh" get ||
  fail "the legs' get ($LEGS_PARTS) failed (rc $?)" 5
[ -f "$D/match/capture/log.rank$RANK.jsonl.gz" ] || fail "no observer log of rank $RANK in $D/match/capture" 3
source "$J/tree_guard.sh" && tree_guard "$ROW" "$SW" "$REF" || fail "tree_guard refused this job's tree" 2
(cd "$TREE/integrations/vllm" && WORK=$W0 VENV=$W0/venv312 HF_HOME=$W0/hf bash verity_vllm/ops/pod_bootstrap.sh --cpu --cases "" --out "$O/bootstrap") \
  >"$O/bootstrap.log" 2>&1 || { tail -30 "$O/bootstrap.log"; fail "the CPU bootstrap failed" 5; }
# row_tp's form of record: the library default (runs, v1) writes a B8 rank's instances.jsonl at ~410 GB, against ~28 GB here
export PY=$W0/venv312/bin/python HF_HOME=$W0/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= VERITY_INSTANCES_FORM=progressions
export PYTHONPATH=$TREE:$TREE/tools/research/src:$TREE/infra/cluster/src:$TREE/integrations/vllm:$TREE/verity/ml/flock/python:$TREE/tools/verity/circuit_check/src
tp=${ROW#*__tp} tp=${tp%%__*}
(cd "$TREE/integrations/vllm" && "$PY" - "$D" "$RANK" "$tp" "workloads/$ROW.json" "derived_${ROLE}_tp$tp" "${FOLD_TIMEOUT:-86400}" "$O/fold_rank.json" <<'EOF'
import glob, json, os, sys
from pathlib import Path
from verity_vllm.pipeline.match import fold_construction_args
from verity_vllm.pipeline.tp import fold_match as FM
d, r, world, wl, profile, timeout, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6]), sys.argv[7]
# row_tp's fold Match: the rank's request Program dirs that hold a result, the first its --build-rank and the rest --derived-rank
dirs = [p for p in sorted(glob.glob(f"{d}/build/rank{r}/build_request_LP*_T*")) if os.path.isfile(f"{p}/result.json")]
if not dirs or not os.path.isfile(wl):
    sys.exit(f"rank {r}: no request Program under {d}/build/rank{r}" if not dirs else f"no workload {wl}")
construction = fold_construction_args(wl, dirs)
cap, rd = Path(d, "match", "capture"), Path(d, "match", "fold_match", f"rank{r}")
rd.mkdir(parents=True, exist_ok=True)
FM._link(rd / "capture" / "log.jsonl.gz", cap / f"log.rank{r}.jsonl.gz")
if (cap / "tokens.json").exists():
    FM._link(rd / "capture" / "tokens.json", cap / "tokens.json")
print(f"rank {r}: folding under {profile} with {construction or 'no construction flags'} ({len(dirs)} request Programs)", flush=True)
f = FM.fold_rank(os.environ["PY"], cap / f"log.rank{r}.jsonl.gz", profile, rd, timeout, construction=construction)
f["construction_args"] = construction
Path(out).write_text(json.dumps(f, indent=1, default=str) + "\n")
print(json.dumps({k: f.get(k) for k in ("ok", "rc", "seconds", "n_instances", "n_steps", "errors", "unresolved", "max_rss_gb", "construction", "tp")}, default=str))
tp = f.get("tp") or {}
bad = [] if f["ok"] else [f"fold not ok (rc {f['rc']}, errors {f['errors']}, unresolved {f['unresolved']}): {json.dumps(f['error_examples'])[:300]}"]
if int(tp.get("rank", -1)) != r or int(tp.get("world", 0)) != world:
    bad.append(f"the log's RunHeader.notes.tp2 states rank {tp.get('rank')} / world {tp.get('world')}, not {r} / {world}")
sys.exit(f"rank {r}: " + "; ".join(bad) if bad else 0)
EOF
); rc=$?
[ "$rc" = 0 ] || fail "rank $RANK's fold failed (rc $rc; fold_rank.json)" "$rc"
(cd "$SW" && find "$ROW/match/fold_match/rank$RANK" -type f -print0 | xargs -0 -r sha256sum) | sort -k2 >"$O/fold.sha256"
echo "fold.sha256: $(wc -l <"$O/fold.sha256") files"
OUT=$O LIST=$O/fold.sha256 NAME=$SET bash "$J/store_tar.sh" put || fail "the put of $SET failed (rc $?)" 5
[ "${CLEAN:-0}" != 1 ] || { rm -rf "$D" "$D.tree" && echo "CLEAN=1: $D removed (the set holds its fold)"; }
