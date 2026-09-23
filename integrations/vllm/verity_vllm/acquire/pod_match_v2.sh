#!/usr/bin/env bash
# The GPU gate on a pod: the row's Match stage (row_pod.sh, stages=match) re-run over the STAGED Build dir with ACQUIRE_ENGINE=v2 -- the
# plan's hooks installed beside the Match observer (m1_capture writes acquisition_plan.json / site_compare.json / acquisition_observed.json
# into match/capture) -- then `match_compare` against the staged v1 Match record.
#
#   ROW=<row id> ROLE=<case> REPO=<hf repo> REV=<revision> [MANIFEST=/workspace/acquire_gate/<row>/manifest.json] bash verity_vllm/acquire/pod_match_v2.sh
#
# The attempt dir is $SWEEP_DIR/$ROW (default /workspace/cp/sweep_v2acq): the staged Build outputs are linked in (never copied, never
# written), the Match record is produced there; compare.json + the small acquisition documents are copied to $OUT (the run dir).
set -uo pipefail
cd "$(dirname "$0")/../.."                      # integrations/vllm
ROW=${ROW:?row id}; ROLE=${ROLE:?case}; REPO=${REPO:?hf repo}; REV=${REV:?revision}
ROWS=${ROWS:-/workspace/rows}; SRC=$ROWS/$ROW
export SWEEP_DIR=${SWEEP_DIR:-/workspace/cp/sweep_v2acq}
D=$SWEEP_DIR/$ROW; mkdir -p "$D"
OUT=${OUT:-${RESEARCH_RUN_DIR:-out}/acquire_match}; mkdir -p "$OUT"
MANIFEST=${MANIFEST:-/workspace/acquire_gate/$ROW/manifest.json}
[ -d "$SRC" ] || { echo "[pod_match_v2] no staged row $SRC"; exit 2; }
[ -f "$MANIFEST" ] || { echo "[pod_match_v2] no required-value manifest $MANIFEST (run pod_gate.sh first)"; exit 2; }
for f in "$SRC"/build_* "$SRC"/build_summary.json; do ln -sfn "$f" "$D/"; done
echo "[pod_match_v2] attempt dir $D over the staged Build of $SRC; manifest $MANIFEST"
export ACQUIRE_ENGINE=v2 ACQUIRE_MANIFEST=$MANIFEST
T0=$(date +%s)
bash verity_capture/sweep/row_pod.sh "$ROW" "$ROLE" "$REPO" "$REV" match; RC=$?
echo "[pod_match_v2] row_pod match rc=$RC in $(( $(date +%s) - T0 )) s"
cat "$D/stages.txt" 2>/dev/null
for f in acquisition_plan.json site_compare.json acquisition_observed.json runtime_tree.json v1_decision.json tokens.json; do
  [ -f "$D/match/capture/$f" ] && cp "$D/match/capture/$f" "$OUT/capture_$f"
done
[ -f "$D/match/control/tokens.json" ] && cp "$D/match/control/tokens.json" "$OUT/control_tokens.json"
for f in global_match.json match_decomp.json stages.txt row.log match.log global_match.log; do [ -f "$D/$f" ] && cp "$D/$f" "$OUT/"; done
[ -f "$D/match/gates.json" ] && cp "$D/match/gates.json" "$OUT/"
grep -h "^\[m1\] ACQUIRE\|^\[m1\] acquisition\|^\[m1\] site compare\|^\[m1\]   " "$D"/match/logs/*.log 2>/dev/null | head -60 > "$OUT/m1_acquire_lines.log"
PY=${PY:-/workspace/venv312/bin/python}
PYTHONPATH=. "$PY" -m verity_vllm.acquire.match_compare --staged "$SRC" --v2 "$D" --out "$OUT/compare.json" | tee "$OUT/compare.log"; CRC=$?
echo "[pod_match_v2] compare rc=$CRC -> $OUT"
exit $(( RC != 0 ? RC : CRC ))
