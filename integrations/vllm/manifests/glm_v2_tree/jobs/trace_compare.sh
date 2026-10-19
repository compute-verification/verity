#!/usr/bin/env bash
# circuits V2PREP: one GLM-5.3 row's Programs on this tree, its AOT Build rehearsed, and its own trace compared with the pins:
#   1. the row's AOT Build (`row chain --stages build --no-publish`, construction record: each derive `predict emit` against the row's
#      pinned `aot.trace` Build, fetched from the store, refused where this tree's predicted Program is not that trace's),
#      Programs only (`--build-programs-only=1`: no manifest), as a vllm.build Attempt of its own; its digests against the pins
#      (aot_neutrality.out).  A PASS means this tree's AOT Build emits the pinned Programs.
#   2. `verity-vllm neutrality ROW --tier heavy` (`row run --stages build --build-construction trace --build-programs-only=1`):
#      this tree's traced Build against the pins (did v2 move a Program in the trace?).
#   3. only when 1 or 2 did not reproduce the pins: the trace preserved (trace_art.sh: $OUT/trace_art.txt, the moved row's
#      `records switch --trace`), and side by side `predict compare` of this tree's predictor against this tree's trace and the
#      AOT Build again against it (`--aot-trace`), with its digests against the trace's (aot2_vs_trace.json).
# SKIP_AOT1=1 skips 1 for a row already known to move (every derive of it is refused only after its whole emission).
# The pinned vLLM imports libcuda.so.1 even on the meta device: on a node without a driver, CUDA_DRIVER_DIR (W/cuda-driver, the H200
# hosts' 580.173.02 library) goes on LD_LIBRARY_PATH; no device is opened.  The CPU venv and checkpoint metadata are the ones
# aot_build_rm.sh (WS-E) keeps on vy-glm53-cpu-2 under W.  Row dirs stay on the node (kept 5d).
# env: ROW, [W (/workspace/cp/circuits)], [BUDGET_GB (100)], [WORKERS (2)], [SKIP_TRACE=1], [SKIP_AOT1=1].  cwd: the source root.
# outputs: aot_chain.out, aot_build_summary.json, aot_row.json, aot_neutrality.out; neutrality.out, neutrality.json,
# build_summary.json, trace.tgz (build_summary.json and every build/rank*/build_*/result.json), rowdir.txt; when moved:
# trace_art.txt, comparison.json, compare.err, aot2_chain.out, aot2_build_summary.json, aot2_vs_trace.json.
set -uo pipefail
: "${ROW:?ROW}"
REV=aca966e4e02791568aa6a4ced368624b3d897f42
TREE=$PWD
W=${W:-/workspace/cp/circuits}
SW=$W/glm-v2/$RESEARCH_RUN_ID
SNAP=$W/hf/hub/models--zai-org--GLM-5.3/snapshots/$REV
[ -s "$SNAP/config.json" ] && [ -s "$SNAP/model.safetensors.index.json" ] || { echo "no GLM-5.3 metadata at $SNAP"; exit 3; }
C=$(dirname "$(dirname "$RESEARCH_RUN_DIR")")/requests/$RESEARCH_RUN_ID/custody
[ -f "$C/cred.json" ] || { echo "no custody key at $C"; exit 4; }
if ! ldconfig -p | grep -q 'libcuda\.so\.1 '; then
  CUDA_DRIVER_DIR=${CUDA_DRIVER_DIR:-$W/cuda-driver}
  [ -e "$CUDA_DRIVER_DIR/libcuda.so.1" ] || { echo "no libcuda.so.1 on the node and none at $CUDA_DRIVER_DIR"; exit 3; }
  export LD_LIBRARY_PATH=$CUDA_DRIVER_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  echo "libcuda.so.1: $CUDA_DRIVER_DIR ($(readlink "$CUDA_DRIVER_DIR/libcuda.so.1"))"
fi
mkdir -p "$SW" || exit 3
git -C "$TREE" rev-parse HEAD >"$OUT/tree.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" >"$OUT/tree.txt"
(cd integrations/vllm && WORK=$W VENV=$W/venv312 HF_HOME=$W/hf bash verity_vllm/ops/pod_bootstrap.sh --cpu --cases "" --out "$OUT/bootstrap") \
  >"$OUT/bootstrap.log" 2>&1 || { tail -30 "$OUT/bootstrap.log"; exit 5; }
tail -1 "$OUT/bootstrap.log"
export PY=$W/venv312/bin/python HF_HOME=$W/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
export TMPDIR=$(mktemp -d /workspace/research/scratch/glmv2.XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT
export PATH=$W/venv312/bin:$PATH
export PYTHONPATH=$TREE:$TREE/tools/research/src:$TREE/infra/cluster/src:$TREE/infra/ops:$TREE/integrations/vllm:$TREE/verity/ml/flock/python:$TREE/tools/verity/circuit_check/src:$TREE/tools/verity/cli:$TREE/verity/core/difftest/python
export RESEARCH_STORE_CONFIG=$C/store.toml
eval "$($PY -m verity_vllm.pipeline.cli epoch custody-env "$C")"
printf '#!/bin/sh\nexec %s -m research "$@"\n' "$PY" >"$SW/research" && chmod +x "$SW/research"
export RESEARCH_BIN=$SW/research
"$RESEARCH_BIN" keep "$SW" --owner @circuits --expires 5d --priority normal \
  --why "GLM v2 AOT and traced Builds of $ROW ($RESEARCH_RUN_ID): a moved row's AOT Build takes the trace as --aot-trace" \
  || echo "research keep failed (continuing)"
cd integrations/vllm || exit 6
B=${BUDGET_GB:-100}

aot() {  # aot TAG SWEEP [--aot-trace DIR]: the AOT Build, Programs only, as a vllm.build Attempt
  local tag=$1 sweep=$2; shift 2
  /usr/bin/time -v "$PY" -m verity_vllm.pipeline.cli row chain --stages build --no-publish --by circuits "$ROW" GLM53 zai-org/GLM-5.3 "$REV" \
    --sweep-dir "$sweep" --build-construction record "$@" --build-programs-only=1 --build-ram-budget-gb "$B" >"$OUT/${tag}_chain.out" 2>&1
  local rc=$?
  grep -E "^build |Elapsed|Maximum resident" "$OUT/${tag}_chain.out" | tail -4
  cp "$sweep/$ROW/build_summary.json" "$OUT/${tag}_build_summary.json" 2>/dev/null
  cp "$sweep/$ROW/row.json" "$OUT/${tag}_row.json" 2>/dev/null
  cp "$sweep/$ROW/stages.txt" "$OUT/${tag}_stages.txt" 2>/dev/null
  return $rc
}

A1=$SW/aot
if [ -n "${SKIP_AOT1:-}" ]; then
  arc=skipped anrc=skipped
  echo "AOT Build against the pinned trace: skipped (SKIP_AOT1)"
else
  aot aot "$A1"; arc=$?
  "$PY" -m verity_vllm.pipeline.cli neutrality "$ROW" --tier heavy --compare --out "$A1" >"$OUT/aot_neutrality.out" 2>&1; anrc=$?
  tail -12 "$OUT/aot_neutrality.out"
  echo "AOT Build against the pinned trace: rc $arc; its digests against the pins: $([ $anrc = 0 ] && echo NEUTRAL || echo MOVED)"
fi
[ -n "${SKIP_TRACE:-}" ] && { [ "$arc" = 0 ] && [ "$anrc" = 0 ] && exit 0; exit 9; }

/usr/bin/time -v "$PY" -m verity_vllm.pipeline.cli neutrality "$ROW" --tier heavy --out "$SW" \
  --row-flag=--build-ram-budget-gb=$B >"$OUT/neutrality.out" 2>&1
nrc=$?
grep -E "^\[neutrality\]|^    rank|Elapsed|Maximum resident" "$OUT/neutrality.out" | tail -30
cp "$SW/neutrality.json" "$OUT/" 2>/dev/null
D=$SW/$ROW
for f in stages.txt row.log build_summary.json; do cp "$D/$f" "$OUT/" 2>/dev/null; done
echo "$D" >"$OUT/rowdir.txt"
grep -q '^build PASS' "$D/stages.txt" 2>/dev/null || { echo "the traced Build did not pass (neutrality rc $nrc): $(grep '^build ' "$D/stages.txt" | tail -1 | cut -c1-300)"; exit 7; }
(cd "$D" && tar czf "$OUT/trace.tgz" build_summary.json $(ls -d build/rank*/build_*/result.json 2>/dev/null))
if [ "$arc" = 0 ] && [ "$anrc" = 0 ] && [ $nrc = 0 ]; then
  echo "v2 moves no Program of $ROW: its AOT Build emits the pins, and its trace is them"; exit 0
fi
[ $nrc = 0 ] || (ROW=$ROW D=$D bash "$TREE/integrations/vllm/manifests/glm_v2_tree/jobs/trace_art.sh") || echo "the trace was not preserved (trace_art.sh)"

/usr/bin/time -v "$PY" -m verity_vllm.pipeline.cli predict compare "$D" --row "$ROW" \
  --config verity_vllm/engine/profiles/hf_configs/GLM53.config.json --out "$OUT/comparison.json" --workers "${WORKERS:-2}" 2>"$OUT/compare.err" &
cpid=$!
A2=$SW/aot2
aot aot2 "$A2" --aot-trace "$D"; a2rc=$?
wait $cpid; crc=$?
grep -E "Elapsed|Maximum resident" "$OUT/compare.err"
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("predict compare equal:", d.get("equal"))' "$OUT/comparison.json" 2>/dev/null
[ "$crc" = 0 ] || { echo "neutrality rc $nrc, predict compare rc $crc: the predictor does not give this tree's trace"; exit 8; }
"$PY" - "$D" "$A2/$ROW" "$OUT/aot2_vs_trace.json" <<'EOF'
import json, sys
from pathlib import Path
from verity_vllm.pipeline import epoch, neutrality
t, a = (epoch.row_digests(Path(p))["programs"] for p in sys.argv[1:3])
diffs = neutrality.compare(t, a) if t and a else ["a Build has no build_summary.json"]
Path(sys.argv[3]).write_text(json.dumps({"trace": sys.argv[1], "aot": sys.argv[2], "equal": not diffs, "differences": diffs,
                                         "programs": a}, indent=1) + "\n")
print("AOT (against this trace) vs this trace:", "EQUAL" if not diffs else "DIFFER", *diffs, sep="\n    ")
EOF
echo "AOT vs pinned trace rc $arc; neutrality rc $nrc; predict compare rc $crc; AOT vs this trace rc $a2rc"
[ "$a2rc" = 0 ] || exit 9
exit 1
