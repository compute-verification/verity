#!/usr/bin/env bash
# The setup run on the served node, which `run.sh setup [--tie]` starts (placement.toml's [run.setup]), once per row and
# node: what that row's window run reads, left on the node under its paths. Steps, each skipped where its result is
# there unless STEPS names it:
#   pin      the INT8 row's pin: the base checkpoint copied from the node's /workspace/hf to HF_HOME and `int8-checkpoint
#            make`, every file held to manifests/checkpoints-int8.json. The BF16 row reads the node's base in place.
#   taps     HIDDEN_SO and NORM_TAP_SO built here from this tree (pod_fa3_tap.sh with its row guard, pod_norm_tap.sh)
#   flock    Flock's prover built in FLOCK_WORK (60-circuit.sh MODE=build)
#   build    `predict row` and the row's traced Build (`row run --stages build --config-run 1`, under one `gpu-lease 1 --on
#            CARD`, as its target check reads the device) in LLAMA_ROOT/<this run's id>/, and `predict compare` of the
#            two, as served_verdicts.sh's step 0 runs it; the INT8 row's prediction is held to the tree's pin
#            (test_predict.py's H200_INT8). Once it passes, placement.toml's builds names this run's id for the row.
# Nothing here serves. What leaves (out/): job.log, the steps' public lines, predict_row.json, predict_compare.json and the
# Build's stage verdict words; no key, salt, request or token.
set -uo pipefail
for v in CARD CORES END ROW_ID ROLE REPO REV VENV CUDA_HOME HF_HOME LLAMA_ROOT FLOCK_WORK HIDDEN_SO NORM_TAP_SO; do
  [ -n "${!v:-}" ] || { echo "setup.sh: no $v: start it with \`adapters/vllm/tools/served_run/run.sh setup\`"; exit 2; }
done
IN=${RESEARCH_RUN_DIR:?run under research run}/inputs OUT=$RESEARCH_RUN_DIR/out
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$OUT/job.log"; }
on() { [[ ",${STEPS:-pin,taps,flock,build}," == *",$1,"* ]]; }
TREE=$PWD R=${LLAMA_ROOT%/*} PY=$VENV/bin/python BASE_REPO=unsloth/Llama-3.2-1B BASE_REV=9535bd9b1d1dea6acafbdc4813b728796aeb28da
end=$(date -u -d "$END" +%s) || { log "END is not a time"; exit 2; }
snap() { echo "$1/hub/models--${2//\//--}/snapshots/$3"; }
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$(nproc) HF_HUB_OFFLINE=1 PATH=$VENV/bin:$CUDA_HOME/bin:$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
paths=$TREE:$TREE/adapters/vllm:$TREE/adapters/torch:$TREE/tools/config:$TREE/tools/research/src
vv() { (cd "$TREE/adapters/vllm" && PYTHONPATH=$paths nice -n 5 "$PY" -m verity_vllm.pipeline.cli "$@"); }
mkdir -p "$R" && chmod 700 "$R" || { log "can't make $R"; exit 2; }
research keep "$R" --owner @vllm --expires 3d --priority high --why "the served run's pin, taps, traced Builds and Flock build" 2>&1 | tail -1 | tee -a "$OUT/job.log"
cp "$IN/$ROW_ID.json" adapters/vllm/workloads/ || exit 2
log "setup of $ROW_ID from $(git rev-parse HEAD) on $(hostname), cores $CORES, card $CARD only under its lease"

if on pin && [[ $REPO == verity-int8/* ]] && { [ -n "${STEPS:-}" ] || [ ! -f "$HF_HOME/.made" ]; }; then
  rm -rf "$HF_HOME" && mkdir -p "$(dirname "$(snap "$HF_HOME" $BASE_REPO $BASE_REV)")"
  cp -rL "$(snap /workspace/hf $BASE_REPO $BASE_REV)" "$(snap "$HF_HOME" $BASE_REPO $BASE_REV)" || { log "pin: the base can't be copied"; exit 2; }
  vv int8-checkpoint make --role "$ROLE" --hf-home "$HF_HOME" --jobs 8 > "$OUT/pin.log" 2>&1 && grep -q '"ok": true' "$OUT/pin.log" \
    || { log "pin: NOT made: $(tail -n 1 "$OUT/pin.log" | cut -c1-300)"; exit 3; }
  touch "$HF_HOME/.made"; log "pin: made, every file's sha256 the manifest's"
fi
. adapters/vllm/tools/served_run/pin_link.sh || exit 2
if on taps && { [ -n "${STEPS:-}" ] || [ ! -s "$HIDDEN_SO" ] || [ ! -s "$NORM_TAP_SO" ]; }; then
  (cd adapters/vllm && FA2=${HIDDEN_SO%/build/*} FA3_TAP_ROW_GUARD=1 FA3_TAP_FORCE=1 bash verity_vllm/ops/pod_fa3_tap.sh \
     && NT=${NORM_TAP_SO%/build/*} NORM_TAP_FORCE=1 bash verity_vllm/ops/pod_norm_tap.sh) > "$OUT/taps.log" 2>&1 \
    || { log "taps: NOT built: $(tail -n 1 "$OUT/taps.log" | cut -c1-300)"; exit 3; }
  log "taps: built ($(grep -E '^(FA3|NORM)-TAP-OK' "$OUT/taps.log" | cut -c1-120 | tr '\n' ' ')): set placement.toml's TAPS_AT to $(git rev-parse HEAD)"
fi
if on flock && { [ -n "${STEPS:-}" ] || [ ! -x "$FLOCK_WORK/flock-gpu-link/flock/target/release/flock-circuit" ]; }; then
  mkdir -p "$FLOCK_WORK" "$OUT/m0-build"
  (unset PYTHONPATH; RESEARCH_RUN_DIR=$OUT/m0-build MODE=build GPU=0 flock "$FLOCK_WORK/.build.lock" \
     bash verity/protocols/proofs/flock/prover/pod/60-circuit.sh) > "$R/flock-build.log" 2>&1 \
    || { tail -n 20 "$R/flock-build.log" | cut -c1-300 > "$OUT/flock_tail.txt"; log "flock: NOT built (out/flock_tail.txt)"; exit 3; }
  log "flock: built"
fi
on build || { log "setup: DONE"; exit 0; }
BASE=$LLAMA_ROOT/$RESEARCH_RUN_ID KEPT=$LLAMA_ROOT/$RESEARCH_RUN_ID-unpublished
mkdir -p "$BASE/predicted" "$KEPT" "$R/cache"
(cd "$TREE" && PYTHONPATH=$paths "$PY" -m verity_vllm.pipeline.cli predict row --row "$ROW_ID" --out "$BASE/predicted") \
  > "$OUT/predict_row.json" 2> "$KEPT/predict_row.err" || { log "predict: refused: $(tail -n 2 "$KEPT/predict_row.err" | cut -c1-400)"; exit 3; }
pin=$(python3 - "$TREE/tests/adapters/vllm/tests/program/predict/test_predict.py" "$ROW_ID" "$BASE/predicted/predicted.json" <<'EOF'
import ast, json, sys
src, row, got = sys.argv[1:]
pins = [ast.literal_eval(n.value) for n in ast.parse(open(src).read()).body if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "H200_INT8"]
want = pins[0].get(row, {}).get("workload") if pins else None
g = json.load(open(got))
if want is None:
    sys.exit(print(f"no pin of {row} in the tree to hold program_digest {g.get('program_digest', '')[:16]}… to") or 0)
bad = [k for k in ("program_digest", "workload_digest") if g.get(k) != want]
print(f"{bad} differ from the tree's pin {want}" if bad else f"program_digest and workload_digest are the tree's pin {want[:16]}…")
sys.exit(1 if bad else 0)
EOF
) || { log "predict: $pin"; exit 3; }
log "predict: $pin"
max=$(( (end - $(date -u +%s)) / 60 - 30 )); [ "$max" -gt 30 ] || { log "build: $max min left before END: not started"; exit 75; }
# the row driver's derives get PYTHONPATH without the torch adapter and the settings package (verity_torch is not in the
# venv): its python is a mirror of the venv's bin whose `python` appends them (VERITY_CORE, which the driver sets)
PYW=$R/py/bin
rm -rf "$R/py" && mkdir -p "$PYW" && for f in "$VENV"/bin/*; do ln -s "$f" "$PYW/${f##*/}"; done
rm -f "$PYW/python" "$PYW/python3"
printf '#!/usr/bin/env bash\n[ -z "${VERITY_CORE:-}" ] || export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}$VERITY_CORE/adapters/torch:$VERITY_CORE/tools/config\nexec %s "$@"\n' \
  "$VENV/bin/python" > "$PYW/python" && chmod +x "$PYW/python" && ln -s python "$PYW/python3"
(cd "$TREE/adapters/vllm" && PYTHONPATH=$paths PY=$PYW/python PY312=$PYW/python TORCH_EXTENSIONS_DIR=$R/cache/torch-extensions \
   TRITON_CACHE_DIR=$R/cache/triton VLLM_CACHE_ROOT=$R/cache/vllm NATIVE_COLLECT_BUILD=$R/cache/jit/nc \
   gpu-lease 1 --on "$CARD" --cores "$CORES" --max-min "$max" --mem-gb 128 -- \
   "$PY" -m verity_vllm.pipeline.cli row run "$ROW_ID" "$ROLE" "$REPO" "$REV" --sweep-dir "$BASE" --research-run-dir "$KEPT" \
     --stages build --config-run 1 --replay-k 460 --hidden-so "$HIDDEN_SO" --norm-tap-so "$NORM_TAP_SO") > "$KEPT/build.out" 2>&1
rc=$?
awk '$2 ~ /^(PASS|FAIL|WAIVED|PENDING)$/ {print $1, $2, $3}' "$BASE/$ROW_ID/stages.txt" 2>/dev/null > "$OUT/build_verdicts.txt"
log "build: exit $rc; stages: $(tr '\n' ' ' < "$OUT/build_verdicts.txt")"
[ "$rc" = 0 ] && [ -s "$BASE/$ROW_ID/manifest.json" ] && [ -s "$BASE/$ROW_ID/build_summary.json" ] || { log "build: NOT done"; exit 3; }
config=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["config"])' "$BASE/predicted/predicted.json")
vv predict compare "$BASE/$ROW_ID" --row "$ROW_ID" --config "$config" --predicted "$BASE/predicted" \
  --out "$OUT/predict_compare.json" > "$OUT/predict_compare.log" 2>&1 || { log "compare: NOT equal: $(tail -n 2 "$OUT/predict_compare.log" | cut -c1-400)"; exit 3; }
log "setup: DONE; the traced Build of $ROW_ID passed: name $RESEARCH_RUN_ID for it in placement.toml's builds"
