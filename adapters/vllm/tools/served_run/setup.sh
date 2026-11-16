#!/usr/bin/env bash
# What a row's window reads, made on its node (`run.sh setup`); a step STEPS names reruns even where its result exists.
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
  S=$R/taps-staged
  rm -rf "$S/exactness" && mkdir -p "$S"
  (cd adapters/vllm && S=$S PY=$PY FA_SRC=${HIDDEN_SO%/build/*}/../vllm-flash-attn TP=$TREE/tests/adapters/vllm:$paths \
     gpu-lease 1 --on "$CARD" --cores "$CORES" --max-min 120 --mem-gb 64 -- bash -euc '
       for g in 0 1; do REPO=$PWD FA2=$S/fa2 FA3_TAP_ROW_GUARD=$g FA3_TAP_FORCE=1 bash verity_vllm/ops/pod_fa3_tap.sh; done
       REPO=$PWD NT=$S/norm_tap NORM_TAP_FORCE=1 bash verity_vllm/ops/pod_norm_tap.sh
       cd ../../tests/adapters/vllm
       PYTHONPATH=$TP "$PY" -m tests.properties.fa_tap_exactness_gpu --so "$S/fa2/build/fa3_matReq/verity_fa3_matReq.so" --out "$S/exactness/fa3"
       PYTHONPATH=$TP "$PY" -m tests.properties.fa_tap_exactness_gpu --so "$S/fa2/build/fa3_matReqG/verity_fa3_matReqG.so" \
         --baseline "$S/exactness/fa3/fa_tap_exactness.json" --construction check-inf-per-iteration --out "$S/exactness/fa3g"') \
    > "$OUT/taps.log" 2>&1 || { log "taps: NOT built or NOT exact, so not live: $(grep -E 'FAIL|Error' "$OUT/taps.log" | tail -n 1 | cut -c1-300)"; exit 3; }
  install -D "$S/fa2/build/fa3_matReqG/verity_fa3_matReqG.so" "$HIDDEN_SO" && install -D "$S/norm_tap/build/verity_norm_tap.so" "$NORM_TAP_SO" || exit 3
  log "taps: built and exact, now live ($(grep -E '^(FA3-TAP-OK|NORM-TAP-OK|FA-TAP-EXACTNESS)' "$OUT/taps.log" | cut -c1-100 | tr '\n' ' ')): set placement.toml's TAPS_AT to $(git rev-parse HEAD)"
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
pin=$(python3 - "$TREE" "$ROW_ID" "$REPO" "$BASE/predicted/predicted.json" <<'EOF'
import ast, json, re, sys
tree, row, repo, got = sys.argv[1:]
g = json.load(open(got))
if repo.startswith("verity-int8/"):
    test = f"{tree}/tests/adapters/vllm/tests/program/predict/test_predict.py"
    pins = [ast.literal_eval(n.value) for n in ast.parse(open(test).read()).body if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "H200_INT8"]
    keys, of = ("program_digest", "workload_digest"), "test_predict.py's H200_INT8 pin"
    want = [(pins[0].get(row) or {}).get("workload") if pins else None] * 2
else:
    keys, of = ("program_digest", "manifest_digest"), "served_verdicts.sh's BUILD_OF_RECORD"
    m = re.search(r"BUILD_OF_RECORD=\$\{BUILD_OF_RECORD:-(\w+):(\w+)\}", open(f"{tree}/benchmarks/one_stage/served_verdicts.sh").read())
    want = list(m.groups()) if m else [None, None]
have = [g.get(k) for k in keys]
print(f"{keys} {':'.join(map(str, have))} are {of}" if have == want else
      f"{keys} {':'.join(map(str, have))} are not {of}, {':'.join(map(str, want))}")
sys.exit(have != want)
EOF
) || { log "predict: $pin"; exit 3; }
log "predict: $pin"
max=$(( (end - $(date -u +%s)) / 60 - 30 )); [ "$max" -gt 30 ] || { log "build: $max min left before END: not started"; exit 75; }
# the row driver's derives need the torch adapter and the settings package, which the venv lacks
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
