#!/usr/bin/env bash
# The two Commit negatives on an H100 / FlashAttention-3 workload, run with the SAME commit_delta command line as the
# workload's positive Commit (row_pod.sh) minus/plus one thing, against the workload's own required-value manifest:
#
#   omitted   no --hidden-so: the FA3 M1 producer is never attached -> exit 3.  With the workload's manifest the identity-coverage
#             check fires first and names the unbound FA3 identities ("identity coverage: 2048 of 27427 required identities have no
#             binding (first: step 0 fa3.m1/L0/g8x4096x8x128x64 ...)"); without a manifest (canary.sh) the required-class check fires
#             ("required class 'fa3_hidden_m1_stream' not declared").  Either is the named fail-closed refusal; both are accepted.
#   lateread  --late-read qkv_proj (the collector reads the module output one step late = a wrong-buffer read): openings still verify
#             (binding != correctness) but the run root MUST differ from the positive run's root (and from the known-good root).
#
#   cd <tree> && bash verity_vllm/ops/fa3_row_negatives.sh <row-id> <ROLE> [<positive commit dir>]
#   -> <SWEEP_DIR>/<row>/negatives/{neg_omitted,neg_lateread}/ + negatives.json ; last line NEGATIVES PASS|FAIL <detail>
# Env as row_pod.sh: PY, HIDDEN_SO (default = the FA-version tap of TargetProfile.from_device(), as row_pod), GPU_UTIL (0.5), SWEEP_DIR.
set -uo pipefail
ROW=${1:?row id}; ROLE=${2:?case role, e.g. LLAMA32_1B}
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}; cd "$REPO"
D=${SWEEP_DIR:-/workspace/cp/sweep}/$ROW; POS=${3:-$D/commit}; OUT=$D/negatives; mkdir -p "$OUT"
WL=workloads/$ROW.json; MAN=$D/manifest.json
PY=${PY:-/workspace/venv-cu129/bin/python}; export PATH=$(dirname "$PY"):${CUDA_HOME:-/usr/local/cuda}/bin:$PATH
export PYTHONPATH=$REPO HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1 TOKENIZERS_PARALLELISM=false
export VERITY_TREE=$REPO NATIVE_COLLECT_BUILD=${NATIVE_COLLECT_BUILD:-/workspace/cp/nc_build}
GPU_UTIL=${GPU_UTIL:-0.5}
FA_VER=$($PY -c "from verity_vllm.program.frontend.target_profile import TargetProfile; print(TargetProfile.from_device().effective_flash_attn_version())" 2>/dev/null || echo 2)
G=$([ "${GUARDED_MAX_TAP:-1}" = 1 ] && echo G)   # the guarded-max build unless GUARDED_MAX_TAP=0 (config.CommitConfig), as a workload selects it
HIDDEN_SO=${HIDDEN_SO:-$([ "$FA_VER" = 3 ] && echo /workspace/cp/fa2/build/fa3_matReq$G/verity_fa3_matReq$G.so || echo /workspace/cp/fa2/build/matReq$G/verity_fa2_matReq$G.so)}
CLASS=$([ "$FA_VER" = 3 ] && echo fa3_hidden_m1_stream || echo fa2_hidden_m1_stream)
"${PY%/*}/verity-vllm" workload-file "$WL" > /dev/null || { echo "NEGATIVES FAIL workload $WL not fetched or not its digest"; exit 2; }
B=$(python3 -c "import json;print(json.load(open('$WL'))['sweep']['batch'])")
MML=$(python3 -c "import json;w=json.load(open('$WL'));s=w['sweep'];print(s['max_input']+s['max_output'])")
log() { echo "$(date -u +%FT%TZ) [neg $ROW] $*" | tee -a "$OUT/negatives.log"; }
[ -f "$WL" ] || { log "no workload $WL"; echo "NEGATIVES FAIL no workload"; exit 2; }
[ -f "$MAN" ] || { log "no required-value manifest $MAN (run the positive Commit first: row_pod builds it)"; echo "NEGATIVES FAIL no manifest"; exit 2; }
POS_ROOT=$(python3 -c "import json;v=json.load(open('$POS/verdict.json'));print((v.get('run_roots') or [''])[0])" 2>/dev/null)
log "start fa=$FA_VER hidden_so=$HIDDEN_SO class=$CLASS B=$B manifest=$MAN positive_root=${POS_ROOT:0:16}"
BASE=("${PY%/*}/verity-vllm" commit --case "$ROLE" --workload "$WL" --committer native_collect_v2b --openings 64
      --pairs 1 --warmup 1 --max-num-seqs "$B" --engine-arg "gpu_memory_utilization=$GPU_UTIL" --engine-arg "max_model_len=$MML"
      --window-mb 256 --window-slots 8 --retain host --open-after-release --required-manifest "$MAN")
reason_of() { python3 -c "import json;print(json.load(open('$1')).get('first_fail_reason') or '')" 2>/dev/null; }
root_of() { python3 -c "import json,sys;v=json.load(open('$1'));print((v.get('run_roots') or [''])[0])" 2>/dev/null; }
R=()
# ---- omitted producer
d=$OUT/neg_omitted; rm -rf "$d"; t0=$(date +%s)
timeout 2400 "${BASE[@]}" --label "${ROW}_neg_omitted" --out "$d" > "$d.log" 2>&1; rc=$?
reason=$(reason_of "$d/verdict.json")
TAP_PATH=$([ "$FA_VER" = 3 ] && echo fa3.m1/ || echo fa2.m1/)
if [ $rc -eq 3 ] && [[ "$reason" =~ required\ class\ \'$CLASS\'\ not\ declared|$CLASS.*never\ fired|identity\ coverage:.*$TAP_PATH ]]; then R+=("omitted|PASS|exit 3 in $(( $(date +%s)-t0 ))s; reason: ${reason:0:220}")
else R+=("omitted|FAIL|expected exit 3 naming $CLASS or unbound $TAP_PATH identities, got rc=$rc reason='${reason:0:200}' $(grep -m1 -E 'Error|Traceback' "$d.log" | cut -c1-120)"); fi
log "${R[-1]}"
# ---- late read
d=$OUT/neg_lateread; rm -rf "$d"; t0=$(date +%s)
timeout 2400 "${BASE[@]}" --hidden-so "$HIDDEN_SO" --late-read qkv_proj --label "${ROW}_neg_lateread" --out "$d" > "$d.log" 2>&1; rc=$?
root=$(root_of "$d/verdict.json"); reason=$(reason_of "$d/verdict.json")
if [ -z "$root" ]; then R+=("lateread|FAIL|no run root (rc=$rc): ${reason:0:160} $(grep -m1 -E 'Error|Traceback|COMMIT FAIL' "$d.log" | cut -c1-160)")
elif [ -n "$POS_ROOT" ] && [ "$root" = "$POS_ROOT" ]; then R+=("lateread|FAIL|late read produced the positive run's root $root: a wrong-buffer read is not detected")
else R+=("lateread|PASS|root ${root:0:16} != positive ${POS_ROOT:0:16} in $(( $(date +%s)-t0 ))s (rc=$rc${reason:+; verdict: ${reason:0:120}})"); fi
log "${R[-1]}"
python3 - "$OUT/negatives.json" "$ROW" "$FA_VER" "$HIDDEN_SO" "$CLASS" "$MAN" "$POS_ROOT" "${R[@]}" <<'EOF'
import json, sys, time
out, row, fa, so, cls, man, pos = sys.argv[1:8]; checks = [dict(zip(("name", "result", "detail"), r.split("|", 2))) for r in sys.argv[8:]]
json.dump({"schema": "verity-sweep/row-negatives/v1", "row_id": row, "flash_attn_version": int(fa), "hidden_so": so, "required_class": cls,
           "required_manifest": man, "positive_run_root": pos, "green": all(c["result"] == "PASS" for c in checks), "checks": checks,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(out, "w"), indent=1)
EOF
if printf '%s\n' "${R[@]}" | grep -q "|FAIL|"; then echo "NEGATIVES FAIL $(printf '%s; ' "${R[@]}")"; exit 1; fi
echo "NEGATIVES PASS $(printf '%s; ' "${R[@]}")"
