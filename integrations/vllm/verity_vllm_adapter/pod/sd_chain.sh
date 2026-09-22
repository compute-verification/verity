#!/bin/bash
# Lane SD: the whole record for the b1-spec port, in order, on the pod (GPU runs sequential, correspondence on CPU).
#   production x2 -> C0 compare -> hash-mode reference replays x2 -> commit-mode replay (values) -> compare-refs
#   -> correspondence of every speculative step, both sides -> conformance harness over a harness-layout results dir.
# usage: sd_chain.sh <results root> <adapter revision> [corr parallel=14]
# env: WHEEL (sha256 of the installed vLLM wheel), VLLM_COMMIT
set -u
R=$1; REV=$2; PAR=${3:-14}
WHEEL=${WHEEL:-7aa52ac71500caf74643135a340eb2d03ae0386d4282fc0b23846940029fbbb3}   # the cu129 build of d9105ea80 (this pod)
VLLM_COMMIT=${VLLM_COMMIT:-d9105ea80}
mkdir -p "$R"
cd /workspace/veritor-sd/vllm-poc
source /workspace/venv-cu129/bin/activate
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONPATH=/workspace/veritor-sd/vllm-poc:/workspace/veritor-sd
PY312=/workspace/venv312/bin/python
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$R/chain.log"; }
python -m verity_vllm_adapter.speculative production --out "$R/prod-1" --adapter-revision "$REV" --wheel-sha256 "$WHEEL" --vllm-commit "$VLLM_COMMIT" > "$R/prod-1.log" 2>&1; log "prod-1 rc=$? $(tail -1 "$R/prod-1.log" | cut -c1-160)"
python -m verity_vllm_adapter.speculative production --out "$R/prod-2" --adapter-revision "$REV" --wheel-sha256 "$WHEEL" --vllm-commit "$VLLM_COMMIT" > "$R/prod-2.log" 2>&1; log "prod-2 rc=$? $(tail -1 "$R/prod-2.log" | cut -c1-160)"
python -m verity_vllm_adapter.speculative compare --a "$R/prod-1" --b "$R/prod-2" --out "$R/C0-prod1-vs-prod2.json" > "$R/C0.log" 2>&1; log "C0 rc=$? $(tail -1 "$R/C0.log" | cut -c1-200)"
for n in ref1 ref2; do
  python -m verity_vllm_adapter.speculative replay --production "$R/prod-1" --out "$R/$n" > "$R/$n.log" 2>&1; log "$n rc=$? $(grep DONE "$R/$n.log" | tail -1 | cut -c1-200)"
done
python -m verity_vllm_adapter.speculative replay --production "$R/prod-1" --out "$R/commit" --store-values > "$R/commit.log" 2>&1; log "commit rc=$? $(grep DONE "$R/commit.log" | tail -1 | cut -c1-200)"
$PY312 -m verity_vllm_adapter.speculative compare-refs --commit "$R/commit" --refs "$R/ref1" "$R/ref2" --out "$R/compare.json" > "$R/compare-refs.log" 2>&1; log "compare-refs rc=$? $(tail -1 "$R/compare-refs.log" | cut -c1-200)"
# harness-layout view of the same records
H="$R/harness"; rm -rf "$H"; mkdir -p "$H/correspondence"
ln -s "$R/prod-1" "$H/production"; ln -s "$R/prod-2" "$H/production-r2"; ln -s "$R/commit" "$H/commit"
ln -s "$R/ref1" "$H/ref1"; ln -s "$R/ref2" "$H/ref2"; ln -s "$R/compare.json" "$H/compare.json"
# correspondence, both sides (CPU; the cos_sin tables were built once from the CUDA caches)
mkdir -p "$R/corr"; cp /workspace/results/sd/corr/cos_sin_draft_*.npy "$R/corr/" 2>/dev/null
bash verity_vllm_adapter/pod/sd_corr_all.sh "$R/commit" "$R/corr" "$PAR" "target draft" > "$R/corr-all.out" 2>&1
log "corr $(grep -c 'rc=0' "$R/corr/all.log") ok / $(grep -c 'rc=' "$R/corr/all.log") runs"
cp "$R"/corr/sd-*-s*-target.json "$R"/corr/sd-*-s*-draft.json "$H/correspondence/"
U=$(date -u +%Y%m%dT%H%M%SZ)
$PY312 -m verity_vllm_adapter.conformance run --port b1-spec --out "$R/CONF-b1-spec-$U" --from-results "$H" > "$R/conf.log" 2>&1; log "conformance rc=$? -> $R/CONF-b1-spec-$U"
tail -8 "$R/conf.log" | tee -a "$R/chain.log"
log "CHAIN_DONE"
