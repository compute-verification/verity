#!/usr/bin/env bash
# A single-GPU Commit of record with its sealed replay bundle, the TP1 counterpart of a5_glm53_commit.sh's deferred TP Commit:
# Llama-3.2-1B bf16 batch-invariant eager on one RTX PRO 6000 (ROW, the demo's workload, its file in demos/vllm/workloads/, which
# the integration finds through VERITY_WORKLOADS: the B4 I256/O32 rtxpro6000 workload, prompt lengths 151, 8, 256 and 87), its Build and
# Match of record and a deferred config-run Commit in one workload dir, then the Commit's C2 replay on CPU. `verity demo vllm commit`
# (run.sh) starts it where placement.toml's [run.commit] says, with the node's VENV, CUDA_HOME, HF_HOME and LLAMA_ROOT and the workload's
# ROW, ROLE, REPO and REV as its environment. Needs on the node: those, the taps of record under /workspace/cp (cli.MACHINE) and
# gpu-lease; the workload leases a GPU only for its Match capture and its Commit's engine (GPU_LEASE, pipeline.gpu_lease).
# A single-GPU workload defers its replay only as a config run (row_stages.SingleRow.commit_cmd), and leaves one bundle,
# commit/replay_bundle_p0/, named in commit/verdict.json's replay_deferred (a TP Commit's are replay_bundle_p0_rank<r>/ and
# replay_deferred.json). The CPU replay deletes a full bundle once it leaves its slim keep (row_config.replay), so the sealed bundle
# is hard-linked aside first and put back after it. The workload dir stays on the node under `research keep`, the only copy of
# its bundle and its slim keep, and nothing stores it. $OUT is published as the run's record, so it holds only the files
# published.sh lists, each with the reason it holds nothing of a request: this script's own machine.txt, steps.txt,
# verdicts.txt and bundle.sha256, and the row's records listed there. Every other file stays on the node and the run names
# it: the row dir's (match/spec.json holds the prompts, commit/ the token record, the bundle and the keep; stages.txt, row.log
# and the stages' logs can quote a served token on a FAIL), and under KEPT, beside the row dir, each stage's output, the
# row dir's sizes and what `row run` writes for a research run (result.json and its evidence/ copies of the row's records:
# the stages get KEPT as their run dir, never the published one).
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
. "$(dirname "${BASH_SOURCE[0]}")/published.sh" || exit 2
for v in ROW ROLE REPO REV VENV CUDA_HOME HF_HOME LLAMA_ROOT; do
  [ -n "${!v:-}" ] || { echo "commit.sh: no $v: start it with \`verity demo vllm commit\` (demos/vllm/run.sh)"; exit 2; }
done
TREE=$PWD
ROOT=$LLAMA_ROOT
BASE=$ROOT/${RESEARCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
D=$BASE/$ROW KEPT=$BASE/unpublished
mkdir -p "$D" "$ROOT/cache" "$BASE/sealed" "$KEPT"
export PY=$VENV/bin/python PY312=$VENV/bin/python HF_HOME HF_HUB_OFFLINE=1
export CUDA_HOME TORCH_EXTENSIONS_DIR=$ROOT/cache/torch-extensions
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export PYTHONPATH=$TREE:$TREE/adapters/vllm:$TREE/tools/research/src VERITY_WORKLOADS=$TREE/demos/vllm/workloads
export GPU_LEASE=${GPU_LEASE:-gpu-lease} GPU_LEASE_WHO=${GPU_LEASE_WHO:-vllm-llama1b-commit}
export TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm
if grep -q staged_sources "$TREE/adapters/vllm/verity_vllm/commit/committer/native_jit.py"; then
  export NATIVE_COLLECT_BUILD=$ROOT/cache/jit/nc
fi
rs() { "$PY" -m research "$@" 2>&1 || research "$@" 2>&1; }
{ id; hostname; git -C "$TREE" rev-parse HEAD 2>/dev/null; echo "row dir $D"; nvidia-smi -L; } | tee "$OUT/machine.txt"
rs keep "$ROOT" --owner @vllm --expires 14d --priority high \
  --why "PROVE-PATH: Llama-3.2-1B TP1 Commits of record with their sealed replay bundles, Builds and Matches" | tail -2

row() {  # NAME ARGS...: one `verity-vllm row run` of the workload dir, its output in KEPT/NAME.out; prints its stages' verdict words
  local name=$1; shift
  echo "$(date -u +%H:%M:%SZ) $name: $*" | tee -a "$OUT/steps.txt"
  (cd "$TREE/adapters/vllm" && "$PY" -m verity_vllm.pipeline.cli row run "$ROW" "$ROLE" "$REPO" "$REV" --sweep-dir "$BASE" \
     --research-run-dir "$KEPT" "$@") > "$KEPT/$name.out" 2>&1
  local rc=$?
  echo "$(date -u +%H:%M:%SZ) $name rc=$rc" | tee -a "$OUT/steps.txt"
  verdicts "$D/stages.txt" "$OUT"
  return $rc
}
records() {  # the workload's records into $OUT, which the run publishes: only the files published.sh lists, and $OUT holds no other
  publish "$D" "$OUT"
  verdicts "$D/stages.txt" "$OUT" > /dev/null
  du -sh "$D" "$D"/* 2>/dev/null > "$KEPT/sizes.txt"
  hold "$OUT" "$KEPT/out"
}
stop() { records; echo "STOPPED at $1"; exit "$2"; }

unset CONFIG_RUN REPLAY_K REPLAY_DEFERRED
row build --stages build --config-run 1 --replay-k 460 || stop build 10
row match --stages match --config-run 0 || stop match 11
row commit --stages commit --config-run 1 --replay-k 460 --replay-deferred 1 || stop commit 12
B=$D/commit/replay_bundle_p0
[ -s "$B/MANIFEST.json" ] || stop "commit (no sealed bundle $B)" 13
cp -al "$B" "$BASE/sealed/" || stop "linking the sealed bundle aside" 14
row replay --stages replay --config-run 1 --replay-k 460; rr=$?
[ -d "$B" ] || cp -al "$BASE/sealed/replay_bundle_p0" "$D/commit/" || stop "restoring the sealed bundle" 15
(cd "$B" && sha256sum MANIFEST.json) | tee "$OUT/bundle.sha256"
records
exit $(( rr != 0 ? 16 : 0 ))
