#!/usr/bin/env bash
# A single-GPU Commit of record with its sealed replay bundle, the TP1 counterpart of a5_glm53_commit.sh's deferred TP Commit:
# Llama-3.2-1B bf16 batch-invariant eager on one RTX PRO 6000 (ROW, the example's workload, its file in demos/vllm/workloads/, which
# the integration finds through VERITY_WORKLOADS: the B4 I256/O32 rtxpro6000 workload, prompt lengths 151, 8, 256 and 87), its Build and
# Match of record and a deferred config-run Commit in one workload dir, then the Commit's C2 replay on CPU. `verity example vllm commit`
# (run.sh) starts it where placement.toml's [run.commit] says, with the node's VENV, CUDA_HOME, HF_HOME and LLAMA_ROOT and the workload's
# ROW, ROLE, REPO and REV as its environment. Needs on the node: those, the taps of record under /workspace/cp (cli.MACHINE) and
# gpu-lease; the workload leases a GPU only for its Match capture and its Commit's engine (GPU_LEASE, pipeline.gpu_lease).
# A single-GPU workload defers its replay only as a config run (row_stages.SingleRow.commit_cmd), and leaves one bundle,
# commit/replay_bundle_p0/, named in commit/verdict.json's replay_deferred (a TP Commit's are replay_bundle_p0_rank<r>/ and
# replay_deferred.json). The CPU replay deletes a full bundle once it leaves its slim keep (row_config.replay), so the sealed bundle
# is hard-linked aside first and put back after it. The workload dir, bundle included, goes to the evidence store as one tree
# (STORED in $OUT/store.txt), put with the run's own custody key, and stays on the node under `research keep`; $OUT keeps the
# records and logs. STORE_ROW=DIR runs none of the phases and only stores an earlier run's workload dir DIR.
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
for v in ROW ROLE REPO REV VENV CUDA_HOME HF_HOME LLAMA_ROOT; do
  [ -n "${!v:-}" ] || { echo "commit.sh: no $v: start it with \`verity example vllm commit\` (demos/vllm/run.sh)"; exit 2; }
done
TREE=$PWD
ROOT=$LLAMA_ROOT
BASE=$ROOT/${RESEARCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
D=$BASE/$ROW
mkdir -p "$D" "$ROOT/cache" "$BASE/sealed"
export PY=$VENV/bin/python PY312=$VENV/bin/python HF_HOME HF_HUB_OFFLINE=1
export CUDA_HOME TORCH_EXTENSIONS_DIR=$ROOT/cache/torch-extensions
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export PYTHONPATH=$TREE:$TREE/integrations/vllm:$TREE/tools/research/src VERITY_WORKLOADS=$TREE/demos/vllm/workloads
export GPU_LEASE=${GPU_LEASE:-gpu-lease} GPU_LEASE_WHO=${GPU_LEASE_WHO:-vllm-llama1b-commit}
export TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm
if grep -q staged_sources "$TREE/integrations/vllm/verity_vllm/commit/committer/native_jit.py"; then
  export HIDDEN_GPU_BUILD=$ROOT/cache/jit/hidden_gpu NATIVE_COLLECT_BUILD=$ROOT/cache/jit/nc
fi
rs() { "$PY" -m research "$@" 2>&1 || research "$@" 2>&1; }
{ id; hostname; git -C "$TREE" rev-parse HEAD 2>/dev/null; echo "row dir $D"; nvidia-smi -L; } | tee "$OUT/machine.txt"
rs keep "$ROOT" --owner @vllm --expires 14d --priority high \
  --why "PROVE-PATH: Llama-3.2-1B TP1 Commits of record with their sealed replay bundles, Builds and Matches" | tail -2

row() {  # NAME ARGS...: one `verity-vllm row run` of the workload dir, its output in $OUT/NAME.out
  local name=$1; shift
  echo "$(date -u +%H:%M:%SZ) $name: $*" | tee -a "$OUT/steps.txt"
  (cd "$TREE/integrations/vllm" && "$PY" -m verity_vllm.pipeline.cli row run "$ROW" "$ROLE" "$REPO" "$REV" --sweep-dir "$BASE" "$@") \
    > "$OUT/$name.out" 2>&1
  local rc=$?
  echo "$(date -u +%H:%M:%SZ) $name rc=$rc" | tee -a "$OUT/steps.txt"
  tail -5 "$OUT/$name.out"
  return $rc
}
records() {  # the workload's records into $OUT (never the bundle, the store or the capture)
  for f in stages.txt row.log verdict.json build_summary.json manifest.json config_record.json strict_word.json target_family.json \
    gpu_lease.json effective_config.json commit.log match.log build.log replay.log manifest.log; do cp "$D/$f" "$OUT/" 2>/dev/null; done
  mkdir -p "$OUT/commit" "$OUT/match"
  find "$D/commit" -maxdepth 1 -type f -size -50M -exec cp {} "$OUT/commit/" \; 2>/dev/null
  find "$D/match" -maxdepth 1 -type f -size -50M -exec cp {} "$OUT/match/" \; 2>/dev/null
  for b in "$D"/commit/replay_bundle_p*/; do
    [ -d "$b" ] && mkdir -p "$OUT/commit/$(basename "$b")" && cp "$b/MANIFEST.json" "$b/context.json" "$OUT/commit/$(basename "$b")/" 2>/dev/null
  done
  du -sh "$D" "$D"/* 2>/dev/null > "$OUT/sizes.txt"
}
stop() { records; echo "STOPPED at $1"; exit "$2"; }

put() {  # the workload dir as one tree, with this run's custody key (a node's job holds no other store credential), then its keep
  echo "$(date -u +%H:%M:%SZ) store: put $D ($(du -sh "$D" | cut -f1))" | tee -a "$OUT/steps.txt"
  local C i art
  C=$(dirname "$(dirname "${RESEARCH_RUN_DIR:?}")")/requests/${RESEARCH_RUN_ID:?}/custody
  for i in 1 2 3; do
    ( if [ -f "$C/cred.json" ]; then
        export RESEARCH_STORE_CONFIG=$C/store.toml
        eval "$("$PY" -m verity_vllm.pipeline.cli epoch custody-env "$C")"
      fi
      rs data put --kind evidence/v1 --tree "$D" --preserve --json --meta "{\"row\": \"$ROW\", \"commit_run\": \"$(basename "$(dirname "$D")")\",
        \"what\": \"Llama-3.2-1B TP1 Commit of record (config run, deferred): Build, Match, commit/ with the sealed replay_bundle_p0\"}"
    ) > "$OUT/put.json" && break
    sleep $((30 * i))
  done
  art=$(grep -o 'art:[0-9a-f]\{64\}' "$OUT/put.json" | head -1)
  echo "STORED ${art:-none} $D" | tee "$OUT/store.txt"
  [ -n "$art" ] || return 1
  rs keep "$(dirname "$D")" --owner @vllm --expires 14d --priority high --art "$art" \
    --why "PROVE-PATH: the Llama-3.2-1B TP1 Commit of record and its sealed bundle, stored as $art" | tail -2
}
if [ -n "${STORE_ROW:-}" ]; then   # an earlier run's workload dir, stored as is
  D=$STORE_ROW; rmdir "$BASE/sealed" "$BASE/$ROW" "$BASE" 2>/dev/null
  [ -s "$D/commit/replay_bundle_p0/MANIFEST.json" ] || { echo "no sealed bundle under $D/commit"; exit 2; }
  records; put; exit
fi

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
put; pr=$?
tail -8 "$OUT/stages.txt" 2>/dev/null
exit $(( rr != 0 ? 16 : pr != 0 ? 17 : 0 ))
