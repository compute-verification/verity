#!/usr/bin/env bash
# [E8 NIGHT 2026-09-12] = E10's record_v5/run_tiny1_pod.sh (eod/e10 HEAD 0d54057b, verbatim logic incl. REQUIRED_FROM_RECORD / DURABLE_DIR / VERITOR_PROVER_TIME)
# with the run id / config / record dir taken from the environment (RUN_ID, CFG, REC) so the same runner drives the one-request SmolLM2-135M epochs
# at every prompt x decode of the night table (configs: out/night/e8/configs/<RUN_ID>.json).  E10's file is not modified.  Regenerated from E10's HEAD
# by three substitutions: the CFG/REC lines, the `.out` tee name, the log tag.
# [E10 EOD 2026-09-11] The one-request TINY record (V5-TINY1-bex0-inc) on the RTX 4090 pod (INTEGRATOR D7), attempt N -- everything attempt 6
# did (record_v5/run_v5.sh, all phases, one protocol transcript) plus the EOD pieces:
#   * the instance is E6's pinned protocol_instance.json (PINNED_INSTANCE; cross-checked, copied verbatim),
#   * the population file is E6's a9/legacy-minus-inputs/v2 (POP),
#   * the interior trees of J are E2's layout over E1's interpreter rows (VERITOR_INTERIOR_VALUES = `e2e.interior.tree stage` of E1's pod output).
# Usage on the pod (after record_v5/ship.sh -> /workspace/a9/veritor and record_v5/pod_bootstrap_a9.sh):
#   bash /workspace/a9/veritor/record_v5/run_tiny1_pod.sh <attempt-tag> [phases...]
# Inputs expected on the pod (E10 puts them there; each is logged with its sha256 into $REC/../inputs-<tag>.sha256):
#   /workspace/a9/pop/population.v5.json            E6 v2 population (built with e2e.partitions.legacy_minus_inputs_v2 from-v5)
#   /workspace/a9/pinned/protocol_instance.json     E6's pinned instance (out/eod/e6/tiny/protocol_instance.json)
#   /workspace/a9/e1/tiny-ru0-4090/                 E1's producer output for request 2663944e (inventory.json, required_gates.bin, interior/...)
set -uo pipefail
TAG=${1:?attempt tag, e.g. a7}; shift 1
# REPO: the frozen source tree the run hashes (freeze.py). Since attempt 7 (00:24Z: another agent wrote a file into the shared
# /workspace/a9/veritor DURING the run -> protocol-end freeze check E_SOURCE_CHANGED, run aborted) every run uses a RUN-PRIVATE copy:
#   cp -a /workspace/a9/veritor /workspace/a9/run-<tag>-veritor && REPO=/workspace/a9/run-<tag>-veritor bash run_tiny1_pod.sh <tag>
export REPO=${REPO:-/workspace/a9/veritor}
export PY=$REPO/.venv/bin/python PY_VLLM=/workspace/venv-cu129/bin/python
export HOST=/workspace/a9/veritor-zk-host
export VOCAB_DIGEST=33ee156d86a1bb72a5f0918413baa9295b15a3d105553f553133007fc1d00623
export WEIGHTS_AUTHORIZED=/workspace/a9/weights/weights_authorized.json
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$REPO:$REPO/vllm-poc
# every hashed source root must resolve INSIDE $REPO (the venv's editable veritor.pth points at the tree it was built in -- a copied tree
# must rewrite it: sed -i "s#<orig>/src#$REPO/src#" $REPO/.venv/lib/python3.*/site-packages/veritor.pth)
for m in veritor e2e.verifier_v3.driver record_v5.freeze verity_vllm_adapter; do
  f=$($PY -c "import importlib,sys;print(importlib.import_module(sys.argv[1]).__file__)" "$m" 2>/dev/null) || { echo "[run_night] FATAL: cannot import $m"; exit 3; }
  case "$f" in "$REPO"/*) ;; *) echo "[run_night] FATAL: $m resolves to $f, outside the frozen tree $REPO"; exit 3;; esac
done
GPU=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)
export VERITOR_HARDWARE=${VERITOR_HARDWARE:-"$GPU (RunPod secure pod $(hostname), $(nproc) vCPU; EOD TINY one-request $TAG)"}
export PINNED_INSTANCE=${PINNED_INSTANCE-/workspace/a9/pinned/protocol_instance.json}   # PINNED_INSTANCE= (empty) -> generated instance (preparatory runs only)
[ -n "$PINNED_INSTANCE" ] && [ ! -f "$PINNED_INSTANCE" ] && { echo "[run_night] WARNING: no pinned instance at $PINNED_INSTANCE -- generating one from the config (NOT the owner's pin; preparatory run only)"; export PINNED_INSTANCE=; }
POP=${POP:-/workspace/a9/pop/population.v5.json}
RUN_ID=${RUN_ID:?run id, e.g. V5-N32x16-1req-bex0-inc}; CFG=${CFG:-$REPO/out/night/e8/configs/$RUN_ID.json}
REC=${REC:-/workspace/a9/rec/$RUN_ID-$TAG}
# [E10 NIGHT 2026-09-12] durable mirror: with DURABLE_DIR set (the network volume, e.g. /vol/a9/rec) a background loop rsyncs the record
# directory (everything but the staged rows) there every 120 s and once more at the end, so a pod loss (00:55Z incident) cannot take the
# record with it; the laptop side still runs record_v5/pull_checkpoints.sh for the small files.
if [ -n "${DURABLE_DIR:-}" ]; then
  mkdir -p "$DURABLE_DIR"
  ( while :; do sleep 120; rsync -a --delete "$REC/" "$DURABLE_DIR/$(basename "$REC")/" 2>/dev/null; cp "$REC/../$RUN_ID-$TAG.out" "$DURABLE_DIR/" 2>/dev/null; done ) &
  MIRROR_PID=$!
  trap 'kill $MIRROR_PID 2>/dev/null; rsync -a --delete "$REC/" "$DURABLE_DIR/$(basename "$REC")/" 2>/dev/null; cp "$REC/../$RUN_ID-$TAG.out" "$DURABLE_DIR/" 2>/dev/null; echo "[run_night] durable mirror final -> $DURABLE_DIR/$(basename "$REC") ($(date -u +%T))"' EXIT
fi
E1OUT=${E1OUT:-/workspace/a9/e1/tiny-ru0-4090}
# STAGED is run-independent (symlinks + decompressed copies of E1's rows, ~7.5 GB): reuse one across attempts (STAGED=... ; the pod disk hit
# 100 % at 00:04Z with three staging copies); a complete staging carries .staged-ok and is not rebuilt.
STAGED=${STAGED:-/workspace/a9/interior-values-$TAG}
# PROVE_ANYWAY=1 as attempt 6: prove the covered statements even if uncovered VUs were selected (the verifier still decides at step 5)
export PROVE_ANYWAY=${PROVE_ANYWAY:-1}
mkdir -p "$REC/logs" "$(dirname "$STAGED")"
{
  echo "# inputs of $TAG $(date -u +%FT%TZ)"
  sha256sum "$POP" "$PINNED_INSTANCE" "$WEIGHTS_AUTHORIZED" "$HOST" "$CFG" 2>/dev/null
  [ -f "$E1OUT/inventory.json" ] && sha256sum "$E1OUT/inventory.json" "$E1OUT/required_gates.bin"
  cat "$REPO/EXPORT.json"
} | tee "$REC/../inputs-$TAG.sha256"
# E1 -> E2 staging: <STAGED>/<req>/<step>/*.jsonl + calls.json + required.json (symlinks into E1's output; the prover reads them in phase 2)
if [ -f "$E1OUT/inventory.json" ]; then
  if [ -f "$STAGED/.staged-ok" ]; then
    echo "[run_night] reusing staged interior values at $STAGED ($(cat "$STAGED/.staged-ok"))" | tee "$REC/logs/interior-stage.log"
  else
  rm -rf "$STAGED"   # a partial staging (e.g. disk full) is never reused
  $PY -m e2e.interior.tree stage --e1 "$E1OUT" --values "$STAGED" 2>&1 | tee "$REC/logs/interior-stage.log"
  # E1's producer writes `*.jsonl.gz` (--gzip, ~10x); E2's reader (tree.py ru_dirs / iter_rows_dir) globs `*.jsonl` only -> without this step
  # the prover commits NO interior tree (observed 22:15Z on the CPU pod: {"committed": []}). The staging dir is E10's: decompress the symlinked
  # rows into plain copies here (E1's and E2's files untouched); reported to E2 (accept .jsonl.gz) in out/eod/reports/e10.md.
  NGZ=$(find "$STAGED" -name '*.jsonl.gz' | wc -l)
  if [ "$NGZ" -gt 0 ]; then
    t0=$(date +%s); find "$STAGED" -name '*.jsonl.gz' -print0 | xargs -0 -P 4 -I{} sh -c 'gzip -dc "$1" > "${1%.gz}" && rm "$1"' _ {} || { echo "[run_night] FATAL: decompression failed (disk?)"; df -h "$STAGED"; exit 3; }
    echo "[run_night] decompressed $NGZ .jsonl.gz row files into $STAGED in $(( $(date +%s)-t0 ))s ($(du -sh "$STAGED" | cut -f1))" | tee -a "$REC/logs/interior-stage.log"
  fi
  [ "$(find "$STAGED" -name '*.jsonl.gz' | wc -l)" -eq 0 ] || { echo "[run_night] FATAL: .jsonl.gz left in $STAGED"; exit 3; }
  echo "staged $(date -u +%FT%TZ) from $E1OUT ($(find "$STAGED" -name '*.jsonl' | wc -l) row files)" > "$STAGED/.staged-ok"
  fi
  export VERITOR_INTERIOR_VALUES=$STAGED
  echo "[run_night] interior values staged at $STAGED: $(find "$STAGED" -name '*.jsonl' | wc -l) row files"
  # v2 population: the prover must frame the VERIFIER's required set (record_v5/interior_required_v2.py re-stages required.json per RU from
  # steps.required_source_live; E1's v1-inventory list is kept as required.e1.json). Provenance copied into the record. Without it, under a
  # v2 population, step 4 refuses INTERIOR_WIDTH_MISMATCH (diag3 00:14Z).
  if [ -f "$STAGED/.required-v2.json" ]; then
    cp "$STAGED/.required-v2.json" "$REC/logs/interior-required-v2.json"
    echo "[run_night] required.json per RU = the verifier's live required layout: $($PY -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['totals'], '|', d['source'][:160])" "$STAGED/.required-v2.json")"
  else
    echo "[run_night] NOTE: required.json per RU is E1's inventory list (v1 rows); under a v2 population run record_v5/interior_required_v2.py over a record with commitments first"
  fi
  # class V (value production): E1's producer ran on the CPU pod; its /usr/bin/time measurement is recorded next to E1's output as
  # E10-value-production.json (E10 writes it from the pod's `.time` file) -> costlog_v5 reads $REC/logs/value-production.json
  [ -f "$E1OUT/E10-value-production.json" ] && cp "$E1OUT/E10-value-production.json" "$REC/logs/value-production.json"
else
  echo "[run_night] WARNING: no E1 output at $E1OUT -- the prover runs WITHOUT --interior-values (capture-only interior trees; V3.interior_layout / V8.interior_regions decide)"
fi
echo "[run_night] $(date -u +%FT%TZ) tag=$TAG rec=$REC hw=$VERITOR_HARDWARE phases=${*:-all}"
ALL="freeze instance coverage serve protocol nonfinite obligations prove verify costlog report"
PH=${*:-$ALL}
# [E10 NIGHT 2026-09-12] the prover's peak RSS is measured (protocol_v5 wraps it in /usr/bin/time -v; costlog_v5 reads logs/prover.time)
export VERITOR_PROVER_TIME=${VERITOR_PROVER_TIME:-$REC/logs/prover.time}
# [E10 NIGHT 2026-09-12] required.json per RU = the VERIFIER's live required layout OF THIS TREE (n0dry 06:01Z: INTERIOR_WIDTH_MISMATCH when the staged rows
# carry E1's v1 list; the layout can change with the tree).  REQUIRED_FROM_RECORD names any earlier record of the same request with commitments
# (boundary id lists); the derivation runs from $REPO (this run's frozen tree) before the protocol phase; provenance in logs/interior-required-v2.json.
if [ -n "${REQUIRED_FROM_RECORD:-}" ] && [[ " $PH " == *" protocol "* ]] && [ -n "${VERITOR_INTERIOR_VALUES:-}" ]; then
  mkdir -p "$REC/logs"
  $PY -m record_v5.interior_required_v2 --record "$REQUIRED_FROM_RECORD" --values "$STAGED" 2>&1 | tee "$REC/logs/interior-required-v2.log" | tail -3
  # [E8 D40 T-3] FATAL if the derivation did not write: n2 (09:56Z) fell through to E1's v1 inventory list here (LazyInteriorLayout had no
  # _all_gates) and the verifier refused INTERIOR_WIDTH_MISMATCH at step 4 -- a run without the verifier's required set is not started.
  [ -f "$STAGED/.required-v2.json" ] || { echo "[run_night] FATAL: record_v5.interior_required_v2 wrote no $STAGED/.required-v2.json (see $REC/logs/interior-required-v2.log); the prover would commit E1's list, not the verifier's required set"; exit 3; }
  cp "$STAGED/.required-v2.json" "$REC/logs/interior-required-v2.json"
  echo "[run_night] required.json per RU re-derived from $REPO over $REQUIRED_FROM_RECORD: $($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['totals'])" "$STAGED/.required-v2.json")"
fi
# [E10 NIGHT 2026-09-12] the serve and protocol phases may run in SEPARATE invocations (serve now, protocol after the census reads 0):
# the interior-calls step runs whenever `protocol` is requested and the record's serve output (adapter/production/run.json) exists.
if [ -n "${VERITOR_INTERIOR_VALUES:-}" ] && [[ " $PH " == *" protocol "* ]] && { [[ " $PH " == *" serve "* ]] || [ -f "$REC/adapter/production/run.json" ]; }; then
  # The interior leaf ids are `p=<profile_id>/r=<req>/call=<c>/chunk=<k>` and the verifier derives <profile_id> from THIS record's commitments
  # (e2e/verifier_v3/expected.py: prefix = f"p={self.profile_id}/r={rid}/").  E2's stage_e1 copies the prefix of E1's inventory -- the profile
  # id of the record E1 interpreted (04137cde: p=65d138443752), and profile_id = sha256(production manifest) changes with the pod / driver /
  # wheel path.  So: serve first, read the record's profile_id, and hand the prover the pinned call order with the RECORD's prefix
  # (--interior-calls via VERITOR_INTERIOR_CALLS); the calls and gate ranges are E1's inventory verbatim (E2's calls.json).
  if [[ "$PH" == protocol* ]]; then PRE=""; POST="$PH"; else PRE=${PH%% protocol*}; POST="protocol${PH#* protocol}"; fi
  [ -n "$PRE" ] && bash "$REPO/record_v5/run_v5.sh" "$CFG" "$REC" "$POP" $PRE 2>&1 | tee -a "$REC/../$RUN_ID-$TAG.out"
  CALLS=$REC/logs/interior-calls.json
  $PY - "$STAGED" "$REC/adapter/production/run.json" "$CALLS" <<'PYEOF' 2>&1 | tee -a "$REC/logs/interior-stage.log"
import json, sys, pathlib
staged, run_json, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
pid = json.loads(run_json.read_text())["profile_id"]
spec, e1_prefixes = {}, set()
for cj in sorted(staged.glob("*/*/calls.json")):
    req, step = cj.parent.parent.name, cj.parent.name
    doc = json.loads(cj.read_text())
    e1_prefixes.add(doc.get("prefix", ""))
    spec[f"{req}/{step}"] = {"calls": doc["calls"], "prefix": f"p={pid}/r={req}/", "source": f"{doc.get('source', 'calls.json')} (calls verbatim); prefix = this record's profile_id"}
out.write_text(json.dumps(spec, indent=1) + "\n")
print(f"[run_night] interior calls for {len(spec)} RU trees -> {out}: prefix p={pid}/r=<req>/ (E1's staged prefix was {sorted(e1_prefixes)})")
PYEOF
  export VERITOR_INTERIOR_CALLS=$CALLS
  bash "$REPO/record_v5/run_v5.sh" "$CFG" "$REC" "$POP" $POST 2>&1 | tee -a "$REC/../$RUN_ID-$TAG.out"
else
  bash "$REPO/record_v5/run_v5.sh" "$CFG" "$REC" "$POP" "$@" 2>&1 | tee "$REC/../$RUN_ID-$TAG.out"
fi
echo "[run_night] $(date -u +%FT%TZ) run_v5 finished; verdict: $($PY -c "import json,sys; d=json.load(open('$REC/verify/verify_v5.json')); print(d.get('verdict') or d.get('status'))" 2>/dev/null || echo 'no verify_v5.json')"
