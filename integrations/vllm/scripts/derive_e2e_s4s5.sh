#!/usr/bin/env bash
# Stages S4 + S5 of `scripts/derive_e2e.sh` (GUIDANCE §43.5), lane CAPCORR.  DERIVE owns `derive_e2e.sh`; this file
# holds CAPCORR's stages as functions so DERIVE's driver can `source` it and call `stage_s4` / `stage_s5` after S3
# has produced the derived program, or run standalone:
#
#   bash scripts/derive_e2e_s4s5.sh                                                    # DERIVED program (default)
#   PROGRAM=registry:vllm-poc/profiles/b0-a100-eager bash scripts/derive_e2e_s4s5.sh   # harness validation (scaffolding)
#
# The default PROGRAM is the derived one: `python:capcorr_derived_program:program` runs DERIVE's importer
# (`verity_vllm.frontend.derive.derive_report`) on `GreedyServe(B0_CONFIG, steps=STEPS)` with a `(prompt int32[LP],)` example
# input in profile mode `b1-eager` (LP/STEPS taken from the record's pins) and hands the resulting Program to every stage.
# Pod (derive-dev): the run of record is `/workspace/capcorr/run_derived.sh` -> `fixtures/results/DERIVE-CORR/`.
#
# The program is the ONE parameter that changes between the harness-validation run (hand-built registry program,
# mandate boundary 6: not the deliverable) and the derived-program run; everything downstream (binding, L2, L3,
# statement, witness, proof, verifier refusals) is identical.
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-/workspace/capcorr-env/venv/bin/python}
[ -x "$PY" ] || PY=python
export PYTHONPATH=.:src:vllm-poc:scripts
PROGRAM=${PROGRAM:-python:capcorr_derived_program:program}
RECORD=${RECORD:-fixtures/results/INT-PAD-2/INTPAD2-B0-20260909T013519Z}
BUNDLE=${BUNDLE:-/workspace/capcorr/bundle/INT-B0-INTPAD2-B0-20260909T013519Z/bundle}
CKPT=${CKPT:-/workspace/capcorr-env/ckpt}
COS_SIN=${COS_SIN:-fixtures/B0-divergence-20260907T1604Z/layers.0.self_attn.rotary_emb/cos_sin_cache.npy}
RUN=${RUN:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT_ROOT=${OUT_ROOT:-/workspace/capcorr/DERIVE-CORR}
S4_OUT=${S4_OUT:-$OUT_ROOT/s4-$RUN}
S5_OUT=${S5_OUT:-$OUT_ROOT/s5-$RUN}
JOBS=${JOBS:-8}
COORD_SAMPLES=${COORD_SAMPLES:-32}
# the pins of the ONE system (GUIDANCE §44.1): kind-vocabulary version DERIVE's S3 recorded (declared vocabulary, see
# verity_vllm.correspondence.program.vocabulary_version) and the record's semantic profile v3 sha; S4 refuses if either differs
# [E1/E6 NIGHT 2026-09-12] vocabulary_version() hashes registry/b1.py's SOURCE: adding SMOL360_CONFIG / QWEN05_CONFIG (data only) moved the digest
# 33ee156d... -> c10e0b2c... (MEASURED, E6 06:20Z / E1 06:30Z). A record derived on the older tree pins 33ee156d...: pass EXPECT_VOCAB=33ee156d... for it.
EXPECT_VOCAB=${EXPECT_VOCAB:-c10e0b2c092f51f2a50d57227410b3e069c7ea5c14f9ac17be8debb6da80727e}
EXPECT_PROFILE_SHA=${EXPECT_PROFILE_SHA:-847913c5f8bd2a546e46b86a4b5579dd4eb596a9f4733d3fd687f74fc0260474}   # certificate.instance.semantic_profile.sha256 (llama-eager-batch-invariant-v3); the manifest's inputs.semantic_profile_sha256 (ac62500f…) is a different hash

stage_s4() {
  echo; echo "=== [$(date -u +%FT%TZ)] S4 correspondence: program=$PROGRAM record=$RECORD -> $S4_OUT"
  local extra=()
  [ -n "$EXPECT_VOCAB" ] && extra+=(--expect-vocabulary "$EXPECT_VOCAB")
  [ -n "$EXPECT_PROFILE_SHA" ] && extra+=(--expect-profile-sha256 "$EXPECT_PROFILE_SHA")
  (cd "$REPO" && "$PY" scripts/derive_corr.py --program "$PROGRAM" --record "$RECORD" --out "$S4_OUT" --stage bind,l2,l3 \
      --bundle "$BUNDLE" --checkpoint "$CKPT/model.safetensors" --cos-sin-npy "$COS_SIN" --jobs "$JOBS" --coord-samples "$COORD_SAMPLES" --resume "${extra[@]}")
  local rc=$?
  echo "S4 exit $rc"
  (cd "$REPO" && "$PY" scripts/derive_corr.py --program "$PROGRAM" --record "$RECORD" --out "$S4_OUT" --self-test-refusals)
  echo "S4 refusals exit $?"
  return $rc
}

stage_s5() {
  echo; echo "=== [$(date -u +%FT%TZ)] S5 prove: program=$PROGRAM s4=$S4_OUT -> $S5_OUT"
  PROGRAM="$PROGRAM" RECORD="$RECORD" S4="$S4_OUT" BUNDLE="$BUNDLE" CKPT="$CKPT" OUT="$S5_OUT" PY="$PY" REPO="$REPO" \
    bash "$REPO/scripts/derive_prove.sh"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  stage_s4; rc4=$?
  if [ $rc4 -eq 2 ]; then echo "S4 REFUSED the program/record binding; S5 not run"; exit 2; fi
  stage_s5; rc5=$?
  echo "S4 exit $rc4, S5 exit $rc5"
  [ $rc4 -eq 0 ] && [ $rc5 -eq 0 ]
fi
