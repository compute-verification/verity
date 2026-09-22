#!/usr/bin/env bash
# LEGACY (DIRECTIVE v4) record pipeline — not the contract path (record_v5/ + e2e.verifier_v3).  Kept runnable for historical
# reproduction only: its rec_bound.py call passes --legacy-directive-v4 (R14).  [A10 2026-09-10, WR-8]
# [REC 2026-09-09] The corrected B0 record, end to end, on one RTX 4090 pod (GUIDANCE §57.10 priority 1).
#
#   stage 1  population   e2e.population maps for the certified partition (validated in-process; digest-pinned)   [done separately, CPU]
#   stage 1b weights      scripts/rec_weights_authorize.py: the AUDITOR's authorized weight commitment (S5), pre-challenge  [vLLM venv]
#   stage 2  instance     e2e.protocol_instance make: pre-challenge Bernoulli instance over the population x N_REQ, with the S5 weights section
#   stage 3  capture      e2e.run_composed --instance-dir: instance + evidence pinned in the manifest BEFORE READY_PHASE0; Bernoulli q over requests
#   stage 4  chronology   e2e.security_sample from-transcript / verify-transcript (adapter-transcript/v1)
#   stage 5  certify      e2e.certificate (strict v3), e2e.verify_composed (compact path, bundle rederive on the pod)
#   stage 6  negatives    tamper a value / drop a sampled obligation / swap the seed -> NOT BOUND
#
# Env: PLAN (default P3/cu=whole), Q (1/2), S (1/4096), DELTA (1/1000000), N_REQ (4), RES (/workspace/rec), STAGES (all)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ENV=${ENV:-/workspace/padint-env}
REPO=${REPO:-/workspace/veritor-proto-a}
POC=$REPO/vllm-poc
PY=$REPO/.venv/bin/python
PYV=$ENV/venv-cu129/bin/python
export HF_HOME=${HF_HOME:-/workspace/hf}
export PYTHONPATH=$POC
export VERITY_CORE_PYTHON=${VERITY_CORE_PYTHON:-$PY}   # the adapter's Bernoulli sampler hops to a core-capable interpreter (numpy needed by veritor.core)
RES=${RES:-/workspace/rec}
PLAN=${PLAN:-P3/cu=whole}
_PD=$RES/population/${PLAN//\//_}
PLAN_DIR=${PLAN_DIR:-${_PD//=/-}}
Q=${Q:-1/2}; S=${S:-1/4096}; DELTA=${DELTA:-1/1000000}; N_REQ=${N_REQ:-4}
STAGES=${STAGES:-weights instance capture chronology certify negatives bound}
V3=$POC/manifests/semantic-profiles/llama-eager-batch-invariant-v3.json
INST_DIR=${INST_DIR:-$RES/instance}
# first boundary: P3/cu=whole -> the request boundary is the adapter statement (committed at T_0, before r_q): REALISED.
# any other partition -> the cross-CU rows are committed in `interior` (after r_q): the instance records the discrepancy itself.
FB_IN=${FB_IN:-$([[ "$PLAN" == "P3/cu=whole" ]] && echo statement || echo interior)}
WEIGHTS_SRC=${WEIGHTS_SRC:-$RES/instance/weights_authorized.json}
HOST=${HOST:-$ENV/veritor-zk-host}
stage() { echo; echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage $*"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
cd "$POC"
mkdir -p "$INST_DIR"

if has weights; then
  stage 1b-weights
  if [[ -f "$WEIGHTS_SRC" && "$WEIGHTS_SRC" != "$INST_DIR/weights_authorized.json" ]]; then cp "$WEIGHTS_SRC" "$INST_DIR/"; echo "reused $WEIGHTS_SRC";
  else $PYV scripts/rec_weights_authorize.py --case B0 --semantic-profile "$V3" --rope-device cuda --out "$INST_DIR"; fi
fi

if has instance; then
  stage 2-instance "($PLAN x $N_REQ, bernoulli q=$Q s=$S)"
  test -f "$PLAN_DIR/compact_validation.json" || { echo "no validated population at $PLAN_DIR"; exit 2; }
  PROFILE_SHA=$(sha256sum "$V3" | cut -d' ' -f1)
  VOCAB_DIGEST=$($PY -c "import json; print(json.load(open('$POC/results/DERIVE-CORR/derived-b0/s4/binding.json'))['program']['vocabulary']['digest'])")
  cp "$PLAN_DIR/compact_validation.json" "$PLAN_DIR/maps.json" "$INST_DIR/"
  $PY -m e2e.population replicate --population "$PLAN_DIR/population.json" --instances "$N_REQ" --out "$INST_DIR/population.json"
  $PY -m e2e.protocol_instance make --population-dir "$PLAN_DIR" --instance-id "B0-REC-$(date -u +%Y%m%dT%H%M%SZ)" \
      --identity '{"family":"Serve_v1","config_sha12":"6f196795e564","case":"B0","LP":32,"STEPS":15,"function":"Serve_v1{C={6f196795e564},LP=32,STEPS=15}"}' \
      --vocabulary "{\"digest\":\"$VOCAB_DIGEST\",\"schema\":\"verity-ir/registry-version/v0\",\"profile_name\":\"llama-eager-batch-invariant-v3\",\"profile_sha256\":\"$PROFILE_SHA\",\"source\":\"$V3 (sha256 of the profile file; DERIVE-CORR s4/binding.json vocabulary digest)\"}" \
      --outputs '[{"name":"token_slot","multiplicity":16,"domain_bits_declared":16,"wire_bits":32,"enforcement":"declared_domain 16 on sampler.select; check_output_domains on every request","source":"Serve_v1 STEPS+1 = 16 root output slots per request; OUT16 (GUIDANCE §32)"}]' \
      --vocabulary-size 49152 --q "$Q" --s "$S" --delta "$DELTA" --instances "$N_REQ" --coverage-size 1000 \
      --first-boundary-rule "outgoing boundary runs of every CU" --first-boundary-committed-in "$FB_IN" --acquisition-granularity request \
      --weights-authorized "$INST_DIR/weights_authorized.json" --chronology-kind adapter-transcript/v1 \
      --seed-derivation "$($PY -c 'from e2e.security_sample import ADAPTER_SEED_DERIVATION as d; print(d)')" \
      $( test -f "$PLAN_DIR/summary_meta.json" && echo --summary-digest "$($PY -c "import json; print(json.load(open('$PLAN_DIR/summary_meta.json'))['summary_digest'])")" ) \
      --out "$INST_DIR/protocol_instance.json"
  $PY -m e2e.protocol_instance validate "$INST_DIR/protocol_instance.json"
fi

RID=${RID:-REC-B0-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=$RES/$RID
if has capture; then
  stage 3-capture "$RID"
  $PY -m e2e.run_composed --case B0 --out "$OUT" --challenge security --preparation selective --selective-q "$Q" --s "$S" --coverage-size 1000 \
      --instance-dir "$INST_DIR" --full-program skip --plan-unit layer --semantic-profile "$V3" \
      --py-vllm "$PYV" --host "$HOST" --sp1-prover cuda --hf-home "$HF_HOME" --cache-root /workspace/padint-cache --prove-retries 2 2>&1 | tail -40
fi

if has chronology; then
  stage 4-chronology
  $PY -m e2e.security_sample from-transcript --record "$OUT" --out "$OUT"
  $PY -m e2e.security_sample verify-transcript --record "$OUT" --out "$OUT" || true
fi

if has certify; then
  stage 5-certify
  $PY -m e2e.certificate --record "$OUT" || true
  $PY -m e2e.verify_composed --dir "$OUT" --host "$HOST" 2>&1 | tail -20 || true
fi

if has negatives; then
  stage 6-negatives
  $PY -m e2e.rec_negatives --record "$OUT" --out "$RES/negatives" || true
fi
if has bound; then
  stage 7-bound "(certified search over the exact Summary; NO exclusion -> cap expected)"
  SUMMARY=${SUMMARY:-$( [[ -f $PLAN_DIR/summary.json ]] && echo $PLAN_DIR/summary.json || echo $PLAN_DIR-summary/summary.json )}
  CV=""; $PY -c "import json,sys; sys.exit(0 if json.load(open('$OUT/certificate.json')).get('certified') else 1)" 2>/dev/null && CV=--covers-verified
  test -f "$SUMMARY" && $PY scripts/rec_bound.py --legacy-directive-v4 --summary "$SUMMARY" --q "$Q" --s "$S" --delta "$DELTA" --record-id "$RID" $CV --out "$OUT/bound.json" || echo "no summary.json at $SUMMARY (bound stage skipped)"
fi
echo "REC-DONE $OUT"
