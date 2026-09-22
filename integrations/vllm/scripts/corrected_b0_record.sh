#!/usr/bin/env bash
# [PROTO-A, brief A.6 -- PHASE 2, POD ONLY; NOT RUN ON 2026-09-09] The corrected B0 record, end to end:
#
#   population := the validated VU partition of the certificate-pinned program (P3/cu=block by default; PLAN=root alternative),
#   Bernoulli law (q over RUs, s over the VUs of J's RUs) with commitment -> challenge chronology, a pre-challenge protocol
#   instance pinned in the manifest before READY_PHASE0, the digest-pinned compact whole-program validation artifact in the
#   record, proofs for every security-sampled obligation, certificate v3 (hard requirements), independent re-verification,
#   the A5 negatives; then the same for the padded early-termination record (PAD-INT workload + padded program).
#
#   stage 1  compact    e2e.population maps: build + plan + VALIDATE the whole pinned program in one process, write
#                       value_map.tsv / vu_map.tsv / ru_map.tsv / population.json / compact_validation.json
#                       (MEASURED on the COMPACT-2 pod: P3/cu=block 340 s / 2,847 MB; root 311 s / 1,874 MB)
#   stage 2  instance   e2e.protocol_instance make: the pre-challenge instance over that population (instances = N_REQ)
#   stage 3  capture    run_composed --case B0 ... WITH THE INSTANCE (see GAP P2-0 below), Bernoulli law, chronology
#   stage 4  certify    e2e.certificate (v3: instance + compact validation + population + chronology REQUIRED)
#   stage 5  reverify   e2e.verify_composed (no --historical-v2: the 'instance' row must be ok)
#   stage 6  negatives  e2e.artefact_negatives + padint_negatives.py + tests/test_proto_a_instance.py negatives on the record
#   stage 7  padded     stages 2-6 again with the PAD-INT workload and --padded-program
#
# Parameters (env):
#   PLAN        validated partition used as the population: P3/cu=block (default; RU = block, 14,197 VUs / 4,295 RUs per instance)
#               | root (RU = Input batch, 4,295 VUs / 2 RUs per instance).  P3/cu=whole (RU = request) has NO validation verdict
#               (docs/data/compact/whole-b0/P3-cu-whole.json: built, not validated) -- validating it is phase-2 item P2-4.
#   Q, S        Bernoulli rates (defaults 1/2 and 1/4096: the RC/INT-PAD-2 values, now as rates, not k/N)
#   DELTA       security threshold (default 1/1000000; bound_record.py DELTAS[0])
#   COVERAGE    diagnostic coverage draw (default 0: no diagnostic units in the corrected record's security accounting)
#   NORM_ROWS   GUIDANCE §50.5 -- Option A (`final`, default): the population is the program's relations; the capture's 124 excess
#               norm.rmsnorm rows (31 non-final prefill rows x 2 ports x 2 requests) are NOT population members (they are outside the
#               pinned program; the address map has no VU for them).  Option B (`all`): pin the all-rows program variant
#               Serve_v1{NORM_ROWS=all} (DERIVE a25fc156..., --rows vllm) so program and production agree by construction; NEW program
#               digest, plan, validation artifact and instance.  This script carries the choice as a parameter; it is a USER GATE, not a
#               decision made here.  Option B additionally needs the family registration (CAP-CORR) before stage 1 can build it.
#   WORKLOAD    B0 workload (default e2e/workload-B0-small.json: 4 requests); PADINT_WORKLOAD for stage 7
#   RID         reuse an existing record id (skip stage 3)
#
# GAP P2-0 (the integration this script depends on; NOT implemented on 2026-09-09, laptop phase):
#   run_composed needs `--instance FILE --population FILE --compact-validation FILE` and must
#     (a) copy the three files into the record and pin protocol_instance.json's sha256 as manifest inputs.protocol_instance_sha256
#         BEFORE READY_PHASE0 (next to sampling_policy_sha256);
#     (b) after the prover's boundary commitment (statement digest), draw r_q with `e2e.security_sample stage1` over the RECORD
#         population's RUs (instances x N_R) instead of the k-of-N / Bernoulli-over-requests selector; J = set of RUs; the prover
#         commits EXACTLY the interiors of J's RUs (message-structure check = the leaves addressed by ru_map/value_map for J);
#     (c) after the interior commitment, draw r_s with `e2e.security_sample stage2`; resolve every sampled VU through
#         e2e.population.Addresser (value_map.tsv: committed position -> VU -> RU -> obligation) to the committed leaves it opens;
#     (d) write chronology.json (every event UTC-stamped; every challenge drawn; rerolls = 0) and reveal the seed only after r_s;
#     (e) fold the Bound with q, s, delta, W_out = 16 x 16 bits (declared_domains on the plan's 16 root output slots) and record it in
#         the instance's security section as bound_value_bits (ANALYTICAL) -- Bernoulli law, so the Bound attaches.
#   Until (a)-(e) land, stage 3 below is the exact intended invocation and FAILS FAST (unknown arguments).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ENV=${ENV:-/workspace/padint-env}
REPO=${REPO:-/workspace/veritor-proto-a}
POC=$REPO/vllm-poc
PY=$REPO/.venv/bin/python
PYV=${PYV:-$ENV/venv-cu129/bin/python}
HOST=${HOST:-$ENV/veritor-zk-host}
RES=${RES:-/workspace/proto-a-results}
V3=$POC/manifests/semantic-profiles/llama-eager-batch-invariant-v3.json
CERT_REC=$POC/results/INT-PAD-2/INTPAD2-B0-20260909T013519Z
PLAN=${PLAN:-P3/cu=block}
Q=${Q:-1/2}
S=${S:-1/4096}
DELTA=${DELTA:-1/1000000}
COVERAGE=${COVERAGE:-0}
NORM_ROWS=${NORM_ROWS:-final}
WORKLOAD=${WORKLOAD:-$POC/e2e/workload-B0-small.json}
PADINT_WORKLOAD=${PADINT_WORKLOAD:-/workspace/padint-results/prompt-search/workload-B0-padint.json}
STAGES=${STAGES:-"compact instance capture certify reverify negatives padded"}
export HF_HOME=$ENV/hf
export VERITY_CORE_PYTHON=$PY
export SP1_PROVER=cuda
export PYTHONPATH=$REPO/src:$REPO/zk/bench:$POC:$REPO
mkdir -p "$RES"
stage() { echo; echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] corrected-b0 stage $*"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
cd "$POC"

$PY -c "import veritor, sys; assert veritor.__file__.startswith('$REPO'), veritor.__file__"   # the lane's code, not mainline

PLAN_DIR=$RES/compact/$(echo "$PLAN" | tr '/=' '__')
N_REQ=$($PY -c "import json,sys; print(len(json.load(open(sys.argv[1]))['requests']))" "$WORKLOAD")

if [ "$NORM_ROWS" = "all" ]; then
  echo "NORM_ROWS=all (Option B): requires the registered Serve_v1{NORM_ROWS=all} family (CAP-CORR) and re-pins the program digest;"
  echo "stage 1 must then build that variant (e2e.population maps --case B0 --norm-rows all: flag NOT implemented; P2-5). Stopping."
  exit 3
fi

# 1. the validated partition, its maps and the compact validation artifact (one process; MEASURED cost printed by the tool)
if has compact && [ ! -f "$PLAN_DIR/compact_validation.json" ]; then
  stage 1-compact "$PLAN"
  /usr/bin/time -v $PY -m e2e.population maps --case B0 --lp 32 --steps 15 --plan "$PLAN" --out "$PLAN_DIR" 2>&1 | grep -E "^\[population\]|Maximum resident|Elapsed"
  $PY -c "
import json; a=json.load(open('$PLAN_DIR/compact_validation.json'))
assert a['status']=='valid', a['status']
assert a['program_digest']=='d4f4b1f1b4d592d697eef5b6c9c803108bdf02d0ef24c945eace6e705832e1b8', a['program_digest']  # the certificate-pinned Serve_v1
print('compact validation: valid; N_V', a['vus'], 'N_R', a['cus'], 'plan', a['plan_digest'][:16])"
  $PY -m e2e.population replicate --population "$PLAN_DIR/population.json" --instances "$N_REQ" --out "$PLAN_DIR/population-x$N_REQ.json"
fi

# 2. the pre-challenge instance (identity / vocabulary from the certificate-pinned record; declared output domain 16 bits x 16 slots)
INST=$RES/instance/protocol_instance-$(echo "$PLAN" | tr '/=' '__')-x$N_REQ.json
if has instance && [ ! -f "$INST" ]; then
  stage 2-instance
  PROFILE_SHA=$(sha256sum "$V3" | cut -d' ' -f1)
  VOCAB_DIGEST=$($PY -c "import json; print(json.load(open('$POC/results/DERIVE-CORR/derived-b0/s4/binding.json'))['program']['vocabulary']['digest'])")
  $PY -m e2e.protocol_instance make --population-dir "$PLAN_DIR" --instance-id "B0-corrected-$(date -u +%Y%m%dT%H%M%SZ)" \
      --identity '{"family":"Serve_v1","config_sha12":"6f196795e564","case":"B0","LP":32,"STEPS":15,"function":"Serve_v1{C={6f196795e564},LP=32,STEPS=15}"}' \
      --vocabulary "{\"digest\":\"$VOCAB_DIGEST\",\"schema\":\"verity-ir/registry-version/v0\",\"profile_name\":\"llama-eager-batch-invariant-v3\",\"profile_sha256\":\"$PROFILE_SHA\",\"source\":\"$V3 (sha256 of the profile file; DERIVE-CORR s4/binding.json vocabulary digest)\"}" \
      --outputs '[{"name":"token_slot","multiplicity":16,"domain_bits_declared":16,"wire_bits":32,"enforcement":"declared_domain 16 on sampler.select; check_output_domains on every request","source":"Serve_v1 STEPS+1 = 16 root output slots per request; OUT16 (GUIDANCE §32)"}]' \
      --vocabulary-size 49152 --q "$Q" --s "$S" --delta "$DELTA" --instances "$N_REQ" --coverage-size "$COVERAGE" --out "$INST"
  $PY -m e2e.protocol_instance validate "$INST"
fi

# 3. capture with the instance (GAP P2-0: the three new run_composed flags)
if [ -z "${RID:-}" ]; then RID="PROTOA-B0-$(date -u +%Y%m%dT%H%M%SZ)"; fi
OUT=$RES/$RID
if has capture && [ ! -f "$OUT/summary.json" ]; then
  stage 3-capture "$RID" "(N_REQ=$N_REQ, law bernoulli q=$Q s=$S, population $PLAN x $N_REQ, NORM_ROWS=$NORM_ROWS)"
  $PY -m e2e.run_composed --case B0 --out "$OUT" --challenge security --preparation selective \
      --instance "$INST" --population "$PLAN_DIR/population-x$N_REQ.json" --compact-validation "$PLAN_DIR/compact_validation.json" \
      --s "$S" --coverage-size "$COVERAGE" --full-program skip \
      --workload "$WORKLOAD" --semantic-profile "$V3" --py-vllm "$PYV" --host "$HOST" --hf-home "$HF_HOME" \
      --cache-root /workspace/proto-a-cache 2>&1 | tail -40
  # (the pre-challenge instance is pinned before READY_PHASE0; --k / --selective-q are NOT given: the law is the instance's)
fi

# 4-5. certificate v3 (hard requirements) + independent re-verification (the 'instance' row must be ok, not skipped)
if has certify; then
  stage 4-certify
  $PY -m e2e.certificate --record "$OUT" --require-coverage --out "$OUT/certificate.json"
  $PY -m e2e.protocol_instance check "$OUT/protocol_instance.json" --record "$OUT" --validation "$OUT/compact_validation.json"
  $PY -m e2e.security_sample verify --instance "$OUT/protocol_instance.json" --population "$OUT/population.json" --out "$OUT"
fi
if has reverify; then
  stage 5-reverify
  $PY -m e2e.verify_composed --dir "$OUT" --out "$OUT/reverify" --program rebuild --plan recorded --bundle-sample 64 --host "$HOST"
  $PY -c "import json; d=json.load(open('$OUT/reverify/reverify.json')); assert d['rows']['instance']['status']=='ok', d['rows']['instance']; print('instance row ok; attests', d['process']['attests_protocol_instance'])"
fi

# 6. negatives on the corrected record: the record's own (artefact / padint) and the A5 set (instance / population / law / domain / unit)
if has negatives; then
  stage 6-negatives
  $PY -m e2e.artefact_negatives --src "$OUT" --out "$OUT/negatives"
  $PY $POC/scripts/padint_negatives.py --record "$OUT" --out "$OUT/negatives/padint" || true
  PROTO_A_RECORD="$OUT" $PY -m pytest "$POC/tests/test_proto_a_instance.py" -q -k "negative or entry_point"
fi

# 7. the padded early-termination record: the same, PAD-INT workload + padded program (morning-2026-09-09.md §2)
if has padded; then
  stage 7-padded
  test -f "$PADINT_WORKLOAD" || { echo "no PAD-INT workload at $PADINT_WORKLOAD (scripts/padint_prompt_search.py)"; exit 2; }
  N_PAD=$($PY -c "import json,sys; print(len(json.load(open(sys.argv[1]))['requests']))" "$PADINT_WORKLOAD")
  $PY -m e2e.population replicate --population "$PLAN_DIR/population.json" --instances "$N_PAD" --out "$PLAN_DIR/population-x$N_PAD.json"
  PINST=$RES/instance/protocol_instance-padded-x$N_PAD.json
  $PY -m e2e.protocol_instance make --population-dir "$PLAN_DIR" --instance-id "B0-padded-corrected-$(date -u +%Y%m%dT%H%M%SZ)" \
      --identity '{"family":"Serve_v1","config_sha12":"6f196795e564","case":"B0","LP":32,"STEPS":15,"function":"Serve_v1{C={6f196795e564},LP=32,STEPS=15}"}' \
      --vocabulary "{\"digest\":\"$VOCAB_DIGEST\",\"schema\":\"verity-ir/registry-version/v0\",\"profile_name\":\"llama-eager-batch-invariant-v3\",\"profile_sha256\":\"$PROFILE_SHA\",\"source\":\"$V3\"}" \
      --outputs '[{"name":"token_slot","multiplicity":16,"domain_bits_declared":16,"wire_bits":32,"enforcement":"declared_domain 16; padded slots after EOS are inactive (PadGuard/PadRowMux proofs)","source":"Serve_v1 STEPS+1 slots; PAD-INT"}]' \
      --vocabulary-size 49152 --q "$Q" --s "$S" --delta "$DELTA" --instances "$N_PAD" --coverage-size "$COVERAGE" --out "$PINST"
  POUT=$RES/PROTOA-B0-PADDED-$(date -u +%Y%m%dT%H%M%SZ)
  $PY -m e2e.run_composed --case B0 --out "$POUT" --challenge security --preparation selective \
      --instance "$PINST" --population "$PLAN_DIR/population-x$N_PAD.json" --compact-validation "$PLAN_DIR/compact_validation.json" \
      --s "$S" --coverage-size "$COVERAGE" --full-program skip --padded-program hand-built \
      --workload "$PADINT_WORKLOAD" --semantic-profile "$V3" --py-vllm "$PYV" --host "$HOST" --hf-home "$HF_HOME" \
      --cache-root /workspace/proto-a-cache 2>&1 | tail -40
  # PAD-INT kind coverage (padint_capture.sh stages pad/embed/prove/attach) then:
  $PY -m e2e.certificate --record "$POUT" --require-coverage --out "$POUT/certificate.json"
  $PY -m e2e.verify_composed --dir "$POUT" --out "$POUT/reverify" --program rebuild --plan recorded --host "$HOST"
fi
echo; echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] corrected-b0 done: $OUT"
