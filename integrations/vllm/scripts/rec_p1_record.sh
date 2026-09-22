#!/usr/bin/env bash
# ============================================================================================================================
# LEGACY (DIRECTIVE v4) — NOT THE CONTRACT RECORD PATH.  Kept only to reproduce the REC-P1 historical record.
#   protocol    -> rec_p1_protocol.py   writes the challenge seed to seed.hex on the prover's host (R7); no exemption request (R8/R16)
#   obligations -> rec_p1_obligations.py "exempt (charged)" VU classes; Input gates as VUs (R2/R8)
#   verify      -> rec_p1_verify.py     B_ex = selected exempt VUs; V5 trusts provenance["units"]; legacy certificate fields (R4/R11/R14)
# Refuses to run unless --legacy-directive-v4 is among its arguments (R14: legacy unselectable by accident); the flag is forwarded
# to the three gated scripts.  With the flag, behaviour is unchanged.  Contract path: e2e/verifier_v3 (A3) composed by
# record_v5/ (A9); contract: docs/PROTOCOL.md; status: docs/PROTOCOL-CONFORMANCE.md.  [A10 2026-09-10]
# ============================================================================================================================
# [L1 RECORD 2026-09-10] One end-to-end record on SMALL over the VALIDATED P1/cu=whole partition (DIRECTIVE v3 §1.1 steps 1-8; brief L1 steps 1-11).
#
#   population   scripts/rec_p1_population.py: population/v2 rows (digest re-derived == validated 56ebcdf5...), COMPACT-P1 verdict cited verbatim   [CPU pod]
#   instance     e2e.protocol_instance make over that directory x N_REQ (Bernoulli q, s; delta) + the registered weight commitment (ROT-INPUT)  [CPU pod]
#   serve        verity_vllm_adapter.production (eager, temporary): N_REQ requests of the B0 workload (LP=32, 16 slots)                         [GPU pod]
#   protocol     scripts/rec_p1_protocol.py: prover phase 0 (request commitments + weights) -> RU challenge (e2e.partition_sample stage1 over the
#                replicated population) -> replay J, commit interiors (phase 2) -> VU challenge (stage2) -> reveal -> verify                     [GPU pod]
#   obligations  scripts/rec_p1_obligations.py: selected VUs -> statements (GemmCoordinate_v2 batches) + input-unit checks + exemptions       [GPU pod]
#   prove        approved ELF 9fd00e60 only, compressed, SP1 CUDA                                                                                   [GPU pod]
#   verify       scripts/rec_p1_verify.py -> verifier.log, certificate.json                                                                        [CPU/GPU]
#
# Env: REPO (/workspace/veritor), ENV (/workspace/rec-env), RES (/workspace/rec), N_REQ (4), Q (1/2), S (1/16384), DELTA (1/1000000000000), STAGES
set -euo pipefail

# [A10 2026-09-10] R14 legacy guard: require --legacy-directive-v4; strip it from the stage list; forward it to the gated scripts.
LEGACY_FLAG="--legacy-directive-v4"
LEGACY_OK=0; ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "$LEGACY_FLAG" ]]; then LEGACY_OK=1; else ARGS+=("$arg"); fi
done
if [[ "$LEGACY_OK" -ne 1 ]]; then
  cat >&2 <<EOF
rec_p1_record.sh: LEGACY (DIRECTIVE v4) — not the contract record path; refusing to run without $LEGACY_FLAG.
  seed.hex on the prover's host (R7); 'exempt (charged)' VUs as B_ex (R8/R11); Input gates as VUs (R2); V5 trusts prover provenance (R4).
  Contract path: e2e/verifier_v3 (driver.py, reverify.py) composed by record_v5/ (A9); docs/PROTOCOL.md governs.
  Historical reproduction of the REC-P1 record: re-run with $LEGACY_FLAG (behaviour is otherwise unchanged).
EOF
  exit 2
fi
set -- "${ARGS[@]+"${ARGS[@]}"}"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.sp1/bin:$PATH"
REPO=${REPO:-/workspace/veritor}
ENV=${ENV:-/workspace/rec-env}
POC=$REPO/vllm-poc
PY=$REPO/.venv/bin/python
PYV=$ENV/venv-cu129/bin/python
export HF_HOME=${HF_HOME:-$ENV/hf}
export PYTHONPATH=$POC
export VERITY_CORE_PYTHON=${VERITY_CORE_PYTHON:-$PY}
RES=${RES:-/workspace/rec}
N_REQ=${N_REQ:-4}; Q=${Q:-1/2}; S=${S:-1/16384}; DELTA=${DELTA:-1/1000000000000}
STAGES=${STAGES:-${*:-instance}}
V3=$POC/manifests/semantic-profiles/llama-eager-batch-invariant-v3.json
POP_DIR=${POP_DIR:-$RES/pop}
INST_DIR=${INST_DIR:-$RES/instance}
HOST=${HOST:-$ENV/veritor-zk-host}
stage() { echo; echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] stage $*"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
cd "$POC"
mkdir -p "$INST_DIR"

if has population; then
  stage population
  $PY scripts/rec_p1_population.py --out "$POP_DIR"
fi

if has instance; then
  stage instance "(P1/cu=whole x $N_REQ, bernoulli q=$Q s=$S delta=$DELTA)"
  test -f "$POP_DIR/population.compact.json" || { echo "no population/v2 at $POP_DIR"; exit 2; }
  test -f "$INST_DIR/weights_authorized.json" || { echo "no registered weight commitment at $INST_DIR/weights_authorized.json (scripts/rec_weights_authorize.py, vLLM venv, GPU pod)"; exit 2; }
  PROFILE_SHA=$(sha256sum "$V3" | cut -d' ' -f1)
  VOCAB_DIGEST=$($PY -c "import json; print(json.load(open('$POC/results/DERIVE-CORR/derived-b0/s4/binding.json'))['program']['vocabulary']['digest'])")
  cp "$POP_DIR/compact_validation.json" "$POP_DIR/maps.json" "$POP_DIR/index.json" "$INST_DIR/"
  ln -sfn "$POP_DIR/population.compact.json" "$INST_DIR/population.compact.json"
  $PY -m e2e.protocol_instance make --population-dir "$POP_DIR" --instance-id "B0-P1-REC-$(date -u +%Y%m%dT%H%M%SZ)" \
      --identity '{"family":"Serve_v1","config_sha12":"6f196795e564","case":"B0","LP":32,"STEPS":15,"function":"Serve_v1{C={6f196795e564},LP=32,STEPS=15}"}' \
      --vocabulary "{\"digest\":\"$VOCAB_DIGEST\",\"schema\":\"verity-ir/registry-version/v0\",\"profile_name\":\"llama-eager-batch-invariant-v3\",\"profile_sha256\":\"$PROFILE_SHA\",\"source\":\"$V3 (sha256 of the profile file; DERIVE-CORR s4/binding.json vocabulary digest)\"}" \
      --outputs '[{"name":"token_slot","multiplicity":16,"domain_bits_declared":16,"wire_bits":32,"enforcement":"declared_domain 16 on sampler.select; check_output_domains on every request","source":"Serve_v1 STEPS+1 = 16 root output slots per request; OUT16 (GUIDANCE §32)"}]' \
      --vocabulary-size 49152 --q "$Q" --s "$S" --delta "$DELTA" --instances "$N_REQ" \
      --first-boundary-rule "outgoing boundary runs of every CU" --first-boundary-committed-in boundary --acquisition-granularity ru \
      --weights-authorized "$INST_DIR/weights_authorized.json" --chronology-kind partition-sample/v1 \
      --seed-derivation 'e2e.partition_sample: seed = 32 verifier bytes (secrets.token_bytes); seed_commitment = H(domain, seed); r_q = bernoulli_subset(seed, q/replay-unit, H(PHASE, q, instance_digest, boundary_digest), N_R, q) with N_R = instances x 1 RU (P1/cu=whole: one RU per request); r_s = bernoulli_subset(seed, s/verification-unit, H(PHASE, s, instance_digest, r_q_digest, interior_digest), |VUs of J|, s), |VUs of J| ranked in J order, ascending VU index inside each RU (population/v2 replicated view: global VU = r*N_V + i); an empty r_q is kept as event r_q_empty[k] and redrawn under H(PHASE, q/replay-unit/redraw, phase, k) (DIRECTIVE 1.5.10; certificate: RU draw conditioned on >= 1 selected (demonstration))' \
      --out "$INST_DIR/protocol_instance.json"
  $PY -m e2e.protocol_instance validate "$INST_DIR/protocol_instance.json"
  $PY -m e2e.protocol_instance digest "$INST_DIR/protocol_instance.json" > "$INST_DIR/instance_digest.txt"
fi

RID=${RID:-REC-P1-B0}
OUT=$RES/$RID
if has serve; then
  stage serve "($N_REQ requests, eager, temporary)"
  mkdir -p "$OUT/adapter"
  WORKLOAD=$OUT/workload.json
  $PY - "$N_REQ" "$WORKLOAD" <<'PYEOF'
import json, sys
from pathlib import Path
n, out = int(sys.argv[1]), Path(sys.argv[2])
w = json.loads((Path("e2e") / "workload-B0-small.json").read_text())
w["requests"] = w["requests"][:n]
if len(w["requests"]) != n:
    raise SystemExit(f"workload has {len(w['requests'])} requests, need {n}")
out.write_text(json.dumps(w, indent=1) + "\n")
print(f"[serve] workload: {n} requests, prompt_len {sorted({r['prompt_len'] for r in w['requests']})}, max_tokens {sorted({r['max_tokens'] for r in w['requests']})}")
PYEOF
  rm -rf "$OUT/adapter/production"
  T0=$(date +%s.%N)
  VLLM_BATCH_INVARIANT=1 $PYV -m verity_vllm_adapter.production --case B0 --checkpoints manifests/checkpoints.json --workload "$WORKLOAD" \
      --out "$OUT/adapter/production" --adapter-revision "$(git -C $REPO rev-parse --short HEAD 2>/dev/null || echo lane-record)" \
      --wheel-sha256 7aa52ac71500caf74643135a340eb2d03ae0386d4282fc0b23846940029fbbb3 --vllm-commit d9105ea80 --enforce-eager --logprobs 2 2>&1 | tail -15
  $PY -c "import json,sys,time; print(json.dumps({'stage':'serve (production: stock pass + recording pass, engine build incl.)','class':'serving','seconds':round(time.time()-float(sys.argv[1]),1),'hardware':'RTX 4090 (record-gpu)','batch':'$N_REQ requests, eager, temporary','label':'MEASURED','note':'adapter/production/timings.json has the per-pass split'}))" "$T0" >> "$OUT/costlog.jsonl"
fi

if has protocol; then
  stage protocol "(phase 0 -> RU challenge -> replay J -> VU challenge)"
  $PY scripts/rec_p1_protocol.py "$LEGACY_FLAG" --record "$OUT" --instance-dir "$INST_DIR" --population "$POP_DIR/population.compact.json" --instances "$N_REQ" \
      --py-vllm "$PYV" --semantic-profile "$V3" --condition-nonempty
fi

if has obligations; then
  stage obligations
  $PY scripts/rec_p1_obligations.py "$LEGACY_FLAG" --record "$OUT" --instance-dir "$INST_DIR" ${OBLIG_ARGS:-}
fi

if has prove; then
  stage prove "(approved ELF only; compressed; SP1 CUDA)"
  $PY scripts/rec_p1_prove.py --record "$OUT" --host "$HOST" ${PROVE_ARGS:-}
fi

if has verify; then
  stage verify
  $PY scripts/rec_p1_verify.py "$LEGACY_FLAG" --record "$OUT" --population "$POP_DIR/population.compact.json" --expressibility "$REPO/out/record/expressibility.json" --host "$HOST" ${VERIFY_ARGS:-}
fi

if has negatives; then
  stage negatives "(9 gate + 3 tamper, on copies)"
  $PY scripts/rec_p1_negatives.py --record "$OUT" --population "$POP_DIR/population.compact.json" --host "$HOST" ${NEG_ARGS:-}
fi
echo "REC-P1-DONE $STAGES $(date -u +%Y-%m-%dT%H:%M:%SZ)"
