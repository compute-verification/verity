#!/usr/bin/env bash
# [PAD-INT] Stage 0 capture + stage 2 padded path on the shared `derive-dev` RTX 4090 pod (GUIDANCE §44.2 PAD-INT).
#
#   0  search     padint_prompt_search.py: early-terminating B0 prompts under port b0-eager -> workload-B0-padint.json
#                 (skipped when WORKLOAD is given)
#   1  capture    run_composed --case B0 (security challenge, selective preparation, k = N so every request is
#                 replayed, v3 profile declared_domain 16): the record PADINT-B0-<ts>
#   2  empty      the empty-sample run (--s ~0, coverage 0) on the same adapter; the corrected pass-bit criterion
#                 (certified == 0 AND output typing ran on every request incl. the early-terminated one)
#   3  pad        padint_pad_statements.py: PadGuard / PadRowMux statements at the REALIZED liveness (live, stop-evidence,
#                 inactive) addressed to the padded program (PROGRAM=hand-built | derived:<descriptor.json.gz>)
#   4  embed      embed_relations.py (prefill + decode embedding rows; the certificate's kind-coverage row needs them)
#   5  prove      SP1 CUDA prove (core + compressed) with the record's shipped cov host; CPU-mode verify of every proof
#   6  attach     e2e.cov_attach -> <record>/proofs/proof-cov/ ; e2e.certificate --require-coverage
#   7  reverify   e2e.verify_composed (program rebuilt, proofs re-verified by the archived host)
#   8  negatives  e2e.negative_audit + padint_negatives.py (live relation claimed on an inactive slot; missing required
#                 inactive proof; altered padded output)
# Env: WORKLOAD (skip search), PROGRAM (hand-built default), RID (reuse an existing record: skip 1-2), STAGES (subset).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
ENV=/workspace/padint-env
REPO=/workspace/veritor-padint
POC=$REPO/vllm-poc
PY=$REPO/.venv/bin/python
PYV=$ENV/venv-cu129/bin/python
HOST=$ENV/veritor-zk-host          # the certified record's shipped CUDA host (tc-ampere guest) -- run_composed stage 7
CERT_REC=$POC/results/INT-PAD-2/INTPAD2-B0-20260909T013519Z
COVHOST=$CERT_REC/bin/veritor-zk-host-cov   # INT-PAD-3's host: guest includes embed-gather + pad-serve
RES=/workspace/padint-results
V3=$POC/manifests/semantic-profiles/llama-eager-batch-invariant-v3.json
PROGRAM=${PROGRAM:-hand-built}
STAGES=${STAGES:-"search capture empty pad embed prove attach reverify negatives"}
export HF_HOME=$ENV/hf
export VERITY_CORE_PYTHON=$PY
export VERITY_ADAPTER_REVISION=${VERITY_ADAPTER_REVISION:-$(cat $REPO/.veritor-commit 2>/dev/null || echo padint)}
export SP1_PROVER=cuda
export PYTHONPATH=$REPO/src:$REPO/zk/bench:$POC:$REPO
mkdir -p "$RES"
stage() { echo; echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] padint stage $*"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
chmod +x "$COVHOST" "$HOST" 2>/dev/null || true

if has search && [ -z "${WORKLOAD:-}" ]; then
  stage 0-search
  $PYV $POC/scripts/padint_prompt_search.py --checkpoints $POC/manifests/checkpoints.json --out $RES/prompt-search 2>&1 | grep -vE "^INFO|^WARNING|^\(EngineCore" | tail -25
  WORKLOAD=$RES/prompt-search/workload-B0-padint.json
fi
WORKLOAD=${WORKLOAD:-$RES/prompt-search/workload-B0-padint.json}
test -f "$WORKLOAD" || { echo "no workload at $WORKLOAD"; exit 2; }
N_REQ=$($PY -c "import json,sys; print(len(json.load(open(sys.argv[1]))['requests']))" "$WORKLOAD")

if [ -z "${RID:-}" ]; then
  TS=$(date -u +%Y%m%dT%H%M%SZ)
  RID="PADINT-B0-$TS"
fi
OUT=$RES/$RID
cd $POC

if has capture && [ ! -f "$OUT/summary.json" ]; then
  stage 1-capture "$RID" "($N_REQ requests, k=$N_REQ)"
  $PY -m e2e.run_composed --case B0 --out "$OUT" --challenge security --preparation selective --k "$N_REQ" \
      --workload "$WORKLOAD" --semantic-profile "$V3" --py-vllm "$PYV" --host "$HOST" --hf-home "$HF_HOME" \
      --cache-root /workspace/padint-cache 2>&1 | tail -40
  echo "verdict: $($PY -c "import json; d=json.load(open('$OUT/summary.json')); print(d['verdict'], d.get('verdict_reasons'))" 2>/dev/null || echo UNREADABLE)"
  $PY -c "
import json; d=json.load(open('$OUT/served_tokens.json'))
for r in d['requests']: print('  ', r['request_id'][:8], 'finish', r['finish_reason'], 'n', len(r['served_token_ids']), 'ids', r['served_token_ids'])"
fi

if has empty && [ ! -f "$OUT-empty-sample/empty_sample_check.json" ]; then
  stage 2-empty-sample
  $PY -m e2e.run_composed --case B0 --out "$OUT-empty-sample" --challenge security --preparation selective --k "$N_REQ" \
      --workload "$WORKLOAD" --semantic-profile "$V3" --py-vllm "$PYV" --host "$HOST" --hf-home "$HF_HOME" \
      --cache-root /workspace/padint-cache --reuse-adapter "$OUT/adapter" --s 1/1073741824 --coverage-size 0 2>&1 | tail -12
  $PY - "$OUT-empty-sample" "$OUT" <<'EOF'
import json, sys, pathlib
out, pos = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sv = json.loads((out / "verify" / "selective_verify.json").read_text())
served = {r["request_id"]: r for r in json.loads((pos / "served_tokens.json").read_text())["requests"]}
domains = sv.get("output_domains") or []
per_req = []
for d in domains:
    rid = d.get("request_id")
    n = len(served.get(rid, {}).get("served_token_ids") or [])
    per_req.append({"request_id": rid, "checked": d.get("checked"), "n_present": n, "finish_reason": served.get(rid, {}).get("finish_reason"),
                    "problems": d.get("problems") or []})
checked = sum(d.get("checked", 0) for d in domains)
certified = sv.get("certified_sample")
ok = checked > 0 and all(not d.get("problems") for d in domains) and (certified or 0) == 0
early = [p for p in per_req if p["finish_reason"] == "stop"]
doc = {"schema": "veritor/empty-sample-output-check/v2",
       "sample_union": sv.get("sample_union"), "certified": certified, "coverage": sv.get("coverage_sample"),
       "output_domain_leaves_checked": checked, "requests_checked": len(domains), "per_request": per_req,
       "early_terminating_requests_checked": len(early),
       "criterion": "certified sample == 0 (the empty-sample condition; integration-pad.md 7.2 reconciliation: NOT sample_union == 0) AND "
                    "output typing ran on every request's present token outputs, including the early-terminated request(s) whose present slots are fewer than max_tokens",
       "pass": bool(ok and len(early) > 0)}
(out / "empty_sample_check.json").write_text(json.dumps(doc, indent=1) + "\n")
print("EMPTY-SAMPLE-CHECK", json.dumps({k: doc[k] for k in ("sample_union", "certified", "coverage", "output_domain_leaves_checked", "requests_checked", "early_terminating_requests_checked", "pass")}))
for p in per_req: print("  ", p)
EOF
fi

BROOT=/workspace/padint-records/INT-B0-$RID/bundle
if has pad || has embed; then
  stage 3-bundle
  mkdir -p /workspace/padint-records
  [ -d "$BROOT" ] || tar xzf "$OUT/bundle/INT-B0-$RID.tar.gz" -C /workspace/padint-records
  ls "$BROOT" | head -3
fi

if has pad; then
  stage 3-pad-statements "($PROGRAM)"
  rm -rf $RES/pb-$RID/batches/pad
  $PY $POC/scripts/padint_pad_statements.py --record "$OUT" --bundle "$BROOT" --program "$PROGRAM" --out $RES/pb-$RID/batches/pad 2>&1 | tail -20
fi

if has embed; then
  stage 4-embed-statements
  SNAP=$(readlink -f $ENV/ckpt)
  WEIGHTS_SHA=d4d9c148303f41396ec906f1dc536da659c5944d9bba12e04db583932f5effa0
  if [ ! -f /workspace/padint-records/embed_tokens_weight.bin ]; then
    $PY - "$SNAP/model.safetensors" <<'EOF'
import json, struct, sys
raw = open(sys.argv[1], "rb")
(n,) = struct.unpack("<Q", raw.read(8))
header = json.loads(raw.read(n))
e = header["model.embed_tokens.weight"] if "model.embed_tokens.weight" in header else header["embed_tokens.weight"]
start, end = e["data_offsets"]; raw.seek(8 + n + start)
open("/workspace/padint-records/embed_tokens_weight.bin", "wb").write(raw.read(end - start)); print("sliced", e["shape"])
EOF
  fi
  echo "$WEIGHTS_SHA  /workspace/padint-records/embed_tokens_weight.bin" | sha256sum -c -
  rm -rf $RES/pb-$RID/batches/embed
  $PY $REPO/zk/bench/embed_relations.py --bundle "$BROOT" --weights /workspace/padint-records/embed_tokens_weight.bin \
      --weights-sha256 "$WEIGHTS_SHA" --label "$RID" --out $RES/pb-$RID/batches/embed --batches all 2>&1 | tail -8
fi

if has prove; then
  stage 5-prove "(cuda, cov host)"
  mkdir -p $RES/pb-$RID/proofs $RES/pb-$RID/verify
  MANIFESTS=""
  [ -f $RES/pb-$RID/batches/pad/manifest.json ] && MANIFESTS="$MANIFESTS --manifest $RES/pb-$RID/batches/pad/manifest.json"
  [ -f $RES/pb-$RID/batches/embed/manifest.json ] && MANIFESTS="$MANIFESTS --manifest $RES/pb-$RID/batches/embed/manifest.json"
  SP1_PROVER=cuda $PY $REPO/zk/bench/pb_prove.py --host "$COVHOST" $MANIFESTS --modes core,compressed --plan --keep-proofs \
      --proof-dir $RES/pb-$RID/proofs --out $RES/pb-$RID/cov-records.jsonl 2>&1 | tail -30
  stage 5b-verify-cpu
  $PY - "$COVHOST" "$RES/pb-$RID" <<'EOF'
import hashlib, json, os, subprocess, sys
sys.path.insert(0, "/workspace/veritor-padint/vllm-poc")
from e2e.run_composed import host_json
host, pb = sys.argv[1], sys.argv[2]
env = dict(os.environ); env["SP1_PROVER"] = "cpu"
ok_all = True
for mpath in (f"{pb}/batches/pad/manifest.json", f"{pb}/batches/embed/manifest.json"):
    if not os.path.exists(mpath):
        continue
    for entry in json.load(open(mpath))["batches"]:
        d = entry["dir"]
        key12 = hashlib.sha256(d.encode()).hexdigest()[:12]
        for mode in ("core", "compressed"):
            proofs = sorted(p for p in os.listdir(f"{pb}/proofs") if p.startswith(f"{key12}-{mode}-"))
            if not proofs:
                print(f"{d.rsplit('/',1)[-1]} {mode}: NO PROOF"); ok_all = False; continue
            r = subprocess.run([host, "verify", "--proof", f"{pb}/proofs/{proofs[0]}", "--statement", f"{d}/statement.bin"], capture_output=True, text=True, env=env)
            doc = host_json(r.stdout) if r.returncode == 0 else {"ok": False, "error": r.stderr[-400:]}
            json.dump(doc, open(f"{pb}/verify/verify-{key12}-{mode}.json", "w"), indent=1)
            good = bool(doc.get("ok")) and bool(doc.get("statement_match")); ok_all &= good
            print(f"{d.rsplit('/',1)[-1]} {mode}: ok={doc.get('ok')} statement_match={doc.get('statement_match')}")
print("CPU-VERIFY", "ALL-OK" if ok_all else "FAILURES")
EOF
fi

if has attach; then
  stage 6-attach+certificate
  BATCHES=""
  [ -f $RES/pb-$RID/batches/pad/manifest.json ] && BATCHES="$BATCHES --batches $RES/pb-$RID/batches/pad"
  [ -f $RES/pb-$RID/batches/embed/manifest.json ] && BATCHES="$BATCHES --batches $RES/pb-$RID/batches/embed"
  $PY -m e2e.cov_attach --record "$OUT" $BATCHES --records $RES/pb-$RID/cov-records.jsonl \
      --proof-dir $RES/pb-$RID/proofs --verify-dir $RES/pb-$RID/verify --host "$COVHOST" 2>&1 | tail -8
  cp "$COVHOST" "$OUT/bin/veritor-zk-host-cov"
  mkdir -p "$OUT/padint"
  cp $RES/pb-$RID/batches/pad/padint_statements.json "$OUT/padint/pad_statements.json" 2>/dev/null || true
  $PY -m e2e.certificate --record "$OUT" --require-coverage 2>&1 | tail -6; echo "certificate rc=${PIPESTATUS[0]}"
fi

if has reverify; then
  stage 7-reverify
  $PY -m e2e.verify_composed --dir "$OUT" --out "$OUT/reverify" --program rebuild --plan recorded --host "$OUT/bin/veritor-zk-host" --skip-bundle 2>&1 | tail -10
fi

if has negatives; then
  stage 8-negatives
  $PY -m e2e.negative_audit --record "$OUT" --out "$RES/$RID-negative-audit" --host "$OUT/bin/veritor-zk-host" 2>&1 | tail -8; echo "negative_audit rc=${PIPESTATUS[0]}"
  $PY $POC/scripts/padint_negatives.py --record "$OUT" --bundle "$BROOT" --batches $RES/pb-$RID/batches/pad --host "$COVHOST" \
      --out "$RES/$RID-padint-negatives" 2>&1 | tail -20; echo "padint_negatives rc=${PIPESTATUS[0]}"
  rm -rf "$OUT/padint/negatives"; cp -r "$RES/$RID-padint-negatives" "$OUT/padint/negatives" 2>/dev/null || true
fi
echo "RID=$RID"
echo PADINT_DONE
