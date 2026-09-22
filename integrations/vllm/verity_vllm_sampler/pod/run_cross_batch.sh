#!/bin/bash
# R1 production-side complement: replay the cap-64 records under OTHER batch compositions (max_num_seqs 8
# and 1) and run sampler.select on those rows.  Same trajectory, different batching -> are the logits rows
# and the unforced sampler choice unchanged?
#   W3_VENV=/workspace/venv-nightly bash verity_vllm_sampler/pod/run_cross_batch.sh
set -uo pipefail
source "$(dirname "$0")/env.sh"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

cross() {  # name cap
  local name=$1 cap=$2
  local dir=$W3_RESULTS/replay/logits_${name}_at_c$cap
  local out=$W3_RESULTS/analysis/select_${name}_at_c${cap}_auto.json
  if [ ! -s "$dir/index.json" ]; then
    echo "[$(date -u +%H:%M:%SZ)] replay $name at cap $cap"
    python -m verity_vllm_sampler.replay_logits --production "$W3_RESULTS/prod/$name.json" --out-dir "$dir" --max-num-seqs "$cap" \
        > "$W3_RESULTS/logs/replay_${name}_at_c$cap.log" 2>&1
    echo "[$(date -u +%H:%M:%SZ)] done   rc=$? $(grep REPLAY_DONE "$W3_RESULTS/logs/replay_${name}_at_c$cap.log" | cut -c1-220)"
  fi
  if [ ! -s "$out" ]; then
    python -m verity_vllm_sampler.analyze --production "$W3_RESULTS/prod/$name.json" --logits-dir "$dir" --out "$out" --same-runtime \
        > "$W3_RESULTS/logs/analyze_${name}_at_c$cap.log" 2>&1
    echo "[$(date -u +%H:%M:%SZ)] done   analyze rc=$? $(grep ANALYZE_DONE "$W3_RESULTS/logs/analyze_${name}_at_c$cap.log" | cut -c1-260)"
  fi
  # logits rows at the same (request, position): identical bytes across batch compositions?
  python - "$W3_RESULTS/replay/logits_$name" "$dir" "$W3_RESULTS/analysis/rows_${name}_c64_vs_c$cap.json" <<'EOF'
import json, os, sys
import numpy as np
a, b, out = sys.argv[1:4]
ia, ib = json.load(open(os.path.join(a, "index.json"))), json.load(open(os.path.join(b, "index.json")))
res = {"a": a, "b": b, "vllm_a": ia.get("vllm_version"), "vllm_b": ib.get("vllm_version"), "requests": 0, "positions": 0,
       "identical_rows": 0, "rows_differing": 0, "max_abs_diff_bf16_as_f32": 0.0, "first_diff": None}
for rid, infa in ia["requests"].items():
    infb = ib["requests"].get(rid)
    if infb is None:
        continue
    A, B = np.load(os.path.join(a, infa["file"])), np.load(os.path.join(b, infb["file"]))
    pa = {p: i for i, p in enumerate(infa["positions"])}
    res["requests"] += 1
    for j, p in enumerate(infb["positions"]):
        if p not in pa:
            continue
        ra, rb = A[pa[p]], B[j]
        res["positions"] += 1
        if np.array_equal(ra, rb):
            res["identical_rows"] += 1
        else:
            res["rows_differing"] += 1
            fa = (ra.astype(np.uint32) << 16).view(np.float32); fb = (rb.astype(np.uint32) << 16).view(np.float32)
            d = float(np.max(np.abs(fa - fb)))
            res["max_abs_diff_bf16_as_f32"] = max(res["max_abs_diff_bf16_as_f32"], d)
            if res["first_diff"] is None:
                n = int(np.sum(ra != rb))
                res["first_diff"] = {"request_id": rid, "pos": int(p), "n_scalars_differing": n, "max_abs_diff": d}
json.dump(res, open(out, "w"), indent=1)
print("ROWS_COMPARE", json.dumps({k: v for k, v in res.items() if k not in ("a", "b")}))
EOF
}

for name in topk_topp_s1_c64_r0 greedy_c64_r0 temp_s1_c64_r0; do
  cross $name 8
done
cross topk_topp_s1_c64_r0 1
echo "CROSS_BATCH_DONE $(date -u +%H:%M:%SZ)"
