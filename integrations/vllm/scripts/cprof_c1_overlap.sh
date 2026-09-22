#!/usr/bin/env bash
# Commitment engine C1 (2026-09-17): does a spare core running the compiled engine slow the GPU replay?
# Runs the as-shipped replay (overlap mode A) alone, then with `vce` hashing the B0 store in a continuous loop pinned to
# 1 core, then 4 cores (continuous = worst case; the engine's real duty cycle for this workload is ~2 %).
# Prints CPERF-C1-OVERLAP-JSON with run_s / forward_s / process_wall_s per condition.
set -uo pipefail
PY=/workspace/venv-cu129/bin/python
SNAP=$(pwd)
OUT=/workspace/cp/cperf/out/c1ov
VCE=$SNAP/verity_vllm/commit_engine/bin/vce-x86_64-linux-musl
PLAN=/workspace/cp/cperf/vce/plan.bin; PACK=/workspace/cp/cperf/vce/pack.bin
chmod +x "$VCE"; mkdir -p "$OUT"
export PYTHONPATH=$SNAP/vllm-poc:$SNAP/src HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "== host"; hostname; lscpu | grep -E "Model name|^CPU\(s\)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; uptime
[ -f "$PLAN" ] || { echo "no plan at $PLAN"; exit 2; }
run_A() {  # $1 = label
  $PY scripts/cprof_overlap.py --production /workspace/cp/cperf/B0/production --out-root "$OUT/$1" \
      --checkpoints manifests/checkpoints.json --arena-gb 4 --skip B,C,D --rev "c1ov-$1" > "$OUT/$1.log" 2>&1
  echo "A[$1] exit=$?"
}
run_A alone1
run_A alone2
# background hashing loop, 1 core
( taskset -c 31 bash -c "while [ ! -f $OUT/stop ]; do $VCE --plan $PLAN --pack $PACK --threads 1 --reps 1 >/dev/null 2>>$OUT/vce1.err; done" ) &
BG=$!
sleep 2; run_A vce1core; touch "$OUT/stop"; wait $BG 2>/dev/null; rm -f "$OUT/stop"
( taskset -c 28-31 bash -c "while [ ! -f $OUT/stop ]; do $VCE --plan $PLAN --pack $PACK --threads 4 --reps 1 >/dev/null 2>>$OUT/vce4.err; done" ) &
BG=$!
sleep 2; run_A vce4core; touch "$OUT/stop"; wait $BG 2>/dev/null; rm -f "$OUT/stop"
run_A alone3
grep -c TIME "$OUT/vce1.err" "$OUT/vce4.err" 2>/dev/null
$PY - <<'EOF'
import json, os, re
OUT = "/workspace/cp/cperf/out/c1ov"
res = {}
for lab in ("alone1", "alone2", "vce1core", "vce4core", "alone3"):
    p = f"{OUT}/{lab}/overlap.json"
    if not os.path.exists(p):
        res[lab] = None
        continue
    a = json.load(open(p)).get("A_replay_as_shipped", {})
    t = a.get("timing", {}) or {}
    res[lab] = {"rc": a.get("rc"), "process_wall_s": a.get("process_wall_s"), "run_s": t.get("run_s"), "forward_s": (t.get("capture") or {}).get("forward_s"),
                "engine_build_s": t.get("engine_build_s"), "emit_s": (t.get("capture") or {}).get("emit_s"), "commit_s": t.get("commit_s"), "commitments_sha": (a.get("commitments") or {}).get("sha256", "")[:16]}
for lab in ("vce1", "vce4"):
    p = f"{OUT}/{lab}.err"
    if os.path.exists(p):
        ts = [float(m.group(1)) for m in re.finditer(r"median_s=([0-9.]+)", open(p).read())]
        res[f"{lab}_iterations"] = len(ts)
        res[f"{lab}_median_s_under_replay"] = sorted(ts)[len(ts) // 2] if ts else None
print("CPERF-C1-OVERLAP-JSON " + json.dumps(res, separators=(",", ":"), default=str))
EOF
