#!/usr/bin/env bash
# The canonical top-k's stress and timing on one GPU node with the integration's pinned vLLM venv (pod_bootstrap.sh's
# venv312): two processes one after the other, then two at once, each running every case unpatched and canonical
# (stress.py check); their comparison (stress.py compare: summary.json); the plugin as vLLM loads it, through the
# indexer's entry points (stress.py plugin: plugin.json); and the timing (stress.py time: time.json).
#   research run --on NODE --project verity --source . --cwd source -- bash benchmarks/dsa_topk/run.sh
# PY: the venv's python (default /workspace/venv312/bin/python); OUT: where to write (research run sets it);
# DSA_TOPK_REPS, DSA_TOPK_DISTS: stress.py check's --reps and --dists; DSA_TOPK_SKIP_TIME=1: no timing.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/workspace/venv312/bin/python}
OUT=${OUT:?set OUT}
mkdir -p "$OUT"
check() {
  "$PY" "$HERE/stress.py" check --out "$OUT" --tag "$1" --reps "${DSA_TOPK_REPS:-4}" \
    --dists "${DSA_TOPK_DISTS:-randn,bf16,int,relu,signed,edge}" > "$OUT/check.$1.log" 2>&1
}
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv > "$OUT/gpu.csv"
check p0
check p1
check p2 & a=$!
check p3 & b=$!
wait "$a"
wait "$b"
"$PY" "$HERE/stress.py" compare --out "$OUT" | tee "$OUT/compare.log"
site=$(mktemp -d)
"$PY" "$HERE/dsa_topk.py" site "$site" > /dev/null
PYTHONPATH="$site" VERITY_DSA_TOPK=canonical "$PY" "$HERE/stress.py" plugin --out "$OUT" --reps 2 --dists randn,relu \
  > "$OUT/plugin.log" 2>&1
rm -rf "$site"
[ "${DSA_TOPK_SKIP_TIME:-0}" = 1 ] || "$PY" "$HERE/stress.py" time --out "$OUT" > "$OUT/time.log" 2>&1
python3 - "$OUT" "$HERE/../harness" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
sys.path.insert(0, sys.argv[2])
from verity_bench import kit
res = kit.writer(out)
v = json.loads((out / "summary.json").read_text())["verdict"]
plugin = json.loads((out / "plugin.json").read_text())
ok = v["canonical_identical"] and v["canonical_is_reference"] and v["short_rows_unchanged"] and plugin["ok"]
res.validation("passed" if ok else "failed", "stress.py compare and plugin: canonical rows identical across processes, "
               "the reference's, short rows unchanged, and the plugin as vLLM loads it", json.dumps(v))
t = out / "time.json"
rows = {r["case"]: r for r in json.loads(t.read_text())["rows"]} if t.exists() else {}
row = rows.get("prefill-8192x32768")
card = res.stamp["card"]
card = "cpu" if card == "none" else card.split(", ")[0].removeprefix("NVIDIA ").lower().replace(" ", "-")
res.measurement("proofs/dsa-topk-added-device-us", row and row["added_us"] if ok else None, "device microseconds a call",
                mark="measured", member=f"glm-5.3-{card}",
                why=None if ok and row else "no timing" if ok else "the canonical rows failed the stress",
                extra={"case": "prefill-8192x32768", "shapes": "GLM-5.3",
                       "unpatched_us": row and row["unpatched_us"], "best": row and row["best"],
                       "cases": {c: r["added_us"] for c, r in rows.items()}})
EOF
