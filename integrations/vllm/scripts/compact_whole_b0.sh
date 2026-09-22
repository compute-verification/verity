#!/usr/bin/env bash
# [COMPACT-2] Step (c) on a CPU pod: whole-B0 validate_plan + Bound on the program the certificate of record pins
# (`Serve_v1{C={6f196795e564},LP=32,STEPS=15}`, record INTPAD2-B0-20260909T013519Z), compared with the certificate
# and the symbolic PLAN/PAD-SPEC table; oracle-equality ladder first; per-stage wall/RSS for VBENCH.
#
#   REPO=/workspace/compact2/veritor OUT=/workspace/compact2/out bash scripts/compact_whole_b0.sh [stage...]
#   stages (default: all): setup oracle whole p3 report
#
# Resources (measured 2026-09-09 on cpu3m 32 vCPU / 256 GB, AMD EPYC 7713; see docs/vllm-poc/compact-planning.md):
# whole-B0 root plan validate + Bound: see docs/data/compact/whole_b0_stages.json.  The pod is single-threaded Python;
# 32 GB is enough for `whole`; the `oracle` rung with compact=False needs ~ 2.6 KB/gate (the pre-§39 expanded path).
set -euo pipefail
REPO=${REPO:-/workspace/compact2/veritor}
OUT=${OUT:-/workspace/compact2/out}
PY=${PY:-/workspace/compact2/venv/bin/python}
RECORD=${RECORD:-$REPO/fixtures/results/INT-PAD-2/INTPAD2-B0-20260909T013519Z}
export PYTHONPATH="$REPO/src:$REPO"
cd "$REPO"
mkdir -p "$OUT/whole-b0" "$OUT/oracle"
stage() { echo; echo "=== [$(date -u +%H:%M:%S)] $*"; }
want() { [ $# -eq 0 ] || [[ " ${STAGES[*]} " == *" $1 "* ]]; }
STAGES=("$@")

if want setup; then
  stage machine; nproc; free -g | head -2; lscpu | grep -E "Model name|^CPU\(s\)"; cat /sys/fs/cgroup/memory.max 2>/dev/null || true
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null || curl -fsSL https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  [ -x "$PY" ] || { uv python install 3.12 >/dev/null 2>&1 || true; uv venv --python 3.12 "$(dirname "$(dirname "$PY")")" >/dev/null; }
  "$PY" -c "import numpy, networkx, psutil, pytest" 2>/dev/null || uv pip install --python "$PY" "numpy>=2.3" "networkx>=3.6" pytest psutil >/dev/null
  test -f "$RECORD/certificate.json"
fi

if want oracle; then
  stage "oracle: the touched test files (ladder + negatives; per-leaf reference + expanded adapter)"
  "$PY" -m pytest tests/veritor/plans/test_compact_boundary.py tests/veritor/plans/test_compact_refs.py tests/veritor/plans/test_validate.py \
    tests/veritor/plans/test_bound.py -q -p no:cacheprovider 2>&1 | tail -5
  stage "oracle: B0 bodies at growth-profile sizes (compact vs compact=False vs per-leaf), largest first that fits"
  # 7.56 M gates: the record's own failed size (20 GB cap).  The expanded path needs ~2.6 KB/gate -> ~20 GB here.
  "$PY" scripts/compact_whole_b0.py --oracle --out "$OUT/oracle/nl1-v1024-lp8s3" --nl 1 --set V=1024 --lp 8 --steps 3 --plans root,P3/cu=whole
  "$PY" scripts/compact_whole_b0.py --oracle --out "$OUT/oracle/nl2-v4096-lp8s3" --nl 2 --set V=4096 --lp 8 --steps 3 --plans root
fi

if want whole; then
  stage "whole-B0: root plan (bridge_diag's P1: one VU per root node, CU(inputs) + CU(calls)) validate + Bound + compare"
  "$PY" scripts/compact_whole_b0.py --out "$OUT/whole-b0/root" --plans root --record "$RECORD"
  cat "$OUT/whole-b0/root.passdiff.txt"
fi

if want p3; then
  stage "whole-B0: P3/cu=whole (operator VUs) validate + Bound + compare"
  "$PY" scripts/compact_whole_b0.py --out "$OUT/whole-b0/p3-whole" --plans P3/cu=whole --record "$RECORD"
  cat "$OUT/whole-b0/p3-whole.passdiff.txt"
fi

if want report; then
  stage "report: redo the symbolic side + comparison on the finished run (17 s), copy the small artifacts into the repo"
  "$PY" scripts/compact_whole_b0.py --recompare "$OUT/whole-b0/root.json" --out "$OUT/whole-b0/rootc" --plans root --record "$RECORD"
  mkdir -p docs/data/compact/whole-b0
  cp "$OUT/whole-b0/root.stages.json" "$OUT/whole-b0/root.log" "$OUT/whole-b0/root.rss.tsv" "$OUT/whole-b0/rootc.json" "$OUT/whole-b0/rootc.passdiff.txt" \
    docs/data/compact/whole-b0/
  "$PY" scripts/compact_stages_report.py "$OUT/whole-b0/root.stages.json" "$OUT/whole-b0/rootc.json" docs/data/compact/whole_b0_stages.json \
    --pod "${POD_DESC:-$(hostname)}" --commit "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  ls -la docs/data/compact/whole-b0/
fi
echo COMPACT-WHOLE-B0-OK
