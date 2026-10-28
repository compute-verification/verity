#!/usr/bin/env bash
# Dither's kernel-level standard number on one leased RTX PRO 6000 (kbench.py; cwd = the source tree's root). The C API over
# cgemm.cu is built and SASS-gated outside the lease; lease 1 runs cgemm.cu's device gate at every shape (KB_SHAPES, default
# all 20); cgemm_ref.py then recomputes the dumped tiles on the CPU; only if every gate and the reference passed does lease 2
# time the arms, interleaved. summary writes result.json (bench-result/v1) and out/table.md. Clocks are read, never set.
#
#   RESEARCH_RUN_DIR=DIR OUT=DIR/out GPU_LEASE_WHO=WHO bash benchmarks/hash/dither/kbench.sh
set -uo pipefail
export O=${OUT:?}
CUDA=${CUDA:-/usr/local/cuda-13.0}
export PATH=$CUDA/bin:$PATH
[ -x /workspace/jobs/venv312/bin/python3 ] && PY=${PY:-/workspace/jobs/venv312/bin/python3}
PY=${PY:-$(. benchmarks/harness/env.sh && echo "$VENV/bin/python3")} || exit 2
REF=${KB_REF_PYTHON:-"uv run --no-project --with numpy --with python-sat --with-editable benchmarks/pouw/exact_kit python3"}
S=$(mktemp -d /tmp/kbench.XXXXXX); trap 'rm -rf "$S"' EXIT
git rev-parse HEAD > "$O/commit.txt"
nvcc -O3 -std=c++17 --no-compress -gencode arch=compute_120a,code=sm_120a -Xptxas -v -Xcompiler -fPIC -shared \
  -o "$S/libkcapi.so" benchmarks/hash/dither/kcapi.cu -lcuda -lcublasLt -lnvidia-ml > "$O/build.log" 2>&1 \
  || { echo "kbench: BUILD_FAIL"; tail -30 "$O/build.log"; exit 4; }
python3 benchmarks/pouw/harness/sass_gate.py "$S/libkcapi.so" --json "$O/sass_gate.json" > "$O/sass_gate.txt" 2>&1
# as crun.sh: the binary holds egemm.cu's never-launched kernels, which the gate may flag; pass only if every flag is in one
python3 - "$O/sass_gate.json" > "$O/sass_scope.txt" 2>&1 <<'EOF'
import json, re, sys
UNLAUNCHED = {"f_plain", "o_plain", "r_plain", "d_duo32", "d_duo2_32", "d_add3_32", "d_mul2_32", "w_duo2_32"}
g = json.load(open(sys.argv[1]))
bad = []
for b in g["binaries"]:
    if b["ptx_bad_n"] or not b["cubins"]:
        bad.append(f"{b['path']}: ptx_bad {b['ptx_bad_n']}, cubins {b['cubins']}")
    for o in b["offending"]:
        m = re.match(r"_Z\d+(\w+?)4Args$", o["function"])
        if not m or m.group(1) not in UNLAUNCHED:
            bad.append(f"{o['function']} {o['address']}: {o['instruction']} ({o.get('why', '')})")
if not g["ok"] and not all(b["offending"] for b in g["binaries"]):
    bad.append(f"the gate failed without a flagged instruction: {g.get('error')}")
print("\n".join(bad) if bad else "pass")
sys.exit(7 if bad else 0)
EOF
sg=$?
echo "sass_gate rc $sg" | tee "$O/rc.txt"
[ "$sg" = 0 ] || { cat "$O/sass_scope.txt"; exit 7; }
SH=${KB_SHAPES:+--shapes $KB_SHAPES}
export S PY SH
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 15 -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  timeout 840 "$PY" benchmarks/hash/dither/kbench.py gate --lib "$S/libkcapi.so" --dir "$S/gates" --out "$O" $SH'
echo "lease 1 (gates) rc $?" | tee -a "$O/rc.txt"
dirs=$(ls -d "$S"/gates/*/ 2>/dev/null)
for d in $dirs; do cp "$d/meta.json" "$O/gate_$(basename "$d")_meta.json" 2>/dev/null; done
# shellcheck disable=SC2086
$REF benchmarks/pouw/edited_gemm/cgemm_ref.py $dirs > "$O/ref.jsonl" 2> "$O/ref.err"
ref=$?
echo "ref rc $ref" | tee -a "$O/rc.txt"
if [ "$ref" != 0 ] || grep -q 'rc [1-9]' "$O/rc.txt" || grep -q '"rc": [1-9]' "$O/gate.jsonl"; then
  echo "kbench: a gate failed, nothing timed"; cat "$O/rc.txt"; tail -5 "$O/gate.jsonl" "$O/ref.jsonl" "$O/ref.err"
  exit 5
fi
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min 25 -- bash -c '
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  timeout 1440 "$PY" benchmarks/hash/dither/kbench.py bench --lib "$S/libkcapi.so" --out "$O" --reps "${KB_REPS:-9}" $SH
  rc=$?; kill $smi; exit $rc'
b=$?
echo "lease 2 (bench) rc $b" | tee -a "$O/rc.txt"
[ "$b" = 0 ] || exit 6
python3 benchmarks/hash/dither/kbench.py summary --out "$O" --result "${RESEARCH_RUN_DIR:?}/result.json" \
  --run-id "${RESEARCH_RUN_ID:-local}" --ref-rc "$ref"
