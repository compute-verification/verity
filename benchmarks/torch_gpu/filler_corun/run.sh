#!/usr/bin/env bash
# corun.py on one leased GPU of an H200 node, from a source tree at a pushed commit (cwd: the tree's root).
# Outside the lease: corun.cu built with the node's nvcc into scratch, refused on a ptxas warning (serialized wgmma
# among them), a spill or local memory, and without IGMMA.64x256x32 in each GEMM kernel; ptxas's report and the SASS
# opcode counts kept in $OUT; a Python with torch and pynvml (VERITY_BENCH_VENV, the node's /workspace/cp/venv312, else
# torch 2.13.0+cu129 in scratch). Then one lease of at most LEASE_MIN (default 15) minutes for corun.py with this
# script's arguments, nvidia-smi sampling the card beside it.
#
#   research run --on NODE --project verity --source . --cwd source --env GPU_LEASE_WHO=WHO -- \
#     bash benchmarks/torch_gpu/filler_corun/run.sh [corun.py's options]
#
# Exit: corun.py's (0 every gate passed and every item counted, 1 one didn't), 3 the build or the venv failed, else
# the lease's code.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run under research run}; OUT=$D/out; mkdir -p "$OUT"
ROOT=$PWD HERE=benchmarks/torch_gpu/filler_corun
TMP=/workspace/tmp; [ -w "$TMP" ] || TMP=${TMPDIR:-/tmp}
S=$(mktemp -d "$TMP/tmp-filler-corun.XXXXXX"); trap 'rm -rf "$S"' EXIT
fail() { tail -30 "$2" 2>/dev/null; echo "$1 3" | tee -a "$OUT/status.txt"; exit 3; }
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"

CUDA=${CUDA:-/usr/local/cuda}
FLAGS=(-O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v -cubin ${CORUN_NVCC_FLAGS:-})
{ "$CUDA/bin/nvcc" --version; echo "nvcc ${FLAGS[*]} -o corun.cubin corun.cu"; } > "$OUT/toolchain.txt"
"$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$S/corun.cubin" "$HERE/corun.cu" > "$OUT/ptxas.txt" 2>&1 || fail build "$OUT/ptxas.txt"
"$CUDA/bin/cuobjdump" -sass "$S/corun.cubin" > "$S/corun.sass" || fail sass "$S/corun.sass"
python3 - "$OUT/ptxas.txt" "$S/corun.sass" > "$OUT/build-gate.txt" <<'EOF' || fail build-gate "$OUT/build-gate.txt"
import re, sys, collections
ptx, sass = open(sys.argv[1]).read(), open(sys.argv[2]).read()
bad = [l for l in ptx.splitlines() if re.search(r"warning|serializ", l, re.I)]
for name, body in re.findall(r"Compiling entry function '(\w+)'.*?\n(.*?)(?=Compiling entry|\Z)", ptx, re.S):
    m = re.search(r"Used (\d+) registers", body)
    s = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", body)
    st = re.search(r"(\d+) bytes stack frame", body)
    print(f"{name}: {m.group(1) if m else '?'} registers, spills {s.groups() if s else '?'}, stack {st.group(1) if st else '?'}")
    if (s and s.groups() != ("0", "0")) or (st and st.group(1) != "0"):
        bad.append(f"{name} spills or has a stack frame")
fns = re.split(r"\n\s+Function : ", sass)
for f in fns[1:]:
    name = f.split("\n", 1)[0].strip()
    ops = collections.Counter(re.findall(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Za-z0-9_.]+)", f))
    tensor = {k: v for k, v in ops.items() if "GMMA" in k}
    print(f"{name}: {sum(ops.values())} instructions, tensor {tensor}, LDL/STL {ops.get('LDL', 0) + ops.get('STL', 0)}")
    if name.startswith("gemm_") and not any(k.startswith("IGMMA.64x256x32") for k in tensor):
        bad.append(f"{name} has no IGMMA.64x256x32")
    if any(k.startswith(("LDL", "STL")) for k in ops):
        bad.append(f"{name} uses local memory")
for b in bad:
    print("REFUSED", b)
sys.exit(1 if bad else 0)
EOF
cat "$OUT/build-gate.txt"
sha256sum "$S/corun.cubin" | sed "s#$S/##" > "$OUT/cubin.sha256"

PY=${VERITY_BENCH_VENV:+$VERITY_BENCH_VENV/bin/python3}
for v in /workspace/cp/venv312 /workspace/jobs/venv312; do [ -z "${PY:-}" ] && [ -x "$v/bin/python3" ] && PY=$v/bin/python3; done
if ! "${PY:-python3}" -c 'import torch, pynvml, numpy' 2>/dev/null; then
  export UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/cache/uv}
  { { uv venv -q -p 3.12 "$S/venv" || uv venv -q "$S/venv"; } && uv pip install -q -p "$S/venv/bin/python3" \
      --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu129 torch==2.13.0+cu129 \
      numpy nvidia-ml-py; } > "$OUT/venv.txt" 2>&1 || fail venv "$OUT/venv.txt"
  PY=$S/venv/bin/python3
fi
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)' > "$OUT/python.txt" 2>&1
export PYTHONPATH=$ROOT/tools/config PYTHONDONTWRITEBYTECODE=1
gpu-lease status > "$OUT/leases-before.txt" 2>&1
echo "run.sh: asking for one GPU at $(date -u +%H:%M:%SZ)"
export OUT PY S HERE
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${LEASE_MIN:-15}" -- bash -c '
  echo "run.sh: leased $GPU_LEASE_UUID at $(date -u +%H:%M:%SZ)"
  q=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,temperature.gpu
  nvidia-smi --query-gpu=$q --format=csv ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} > "$OUT/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv -lms 200 \
    ${GPU_LEASE_UUID:+-i "$GPU_LEASE_UUID"} > "$OUT/smi.csv" 2>&1 &
  smi=$!
  timeout $((60 * ${LEASE_MIN:-15} - 30)) "$PY" "$HERE/corun.py" --cubin "$S/corun.cubin" --out "$OUT" "$@" \
    > "$OUT/corun.log" 2>&1
  r=$?
  kill $smi 2>/dev/null
  echo "run.sh: corun.py exited $r at $(date -u +%H:%M:%SZ)"
  exit $r' bash "$@"
rc=$?
tail -40 "$OUT/corun.log"
echo "corun $rc" | tee -a "$OUT/status.txt"
exit $rc
