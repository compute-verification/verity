#!/usr/bin/env bash
# The WP1 exhaustive comparison as one recorded pod run (stage wp1.exhaustive).  Run from the root of the shipped checkout:
#
#   uv run research run --on vy-sp1 --project verity --source . --exclusive --stage wp1.exhaustive --require-result \
#     -- bash benchmarks/numerical/tests/reference/exhaustive_pod.sh
#
# 1. clones Hawkeye's simulator at the pinned commit (verity/ml/circuits/silicon/hawkeye/PIN.json) and checks the eight blob shas with git hash-object;
# 2. makes a Python 3.12 venv with a CPU libtorch (the pinned sources include <torch/torch.h>), pytest and z3;
# 3. compiles hawkeye_harness.cpp against the unmodified pinned sources;
# 4. runs exhaustive_driver.py (the fast suite first, then steps a-g) writing wp1_exhaustive.json and result.json to $RESEARCH_RUN_DIR.
set -euo pipefail

PY=${WP1_PYTHON:-/root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12}
UV=${WP1_UV:-/root/.local/bin/uv}
WORK=${WP1_WORK:-/workspace/wp1}
OUT=${RESEARCH_RUN_DIR:-/tmp/wp1}
RANDOM_CASES=${WP1_RANDOM:-1000000}
PIN=verity/ml/circuits/silicon/hawkeye/PIN.json
HERE=benchmarks/numerical/tests/reference
mkdir -p "$WORK" "$OUT"

# 1. the pinned source
COMMIT=$("$PY" -c "import json; print(json.load(open('$PIN'))['commit'])")
REPO=$("$PY" -c "import json; print(json.load(open('$PIN'))['repository'])")
HAWKEYE="$WORK/hawkeye"
if [ ! -d "$HAWKEYE/.git" ]; then git clone -q "$REPO" "$HAWKEYE"; fi
git -C "$HAWKEYE" checkout -q "$COMMIT"
{
  echo "commit $(git -C "$HAWKEYE" rev-parse HEAD)"
  "$PY" -c "import json; [print(sha, path) for path, sha in json.load(open('$PIN'))['files'].items()]" | while read -r want path; do
    got=$(git -C "$HAWKEYE" hash-object "$path")
    if [ "$got" = "$want" ]; then echo "ok   $got $path"; else echo "BAD  want $want got $got $path"; exit 3; fi
  done
} | tee "$OUT/pin_check.txt"
grep -q BAD "$OUT/pin_check.txt" && { echo "pin check failed"; exit 3; }

# 2. the venv: CPU torch (headers + libtorch for the harness), pytest, z3
VENV="$WORK/venv"
if [ ! -x "$VENV/bin/python" ]; then "$UV" venv -q --python "$PY" "$VENV"; fi
# torch 2.4.1: the newest CPU wheel whose headers compile with the image's g++ 11.4 (2.14's ATen/ExpandUtils.h does not)
"$UV" pip install -q --python "$VENV/bin/python" --index-url https://download.pytorch.org/whl/cpu "torch==2.4.1"
"$UV" pip install -q --python "$VENV/bin/python" pytest z3-solver
TORCH_BASE=$("$VENV/bin/python" -c "import torch, os; print(os.path.dirname(torch.__file__))")
ABI=$("$VENV/bin/python" -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
"$VENV/bin/python" -c "import torch, z3, pytest, sys; print('torch', torch.__version__, 'z3', z3.get_version_string(), 'pytest', pytest.__version__, 'python', sys.version.split()[0])" | tee "$OUT/versions.txt"
g++ --version | head -1 | tee -a "$OUT/versions.txt"

# 3. the harness against the unmodified pinned sources
BUILD="$OUT/build"
mkdir -p "$BUILD"
CXXFLAGS=(-std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI="$ABI" -I"$HAWKEYE" -I"$TORCH_BASE/include" -I"$TORCH_BASE/include/torch/csrc/api/include")
g++ "${CXXFLAGS[@]}" -c "$HERE/hawkeye_harness.cpp" -o "$BUILD/harness.o" &
g++ "${CXXFLAGS[@]}" -c "$HAWKEYE/src/Ampere_simulator.cpp" -o "$BUILD/Ampere_simulator.o" &
g++ "${CXXFLAGS[@]}" -c "$HAWKEYE/src/gfloat.cpp" -o "$BUILD/gfloat.o" &
g++ "${CXXFLAGS[@]}" -c "$HAWKEYE/utils/utils.cpp" -o "$BUILD/utils.o" &
wait
g++ "$BUILD"/harness.o "$BUILD"/Ampere_simulator.o "$BUILD"/gfloat.o "$BUILD"/utils.o \
  -L"$TORCH_BASE/lib" -ltorch -ltorch_cpu -lc10 -Wl,-rpath,"$TORCH_BASE/lib" -pthread -o "$OUT/hawkeye_harness"
printf '3f800000 %s %s\n' "$(printf '3f80%.0s' $(seq 16))" "$(printf '3f80%.0s' $(seq 16))" | "$OUT/hawkeye_harness" | tee "$OUT/harness_smoke.txt"
# 1.0 + 16 * 1.0 = 17.0 = 0x41880000 through the adapter and the finalisation; the wrapper column is "-" (nonzero accumulator)
grep -q "^41880000 41880000 -$" "$OUT/harness_smoke.txt"

# 4. the driver
export PYTHONPATH=.:benchmarks/numerical/python
"$VENV/bin/python" "$HERE/exhaustive_driver.py" --out "$OUT" --harness "$OUT/hawkeye_harness" --random "$RANDOM_CASES" --pytest
