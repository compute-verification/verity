#!/usr/bin/env bash
# Scratch, never to land: the checks of the adapter's two computed paths to the PoUW kernel sources, as steps of a run.
#
#   research run --on auto --cwd source -- bash -lc 'bash tools/move/kernelpath_scratch/prove.sh STEP...'
#   research run --on auto --cwd clone  -- bash -lc 'bash tools/move/kernelpath_scratch/prove.sh --at REV STEP...'
#
# It runs in the tree's root. Under --cwd source it writes only under $OUT and a temporary directory; the steps that write
# into the tree (--at, generator) are for a run's own clone. Each step's output is $OUT/STEP.log and its exit a line of
# $OUT/exits.txt; a failed step doesn't stop the next, and the script exits 1 when any failed.
#
# Steps: sync tools host heavy guard cold wheels boundaries root lints generator movecheck ratchet terms
# MOVES and RANGES, each "BASE TOP;BASE TOP;...", are what movecheck and ratchet take; the defaults are the first proof's.
set -u
: "${OUT:?run it under research run, or set OUT to a directory}"
mkdir -p "$OUT"
ROOT=$(pwd)
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
cp "$ROOT/tools/move/kernelpath_scratch/paths.py" "$d/paths.py"

MOVES=${MOVES:-21ba413aa08c54cd8897c4fb41f7834c56101109 054ce0152301da7a0480651b5c0435334eabb18e}
RANGES=${RANGES:-21ba413aa08c54cd8897c4fb41f7834c56101109 054ce0152301da7a0480651b5c0435334eabb18e}
HOST="tests/protocols/test_pouw_device.py tests/commit/test_row_tables_device.py tests/protocols/test_pouw_h1_registered_device.py
      tests/commit/test_history_tables_device.py tests/protocols/test_pouw_h1_tables_device.py tests/protocols/test_pouw_h1_nodes.py"
HEAVY="tests/check/test_pouw_ncp_v2_end_to_end.py tests/pipeline/test_serve_h1.py tests/protocols/test_pouw_h1_served.py"
LINTS="tests/lint/test_ruff.py tests/lint/test_ruff_vllm.py tests/lint/test_p09_layering.py tests/lint/test_mypy.py
       tests/test_imports_resolve.py tests/test_no_dead_modules.py"
ROOT_TESTS="tests/test_integration_boundaries.py tests/test_tree.py tests/test_doc_paths.py tests/test_gate_paths.py
            tests/test_module_strings.py tests/test_pythonpath_entries.py tests/test_public_boundary.py tests/test_vllm_boundary.py
            tests/test_vllm_public_api.py tests/lint/test_repo_retired_terms.py"
PY="uv run --no-sync"
NOCACHE="-p no:cacheprovider"

sync() { uv sync -q --all-packages --extra torch-cpu; }

tools() {
  for t in g++ c++ clang++ nvcc nvidia-smi; do echo "$t: $(command -v "$t" || echo none)"; done
  $PY python -c 'import torch; print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())'
  nproc; uname -m
}

# where this environment imports the two distributions and the prover package from, printed before the tests that read them
where() {
  echo "HEAD $(git rev-parse HEAD); PYTHONPATH is $([ -n "${PYTHONPATH:-}" ] && echo set || echo unset)"
  $PY python -c 'import verity, verity_vllm
from verity.protocols.pouw import prover
from verity_vllm.commit import row_tables_device as D
from verity_vllm.protocols.pouw import device as PD
print("verity:", verity.__file__); print("verity_vllm:", verity_vllm.__file__); print("prover:", prover.__file__)
print("CUDA_DIR:", D.CUDA_DIR); print("HERE:", PD.HERE)'
}

host() { where; cd adapters/vllm && $PY pytest $HOST -q -rs $NOCACHE; }

heavy() { where; cd adapters/vllm && $PY pytest $HEAVY -q -rs $NOCACHE --durations=8; }

# the same files as one suite under the read guard, which fails a suite that reads a file outside its declared inputs
guard() {
  where
  uv run tools/verity/check/suites.py adapters/vllm --logs "$OUT/guard-logs" --json "$OUT/guard.json" -- $HOST $HEAVY -rs
}

# every host library compiled from the directory into an empty cache, in the checkout's own environment
cold() { mkdir -p "$d/home-cold" && $PY env HOME="$d/home-cold" python "$d/paths.py" --build ${UNDER:-}; }

# outside a checkout: the distributions built (an sdist, and the wheel from it), installed into a fresh environment with
# no editable install, and the two paths read there from a directory outside the tree
wheels() {
  w="$d/wheels"; mkdir -p "$w/dist" "$w/home" "$w/cwd"
  uv build -q --out-dir "$w/dist" . || return 1
  uv build -q --out-dir "$w/dist" --package verity-vllm || return 1
  uv build -q --out-dir "$w/dist" --package verity-config || return 1
  ls -l "$w/dist"
  for f in "$w"/dist/*.whl "$w"/dist/*.tar.gz; do
    case "$f" in
      *.whl) names=$(python3 -m zipfile -l "$f" | awk '{print $1}') ;;
      *) names=$(tar -tzf "$f") ;;
    esac
    echo "$(basename "$f"): $(echo "$names" | grep -c '/pouw_cuda/.') files under a pouw_cuda/, $(echo "$names" | grep -c '/definitions/kernels/.*\.cuh$') .cuh under definitions/kernels/"
    echo "$names" | grep '/pouw_cuda/.' | sed 's/^/    /'
  done
  cp "$d/paths.py" "$w/cwd/paths.py"
  base=$($PY python -c 'import sys; print(sys._base_executable)')
  ( cd "$w/cwd" || exit 1
    uv venv -q --python "$base" "$w/venv" || exit 1
    uv pip install -q --python "$w/venv/bin/python" --index-url https://download.pytorch.org/whl/cpu torch || exit 1
    uv pip install -q --python "$w/venv/bin/python" "$w"/dist/verity-*.whl "$w"/dist/verity_vllm-*.whl "$w"/dist/verity_config-*.whl || exit 1
    echo "installed distributions:"; uv pip list --python "$w/venv/bin/python"
    echo "editable installs:"; uv pip list --python "$w/venv/bin/python" --editable
    HOME="$w/home" "$w/venv/bin/python" -I paths.py --build --outside "$ROOT" ${UNDER:-} )
}

boundaries() { $PY pytest tests/test_boundaries.py -q $NOCACHE; }

root() { $PY pytest $ROOT_TESTS -q -rs $NOCACHE; }

lints() { cd adapters/vllm && $PY pytest $LINTS -q -rs $NOCACHE; }

# the import listing's generator without --grow, in a clone: what it writes is the diff
generator() {
  $PY python tools/verity/check/import_order.py; rc=$?
  git diff --stat -- tests/import_order.toml
  git diff -- tests/import_order.toml > "$OUT/import_order.diff"
  echo "the generator's diff: $(wc -l < "$OUT/import_order.diff") lines ($OUT/import_order.diff)"
  git checkout -q -- tests/import_order.toml
  return $rc
}

movecheck() {
  rc=0
  while IFS= read -r -d ';' m; do
    [ -n "$m" ] || continue
    $PY python tools/verity/check/move_check.py $m < /dev/null; r=$?
    echo "move_check $m: exit $r"; [ $r -eq 0 ] || rc=1
  done <<< "$MOVES;"
  return $rc
}

# the no-growth lints on each range of RANGES; the step's exit is 1 when any range's is not 0
ratchet() {
  rc=0
  while IFS= read -r -d ';' range; do
    [ -n "$range" ] || continue
    python3 adapters/vllm/tools/ratchet_gate.py $range < /dev/null; r=$?
    echo "ratchet_gate $range: exit $r"; [ $r -eq 0 ] || rc=1
  done <<< "$RANGES;"
  return $rc
}

# the retired-terms lint on this tree, and whether its list stands in sorted order
terms() {
  python3 -c 'import json
fs = [e["file"] for e in json.load(open("tests/lint/allowlists/repo_retired_terms.json"))["entries"]]
print(len(fs), "entries;", sum(1 for a, b in zip(fs, fs[1:]) if b < a), "stand before their place in sorted order")
print([f for f in fs if "pouw_cuda" in f])'
  $PY pytest tests/lint/test_repo_retired_terms.py -q $NOCACHE
}

# one compound command, so the shell has read all of it before --at takes this file out of the tree
{
  if [ "${1:-}" = "--at" ]; then
    git checkout -q --detach "$2" || { echo "at $2 128" >> "$OUT/exits.txt"; exit 1; }
    shift 2
  fi
  git rev-parse HEAD > "$OUT/head.txt"
  fail=0
  for s in "$@"; do
    echo "== $s"
    ( cd "$ROOT" && "$s" ) > "$OUT/$s.log" 2>&1; rc=$?
    echo "$s $rc" >> "$OUT/exits.txt"
    tail -n 40 "$OUT/$s.log"
    echo "== $s: exit $rc"
    [ $rc -eq 0 ] || fail=1
  done
  echo "== exits at $(cat "$OUT/head.txt")"; cat "$OUT/exits.txt"
  exit $fail
}
