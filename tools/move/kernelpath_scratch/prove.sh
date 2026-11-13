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
# Steps: sync tools host heavy guard cold sheets wheels boundaries root lints generator movecheck ratchet terms sites remedy
# SITES_BASE is the commit sites reads the retired-terms list's growth from.
# MOVES and RANGES, each "BASE TOP;BASE TOP;...", are what movecheck and ratchet take; the defaults are the first proof's.
set -u
: "${OUT:?run it under research run, or set OUT to a directory}"
mkdir -p "$OUT"
ROOT=$(pwd)
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
cp "$ROOT"/tools/move/kernelpath_scratch/{paths,sheets,sites}.py "$d/"
mkdir "$d/plug" && cp "$ROOT/tools/move/kernelpath_scratch/where_plugin.py" "$d/plug/"

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
from verity_vllm.commit import row_tables_device as D
from verity_vllm.protocols.pouw import device as PD
prover = PD.prover
print("verity:", verity.__file__); print("verity_vllm:", verity_vllm.__file__); print("prover:", prover.__name__, prover.__file__)
print("CUDA_DIR:", D.CUDA_DIR); print("HERE:", PD.HERE)'
}

# the plugin's directory goes last on the path, so it changes nothing about where the tests import from; it prints, inside
# the pytest process, the files the tests' own imports gave
host() {
  where
  cd adapters/vllm && PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$d/plug" $PY pytest $HOST -q -rs $NOCACHE -p where_plugin
}

heavy() { where; cd adapters/vllm && $PY pytest $HEAVY -q -rs $NOCACHE --durations=8; }

# the same files as one suite under the read guard, which fails a suite that reads a file outside its declared inputs
guard() {
  where
  uv run tools/verity/check/suites.py adapters/vllm --logs "$OUT/guard-logs" --json "$OUT/guard.json" -- $HOST $HEAVY -rs
}

# every host library compiled from the directory into an empty cache, in the checkout's own environment
cold() { mkdir -p "$d/home-cold" && $PY env HOME="$d/home-cold" python "$d/paths.py" --build ${UNDER:-}; }

# what the data sheets' loaders find in the checkout's own environment: from the tree's root, where `python -c` puts the
# tree first on the path, and from a script outside the tree, which imports verity the way the environment hands it
sheets() {
  echo "-- from the tree's root (python -c)"
  $PY python -c 'import runpy, sys; runpy.run_path(sys.argv[1])' "$d/sheets.py"
  echo "-- a script outside the tree"
  $PY python "$d/sheets.py"
}

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
    echo "$(basename "$f"): $(echo "$names" | grep -c '/pouw_cuda/.') files under a pouw_cuda/, $(echo "$names" | grep -c -E '/(definitions|circuits)/kernels/.*\.cuh$') .cuh under a kernels/ of the circuits"
    echo "$names" | grep '/pouw_cuda/.' | sed 's/^/    /'
    echo "    $(echo "$names" | grep -c 'hardware/cards/.*\.toml$') sheets under a hardware/cards/, $(echo "$names" | grep -c 'hardware/nodes/.*\.toml$') under a hardware/nodes/"
  done
  cp "$d/paths.py" "$d/sheets.py" "$w/cwd/"
  base=$($PY python -c 'import sys; print(sys._base_executable)')
  ( cd "$w/cwd" || exit 1
    uv venv -q --python "$base" "$w/venv" || exit 1
    uv pip install -q --python "$w/venv/bin/python" --index-url https://download.pytorch.org/whl/cpu torch || exit 1
    uv pip install -q --python "$w/venv/bin/python" "$w"/dist/verity-*.whl "$w"/dist/verity_vllm-*.whl "$w"/dist/verity_config-*.whl || exit 1
    echo "installed distributions:"; uv pip list --python "$w/venv/bin/python"
    echo "editable installs:"; uv pip list --python "$w/venv/bin/python" --editable
    HOME="$w/home" "$w/venv/bin/python" -I paths.py --build --outside "$ROOT" ${UNDER:-}; rc=$?
    echo "-- the data sheets, in the same environment"
    HOME="$w/home" "$w/venv/bin/python" -I sheets.py
    exit $rc )
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

# the no-growth lints on each range of RANGES, the gate as the range's base has it (as the check runs it); the step's exit
# is 1 when any range's is not 0
ratchet() {
  rc=0
  while IFS= read -r -d ';' range; do
    [ -n "$range" ] || continue
    git show "${range%% *}:adapters/vllm/tools/ratchet_gate.py" > "$d/gate.py" || { rc=1; continue; }
    python3 "$d/gate.py" $range < /dev/null; r=$?
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

sites() { $PY python "$d/sites.py" "$SITES_BASE"; }

# The other way to pass the gate, tried in the run's own clone with commits that are never pushed: a change before the move
# records the three row_table sources' sites at the adapter's path, and the mover's commit carries the three entries to the
# new path as it carries ncp2_fused.cu's. REMEDY_BASE is the commit under the mover's commit REMEDY_MOVER, REMEDY_PICKS the
# hand commits above it, REMEDY_SAME a commit whose tree the result should equal. The gate runs as REMEDY_BASE has it.
remedy() {
  list=tests/lint/allowlists/repo_retired_terms.json
  old=adapters/vllm/verity_vllm/protocols/pouw/pouw_cuda; new=verity/protocols/pouw/prover/kernels/pouw_cuda
  g="git -c user.name=kernelpath -c user.email=kernelpath@users.noreply.github.com"
  git show "$REMEDY_BASE:adapters/vllm/tools/ratchet_gate.py" > "$d/gate.py" || return 1
  add() {
    python3 - "$list" "$1" <<'EOF'
import sys
path, at = sys.argv[1:]
text = open(path).read()
line = f'    {{"file": "{at}/ncp2_fused.cu", "count": 4}},\n'
assert text.count(line) == 1, (at, text.count(line))
more = "".join(f'    {{"file": "{at}/{f}", "count": {n}}},\n'
               for f, n in (("row_table.cu", 5), ("row_table.cuh", 13), ("row_table_host.cpp", 4)))
open(path, "w").write(text.replace(line, line + more))
EOF
  }
  gate() { python3 "$d/gate.py" "$1" "$2" < /dev/null 2>&1 | cut -c1-1200; r=${PIPESTATUS[0]}; echo "ratchet_gate, $3 ($1 $2): exit $r"; return $r; }
  lint() { $PY pytest tests/lint/test_repo_retired_terms.py -q $NOCACHE 2>&1 | tail -n 3; r=${PIPESTATUS[0]}; echo "retired-terms lint at $1: exit $r"; return $r; }
  rc=0
  git checkout -q --detach "$REMEDY_BASE" || return 1
  add "$old" && git add "$list" && $g commit -q -m "scratch: the three row_table sources' sites, recorded at the adapter's path" || return 1
  X=$(git rev-parse HEAD); lint "the entries at the old path" || rc=1
  git read-tree -u --reset "$REMEDY_MOVER" && add "$new" && git add "$list" \
    && $g commit -q -m "scratch: the mover's commit, carrying the three entries" || return 1
  M=$(git rev-parse HEAD)
  $g cherry-pick $REMEDY_PICKS > /dev/null || return 1
  H=$(git rev-parse HEAD); lint "the result" || rc=1
  echo "the list's diff across the scratch mover's commit:"; git diff "$X" "$M" -- "$list" | grep '^[+-] ' | cut -c1-140
  echo "tree of the result: $(git rev-parse "$H^{tree}"); tree of $REMEDY_SAME: $(git rev-parse "$REMEDY_SAME^{tree}")"
  echo "files that differ from $REMEDY_SAME: $(git diff --name-only "$H" "$REMEDY_SAME" | tr '\n' ' ')"
  [ "$(git show "$H:$list" | sort | sed 's/,$//')" = "$(git show "$REMEDY_SAME:$list" | sort | sed 's/,$//')" ] \
    && echo "the list holds the same entries as at $REMEDY_SAME" || { echo "the list's entries differ from $REMEDY_SAME's"; rc=1; }
  $PY python tools/verity/check/move_check.py "$X" "$M" < /dev/null; r=$?; echo "move_check $X $M: exit $r"; [ $r -eq 0 ] || rc=1
  gate "$REMEDY_BASE" "$X" "the entries at the old path, alone" || rc=1
  gate "$X" "$M" "the mover's commit carrying them, alone" || rc=1
  gate "$X" "$H" "the mover's commit, the mend and the test as one change" || rc=1
  gate "$REMEDY_BASE" "$H" "the entries, the move, the mend and the test as one change (told, not counted in this step's exit)"
  return $rc
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
