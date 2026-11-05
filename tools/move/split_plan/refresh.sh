#!/usr/bin/env bash
# `bash tools/move/split_plan/refresh.sh MAP OUT`, at the root of a tree that may be written (a research run's `--cwd clone`):
# builds Proofs.Flock from the Lean cache, extracts its constant graph (Extract.lean), rederives MAP's operations on it
# (`splitmap.py --refresh`), applies them (`leansplit.py`) and builds the result. OUT gets the graph (deps.tsv.gz), the
# rederived map (map.toml), what leansplit did (leansplit.log), the split's diff (split.patch) and each step's log;
# steps.log has each step's exit status, and the script's is the first that failed.
set -uo pipefail
map=$(realpath "$1") out=$(mkdir -p "$2" && realpath "$2")
here=$(cd "$(dirname "$0")" && pwd)
export PATH="$HOME/.elan/bin:$PATH"

step() {
    local name=$1 t=$SECONDS rc
    shift
    "$@" > "$out/$name.log" 2>&1
    rc=$?
    echo "$name: rc $rc in $((SECONDS - t)) s" | tee -a "$out/steps.log"
    return $rc
}

for p in Security Security/Properties Security/Proofs; do
    step "setup-${p//\//_}" bash tools/verity/lean/setup.sh "$p" || exit
done
step cache-get python3 tools/verity/lean/cache.py get || exit
step build-tree bash -c 'cd Security/Proofs && lake build Proofs.Flock' || exit
python3 "$here/graph.py" . Proofs.Flock mods > "$out/mods.txt" || exit
step extract bash -c "cd Security/Proofs && lake env lean --run '$here/Extract.lean' Proofs.Flock '$out/mods.txt' '$out/deps.tsv'" \
    || exit
gzip -f "$out/deps.tsv"
step refresh python3 "$here/splitmap.py" --repo . --deps "$out/deps.tsv.gz" --refresh "$map" --out "$out/map.toml" || exit
git status --porcelain > "$out/status-before-split.txt"
step leansplit python3 tools/move/leansplit.py . --map "$out/map.toml" || exit
git add -A -N && git diff > "$out/split.patch" && git diff --stat > "$out/split.stat"
step build-split bash -c 'cd Security/Proofs && lake build Proofs.Flock'
