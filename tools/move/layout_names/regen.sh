#!/usr/bin/env bash
# ci: `bash tools/move/layout_names/regen.sh TIP_SHA BRANCH` makes, from TIP_SHA in a clean checkout, the protocol groups'
# rename and input_datasets/'s move to library/datasets/ as two commits on BRANCH (created or reset; nothing is pushed).
# A lane, on its branch made before the rename: `bash tools/move/layout_names/carry.sh`.
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: regen.sh TIP_SHA BRANCH" >&2; exit 2; }
tip=$1 branch=$2
here=$(cd "$(dirname "$0")" && pwd)
if [ -z "${REGEN_COPY:-}" ]; then     # the checkout below may change this directory, so run from a copy of it
  copy=$(mktemp -d); cp -r "$here"/. "$copy"; REGEN_COPY=1 exec bash "$copy/regen.sh" "$@"
fi
cd "$(git rev-parse --show-toplevel)"
map=tools/move/protocol_names_map.toml
[ -z "$(git status --porcelain)" ] || { echo "regen.sh: the checkout is dirty; commit or stash first" >&2; exit 1; }
git checkout -q -B "$branch" "$tip"
for d in proof accounting policy run; do
  if [ -e "verity/protocols/$d" ] || [ -e "tests/verity/protocols/$d" ]; then
    echo "regen.sh: verity/protocols/$d already exists at $tip: stop and report" >&2; exit 1
  fi
done
if git grep -qwE 'verity\.protocols\.(proof|accounting|policy|run)\b' -- '*.py'; then
  echo "regen.sh: a module named verity.protocols.{proof,accounting,policy,run} is already imported at $tip" >&2; exit 1
fi
cp "$here/protocol_names_map.toml" "$map"
python3 tools/move/layout.py --map "$map" --no-lock --allow-dirty > /tmp/regen-layout.log
git add -A
python3 "$here/hand.py" "$map" > /dev/null
python3 tools/check/import_order.py > /dev/null     # the listing's tables sort by path, so they reorder
python3 -c 'import subprocess, sys; sys.path.insert(0, "tools/check"); import pins
[subprocess.run(p.shell(pins.ROOT), shell=True, check=True, cwd=pins.ROOT) for p in pins.PINS if p.key is None]'   # sha256sum pins: no Lean
hand=$(git diff --name-only | sort | tr '\n' ' ')
git add -A
GIT_AUTHOR_DATE="2026-11-20T00:00:00Z" GIT_COMMITTER_DATE="2026-11-20T00:00:00Z" git commit -q \
  -m "Move by tools/move/layout.py --map $map --no-lock: verity/protocols/ groups named proof, accounting, policy and run" \
  -m "The references layout.py doesn't read are rewritten after it by hand.py (tools/move/layout_names/), and the import order's listing by its generator." \
  -m "Hand-paths: ${hand% }" -m "Layout-move: $map --no-lock"
echo "$branch $(git rev-parse HEAD) tree $(git rev-parse 'HEAD^{tree}')"

lib=tools/move/library_datasets_map.toml
[ ! -e library ] || { echo "regen.sh: library/ already exists at $tip: stop and report" >&2; exit 1; }
cp "$here/library_datasets_map.toml" "$lib"
python3 tools/move/layout.py --map "$lib" --no-lock --allow-dirty > /tmp/regen-library.log
git add -A
python3 "$here/library_hand.py" > /dev/null
python3 tools/check/import_order.py > /dev/null
python3 -c 'import subprocess, sys; sys.path.insert(0, "tools/check"); import pins
[subprocess.run(p.shell(pins.ROOT), shell=True, check=True, cwd=pins.ROOT) for p in pins.PINS if p.key is None]'
hand=$(git diff --name-only | sort | tr '\n' ' ')
git add -A
GIT_AUTHOR_DATE="2026-11-20T00:00:00Z" GIT_COMMITTER_DATE="2026-11-20T00:00:00Z" git commit -q \
  -m "Move by tools/move/layout.py --map $lib --no-lock: input_datasets/ to library/datasets/ (the import name stays)" \
  -m "The references layout.py doesn't read are rewritten after it by library_hand.py (tools/move/layout_names/), and the import order's listing by its generator." \
  -m "Hand-paths: ${hand% }" -m "Layout-move: $lib --no-lock"
echo "$branch $(git rev-parse HEAD) tree $(git rev-parse 'HEAD^{tree}')"
