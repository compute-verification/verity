#!/usr/bin/env bash
# `nci-desc-fuzz` against the checkout's NciCheck (`head`) and against NciCheck at each COMMIT (this checkout's fuzz/
# copied into a worktree there), into $OUT/bin/NAME, from the checkout's root in vy-nebius-1's Lean-record pool:
#
#   research run --on vy-nebius-1 --project verity --source . --cwd clone -- \
#     env CHECK_SLOTS=/workspace/research/locks/leanrec/slots python3 tools/verity/check/slot.py --by LANE -- \
#     bash verity/core/protocols/network/nci/fuzz/build.sh [NAME=COMMIT ...]
#
# `run.sh` then checks the same Programs with each on a CPU node (`--send` them, FUZZ_EXES="NAME=PATH ...").
set -euo pipefail
FUZZ=verity/core/protocols/network/nci/fuzz
N=verity/core/protocols/network/nci/lean
H=$(pwd)
d=$(mktemp -d)
trap 'for w in "$d"/*/; do git worktree remove --force "$w" 2>/dev/null || true; done; rm -rf "$d"' EXIT
export LAKE_ARTIFACT_CACHE=false

build() { # name root
  mkdir -p "$OUT/bin/$1"
  (cd "$2/$FUZZ" && lake build nci-desc-fuzz) > "$OUT/$1.build.log" 2>&1 || { tail -30 "$OUT/$1.build.log"; exit 1; }
  cp "$2/$FUZZ/.lake/build/bin/nci-desc-fuzz" "$OUT/bin/$1/"
  { git -C "$2" log --oneline -1; sha256sum "$2/$N/NciCheck/Desc.lean" "$OUT/bin/$1/nci-desc-fuzz"; } \
    | tee "$OUT/$1.tree.txt"
}

for spec in "$@"; do
  name=${spec%%=*}
  git worktree add -q "$d/$name" "${spec#*=}"
  rm -rf "$d/$name/$FUZZ"
  cp -r "$H/$FUZZ" "$d/$name/$FUZZ"
  rm -rf "$d/$name/$FUZZ/.lake"
done
build head "$H"
for spec in "$@"; do
  build "${spec%%=*}" "$d/${spec%%=*}"
done
