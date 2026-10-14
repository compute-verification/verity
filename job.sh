#!/usr/bin/env bash
set -uo pipefail
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
export TMPDIR="$d"
export PATH="$HOME/.elan/bin:$PATH"
rc=0
step() {
  local name=$1 must=$2
  shift 2
  echo "== $name: $*"
  "$@" > "$OUT/$name.log" 2>&1
  local got=$?
  echo "$name $got" | tee -a "$OUT/steps.txt"
  if [ "$must" = 1 ] && [ "$got" -ne 0 ]; then rc=1; fi
}
step security 1 python3 tools/verity/lean/lean_changed.py --records Security --update --out "$OUT/security"
step flock-records 0 python3 tools/verity/lean/lean_changed.py --records verity/core/service/flock --out "$OUT/flock-records"
step flock-build 1 bash -c 'cd verity/core/service/flock && lake build && lake build Flock.Test.InnerStreams Flock.Test.Full Flock.Test.Labels Flock.Test.MsgNames Flock.Test.ReadWhole Flock.Test.RecCoins Flock.Test.RecOpening Flock.Test.RecOpen Flock.Test.RecOpenLeaves Flock.Test.Sha512'
step flock-suite 1 uv run --locked --extra torch-cpu python tools/verity/check/suites.py verity/ml/flock --fresh
up= in=
for f in "$RESEARCH_RUN_DIR"/inputs/*; do
  case "$(sha256sum "$f" | cut -d' ' -f1)" in
    2e848f9881a225ed00be438bbebf64b2c9c83e4f1648cebaf51f535acecf9763) up=$f ;;
    af87c5f9cce005f0aefd1b132dc1e9c81b23a6f58c295f2e0f3590e1a6b55619) in=$f ;;
  esac
done
step lean-agreement 1 uv run --locked --extra torch-cpu python tools/verity/check/check.py --lean-agreement "$d/agreement" "$up" "$in"
mkdir -p "$OUT/agreement"
(cd "$d/agreement" && find . -path ./flock-agreement/upstream -prune -o -path ./flock-agreement/inputs -prune -o -path ./flock-agreement/work -prune -o -type f -size -20M -print | cpio -pdm --quiet "$OUT/agreement") || true
grep -rh "depends on axioms" "$OUT" --exclude=probes.txt > "$OUT/probes.txt" || true
exit "$rc"
