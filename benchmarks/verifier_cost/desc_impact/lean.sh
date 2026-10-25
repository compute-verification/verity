#!/usr/bin/env bash
# The Lean half of the NciCheck.Desc impact study (#4066's Seg.split), from the checkout's root, in vy-nebius-1's
# Lean-record pool:
#
#   research run --on vy-nebius-1 --project verity --source . --cwd clone -- \
#     env CHECK_SLOTS=/workspace/research/locks/leanrec/slots python3 tools/verity/check/slot.py --by training -- \
#     bash benchmarks/verifier_cost/desc_impact/lean.sh
#
# Three trees: pre (PRE, NciCheck.Desc before #3896's closed-form batch reads), main (MAIN) and fix (MAIN with
# fix-4066.patch: Seg.split and the 12 `shared` difftest vectors). In each it builds nci-check, nci-desc-difftest and
# nci-desc-fuzz (fuzz/, cursor/nci-desc-fuzz-4207's falsification search), into $OUT/bin/TREE. It writes #4066's 3,670
# vectors once (the vector script and the Python reference are the same at PRE and MAIN, so pre and main read the 12
# shared vectors beside their own 3,658), runs every tree's nci-desc-difftest on them ($OUT/TREE/difftest.txt), and
# shared.py's table of the 12 under every tree's nci-check ($OUT/shared.json).
set -uo pipefail
O=${OUT:?}
H=$(pwd)
I=benchmarks/verifier_cost/desc_impact
N=verity/core/protocols/network/nci/lean
F=verity/core/protocols/network/nci/fuzz
PRE=0249014fcd
MAIN=14acc70d97
TREES="pre main fix"
d=$(mktemp -d)
trap 'for t in $TREES; do git worktree remove --force "$d/$t" 2>/dev/null; done; rm -rf "$d"' EXIT
export LAKE_ARTIFACT_CACHE=false
uv sync -q --all-packages --extra torch-cpu || exit 1
git worktree add -q "$d/pre" $PRE && git worktree add -q "$d/main" $MAIN && git worktree add -q "$d/fix" $MAIN || exit 1
git -C "$d/fix" apply "$H/$I/fix-4066.patch" || exit 1
for t in $TREES; do
  mkdir -p "$O/$t" "$O/bin/$t"
  cp -r "$I/fuzz" "$d/$t/$F"
  { git -C "$d/$t" log --oneline -1; git -C "$d/$t" status --short; sha256sum "$d/$t/$N/NciCheck/Desc.lean"; } > "$O/$t/tree.txt"
done

for t in $TREES; do
  echo "== $t: build"
  s=$SECONDS
  (cd "$d/$t/$N" && lake build nci-check nci-desc-difftest) > "$O/$t/build.log" 2>&1 || { tail -30 "$O/$t/build.log"; exit 1; }
  cp "$d/$t/$N/.lake/build/bin/nci-check" "$d/$t/$N/.lake/build/bin/nci-desc-difftest" "$O/bin/$t/"
  if (cd "$d/$t/$F" && lake build nci-desc-fuzz) > "$O/$t/fuzz-build.log" 2>&1; then
    cp "$d/$t/$F/.lake/build/bin/nci-desc-fuzz" "$O/bin/$t/"
  else
    echo "$t: nci-desc-fuzz did not build"; tail -20 "$O/$t/fuzz-build.log"
  fi
  echo "   $((SECONDS - s)) s"
done

echo "== vectors"
V=$O/data/desc_difftest
(cd "$d/fix/$N" && "$H/.venv/bin/python" scripts/desc_difftest_vectors.py "$H" --out "$O/data") > "$O/data.log" 2>&1 \
  || { tail -30 "$O/data.log"; exit 1; }
"$H/.venv/bin/python" - "$d/fix/$N/lean-audit.json" "$O/data" <<'EOF' | tee "$O/data-pins.txt"
import hashlib, json, sys
from pathlib import Path
pins = json.loads(Path(sys.argv[1]).read_text())["runs"]["NciCheck.DescDifftestMain"]["data"]
for name, want in sorted(pins.items()):
    have = hashlib.sha256((Path(sys.argv[2]) / name).read_bytes()).hexdigest()
    print(name, have, "pinned" if have == want else f"DIFFERS from the pin {want}")
EOF

rc=0
for t in $TREES; do
  echo "== $t: nci-desc-difftest"
  "$O/bin/$t/nci-desc-difftest" "$V/instances.jsonl" "$V/expanded.jsonl" "$V/python.jsonl" 3670 > "$O/$t/difftest.txt" 2>&1
  echo "rc $?" >> "$O/$t/difftest.txt"
  tail -2 "$O/$t/difftest.txt"
done

echo "== shared"
"$H/.venv/bin/python" "$I/shared.py" "$V" --exe pre "$O/bin/pre/nci-check" --exe main "$O/bin/main/nci-check" \
  --exe fix "$O/bin/fix/nci-check" --out "$O/shared.json" | tee "$O/shared.txt" || rc=1
gzip "$V"/*.jsonl
exit $rc
