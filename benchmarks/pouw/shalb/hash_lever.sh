#!/usr/bin/env bash
# HASH's FOLD/TENSOR lane, the lever, on CPU: the lead's baseline from N-CARD's calculator (hash_sha_lead.py), the
# lever's floor and attacks (hash_lever.py), and the core-Lean floors (TensorFloor.lean, HashFloor.lean). cwd: the
# source tree's root. The calculator's inputs come as in hash_tensor.sh.
# Output: $OUT/{lead.json, hash_lever.json, hash_lever.txt, verdict.json, lean.txt, card.sha256}.
set -euo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
HERE=benchmarks/pouw/shalb
SCRATCH=$(mktemp -d); trap 'rm -rf "$SCRATCH"' EXIT
CARD=${CARD_DIR:-}
if [[ -z $CARD && -f ${RESEARCH_RUN_DIR:-.}/inputs/pouw_space.py ]]; then CARD=$RESEARCH_RUN_DIR/inputs; fi
if [[ -z $CARD ]]; then
  REF=${CARD_REF:-origin/cursor/compute-research-np-card-7c2d}
  git fetch -q origin "${REF#origin/}" 2>/dev/null || true
  git archive "$REF" benchmarks/pouw/theory/pouw_space.py benchmarks/pouw/theory/pouw_space.toml | tar -x -C "$SCRATCH"
  CARD=$SCRATCH/benchmarks/pouw/theory
fi
cp "$CARD/pouw_space.py" "$CARD/pouw_space.toml" "$SCRATCH/"
(cd "$SCRATCH" && sha256sum pouw_space.py pouw_space.toml) > "$OUT/card.sha256"
PYTHONPATH="$SCRATCH:$PWD/benchmarks/pouw/exact_kit" \
  uv run --no-project --with numpy python "$HERE/hash_sha_lead.py" "$OUT/lead.json"
LEAN=$(command -v lean || echo "$HOME/.elan/bin/lean")
{ "$LEAN" --version; for f in TensorFloor HashFloor; do echo "== $f.lean"; "$LEAN" "$HERE/$f.lean"; done; } \
  2>&1 | tee "$OUT/lean.txt"
uv run --no-project --with numpy python "$HERE/hash_lever.py" "$OUT/lead.json" --verdict "$OUT/verdict.json" \
  --json "$OUT/hash_lever.json" | tee "$OUT/hash_lever.txt"
