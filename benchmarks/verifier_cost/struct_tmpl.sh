#!/usr/bin/env bash
# The H1 tile unit's structural template: per SHAPE ("K" for the 4×8 tile, "K R C" otherwise),
# `python -m verity_flock.struct_lower` lowers the unit with each specialized Definition once and its calls kept as calls,
# writes it as flock-ir-tmpl/v2 and prints its STRUCT statistics, and flock-tmpl (flock_tmpl/Main.lean, built in a scratch
# Lake package against the checkout's verifier) times the templated read (`Tmpl.parse`, its order check included). With
# --check K, the unit at K is unrolled and evaluated against the flat lowering first; after --unroll, each later shape's
# template is also unrolled to its flat net (`flock-tmpl unroll`, unroll-SHAPE.jsonl); after --layouts, each later shape's
# template gets tmpl_layouts.py's today and option (iv) layouts, lenient, strict, hybrid and hybrid/4 (layouts-SHAPE.jsonl),
# and the Lean layout's (`flock-tmpl layout`, lean-layout-SHAPE.jsonl), whose column maps must agree with them; after
# --fold, each later shape's `Tmpl.fold` on the layout --mode names (default iv; fold-SHAPE.jsonl), after --split
# `Tmpl.foldSplit` there, checked against `Tmpl.fold`, and after --fold-ref, also the reference `Tmpl.foldRef`
# (O(unrolled)), which it must equal. Run from a clone:
#   research run --as proofs --project verity --on vy-nebius-cpu-1 --source . --pool proofs --cwd clone -- \
#     bash benchmarks/verifier_cost/struct_tmpl.sh --check 32 256 4096 "256 32 32"
# $OUT gets check-K.json, and per shape struct-SHAPE.json, read-SHAPE.jsonl and each step's /usr/bin/time -v; a template
# under 1 GB is kept as unit-SHAPE.tmpl.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
L=$PWD/verity/core/service/flock
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu
mkdir -p "$S/tmpl"
cp benchmarks/verifier_cost/flock_tmpl/Main.lean "$L/lean-toolchain" "$S/tmpl/"
cat > "$S/tmpl/lakefile.toml" <<EOF
name = "flock_tmpl"
defaultTargets = ["flock-tmpl"]

[[require]]
name = "flock_verifier"
path = "$L"

[[lean_exe]]
name = "flock-tmpl"
root = "Main"
EOF
(cd "$S/tmpl" && lake build flock-tmpl) > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
X=$S/tmpl/.lake/build/bin/flock-tmpl
unroll=0
layouts=0
fold=
mode=iv
split=()
while [ $# -gt 0 ]; do
  if [ "$1" = --mode ]; then
    mode=$2
    shift 2
    continue
  fi
  if [ "$1" = --split ]; then
    split=(--split --plain)
    shift
    continue
  fi
  if [ "$1" = --fold ] || [ "$1" = --fold-ref ]; then
    fold=$1
    shift
    continue
  fi
  if [ "$1" = --unroll ]; then
    unroll=1
    shift
    continue
  fi
  if [ "$1" = --layouts ]; then
    layouts=1
    shift
    continue
  fi
  if [ "$1" = --check ]; then
    k=$2
    shift 2
    /usr/bin/time -v uv run python -m verity_flock.struct_lower "$k" --check > "$O/check-$k.out" 2> "$O/check-time-$k.txt" \
      || { tail -40 "$O/check-time-$k.txt" >&2; exit 1; }
    grep '^STRUCT ' "$O/check-$k.out" | sed 's/^STRUCT //' | tee "$O/check-$k.json"
    grep -q '"mismatched_bits": 0,' "$O/check-$k.json" || { echo "K=$k: the structural unit's outputs differ" >&2; exit 1; }
    continue
  fi
  tag=${1// /x}
  shift
  t=$S/unit-$tag.tmpl
  /usr/bin/time -v uv run python -m verity_flock.struct_lower ${tag//x/ } --out "$t" > "$O/struct-$tag.out" \
    2> "$O/struct-time-$tag.txt" || { tail -40 "$O/struct-time-$tag.txt" >&2; exit 1; }
  grep '^STRUCT ' "$O/struct-$tag.out" | sed 's/^STRUCT //' | tee "$O/struct-$tag.json"
  /usr/bin/time -v "$X" read "$t" > "$O/read-$tag.out" 2> "$O/read-time-$tag.txt" \
    || { cat "$O/read-$tag.out" "$O/read-time-$tag.txt" >&2; exit 1; }
  grep '^PHASE ' "$O/read-$tag.out" | sed 's/^PHASE //' | tee "$O/read-$tag.jsonl"
  if [ $unroll = 1 ]; then
    /usr/bin/time -v "$X" unroll "$t" > "$O/unroll-$tag.out" 2> "$O/unroll-time-$tag.txt" \
      || { cat "$O/unroll-$tag.out" "$O/unroll-time-$tag.txt" >&2; exit 1; }
    grep '^PHASE ' "$O/unroll-$tag.out" | sed 's/^PHASE //' | tee "$O/unroll-$tag.jsonl"
    grep -E 'Maximum resident|Elapsed' "$O/unroll-time-$tag.txt"
  fi
  if [ $layouts = 1 ]; then
    /usr/bin/time -v uv run python benchmarks/verifier_cost/tmpl_layouts.py "$t" --modes today,iv,iv/strict,iv/hybrid,iv/hybrid/4 --digest \
      > "$O/layouts-$tag.out" 2> "$O/layouts-time-$tag.txt" || { tail -40 "$O/layouts-time-$tag.txt" >&2; exit 1; }
    grep '^LAYOUT ' "$O/layouts-$tag.out" | sed 's/^LAYOUT //' | tee "$O/layouts-$tag.jsonl" | cut -c1-400
    grep -E 'Maximum resident|Elapsed' "$O/layouts-time-$tag.txt"
    /usr/bin/time -v "$X" layout "$t" iv iv/strict iv/hybrid iv/hybrid/4 > "$O/lean-layout-$tag.out" 2> "$O/lean-layout-time-$tag.txt" \
      || { cat "$O/lean-layout-$tag.out" "$O/lean-layout-time-$tag.txt" >&2; exit 1; }
    grep '^PHASE ' "$O/lean-layout-$tag.out" | sed 's/^PHASE //' | tee "$O/lean-layout-$tag.jsonl"
    python3 - "$O/layouts-$tag.jsonl" "$O/lean-layout-$tag.jsonl" <<'EOF' || exit 1
import json, sys
name = lambda r: "iv/strict" if r["strict"] else (f"iv/hybrid/{r['coarse']}" if r["coarse"] else "iv/hybrid") if r["hybrid"] else "iv"
py = {name(r): r["digest"] for r in map(json.loads, open(sys.argv[1])) if r.get("fallback") == "interleave"}
ln = {r["phase"]: r.get("digest") for r in map(json.loads, open(sys.argv[2]))}
bad = [m for m in ("iv", "iv/strict", "iv/hybrid", "iv/hybrid/4") if py.get(m) is None or py.get(m) != ln.get(m + "/map")]
sys.exit(f"Tmpl.layout's map is not tmpl_layouts.py's: {bad}" if bad else 0)
EOF
  fi
  if [ -n "$fold" ]; then
    ref=()
    [ "$fold" = --fold-ref ] && ref=(--ref)
    /usr/bin/time -v "$X" fold "$t" --mode "$mode" "${split[@]}" "${ref[@]}" > "$O/fold-$tag.out" 2> "$O/fold-time-$tag.txt" \
      || { cat "$O/fold-$tag.out" "$O/fold-time-$tag.txt" >&2; exit 1; }
    grep '^PHASE ' "$O/fold-$tag.out" | sed 's/^PHASE //' | tee "$O/fold-$tag.jsonl"
    grep -E 'Maximum resident|Elapsed' "$O/fold-time-$tag.txt"
  fi
  grep -E 'Maximum resident|Elapsed' "$O"/{struct,read}-time-$tag.txt
  [ "$(stat -c %s "$t")" -lt 1000000000 ] && mv "$t" "$O/" || rm -f "$t"
done
