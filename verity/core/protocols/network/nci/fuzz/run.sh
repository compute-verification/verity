#!/usr/bin/env bash
# The falsification search on a node, from the checkout's root:
#
#   research run --on auto --cpus 16 --mem-gb 32 --cwd clone -- \
#     bash verity/core/protocols/network/nci/fuzz/run.sh [RANDOM] [CHAIN] [EXTRA] [SEED]
#
# builds `nci-desc-fuzz`, writes the Programs (`gen/`: the hand family, the NCI difftest vectors with EXTRA more seeds of
# each of their generators, RANDOM random descriptors and CHAIN deep chains from seed SEED, with mutants), checks them,
# and writes to $OUT `summary.json` (`gen/summarize.py`), `failures.jsonl` (each result where a statement fails, with
# its Program) and every result, gzipped, per family.
set -euo pipefail
FUZZ=verity/core/protocols/network/nci/fuzz
RANDOM_N=${1:-100000}
CHAIN_N=${2:-10000}
EXTRA=${3:-10000}
SEED=${4:-0}
JOBS=$(nproc)
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT

(cd "$FUZZ" && lake build)
EXE=$FUZZ/.lake/build/bin/nci-desc-fuzz
uv sync -q --all-packages --extra torch-cpu

shards() { # family count mutate: JOBS generators over consecutive seed ranges, one file
  local fam=$1 n=$2 mut=$3 per=$(( ($2 + JOBS - 1) / JOBS )) i
  for ((i = 0; i < JOBS; i++)); do
    local lo=$(( SEED + i * per )) cnt=$per
    (( i * per >= n )) && break
    (( i * per + per > n )) && cnt=$(( n - i * per ))
    python3 "$FUZZ/gen/programs.py" "$fam" --seed "$lo" --count "$cnt" --mutate "$mut" --out "$S/$fam.$i.jsonl" &
  done
  wait
  cat "$S/$fam".*.jsonl > "$S/$fam.jsonl"
  rm -f "$S/$fam".*.jsonl
}

python3 "$FUZZ/gen/programs.py" hand --out "$S/hand.jsonl"
uv run python "$FUZZ/gen/vectors.py" --out "$S/vectors.jsonl" --extra "$EXTRA" &
shards random "$RANDOM_N" 0.3
shards chain "$CHAIN_N" 0.2
wait

: > "$OUT/failures.jsonl"
for f in hand vectors random chain; do
  echo "== $f: $(wc -l < "$S/$f.jsonl") programs"
  t=$SECONDS
  "$EXE" "$S/$f.jsonl" "$S/$f.out.jsonl"
  echo "   $((SECONDS - t)) s"
  grep '"program"' "$S/$f.out.jsonl" >> "$OUT/failures.jsonl" || true
  gzip -c "$S/$f.out.jsonl" > "$OUT/$f.out.jsonl.gz"
done
python3 "$FUZZ/gen/summarize.py" "$S"/*.out.jsonl --json "$OUT/summary.json"
