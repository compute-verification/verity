#!/usr/bin/env bash
# The falsification search on a node, from the checkout's root:
#
#   research run --on auto --cwd clone -- \
#     bash verity/core/protocols/network/nci/fuzz/run.sh [RANDOM] [CHAIN] [EXTRA] [SEED] [COPIES] [UNWALK] [BIG] [SHARED]
#
# builds `nci-desc-fuzz`, writes the Programs (`gen/`: the hand family, the NCI difftest vectors with EXTRA more seeds of
# each of their generators, RANDOM random descriptors, CHAIN deep chains, COPIES batches read as segment copies, UNWALK
# random descriptors past the walk's bound, BIG random descriptors three times the size and SHARED shared-members
# declarations after `shared()`'s six, each with its IU assignments (`gen/shared.py`), from seed SEED, with mutants),
# checks them, and writes to $OUT `summary.json` (`gen/summarize.py`), `failures.jsonl.gz` (each result where a statement
# fails, with its Program) and every result, gzipped, per family. With EXTRA negative, no hand family and no vectors: a
# run beside one that has them, on other seeds. With FUZZ_EXES="NAME=PATH ...", no build: each prebuilt `nci-desc-fuzz`
# (`build.sh`'s, against another commit's NciCheck; PATH a file, or one sent to the run with `research run --send`)
# checks the same Programs, into $OUT/NAME.
set -euo pipefail
FUZZ=verity/core/protocols/network/nci/fuzz
RANDOM_N=${1:-100000}
CHAIN_N=${2:-10000}
EXTRA=${3:-10000}
SEED=${4:-0}
COPIES_N=${5:-0}
UNWALK_N=${6:-0}
BIG_N=${7:-0}
SHARED_N=${8:-0}
JOBS=$(nproc)
(( JOBS > 16 )) && JOBS=16
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT

if [[ -n ${FUZZ_EXES:-} ]]; then
  EXES=$FUZZ_EXES
else
  (cd "$FUZZ" && lake build)
  EXES=".=$FUZZ/.lake/build/bin/nci-desc-fuzz"
fi

shards() { # family count mutate: JOBS generators over consecutive seed ranges, one file
  local fam=$1 n=$2 mut=$3 per=$(( ($2 + JOBS - 1) / JOBS )) i
  : > "$S/$fam.0.jsonl"
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

FAMILIES="random chain copies unwalk big"
(( EXTRA >= 0 || SHARED_N > 0 )) && uv sync -q --all-packages --extra torch-cpu
if (( EXTRA >= 0 )); then
  FAMILIES="hand vectors $FAMILIES"
  python3 "$FUZZ/gen/programs.py" hand --out "$S/hand.jsonl"
  uv run python "$FUZZ/gen/vectors.py" --out "$S/vectors.jsonl" --extra "$EXTRA" &
fi
if (( SHARED_N > 0 )); then
  FAMILIES="$FAMILIES shared"
  uv run python "$FUZZ/gen/shared.py" --seeds --seed "$SEED" --count "$SHARED_N" --mutate 0.3 --out "$S/shared.jsonl" &
fi
shards random "$RANDOM_N" 0.3
shards chain "$CHAIN_N" 0.2
shards copies "$COPIES_N" 0.2
shards unwalk "$UNWALK_N" 0.3
shards big "$BIG_N" 0.3
wait

for e in $EXES; do
  name=${e%%=*}
  exe=${e#*=}
  [[ -f $exe ]] || exe=${RESEARCH_RUN_DIR:?}/inputs/$exe
  O=$OUT/$name
  mkdir -p "$O" "$S/$name"
  cp "$exe" "$S/$name/nci-desc-fuzz" && chmod +x "$S/$name/nci-desc-fuzz"
  exe=$S/$name/nci-desc-fuzz
  sha256sum "$exe" > "$O/exe.sha256"
  : > "$O/failures.jsonl"
  for f in $FAMILIES; do
    echo "== $name $f: $(wc -l < "$S/$f.jsonl") programs"
    t=$SECONDS
    "$exe" "$S/$f.jsonl" "$S/$name/$f.out.jsonl"
    echo "   $((SECONDS - t)) s"
    grep '"program"' "$S/$name/$f.out.jsonl" >> "$O/failures.jsonl" || true
    gzip -c "$S/$name/$f.out.jsonl" > "$O/$f.out.jsonl.gz"
  done
  gzip -f "$O/failures.jsonl"
  python3 "$FUZZ/gen/summarize.py" "$S/$name"/*.out.jsonl --json "$O/summary.json"
done
