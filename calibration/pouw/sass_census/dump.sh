#!/usr/bin/env bash
# The SASS opcode census of a node's binaries: every file with device code under the given roots, its ELFs extracted
# (`cuobjdump -xelf all`, into a work directory under TMPDIR) and disassembled with `cuobjdump -sass` in runs of
# consecutive ELFs, $JOBS runs at once, each run's opcodes counted by `sass_count.py` and the runs' counts joined in order
# into the binary's, and the opcodes the machine model doesn't price counted by `diff_menu.py record`. CPU only; no GPU.
# Writes under $OUT:
#   toolkit.txt    cuobjdump and nvcc --version
#   corpus.tsv     a line a binary kept: sha256, bytes, the architectures it holds, its path
#   counts/*.json  `sass_count.py count` for each binary (every architecture it holds)
#   cached.tsv     the binaries whose count was read from the cache, when any was
#   failed.txt     the binaries whose disassembly or count failed, when any did
#   census.json    `sass_count.py merge` of them all, a device family each; summary.txt its totals
#
#   research run --on MACHINE --cwd source -- bash calibration/pouw/sass_census/dump.sh [ROOT ...]
#
# ROOT: a directory searched for *.so, *.so.*, *.cubin and *.fatbin over 100 kB. Default: the CUDA toolkit of the nvcc
# it runs, /usr/lib/x86_64-linux-gnu (the driver's and NCCL's libraries) and this checkout. MAXBYTES skips binaries
# larger than it. A binary's count is kept in SASS_CENSUS_CACHE (default ~/.cache/verity/sass-census; empty: no cache)
# under the dump's version (`sass_count.py version`: this script, sass_count.py and cuobjdump's version) and the
# binary's SHA-256, and read back from there on a later run, which the log and cached.tsv name. NVCC (default: the newest
# /usr/local/cuda-13*), CUOBJDUMP and JOBS (default: nproc) are read too.
set -uo pipefail
case "${1:-}" in -h|--help) sed -n '2,/^set /p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;; esac
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "$0")" && pwd)
NVCC=${NVCC:-$(ls -d /usr/local/cuda-13*/bin/nvcc 2>/dev/null | sort -V | tail -1)}
[ -x "${NVCC:-}" ] || NVCC=$(command -v nvcc)
BIN=$(dirname "$NVCC")
CUOBJDUMP=${CUOBJDUMP:-$BIN/cuobjdump}
JOBS=${JOBS:-$(nproc)}
CACHE=${SASS_CENSUS_CACHE-${HOME:+$HOME/.cache/verity/sass-census}}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT/counts" "$WORK/lelf" "$WORK/sha256" "$WORK/parts" "$WORK/elf"
{ "$CUOBJDUMP" --version; "$NVCC" --version; } >"$OUT/toolkit.txt" 2>&1
VERSION=$(python3 "$HERE/sass_count.py" version --tool "$("$CUOBJDUMP" --version 2>&1)" "$HERE/dump.sh" "$HERE/sass_count.py")

roots=("$@")
[ ${#roots[@]} -gt 0 ] || roots=("$(dirname "$BIN")" /usr/lib/x86_64-linux-gnu "$HERE/../../../")
for r in "${roots[@]}"; do
  [ -e "$r" ] || continue
  find -L "$r" -xdev -type f \( -name '*.so' -o -name '*.so.*' -o -name '*.cubin' -o -name '*.fatbin' \) \
    -size +100k 2>/dev/null
done | sort -u >"$WORK/candidates"

# One copy of each binary (the same wheel sits in several venvs), the first in path order, and only binaries with device
# code; the candidates are read $JOBS at once.
export CUOBJDUMP WORK MAXBYTES
scan() {
  local i f lelf archs sha
  IFS=$'\t' read -r i f <<<"$1"
  [ -z "${MAXBYTES:-}" ] || [ "$(stat -L -c %s "$f")" -le "$MAXBYTES" ] || return 0
  lelf=$("$CUOBJDUMP" -lelf "$f" 2>/dev/null)
  archs=$(grep -o 'sm_[0-9]*[a-z]\?' <<<"$lelf" | sort -u | tr '\n' ' ')
  [ -n "$archs" ] || archs=$("$CUOBJDUMP" -sass "$f" 2>/dev/null | grep -m1 -o 'sm_[0-9]*[a-z]\?')
  [ -n "$archs" ] || return 0
  sha=$(sha256sum "$f" | cut -d' ' -f1)
  printf '%s\n' "$lelf" >"$WORK/lelf/${sha:0:16}"
  echo "$sha" >"$WORK/sha256/${sha:0:16}"
  printf '%s\t%s\t%s\t%s\t%s\n' "$i" "${sha:0:16}" "$(stat -L -c %s "$f")" "$archs" "$f" >"$WORK/scan/$i"
}
export -f scan
mkdir -p "$WORK/scan"
awk '{printf "%d\t%s\0", NR, $0}' "$WORK/candidates" | xargs -0 -r -P "$JOBS" -I{} bash -c 'scan "$@"' _ {}
find "$WORK/scan" -type f -exec cat {} + | sort -t$'\t' -k1,1n | cut -f2- | sort -t$'\t' -k1,1 -u >"$OUT/corpus.tsv"
wc -l <"$OUT/corpus.tsv" >"$OUT/corpus.count"
echo "dump: $(cat "$OUT/corpus.count") binaries with device code, found in ${SECONDS}s"

# A binary whose count the cache holds is read from it; the rest have their ELFs extracted and are cut into units.
: >"$WORK/fresh"
while IFS=$'\t' read -r sha bytes archs f; do
  src="$(basename "$f")@$sha"
  entry=
  if [ -n "$CACHE" ]; then
    entry=$CACHE/$(python3 "$HERE/sass_count.py" key "$VERSION" "$(cat "$WORK/sha256/$sha")" "$src")
    if [ -f "$entry" ]; then
      cp "$entry" "$OUT/counts/$sha.json"
      printf '%s\t%s\n' "$sha" "$f" >>"$OUT/cached.tsv"
      echo "dump: $src read from the cache: $entry"
      continue
    fi
  fi
  printf '%s\t%s\t%s\n' "$sha" "$f" "$entry" >>"$WORK/fresh"
  mkdir -p "$WORK/parts/$sha" "$WORK/elf/$sha"
  (cd "$WORK/elf/$sha" && "$CUOBJDUMP" -xelf all "$(readlink -f "$f")" >/dev/null 2>&1) || true
done <"$OUT/corpus.tsv"
python3 "$HERE/sass_count.py" plan --jobs "$JOBS" --work "$WORK" >"$WORK/units" || exit 1
echo "dump: $(wc -l <"$WORK/units") units of $(wc -l <"$WORK/fresh") binaries to disassemble, $JOBS at once," \
  "extracted by ${SECONDS}s"

export CUOBJDUMP HERE WORK
unit() {
  set -o pipefail
  IFS=$'\t' read -r _ sha i f <<<"$1"
  src="$(basename "$f")@$sha"
  if [ "$i" = - ]; then
    "$CUOBJDUMP" -sass "$f" 2>/dev/null | python3 "$HERE/sass_count.py" count --source "$src" >"$WORK/parts/$sha/-.json"
  else
    while IFS= read -r c; do "$CUOBJDUMP" -sass "$c" 2>/dev/null || exit 1; done <"$WORK/parts/$sha/$i.list" \
      | python3 "$HERE/sass_count.py" count --source "$src" >"$WORK/parts/$sha/$i.json" \
      && xargs -d '\n' rm -f <"$WORK/parts/$sha/$i.list"
  fi || printf '%s\n' "$sha" >>"$WORK/failed"
}
export -f unit
# the largest units first, so the long ones start early
sort -t$'\t' -k1,1nr "$WORK/units" | tr '\n' '\0' | xargs -0 -r -P "$JOBS" -I{} bash -c 'unit "$@"' _ {}
echo "dump: disassembled by ${SECONDS}s"

touch "$WORK/failed"
while IFS=$'\t' read -r sha f entry; do
  if grep -qx "$sha" "$WORK/failed"; then
    printf '%s\t%s\n' "$sha" "$f" >>"$OUT/failed.txt"
    continue
  fi
  if [ -f "$WORK/parts/$sha/-.json" ]; then
    mv "$WORK/parts/$sha/-.json" "$OUT/counts/$sha.json"
  else
    python3 "$HERE/sass_count.py" join --source "$(basename "$f")@$sha" "$WORK/parts/$sha"/*.json >"$OUT/counts/$sha.json" || { printf '%s\t%s\n' "$sha" "$f" >>"$OUT/failed.txt"; continue; }
  fi
  if [ -n "$entry" ]; then
    mkdir -p "$(dirname "$entry")" && cp "$OUT/counts/$sha.json" "$entry.$$" && mv "$entry.$$" "$entry"
  fi
done <"$WORK/fresh"

python3 "$HERE/sass_count.py" merge "$OUT"/counts/*.json >"$OUT/census.json"
python3 "$HERE/sass_count.py" summary "$OUT/census.json" | tee "$OUT/summary.txt"
ROOT=$HERE/../../..
PYTHONPATH="$ROOT/benchmarks/harness:$ROOT/tools/config${PYTHONPATH:+:$PYTHONPATH}" \
  python3 "$HERE/diff_menu.py" record "$OUT"
echo "dump: done in ${SECONDS}s"
