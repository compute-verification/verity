#!/usr/bin/env bash
# Partial completion against the masked k = 2 word, CPU only (mask_attack.py): every experiment at 512-2048 bits, the
# Coppersmith peel at 512-1024. Writes $OUT/mask_attack.jsonl ($OUT is the run's out/, or ./out outside a run).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${OUT:-out}
mkdir -p "$out"
WORDS=${WORDS:-200}
deps=(--with gmpy2 --with python-flint --with fpylll --with cysignals)
{
  uv run --no-project "${deps[@]}" python "$here/mask_attack.py" "$WORDS" quartic,advice,peel,intermediate,wrap,floor
  uv run --no-project "${deps[@]}" python "$here/mask_attack.py" "$WORDS" copper p512 p640 p768 p1024
} | tee "$out/mask_attack.jsonl"
