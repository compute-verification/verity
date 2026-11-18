#!/usr/bin/env bash
# verity bench mempool [ARGS]: cost.py with ARGS (default --quick), its report in $OUT/cost.json.
set -euo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
cd benchmarks/mempool
exec python cost.py --json "$OUT/cost.json" "${@:---quick}"
