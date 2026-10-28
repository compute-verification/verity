#!/usr/bin/env bash
# Re-read served_util.sh captures with tensor_util.py --steady (no GPU), and count its forwards against the step trace.
#   bash benchmarks/one_stage/reread_cut.sh RUN:T ...   (writes $OUT/<run>.tu.{json,md}, $OUT/<run>.cut.{json,txt})
set -u
OUT=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$OUT"
PY=${PY:-/workspace/cp/venv312/bin/python}; SCR=${SCR:-/workspace/research/vllm-util80/scratch}; P=${P:-4}
one() {
  IFS=: read -r r T <<< "$1"; S=$SCR/$r
  [ -f "$S/prof.sqlite" ] || { echo "$r MISSING $S/prof.sqlite"; return; }
  nice -n 19 "$PY" benchmarks/hash/exactlin/tensor_util.py "$S/prof.sqlite" --tokens "$T" --steady \
    --out "$OUT/$r.tu.json" > "$OUT/$r.tu.md" 2> "$OUT/$r.tu.err"
  local rc=$?
  nice -n 19 "$PY" benchmarks/one_stage/forward_count.py --sqlite "$S/prof.sqlite" --trace "$S/serve/advice_trace.jsonl" \
    --tu "$OUT/$r.tu.json" --T "$T" --out "$OUT/$r.cut.json" > "$OUT/$r.cut.txt" 2>&1
  echo "$r T $T tensor_util rc=$rc $(tail -3 "$OUT/$r.cut.txt")"
}
export -f one; export OUT PY SCR
for x in "$@"; do ls -l "$SCR/${x%%:*}/prof.sqlite" 2>&1; done
printf '%s\n' "$@" | xargs -P "$P" -I{} bash -c 'one "$1"' _ {}
