#!/usr/bin/env bash
# CIR-286: circuit-check's report on the Definitions the no-indexer forms are: DsaRMSNorm_v1 at N 2048 and 512 (eps 1e-5) and
# RoPEHeadGptJ_v2 at D 64, by id; the pins' diff to $OUT.  Runs in a private clone (--cwd clone).
set -uo pipefail
date -u +"sync %FT%TZ"; git log -1 --format="head %H"
uv sync -q --all-packages --extra torch-cpu || exit 2
T=('DsaRMSNorm_v1{N=2048,EPS=1e-05}' 'DsaRMSNorm_v1{N=512,EPS=1e-05}' 'RoPEHeadGptJ_v2{D=64}')
date -u +"report %FT%TZ"
uv run circuit-check --mem-gb 24 --out "$OUT/report.json" "${T[@]}" > "$OUT/report.log" 2>&1
rc=$?
echo "report rc=$rc"; tail -60 "$OUT/report.log"
git status --porcelain > "$OUT/status.txt"; echo "status:"; cat "$OUT/status.txt"
date -u +"end %FT%TZ"
exit $rc
