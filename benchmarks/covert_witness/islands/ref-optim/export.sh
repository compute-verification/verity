#!/usr/bin/env bash
# export.sh ART...: the ladder's starts as one artifact, and its report. Each ART is a ladder run's run record (custody's
# run-record/v1); its out/ is fetched from the evidence store (launch with --store-read), ladder_report.py writes
# $OUT/report.json and $OUT/F.json, and starts_store.py export checks every θ against its digest and writes
# $OUT/starts/<name>/ with $OUT/starts_store.json. This run's own run record is then the starts' artifact (path out/starts).
#
#   research run --on NODE --project verity --source . --pool POOL --as training --store-read --timeout 1h \
#     --declared-output 'out/*' -- bash benchmarks/covert_witness/islands/ref-optim/export.sh art:... art:... art:...
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/pyenv.sh"
research() { PYTHONPATH="$here/../../../../tools/research/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m research "$@"; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
runs=()
for art in "$@"; do
  d="$tmp/${art#art:}"
  research data fetch "$art" --to "$d" --path 'out/*' >/dev/null
  runs+=("$(find "$d" -type f -path '*/out/ladder.json' -printf '%h\n' | head -n 1)")
done
"$P" "$here/ladder_report.py" --runs "${runs[@]}" --out "$OUT"
mkdir -p "$tmp/all"
names=()
for r in "${runs[@]}"; do
  for s in "$r"/starts/*/; do
    names+=("$(basename "$s")")
    ln -s "$s" "$tmp/all/$(basename "$s")"
  done
done
cd "$here/../.."
CW_DEVICE=cpu "$P" starts_store.py export --from "$tmp/all" --out "$OUT" --names "${names[@]}"
