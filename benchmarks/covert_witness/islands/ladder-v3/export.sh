#!/usr/bin/env bash
# export.sh RUN...: the ladder's starts as one artifact, and its report. A RUN is either art:ID, a finished ladder run's
# run record (custody's run-record/v1), whose out/ is fetched from the evidence store (launch with --store-read), or
# OUT_DIR=WORK_DIR, a ladder run on this node, finished or still going: its out/ (/workspace/research/runs/ID/out) and its
# WORK (ladder.sh's third argument), where each start is registered as its branch anneals. ladder_report.py writes
# $OUT/report.json and $OUT/F.json, and starts_store.py export checks every θ against its digest and writes
# $OUT/starts/<name>/ with $OUT/starts_store.json. This run's own run record is then the starts' artifact (path out/starts).
#
#   research run --on NODE --project verity --source . --pool POOL --as training [--store-read] --timeout 1h \
#     --declared-output 'out/*' -- bash benchmarks/covert_witness/islands/ref-optim/export.sh art:... OUT=WORK ...
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/pyenv.sh"
research() { PYTHONPATH="$here/../../../../tools/research/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m research "$@"; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
runs=()
for spec in "$@"; do
  case "$spec" in
    art:*)
      d="$tmp/${spec#art:}"
      research data fetch "$spec" --to "$d" --path 'out/*' >/dev/null
      runs+=("$(find "$d" -type f -path '*/out/count.json' -printf '%h\n' | head -n 1)") ;;
    *=*)
      src=${spec%%=*} work=${spec#*=} d="$tmp/live-${#runs[@]}"
      mkdir -p "$d/starts"
      cp "$src"/*.json "$d/"
      for s in "$work"/starts/*/; do
        [ -f "$s/start.json" ] && ln -s "$s" "$d/starts/$(basename "$s")"
      done
      runs+=("$d") ;;
    *) echo "export.sh: $spec is neither art:ID nor OUT_DIR=WORK_DIR" >&2; exit 2 ;;
  esac
done
"$P" "$here/ladder_report.py" --runs "${runs[@]}" --out "$OUT"
mkdir -p "$tmp/all"
names=()
for r in "${runs[@]}"; do
  for s in "$r"/starts/*/; do
    [ -f "$s/start.json" ] || continue
    names+=("$(basename "$s")")
    ln -s "$(readlink -f "$s")" "$tmp/all/$(basename "$s")"
  done
done
cd "$here/../.."
CW_DEVICE=cpu "$P" starts_store.py export --from "$tmp/all" --out "$OUT" --names "${names[@]}"
