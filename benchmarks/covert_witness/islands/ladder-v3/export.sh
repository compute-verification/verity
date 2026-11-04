#!/usr/bin/env bash
# export.sh RUN...: the v3 ladder's starts as one artifact, and its report. A RUN is either art:ID, a finished ladder run's
# run record (custody's run-record/v1), whose out/ is fetched from the evidence store (launch with --store-read), or
# OUT_DIR=WORK_DIR, a ladder run on this node, finished or still going: its out/ (/workspace/research/runs/ID/out) and its
# WORK (ladder.sh's third argument), where each start is registered as its branch anneals. ladder_report.py writes
# $OUT/report.json, $OUT/F.json and $OUT/FIT.json (the scoring cap's ladder fit), and starts_store.py export checks every θ
# against its digest and writes starts/<name>/ and starts_store.json at the top of the run dir, so this run's run record
# is the starts' artifact at the manifest's path `starts`, where the pinned starts_store.py's pack and install --tar read.
#
#   research run --on NODE --project verity --source . --pool research --as training [--store-read] --timeout 1h \
#     --declared-output 'out/*' --declared-output 'starts/*' --declared-output starts_store.json \
#     -- bash benchmarks/covert_witness/islands/ladder-v3/export.sh art:... OUT=WORK ...
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/pyenv.sh"
research() { PYTHONPATH="$here/../../../../tools/research/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m research "$@"; }
top=${RESEARCH_RUN_DIR:?export.sh runs under research run}
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
CUDA_VISIBLE_DEVICES= CW_DEVICE=cpu "$P" starts_store.py export --from "$tmp/all" --out "$top" --names "${names[@]}"
cp "$top/starts_store.json" "$OUT/starts_store.json"
