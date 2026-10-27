#!/usr/bin/env bash
# The scheduler replay's table (replay.py's `main`) on a CPU node, from the shipped tree's root: over a trace-corpus
# artifact (store kind vllm-trace-corpus/v1, fetched with the run's read-only key) or over trace paths in the tree, with
# advice.py (the advice module) sent along for I1 v0's charge (without it, v0 is `v0_widths`):
#   research run --on vy-nebius-cpu-2 --mem-gb 8 --mem-wait 3600 --project verity --campaign glm-e2e-demo --cwd source \
#     --store-read [--send advice.py] -- bash benchmarks/network_e2e/encodings/run_corpus.sh art:...|TRACE...
# It writes OUT/table-<egress>-<ingress>.json (an artifact's four times: outputs released at their step's time or at
# their streamed chunk's, and requests' ingress at the client's send or at the engine's own arrival stamp, `replay.py --ingress`)
# and OUT/table.md; `research data put --kind advice-encoding/v1 --tree OUT --preserve` runs where `research fetch RUN
# --all` brought it.
set -euo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out
[ $# -ge 1 ] || { echo "usage: run_corpus.sh art:ID | TRACE..."; exit 2; }
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/tree.txt" 2>/dev/null || true
printf '%s\n' "$@" > "$OUT/inputs.txt"
if [[ $1 == art:* ]]; then
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  research data fetch "$1" --to "$tmp"
  mapfile -t traces < <(find "$tmp" -name steps.jsonl -printf '%h\n' | sort)
else
  traces=("$@")
fi
v0=()
for f in "$D"/inputs/*.py; do [ -f "$f" ] && v0=(--v0 "$f"); done
runs=("engine client")
[[ $1 == art:* ]] && runs=("engine client" "client client" "engine warden" "client warden")
for r in "${runs[@]}"; do
  read -r e i <<< "$r"
  echo "## egress: $e, ingress: $i" | tee -a "$OUT/table.md"
  python3 -P benchmarks/network_e2e/encodings/replay.py "${traces[@]}" "${v0[@]}" --egress "$e" --ingress "$i" \
    --out "$OUT/table-$e-$i.json" | tee -a "$OUT/table.md"
done
