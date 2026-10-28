#!/usr/bin/env bash
# run.sh's join on a CPU node, for a GPU run made with JOIN=0, from the shipped tree's root, in a slot of network's cpu-1 pool:
#   research run --on vy-nebius-cpu-1 --project verity --campaign glm-e2e-demo --cwd source --store-read -- \
#     env CHECK_SLOTS=/workspace/research/locks/network/slots python3 tools/verity/check/slot.py --by LANE -- \
#     bash benchmarks/network_e2e/traces/join.sh art:RUN_RECORD [art:CORPUS[:TRACE,...]...]
# art:RUN_RECORD is the GPU run's record (`research data custody RUN --json`): each of its out/corpus/* is summarized into
# OUT/corpus/ (a warden trace's `warden_serve.py check` first), and compared request by request into OUT/compare/ with each
# art:CORPUS, a summarized vllm-trace-corpus/v1 (of the same plan and sampling), on the traces named. With the replay's
# benchmarks/network_e2e/encodings/replay.py in the tree, OUT/replay/ has its table over every trace and warden view
# (`summary.warden_views`), outputs released at their engine step and at the client's read (table-engine.json,
# table-client.json, table.md). OUT/corpus/<model>-<variant> goes in as vllm-trace-corpus/v1 and OUT/replay as
# advice-encoding/v1.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out
[ $# -ge 1 ] && [[ $1 == art:* ]] || { echo "usage: join.sh art:RUN_RECORD [art:CORPUS[:TRACE,...]...]"; exit 2; }
PY=${PY:-python3}
mkdir -p "$OUT/corpus"
git rev-parse HEAD > "$OUT/tree.txt" 2>/dev/null || true
printf '%s\n' "$@" > "$OUT/inputs.txt"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
rc=0
research data fetch "$1" --to "$tmp/run" || exit 1
shift
cp -r "$tmp"/run/out/corpus/* "$OUT/corpus/" || { echo "join: $tmp/run has no out/corpus"; exit 1; }
corpora=()
for dir in "$OUT"/corpus/*/; do
  dir=${dir%/}
  [ -f "$dir/meta.json" ] || continue
  rm -rf "$dir/.site"
  echo "join: summarizing $dir at $(date -u +%H:%M:%SZ)"
  "$PY" benchmarks/network_e2e/traces/corpus.py summarize "$dir" || rc=1
  corpora+=("$dir")
done
if [ $# -gt 0 ]; then
  specs=("${corpora[@]}")
  for a in "$@"; do
    id=${a#art:}; traces=; [[ $id == *:* ]] && traces=${id#*:}; id=${id%%:*}
    research data fetch "art:$id" --to "$tmp/$id" || { rc=1; continue; }
    root=$(find "$tmp/$id" -name summary.json -printf '%h\n' | sort | head -1)
    [ -n "$root" ] || { echo "join: art:$id has no summary.json"; rc=1; continue; }
    specs+=("$root${traces:+:$traces}")
  done
  "$PY" benchmarks/network_e2e/traces/corpus.py compare --out "$OUT/compare" "${specs[@]}" || rc=1
fi
REPLAY=benchmarks/network_e2e/encodings/replay.py
if [ -f "$REPLAY" ]; then
  mkdir -p "$OUT/replay"
  mapfile -t traces < <(find "$OUT/corpus" -name steps.jsonl -printf '%h\n' | sort)
  printf '%s\n' "${traces[@]#"$OUT"/}" > "$OUT/replay/traces.txt"
  for e in engine client; do
    echo "## egress: $e" | tee -a "$OUT/replay/table.md"
    "$PY" -P "$REPLAY" "${traces[@]}" --egress "$e" --out "$OUT/replay/table-$e.json" | tee -a "$OUT/replay/table.md" || rc=1
  done
fi
echo "join: done (rc $rc) at $(date -u +%H:%M:%SZ)"
exit "$rc"
