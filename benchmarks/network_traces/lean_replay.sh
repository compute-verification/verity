#!/usr/bin/env bash
# lean_replay.py over the recorded network-traces runs, every input fetched from the store into scratch, from the shipped
# tree's root (`--cwd clone` when the verifier is Lean's, which `lake build`s network-records in the tree):
#   research run --on MACHINE --project verity --campaign network-accounting --cwd clone --declared-output 'out/*' \
#     --send runs.json -- python3 tools/verity/check/slot.py --by network-accounting -- \
#     bash benchmarks/network_traces/lean_replay.sh [traces] [active] [-- lean_replay.py flags, e.g. --verifier M:C]
# runs.json: {"traces": {"verdicts": ART, "calibration": ART, "calibration_path": P, "bundles": [ART, ...],
#             "runs": [RUN, ...]}, "active": {RUN: RUN_RECORD_ART, ...}}. A trace RUN is a bundle's RUN/trace.json, its
# recorded rows those of `labels.run` RUN in the verdicts artifact's out/all@plaintext/verdicts.jsonl. A machine whose
# store has no remote takes them shipped instead (`--send`): traces.tar (rec/, cal/ and bundle/ as fetched here) and
# active.tar (runs/RUN/ as fetched here), each used in place of its fetches when present in the inputs. JOBS (default
# nproc) traces run at once; active configurations at most ACTIVE_JOBS (8) at once within ACTIVE_GB (400) of estimated
# peak memory, those over MAX_FRAMES (100000) Program frames left out but each run's smallest up to EACH_RUN_UP_TO (220000),
# and a run's controls injected only if its smallest reads at most CONTROLS_UP_TO (25000).
set -euo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:?run this under research run}/out}
IN=${IN:-$RESEARCH_RUN_DIR/inputs}
mkdir -p "$OUT"
kinds=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do kinds+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
[ ${#kinds[@]} -gt 0 ] || kinds=(traces active)
s=$(mktemp -d /workspace/research/scratch/lean-replay.XXXXXX 2>/dev/null || mktemp -d)
trap 'rm -rf "$s"' EXIT
export TMPDIR=$s
JOBS=${JOBS:-$(nproc)}
py() { uv run -q --locked python "$@"; }
fetch() { uv run -q --locked research data fetch "$@" > /dev/null; }
case " $* " in
  *" --verifier "*) ;;
  *) export PATH=$HOME/.elan/bin:$PATH
     (cd verity/core/protocols/network/lean && lake build network-records) > "$OUT/lake-build.log" 2>&1 ;;
esac
git rev-parse HEAD > "$OUT/head.txt" 2>/dev/null || true
q() { python3 -c "import json,sys; d=json.load(open('$IN/runs.json')); print(eval(sys.argv[1]))" "$1"; }
for kind in "${kinds[@]}"; do
  case $kind in
    traces)
      if [ -f "$IN/traces.tar" ]; then
        tar xf "$IN/traces.tar" -C "$s"
      else
        fetch "$(q 'd["traces"]["verdicts"]')" --to "$s/rec" --path 'out/all@plaintext/verdicts.jsonl'
        fetch "$(q 'd["traces"]["calibration"]')" --to "$s/cal" --path "$(q 'd["traces"]["calibration_path"]')"
        for a in $(q '" ".join(d["traces"]["bundles"])'); do fetch "$a" --to "$s/bundle/${a#art:}"; done
      fi
      mkdir -p "$s/tr"
      find "$s/bundle" -name '*.tar.gz' -exec tar xzf {} -C "$s/tr" \;
      rm -rf "$s/bundle"
      traces=()
      for r in $(q '" ".join(d["traces"]["runs"])'); do traces+=("$s/tr/$r/trace.json"); done
      py benchmarks/network_traces/lean_replay.py traces --calibration "$s/cal/$(q 'd["traces"]["calibration_path"]')" \
        --recorded "$s/rec/out/all@plaintext/verdicts.jsonl" --out "$OUT/traces" --jobs "$JOBS" "$@" "${traces[@]}" \
        2>&1 | tee "$OUT/traces.log" || true
      rm -rf "$s/tr" ;;
    active)
      if [ -f "$IN/active.tar" ]; then
        tar xf "$IN/active.tar" -C "$s"
      else
        for r in $(q '" ".join(d["active"])'); do
          fetch "$(q "d['active']['$r']")" --to "$s/runs/$r" --path 'out/*' --path 'inputs/*calibration.json'
        done
      fi
      py benchmarks/network_traces/lean_replay.py active --out "$OUT/active" --jobs "${ACTIVE_JOBS:-8}" \
        --budget-gb "${ACTIVE_GB:-400}" --max-frames "${MAX_FRAMES:-100000}" --each-run-up-to "${EACH_RUN_UP_TO:-220000}" \
        --controls-up-to "${CONTROLS_UP_TO:-25000}" \
        "$@" "$s"/runs/* 2>&1 | tee "$OUT/active.log" || true ;;
  esac
done
