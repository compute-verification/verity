#!/usr/bin/env bash
# `Run.run`'s cost against the population: whole_run.py writes, per SHAPE and N, an accepted `run` input (MODE run) or the
# program and query alone (MODE desc), and whole-run-time (whole_run/Main.lean, built in a scratch Lake package against
# the checkout's verifier) times the whole run and each of its steps, or the partition's per-unit gate lookups on the
# program alone. Each input runs under `timeout STEP_S` (default 3600). With PERF_N set, the run at N is also sampled
# by perf where the node has it. Run from a clone (the build writes the verifier's .lake):
#   research run --as proofs --project verity --on vy-nebius-cpu-12 --source . --pool proofs --cwd clone -- \
#     bash benchmarks/verifier_cost/whole_run.sh run toy 146 1000 10000 50000
#   ... -- bash benchmarks/verifier_cost/whole_run.sh desc l5 146 1000 10000 373792
# $OUT gets whole_run.jsonl (one row per input and step: shape, n, ms, peak RSS) and inputs.jsonl.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
[ $# -gt 2 ] || { echo "usage: whole_run.sh run|desc SHAPE N... [-- FLAGS]" >&2; exit 2; }
mode=$1 shape=$2
shift 2
ns=() flags=()
while [ $# -gt 0 ]; do
  [ "$1" = "--" ] && { shift; flags=("$@"); break; }
  ns+=("$1"); shift
done
L=$PWD/verity/protocols/spot/verifier/lean
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu
mkdir -p "$S/time"
cp benchmarks/verifier_cost/whole_run/Main.lean "$L/lean-toolchain" "$S/time/"
cat > "$S/time/lakefile.toml" <<EOF
name = "whole_run_time"
defaultTargets = ["whole-run-time"]

[[require]]
name = "core_spot"
path = "$L"

[[lean_exe]]
name = "whole-run-time"
root = "Main"
EOF
(cd "$S/time" && lake build whole-run-time) > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
desc=(); [ "$mode" = desc ] && desc=(--desc)
exe=$S/time/.lake/build/bin/whole-run-time
for n in "${ns[@]}"; do
  uv run python benchmarks/verifier_cost/whole_run.py "$S/in" "$shape" "$n" "${desc[@]}" | tee -a "$O/inputs.jsonl"
  f=$S/in/$mode-$shape-$n.json
  tag="{\"shape\":\"$shape\",\"n\":$n,"
  rc=0
  timeout "${STEP_S:-3600}" "$exe" "$mode" "$f" "${flags[@]}" | grep --line-buffered '^PHASE ' \
    | sed -u "s/^PHASE {/$tag/" | tee -a "$O/whole_run.jsonl" || rc=$?
  [ "$rc" = 0 ] || echo "$tag\"step\":\"timeout-or-error\",\"rc\":$rc}" | tee -a "$O/whole_run.jsonl"
  if [ "${PERF_N:-}" = "$n" ] && command -v perf > /dev/null; then
    timeout "${STEP_S:-3600}" perf record -F 499 -g -o "$S/perf.data" "$exe" "$mode" "$f" --no-whole "${flags[@]}" \
      > /dev/null 2>&1 || true
    perf report -i "$S/perf.data" --stdio --no-children --percent-limit 1 2> /dev/null | head -200 > "$O/perf-$shape-$n.txt" || true
  fi
  rm -f "$f"
done
