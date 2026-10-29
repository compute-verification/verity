#!/usr/bin/env bash
# synth.py's cases in parallel, one JSON a case under $OUT (a `research run` job; the scratch venv is outside the run dir).
#   research run --on NODE --project verity --send benchmarks/pouw/sketch/synth.py \
#     --send benchmarks/pouw/sketch/run_synth.sh --timeout 110m -- bash inputs/run_synth.sh [PARALLEL] [SECONDS_A_QUERY]
set -euo pipefail
par=${1:-24}
per=${2:-3000}
out=${OUT:-$PWD/out}
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
if command -v uv >/dev/null; then
  uv venv -q "$d/venv" && VIRTUAL_ENV="$d/venv" uv pip install -q z3-solver
else
  python3 -m venv "$d/venv" && "$d/venv/bin/pip" install -q z3-solver
fi
py="$d/venv/bin/python"
"$py" -c 'import z3; print("z3", z3.get_version_string())' | tee "$out/z3_version.txt"
# sketch, steps, width, library: the positive controls (linear, sum) first, then the candidates. `ring` (shifts left
# only) at width 4, whose UNSAT holds at 32 bits; `full` (right shifts and rotates too) at width 8, whose UNSAT holds there
cases="linear 2 4 ring
linear 3 4 ring
linear 4 4 ring
sum 2 4 ring
sum 3 4 ring
sum 4 4 ring
nh 2 4 ring
nh 4 4 ring
nh2 2 4 ring
nh2 3 4 ring
xor 2 4 ring
xor 3 4 ring
xor 4 4 ring
chain 2 4 ring
chain 4 4 ring
sumsq 2 4 ring
sumsq 3 4 ring
sumsq 4 4 ring
nh 4 8 full
nh2 2 8 full
xor 4 8 full
chain 4 8 full
sumsq 3 8 full"
run_case() {
  local s=$1 T=$2 w=$3 lib=$4 flag=""
  [ "$lib" = ring ] && flag=--ring
  local tag="$s-T$T-w$w-$lib"
  "$py" "$here/synth.py" --sketch "$s" --steps "$T" --width "$w" $flag --timeout "$per" --json "$out/$tag.json" \
    > "$out/$tag.log" 2>&1 || echo "$tag rc $?" >> "$out/failures.txt"
}
while read -r s T w lib; do
  while [ "$(jobs -rp | wc -l)" -ge "$par" ]; do wait -n || true; done
  run_case "$s" "$T" "$w" "$lib" &
done <<< "$cases"
wait
"$py" - "$out" <<'EOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
rows = {}
for f in sorted(out.glob("*-T*-w*.json")):
    for k, v in json.loads(f.read_text())["verdicts"].items():
        rows[f"{k}/w{f.stem.split('-w')[-1]}"] = v
(out / "verdicts.json").write_text(json.dumps(rows, indent=1))
for k, v in rows.items():
    print(f"- {k}: honest {v['honest_len']}; cheaper at {v['cheaper']}; UNSAT at {v['unsat_below']}; unknown at {v['unknown']}")
EOF
