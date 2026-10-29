#!/usr/bin/env bash
# synth.py's cases in parallel, one JSON a case under $OUT (a `research run` job; the scratch venv is outside the run dir).
#   research run --on NODE --project verity --send benchmarks/pouw/sketch/synth.py \
#     --send benchmarks/pouw/sketch/run_synth.sh --timeout 110m -- bash run_synth.sh [PARALLEL] [SECONDS_A_QUERY]
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
# sketch, steps, width: the positive controls (linear, sum) first, then the candidates
cases="linear 2 8
linear 3 8
linear 4 8
sum 2 8
sum 3 8
sum 4 8
nh 2 8
nh 4 8
nh 4 6
nh2 2 8
nh2 3 6
xor 2 8
xor 3 8
xor 4 8
xor 4 6
chain 2 8
chain 4 8
chain 4 6
sumsq 2 8
sumsq 3 8
sumsq 3 6
sumsq 4 6"
run_case() {
  local s=$1 T=$2 w=$3
  "$py" "$here/synth.py" --sketch "$s" --steps "$T" --width "$w" --timeout "$per" --json "$out/$s-T$T-w$w.json" \
    > "$out/$s-T$T-w$w.log" 2>&1 || echo "$s T$T w$w rc $?" >> "$out/failures.txt"
}
while read -r s T w; do
  while [ "$(jobs -rp | wc -l)" -ge "$par" ]; do wait -n || true; done
  run_case "$s" "$T" "$w" &
done <<< "$cases"
wait
"$py" - "$out" <<'EOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
rows = {}
for f in sorted(out.glob("*.json")):
    for k, v in json.loads(f.read_text())["verdicts"].items():
        rows[f"{k}/w{f.stem.split('-w')[-1]}"] = v
(out / "verdicts.json").write_text(json.dumps(rows, indent=1))
for k, v in rows.items():
    print(f"- {k}: honest {v['honest_len']}; cheaper at {v['cheaper']}; UNSAT at {v['unsat_below']}; unknown at {v['unknown']}")
EOF
