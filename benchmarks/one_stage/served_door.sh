#!/usr/bin/env bash
# `served_door.py` on a CPU node: the served circuit built once, then for each template's first unit the door's steps,
# each in its own process under /usr/bin/time (wall and peak RSS). TEMPLATES: one descriptor id a line.
# research run --on auto --cwd source --mem-gb 64 --timeout 2h -- bash benchmarks/one_stage/served_door.sh
set -uo pipefail
O=${OUT:-${RESEARCH_RUN_DIR:-/tmp/served-door}/out}; mkdir -p $O
S=$(mktemp -d); trap 'rm -rf $S' EXIT
TEMPLATES=${TEMPLATES:-'OutCollect_v1{KO=2,FIRST=true}
RMSNormTriton_v5{N=2048,EPS={"f64":"0x1.4f8b588e368f1p-17"}}'}
timed() { local name=$1; shift; /usr/bin/time -o $O/time-$name.txt -f "TIME $name wall_s=%e maxrss_kb=%M rc=%x" "$@"; cat $O/time-$name.txt; }
uv sync -q --all-packages --extra torch-cpu || exit 4
timed build uv run -q python benchmarks/one_stage/served_door.py build --tree . --out $S 2>&1 | tail -5
sha256sum $S/C.json | tee $O/C.sha256
i=0
while IFS= read -r T; do
  i=$((i + 1))
  for s in block restrict lower door one; do
    timed $s-$i uv run -q python benchmarks/one_stage/served_door.py step $s --dir $S --template "$T" 2>&1 | tail -4 | cut -c1-3000
  done
done <<< "$TEMPLATES"
