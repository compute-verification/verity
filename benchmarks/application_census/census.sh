#!/usr/bin/env bash
# One application through `verity run NAME` on a processor node, timestamped, its logs and record kept under $OUT.
# census.sh NAME [BINDINGS_LINES]: BINDINGS_LINES (TOML, `\n` for a newline) are added to the workload's table, and with
# any the preflight's ROW, HIDDEN_SO and VENV are stand-ins, so the run shows the refusal after the first one. Launch:
#   research run --on auto --cwd clone -- bash -lc 'set -- NAME; ...this script...'
set -o pipefail
name=$1 extra=${2-}
uv sync -q --all-packages --extra torch-cpu || exit 9
s=$(mktemp -d); trap 'rm -rf "$s"' EXIT
git rev-parse HEAD > "$OUT/commit.txt"
args=()
if [ -n "$extra" ]; then
  mkdir -p "$s/row"; echo '{}' > "$s/row/manifest.json"; echo '{}' > "$s/row/build_summary.json"; : > "$s/hidden.so"
  printf '[workers.served-window]\nROW = "%s"\nHIDDEN_SO = "%s"\nVENV = "%s"\n%b' "$s/row" "$s/hidden.so" "$PWD/.venv" "$extra" \
    > "$s/bindings.toml"
  cp "$s/bindings.toml" "$OUT/bindings.toml.txt"
  args=(--bindings "$s/bindings.toml")
fi
stamp() { while IFS= read -r l; do printf '%s %s\n' "$(date -u +%H:%M:%S)" "$l"; done; }
echo "$(date -u +%H:%M:%S) verity run $name begins" > "$OUT/run.ts"
uv run --no-sync verity run "$name" "${args[@]}" --state "$s/state" > >(stamp >> "$OUT/run.ts") 2> >(stamp >> "$OUT/run.ts.err")
echo $? > "$OUT/run.rc"
sleep 2
echo "$(date -u +%H:%M:%S) ended rc=$(cat "$OUT/run.rc")" >> "$OUT/run.ts"
for d in logs record public run.json out; do [ -e "$s/state/$d" ] && cp -r "$s/state/$d" "$OUT/"; done
find "$OUT" \( -name '*salt*' -o -name 'seed*' \) -exec rm -rf {} +
(gpu-lease 1 --wait --max-min 1 -- nvidia-smi -L) > "$OUT/gpu-lease.txt" 2>&1
echo "rc=$?" >> "$OUT/gpu-lease.txt"
