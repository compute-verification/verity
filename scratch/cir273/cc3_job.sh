# circuit-check on the three sidecars in the run's clone, run with `research run --on auto --project verity --source
# CHECKOUT --cwd clone --mem-gb 32 --timeout 120m -- bash -lc "$(cat scratch/cir273/cc3_job.sh)"`: --update-pins on each,
# then the clone's git status and git diff (empty when it wrote no pin), then each sidecar's report without it.
set -u
uv sync -q --all-packages --extra torch-cpu </dev/null
git rev-parse HEAD > "$OUT/head.txt"
for s in topp_stages sampled_accept boolean_topp_stages; do
  uv run circuit-check verity/definitions/registry/$s.circuit-check.py --update-pins --out "$OUT/pins-$s.json" > "$OUT/pins-$s.log" 2>&1
  echo "pins $s exit $?" >> "$OUT/exits.txt"
done
git status --porcelain > "$OUT/pins-status.txt"
git diff > "$OUT/pins.diff"
for f in $(git status --porcelain | awk '{print $2}'); do mkdir -p "$OUT/tree/$(dirname "$f")"; cp "$f" "$OUT/tree/$f"; done
for s in topp_stages sampled_accept boolean_topp_stages; do
  uv run circuit-check verity/definitions/registry/$s.circuit-check.py --out "$OUT/cc-$s.json" > "$OUT/cc-$s.log" 2>&1
  echo "cc $s exit $?" >> "$OUT/exits.txt"
  tail -6 "$OUT/cc-$s.log"
done
cat "$OUT/exits.txt"; echo "status lines: $(wc -l < "$OUT/pins-status.txt"); diff bytes: $(wc -c < "$OUT/pins.diff")"
