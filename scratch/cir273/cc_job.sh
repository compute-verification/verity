# circuit-check on the three sidecars, run with `research run --on auto --project verity --cwd clone --wait --mem-gb 32
# --timeout 120m -- bash -lc "$(cat scratch/cir273/cc_job.sh)"` from the branch's checkout.
# 1. --update-pins on topp_stages and sampled_accept, then the clone's diff (what it wrote);
# 2. the three sidecars without --update-pins (the reports);
# 3. the accept sidecar at ACCEPT's head ($ACCEPT_HEAD, default #9693's 66f957e940), the whole tree checked out.
set -u
uv sync -q --all-packages --extra torch-cpu </dev/null
git rev-parse HEAD > "$OUT/head.txt"
for s in topp_stages sampled_accept; do
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
git checkout -q -f --detach "${ACCEPT_HEAD:-66f957e940d7428d0904626920d265a4f38e93c7}" && git rev-parse HEAD > "$OUT/accept-head.txt"
uv run circuit-check verity/definitions/registry/sampled_accept.circuit-check.py --out "$OUT/cc-accept-head.json" > "$OUT/cc-accept-head.log" 2>&1
echo "cc accept-head exit $?" >> "$OUT/exits.txt"
tail -6 "$OUT/cc-accept-head.log"
cat "$OUT/exits.txt" "$OUT/pins-status.txt"
