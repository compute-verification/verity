#!/usr/bin/env bash
# trails.c's searches (cwd = the source tree), one bound ladder each over all 256 start words (`trails.py ladder`): duo
# with the key secret, with phi.cu's key and with 16 random keys; each duo2 lane alone; duo2 with the key secret and
# known. Each found trail is measured on the function itself. One JSON line per ladder in $OUT/ladders.jsonl.
#
#   research run --on vy-nebius-cpu-1 --project verity --campaign pouw --source . --cwd clone --send benchmarks/receipt_phi/trails.sh \
#     -- python3 tools/verity/check/slot.py --by NAME -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/trails.sh"'
set -uo pipefail
O=${OUT:?}
D=benchmarks/receipt_phi
B=$(mktemp -d /tmp/receipt-trails.XXXXXX); trap 'rm -rf "$B"' EXIT
gcc -O3 -march=native -o "$B/trails" $D/trails.c || exit 4
"$B/trails" selftest > "$O/selftest.json" || exit 5
lad() {
  timeout "${JOB_MAX:-3600}" python3 $D/trails.py ladder --bin "$B/trails" "$@" >> "$O/ladders.jsonl" 2>> "$O/errors.txt" \
    || echo "ladder $* rc $?" >> "$O/errors.txt"
}
lad --kind duo --key secret --max-bound 12
lad --kind duo --key known --max-bound 12
for ks in $(seq 1 16); do lad --kind duo --key known --kseed "$ks" --max-bound 8; done
for k in duo2-lane1 duo2-lane2; do
  lad --kind $k --key known --max-bound 12
  lad --kind $k --key secret --max-bound 12
done
lad --kind duo2 --key secret --max-bound "${SECRET_MAX:-17}" --max-states "${MAX_STATES:-6000000}"
lad --kind duo2 --key known --max-bound "${KNOWN_MAX:-13}" --max-states "${MAX_STATES:-6000000}"
