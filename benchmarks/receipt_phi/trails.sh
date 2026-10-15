#!/usr/bin/env bash
# trails.py on a CPU pod (cwd = the source tree): one Dijkstra search per model, key and start word, in parallel, then
# the cheapest trail of each (model, key) measured on duo itself, merged into $OUT/trails.json.
#
#   research run --on vy-nebius-cpu-1 --project verity --campaign pouw --source . --cwd clone \
#     --send benchmarks/receipt_phi/trails.sh -- bash -c 'python3 tools/verity/check/slot.py -- bash "$RESEARCH_RUN_DIR/inputs/trails.sh"'
set -uo pipefail
O=${OUT:?}
D=benchmarks/receipt_phi
P=$(mktemp -d /tmp/receipt-trails.XXXXXX); trap 'rm -rf "$P"' EXIT
J=$(python3 -c 'import os; print(min(32, len(os.sched_getaffinity(0))))')
echo "jobs $J" >> "$O/errors.txt"
jobs() {
  for t in $(seq 0 255); do
    echo "secret 0xC0FFEE11 ${SECRET_BOUND:-9} $t"
    echo "known 0xC0FFEE11 ${KNOWN_BOUND:-8} $t"
    for ks in 0x1 0x2; do echo "known $ks ${RANDOM_BOUND:-6} $t"; done
  done
}
export D P O
jobs | xargs -P "$J" -L 1 bash -c 'timeout "${JOB_MAX:-1200}" python3 "$D/trails.py" search --key "$0" --kseed "$1" --k1 3 --k2 3 \
  --bound "$2" --starts "$3" > "$P/$0-$1-$3.json" 2>> "$O/errors.txt" || echo "job $0 $1 $3 rc $?" >> "$O/errors.txt"'
python3 - "$P" "$D" > "$O/trails.json" <<'EOF'
import json, pathlib, sys
sys.path.insert(0, sys.argv[2])
import trails as T

res = {}
for f in sorted(pathlib.Path(sys.argv[1]).glob("*.json")):
    txt = f.read_text().strip()
    key, kseed, start = f.stem.split("-")
    g = res.setdefault(f"{key}-{kseed}", {"key": key, "kseed": kseed, "starts": 0, "done": 0, "found": [], "expanded": 0})
    g["starts"] += 1
    if not txt:
        continue
    d = json.loads(txt)
    g["done"] += 1
    g["bound"] = min(g.get("bound", 99), d["bound"])
    g["expanded"] += d["stats"]["expanded"]
    if d["trail"]:
        g["found"].append(d["trail"])
for g in res.values():
    g["found"].sort(key=lambda tr: (tr["cost"], tr["steps"][0]["t"]))
    g["costs"] = sorted({tr["cost"] for tr in g["found"]})
    g["starts_with_trail"] = len(g["found"])
    g["exhaustive"] = g["done"] == g["starts"] == 256
    g["best"] = g["found"][:4]
    for tr in g["best"]:
        s = T.Schedule(int(g["kseed"], 0))
        diffs = T.word_diffs(tr)
        tr["word_diffs"] = [[t, hex(d)] for t, d in diffs]
        tr["measured"] = T.simulate(s, diffs, 1 << 20, 7, g["key"] == "secret", tr["end"])
    del g["found"]
print(json.dumps(res, indent=1))
EOF
