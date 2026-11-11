#!/usr/bin/env bash
# Scratch, never for landing: the benchmark's check on this tree (main with the record check whose ownRows reads
# dims), and the own-rows probe (OwnProbe.lean) on the Programs the benchmark builds and on the small served phase's.
#   bash benchmarks/inference_only/own_probe.sh        (from the repository's root, in a clone; $OUT set)
set -euo pipefail
root=$(pwd)
pkg=verity/protocols/inference_only/verifier/nci/lean
scratch=$(mktemp -d)
uv sync -q --all-packages --extra torch-cpu
bash tools/verity/lean/setup.sh "$pkg" > "$scratch/setup.log" 2>&1 || { tail -40 "$scratch/setup.log"; exit 1; }
export PATH="$HOME/.elan/bin:$PATH"
mkdir -p "$OUT/bench" "$OUT/probe"
echo "== 1. the benchmark's command, as its README gives it"
uv run --no-sync verity bench inference_only --out "$OUT/bench" > "$OUT/bench/report.txt"
grep -n "check " "$OUT/bench/report.txt" || true
echo "== 2. the same report, the count's Program and the class's whole Program kept"
uv run --no-sync python benchmarks/inference_only/own_probe.py "$OUT/probe" > "$OUT/probe/report.txt"
python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
a = json.loads((out / "bench/inference_only.json").read_text())
b = json.loads((out / "probe/inference_only.json").read_text())
strip = lambda j: json.dumps({k: v for k, v in j.items() if k not in ("run",)}, sort_keys=True)
def checks(j):
    return [(p.get("label"), {k: v for k, v in p["check"].items() if k != "cost"}) for p in j["policies"]]
print("the two reports' checks equal:", checks(a) == checks(b))
for label, c in checks(a):
    print("CHECK", label, json.dumps(c, sort_keys=True))
print("VALUE", a["value"], a["unit"][:60])
PY
echo "== 3. the small served phase's Program"
(cd adapters/vllm && uv run --no-sync python own_small.py "$OUT/probe")
echo "== 4. the probe: ownRows with the test of before against the test of now, and every batch node's axes against dims"
cd "$root/$pkg"
lake build NciCheck.Record > "$scratch/build.log" 2>&1 || { tail -60 "$scratch/build.log"; exit 1; }
tail -2 "$scratch/build.log"
for f in small-program count-instance class-program; do
  ls -la "$OUT/probe/$f.json"
  /usr/bin/time -v timeout 3000 lake env lean --run "$root/benchmarks/inference_only/OwnProbe.lean" instance "$OUT/probe/$f.json" \
    2> "$scratch/$f.time" || echo "PROBE $f: exit $?"
  grep -E "Elapsed|Maximum resident" "$scratch/$f.time" || tail -5 "$scratch/$f.time"
done
echo "== done"
