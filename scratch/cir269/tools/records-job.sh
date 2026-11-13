# CIR-269's records-first job: R_MEMGB, R_JOBS for the records pass; SHARD ("" for none), S_JOBS, S_MEMGB for the shard.
# Step 1 builds every catalog binding's record into a scratch cache and lists them; its reads are dumped by binding.
# Step 2 checks one shard from that cache.  Everything kept is under $OUT.
set -uo pipefail
uv sync -q --all-packages --extra torch-cpu </dev/null > "$OUT/sync.log" 2>&1 || { tail -20 "$OUT/sync.log"; exit 9; }
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
git rev-parse HEAD > "$OUT/head.txt"
echo "J head $(cat "$OUT/head.txt") $(nproc) cpus $(date -u +%H:%M:%SZ)"
t=$(date +%s)
uv run python -m circuit_check --all --records-only --cache "$d/cache" -j "$R_JOBS" --mem-gb "$R_MEMGB" \
  --records-manifest "$OUT/records.json" --out "$OUT/records-report.json" > "$OUT/records.log" 2>&1
rc1=$?
echo "J step1 rc=$rc1 seconds=$(( $(date +%s) - t )) $(date -u +%H:%M:%SZ)"
grep -E '^circuit-check --records-only|^warn' "$OUT/records.log" | cut -c1-2000
uv run python - "$d/cache" "$OUT/records.json" "$OUT/reads.json" <<'PY'
import json, sys
from pathlib import Path
from circuit_check import cache as CC
top, man = Path(sys.argv[1]), Path(sys.argv[2])
if not man.exists():
    print("J reads: no manifest"); sys.exit(0)
doc = json.loads(man.read_text())
out = {}
for row in doc["records"]:
    e = CC.read(top / row["record"])
    out[row["binding"]] = {"built": row["built"], "reads": e["reads"]}
Path(sys.argv[3]).write_text(json.dumps({"version": doc["version"], "unstored": doc["unstored"], "bindings": out}, sort_keys=True) + "\n")
print(f"J reads: {len(out)} records dumped, {len(doc['unstored'])} unstored, {len(doc['needs'])} needs")
PY
[ -z "$SHARD" ] && exit $rc1
[ "$rc1" = 0 ] || { echo "J step1 failed; no shard"; exit $rc1; }
t=$(date +%s)
uv run python -m circuit_check --all -q --jobs "$S_JOBS" --cache "$d/cache" --catalog-units --mem-gb "$S_MEMGB" \
  --shard "$SHARD" --shard-dir "$OUT/shard" --out "$OUT/circuit-check.json" > "$OUT/shard.log" 2>&1
rc2=$?
echo "J step2 rc=$rc2 seconds=$(( $(date +%s) - t )) $(date -u +%H:%M:%SZ)"
grep -E '^warn|^circuit-check: [0-9]+ target|^FAIL' "$OUT/shard.log" | head -60 | cut -c1-600
uv run python - "$OUT/circuit-check.json" <<'PY'
import json, sys
from pathlib import Path
from circuit_check.cli import heavy
p = Path(sys.argv[1])
if not p.exists():
    print("J shard: no report"); sys.exit(0)
rep = json.loads(p.read_text()); s = rep["summary"]; ts = rep["targets"]; h = heavy()
ran = [r for r in ts if r.get("peak_gb") is not None]
starts = [r["rss_start_gb"] for r in ran if r.get("rss_start_gb") is not None]
print(f"J shard: {len(ts)} targets, {s['failures']} failures, {s['unchecked']['items']} unchecked on {s['unchecked']['definitions']} Definitions; "
      f"{len(ran)} with a peak; parent rss at fork min {min(starts, default=0):.1f} max {max(starts, default=0):.1f} GiB")
own = sorted(((r["peak_gb"] - (r.get("rss_start_gb") or 0), r["peak_gb"], r["target"]) for r in ran), reverse=True)
for g, pk, name in own[:15]:
    print(f"J own {g:.2f} GiB (peak {pk:.2f}) {'listed' if name in h else 'UNLISTED'} {name}")
print("J over16 unlisted:", [(n, round(pk, 2)) for g, pk, n in own if pk > 16 and n not in h])
PY
exit $rc2
