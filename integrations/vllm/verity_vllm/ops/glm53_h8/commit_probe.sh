# Not a Commit of record. The row's config-run Commit needs the Build's GP-01 workload digests and manifest, and this tree's Build
# refuses GLM-5.3; this is the TP8 Commit's one instrumented run without them: per-rank roots, coverage of the required classes and
# the committer's staging (#920's GPU smoke), in the engine of record. Without --required-manifest there is no sampled replay and no
# weights pin, so its verdict fails closed by name. Each GPU's memory every 30 s. Exit 0 when tp-commit ran to its summary.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
export NATIVE_COLLECT_BUILD=/workspace/cp/nc_build
d=$SWEEP_DIR/commit-probe
rm -rf "$d"; mkdir -p "$d"
stamp commit-probe | tee "$O/commit-probe.log"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv -l 30 > "$O/gpu-mem.csv" & s=$!
t0=$(date +%s)
VV tp-commit --workload "$WL" --case "$ROLE" --tp 8 --committer native_collect_v2b \
  --hidden-so /workspace/cp/fa2/build/matReqG/verity_fa2_matReqG.so --openings 64 --max-num-seqs 1 "${ENGINE_ARGS[@]}" \
  --pairs 1 --warmup 1 --only-arm instrumented --label "$ROW" --out "$d" > "$O/tp-commit.log" 2>&1
rc=$?
kill $s 2>/dev/null
echo "[commit-probe] tp-commit rc=$rc wall=$(($(date +%s) - t0))s" | tee -a "$O/commit-probe.log"
du -sb "$d" "$d"/* 2>/dev/null | tee "$O/commit-dir-bytes.txt"
for f in summary.json runs.jsonl versions.json committer_make.json weights_pin.json timeline.jsonl; do
  [ -f "$d/$f" ] && cp "$d/$f" "$O/"
done
"$PY" - "$O" <<'EOF' 2>&1 | tee -a "$O/commit-probe.log"
import csv, json, sys
from collections import defaultdict
o = sys.argv[1]
used, total = defaultdict(int), {}
for row in csv.reader(open(f"{o}/gpu-mem.csv")):
    if len(row) < 4 or not row[1].strip().isdigit():
        continue
    i, u, t = int(row[1]), int(row[2].split()[0]), int(row[3].split()[0])
    used[i], total[i] = max(used[i], u), t
print("[commit-probe] gpu memory, peak used / total MiB:", " ".join(f"{i}:{used[i]}/{total[i]}" for i in sorted(total)))
try:
    sm = json.load(open(f"{o}/summary.json"))
except (OSError, ValueError) as e:
    print("[commit-probe] no summary:", e)
    sys.exit(0)
print("[commit-probe] summary keys:", sorted(sm))
for k in ("pass", "tp_run_root", "roots", "why", "fail", "verdict", "replay_deferred", "weights_pin", "required_classes"):
    if k in sm:
        print(f"[commit-probe] {k}: {json.dumps(sm[k], default=str)[:1500]}")
EOF
[ -f "$O/summary.json" ]
