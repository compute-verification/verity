# Runbook steps 3-4 in one engine load (build-fold.md's observed smoke): TP8 eager, one B1 i256 o32 greedy request with the observer
# on rank 0..7, the live inventory before any request, then the capture-map check; each GPU's memory every 30 s. A failed observer
# reruns as the plain smoke (--no-observer), which still writes the inventory.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
d=$SWEEP_DIR/smoke-observe
rm -rf "$d"; mkdir -p "$d"
stamp smoke | tee "$O/smoke.log"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv -l 30 > "$O/gpu-mem.csv" & s=$!
cap() {  # cap LOG INVENTORY [ARG...]
  local log=$1 inv=$2; shift 2
  VV tp-capture --workload "$WL" --case "$ROLE" --tp 8 --max-num-seqs 1 "${ENGINE_ARGS[@]}" --snapshot-steps "" \
    --inventory-out "$inv" --out "$d" "$@" > "$log" 2>&1
}
t0=$(date +%s)
cap "$O/tp-capture.log" "$O/inv-live-tp8.json"; rc=$?
echo "[smoke] tp-capture rc=$rc wall=$(($(date +%s) - t0))s" | tee -a "$O/smoke.log"
inv=$O/inv-live-tp8.json
obs=ok
if [ "$rc" != 0 ] || grep -Eq '^\[tp2\] rank [0-7]: .* errors [1-9]' "$O/tp-capture.log"; then
  obs=failed
  cp "$d"/*.json "$O"/ 2>/dev/null
  rm -rf "$d"; mkdir -p "$d"
  t0=$(date +%s)
  cap "$O/tp-control.log" "$O/inv-live-tp8-control.json" --no-observer; rc=$?
  inv=$O/inv-live-tp8-control.json
  echo "[smoke] observer failed; plain smoke rc=$rc wall=$(($(date +%s) - t0))s" | tee -a "$O/smoke.log"
fi
kill $s 2>/dev/null
cp "$d"/log.rank0.jsonl.gz "$d"/ranks.json "$d"/topology.json "$d"/versions.json "$d"/tokens.json "$O"/ 2>/dev/null
cm=9
if [ -f "$inv" ]; then
  VV capture-map check --map "$CMAP" --config "$CK/config.json" --inventory "$inv" > "$O/capture-map-check.log" 2>&1; cm=$?
fi
"$PY" - "$O" "$inv" <<'EOF' 2>&1 | tee -a "$O/smoke.log"
import csv, json, re, sys
from collections import defaultdict
o, inv = sys.argv[1], sys.argv[2]
used, total = defaultdict(int), {}
for row in csv.reader(open(f"{o}/gpu-mem.csv")):
    if len(row) < 4 or not row[1].strip().isdigit():
        continue
    i, u, t = int(row[1]), int(row[2].split()[0]), int(row[3].split()[0])
    used[i], total[i] = max(used[i], u), t
print("[smoke] gpu memory, peak used / total MiB (free GiB):",
      " ".join(f"{i}:{used[i]}/{total[i]}({(total[i] - used[i]) / 1024:.1f})" for i in sorted(total)))
print("[smoke] min free GiB over GPUs:", round(min(((total[i] - used[i]) / 1024 for i in total), default=-1), 1))
for name in ("tp-capture.log", "tp-control.log"):
    try:
        txt = open(f"{o}/{name}", errors="replace").read()
    except OSError:
        continue
    for pat in (r"\[tp2\] engine built in [0-9.]+s", r"\[tp2\] (capture|control) run [0-9.]+s, \d+ engine steps", r"\[tp2\] live inventory of ranks .*",
                r"\[tp2\] rank \d: \d+ events.*", r"Using configuration from .*", r"[Uu]sing default .*MoE config.*"):
        for m in sorted(set(x.group(0) for x in re.finditer(pat, txt)))[:10]:
            print(f"[smoke] {name}: {m[:300]}")
try:
    doc = json.load(open(inv))
except (OSError, ValueError) as e:
    print("[smoke] no inventory:", e)
    sys.exit(0)
hits = set()
def walk(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            walk(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            walk(v, f"{path}[{i}]")
    elif isinstance(x, (str, bool)) and re.search(r"ll_bf16|cute|GateLinear", f"{path}={x}", re.I):
        hits.add(re.sub(r"\[\d+\]", "[]", f"{path}={x}")[:240])
walk(doc)
for h in sorted(hits)[:40]:
    print("[smoke] inventory:", h)
EOF
echo "[smoke] tp-capture rc=$rc capture-map rc=$cm observer=$obs" | tee -a "$O/smoke.log"
[ "$rc" = 0 ] && [ "$cm" = 0 ]
