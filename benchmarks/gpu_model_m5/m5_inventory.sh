#!/usr/bin/env bash
# M5's store and engine inventory of a node (no GPU work, no lease): every block device a PoUS store or a bandwidth
# tier could live on, the host's installed DRAM, and the GPUs' fixed-function engines, for the model's StoreCap and
# Closed (the attack catalog's row 25).
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd clone \
#     --send m5_inventory.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/m5_inventory.sh"'
set -uo pipefail
OUT=${OUT:?run under research run}
lsblk -b -d -o NAME,SIZE,ROTA,TRAN,TYPE,MODEL >"$OUT/lsblk.txt" 2>&1
lsblk -b -o NAME,SIZE,TYPE,MOUNTPOINT >"$OUT/lsblk-tree.txt" 2>&1
df -B1 --output=source,size,used,avail,target -x tmpfs -x devtmpfs -x overlay >"$OUT/df.txt" 2>&1
grep -E '^(MemTotal|MemAvailable|HugePages_Total|Hugepagesize):' /proc/meminfo >"$OUT/meminfo.txt"
lscpu >"$OUT/lscpu.txt" 2>&1
nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv >"$OUT/gpus.csv" 2>&1
nvidia-smi -q -i 0 | grep -A3 -iE '^\s*(Encoder|Decoder|JPEG|OFA)' >"$OUT/engines-gpu0.txt" 2>&1 || true
nvidia-smi topo -m >"$OUT/topo.txt" 2>&1 || true
ls /sys/class/infiniband 2>/dev/null >"$OUT/ib.txt" || true
for d in /sys/class/net/*; do
  n=$(basename "$d"); [ "$n" = lo ] && continue
  printf '%s %s\n' "$n" "$(cat "$d/speed" 2>/dev/null || echo ?)"
done >"$OUT/net.txt"
python3 - "$OUT" <<'EOF'
import json, re, sys, pathlib
out = pathlib.Path(sys.argv[1])
disks = []
for line in (out / "lsblk.txt").read_text().splitlines()[1:]:
    f = line.split(None, 5)
    if len(f) >= 5 and f[4] == "disk":
        disks.append({"name": f[0], "bytes": int(f[1]), "rota": f[2], "tran": f[3], "model": f[5] if len(f) > 5 else ""})
mem = dict(re.findall(r"^(\w+):\s+(\d+)", (out / "meminfo.txt").read_text(), re.M))
rec = {"probe": "inventory", "disks": disks, "disk_bytes": sum(d["bytes"] for d in disks),
       "nvme_bytes": sum(d["bytes"] for d in disks if d["name"].startswith("nvme")),
       "mem_total_bytes": int(mem.get("MemTotal", 0)) * 1024}
(out / "summary.jsonl").open("a").write(json.dumps(rec) + "\n")
print(json.dumps(rec))
EOF
cat "$OUT/engines-gpu0.txt" "$OUT/net.txt" "$OUT/ib.txt"
