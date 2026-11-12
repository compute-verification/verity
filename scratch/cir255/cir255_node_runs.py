"""On a node: the runs whose span overlaps given UTC windows, with their slot, cores, mem, start, end and command head.

python3 - LO1 HI1 [LO2 HI2 ...] < cir255_node_runs.py   (ISO UTC, e.g. 2026-11-12T00:50:00)
"""
import json
import sys
from pathlib import Path

wins = list(zip(sys.argv[1::2], sys.argv[2::2]))
root = Path("/workspace/research/runs")
for d in sorted(root.glob("r20261112-0[0-3]*")) + sorted(root.glob("r20261111-2*")):
    def rd(n):
        try:
            return json.loads((d / n).read_text())
        except Exception:
            return {}
    st, sl, job = rd("status.json"), rd("slot.json"), rd("job.json")
    start = sl.get("took") or st.get("started_utc") or (job.get("run") or {}).get("started_utc") or d.name
    end = st.get("ended_utc") or (job.get("run") or {}).get("ended_utc") or ""
    start, end = str(start)[:19], str(end)[:19]
    hit = [w for w in wins if start <= w[1] and (not end or end >= w[0])]
    if not hit:
        continue
    cmd = (job.get("run") or {}).get("command") or st.get("command") or []
    cmd = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
    for tag in ("--shard", "suites.py", "circuit-check", "pytest"):
        pass
    print(d.name, "slot", sl.get("slot"), sl.get("cores"), "mem", sl.get("mem"), "took", start, "end", end, "|", cmd[:60], "...",
          cmd[cmd.find("--shard"):cmd.find("--shard") + 12] if "--shard" in cmd else cmd[-60:])
