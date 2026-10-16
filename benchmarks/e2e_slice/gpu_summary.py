"""gpu_summary.py BENCH COMMIT: o_proj's arms against plain cuBLASLt, and the operand commitment's cost a call."""
import json
import sys

j = json.loads(open(sys.argv[1]).readline())
p = j["arms"]["plain"]["ms"]
for a, v in sorted(j["arms"].items(), key=lambda kv: kv[1]["ms"]):
    print(f"{a:10s} {v['ms']:.4f} ms  {v['ms'] / p:.3f}x  {v['mhz']:.0f} MHz")
c = json.loads(open(sys.argv[2]).readline())["commit_cost"]
print(f"commit_cost {c['ms']:.4f} ms a call ({c['leaves']} leaves), {c['ms'] / p:.3f}x plain, {c['mhz']:.0f} MHz")
