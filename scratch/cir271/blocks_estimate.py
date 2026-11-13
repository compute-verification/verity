"""CIR-271 (REQBLOCK): proof blocks at a ladder shape, estimated from the card's blocks per call of each row family at (256, 32).

    python3 scratch/cir271/blocks_estimate.py NODE_JSON SMALLEST_JSON SHAPE

NODE_JSON is node1.py's output at (256, 32) (its ``node.templates_by_id``), SMALLEST_JSON smallest.py's. Per family: card
blocks (`benchmarks/sampling_plan/glm53_blocks.json`, each Definition's prefill + decode split evenly over its ``of``
families) over the node's calls, times the shape's calls. An estimate: a family's blocks per call vary with its bindings."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
node, small, shape = json.loads(Path(sys.argv[1]).read_text())["node"], json.loads(Path(sys.argv[2]).read_text()), sys.argv[3]
card = json.loads((ROOT / "benchmarks/sampling_plan/glm53_blocks.json").read_text())
calls = Counter()
for name, n in node["templates_by_id"].values():
    calls[name.split("_v")[0]] += n
blocks = Counter()
for d in card["definitions"].values():
    for of in d["of"]:
        blocks[of.split("_v")[0]] += (d["prefill"] + d["decode"]) / len(d["of"])
fam = small["shapes"][shape]["families"]
est = {f: n * blocks[f] / calls[f] for f, n in fam.items() if calls[f]}
print(json.dumps({"shape": shape, "layers": small["layers"], "calls": sum(fam.values()), "blocks_estimate": round(sum(est.values())),
                  "top": {f: round(v) for f, v in sorted(est.items(), key=lambda x: -x[1])[:8]}}))
