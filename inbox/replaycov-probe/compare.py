"""Compare two probe replays' prefill top-k digests call by call, per device: the first call where the indexer logits, the
selected set or the raw order differ, and how many calls differ in each."""
import json
import sys
from pathlib import Path


def load(d: Path) -> dict[str, list[dict]]:
    return {f.name: [json.loads(x) for x in f.read_text().splitlines() if x.strip()] for f in sorted(d.glob("dev*.jsonl"))}


a, b = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
out = {}
for dev in sorted(set(a) & set(b)):
    xs, ys = a[dev], b[dev]
    n = min(len(xs), len(ys))
    shape = next((i for i in range(n) if (xs[i]["rows"], xs[i]["cols"], xs[i]["k"]) != (ys[i]["rows"], ys[i]["cols"], ys[i]["k"])), None)
    row = {"calls": [len(xs), len(ys)], "first_shape_mismatch": shape}
    m = shape if shape is not None else n
    for key in ("logits", "set", "order"):
        diff = [i for i in range(m) if xs[i][key] != ys[i][key]]
        row[key] = {"differing": len(diff), "first": diff[0] if diff else None,
                    "first_call": xs[diff[0]] if diff else None}
    row["long_calls"] = sum(1 for i in range(m) if xs[i]["cols"] > xs[i]["k"])
    out[dev] = row
print(json.dumps(out, indent=1))
