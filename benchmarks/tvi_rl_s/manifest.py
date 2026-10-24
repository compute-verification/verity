"""Merged checkpoint manifest for crux 1: one row per saved checkpoint (node, path, S in bits, step, task, scores).

`python manifest.py NODEMAP.json RES_ROOT > manifest.json`, where RES_ROOT/<run id>/ holds a run's result-*.json and
<label>.jsonl (fetched from its out/) and NODEMAP maps run id to node.
"""
import glob
import json
import sys
from pathlib import Path


def main(nodemap_path, root):
    nodemap = json.load(open(nodemap_path))
    rows = []
    for d in sorted(Path(root).iterdir()):
        node = nodemap.get(d.name)
        if node is None:
            continue
        for p in sorted(glob.glob(str(d / "result-*.json"))):
            r = json.load(open(p))
            if r.get("method") == "base" or r.get("ckpt"):
                continue
            label = r["label"]
            rewards = {}
            jl = d / f"{label}.jsonl"
            if jl.exists():
                for line in open(jl):
                    e = json.loads(line)
                    if e.get("event") == "step":
                        rewards[e["step"]] = e["reward"]
            evals = {e["step"]: e["acc"] for e in r.get("evals", [])}
            for step, path in sorted(r.get("ckpts", {}).items(), key=lambda kv: int(kv[0])):
                if not path:
                    continue
                s = int(step)
                w = [rewards[i] for i in range(s - 24, s + 1) if i in rewards]
                rows.append({"node": node, "path": path, "run_id": d.name, "task": r["task"], "method": r["method"],
                             "seed": r["seed"], "S_bits": r["S_bits"], "n_params": r["n_params"], "step": s,
                             "steps_total": r["steps"], "model": r["model"], "lr": r["lr"],
                             "score_greedy": evals.get(s), "score_online_T1": sum(w) / len(w) if w else None})
    return rows


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], sys.argv[2]), indent=1))
