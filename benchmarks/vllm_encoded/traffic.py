"""HBM bytes a decode step moves, measured: Nsight Compute's DRAM counters over each profiled arm's windows (`serve.py
--profile-steps`, run under `ncu --profile-from-start off --graph-profiling graph`, exported as OUT/ncu-<tag>.csv with
`--page raw --print-units base`; on sm_120 the counters are `dram__bytes_op_read` and `dram__bytes_op_write`).

A window opens with torch's ``spin_kernel`` and holds a few steady decode steps of one point.  A step's forward is one CUDA
graph (one profile result reading at least the linear weights); the window's bytes over its graph count are the step's.
The linear weights (the encoded words, or the stock FP8 matrices) are read once a step, since they are far larger than L2,
so the uncertified bytes are the step's bytes less the weights': KV cache, activations, the LM head and embeddings, and
whatever the fused kernel's chunk sums cost beyond L2.  Writes OUT/traffic.json beside `compare.py`'s accounting.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import compare

READ, WRITE = "dram__bytes_op_read.sum", "dram__bytes_op_write.sum"
GRAPH_MIN = 2 << 30


def rows(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text().splitlines() if ln.startswith('"')]
    r = list(csv.reader(lines))
    head, body = r[0], r[2:]
    if READ not in head or WRITE not in head:
        raise SystemExit(f"{path.name}: no {READ} / {WRITE} columns (the metrics were not collected)")
    out = []
    for row in body:
        d = dict(zip(head, row))
        out.append({"marker": any("spin_kernel" in v for v in row),
                    "read": float(d[READ].replace(",", "")), "write": float(d[WRITE].replace(",", ""))})
    return out


def windows(rs: list[dict]) -> list[list[dict]]:
    out = []
    for r in rs:
        if r["marker"]:
            out.append([])
        elif out:
            out[-1].append(r)
    return out


def main() -> int:
    out = Path(sys.argv[1])
    result, fails = [], []
    for a in compare.load(out):
        if not a.get("profile_steps"):
            continue
        csv_path = out / f"ncu-{a['tag']}.csv"
        if not csv_path.exists():
            fails.append(f"{a['tag']}: no {csv_path.name}")
            continue
        ws = windows(rows(csv_path))
        if len(ws) != len(a["profiled"]):
            fails.append(f"{a['tag']}: {len(ws)} profile windows for {len(a['profiled'])} profiled points")
            continue
        inst = a["install"]
        weights = inst["store_bytes"] if a["arm"] in compare.ENCODED else inst["plain_bytes"]
        cod = inst.get("codec") or {}
        for pt, w in zip(a["profiled"], ws):
            tags = ((32 * compare.tag_blocks(cod, pt["batch"]) if "tag_blocks" in cod else inst["plain_bytes"] // 8)
                    if compare.codec_tag(a) else 0)
            graphs = sum(r["read"] >= GRAPH_MIN for r in w)
            if not graphs:
                fails.append(f"{a['tag']} {pt}: no decode graph in the window")
                continue
            rd, wr = sum(r["read"] for r in w) / graphs, sum(r["write"] for r in w) / graphs
            total = rd + wr
            timed = next((p for p in a["points"] if p["batch"] == pt["batch"] and p["context"] == pt["context"]), {})
            analytic = compare.working_set(a["model_shape"], pt["batch"], pt["context"] + timed.get("max_tokens", 0) / 2,
                                           weights, a["arm"] != "stock", tags)
            result.append({
                "tag": a["tag"], "arm": a["arm"], "k": a["k"], "compiled": bool(a.get("compiled")), **pt,
                "graphs": graphs, "results": len(w), "read_bytes": rd, "write_bytes": wr, "total_bytes": total,
                "weight_bytes": weights, "uncertified_bytes": total - weights,
                "uncertified_share": (total - weights) / total,
                "analytic_total_bytes": analytic["total"], "analytic_uncertified_share": analytic["uncertified_share"],
            })
    print(f"{'arm':10} {'B':>3} {'ctx':>5} {'read GB':>8} {'write GB':>8} {'weights':>8} {'uncert':>7} {'analytic':>8}")
    for r in result:
        print(f"{r['tag']:10} {r['batch']:3d} {r['context']:5d} {r['read_bytes'] / 1e9:8.3f} {r['write_bytes'] / 1e9:8.3f} "
              f"{r['weight_bytes'] / 1e9:8.3f} {r['uncertified_share']:7.1%} {r['analytic_uncertified_share']:8.1%}")
    (out / "traffic.json").write_text(json.dumps({"points": result, "fails": fails}, indent=1))
    for x in fails:
        print("FAIL", x)
    return 1 if fails or not result else 0


if __name__ == "__main__":
    sys.exit(main())
