"""The split attacks' grid: the cheapest covering of every variant of the toy at every k, in parallel.

    python run.py --out DIR [--workers N] [--budget S] [--bits-budget S] [--k 12,16,24,32]

Writes `DIR/results.jsonl` (one line per run) and `DIR/summary.md` (blocks per variant and k, with lower bounds and the
gain against `mono`, the unsplit optimum). `local` runs add the locality rule: every counted value's pieces (its shares
or planes) held by one block.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import time
from pathlib import Path

import cover
import toy


def value_groups(g: toy.Graph) -> list[list[int]]:
    """The pieces of each counted value h_t[j,r], t >= 1: its shares' outputs, or their planes."""
    groups: dict[str, list[int]] = {}
    for i, nd in enumerate(g.nodes):
        if nd.tag == "value":
            key = re.match(r"(h\d+\[\d+,\d+\])", nd.name).group(1)
            groups.setdefault(key, []).append(i)
    return [groups[k] for k in sorted(groups)]


def task(args):
    model, variant, k, local, budget = args
    g = toy.training_graph(variant=variant, model=model)
    t = time.monotonic()
    try:
        r = cover.colgen(g, k=k, local=value_groups(g) if local else None, budget=budget, max_iter=100000)
    except ValueError as e:
        r = {"k": k, "blocks": None, "error": str(e)}
    r.update(model=model, variant=variant, local=local, seconds=round(time.monotonic() - t, 1), **g.stats())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--bits-budget", type=float, default=2400.0)
    ap.add_argument("--k", default="12,16,24,32")
    ap.add_argument("--models", default="norm,linear")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ks = [int(x) for x in args.k.split(",")]
    for model in args.models.split(","):
        ok = toy.check_equivalent(model=model)
        assert all(ok.values()), ok
    tasks = []
    for model in args.models.split(","):
        for variant in toy.VARIANTS:
            for k in ks:
                b = args.bits_budget if "bits" in variant else args.budget
                tasks.append((model, variant, k, False, b))
                if variant != "mono" and (model == "linear" or "bits" in variant):
                    tasks.append((model, variant, k, True, b))
    tasks.sort(key=lambda a: -a[4])
    results = []
    with mp.Pool(args.workers) as pool, open(out / "results.jsonl", "w") as f:
        for r in pool.imap_unordered(task, tasks):
            results.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["model"], r["variant"], "local" if r["local"] else "", r["k"], r.get("blocks"),
                  r.get("lp_bound"), r.get("optimal"), r["seconds"], flush=True)
    (out / "summary.md").write_text(summary(results))


def summary(results) -> str:
    by = {(r["model"], r["variant"], r["local"], r["k"]): r for r in results}
    ks = sorted({r["k"] for r in results})
    lines = []
    for model in sorted({r["model"] for r in results}):
        lines.append(f"\n## {model}\n")
        lines.append("| variant | " + " | ".join(f"k={k}" for k in ks) + " |")
        lines.append("|---|" + "---|" * len(ks))
        rows = sorted({(r["variant"], r["local"]) for r in results if r["model"] == model},
                      key=lambda x: (toy.VARIANTS.index(x[0]), x[1]))
        for variant, local in rows:
            cells = []
            for k in ks:
                r = by.get((model, variant, local, k))
                if r is None:
                    cells.append("")
                elif r.get("blocks") is None:
                    cells.append("infeasible")
                else:
                    base = by.get((model, "mono", False, k), {}).get("blocks")
                    lb = r.get("lp_bound")
                    cell = f"{r['blocks']}" if r.get("optimal") else f"{lb}-{r['blocks']}"
                    if base:
                        cell += f" ({base / r['blocks']:.2f}x)"
                    cells.append(cell)
            lines.append(f"| {variant}{' (local)' if local else ''} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
