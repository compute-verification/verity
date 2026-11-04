"""HT-DEFENCE's quality table: each arm's perplexity on WikiText-2 test and FineWeb-Edu val (1,048,576 tokens; a 70B
arm's two halves joined by their nll summed over their tokens), and its change against bf16.

    python benchmarks/hardness_defence/quality.py [--runs ~/.research/runs]

The arms are the runs in ARMS, fetched first with `research fetch RUN --all --on vy-nebius-2`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# model, arm, run, its output directories' prefix (8B: one directory; 70B: PREFIX-wt, -fw0 and -fw1)
ARMS = [
    ("8b", "bf16", "r20261104-081323-5f94", "8b-bf16"),
    ("8b", "design U (2%)", "r20261104-081323-5f94", "8b-u2"),
    ("8b", "layer 0 q, k at 6%", "r20261104-081323-5f94", "8b-clip"),
    ("8b", "layer 0 q, k uncredited", "r20261104-081323-5f94", "8b-broken-out"),
    ("8b", "q, k, v uncredited", "r20261104-081323-5f94", "8b-qkv-out"),
    ("8b", "layer 0 q at 8%, k at 6%", "r20261104-110454-7328", "8b-op"),
    ("8b", "design U, live weight noise f = 1 on q, k, v", "r20261104-111958-9ae0", "8b-u2-w1"),
    ("8b", "design U, live weight noise f = 4 on q, k, v", "r20261104-111958-9ae0", "8b-u2-w4"),
    ("70b", "bf16", "r20261104-081312-879a", "70b-bf16"),
    ("70b", "design U (2%)", "r20261104-082313-bf53", "70b-u2"),
    ("70b", "q, k, v uncredited", "r20261104-091006-89f8", "70b-qkv-out"),
    ("70b", "39 broken or concentrated uncredited", "r20261104-091026-b6ff", "70b-broken-out"),
    ("70b", "8 concentrated gate/up uncredited", "r20261104-095716-8e45", "70b-gu-out"),
    ("70b", "31 broken q, k, v uncredited", "r20261104-095739-e299", "70b-qkvb-out"),
    ("70b", "layer 0 q, k, v and layer 1 v uncredited", "r20261104-095749-e38f", "70b-unfix-out"),
    ("70b", "56 at their least clip that clears, up to 30%", "r20261104-110454-943c", "70b-clip"),
    ("70b", "50 clipped up to 30%, layer 0 q, k, v and layer 1 q uncredited", "r20261104-110454-c096", "70b-op"),
    ("70b", "24 clipped at 5%, layer 0 q, k, v and layer 1 v uncredited", "r20261104-113440-42a5", "70b-op5"),
    ("70b", "design U, live weight noise f = 1 on q, k, v", "r20261104-112058-99e8", "70b-u2-w1"),
    ("70b", "design U, live weight noise f = 4 on q, k, v", "r20261104-112058-99e8", "70b-u2-w4"),
]


def evals(d: Path) -> dict:
    r = json.loads((d / "result.json").read_text())
    return {k: v for k, v in r["eval_after"].items() if k != "modes"}


def arm(runs: Path, model: str, run: str, prefix: str) -> dict | None:
    out = runs / run / "out"
    if model == "8b":
        d = out / prefix
        if not (d / "result.json").exists():
            return None
        e = evals(d)
        return {"wikitext2": e["wikitext2"]["ppl"], "fineweb": e["fineweb_val"]["ppl"]}
    parts = {p: out / f"{prefix}-{p}" for p in ("wt", "fw0", "fw1")}
    if not all((d / "result.json").exists() for d in parts.values()):
        return None
    wt = evals(parts["wt"])["wikitext2"]["ppl"]
    fw = [evals(parts[p])["fineweb_val"] for p in ("fw0", "fw1")]
    nll = sum(f["nll"] * f["tokens"] for f in fw) / sum(f["tokens"] for f in fw)
    return {"wikitext2": wt, "fineweb": math.exp(nll)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=Path.home() / ".research" / "runs")
    args = ap.parse_args()
    base = {}
    print("| model | arm | WikiText-2 | vs bf16 | FineWeb-Edu | vs bf16 | run |")
    print("|---|---|---|---|---|---|---|")
    for model, name, run, prefix in ARMS:
        r = arm(args.runs, model, run, prefix)
        if r is None:
            print(f"| {model} | {name} | – | – | – | – | `{run}` (not fetched or not done) |")
            continue
        if name == "bf16":
            base[model] = r
        b = base.get(model)
        dw = f"{100 * (r['wikitext2'] / b['wikitext2'] - 1):+.2f}%" if b else "–"
        df = f"{100 * (r['fineweb'] / b['fineweb'] - 1):+.2f}%" if b else "–"
        print(f"| {model} | {name} | {r['wikitext2']:.4f} | {dw} | {r['fineweb']:.4f} | {df} | `{run}` |")


if __name__ == "__main__":
    main()
