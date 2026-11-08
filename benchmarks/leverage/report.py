"""Markdown tables from reach.py's and extract.py's outputs, and the leverage factor.

The cost table charges an incorrect proof block c + l bits, its content and its location. An attack that recovers B
bits of the weights per fault beyond honest use witnesses lambda >= 1 + B / (c + l); the tables give lambda - 1 at
each charge --charge names. The net figure subtracts what one honest query with the same observables recovers
(extract.py --part honest), which honest use gives for free; an attack on weights no honest attack reaches (the inner
layers' rows) has nothing to subtract.

  python benchmarks/leverage/report.py --reach DIR/summary.json --extract DIR/extract-*.json [--charge 54.7 72.7]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def table(head: list[str], rows: list[list]) -> str:
    fmt = lambda x: f"{x:.3g}" if isinstance(x, float) else str(x)  # noqa: E731
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(fmt(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def reach_tables(s: dict) -> str:
    bpt = s["bits_per_token"]
    best = {}
    for g in s["groups"]:
        key = (g["site"], g["layer"])
        if key not in best or g["direct_bits_bound"] > best[key]["direct_bits_bound"]:
            best[key] = g
    codes = ("rank_bits_bound", "xent_bits", "kl_bits")
    rows = [[k[0], k[1], g["mult"], g["n"], g["flip"], g["raw"], g["direct"], g["direct_after_first"],
             g["direct_bits_bound"], g["direct_bits_bound"] / bpt] + [g.get(c, "-") for c in codes]
            + [g["lp"], g["nan"]] for k, g in best.items()]
    head = ["site", "layer", "m (most reach)", "faults", "flip", "raw", "direct", "direct after 1st",
            "bound (bits)", "bound (tokens)", "rank code (bits)", "cross-entropy (bits)", "sum KL (bits)",
            "lp positions", "nan"]
    by_mult = {}
    for g in s["groups"]:
        by_mult.setdefault(g["mult"], []).append(g)
    mrows = []
    for m, gs in sorted(by_mult.items()):
        n = sum(g["n"] for g in gs)
        mean = lambda key: sum(g[key] * g["n"] for g in gs) / n if all(key in g for g in gs) else "-"  # noqa: E731
        mrows.append([m, n, mean("flip"), mean("direct"), mean("direct_bits_bound")] + [mean(c) for c in codes])
    temp = s.get("temperature", 0.0)
    return (f"Reach, {s['model']}, N = {s['new']} served positions, {s['faults']} faults, served logits "
            f"{s['precision']}, {'greedy' if not temp else f'sampled at temperature {temp:g}'}, top-{s['topk']} lp "
            f"tolerance {s['lp_tol']}:\n\n" + table(head, rows)
            + "\n\nBy multiple m of the site's RMS, over all sites:\n\n"
            + table(["m", "faults", "flip", "direct", "bound (bits)", "rank code (bits)", "cross-entropy (bits)",
                     "sum KL (bits)"], mrows))


def lam_tables(ex: list[dict], charges: list[float], ks: tuple[int, ...] = (20,)) -> str:
    parts = {e["part"]: e for e in ex}
    out = []
    honest = {}
    if "honest" in parts:
        h = parts["honest"]
        for r in h["rows"]:
            key = (r["precision"], r["k"])
            honest[key] = max(honest.get(key, 0.0), r["bits_per_query"])
        out.append(f"Honest SVD baseline, {h['model']}, {h['states']} states:\n\n" + table(
            ["precision", "n", "k", "queries", "mse/var (mean over directions)", "median", "bits total", "bits/query",
             "bits/observed bit"],
            [[r["precision"], r["n"], r["k"], r["queries"], r["mse_over_var"], r.get("median_mse_over_var", "-"),
              r["bits_total"], r["bits_per_query"], r["bits_per_observed_bit"]] for r in h["rows"]]))
    lh = [f"lambda-1 at c+l={x:g}" for x in charges]
    if "final" in parts:
        f = parts["final"]
        rows = []
        for r in f["rows"]:
            hb = honest.get((r["precision"], r["k"]), 0.0)
            B = r["bits_per_fault"]
            rows.append([r["amp"], r["precision"], r["k"], r["faults"], r["mse_over_var"], r.get("capped", "-"),
                         r["bits_per_entry"], B, hb] + [B / x for x in charges] + [(B - hb) / x for x in charges])
        out.append(f"Unembedding-input attack, {f['model']}, {f['columns']} columns, stored code "
                   f"{f.get('code_bits', float('nan')):.3g} bits an entry:\n\n" + table(
            ["A", "precision", "k", "faults", "mse/var", "exact", "bits/entry", "B bits/fault", "honest bits/query"] + lh
            + [f"net {x}" for x in lh], rows))
    for e in ex:
        if e["part"] != "rows":
            continue
        rows = [[r["site"], r["layer"], r["value"], r["precision"], r["k"], r["units"], r["read_tokens"],
                 r["faults_per_row"], r["contamination_mse"], r["mse"], r.get("capped", "-"), r["bits_per_row"],
                 r["mi_bits_per_row"],
                 r["bits_per_fault"]] + [r["bits_per_fault"] / x for x in charges] for r in e["rows"]
                if r["k"] in ks]
        out.append(f"Row readout, {e['model']}:\n\n" + table(
            ["site", "layer", "value", "precision", "k", "units", "logits read", "faults/row", "contamination mse",
             "mse", "exact", "bits/row", "Gaussian MI bits/row", "B bits/fault"] + lh, rows))
    return "\n\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach", type=Path, nargs="*", default=[])
    ap.add_argument("--extract", type=Path, nargs="*", default=[])
    ap.add_argument("--charge", type=float, nargs="+", default=[54.7, 72.7],
                    help="c + l, the table's charge per incorrect proof block, in bits")
    ap.add_argument("--ks", type=int, nargs="+", default=[20], help="the row readout's k to list")
    a = ap.parse_args()
    for p in a.reach:
        print(reach_tables(json.loads(p.read_text())) + "\n")
    if a.extract:
        print(lam_tables([json.loads(p.read_text()) for p in a.extract], a.charge, tuple(a.ks)))


if __name__ == "__main__":
    main()
