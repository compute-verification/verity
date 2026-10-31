"""HT-USEFUL's tables, rendered from fetched run outputs.

    python table.py quality DIR...   # every result.json under DIR (qat.py's outputs)
    python table.py price DIR...     # every summary.json under DIR (price.py's outputs)

quality: per arm, the measured Strassen level-1 overflow (median over matrices), the training tokens, perplexity on
WikiText-2 and FineWeb-Edu val before and after training, and zero-shot accuracy (acc_norm where lm-eval reports it,
else acc). price: per pricing, for each correction price and chip at the fast reading, the least `cheapest` over the
matrices (W1 per MAC against the honest kernel's 1), the MAC-weighted `cheapest` and the matrices under 1.05.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TASK_METRIC = {"arc_easy": "acc", "arc_challenge": "acc_norm", "hellaswag": "acc_norm", "piqa": "acc_norm",
               "winogrande": "acc"}


def zs_scores(ev: dict) -> dict[str, float]:
    out = {}
    for t, r in (ev.get("zeroshot") or {}).items():
        key = f"{TASK_METRIC.get(t, 'acc')},none"
        if key in r:
            out[t] = r[key]
    return out


def quality(dirs: list[Path]) -> str:
    rows = []
    for d in dirs:
        for p in sorted(d.rglob("result.json")):
            r = json.loads(p.read_text())
            cfg = r["config"]
            ov = sorted(v["strassen1"] for v in (r.get("calibration_after") or r.get("calibration_before") or {}).values())
            before, after = r["eval_before"], r.get("eval_after", r["eval_before"])
            zb, za = zs_scores(before), zs_scores(after)
            tok = (r.get("train") or {}).get("tokens", 0)
            rows.append([p.parent.name, "yes" if cfg.get("rotate") else "no", cfg["target"],
                         f"{ov[len(ov) // 2]:.2%}" if ov else "-", f"{tok / 1e6:.0f}M",
                         f"{before['wikitext2']['ppl']:.3f}", f"{after['wikitext2']['ppl']:.3f}",
                         f"{before['fineweb_val']['ppl']:.3f}", f"{after['fineweb_val']['ppl']:.3f}",
                         *(f"{za[t]:.4f}" if t in za else (f"({zb[t]:.4f})" if t in zb else "-") for t in TASK_METRIC),
                         f"{sum(za.values()) / len(za):.4f}" if len(za) == len(TASK_METRIC) else "-"])
    head = ["arm", "rotate", "target", "strassen1 median", "tokens", "WT2 before", "WT2 after", "FW before",
            "FW after", *TASK_METRIC, "zero-shot mean"]
    return md(head, rows)


def price(dirs: list[Path]) -> str:
    rows, keys = [], None
    for d in dirs:
        for p in sorted(d.rglob("summary.json")):
            s = json.loads(p.read_text())
            by = {k: v for k, v in s["by_price"].items() if k.endswith("/fast")}
            keys = keys or list(by)
            vs_leaf = {}
            for line in (p.parent / "matrices.jsonl").read_text().splitlines():
                for row in json.loads(line)["rows"]:
                    k = f"{row['alu']}/{row['chip']}/{row['reading']}"
                    vs_leaf[k] = min(vs_leaf.get(k, float("inf")), row["cheapest"] / row["plain"])
            rows.append([p.parent.name, f"{s['strassen1']['median']:.2%}", "yes" if s["exact"] else "NO",
                         *(f"{by[k]['min_cheapest']:.3f} / {vs_leaf[k]:.3f} / {by[k]['cheapest_macw']:.3f} / "
                           f"{by[k]['below_1.05']}" for k in keys)])
    head = ["codes", "strassen1 median", "exact", *(k.removesuffix("/fast") for k in keys or [])]
    return ("least cheapest route over the matrices / least cheapest over the matrix's plain leaf (1 + 32/K) / "
            "MAC-weighted cheapest / matrices whose cheapest is under 1.05\n\n" + md(head, rows))


def md(head: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


if __name__ == "__main__":
    kind, *paths = sys.argv[1:]
    print({"quality": quality, "price": price}[kind]([Path(p) for p in paths]))
