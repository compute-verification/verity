"""`real.py`'s summaries as one table per linear: admission (D-NF over the split unit, non-finite rows, the screened share),
and each linear's tiles' replayed debit split by item as a share of the cap at ρ = 1/400, with the tiles over the cap at
ρ = 1/400 and at a tighter ρ (a tile's cap scales with ρ, so it is over at ρ' when its ratio exceeds ρ'/ρ).  The first and
the last layer's ``down_proj`` are flagged.

  python real_table.py SUMMARY.json... [--rho 1/1000] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

RHO = Fraction(1, 400)
TERMS = ("two_four", "identity", "dead_pairs", "forming", "subgrid", "windows")


def x_files(summary: dict) -> dict[str, str]:
    """Each linear's X file: the capture's manifest for a directory, ``<name>.npy`` for an npz."""
    cap = Path(summary["capture"])
    if cap.suffix == ".npz" or not (cap / "manifest.json").exists():
        return {name: name + ".npy" for name in summary["tiles"]}
    return {name: e["file"] for name, e in json.loads((cap / "manifest.json").read_text())["inputs"].items()}


def layer(name: str) -> int:
    return int(re.search(r"layers\.(\d+)\.", name).group(1))


def rows(summaries: list[dict], rho: Fraction) -> list[dict]:
    out = []
    for s in summaries:
        files = x_files(s)
        for name, tiles in s["tiles"].items():
            x, w = s["inputs"].get(files.get(name, ""), {}), s["weights"].get(name, {})
            debited = [t for t in tiles if "ratio" in t]
            split = [{term: float(Fraction(t["debit"][term]) / Fraction(t["cap"])) for term in TERMS} for t in debited]
            out.append({
                "linear": name, "model": s.get("model"), "capture_sha256": s.get("manifest_sha256"),
                "x_rows": x.get("rows"), "x_dnf_fail_split": x.get("dnf_fail_split"), "x_non_finite": x.get("non_finite"),
                "x_screened_share": x["screened_blocks"] / x["blocks"] if x.get("blocks") else None,
                "x_rows_screened": x.get("rows_screened"),
                "w_rows": w.get("rows"), "w_dnf_fail_split": w.get("dnf_fail_split"), "w_non_finite": w.get("non_finite"),
                "tiles": len(tiles), "debited": len(debited), "rejected": sum(bool(t.get("rejected")) for t in tiles),
                "max_ratio": max((t["ratio"] for t in debited), default=None),
                "over_cap": sum(t["ratio"] > 1 for t in debited),
                "over_cap_tight": sum(t["ratio"] > float(rho / RHO) for t in debited),
                "mean_share": {term: sum(d[term] for d in split) / len(split) for term in TERMS} if split else None,
                "tile_shares": split, "screened_only_elements": [t["debit"]["counts"].get("screened_only_elements")
                                                                  for t in debited]})
    downs = sorted((layer(r["linear"]), r["model"], i) for i, r in enumerate(out) if r["linear"].endswith("down_proj"))
    for model in {m for _, m, _ in downs}:
        mine = [d for d in downs if d[1] == model]
        out[mine[0][2]]["flag"], out[mine[-1][2]]["flag"] = "first down_proj", "last down_proj"
    return out


def markdown(table: list[dict], rho: Fraction) -> str:
    def pct(v):
        return "" if v is None else f"{100 * v:.2f}%"

    head = ("| Linear | X D-NF fail (split) | X screened | W D-NF fail (split) | Tiles | Max debit/cap | "
            + " | ".join(TERMS) + f" | Over cap at 1/400 | Over cap at {rho} |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in table:
        name = r["linear"] + (f" (**{r['flag']}**)" if r.get("flag") else "")
        share = r["mean_share"] or {}
        lines.append(f"| {name} | {r['x_dnf_fail_split']}/{r['x_rows']} | {pct(r['x_screened_share'])} | "
                     f"{r['w_dnf_fail_split']}/{r['w_rows']} | {r['debited']}/{r['tiles']} | {pct(r['max_ratio'])} | "
                     + " | ".join(pct(share.get(t)) for t in TERMS) + f" | {r['over_cap']} | {r['over_cap_tight']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("summaries", nargs="+", type=Path)
    ap.add_argument("--rho", type=Fraction, default=Fraction(1, 1000))
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    table = rows([json.loads(p.read_text()) for p in a.summaries], a.rho)
    if a.json:
        a.json.write_text(json.dumps({"rho": str(RHO), "rho_tight": str(a.rho), "linears": table}, indent=1) + "\n")
    sys.stdout.write(markdown(table, a.rho))
    return 0


if __name__ == "__main__":
    sys.exit(main())
