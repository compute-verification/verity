"""`real.py`'s summaries as one table per linear: admission (D-NF over the split unit, non-finite rows, the screened share),
and each linear's tiles' replayed debit split by item as a share of the cap at ρ = 1/400, with the tiles over the cap at
ρ = 1/400 and at a tighter ρ (a tile's cap scales with ρ, so it is over at ρ' when its ratio exceeds ρ'/ρ).  The first and
the last layer's ``down_proj`` are flagged.  Per model, the share of MACs over the cap at both ρ: each linear's MACs
(m·n·k over the split unit) times the fraction of its sampled tiles over the cap.

  python real_table.py SUMMARY.json... [--rho 1/1000] [--json OUT] [--manifest CAPTURE_MANIFEST.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

RHO = Fraction(1, 400)
#: The debit's items (F1′ is `split`, F2 `int8`); `two_four` is the retired item 1, which summaries replayed before F1′
#: carry in their place.
TERMS = ("split", "int8", "two_four", "identity", "dead_pairs", "forming", "subgrid", "windows")


def x_files(summary: dict, manifest: Path | None = None) -> dict[str, str]:
    """Each linear's X file: the capture's manifest for a directory (or ``manifest``, a copy of it, where the capture
    isn't), ``<name>.npy`` for an npz."""
    cap = Path(summary["capture"])
    if cap.suffix != ".npz" and manifest is None and (cap / "manifest.json").exists():
        manifest = cap / "manifest.json"
    if cap.suffix == ".npz" or manifest is None:
        return {name: name + ".npy" for name in summary["tiles"]}
    return {name: e["file"] for name, e in json.loads(manifest.read_text())["inputs"].items()}


def layer(name: str) -> int:
    return int(re.search(r"layers\.(\d+)\.", name).group(1))


def rows(summaries: list[dict], rho: Fraction, manifest: Path | None = None) -> list[dict]:
    out = []
    for s in summaries:
        files = x_files(s, manifest)
        for name, tiles in s["tiles"].items():
            x, w = s["inputs"].get(files.get(name, ""), {}), s["weights"].get(name, {})
            debited = [t for t in tiles if "ratio" in t]
            split = [{term: float(Fraction(t["debit"].get(term, 0)) / Fraction(t["cap"])) for term in TERMS} for t in debited]
            size = next((t for t in tiles if "k_kept" in t), {})
            out.append({
                "linear": name, "model": s.get("model"), "capture_sha256": s.get("manifest_sha256"),
                "macs": size.get("m", 0) * size.get("n", 0) * size.get("k_kept", 0),
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


def mac_shares(table: list[dict]) -> dict[str, dict]:
    """Per model, the share of its measured linears' MACs whose sampled tiles are over the cap (at 1/400 and at the tighter
    ρ) or rejected: each linear weighs the fraction of its tiles by its MACs."""
    out: dict[str, dict] = {}
    for r in table:
        if not r["tiles"]:
            continue
        acc = out.setdefault(r["model"], {"linears": 0, "tiles": 0, "macs": 0, "over_cap": 0.0, "over_cap_tight": 0.0,
                                          "rejected": 0.0})
        acc["linears"] += 1
        acc["tiles"] += r["tiles"]
        acc["macs"] += r["macs"]
        for key in ("over_cap", "over_cap_tight", "rejected"):
            acc[key] += r["macs"] * r[key] / r["tiles"]
    for acc in out.values():
        for key in ("over_cap", "over_cap_tight", "rejected"):
            acc[key] = acc[key] / acc["macs"] if acc["macs"] else None
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
    lines.append("")
    for model, acc in mac_shares(table).items():
        lines.append(f"- {model}: over the cap on {pct(acc['over_cap'])} of MACs at 1/400 and {pct(acc['over_cap_tight'])} "
                     f"at {rho}; rejected tiles on {pct(acc['rejected'])} ({acc['tiles']} tiles over {acc['linears']} "
                     "linears, each linear's share its sampled tiles' fraction weighted by m·n·k)")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("summaries", nargs="+", type=Path)
    ap.add_argument("--rho", type=Fraction, default=Fraction(1, 1000))
    ap.add_argument("--json", type=Path)
    ap.add_argument("--manifest", type=Path, help="a copy of the capture's manifest.json, where the capture isn't")
    a = ap.parse_args(argv)
    table = rows([json.loads(p.read_text()) for p in a.summaries], a.rho, a.manifest)
    if a.json:
        a.json.write_text(json.dumps({"rho": str(RHO), "rho_tight": str(a.rho), "mac_shares": mac_shares(table),
                                      "linears": table}, indent=1) + "\n")
    sys.stdout.write(markdown(table, a.rho))
    return 0


if __name__ == "__main__":
    sys.exit(main())
