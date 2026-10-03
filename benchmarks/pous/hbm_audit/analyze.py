"""Turn the HBM-audit measurement JSONs into the separation table and the recommended `(R, T)`.

Reads `honest.json` and `cheater.json` (from `driver.py`) in `--dir` and writes `separation.md` and `summary.json`:

- the honest digest-answer latency over `R` HBM-resident blocks (its `p50`, `p99`, `p999` and the device read rate it
  achieves), from `honest.json`'s `digest1`;
- the cheater's best honest time when a fraction `f` of the challenged blocks live in pinned host DRAM, the faster of the
  two methods (async H2D copy overlapped with the device digest, or UVA zero-copy reads), from `cheater.json`;
- the separation: cheater(R, f) / honest(R) time ratio, and for each `R` the deadline `T` and the fraction `f` that a
  deadline tuned to HBM speed distinguishes with margin.

A deadline `T` is called *distinguishing at `(R, f)` with margin* when the honest answer is reliably under it and the
cheater reliably over it: `honest.p999(R) <= T` and `cheater_best.min(R, f) > T`. The recommended `T` for an `R` is set at
the honest `p999` with a small guard; the headline is the least `f` such a `T` separates with the required margin ratio
`cheater_best.min / honest.p999 >= --margin`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BLOCK_BYTES = 65536


def _rate_gbps(nbytes: int, ms: float) -> float:
    return nbytes / (ms / 1e3) / 1e9 if ms > 0 else 0.0


def load(dir_: Path):
    honest = json.loads((dir_ / "honest.json").read_text())
    cheater = json.loads((dir_ / "cheater.json").read_text())
    return honest, cheater


def honest_by_r(honest: dict, c_gib: int) -> dict:
    """`{R: digest1-distribution}` at the given `C` size (the audit answer is the one-lane digest)."""
    out = {}
    for pt in honest["points"]:
        if pt["c_gib"] == c_gib:
            out[pt["R"]] = pt["digest1"]
    return out


def cheater_best_by_rf(cheater: dict) -> dict:
    """`{(R, f): {method, ms_p50, ms_min, ms_p99}}`, the faster of async-copy and zero-copy at each `(R, f)`."""
    out = {}
    for pt in cheater["points"]:
        r, f = pt["R"], pt["f"]
        a, z = pt.get("async", {}), pt.get("zerocopy", {})
        best = min((a, z), key=lambda d: d.get("ms_p50", float("inf")))
        method = "async" if best is a else "zerocopy"
        out[(r, f)] = {"method": method, "ms_p50": best["ms_p50"], "ms_min": best["ms_min"],
                       "ms_p99": best.get("ms_p99", best["ms_p50"]), "R_host": pt.get("R_host", 0),
                       "async_p50": a.get("ms_p50"), "zerocopy_p50": z.get("ms_p50"),
                       "pcie_gbps_p50": best.get("pcie_gbps_at_p50", 0.0)}
    return out


def analyze(honest: dict, cheater: dict, margin: float):
    c_gib = cheater["c_gib"]
    hon = honest_by_r(honest, c_gib)
    che = cheater_best_by_rf(cheater)
    fracs = sorted({f for (_, f) in che})
    rs = sorted(hon)
    rows = []
    for r in rs:
        h = hon.get(r)
        if h is None:
            continue
        nbytes = r * BLOCK_BYTES
        row = {"R": r, "honest_p50": h["ms_p50"], "honest_p99": h["ms_p99"], "honest_p999": h["ms_p999"],
               "honest_gbps_p50": _rate_gbps(nbytes, h["ms_p50"]), "fracs": {}}
        T = h["ms_p999"]  # deadline tuned to the honest HBM answer
        row["T_ms"] = T
        for f in fracs:
            c = che.get((r, f))
            if not c:
                continue
            ratio_typ = c["ms_p50"] / h["ms_p50"] if h["ms_p50"] else 0.0
            ratio_margin = c["ms_min"] / T if T else 0.0  # cheater's best case vs the deadline
            row["fracs"][f] = {"cheater_p50": c["ms_p50"], "cheater_min": c["ms_min"], "method": c["method"],
                               "ratio_typ": ratio_typ, "ratio_margin": ratio_margin,
                               "separated": c["ms_min"] > T and h["ms_p999"] <= T and ratio_margin >= margin,
                               "pcie_gbps_p50": c["pcie_gbps_p50"]}
        rows.append(row)
    # the least f separated with margin, per R, and a recommended (R, T)
    for row in rows:
        sep_fs = [f for f, v in row["fracs"].items() if v["separated"]]
        row["min_separated_f"] = min(sep_fs) if sep_fs else None
    rec = pick_recommendation(rows, margin)
    return {"c_gib": c_gib, "margin": margin, "fracs": fracs, "rows": rows, "recommendation": rec,
            "raw": honest.get("raw", {})}


def pick_recommendation(rows: list[dict], margin: float):
    """The smallest `R` that separates `f = 5%` with the margin, and its `T`; else the `R` with the best 5% margin."""
    target = 0.05
    with5 = [r for r in rows if target in r["fracs"]]
    sep = [r for r in with5 if r["fracs"][target]["separated"]]
    pick = min(sep, key=lambda r: r["R"]) if sep else (
        max(with5, key=lambda r: r["fracs"][target]["ratio_margin"]) if with5 else None)
    if not pick:
        return None
    v = pick["fracs"][target]
    return {"R": pick["R"], "T_ms": pick["T_ms"], "honest_p999_ms": pick["honest_p999"],
            "cheater5_min_ms": v["cheater_min"], "cheater5_p50_ms": v["cheater_p50"],
            "margin_ratio": v["ratio_margin"], "separated": v["separated"], "method": v["method"]}


def to_markdown(a: dict) -> str:
    fracs = a["fracs"]
    nonzero = [f for f in fracs if f > 0]
    lines = [f"## Separation on C = {a['c_gib']} GiB (RTX PRO 6000 Blackwell)", "",
             "Honest = one-lane digest over R HBM-resident 64 KiB blocks (device). "
             "Cheater = faster of async-copy-overlap and UVA zero-copy, fraction f of blocks in pinned host DRAM. "
             "Times in ms; ratio = cheater_p50 / honest_p50.", ""]
    head = ["R", "honest p50", "honest p999 (=T)", "GB/s"] + [f"f={int(f*100)}% p50 (x)" for f in nonzero]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] * len(head)) + "|")
    for row in a["rows"]:
        cells = [str(row["R"]), f"{row['honest_p50']:.3f}", f"{row['honest_p999']:.3f}",
                 f"{row['honest_gbps_p50']:.0f}"]
        for f in nonzero:
            v = row["fracs"].get(f)
            if v:
                star = "*" if v["separated"] else ""
                cells.append(f"{v['cheater_p50']:.3f} ({v['ratio_typ']:.2f}x){star}")
            else:
                cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "`*` = deadline T = honest p999 separates this (R, f) with the required margin "
              f"(cheater best-case / T >= {a['margin']}).", ""]
    rec = a["recommendation"]
    if rec:
        lines += ["### Recommendation", "",
                  f"- R = {rec['R']}, T = {rec['T_ms']*1000:.0f} us ({rec['T_ms']:.3f} ms)",
                  f"- honest p999 = {rec['honest_p999_ms']:.3f} ms; cheater(f=5%) best-case = {rec['cheater5_min_ms']:.3f} ms "
                  f"(p50 {rec['cheater5_p50_ms']:.3f} ms), method = {rec['method']}",
                  f"- margin ratio (cheater best-case / T) = {rec['margin_ratio']:.2f}x; "
                  f"separates f>=5% with margin: {rec['separated']}", ""]
    else:
        lines += ["### Recommendation", "", "No f=5% point in the cheater sweep.", ""]
    raw = a.get("raw", {})
    if raw:
        lines += ["### Reference rates", ""]
        for k, v in raw.items():
            if isinstance(v, dict) and "gbps_at_p50" in v:
                lines.append(f"- {k}: {v['gbps_at_p50']:.0f} GB/s (p50 {v['ms_p50']:.3f} ms)")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--margin", type=float, default=1.5, help="required cheater-best-case / T ratio to call f separated")
    args = ap.parse_args()
    d = Path(args.dir)
    honest, cheater = load(d)
    a = analyze(honest, cheater, args.margin)
    (d / "summary.json").write_text(json.dumps(a, indent=2))
    md = to_markdown(a)
    (d / "separation.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
