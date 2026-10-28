"""The baseline curve: ANDs, bytes, peak memory and wall time of the flat lowering against the tile's terms, fitted to measured
shapes and projected to others.

    fit.py RECORD... [--project K,R,C ...] [--out FILE]

A RECORD is a run.py ``unit_lowering.json`` or a run's directory holding one (``out/unit_lowering.json``), all of one unit and
form, and for a lane or combine of one leaf. Each quantity is fitted by least squares on the unit's terms (`TERMS`): for the tile
a constant (its fixed part), R C (its cells, each lane R C / 32 of them), R C K (its multiply-accumulates) and (R + C) K (its
operand rows and columns); for a leaf's lane and combine a constant and K. Peak and wall are fitted on the same terms as ANDs,
so a projection far past the measured K says which term carries it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TERMS = {"tile": ("1", "RC", "RCK", "(R+C)K"), "lane": ("1", "K"), "combine": ("1", "K")}
QUANTITIES = ("ands", "bytes", "peak_gib", "wall_min")
#: the tile's projections, and each K a leaf's lane and combine are projected to at their own leaf
PROJECT = ((2048, 32, 32), (8192, 32, 32), (128, 32, 32))
LEAF_K = (2048, 8192, 14336)


def terms(unit: str, k: int, r: int, c: int) -> list[float]:
    return [1.0, r * c, r * c * k, (r + c) * k] if unit == "tile" else [1.0, k]


def load(path: Path) -> dict:
    for p in (path, path / "unit_lowering.json", path / "out" / "unit_lowering.json"):
        if p.is_file():
            d = json.loads(p.read_text())
            return {"unit": d.get("unit", "tile"), "k": d["k"], "tile": d["tile"], "form": d["form"], "ands": d["ands"], "bytes": d["bytes"],
                    "peak_gib": d["peak_gib"], "wall_min": d["wall_min"], "record": str(p)}
    raise SystemExit(f"fit: no unit_lowering.json at {path}")


def fit(points: list[dict], project: tuple[tuple[int, int, int], ...] | None = None) -> dict:
    """Each quantity's coefficients on its unit's `TERMS`, its largest relative residual over the points, and its value at each
    ``project`` shape (by default the tile's `PROJECT`, or a leaf's `LEAF_K` at its own R x C)."""
    unit = points[0]["unit"]
    if project is None:
        project = PROJECT if unit == "tile" else tuple((k, *points[0]["tile"]) for k in LEAF_K)
    X = np.array([terms(unit, p["k"], *p["tile"]) for p in points])
    rank = int(np.linalg.matrix_rank(X))
    out = {"unit": unit, "terms": list(TERMS[unit]), "rank": rank, "points": points, "fits": {}, "projections": []}
    for q in QUANTITIES:
        y = np.array([float(p[q]) for p in points])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = float(np.max(np.abs(X @ coef - y) / y))
        out["fits"][q] = {"coef": dict(zip(TERMS[unit], map(float, coef))), "max_rel_residual": resid}
    for k, r, c in project:
        x = np.array(terms(unit, k, r, c))
        row = {"k": k, "tile": [r, c]}
        for q in QUANTITIES:
            row[q] = float(x @ np.array(list(out["fits"][q]["coef"].values())))
        row["peak_b_per_and"] = row["peak_gib"] * (1 << 30) / row["ands"]
        out["projections"].append(row)
    return out


def table(doc: dict) -> str:
    names = doc["terms"]
    lines = [f"{doc['unit']}: fit on {len(doc['points'])} shapes (rank {doc['rank']} of {len(names)} terms)", "",
             "| quantity | " + " | ".join(names) + " | max residual |", "| --- " * (len(names) + 2) + "|"]
    for q, f in doc["fits"].items():
        lines.append(f"| {q} | " + " | ".join(f"{v:.4g}" for v in f["coef"].values()) + f" | {f['max_rel_residual']:.1%} |")
    lines += ["", "| K | tile | ANDs | bytes | peak GiB | B/AND at peak | wall min |", "| --- " * 7 + "|"]
    for p in doc["projections"]:
        lines.append(f"| {p['k']} | {p['tile'][0]}x{p['tile'][1]} | {p['ands']:,.0f} | {p['bytes']:,.0f} | {p['peak_gib']:,.1f} | "
                     f"{p['peak_b_per_and']:,.0f} | {p['wall_min']:,.1f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("records", nargs="+", type=Path)
    ap.add_argument("--project", action="append", type=lambda v: tuple(int(x) for x in v.split(",")), metavar="K,R,C")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    points = [load(p) for p in a.records]
    if len({(p["unit"], p["form"], None if p["unit"] == "tile" else tuple(p["tile"])) for p in points}) > 1:
        raise SystemExit("fit: the records mix units, forms or a lane's or combine's leaves; fit one of each at a time")
    doc = fit(points, tuple(a.project) if a.project else None)
    if a.out:
        a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(table(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
