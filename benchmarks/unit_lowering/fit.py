"""The baseline curve: ANDs, bytes, peak memory and wall time of the flat lowering against the tile's terms, fitted to measured
shapes and projected to others.

    fit.py RECORD... [--project K,R,C ...] [--out FILE]

A RECORD is a run.py ``unit_lowering.json`` or a run's directory holding one (``out/unit_lowering.json``). Each quantity is fitted
by least squares on `TERMS`: a constant (the unit's fixed part), R C (the tile's cells, each lane R C / 32 of them), R C K (its
multiply-accumulates) and (R + C) K (its operand rows and columns). The fit is exact for ANDs where the measured shapes span the
terms; peak and wall are fitted on the same terms, so a projection far past the measured K says which term carries it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TERMS = ("1", "RC", "RCK", "(R+C)K")
QUANTITIES = ("ands", "bytes", "peak_gib", "wall_min")
PROJECT = ((2048, 32, 32), (8192, 32, 32), (128, 32, 32))


def terms(k: int, r: int, c: int) -> list[float]:
    return [1.0, r * c, r * c * k, (r + c) * k]


def load(path: Path) -> dict:
    for p in (path, path / "unit_lowering.json", path / "out" / "unit_lowering.json"):
        if p.is_file():
            d = json.loads(p.read_text())
            return {"k": d["k"], "tile": d["tile"], "form": d["form"], "ands": d["ands"], "bytes": d["bytes"],
                    "peak_gib": d["peak_gib"], "wall_min": d["wall_min"], "record": str(p)}
    raise SystemExit(f"fit: no unit_lowering.json at {path}")


def fit(points: list[dict], project: tuple[tuple[int, int, int], ...] = PROJECT) -> dict:
    """Each quantity's coefficients on `TERMS`, its largest relative residual over the points, and its value at each
    `project` shape."""
    X = np.array([terms(p["k"], *p["tile"]) for p in points])
    rank = int(np.linalg.matrix_rank(X))
    out = {"terms": list(TERMS), "rank": rank, "points": points, "fits": {}, "projections": []}
    for q in QUANTITIES:
        y = np.array([float(p[q]) for p in points])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = float(np.max(np.abs(X @ coef - y) / y))
        out["fits"][q] = {"coef": dict(zip(TERMS, map(float, coef))), "max_rel_residual": resid}
    for k, r, c in project:
        x = np.array(terms(k, r, c))
        row = {"k": k, "tile": [r, c]}
        for q in QUANTITIES:
            row[q] = float(x @ np.array(list(out["fits"][q]["coef"].values())))
        row["peak_b_per_and"] = row["peak_gib"] * (1 << 30) / row["ands"]
        out["projections"].append(row)
    return out


def table(doc: dict) -> str:
    lines = [f"fit on {len(doc['points'])} shapes (rank {doc['rank']} of {len(TERMS)} terms)", "",
             "| quantity | " + " | ".join(TERMS) + " | max residual |", "| --- " * (len(TERMS) + 2) + "|"]
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
    if len({p["form"] for p in points}) > 1:
        raise SystemExit("fit: the records mix forms; fit one form at a time")
    doc = fit(points, tuple(a.project or PROJECT))
    if a.out:
        a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(table(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
