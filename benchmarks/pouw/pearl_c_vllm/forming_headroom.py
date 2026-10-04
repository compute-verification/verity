"""Whether the forming term the served debit leaves out could flip a measured tile's verdict.  `pearl_c_debit.forming_elements`
is a stub (0), and Lean's tile debit (`PearlCSem.tileDebitRev1`) adds fs·(|C|/n)·elems, elems the elements of the tile's
passing rows R that are salt-dead (`saltDeadP`) or unneeded (every credited column flagged at the element's word).

From each recorded line (`served_debit.py`'s ``*-tiles.jsonl``), exactly, at the cited statement's prices (`STATEMENT`,
`Prices.sm120Loop`; c moves W_ref only, not capOK): the unneeded elements are at most 32·⌊atoms/|C|⌋, since each is in a
row and word whose |C| columns are all flagged and `atoms` counts them; so the verdict flips only if the salt-dead elements
exceed

    (ρ·creditDevRev1(m, k, n)·|R|·|C|/(m·n) − (32 + add/G)·atoms − add·zeros)·n/(fs·|C|) − 32·⌊atoms/|C|⌋,

given as `dead_to_flip`, a fraction of the tile's |R|·k elements.  Lean bounds a unit's salt-dead elements at ρ_D = 1/64 of
its m·k (`FormingP.saltDeadP_card_le`, under its domain conditions), which covers a tile whose rows are its unit's
(`unit_rows`: decode, |R| = m); a tile of fewer rows has the threshold alone.  Every line's own debit and cap are first
recomputed at `Prices.sm120` (the scheme's, `LINE`), and a line they don't match is refused.

  python benchmarks/pouw/pearl_c_vllm/forming_headroom.py LINES.jsonl ... --out FILE
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

RHO_D = Fraction(1, 64)
#: (add, bf16, fs, G, r, ρ): the served lines' prices (`Prices.sm120`) and the cited statement's (`Prices.sm120Loop`)
LINE = (Fraction(8), 2, 40, 4, 32, Fraction(1, 1000))
STATEMENT = (Fraction(1047, 125), 2, 40, 4, 32, Fraction(1, 1000))


def credit_rev1(m: int, k: int, n: int, prices: tuple) -> Fraction:
    """Lean's `creditDevRev1` at `prices`."""
    add, bf16, fs, g, r, _ = prices
    mkn = Fraction(m * k * n)
    return mkn + mkn * add / (32 * g) + m * k * r + 2 * bf16 * r * m * n + fs * m * k - add * m * n


def tile(line: dict, prices: tuple) -> tuple[Fraction, Fraction]:
    """A line's (debit without forming, cap) at `prices`."""
    add, _, _, g, _, rho = prices
    m, k, n, rows, cols = line["m"], line["k"], line["n"], line["credited_rows"], line["credited_cols"]
    debit = (32 + add / g) * line["atoms"] + add * line["later_zero_promotions"]
    return debit, rho * credit_rev1(m, k, n, prices) * Fraction(rows * cols, m * n)


def dead_to_flip(line: dict, prices: tuple = STATEMENT) -> Fraction:
    """The fraction of the tile's |R|·k elements that must be salt-dead for the forming term to flip its verdict."""
    fs, k, n, rows, cols = prices[2], line["k"], line["n"], line["credited_rows"], line["credited_cols"]
    debit, cap = tile(line, prices)
    elems = (cap - debit) * n / (fs * cols) - 32 * (line["atoms"] // cols)
    return elems / (rows * k)


def check_line(line: dict) -> None:
    debit, cap = tile(line, LINE)
    if line["credited_rows"] == 0 or not line["ok"]:
        raise ValueError(f"tile {line['tile']}: a rejected or rowless tile has no headroom to report")
    if Fraction(line["debit_exact"]) != debit or abs(Fraction(line["cap"]) - cap) > Fraction(1, 10 ** 6) * cap:
        raise ValueError(f"tile {line['tile']}: its recorded debit {line['debit_exact']} or cap {line['cap']} is not "
                         f"Prices.sm120's {debit} / {float(cap)}")


def report(paths: list[Path]) -> dict:
    inputs = {}
    for path in paths:
        for raw in path.read_text().splitlines():
            line = json.loads(raw)
            check_line(line)
            key = (path.parent.name, path.name.split(".")[0], line["matmul"])
            r = inputs.setdefault(key, {"measurement": key[0], "pass": key[1], "matmul": line["matmul"], "k": line["k"],
                                        "n": line["n"], "tiles": 0, "unit_rows": True, "dead_to_flip": None, "worst_tile": None})
            f = dead_to_flip(line)
            r["tiles"] += 1
            r["unit_rows"] &= line["credited_rows"] == line["m"]
            if r["dead_to_flip"] is None or f < r["dead_to_flip"]:
                r["dead_to_flip"], r["worst_tile"] = f, line["tile"]
    rows = sorted(inputs.values(), key=lambda r: (r["measurement"], r["pass"], r["matmul"]))
    worst = min(rows, key=lambda r: r["dead_to_flip"])
    covered = all(r["dead_to_flip"] > RHO_D for r in rows if r["unit_rows"])
    return {"tool": "pouw_pearl_c_forming_headroom", "prices": "Prices.sm120Loop", "rho_D": str(RHO_D),
            "lines": [str(p) for p in paths], "tiles": sum(r["tiles"] for r in rows),
            "dead_to_flip_min": float(worst["dead_to_flip"]),
            "worst": {k: worst[k] for k in ("measurement", "pass", "matmul", "k", "n")},
            "unit_rows_covered_by_rho_D": covered,
            "inputs": [r | {"dead_to_flip": float(r["dead_to_flip"])} for r in rows]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("lines", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    doc = report(a.lines)
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{doc['tiles']} tiles: the fewest salt-dead elements that flip a verdict is {doc['dead_to_flip_min']:.4f} of a "
          f"tile's |R|·k ({doc['worst']}); ρ_D = 1/64 = {float(RHO_D):.4f}; tiles of a unit's rows covered by ρ_D: "
          f"{doc['unit_rows_covered_by_rho_D']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
