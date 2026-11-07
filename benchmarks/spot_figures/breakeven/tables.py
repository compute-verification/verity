"""The break-even note's tables from sweep.py's surfaces: the levers alone, the replay design and its floor, the paths
to beta at 5% and 10%, and the paths over the six scopes.

    python benchmarks/spot_figures/breakeven/tables.py SURFACE.json...
"""
import json
import sys
from pathlib import Path

from report import tax

WEEK = "world week"


def load(paths):
    out = {}
    for p in paths:
        for r in json.loads(Path(p).read_text())["rows"]:
            out[(r["case"], r["scope"])] = r
    return out


def beta(b):
    return "never" if b is None else (f"{b:.3g}×" if b >= 1 else f"{b:.3g}" if b < 0.001 else f"{b:.2%}")


def total(case, b):
    return "never" if b is None else (f"{tax(case)[0] + b:.3g}×" if tax(case)[0] + b >= 1 else f"{tax(case)[0] + b:.2%}")


def betas(rows, case, scope=WEEK):
    r = rows.get((case, scope))
    return ("–", "–") if r is None else (beta(r["beta"]["FP8"]), beta(r["beta"]["FP4"]))


def alone(rows):
    spec = [
        ("none: the figure", "", "A", "P", "F"),
        ("(a) per-expert types at the card's 5.53e8, ×0.805", "measured", None, "P-card", "F-card"),
        ("(a) fused read's fixed cost summed with its GEMM's", "modeled", None, None, "F-fixedsum"),
        ("(b) statement's cap ×2", "assumed", "A-cap2", None, "F-cap2"),
        ("(b) statement's cap ×4", "assumed", "A-cap4", None, "F-cap4"),
        ("(b) statement's cap ×8", "assumed", "A-cap8", None, "F-cap8"),
        ("(b) cap ×4, statements shared in a datacenter's registration", "assumed", "A-cap4-dc", None, "F-cap4-dc"),
        ("(b) statements shared in a datacenter's registration", "planner", "A-dc", None, "F-dc"),
        ("(b) no fixed cost (the limit)", "limit", "A-nofixed", None, "F-nofixed"),
        ("(b) verifier waits overlapped: the busy reading", "measured readings", "A-busy", "P-busy", "F-busy"),
        ("(c) in-proof hashing ÷2 at 25% of the marginal", "assumed", "A-hash2-s25", None, "F-hash2-s25"),
        ("(c) ÷2 at 50%", "assumed", "A-hash2-s50", None, "F-hash2-s50"),
        ("(c) ÷4 at 50%", "assumed", "A-hash4-s50", None, "F-hash4-s50"),
        ("(c) ÷4 at 75%", "assumed", "A-hash4-s75", None, "F-hash4-s75"),
        ("(d) prover 2× faster", "assumed", "A-fast2", None, "F-fast2"),
        ("(d) prover 4× faster", "assumed", "A-fast4", None, "F-fast4"),
        ("(e) edges every 60 s, snapshot each", "modeled", "A-edge60-T90", None, "F-edge60-T90"),
        ("(e) edges every 10 s, snapshot each", "modeled", "A-edge10-T90", None, "F-edge10-T90"),
        ("(e) edges every 10 s, live K/V hashed", "modeled", "A-edge10-T90-live", None, "F-edge10-T90-live"),
        ("(e) a request's lifetime as its replay block", "modeled", "A-edge0-T90-live", None, "F-edge0-T90-live"),
    ]
    print("| lever | factor | (A) FP8 | (A) FP4 | (A′) FP8 | (A′) FP4 | (A′) fused FP8 | (A′) fused FP4 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for name, label, a, p, f in spec:
        cells = []
        for c in (a, p, f):
            cells += list(betas(rows, c)) if c else ["", ""]
        print(f"| {name} | {label} | " + " | ".join(cells) + " |")


def replay(rows):
    spec = [("600 s, one window a request (the figure)", "fig", 600, None),
            ("600 s, every window a request spans", "T90", 600, None),
            ("60 s, snapshot at each edge", "T90", 60, None),
            ("10 s, snapshot at each edge", "T90", 10, None),
            ("10 s, live K/V hashed at each edge", "T90-live", 10, None),
            ("a request's lifetime, replayed live", "T90-live", 0, None)]
    print("| replay block | replay a sampled request, GPU-s | serve tax | (A) at β = 1% | (A) floor | (A′) fused floor "
          "| (A) β at FP8 | (A′) fused β at FP8 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, kind, edge, _ in spec:
        a, f = f"A-edge{edge}-{kind}", f"F-edge{edge}-{kind}"
        ra, rf = rows.get((a, WEEK)), rows.get((f, WEEK))
        gpu_s = 8 * edge if kind == "fig" else 8 * (edge + 90)
        if ra is None:
            continue
        print(f"| {name} | {gpu_s:,} | {tax(a)[0]:.2%} | {ra['bound']['0.01'] / 8e12:,.0f} TB | "
              f"{ra['floor'] / 8e9:.3g} GB | {rf['floor'] / 8e9:.3g} GB | {beta(ra['beta']['FP8'])} | "
              f"{beta(rf['beta']['FP8'])} |" if rf else "")
    print()
    print("| request lifetime T | 600 s windows | 60 s | 10 s | lifetime replay |")
    print("|---|---:|---:|---:|---:|")
    for t in (30, 90, 300):
        cells = []
        for edge, kind in ((600, f"T{t}"), (60, f"T{t}"), (10, f"T{t}"), (0, f"T{t}-live")):
            r = rows.get((f"A-edge{edge}-{kind}", WEEK))
            cells.append("–" if r is None else f"{r['floor'] / 8e9:.3g} GB")
        print(f"| {t} s | " + " | ".join(cells) + " |")


def paths(rows, cases):
    print("| path | levers | bound at β = 1% | β at FP8 | total at FP8 | β at FP4 | total at FP4 | floor |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for name, levers in cases:
        r = rows.get((name, WEEK))
        if r is None:
            continue
        b8, b4 = r["beta"]["FP8"], r["beta"]["FP4"]
        print(f"| {name} | {levers} | {r['bound']['0.01'] / 8e12:,.3g} TB | {beta(b8)} | {total(name, b8)} | "
              f"{beta(b4)} | {total(name, b4)} | {r['floor'] / 8e9:.3g} GB |")


def scopes(rows, cases):
    order = ["world week", "world month", "world year", "datacenter week", "datacenter month", "datacenter year"]
    print("| path | " + " | ".join(order) + " |")
    print("|---|" + "---:|" * len(order))
    for name in cases:
        cells = []
        for s in order:
            r = rows.get((name, s))
            cells.append("–" if r is None else f"{beta(r['beta']['FP8'])} / {beta(r['beta']['FP4'])}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print()
    print("| path | " + " | ".join(order) + " |")
    print("|---|" + "---:|" * len(order))
    for name in cases:
        cells = []
        for s in order:
            r = rows.get((name, s))
            cells.append("–" if r is None else f"{r['bound']['0.05'] / 8e12:,.3g} TB")
        print(f"| {name} at β = 5% | " + " | ".join(cells) + " |")


PATHS = [
    ("A", "the figure's (A)"),
    ("F", "fused cut"),
    ("F-busy", "fused cut, overlapped waits"),
    ("F+busy+fast2", "+ prover ×2"),
    ("F+busy+fast4", "+ prover ×4"),
    ("F+busy+fast2+hash2", "+ prover ×2, hashing ÷2 at 50%"),
    ("F+busy+fast2+hash4", "+ prover ×2, hashing ÷4 at 50%"),
    ("F+busy+fast2+hash2+life", "+ prover ×2, hashing ÷2, lifetime replay"),
    ("F+busy+fast2+hash4+life", "+ prover ×2, hashing ÷4, lifetime replay"),
    ("F+busy+fast4+life", "+ prover ×4, lifetime replay"),
    ("F+busy+fast4+hash4", "+ prover ×4, hashing ÷4 at 50%"),
    ("F+busy+fast4+hash4+life", "+ prover ×4, hashing ÷4, lifetime replay"),
    ("F+busy+fast4+hash4+live10", "+ prover ×4, hashing ÷4, live 10 s edges"),
    ("F+busy+fast4+hash4s75", "+ prover ×4, hashing ÷4 at 75%"),
    ("F+busy+fast4+hash4s75+life", "+ prover ×4, hashing ÷4 at 75%, lifetime replay"),
    ("F+busy+fast8", "+ prover ×8"),
    ("F+busy+fast8+hash4", "+ prover ×8, hashing ÷4 at 50%"),
    ("F+busy+fast8+hash4+life", "+ prover ×8, hashing ÷4, lifetime replay"),
    ("F+busy+fast8+hash4s75+life", "+ prover ×8, hashing ÷4 at 75%, lifetime replay"),
    ("F+cap4+fast4", "fused cut, cap ×4, prover ×4"),
    ("F+cap4+fast4+hash4", "fused cut, cap ×4, prover ×4, hashing ÷4"),
    ("F+nofixed+fast4+hash4", "fused cut, no fixed cost, prover ×4, hashing ÷4"),
    ("A+busy+fast4+hash4+live10", "(A), overlapped waits, prover ×4, hashing ÷4, live 10 s edges"),
]


def main():
    rows = load(sys.argv[1:])
    print("## Levers alone, world week: β at FP8 and FP4\n")
    alone(rows)
    print("\n## The replay design\n")
    replay(rows)
    print("\n## Paths\n")
    paths(rows, PATHS)
    print("\n## Paths over six scopes: β at FP8 / FP4\n")
    scopes(rows, ["A", "F", "F-busy", "F+busy+fast2", "F+busy+fast2+hash4", "F+busy+fast4", "F+busy+fast4+hash4",
                  "F+busy+fast2+hash4+live10", "F+busy+fast4+hash4+live10"])


if __name__ == "__main__":
    main()
