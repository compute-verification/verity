"""RT2-LOGIC-1 at #2699's round-4 measured data: FOLD-OPT at every certificate against an ALU-only fold of ♭'s IMADs.

On DC-90-ri and DC-100-ri (explicit rows, `rowsMachine`), price class k at certificate λ as Σ_r λ_r a_rk. At
λ = 1/20·I + 19/20·UP and at the generated π*, compare N_w·φ*(λ), the cheapest fold route's price, with ♭'s fold
(`flatFold`) and with ♭'s fold with each IMAD replaced by 32 ALU shift-adds, and print whether FoldOptFolded at λ
(μ_f = 0, folded = N_w) is false there.
"""
import re
import sys
from fractions import Fraction as F

ROOT = sys.argv[1]
DIR = f"{ROOT}/Security/Definitions/Accounting/HeadlineDc"


def sparse(s):
    return [(int(a), F(int(b), int(c))) for a, b, c in re.findall(r"\((\d+), (\d+), (\d+)\)", s)]


for mod, rows_name in (("Sm90", "dc90ri"), ("Sm100", "dc100ri")):
    base = open(f"{DIR}/{mod}.lean").read()
    meas = open(f"{DIR}/{mod}Measured.lean").read()
    pipes = [p.strip() for p in re.search(r"pipe := \[(.*?)\]\s*$", base, re.S | re.M).group(1)
             .replace("\n", " ").split(",")]
    imad = {i for i, p in enumerate(pipes) if p == "some .imad"}
    names = re.findall(r'"(\w+)"', re.search(r"def rowNames : List String := \[(.*?)\]", meas, re.S).group(1))
    body = re.search(rf"def {rows_name} : List Sparse := \[(.*?)\]\]\s*$", meas, re.S | re.M).group(1) + "]"
    chunks = re.split(r"--\s*\d+:\s*\w+", body)[1:]
    rows = [dict(sparse(c)) for c in chunks]
    assert len(rows) == len(names), (len(rows), len(names))
    folds = re.search(r"folds := \[(.*?)\]\]", base, re.S).group(1)
    routes = [sparse(r) for r in re.findall(r"\[(\([^\]]*)\]", folds + "]")]
    nw = int(re.search(r"Nw := (\d+)", base).group(1))
    fold = sparse(re.search(r"def flatFold : Sparse := \[(.*?)\]", meas).group(1))
    star = sparse(re.search(r"cert \d+ \[(.*?)\]\s*$", meas, re.S | re.M).group(1))
    lam_star = {r: w for r, w in star}
    assert sum(lam_star.values()) == 1
    ix = {n: i for i, n in enumerate(names)}
    lam_rt2 = {ix["I"]: F(1, 20), ix["UP"]: F(19, 20)}

    print(f"{mod}Measured: rows {len(rows)}, IMAD pipe classes {sorted(imad)}, N_w {nw}")
    for label, lam in (("1/20 I + 19/20 UP", lam_rt2), ("pi* (generated)", lam_star)):
        def price(sp):
            return sum(q * sum(w * rows[r].get(k, 0) for r, w in lam.items()) for k, q in sp)

        per_word = [price(r) for r in routes]
        phi = min(per_word)
        imads = sum(q for k, q in fold if k in imad)
        lea = [(k, q) for k, q in fold if k not in imad] + [(0, 32 * imads)]
        print(f"  at {label}: ALU price {float(price([(0, 1)])):.6f}, IMAD price "
              f"{float(price([(min(imad), 1)])):.6f}")
        for i, p in enumerate(per_word):
            print(f"    route {i}: {float(p):.7f} a word")
        print(f"    N_w·φ* = {float(nw * phi):.4f}; ♭'s fold {float(price(fold)):.4f} ({float(imads)} IMAD a unit); "
              f"♭'s fold, IMAD→32 ALU: {float(price(lea)):.4f}; FoldOptFolded at λ false: "
              f"{'yes' if price(lea) < nw * phi else 'no'}")
