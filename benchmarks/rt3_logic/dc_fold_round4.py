"""RT2-LOGIC-1 at #2699's round-4 data: FOLD-OPT at every certificate against an ALU-only fold of ♭'s IMADs.

At λ = 1/20·issue + 19/20·IMAD (sheetCoef: issue 1 a class, a pipe 128/width), compare N_w·φ*(λ), the cheapest fold
route's price, with the price of ♭'s words, skippable work and fold with each IMAD replaced by 32 ALU shift-adds.
"""
import re
import sys
from fractions import Fraction as F

ROOT = sys.argv[1]
LAM_ISSUE, LAM_IMAD = F(1, 20), F(19, 20)


def block(text, name):
    m = re.search(rf"{name} := \[(.*?)\]\s*$", text, re.S | re.M)
    return m.group(1)


def sparse(s):
    return [(int(a), F(int(b), int(c))) for a, b, c in re.findall(r"\((\d+), (\d+), (\d+)\)", s)]


for mod, n in (("Sm90", 29), ("Sm100", 26)):
    text = open(f"{ROOT}/Security/Definitions/Accounting/HeadlineDc/{mod}.lean").read()
    pipes = [p.strip() for p in block(text, "pipe").replace("\n", " ").split(",")]
    imad = {i for i, p in enumerate(pipes) if p == "some .imad"}
    widths = dict(re.findall(r"\| \.(\w+) => (\d+)", text.split("def Pipe.width")[1].split("def ")[0]))
    c_imad = F(128, int(widths["imad"]))

    def price(sp):
        return sum(q * (LAM_ISSUE + (LAM_IMAD * c_imad if k in imad else 0)) for k, q in sp)

    folds = re.search(r"folds := \[(.*?)\]\]", text, re.S).group(1)
    routes = [sparse(r) for r in re.findall(r"\[(\([^\]]*)\]", folds + "]")]
    nw = int(re.search(r"Nw := (\d+)", text).group(1))
    per_word = [price(r) for r in routes]
    phi = min(per_word)
    words = sparse(re.search(r"def flatWords : Sparse := \[(.*?)\]", text).group(1))
    fold = sparse(re.search(r"def flatFold : Sparse := \[(.*?)\]", text).group(1))
    skip = sparse(re.search(r"def flatSkip : Sparse := \[(.*?)\]", text).group(1))
    imads = sum(q for k, q in fold if k in imad)
    lea_fold = [(k, q) for k, q in fold if k not in imad] + [(0, 32 * imads)]
    lea = price(words) + price(lea_fold) + price(skip)
    print(f"{mod}: IMAD pipe classes {sorted(imad)}, c_imad {c_imad}, N_w {nw}")
    for i, (r, p) in enumerate(zip(routes, per_word)):
        print(f"  route {i}: {p} = {float(p):.7f} a word")
    print(f"  N_w·φ*(λ) = {nw * phi} = {float(nw * phi):.4f}; ♭'s fold IMADs {imads} a unit")
    print(f"  LEA program at λ: {float(lea):.4f}; FOLD-OPT fails once folded > {float(lea / phi):.1f} of {nw} words: "
          f"{'yes' if lea < nw * phi else 'no'}")
