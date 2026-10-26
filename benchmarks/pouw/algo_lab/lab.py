"""Every route of (0) and (a) to (e), run exactly and checked against `segment.py` word for word and receipt for receipt,
then priced on DS-90 a warp-word against the reference.

``python -m algo_lab.lab`` (from benchmarks/pouw) prints both tables. Columns:

- **time**: max over DS-90's resources (tensor, issue, ALU pipe, IMAD pipe, SHFL, uniform, shared memory) of the
  route's SMSP-cycles a warp-word, over the reference's (1.0625, ALU-bound by LI-CP-W's cp-seg4).
- **issue**: the issue slots alone over the reference's (1.0469): the Lean bound's DS-90 certificate, `DS_issue`, which
  prices every instruction at one slot.
- **pooled**: time with the ALU and IMAD pipes pooled into one of twice the width, over the reference's time: a lower
  bound for any rebalancing of the route's integer work between the two pipes, so it favours the route.
- **setup**: SMSP-cycles a warp-word of the work done once (tables, operand forming), at n = 64; it shrinks as k grows.

The fold table (e) prices each fold program on the reference's words and tensor path, and gives its lane instructions a
word (the fold's own, then the checkpoint's), with the fan-in floor of 1/2 a word beside them.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import bitslice as BS
from . import dither as D
from . import folds as F
from . import packed as P
from . import price as PR
from . import reference as REF
from . import russians as RU
from . import strassen as ST
from .isa import Kernel

N, KS, SEED, FSEED = 64, 2, 7, 1


@dataclass
class Row:
    name: str
    load: dict[str, float]
    setup: float
    exact: bool

    def cols(self, ref: dict[str, float]) -> dict[str, float | str]:
        t, bind = PR.time(self.load)
        pool = dict(self.load)
        pool["ALU"] = pool["IMAD"] = (self.load["ALU"] + self.load["IMAD"]) / 2
        return {"time": t / PR.time(ref)[0], "bind": bind, "issue": self.load["issue"] / ref["issue"],
                "pooled": PR.time(pool)[0] / PR.time(ref)[0], "tensor": self.load["tensor"], "setup": self.setup}


def instances() -> tuple[D.Instance, D.Instance]:
    return D.instance(N, KS * D.L, SEED), D.instance(N, KS * D.L, SEED, general=True)


def priced(name: str, inst: D.Instance, run: REF.Run, fold: str = "li") -> Row:
    words = inst.ks * inst.n * inst.n
    exact = bool(np.array_equal(run.words, D.words(inst))
                 and np.array_equal(run.receipt, D.receipt(run.words, fold, FSEED)))
    setup = PR.time(PR.loads(run.kernel.by_phase("setup"), words))[0]
    return Row(name, PR.loads(run.kernel.by_phase("word"), words), setup, exact)


#: (name, general A and B, the run)
ROUTES: list[tuple[str, bool, Callable[[D.Instance], REF.Run]]] = [
    ("(0) reference: wgmma s8 k32, LI, cp-seg4", False, lambda i: REF.run(i, seed=FSEED)),
    ("(a) Strassen depth 1, A = B = 0", False, lambda i: ST.run(i, 1, seed=FSEED)),
    ("(a) Strassen depth 2, A = B = 0", False, lambda i: ST.run(i, 2, seed=FSEED)),
    ("(a) Strassen depth 1, general", True, lambda i: ST.run(i, 1, seed=FSEED)),
    ("(a) Strassen depth 2, general", True, lambda i: ST.run(i, 2, seed=FSEED)),
    ("(b) packed IMAD, two 16-bit fields", False, lambda i: P.fields(i, seed=FSEED)),
    ("(b) packed IMAD.WIDE polynomial, w 12", False, lambda i: P.poly(i, seed=FSEED)),
    ("(b) IDP4A, general", True, lambda i: P.idp4a(i, seed=FSEED)),
    ("(c) bit-sliced planes, LOP3 + POPC", False, lambda i: BS.planes(i, seed=FSEED)),
    ("(c) bit-sliced carry-save, LOP3", False, lambda i: BS.csa(i, seed=FSEED)),
    ("(d) Four Russians g 8, 16-bit, 8 columns a lane", False, lambda i: RU.run(i, 8, 2, 8, seed=FSEED)),
    ("(d) Four Russians g 8, 32-bit, 4 columns a lane", False, lambda i: RU.run(i, 8, 4, 4, seed=FSEED)),
    ("(d) Four Russians g 4, 16-bit, 8 columns a lane", False, lambda i: RU.run(i, 4, 2, 8, seed=FSEED)),
]


def word_table() -> list[Row]:
    zero, gen = instances()
    return [priced(name, gen if general else zero, run(gen if general else zero)) for name, general, run in ROUTES]


# ---- (e) the folds -----------------------------------------------------------------------------------------------

FOLDS: list[tuple[str, str, dict]] = [
    ("LI, cp-seg4 (reference)", "li", {"cp": "redux"}),
    ("LI, cp-red", "li", {"cp": "redg"}),
    ("LI, cp-red2", "li", {"cp": "redg2"}),
    ("LI, SHFL.BFLY checkpoint", "li", {"cp": "shfl"}),
    ("LI, key XOR on the IMAD pipe, cp-seg4", "li", {"cp": "redux", "absorb": F.li_kimad_absorb}),
    ("duo, LEA.HI (reference dc90-l32)", "duo", {"form": lambda q: "leahi"}),
    ("duo, IMAD.HI + IMAD", "duo", {"form": lambda q: "imad"}),
    ("duo, SHF.L.W + IADD3", "duo", {"form": lambda q: "shf"}),
    ("duo, IMAD.WIDE + IADD3", "duo", {"form": lambda q: "wide"}),
    ("duo, LEA.HI, cp-red2", "duo", {"form": lambda q: "leahi", "cp": "redg2"}),
    ("duo, IMAD.HI + IMAD, cp-red2", "duo", {"form": lambda q: "imad", "cp": "redg2"}),
]


@dataclass
class FoldRow:
    name: str
    fold: str
    load: dict[str, float]
    bare: dict[str, float]
    bare_hi: dict[str, float]
    fold_instr: float
    cp_instr: float
    exact: bool


def fold_row(name: str, fold: str, kw: dict, inst: D.Instance | None = None) -> FoldRow:
    """The fold on the reference's words, with the reference's tensor path; instructions are lane instructions a word.
    ``bare`` is the load without the checkpoint, as the duo designs price their fold, and ``bare_hi`` the same with
    IMAD.HI at its measured H200 price (`price.IMAD_HI_H200`)."""
    inst = inst or instances()[0]
    k = Kernel()
    W = REF.tensor_words(k, inst)
    lanes = W[0].size // D.WPL
    words = inst.ks * inst.n * inst.n
    out = REF.receipt(k, W, fold, FSEED, **kw)
    exact = bool(np.array_equal(out, D.receipt(W, fold, FSEED)))
    kc = Kernel()
    st = np.array(D.start(fold, lanes, FSEED), dtype=np.uint32)
    F.checkpoint(kc, st, kw.get("cp", "redux"))
    cp = sum(v for c, v in kc.by_phase("word").items() if c not in PR.UNCOSTED) / (D.WPL * lanes)
    total = sum(v for c, v in k.by_phase("word").items() if not c.startswith(("MMA", "SMEM", "GMEM"))) / words
    load = PR.loads(k.by_phase("word"), words)
    hi = PR.loads(k.by_phase("word"), words, PR.IMAD_HI_H200)
    cpl = PR.loads(kc.by_phase("word"), D.WPL * lanes)
    return FoldRow(name, fold, load, {r: load[r] - cpl[r] for r in load}, {r: hi[r] - cpl[r] for r in hi},
                   total - cp, cp, exact)


def fold_table() -> tuple[list[FoldRow], dict[str, tuple]]:
    """The fold rows, and the cheapest per-word mixes of duo's LEA.HI and IMAD forms (`price.mix2`), with the
    checkpoint (cp-red2) and without it, each checked exact at the nearest share in 32nds."""
    rows = [fold_row(*f) for f in FOLDS]
    by = {r.name: r for r in rows}
    mixes = {}
    a, b = by["duo, LEA.HI, cp-red2"], by["duo, IMAD.HI + IMAD, cp-red2"]
    x, t = PR.mix2(a.load, b.load)
    mixes["duo LEA.HI/IMAD, cp-red2"] = (x, t, fold_row(f"duo, IMAD form on {round(float(x) * 32)}/32, cp-red2", "duo",
                                                        {"form": F.mixed(float(x)), "cp": "redg2"}))
    x, t = PR.mix2(a.bare, b.bare)
    mixes["duo LEA.HI/IMAD, no checkpoint"] = (x, t, fold_row(f"duo, IMAD form on {round(float(x) * 32)}/32", "duo",
                                                              {"form": F.mixed(float(x))}))
    x, t = PR.mix2(a.bare_hi, b.bare_hi)
    mixes["duo LEA.HI/IMAD, no checkpoint, IMAD.HI at 4.18"] = (
        x, t, fold_row(f"duo, IMAD form on {round(float(x) * 32)}/32 (IMAD.HI 4.18)", "duo", {"form": F.mixed(float(x))}))
    return rows, mixes


FANIN_FLOOR = 0.5


def main() -> None:
    rows = word_table()
    ref = rows[0].load
    print(f"DS-90 a warp-word, n = {N}, {KS} segments; reference time {PR.time(ref)[0]:.4f} ({PR.time(ref)[1]}), "
          f"issue {ref['issue']:.4f}")
    print(f"{'route':52s} {'exact':5s} {'time':>7s} {'bind':>7s} {'issue':>7s} {'pooled':>7s} {'tensor':>7s} "
          f"{'setup':>7s}")
    for r in sorted(rows, key=lambda r: r.cols(ref)["time"]):
        c = r.cols(ref)
        print(f"{r.name:52s} {str(r.exact):5s} {c['time']:7.3f} {c['bind']:>7s} {c['issue']:7.3f} {c['pooled']:7.3f} "
              f"{c['tensor']:7.3f} {c['setup']:7.3f}")
    frows, mixes = fold_table()
    print(f"\n(e) folds: lane instructions a word (fold + checkpoint; fan-in floor {FANIN_FLOOR}), DS-90 time and issue "
          "a warp-word with the reference's tensor path")
    print(f"{'fold route':44s} {'exact':5s} {'fold':>6s} {'cp':>7s} {'time':>7s} {'bind':>6s} {'issue':>7s} "
          f"{'ALU':>6s} {'IMAD':>6s} {'no cp':>7s} {'hi4.18':>7s}")
    for r in frows + [m[2] for m in mixes.values()]:
        t, b = PR.time(r.load)
        print(f"{r.name:44s} {str(r.exact):5s} {r.fold_instr:6.3f} {r.cp_instr:7.4f} {t:7.4f} {b:>6s} "
              f"{r.load['issue']:7.4f} {r.load['ALU']:6.3f} {r.load['IMAD']:6.3f} {PR.time(r.bare)[0]:7.4f} "
              f"{PR.time(r.bare_hi)[0]:7.4f}")
    for name, (x, t, _) in mixes.items():
        print(f"mix {name}: the LP's share of the IMAD form {x} ({float(x):.4f}), time {t:.4f}")


if __name__ == "__main__":
    main()
