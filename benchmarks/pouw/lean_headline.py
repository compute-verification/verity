"""`lean_headline`: a family's headline instance as exact Lean data, the file its headline guarantee reads.

    python benchmarks/pouw/lean_headline.py           write Security/Definitions/Accounting/Headline/Sm120.lean
    python benchmarks/pouw/lean_headline.py --check   exit 1 when the committed file is not what it would write

sm_120's instance (`Accounting.TS.Headline.Sm120`, read by `Accounting.Guarantees.TS.Sm120SheetHeadline`): the RTX PRO
6000's data sheet (DS-120) as `Accounting.TS.Sheet` data, H1 + fx1 (`dither-c-peel-fx1`) as a TS model on it, its floor
program ♭ with ε_skip's skippable work, and π*, the certificate the machine model's LP finds at q* = ½. Every number is
exact: a census value or a TOML value read as its decimal, and eps_budget's ε_skip as the decimal Python prints.

`instance()` derives it and checks it against Python's own numbers: the sheet reproduces every DS-120 coefficient, ♭ is
`setup`'s floor program at ω's words, π* is one facet, and the bound the Lean proof reads off the identity (w* and φ*
at π*, ln 2 < 0.6931471808, ln(4/3) ≥ 1/4) is below the headline's 0.585%.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from verity_machine_model import eps as E

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "Security" / "Definitions" / "Accounting" / "Headline" / "Sm120.lean"

DEVICE = "DS-120"
DESIGN = "dither-c-peel-fx1"
Q = Fraction(1, 2)                 # A6's floors: two words a lane, so q = q_T = ½ and b_T = 0 (SheetScalarRouteHalf)
PIPES = {"DS_imad": "imad", "DS_lop": "lop", "DS_xu": "xu", "DS_shfl": "shfl"}
WIDTH = {"imad": 64, "lop": 64, "xu": 16, "shfl": 32}   # results an SM clock (rows.csv's DS rows, the Programming Guide)
PEAKS = (256, 512, 1024, 2048)     # dense MACs an SM clock: tf32, f16/bf16, s8 and e4m3, e2m1 (rows.csv DS_tensor)
FMAX = 2430                        # MHz, SPEC's P1 (designs.toml [clock])
UNIFORM = {"UNI": 1, "UNI64": 2}   # the uniform writer kinds and ω, their destination's 32-bit words
BYTES = "LDG32"                    # the table level's byte class, priced 0 at b_T = 0
SKIP = "NOROW"                     # the class ♭'s skippable work is counted in: issue only, as eps_budget prices it
EPS = Fraction(117, 20000)         # the headline: 0.585%
LN2 = Fraction(6931471808, 10**10)  # Mathlib's Real.log_two_lt_d9
LN_P = Fraction(1, 4)              # 1 − 3/4 ≤ ln(4/3) (Real.one_sub_inv_le_log_of_pos)


def _eps_budget():
    spec = importlib.util.spec_from_file_location("pouw_eps_budget", HERE / "eps_budget.py")
    m = sys.modules.setdefault(spec.name, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(m)
    return m


def exact(x: float) -> Fraction:
    """A float as the decimal Python prints for it."""
    return Fraction(repr(x))


@dataclass(frozen=True)
class Instance:
    classes: list[str]
    macs: list[int]
    peak: list[int]
    pipe: list[str | None]
    smem: list[int]
    writers: list[tuple[str, int]]
    vec: list[str]
    folds: list[dict[str, Fraction]]
    nw: dict[str, Fraction]
    nf: dict[str, Fraction]
    ns: dict[str, Fraction]
    Nw: int
    checks: int
    pistar: str
    eps_skip: Fraction
    B: Fraction
    kappa: Fraction
    bound: Fraction


def coef(i: Instance, row: str, k: str) -> Fraction:
    """The data-sheet machine's coefficient (`Accounting.TS.Sheet.machine`) on a DS row."""
    j = i.classes.index(k)
    if row == "DS_issue":
        return Fraction(1)
    if row == "DS_tensor":
        return Fraction(i.macs[j]) / (Fraction(i.peak[j]) / 4) if i.peak[j] else Fraction(0)
    if row == "DS_smem":
        return Fraction(i.smem[j], 32)
    p = PIPES[row]
    return Fraction(128, WIDTH[p]) if i.pipe[j] == p else Fraction(0)


def load(i: Instance, row: str, n: dict[str, Fraction]) -> Fraction:
    return sum((coef(i, row, k) * v for k, v in n.items()), Fraction(0))


def instance() -> Instance:
    B_ = _eps_budget()
    pi = E.machine(DEVICE)
    d = E.design(DESIGN)
    h = E.honest(d.honest)
    fms = {f.kind: f for f in E.forms()}
    if (d.delta, d.p, d.checks) != (2.0 ** -41, 0.75, 48000):
        raise SystemExit(f"{DESIGN}: the statement takes δ′ = 2^-41, p = 3/4 and 48000 checks; the design has "
                         f"{d.delta}, {d.p}, {d.checks}")
    if d.filler:
        raise SystemExit(f"{DESIGN}: the instance has no filler route (FillOpt is proved for an empty menu)")
    classes = sorted(pi.c)
    macs, peak, pipe, smem = [], [], [], []
    for k in classes:
        c = {r: exact(v) for r, v in pi.c[k].items() if v}
        if c.get("DS_issue") != 1 or set(c) - set(pi.rows):
            raise SystemExit(f"{DEVICE}: {k} is not 1 on DS_issue, or has a row outside {pi.rows}")
        if "DS_tensor" in c:
            f = fms.get(k)
            m = Fraction(f.words * round(f.depth * f.split) if f else h.words * h.depth)
            pk = 4 * m / c["DS_tensor"]
            if m.denominator != 1 or pk.denominator != 1 or pk not in PEAKS:
                raise SystemExit(f"{DEVICE}: {k}'s {m} MACs at {c['DS_tensor']} SMSP-cycles is no sheet peak")
            macs.append(int(m))
            peak.append(int(pk))
        else:
            macs.append(0)
            peak.append(0)
        on = [PIPES[r] for r in PIPES if r in c]
        if len(on) > 1 or (on and c[next(r for r in PIPES if r in c)] != Fraction(128, WIDTH[on[0]])):
            raise SystemExit(f"{DEVICE}: {k} is on pipes {on} at {c}, not one pipe at 128/width")
        pipe.append(on[0] if on else None)
        smem.append(int(c["DS_smem"] * 32) if "DS_smem" in c else 0)
    words = h.words * h.depth / d.L
    if words != int(words):
        raise SystemExit(f"{DESIGN}: {words} committed words an MMA is not whole")
    Nw = int(words)
    writers = [(k, fms[k].words) for k in sorted(fms) if k in pi.c]
    if E.MMA in pi.c and E.MMA not in fms:
        writers.append((E.MMA, h.words))
    writers += [(k, w) for k, w in UNIFORM.items() if k in pi.c]
    folds = [{k: exact(v) / E.LANES for k, v in d.fold.items()}]
    if d.uniform and E.UNI in pi.c:
        folds.append({E.UNI: exact(d.fold_cost())})
    nw = {k: exact(v) for k, v in h.tensor("floor", d.floor).items()}
    nf = {k: exact(v) * Nw / E.LANES for k, v in d.fold.items()}
    eps_skip = exact(B_.design_skip(d, vector=DEVICE).eps_time)
    i = Instance(classes, macs, peak, pipe, smem, writers, list(pi.cuda), folds, nw, nf, {}, Nw, int(d.checks), "",
                 eps_skip, Fraction(0), Fraction(0), Fraction(0))
    for k in classes:
        for r in pi.rows:
            if coef(i, r, k) != exact(pi.c[k].get(r, 0.0)):
                raise SystemExit(f"{DEVICE}: the sheet gives {k} {coef(i, r, k)} on {r}, the census {pi.c[k].get(r)}")
    B0 = load(i, "DS_issue", {k: nw.get(k, 0) + nf.get(k, 0) for k in {*nw, *nf}})
    ns = {SKIP: eps_skip / (1 - eps_skip) * B0}
    n = {k: nw.get(k, 0) + nf.get(k, 0) + ns.get(k, 0) for k in {*nw, *nf, *ns}}
    B = max(load(i, r, n) for r in pi.rows)
    if B != load(i, "DS_issue", n):
        raise SystemExit(f"{DESIGN} on {DEVICE}: ♭'s bound is not on DS_issue")
    s = E.setup(d, pi, operands="free", credit="floor")
    if abs(s.baseline.T * Nw - float(B0)) > 1e-9 or any(
            abs(s.baseline.n.get(k, 0.0) * Nw - float(nw.get(k, 0) + nf.get(k, 0))) > 1e-9 for k in {*s.baseline.n, *nw, *nf}):
        raise SystemExit(f"{DESIGN} on {DEVICE}: ♭ is not setup's floor program ({s.baseline.n}, T {s.baseline.T})")
    e = E.eps(d, pi, float(Q), operands="free", credit="floor")
    star = {r: w for r, w in e.cheat.pi.items() if w > 1e-9}
    if len(star) != 1 or abs(next(iter(star.values())) - 1) > 1e-9:
        raise SystemExit(f"{DESIGN} on {DEVICE} at q* = {Q}: π* is {star}, not one facet")
    pistar = next(iter(star))
    if pistar != "DS_issue":
        raise SystemExit(f"π* is {pistar}: the proof's w* and φ* are read at DS_issue")
    w = min(Fraction(1, om) for _, om in writers)
    w = min(w, Q / E.LANES)
    phi = min(load(i, pistar, f) for f in folds)
    kappa = Nw * w / B
    bound = 1 - (Nw * w + Nw * phi) / B + kappa * 41 * LN2 / (LN_P * i.checks)
    if bound >= EPS:
        raise SystemExit(f"{DESIGN} on {DEVICE}: the Lean bound {float(bound):.6%} is not below {float(EPS):.4%}")
    if abs(float(1 - (Nw * w + Nw * phi) / B) - (e.machine_eps * (1 - float(eps_skip)) + float(eps_skip))) > 1e-9:
        raise SystemExit(f"{DESIGN} on {DEVICE}: the terms at π* are not ε_machine and ε_skip over ♭ with its skip")
    return Instance(classes, macs, peak, pipe, smem, writers, list(pi.cuda), folds, nw, nf, ns, Nw, i.checks, pistar,
                    eps_skip, B, kappa, bound)


def exact_logs(i: Instance) -> float:
    """The bound with ln 2 and ln(4/3) exact, for the record."""
    return float(i.bound) - float(i.kappa * 41) * (float(LN2) / (float(LN_P) * i.checks)
                                                   - math.log(2) / (math.log(4 / 3) * i.checks))


def _wrap(items: list[str], indent: str, width: int = 120) -> str:
    lines, cur = [], indent
    for s in items:
        piece = s + ", "
        if len(cur) + len(piece.rstrip()) > width and cur.strip():
            lines.append(cur.rstrip())
            cur = indent
        cur += piece
    lines.append(cur.rstrip().rstrip(","))
    return "\n".join(lines)


def _list(items: list[str], indent: str) -> str:
    return "[\n" + _wrap(items, indent) + "]"


def _sparse(i: Instance, n: dict[str, Fraction]) -> str:
    return "[" + ", ".join(f"({i.classes.index(k)}, {v.numerator}, {v.denominator})"
                           for k, v in sorted(n.items(), key=lambda kv: i.classes.index(kv[0]))) + "]"


def render(i: Instance) -> str:
    ix = i.classes.index
    star = {"DS_issue": "issue", "DS_tensor": "tensor", "DS_smem": "smem"}[i.pistar]
    pipes = [f"some .{p}" if p else "none" for p in i.pipe]
    writer_lines = "\n".join(f"    ({ix(k)}, {om}){',' if j + 1 < len(i.writers) else ''}  -- {k}"
                             for j, (k, om) in enumerate(i.writers))
    fold_lines = ",\n".join(f"    {_sparse(i, f)}" for f in i.folds)
    return f'''import Definitions.Accounting.Headline.Encoding

/-!
# sm_120's headline instance: DS-120 and H1 + fx1 (trusted; generated)

Written by `benchmarks/pouw/lean_headline.py` from the machine model's data (`benchmarks/pouw/machine_model/data`), and
checked against it by `benchmarks/pouw/tests/test_lean_headline.py`: edit the data and rerun the exporter, never this
file. `Accounting.Guarantees.TS.Sm120SheetHeadline` reads it.

* `classes`: the {len(i.classes)} instruction classes of the {DEVICE} census, in the order `Fin {len(i.classes)}`
  indexes them.
* `ds120`, `sheet`: the RTX PRO 6000's data sheet ({DEVICE}): f_max {FMAX} MHz (SPEC's P1); each tensor class's dense
  MACs a warp instruction (`m·n·k`, a 2:4 sparse form's dense equivalent) and its operand type's dense MACs an SM clock;
  the Nsight Compute pipe of IMAD, LOP3, XU and SHFL with its Programming Guide width; 128 B for every
  shared-memory class. Its machine reproduces every {DEVICE} coefficient.
* `h1fx1`, `model`: H1 + fx1 (`{DESIGN}`) as a TS model on the sheet's machine: writer kinds the census's MMA
  forms (ω = m·n), the generic MMA (128) and the uniform `UNI` (1) and `UNI64` (2); the vector classes CUDA's lane
  classes; fx1's fold on lanes (ALU 2 + IMAD 2 lane instructions a word) and alone on the uniform datapath (4 a word);
  A6's floors `q = q_T = ½`, `b_T = 0`; {i.Nw} committed words a unit of honest work (an IMMA.16832 at L = 32).
* `flat`: the floor program ♭ a unit: the MMA and the design's operand floor (word production), the fold, no filler,
  and ε_skip's skippable work at its issue price, {float(i.eps_skip):.6%} of ♭'s bound (`eps_budget.design_skip`
  on {DEVICE}).
* `piStar`: π*, the facet the machine model's LP certificate sits on at q* = ½.
* `checks`: the checks a window, one a round (Theorem S's S).
-/

namespace Accounting.TS.Headline.Sm120

open Accounting.TS

/-- The {DEVICE} census's instruction classes, in index order. -/
def classes : List String := {_list([f'"{k}"' for k in i.classes], "  ")}

/-- The Nsight Compute pipes the sheet rates (A4). -/
inductive Pipe
  | imad
  | lop
  | xu
  | shfl
  deriving DecidableEq, Fintype

/-- Each pipe's Programming Guide results an SM clock. -/
def Pipe.width : Pipe → ℕ
  | .imad => {WIDTH["imad"]}
  | .lop => {WIDTH["lop"]}
  | .xu => {WIDTH["xu"]}
  | .shfl => {WIDTH["shfl"]}

/-- The RTX PRO 6000's data sheet, by class index. -/
def ds120 : SheetData Pipe where
  fmax := {FMAX}
  macs := {_list([str(m) for m in i.macs], "    ")}
  peak := {_list([str(p) for p in i.peak], "    ")}
  pipe := {_list(pipes, "    ")}
  width := Pipe.width
  smem := {_list([str(s) for s in i.smem], "    ")}

/-- The data sheet over the census's classes. -/
noncomputable def sheet : Sheet (Fin {len(i.classes)}) Pipe := ds120.sheet {len(i.classes)}

/-- H1 + fx1's design data: writer kinds as (class, ω), vector classes, fold routes. -/
def h1fx1 : DesignData where
  writers := [
{writer_lines}
  ]
  vec := [{", ".join(str(ix(k)) for k in i.vec)}]  -- {", ".join(i.vec)}
  folds := [
{fold_lines}]
  bytes := {ix(BYTES)}  -- {BYTES}
  q := {Q.numerator} / {Q.denominator}
  qT := {Q.numerator} / {Q.denominator}
  Nw := {i.Nw}

/-- H1 + fx1's TS model on the data sheet's machine. -/
noncomputable def model : Model (Fin {len(i.classes)}) (SheetRow Pipe) (Fin {len(i.writers)}) Unit Empty (Fin {len(i.folds)}) Unit :=
  h1fx1.model {len(i.classes)} {len(i.writers)} {len(i.folds)} sheet.machine

/-- ♭'s word production a unit: the MMA and the operand floor. -/
def flatWords : Sparse := {_sparse(i, i.nw)}

/-- ♭'s fold a unit: fx1 on lanes for {i.Nw} words. -/
def flatFold : Sparse := {_sparse(i, i.nf)}

/-- ♭'s skippable work a unit, at its issue price ({SKIP}). -/
def flatSkip : Sparse := {_sparse(i, i.ns)}

/-- The floor program ♭, labelled. -/
noncomputable def flat : Labelled (Fin {len(i.classes)}) := labelled {len(i.classes)} flatWords flatFold [] flatSkip

/-- π*: the machine model's certificate at q* = ½. -/
def piStar : SheetRow Pipe := .{star}

/-- The checks a window. -/
def checks : ℕ := {i.checks}

end Accounting.TS.Headline.Sm120
'''


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 when the committed file differs")
    a = ap.parse_args(argv)
    i = instance()
    text = render(i)
    print(f"{DESIGN} on {DEVICE}: π* {i.pistar}, ♭'s bound {float(i.B):.6f} a unit, ε_skip {float(i.eps_skip):.6%}; "
          f"the Lean bound {float(i.bound):.5%} (exact logs {exact_logs(i):.5%}) < {float(EPS):.4%}")
    if a.check:
        if not OUT.is_file() or OUT.read_text() != text:
            print(f"{OUT.relative_to(ROOT)} is not what lean_headline.py writes: rerun it", file=sys.stderr)
            return 1
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
