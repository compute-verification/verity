"""`lean_headline`: the families' headline instances as exact Lean data, the files their headline guarantees read.

    python benchmarks/pouw/lean_headline.py           write Security/Definitions/Accounting/Headline/Sm*.lean
    python benchmarks/pouw/lean_headline.py --check   exit 1 when a committed file is not what it would write

A family's data-sheet instance (`Accounting.TS.Headline.SmNN`, read by `Accounting.Guarantees.TS.SmNNSheetHeadline`):
its device's data sheet (`DS-NN`) as `Accounting.TS.Sheet` data, its headline design as a TS model on it, the floor
program ♭ with ε_skip's skippable work, and π*, the certificate the machine model's LP finds at q* = ½. Every number is
exact: a census value or a TOML value read as its decimal, and eps_budget's ε_skip as the decimal Python prints.
`FAMILIES` lists them: sm_120 (H1 + fx1) and sm_80 (Dither-2S at L = 16).

`instance()` derives one and checks it against Python's own numbers: the sheet reproduces every DS coefficient, ♭ is
`setup`'s floor program at ω's words, π* is one facet, and the bound the Lean proof reads off the identity (w* and φ*
at π*, ln 2 < 0.6931471808, ln(4/3) ≥ 1/4) is at most the family's headline.

sm_120's measured instance (`Accounting.TS.Headline.Sm120Measured`, read by
`Accounting.Guarantees.TS.Sm120MeasuredHeadline`): the same design and ♭ on the measured vector (VF3r-16, its rows
sparse), the writer kinds at forms.toml's per-word count (`ω = m·n·k_dense/(L·split)`, rounded up), the scalar routes
at `q = q_T = 9/8`, ♭'s skippable work at VF3r-16's ε_skip, and π* at q* = 9/8. `measured()` checks it the same way,
against 0.540%.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from verity_machine_model import eps as E
from verity_machine_model.model import device

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "Security" / "Definitions" / "Accounting" / "Headline" / "Sm120.lean"
OUT_M = OUT.with_name("Sm120Measured.lean")

Q = Fraction(1, 2)                 # A6's floors: two words a lane, so q = q_T = ½ and b_T = 0 (SheetScalarRouteHalf)
LN2 = Fraction(6931471808, 10**10)  # Mathlib's Real.log_two_lt_d9
LN_P = Fraction(1, 4)              # 1 − 3/4 ≤ ln(4/3) (Real.one_sub_inv_le_log_of_pos)
TENSOR = re.compile(r"([HIBDQO])MMA(?:SP|SF)?(16|8)8(\d+)")   # a census MMA class: its type letter and m (n = 8), k
SPARSE = re.compile(r"[HIBDQO]MMASP")


@dataclass(frozen=True)
class Family:
    """A family's data-sheet instance: its device's data, its sheet's constants and its headline design."""
    ns: str                              # the Lean namespace's last part, and the file's stem
    gpu: str
    data: Path                           # the device's machine-model data
    device: str                          # its data-sheet vector
    design: str
    dvar: str                            # the design data's Lean name
    title: str                           # the design, in prose
    pipes: dict[str, tuple[str, int]]    # DS row: (Lean pipe, the Programming Guide's results an SM clock)
    peaks: tuple[int, ...]               # the sheet's dense MACs an SM clock, by operand type (rows.csv DS_tensor)
    fmax: int                            # MHz
    fmax_src: str
    uniform: dict[str, int]              # the uniform writer kinds and ω, their destination's 32-bit words
    bytes: str                           # the table level's byte class, priced 0 at b_T = 0
    skip: str                            # the class ♭'s skippable work is counted in, priced on π*'s facet
    eps: Fraction                        # the headline
    honest: str                          # a unit of honest work, in prose
    fold: str                            # the fold routes, in prose

    @property
    def key(self) -> str:
        return "sm_" + self.ns[2:]

    @property
    def out(self) -> Path:
        return OUT.with_name(f"{self.ns}.lean")


SM120 = Family(
    "Sm120", "RTX PRO 6000", device("sm_120"), "DS-120", "dither-c-peel-fx1", "h1fx1", "H1 + fx1",
    {"DS_imad": ("imad", 64), "DS_lop": ("lop", 64), "DS_xu": ("xu", 16), "DS_shfl": ("shfl", 32)},
    (256, 512, 1024, 2048), 2430, "SPEC's P1", {"UNI": 1, "UNI64": 2}, "LDG32", "NOROW", Fraction(117, 20000),
    "an IMMA.16832 at L = 32",
    "fx1's fold on lanes (ALU 2 + IMAD 2 lane instructions a word) and alone on the uniform datapath (4 a word)")
SM80 = Family(
    "Sm80", "A100", device("sm_80"), "DS-80", "dither-2s-l16", "d2s", "Dither-2S at L = 16",
    {"DS_alu": ("alu", 64), "DS_fma": ("fma", 64), "DS_xu": ("xu", 16), "DS_shfl": ("shfl", 32)},
    (64, 512, 1024, 2048, 4096, 16384), 1410, "the data sheet's boost clock", {}, "LDS32", "ALU",
    Fraction(147, 100000), "an IMMA.16816 at L = 16", "duo's fold on lanes (ALU 2 + IMAD 1 lane instructions a word)")
FAMILIES = (SM120, SM80)

DEVICE, DESIGN, FMAX = SM120.device, SM120.design, SM120.fmax
UNIFORM, SKIP, EPS = SM120.uniform, SM120.skip, SM120.eps
VECTOR = "VF3r-16"                 # the measured reading's vector
Q_M = Fraction(9, 8)               # its scalar floors: q*₀ is 1.120 (operands free), FLOORS's lane floor ≥ 9/8 at w ≤ 15
EPS_M = Fraction(27, 5000)         # the measured reading: 0.540%
OFF_MENU = ("MMA", "UNI64")        # census lines no writer kind is: the forms' fallback line, and UNI64 (on UP alone)
STAR = {"DS_issue": "issue", "DS_tensor": "tensor", "DS_smem": "smem"}


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
    fam: Family
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
    p, width = i.fam.pipes[row]
    return Fraction(128, width) if i.pipe[j] == p else Fraction(0)


def load(i: Instance, row: str, n: dict[str, Fraction]) -> Fraction:
    return sum((coef(i, row, k) * v for k, v in n.items()), Fraction(0))


def shape(k: str, fms: dict[str, E.Form], h: E.Honest) -> tuple[int, int] | None:
    """A tensor class's dense MACs a warp instruction (a 2:4 sparse form's dense equivalent) and ω, its destination's
    32-bit words (m·n; f64's two words an element): forms.toml's form, honest's for the generic MMA, else the census
    name's shape. None for a class that is no MMA."""
    if k in fms:
        f = fms[k]
        return f.words * round(f.depth * f.split), f.words
    if k == E.MMA:
        return h.words * h.depth, h.words
    m = TENSOR.fullmatch(re.sub(r"(F16|T|S4)$", "", k))
    if not m:
        return None
    t, rows, depth = m.group(1), int(m.group(2)), int(m.group(3))
    return rows * 8 * depth // (2 if SPARSE.match(k) else 1), rows * 8 * (2 if t == "D" else 1)


def instance(fam: Family = SM120) -> Instance:
    B_ = _eps_budget()
    pi = E.machine(fam.device, fam.data)
    d = E.design(fam.design, fam.data)
    h = E.honest(d.honest, fam.data)
    fms = {f.kind: f for f in E.forms(fam.data)}
    if (d.delta, d.p, d.checks) != (2.0 ** -41, 0.75, 48000):
        raise SystemExit(f"{fam.design}: the statement takes δ′ = 2^-41, p = 3/4 and 48000 checks; the design has "
                         f"{d.delta}, {d.p}, {d.checks}")
    if d.filler:
        raise SystemExit(f"{fam.design}: the instance has no filler route (FillOpt is proved for an empty menu)")
    pipes = {r: p for r, (p, _) in fam.pipes.items()}
    classes = sorted(pi.c)
    macs, peak, pipe, smem = [], [], [], []
    for k in classes:
        c = {r: exact(v) for r, v in pi.c[k].items() if v}
        if c.get("DS_issue") != 1 or set(c) - set(pi.rows):
            raise SystemExit(f"{fam.device}: {k} is not 1 on DS_issue, or has a row outside {pi.rows}")
        if "DS_tensor" in c:
            s = shape(k, fms, h)
            pk = 4 * Fraction(s[0]) / c["DS_tensor"] if s else None
            if pk is None or pk.denominator != 1 or pk not in fam.peaks:
                raise SystemExit(f"{fam.device}: {k}'s {s} MACs at {c['DS_tensor']} SMSP-cycles is no sheet peak")
            macs.append(s[0])
            peak.append(int(pk))
        else:
            macs.append(0)
            peak.append(0)
        on = [r for r in pipes if r in c]
        if len(on) > 1 or (on and c[on[0]] != Fraction(128, fam.pipes[on[0]][1])):
            raise SystemExit(f"{fam.device}: {k} is on pipes {on} at {c}, not one pipe at 128/width")
        pipe.append(pipes[on[0]] if on else None)
        smem.append(int(c["DS_smem"] * 32) if "DS_smem" in c else 0)
    words = h.words * h.depth / d.L
    if words != int(words):
        raise SystemExit(f"{fam.design}: {words} committed words an MMA is not whole")
    Nw = int(words)
    writers = [(k, shape(k, fms, h)[1]) for k in classes if k != E.MMA and shape(k, fms, h)]
    if E.MMA in pi.c:
        writers.append((E.MMA, shape(E.MMA, fms, h)[1]))
    writers += [(k, w) for k, w in fam.uniform.items() if k in pi.c]
    folds = [{k: exact(v) / E.LANES for k, v in d.fold.items()}]
    if d.uniform and E.UNI in pi.c:
        folds.append({E.UNI: exact(d.fold_cost())})
    nw = {k: exact(v) for k, v in h.tensor("floor", d.floor).items()}
    nf = {k: exact(v) * Nw / E.LANES for k, v in d.fold.items()}
    eps_skip = exact(B_.design_skip(d, data=fam.data, vector=fam.device).eps_time)
    i = Instance(fam, classes, macs, peak, pipe, smem, writers, list(pi.cuda), folds, nw, nf, {}, Nw, int(d.checks),
                 "", eps_skip, Fraction(0), Fraction(0), Fraction(0))
    for k in classes:
        for r in pi.rows:
            if coef(i, r, k) != exact(pi.c[k].get(r, 0.0)):
                raise SystemExit(f"{fam.device}: the sheet gives {k} {coef(i, r, k)} on {r}, the census {pi.c[k].get(r)}")
    e = E.eps(d, pi, float(Q), fam.data, operands="free", credit="floor")
    star = {r: w for r, w in e.cheat.pi.items() if w > 1e-9}
    if len(star) != 1 or abs(next(iter(star.values())) - 1) > 1e-9:
        raise SystemExit(f"{fam.design} on {fam.device} at q* = {Q}: π* is {star}, not one facet")
    pistar = next(iter(star))
    n0 = {k: nw.get(k, 0) + nf.get(k, 0) for k in {*nw, *nf}}
    B0 = load(i, pistar, n0)
    if max(load(i, r, n0) for r in pi.rows) != B0:
        raise SystemExit(f"{fam.design} on {fam.device}: ♭'s bound is not on π*'s facet {pistar}")
    ns = {fam.skip: eps_skip / (1 - eps_skip) * B0 / coef(i, pistar, fam.skip)}
    n = {k: n0.get(k, 0) + ns.get(k, 0) for k in {*n0, *ns}}
    B = load(i, pistar, n)
    if max(load(i, r, n) for r in pi.rows) != B:
        raise SystemExit(f"{fam.design} on {fam.device}: ♭'s bound with its skip is not on {pistar}")
    s = E.setup(d, pi, fam.data, operands="free", credit="floor")
    if abs(s.baseline.T * Nw - float(B0)) > 1e-9 or any(
            abs(s.baseline.n.get(k, 0.0) * Nw - float(n0.get(k, 0))) > 1e-9 for k in {*s.baseline.n, *n0}):
        raise SystemExit(f"{fam.design} on {fam.device}: ♭ is not setup's floor program ({s.baseline.n}, T "
                         f"{s.baseline.T})")
    vmin = min(coef(i, pistar, k) for k in i.vec)
    w = min([coef(i, pistar, k) / om for k, om in writers] + [Q / E.LANES * vmin])
    phi = min(load(i, pistar, f) for f in folds)
    kappa = Nw * w / B
    bound = 1 - (Nw * w + Nw * phi) / B + kappa * 41 * LN2 / (LN_P * i.checks)
    if bound > fam.eps:
        raise SystemExit(f"{fam.design} on {fam.device}: the Lean bound {float(bound):.6%} is above "
                         f"{float(fam.eps):.4%}")
    if abs(float(1 - (Nw * w + Nw * phi) / B) - (e.machine_eps * (1 - float(eps_skip)) + float(eps_skip))) > 1e-9:
        raise SystemExit(f"{fam.design} on {fam.device}: the terms at π* are not ε_machine and ε_skip over ♭ with its "
                         f"skip")
    return Instance(fam, classes, macs, peak, pipe, smem, writers, list(pi.cuda), folds, nw, nf, ns, Nw, i.checks,
                    pistar, eps_skip, B, kappa, bound)


@dataclass(frozen=True)
class Measured:
    rows: list[str]
    c: dict[str, dict[str, Fraction]]       # VF3r-16's coefficients, class by row
    writers: list[tuple[str, int]]
    ns: dict[str, Fraction]
    eps_skip: Fraction
    pistar: str
    wstar: str                              # the writer kind w* is at π*
    B: Fraction
    kappa: Fraction
    bound: Fraction
    checks: int


def mload(m: Measured, row: str, n: dict[str, Fraction]) -> Fraction:
    return sum((m.c[k].get(row, Fraction(0)) * v for k, v in n.items()), Fraction(0))


def measured(i: Instance | None = None) -> Measured:
    """sm_120's measured instance: H1 + fx1 and ♭ (`i`'s word and fold counts) on VF3r-16 at q = q_T = 9/8."""
    i = i or instance()
    B_ = _eps_budget()
    pi = E.machine(VECTOR)
    d = E.design(DESIGN)
    h = E.honest(d.honest)
    if sorted(pi.c) != i.classes or list(pi.cuda) != i.vec:
        raise SystemExit(f"{VECTOR}: its classes or CUDA lane classes are not {DEVICE}'s, which ♭'s counts index")
    c = {k: {r: exact(v) for r, v in pi.c[k].items() if v} for k in i.classes}
    writers = []
    for k in sorted(E.forms(), key=lambda f: f.kind):
        if k.kind in pi.c and k.kind not in OFF_MENU:
            writers.append((k.kind, math.ceil(Fraction(k.words) * Fraction(round(k.depth * k.split), k.split) / d.L)))
    writers += [(k, w) for k, w in UNIFORM.items() if k in pi.c and k not in OFF_MENU]
    eps_skip = exact(B_.design_skip(d, vector=VECTOR).eps_time)
    n0 = {k: i.nw.get(k, 0) + i.nf.get(k, 0) for k in {*i.nw, *i.nf}}
    e = E.eps(d, pi, float(Q_M), operands="free", credit="floor")
    star = {r: w for r, w in e.cheat.pi.items() if w > 1e-9}
    if len(star) != 1 or abs(next(iter(star.values())) - 1) > 1e-9:
        raise SystemExit(f"{DESIGN} on {VECTOR} at q* = {Q_M}: π* is {star}, not one facet")
    pistar = next(iter(star))
    m = Measured(list(pi.rows), c, writers, {}, eps_skip, pistar, "", Fraction(0), Fraction(0), Fraction(0),
                 i.checks)
    B0 = mload(m, pistar, n0)
    if max(mload(m, r, n0) for r in pi.rows) != B0:
        raise SystemExit(f"{DESIGN} on {VECTOR}: ♭'s bound is not on π*'s facet {pistar}")
    ns = {SKIP: eps_skip / (1 - eps_skip) * B0 / c[SKIP][pistar]}
    n = {k: n0.get(k, 0) + ns.get(k, 0) for k in {*n0, *ns}}
    B = mload(m, pistar, n)
    if max(mload(m, r, n) for r in pi.rows) != B:
        raise SystemExit(f"{DESIGN} on {VECTOR}: ♭'s bound with its skip is not on {pistar}")
    s = E.setup(d, pi, operands="free", credit="floor")
    if abs(s.baseline.T * i.Nw - float(B0)) > 1e-9:
        raise SystemExit(f"{DESIGN} on {VECTOR}: ♭ is not setup's floor program (T {s.baseline.T})")
    demand = {k: c[k].get(pistar, Fraction(0)) / om for k, om in writers}
    wk = min(demand, key=demand.get)
    vec = Q_M / E.LANES * min(c[k].get(pistar, Fraction(0)) for k in i.vec)
    if wk != h.form or demand[wk] > vec:
        raise SystemExit(f"{DESIGN} on {VECTOR}: w* at {pistar} is {wk} ({float(demand[wk])}), not honest's "
                         f"{h.form} below the vector route's {float(vec)}")
    w = demand[wk]
    phi = min(mload(m, pistar, f) for f in i.folds)
    kappa = i.Nw * w / B
    bound = 1 - (i.Nw * w + i.Nw * phi) / B + kappa * 41 * LN2 / (LN_P * i.checks)
    if bound >= EPS_M:
        raise SystemExit(f"{DESIGN} on {VECTOR}: the Lean bound {float(bound):.6%} is not below {float(EPS_M):.4%}")
    if abs(float(1 - (i.Nw * w + i.Nw * phi) / B) - (e.machine_eps * (1 - float(eps_skip)) + float(eps_skip))) > 1e-9:
        raise SystemExit(f"{DESIGN} on {VECTOR}: the terms at π* are not ε_machine and ε_skip over ♭ with its skip")
    return Measured(list(pi.rows), c, writers, ns, eps_skip, pistar, wk, B, kappa, bound, i.checks)


def exact_logs(i: Instance | Measured) -> float:
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


def _words(text: str, first: str, rest: str, width: int = 120) -> str:
    """Prose wrapped at `width`, its first line after `first`, the others after `rest`."""
    lines, cur = [], first
    for word in text.split():
        if len(cur) + 1 + len(word) > width and cur.strip():
            lines.append(cur)
            cur = rest + word
        else:
            cur = cur + word if cur in (first, rest) else cur + " " + word
    lines.append(cur)
    return "\n".join(lines)


def _sparse(i: Instance, n: dict[str, Fraction]) -> str:
    return "[" + ", ".join(f"({i.classes.index(k)}, {v.numerator}, {v.denominator})"
                           for k, v in sorted(n.items(), key=lambda kv: i.classes.index(kv[0]))) + "]"


def render(i: Instance) -> str:
    f = i.fam
    ix = i.classes.index
    star = STAR.get(i.pistar) or f"pipe .{f.pipes[i.pistar][0]}"
    lane = [p for p, _ in f.pipes.values()]
    pipes = [f"some .{p}" if p else "none" for p in i.pipe]
    writer_lines = "\n".join(f"    ({ix(k)}, {om}){',' if j + 1 < len(i.writers) else ''}  -- {k}"
                             for j, (k, om) in enumerate(i.writers))
    fold_lines = ",\n".join(f"    {_sparse(i, fl)}" for fl in i.folds)
    uni = [f"`{k}` ({w})" for k, w in f.uniform.items() if k in i.classes]
    f64 = ", f64's two words an element" if any(k.startswith("DMMA") for k, _ in i.writers) else ""
    menu = f"the census's MMA classes (ω = m·n{f64})" + (
        ", the generic MMA (128)" if E.MMA in i.classes else "") + (
        f" and the uniform {' and '.join(uni)}" if uni else "")
    bullets = [
        f"`classes`: the {len(i.classes)} instruction classes of the {f.device} census, in the order `Fin "
        f"{len(i.classes)}` indexes them.",
        f"`{f.device.replace('-', '').lower()}`, `sheet`: the {f.gpu}'s data sheet ({f.device}): f_max {f.fmax} MHz "
        f"({f.fmax_src}); each tensor class's dense MACs a warp instruction (`m·n·k`, a 2:4 sparse form's dense "
        f"equivalent) and its operand type's dense MACs an SM clock; the Nsight Compute pipes "
        f"{', '.join(p.upper() for p in lane[:-1])} and {lane[-1].upper()} with their Programming Guide widths; 128 B "
        f"for every shared-memory class. Its machine reproduces every {f.device} coefficient.",
        f"`{f.dvar}`, `model`: {f.title} (`{f.design}`) as a TS model on the sheet's machine: writer kinds {menu}; "
        f"the vector classes CUDA's lane classes; {f.fold}; A6's floors `q = q_T = ½`, `b_T = 0`; {i.Nw} committed "
        f"words a unit of honest work ({f.honest}).",
        f"`flat`: the floor program ♭ a unit: the MMA and the design's operand floor (word production), the fold, no "
        f"filler, and ε_skip's skippable work at {f.skip}'s price, {float(i.eps_skip):.6%} of ♭'s bound "
        f"(`eps_budget.design_skip` on {f.device}).",
        f"`piStar`: π*, the facet ({i.pistar}) the machine model's LP certificate sits on at q* = ½.",
        "`checks`: the checks a window, one a round (Theorem S's S).",
    ]
    doc = "\n".join(_words(b, "* ", "  ") for b in bullets)
    sv = f.device.replace("-", "").lower()
    mtype = f"Model (Fin {len(i.classes)}) (SheetRow Pipe) (Fin {len(i.writers)}) Unit Empty (Fin {len(i.folds)}) Unit"
    mhead = (f"noncomputable def model : {mtype} :=" if len(mtype) < 90 else
             f"noncomputable def model :\n    {mtype} :=")
    pipe_ctors = "\n".join(f"  | {p}" for p in lane)
    width_lines = "\n".join(f"  | .{p} => {w}" for p, w in f.pipes.values())
    return f'''import Definitions.Accounting.Headline.Encoding

/-!
# {f.key}'s headline instance: {f.device} and {f.title} (trusted; generated)

Written by `benchmarks/pouw/lean_headline.py` from the machine model's data (`benchmarks/pouw/machine_model/data`), and
checked against it by `benchmarks/pouw/tests/test_lean_headline.py`: edit the data and rerun the exporter, never this
file. `Accounting.Guarantees.TS.{f.ns}SheetHeadline` reads it.

{doc}
-/

namespace Accounting.TS.Headline.{f.ns}

open Accounting.TS

/-- The {f.device} census's instruction classes, in index order. -/
def classes : List String := {_list([f'"{k}"' for k in i.classes], "  ")}

/-- The Nsight Compute pipes the sheet rates (A4). -/
inductive Pipe
{pipe_ctors}
  deriving DecidableEq

instance : Fintype Pipe where
  elems := {{{", ".join(f"Pipe.{p}" for p in lane)}}}
  complete x := by cases x <;> simp

/-- Each pipe's Programming Guide results an SM clock. -/
def Pipe.width : Pipe → ℕ
{width_lines}

/-- The {f.gpu}'s data sheet, by class index. -/
def {sv} : SheetData Pipe where
  fmax := {f.fmax}
  macs := {_list([str(m) for m in i.macs], "    ")}
  peak := {_list([str(p) for p in i.peak], "    ")}
  pipe := {_list(pipes, "    ")}
  width := Pipe.width
  smem := {_list([str(s) for s in i.smem], "    ")}

/-- The data sheet over the census's classes. -/
noncomputable def sheet : Sheet (Fin {len(i.classes)}) Pipe := {sv}.sheet {len(i.classes)}

/-- {f.title}'s design data: writer kinds as (class, ω), vector classes, fold routes. -/
def {f.dvar} : DesignData where
  writers := [
{writer_lines}
  ]
  vec := [{", ".join(str(ix(k)) for k in i.vec)}]  -- {", ".join(i.vec)}
  folds := [
{fold_lines}]
  bytes := {ix(f.bytes)}  -- {f.bytes}
  q := {Q.numerator} / {Q.denominator}
  qT := {Q.numerator} / {Q.denominator}
  Nw := {i.Nw}

/-- {f.title}'s TS model on the data sheet's machine. -/
{mhead}
  {f.dvar}.model {len(i.classes)} {len(i.writers)} {len(i.folds)} sheet.machine

/-- ♭'s word production a unit: the MMA and the operand floor. -/
def flatWords : Sparse := {_sparse(i, i.nw)}

/-- ♭'s fold a unit: the lane fold for {i.Nw} words. -/
def flatFold : Sparse := {_sparse(i, i.nf)}

/-- ♭'s skippable work a unit, at {f.skip}'s price. -/
def flatSkip : Sparse := {_sparse(i, i.ns)}

/-- The floor program ♭, labelled. -/
noncomputable def flat : Labelled (Fin {len(i.classes)}) := labelled {len(i.classes)} flatWords flatFold [] flatSkip

/-- π*: the machine model's certificate at q* = ½. -/
def piStar : SheetRow Pipe := .{star}

/-- The checks a window. -/
def checks : ℕ := {i.checks}

end Accounting.TS.Headline.{f.ns}
'''


def render_measured(i: Instance, m: Measured) -> str:
    ix = i.classes.index
    rows = []
    for j, r in enumerate(m.rows):
        entries = [f"({ix(k)}, {m.c[k][r].numerator}, {m.c[k][r].denominator})" for k in i.classes if r in m.c[k]]
        body = _wrap(entries, "   ", 116)[3:] if entries else ""
        rows.append(f"  -- {j}: {r}\n  [{body}]{',' if j + 1 < len(m.rows) else ''}")
    writer_lines = "\n".join(f"    ({ix(k)}, {om}){',' if j + 1 < len(m.writers) else ''}  -- {k}"
                             for j, (k, om) in enumerate(m.writers))
    nt = len(m.writers)
    return f'''import Definitions.Accounting.Headline.Sm120

/-!
# sm_120's measured headline instance: {VECTOR} and H1 + fx1 (trusted; generated)

Written by `benchmarks/pouw/lean_headline.py` from the machine model's data (`benchmarks/pouw/machine_model/data`), and
checked against it by `benchmarks/pouw/tests/test_lean_headline.py`: edit the data and rerun the exporter, never this
file. `Accounting.Guarantees.TS.Sm120MeasuredHeadline` reads it.

* `rowNames`, `vf3r16`, `machine`: the measured vector {VECTOR}, {len(m.rows)} rows of SMSP-cycles a warp
  instruction at its epoch's census, each sparse over `Sm120.classes`; every facet SM-clocked, f_max {FMAX} MHz.
* `h1fx1`, `model`: H1 + fx1 (`{DESIGN}`) on that machine. Its writer kinds are the census's MMA forms, each at
  the committed words an instruction of it writes (forms.toml: `m·n·k_dense/(L·split)` at L = {E.design(DESIGN).L},
  rounded up), and the uniform `UNI` (1); `MMA` (the forms' fallback line) and `UNI64` ({VECTOR} prices it on the
  uniform pipe alone) are not on the menu. `Sm120.h1fx1`'s vector classes, fold routes and `N_w`; SCALAR-ROUTE's
  floors `q = q_T = {Q_M.numerator}/{Q_M.denominator}`, `b_T = 0`.
* `flatSkip`, `flat`: ♭ is `Sm120.flat`'s word production and fold, with ε_skip's skippable work at {VECTOR}'s
  {float(m.eps_skip):.6%} (`eps_budget.design_skip` on {VECTOR}), at {SKIP}'s price on π*.
* `piStar`, `wStarKind`: π*, the facet ({m.pistar}) the machine model's LP certificate sits on at
  q* = {Q_M.numerator}/{Q_M.denominator}, and the writer kind (honest's {m.wstar}) that is the cheapest word route there.
-/

namespace Accounting.TS.Headline.Sm120Measured

open Accounting.TS

/-- {VECTOR}'s rows, in index order. -/
def rowNames : List String := {_list([f'"{r}"' for r in m.rows], "  ")}

/-- {VECTOR}'s coefficients, row by row: `(class index, numerator, denominator)`. -/
def vf3r16 : List Sparse := [
{chr(10).join(rows)}]

/-- The measured machine: {VECTOR}'s rows over `Sm120.classes`. -/
noncomputable def machine : Machine (Fin {len(i.classes)}) (Fin {len(m.rows)}) :=
  rowsMachine {len(i.classes)} {len(m.rows)} vf3r16 {FMAX}

/-- H1 + fx1's design data on {VECTOR}: writer kinds as (class, ω), and `Sm120.h1fx1`'s routes. -/
def h1fx1 : DesignData where
  writers := [
{writer_lines}
  ]
  vec := Sm120.h1fx1.vec
  folds := Sm120.h1fx1.folds
  bytes := Sm120.h1fx1.bytes
  q := {Q_M.numerator} / {Q_M.denominator}
  qT := {Q_M.numerator} / {Q_M.denominator}
  Nw := Sm120.h1fx1.Nw

/-- H1 + fx1's TS model on the measured machine. -/
noncomputable def model : Model (Fin {len(i.classes)}) (Fin {len(m.rows)}) (Fin {nt}) Unit Empty (Fin {len(i.folds)}) Unit :=
  h1fx1.model {len(i.classes)} {nt} {len(i.folds)} machine

/-- ♭'s skippable work a unit on {VECTOR}, at its {SKIP} price. -/
def flatSkip : Sparse := {_sparse(i, m.ns)}

/-- The floor program ♭ on {VECTOR}, labelled. -/
noncomputable def flat : Labelled (Fin {len(i.classes)}) :=
  labelled {len(i.classes)} Sm120.flatWords Sm120.flatFold [] flatSkip

/-- π*: the machine model's certificate at q* = {Q_M.numerator}/{Q_M.denominator}, {m.pistar}. -/
def piStar : Fin {len(m.rows)} := {m.rows.index(m.pistar)}

/-- The writer kind of w* at π*: {m.wstar}. -/
def wStarKind : Fin {nt} := {[k for k, _ in m.writers].index(m.wstar)}

end Accounting.TS.Headline.Sm120Measured
'''


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 when a committed file differs")
    a = ap.parse_args(argv)
    out = {}
    for fam in FAMILIES:
        i = instance(fam)
        print(f"{fam.design} on {fam.device}: π* {i.pistar}, ♭'s bound {float(i.B):.6f} a unit, ε_skip "
              f"{float(i.eps_skip):.6%}; the Lean bound {float(i.bound):.5%} (exact logs {exact_logs(i):.5%}) ≤ "
              f"{float(fam.eps):.4%}")
        out[fam.out] = render(i)
        if fam is SM120:
            m = measured(i)
            print(f"{DESIGN} on {VECTOR}: π* {m.pistar}, w* {m.wstar}, ♭'s bound {float(m.B):.6f} a unit, "
                  f"ε_skip {float(m.eps_skip):.6%}; the Lean bound {float(m.bound):.5%} (exact logs "
                  f"{exact_logs(m):.5%}) < {float(EPS_M):.4%}")
            out[OUT_M] = render_measured(i, m)
    if a.check:
        stale = [p for p, text in out.items() if not p.is_file() or p.read_text() != text]
        for p in stale:
            print(f"{p.relative_to(ROOT)} is not what lean_headline.py writes: rerun it", file=sys.stderr)
        return 1 if stale else 0
    for p, text in out.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
