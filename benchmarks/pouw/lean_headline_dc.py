"""`lean_headline_dc`: the H200's and B200's headline instances in `lean_headline.py`'s Lean form, exact.

    python benchmarks/pouw/lean_headline_dc.py            print each instance's bound, its branches and their terms
    python benchmarks/pouw/lean_headline_dc.py --json F   the same as JSON

`lean_headline.py`'s bound (`Whole.headline_of_branches`) for the data-centre families, whose instances its
`instance()` refuses. The bound is the larger of the folded branch, at w* and φ* on π* with p = 7/8, and the unfolded
one, `1 − Y·(1 − 41/S)`, where `Y = (N_w w* + N_w φ* + χ*)/B_♭`. Three things differ from sm_120's and sm_80's instances:

- the H200's design runs a verified filler, ffill at p = 0 with pinned receipts. FILL-OPT prices it at χ*, the
  filler's load on π*, and the pinned check (`PinnedFillLuckAt`, (I7) at 0 under full checking) leaves it no luck
  term, so χ* enters Y whole. The B200's has none, and χ* is 0;
- π* is one facet that ♭'s bound sits on. The machine model's LP certificate may be mixed (the B200 base design on
  DS-100: DS_issue and DS_alu). Any one facet is a certificate, so where the LP's is mixed π* is ♭'s binding facet
  and the bound is at least the LP's: `1 − Y` is then at least ε_machine and ε_skip over ♭, not equal to them;
- neither device has a forms.toml, so a tensor class's shape is read from its census name: sm_90's wgmma as a warp's
  m16 slice of the m64 warpgroup instruction, sm_100's tcgen05 whole, and the mma.sync forms as m16n8. ω is the
  destination's 32-bit words (A6's count), and each class's MACs must give one of the sheet's peaks.

Each family is read on its data sheet (DS-90, DS-100) and on its measured vector (DC-90-ri, DC-100-ri), both at A6's
floors `q = q_T = ½`, `b_T = 0` (`SheetScalarRouteHalf` proves (I3) there on the sheet; on the measured rows it is the
same two-words-a-lane count). The checks are `lean_headline.instance()`'s: the sheet reproduces every DS coefficient,
♭ is `setup`'s floor program, the filler's included, and the bound is at most the family's target.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _lean_headline():
    spec = importlib.util.spec_from_file_location("pouw_lean_headline", HERE / "lean_headline.py")
    if spec.name not in sys.modules:
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
    return sys.modules[spec.name]


LH = _lean_headline()
E = LH.E
exact, branches, Q = LH.exact, LH.branches, LH.Q

from verity_machine_model.model import device  # noqa: E402

DEPTH = {"S8": 32, "U8S8": 32, "E4M3": 32, "E5M2": 32, "BF16": 16, "F16": 16, "TF32": 8, "B1": 256}
GMMA = re.compile(r"[HIBQ]GMMA_(S8|U8S8|E4M3|E5M2|BF16|F16|TF32|B1)_N(\d+)(?:_RS)?")
UTC = re.compile(r"UTC[HIQ]MMA_(?:(F16|E4M3)_)?M(\d+)_N(\d+)")
SYNC = re.compile(r"[HIB]MMA168(\d+)_(S8|F16|BF16|TF32|B1)")


@dataclass(frozen=True)
class Family:
    ns: str
    gpu: str
    key: str                             # the device directory
    design: str
    sheet: str                           # its data-sheet vector
    measured: str                        # its measured vector
    pipes: dict[str, tuple[str, int]]    # DS row: (Lean pipe, the Programming Guide's results an SM clock)
    peaks: tuple[int, ...]               # the sheet's dense MACs an SM clock, by operand type
    fmax: int
    fmax_src: str
    uniform: dict[str, int]              # the uniform writer kinds and ω
    eps: Fraction                        # the sheet's headline target
    eps_m: Fraction                      # the measured reading's

    @property
    def data(self) -> Path:
        return device(self.key)


SM90 = Family("Sm90", "H200", "sm_90", "dither-2s-licpw-redux-l32", "DS-90", "DC-90-ri",
              {"DS_alu": ("alu", 64), "DS_imad": ("imad", 64), "DS_shfl": ("shfl", 32)},
              (1024, 2048, 4096, 16384), 1980, "the data sheet's boost clock", {"UNI": 1},
              Fraction(1, 100), Fraction(1, 100))
SM100 = Family("Sm100", "B200", "sm_100", "dither-2s-licpw-redux-heavy-ldtm0-l32", "DS-100", "DC-100-ri",
               {"DS_alu": ("alu", 64), "DS_imad": ("imad", 64), "DS_shfl": ("shfl", 32)},
               (2048, 4096, 8192, 32768), 1965, "the driver's reported maximum (the data sheet gives no clock)", {},
               Fraction(1, 100), Fraction(1, 100))
FAMILIES = (SM90, SM100)


def shape(k: str, h: E.Honest) -> tuple[int, int] | None:
    """A tensor class's dense MACs a warp instruction and ω, its destination's 32-bit words; None for no MMA."""
    if k == h.form:
        return h.words * h.depth, h.words
    if m := GMMA.fullmatch(k):
        n = int(m.group(2))
        return 16 * n * DEPTH[m.group(1)], 16 * n
    if m := UTC.fullmatch(k):
        rows, n = int(m.group(2)), int(m.group(3))
        return rows * n * DEPTH[m.group(1) or "S8"], rows * n
    if m := SYNC.fullmatch(k):
        return 16 * 8 * int(m.group(1)), 16 * 8
    return None


@dataclass
class Reading:
    family: str
    vector: str
    design: str
    sheet: bool
    lp_certificate: dict[str, float]
    pistar: str
    wstar: str
    Nw: int
    checks: int
    eps_skip: float
    skip_class: str
    B: float                             # ♭'s bound a unit, its skip included, on π*
    w: float
    phi: float
    chi: float
    kappa: float
    one_minus_Y: float
    folded: float
    unfolded: float
    bound: float
    target: float
    machine_core: float                  # the machine model's core ε at q* = ½, for comparison
    machine_eps: float
    luck: float
    exact: dict


def _coefs(pi: E.Machine, classes: list[str]) -> dict[str, dict[str, Fraction]]:
    return {k: {r: exact(v) for r, v in pi.c[k].items() if v} for k in classes}


def _sheet_check(fam: Family, pi: E.Machine, c: dict, h: E.Honest) -> None:
    """The data-sheet machine (`Accounting.TS.Sheet.machine`) reproduces every DS coefficient."""
    for k, row in c.items():
        if row.get("DS_issue") != 1 or set(row) - set(pi.rows):
            raise SystemExit(f"{fam.sheet}: {k} is not 1 on DS_issue, or has a row outside {pi.rows}")
        if "DS_tensor" in row:
            s = shape(k, h)
            pk = 4 * Fraction(s[0]) / row["DS_tensor"] if s else None
            if pk is None or pk.denominator != 1 or pk not in fam.peaks:
                raise SystemExit(f"{fam.sheet}: {k}'s {s} MACs at {row['DS_tensor']} SMSP-cycles is no sheet peak")
        on = [r for r in fam.pipes if r in row]
        if len(on) > 1 or (on and row[on[0]] != Fraction(128, fam.pipes[on[0]][1])):
            raise SystemExit(f"{fam.sheet}: {k} is on pipes {on} at {row}, not one pipe at 128/width")


def reading(fam: Family, vector: str) -> Reading:
    B_ = LH._eps_budget()
    sheet = vector == fam.sheet
    pi = E.machine(vector, fam.data)
    d = E.design(fam.design, fam.data)
    h = E.honest(d.honest, fam.data)
    if (d.delta, d.p, d.checks) != (2.0 ** -41, float(LH.P_OPERAND), 48000):
        raise SystemExit(f"{fam.design}: the statement takes δ′ = 2^-41, operand luck 3/4 and 48000 checks")
    if any(f.p != 0 for f in d.filler):
        raise SystemExit(f"{fam.design}: a filler enters Y whole only at p = 0 (pinned receipts)")
    classes = sorted(pi.c)
    c = _coefs(pi, classes)
    if sheet:
        _sheet_check(fam, pi, c, h)
    load = lambda r, n: sum((c[k].get(r, Fraction(0)) * v for k, v in n.items()), Fraction(0))
    words = Fraction(h.words * h.depth, d.L)
    if words.denominator != 1:
        raise SystemExit(f"{fam.design}: {words} committed words an MMA is not whole")
    Nw = int(words)
    nw = {k: exact(v) for k, v in h.tensor("floor", d.floor).items()}
    nf = {k: exact(v) * Nw / E.LANES for k, v in d.fold.items()}
    nx: dict[str, Fraction] = {}
    for f in d.filler:
        tot = sum(exact(v) for v in f.mix.values())
        for k, v in f.mix.items():
            nx[k] = nx.get(k, Fraction(0)) + exact(f.per_word) * Nw * exact(v) / tot / E.LANES
    n0: dict[str, Fraction] = {}
    for part in (nw, nf, nx):
        for k, v in part.items():
            n0[k] = n0.get(k, Fraction(0)) + v
    s = E.setup(d, pi, fam.data, operands="free", credit="floor")
    if abs(s.baseline.T * Nw - float(max(load(r, n0) for r in pi.rows))) > 1e-6 or any(
            abs(s.baseline.n.get(k, 0.0) * Nw - float(n0.get(k, 0))) > 1e-6 for k in {*s.baseline.n, *n0}):
        raise SystemExit(f"{fam.design} on {vector}: ♭ is not setup's floor program ({s.baseline.n}, T "
                         f"{s.baseline.T})")
    e = E.eps(d, pi, float(Q), fam.data, operands="free", credit="floor")
    cert = {r: w for r, w in e.cheat.pi.items() if w > 1e-9}
    B0 = max(load(r, n0) for r in pi.rows)
    binds = [r for r in pi.rows if load(r, n0) == B0]
    if len(cert) == 1 and abs(next(iter(cert.values())) - 1) < 1e-9 and next(iter(cert)) in binds:
        pistar = next(iter(cert))
    else:
        pistar = binds[0]
    eps_skip = exact(B_.design_skip(d, data=fam.data, vector=vector).eps_time)
    lanes = [k for k in pi.cuda if c[k].get(pistar)]
    skip = min(lanes, key=lambda k: max(c[k].get(r, Fraction(0)) / c[k][pistar] for r in pi.rows))
    ns = {skip: eps_skip / (1 - eps_skip) * B0 / c[skip][pistar]}
    n = {k: n0.get(k, 0) + ns.get(k, 0) for k in {*n0, *ns}}
    B = load(pistar, n)
    if max(load(r, n) for r in pi.rows) != B:
        raise SystemExit(f"{fam.design} on {vector}: ♭'s bound with its skip ({skip}) is not on {pistar}")
    writers = [(k, shape(k, h)[1]) for k in classes if shape(k, h)]
    writers += [(k, om) for k, om in fam.uniform.items() if k in pi.c]
    demand = {k: c[k].get(pistar, Fraction(0)) / om for k, om in writers}
    vmin = min(c[k].get(pistar, Fraction(0)) for k in pi.cuda)
    vec = Q / E.LANES * vmin
    wk = min(demand, key=demand.get)
    w = min(demand[wk], vec)
    folds = [{k: exact(v) / E.LANES for k, v in d.fold.items()}]
    phi = min(load(pistar, f) for f in folds)
    chi = load(pistar, nx)
    Y = (Nw * w + Nw * phi + chi) / B
    lp = e.machine_eps * (1 - float(eps_skip)) + float(eps_skip)
    if float(1 - Y) < lp - 1e-9 or (pistar in cert and len(cert) == 1 and abs(float(1 - Y) - lp) > 1e-9):
        raise SystemExit(f"{fam.design} on {vector}: 1 − Y {float(1 - Y):.6%} at {pistar} is not ε_machine and ε_skip "
                         f"over ♭ ({lp:.6%}) or above them")
    folded = 1 - Y + Nw * w / B * 41 * LH.LN2 / (LH.LN_P * d.checks)
    unfolded = 1 - Y * (1 - Fraction(41, d.checks))
    bound = max(folded, unfolded)
    target = fam.eps if sheet else fam.eps_m
    if bound > target:
        raise SystemExit(f"{fam.design} on {vector}: the Lean bound {float(bound):.6%} is above {float(target):.4%}")
    fr = lambda x: f"{x.numerator}/{x.denominator}"
    return Reading(fam.ns, vector, fam.design, sheet, cert, pistar, wk if demand[wk] <= vec else "vector", Nw,
                   int(d.checks), float(eps_skip), skip, float(B), float(w), float(phi), float(chi),
                   float(Nw * w / B), float(1 - Y), float(folded), float(unfolded), float(bound), float(target),
                   e.core(), e.machine_eps, e.luck,
                   {"B": fr(B), "w": fr(w), "phi": fr(phi), "chi": fr(chi), "bound": fr(bound)})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="write every reading here as JSON")
    a = ap.parse_args(argv)
    out = []
    for fam in FAMILIES:
        for v in (fam.sheet, fam.measured):
            r = reading(fam, v)
            out.append(r)
            cert = " + ".join(f"{k} {x:.3f}" for k, x in r.lp_certificate.items())
            print(f"{r.design} on {v}: π* {r.pistar} (LP: {cert}), w* {r.wstar}, ♭'s bound {r.B:.4f} a unit, ε_skip "
                  f"{r.eps_skip:.4%} (skip as {r.skip_class}), κ {r.kappa:.6f}, χ* {r.chi:.4f}; 1 − Y "
                  f"{r.one_minus_Y:.4%}; folded {r.folded:.4%}, unfolded {r.unfolded:.4%}: the Lean bound "
                  f"{r.bound:.4%} ≤ {r.target:.2%} (machine model: core {r.machine_core:.4%}, ε_machine "
                  f"{r.machine_eps:.4%}, ε_luck {r.luck:.4%})")
    if a.json:
        Path(a.json).write_text(json.dumps([asdict(r) for r in out], indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
