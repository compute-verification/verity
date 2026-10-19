"""`lean_headline_dc`: the H200's and B200's headline instances in `lean_headline.py`'s Lean form, exact.

    python benchmarks/pouw/lean_headline_dc.py            print each instance's bound, its branches and their terms
    python benchmarks/pouw/lean_headline_dc.py --json F   the same as JSON

`lean_headline.py`'s bound (`Whole.headline_of_branches`) for the data-centre families, whose instances its
`instance()` refuses. The bound is the larger of the folded branch, at w* and φ* on π* with p = 7/8, and the unfolded
one, `1 − Y·(1 − 41/S)`, where `Y = (N_w w* + N_w φ* + χ*)/B_♭`. Three things differ from sm_120's and sm_80's instances:

- the H200's design runs a verified filler, ffill at p = 0 with pinned receipts. FILL-OPT prices it at χ*, the
  filler's load on π*, and the pinned check (`PinnedFillLuckAt`, (I7) at 0 under full checking) leaves it no luck
  term, so χ* enters Y whole. The B200's has none, and χ* is 0;
- π* is the certificate that minimises the Lean bound over the Lean menu (`_best`, an LP over `λ`), made exact
  (`certificate`: weights `λ ≥ 0` summing to 1, `IsCert`), and may be mixed. Every price is π*'s: `w*`, `φ*` and
  `χ*` are the cheapest word route's, fold's and filler's prices at π*, each route's demand read facet by facet as
  `Accounting.TS.Model.demand` reads it (the vector route at the cheapest vector class on each facet), and `B_♭` is
  ♭'s bound, the facet it sits on. Any certificate gives a valid bound; at a mixed one `1 − Y` holds ε_cert, π*'s
  gap at ♭. The machine model's own certificate doesn't serve: its cheater folds a tensor word on lanes only, where
  the Lean menu prices the fold alone on any route (the uniform datapath's among them, which DS-100's ALU-heavy
  certificate leaves nearly free), so `1 − Y` is at least the LP's ε_machine and ε_skip over ♭, and more;
- neither device has a forms.toml, so a tensor class's shape is read from its census name: sm_90's wgmma as a warp's
  m16 slice of the m64 warpgroup instruction, sm_100's tcgen05 whole, and the mma.sync forms as m16n8. ω is the
  destination's 32-bit words (A6's count), and each class's MACs must give one of the sheet's peaks.

Each family is read on its data sheet (DS-90, DS-100) and on its measured vector (DC-90-ri, DC-100-ri), both at A6's
floors `q = q_T = ½`, `b_T = 0` (`SheetScalarRouteHalf` proves (I3) there from A6, on any machine). The menu's writer
kinds are every census MMA class, except on DC-90-ri: its joint rows price every tensor form but honest's at the scalar
coefficient, a lower bound on its issue cost, and no certificate prices them above it, so its menu is honest's form
and the uniform class, the machine model's, and NoSum carries the rest; the full menu's bound is printed for the
record. The checks are `lean_headline.instance()`'s: the sheet reproduces every DS coefficient, ♭ is `setup`'s floor
program, the filler's included, and the bound is at most the family's target.
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
    honest_menu: bool = False            # the measured reading's writer kinds: honest's form and the uniform ones only

    @property
    def data(self) -> Path:
        return device(self.key)


# DC-90-ri's joint rows price every tensor form but honest's IGMMA_S8_N256 at the scalar coefficient (rows.csv: "a
# lower bound on their issue cost"), and no certificate prices them otherwise (M gives a GMMA 32 SMSP-cycles to
# IGMMA's 62.7 on JI90_0), so its measured menu is the machine model's: honest's form, the rest off it (NoSum).
SM90 = Family("Sm90", "H200", "sm_90", "dither-2s-licpw-redux-l32", "DS-90", "DC-90-ri",
              {"DS_alu": ("alu", 64), "DS_imad": ("imad", 64), "DS_shfl": ("shfl", 32)},
              (1024, 2048, 4096, 16384), 1980, "the data sheet's boost clock", {"UNI": 1},
              Fraction(1, 100), Fraction(1, 100), honest_menu=True)
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


@dataclass(frozen=True)
class _Exact:
    """What the Lean data file of a reading reads: classes, coefficients, routes, ♭'s parts and the certificate."""
    classes: list[str]
    c: dict[str, dict[str, Fraction]]
    writers: list[tuple[str, int]]
    folds: list[dict[str, Fraction]]
    nw: dict[str, Fraction]
    nf: dict[str, Fraction]
    nx: dict[str, Fraction]
    ns: dict[str, Fraction]
    lam: dict[str, Fraction]
    flat: str
    B: Fraction
    bound: Fraction


@dataclass
class Reading:
    family: str
    vector: str
    design: str
    sheet: bool
    menu: str                            # the writer kinds: every census MMA class ("all") or honest's form
    certificate: dict[str, float]
    lam: dict[str, str]                  # π*: the certificate, exact
    flat_facet: str                      # the facet ♭'s bound sits on
    wstar: str
    Nw: int
    checks: int
    eps_skip: float
    skip_class: str
    B: float                             # ♭'s bound a unit, its skip included
    eps_cert: float                      # 1 − π*·n(♭)/B: 0 when π* is ♭'s facet
    w: float
    phi: float
    chi: float
    kappa: float
    one_minus_Y: float
    lp: float                            # the machine model's ε_machine and ε_skip over ♭, which 1 − Y is at least
    folded: float
    unfolded: float
    bound: float
    target: float
    machine_core: float                  # the machine model's core ε at q* = ½, for comparison
    machine_eps: float
    luck: float
    exact: dict
    data: _Exact


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


def reading(fam: Family, vector: str, full: bool = False) -> Reading:
    """The family's reading on `vector`; `full` puts every census MMA class on a measured menu `honest_menu` limits."""
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
    B0 = max(load(r, n0) for r in pi.rows)
    flat = next(r for r in pi.rows if load(r, n0) == B0)
    eps_skip = exact(B_.design_skip(d, data=fam.data, vector=vector).eps_time)
    lanes = [k for k in pi.cuda if c[k].get(flat)]
    skip = min(lanes, key=lambda k: max(c[k].get(r, Fraction(0)) / c[k][flat] for r in pi.rows))
    ns = {skip: eps_skip / (1 - eps_skip) * B0 / c[skip][flat]}
    n = {k: n0.get(k, 0) + ns.get(k, 0) for k in {*n0, *ns}}
    B = load(flat, n)
    if max(load(r, n) for r in pi.rows) != B:
        raise SystemExit(f"{fam.design} on {vector}: ♭'s bound with its skip ({skip}) is not on {flat}")
    menu = "honest" if fam.honest_menu and not sheet and not full else "all"
    writers = [(k, shape(k, h)[1]) for k in classes if shape(k, h) and (menu == "all" or k == h.form)]
    writers += [(k, om) for k, om in fam.uniform.items() if k in pi.c]
    folds = [{k: exact(v) / E.LANES for k, v in d.fold.items()}]
    if d.uniform and E.UNI in pi.c:
        folds.append({E.UNI: exact(d.fold_cost())})
    # Each route's demand on every facet, as `Accounting.TS.Model.demand` reads it: a writer kind 1/ω of its class, the
    # vector route (and the table route, at q_T = q, b_T = 0) q/32 of the cheapest vector class on that facet.
    words = {k: {r: c[k].get(r, Fraction(0)) / om for r in pi.rows} for k, om in writers}
    words["vector"] = {r: Q / E.LANES * min(c[k].get(r, Fraction(0)) for k in pi.cuda) for r in pi.rows}
    fold_d = [{r: load(r, f) for r in pi.rows} for f in folds]
    fill_d = [{r: load(r, nx) for r in pi.rows}] if nx else []
    lam = certificate(_best(pi.rows, Nw, B, int(d.checks), words, fold_d, fill_d))
    price = lambda dem: sum((x * dem[r] for r, x in lam.items()), Fraction(0))
    wk = min(words, key=lambda k: price(words[k]))
    w = price(words[wk])
    phi = min(price(f) for f in fold_d)
    chi = min((price(f) for f in fill_d), default=Fraction(0))
    Y = (Nw * w + Nw * phi + chi) / B
    lp = e.machine_eps * (1 - float(eps_skip)) + float(eps_skip)
    if float(1 - Y) < lp - 1e-6 and menu == "all":
        raise SystemExit(f"{fam.design} on {vector}: 1 − Y {float(1 - Y):.6%} at {lam} is below ε_machine and ε_skip "
                         f"over ♭ ({lp:.6%}): the menu here misses a route the machine model prices")
    folded = 1 - Y + Nw * w / B * 41 * LH.LN2 / (LH.LN_P * d.checks)
    unfolded = 1 - Y * (1 - Fraction(41, d.checks))
    bound = max(folded, unfolded)
    target = fam.eps if sheet else fam.eps_m
    fr = lambda x: f"{x.numerator}/{x.denominator}"
    nprice = sum((x * load(r, n) for r, x in lam.items()), Fraction(0))
    return Reading(fam.ns, vector, fam.design, sheet, menu, {r: float(x) for r, x in lam.items()},
                   {r: fr(x) for r, x in lam.items()}, flat, wk, Nw, int(d.checks), float(eps_skip), skip, float(B),
                   float(1 - nprice / B), float(w), float(phi), float(chi), float(Nw * w / B), float(1 - Y), lp,
                   float(folded), float(unfolded), float(bound), float(target), e.core(), e.machine_eps, e.luck,
                   {"B": fr(B), "w": fr(w), "phi": fr(phi), "chi": fr(chi), "bound": fr(bound)},
                   _Exact(classes, c, writers, folds, nw, nf, nx, ns, lam, flat, B, bound))


def _best(rows: list[str], Nw: int, B: Fraction, checks: int, words: dict, folds: list, fills: list) -> dict[str, float]:
    """The certificate that minimises the Lean bound over this menu: min z over `λ ≥ 0`, `Σ λ = 1`, with
    `t_w ≤ λ·v_ρ` for every word route, `t_f ≤ λ·u_φ` for every fold route, `t_x ≤ λ·x_ψ` for the filler, and z at
    least both branches, `1 − Y + c·N_w t_w/B` and `1 − Y·(1 − 41/S)`, `Y = (N_w t_w + N_w t_f + t_x)/B`."""
    from scipy.optimize import linprog

    R = len(rows)
    iw, i_f, ix, iz = R, R + 1, R + 2, R + 3
    nv = R + 4
    cfold = float(Fraction(41) * LH.LN2 / (LH.LN_P * checks))
    b = float(B)
    A, ub = [], []
    for dem, i in [*((v, iw) for v in words.values()), *((v, i_f) for v in folds), *((v, ix) for v in fills)]:
        a = [0.0] * nv
        a[i] = 1.0
        for j, r in enumerate(rows):
            a[j] = -float(dem[r])
        A.append(a)
        ub.append(0.0)
    y = {iw: Nw / b, i_f: Nw / b, ix: 1 / b}
    folded = [0.0] * nv                                  # z ≥ 1 − Y + c·N_w t_w/B
    unfolded = [0.0] * nv                                # z ≥ 1 − Y·(1 − 41/S)
    folded[iz] = unfolded[iz] = -1.0
    for i, coef in y.items():
        folded[i] = -coef
        unfolded[i] = -coef * (1 - 41 / checks)
    folded[iw] += Nw * cfold / b
    A += [folded, unfolded]
    ub += [-1.0, -1.0]
    bounds = [(0, None)] * R + [(0, None), (0, None), (0, None if fills else 0), (None, None)]
    obj = [0.0] * nv
    obj[iz] = 1.0
    s = linprog(obj, A_ub=A, b_ub=ub, A_eq=[[1.0] * R + [0.0] * 4], b_eq=[1.0], bounds=bounds, method="highs")
    if s.status:
        raise SystemExit(f"the certificate LP failed ({s.message})")
    return {r: float(s.x[j]) for j, r in enumerate(rows) if s.x[j] > 1e-12}


def certificate(cert: dict[str, float], digits: int = 9) -> dict[str, Fraction]:
    """A certificate as exact weights `λ ≥ 0` with `Σ λ = 1` (`IsCert`): each rounded to `digits`, the largest taking
    the rest. Any certificate gives a valid bound, so rounding only moves it by about 10^-digits."""
    lam = {r: Fraction(round(w * 10 ** digits), 10 ** digits) for r, w in cert.items()}
    top = max(lam, key=lam.get)
    lam[top] += 1 - sum(lam.values())
    return {r: x for r, x in lam.items() if x > 0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="write every reading here as JSON")
    a = ap.parse_args(argv)
    out, record = [], []
    for fam in FAMILIES:
        for v, full in ((fam.sheet, False), (fam.measured, False), *(((fam.measured, True),) if fam.honest_menu else ())):
            r = reading(fam, v, full)
            (record if full else out).append(r)
            cert = " + ".join(f"{k} {x:.4f}" for k, x in r.certificate.items())
            print(f"{r.design} on {v} ({r.menu} menu{', for the record' if full else ''}): π* {cert}, ♭ on "
                  f"{r.flat_facet} at {r.B:.4f} a unit (ε_cert {r.eps_cert:.4%}), w* {r.wstar}, ε_skip "
                  f"{r.eps_skip:.4%} (skip as {r.skip_class}), κ {r.kappa:.6f}, χ* {r.chi:.4f}; 1 − Y "
                  f"{r.one_minus_Y:.4%} (the LP's {r.lp:.4%}); folded {r.folded:.4%}, unfolded {r.unfolded:.4%}: the "
                  f"Lean bound {r.bound:.4%} {'≤' if r.bound <= r.target else 'ABOVE'} {r.target:.2%} (machine "
                  f"model: core {r.machine_core:.4%}, ε_machine {r.machine_eps:.4%}, ε_luck {r.luck:.4%})")
    if a.json:
        rows = [{k: v for k, v in asdict(r).items() if k != "data"} for r in out + record]
        Path(a.json).write_text(json.dumps(rows, indent=1) + "\n")
    return 1 if any(r.bound > r.target for r in out) else 0


if __name__ == "__main__":
    raise SystemExit(main())
