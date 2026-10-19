"""`lean_headline_dc`: the H200's and B200's headline instances in `lean_headline.py`'s Lean form, exact.

    python benchmarks/pouw/lean_headline_dc.py            write Security/Definitions/Accounting/HeadlineDc/Sm*.lean
    python benchmarks/pouw/lean_headline_dc.py --check    exit 1 when a committed file is not what it would write
    python benchmarks/pouw/lean_headline_dc.py --numbers  print each instance's bound, its branches and their terms
    python benchmarks/pouw/lean_headline_dc.py --json F   also write the readings as JSON

`lean_headline.py`'s bound (`Whole.headline_of_branches`) for the data-centre families, whose instances its
`instance()` refuses. The bound is the larger of the folded branch, at w* and φ* on π* with p = 7/8, and the unfolded
one, `1 − Y·(1 − 41/S)`, where `Y = (N_w w* + N_w φ* + χ*)/B_♭`. Three things differ from sm_120's and sm_80's instances:

- a design credits only work a check reads (RT-LOGIC-12): a filler at p = 0 (pinned receipts) would enter Y whole at
  χ*, its load on π*, and one at p = 1 enters ♭ not at all. The H200's ffill is at p = 1, since no check reads its
  chains, and the B200's fold is the plain LI-CP-W redux `PouwWarpCheck` recomputes, so χ* is 0 on every instance,
  the form `headline_of_branches` proves;
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
program (on a `flat_honest` vector, DC-90-ri, honest's program, a larger ♭ and so a larger bound), and the bound is at
most the family's target. A uniform writer heads 2 words (`UIMAD.WIDE.U32` writes a UR pair, RT-HOPPER-7).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
    flat_honest: tuple[str, ...] = ()    # vectors whose ♭ is honest's program, where `setup`'s floor mixes word routes
    bytes: str = "ALU"                   # the table level's byte class, priced 0 at b_T = 0 (neither census has a load)

    @property
    def data(self) -> Path:
        return device(self.key)


# DC-90-ri's joint rows price every tensor form but honest's IGMMA_S8_N256 at the scalar coefficient (rows.csv: "a
# lower bound on their issue cost"), and no certificate prices them otherwise (M gives a GMMA 32 SMSP-cycles to
# IGMMA's 62.7 on JI90_0), so its measured menu is the machine model's: honest's form, the rest off it (NoSum).
# The H200's design credits ffill at p = 1, since no check reads its chains (RT-LOGIC-12, decision 99). Without ffill
# in ♭, `setup`'s floor on DC-90-ri heads 0.3% of its words on honest's lane baseline (IMAD) beside the IGMMA, so ♭
# there is honest's program, the IGMMA heading every word: a larger B_♭ (199.31 a unit against the floor's 199.11), so
# the bound is at least the machine model's. It is over 1%.
# A uniform instruction heads up to 2 words (UIMAD.WIDE.U32 writes a UR pair, RT-HOPPER-7), so UNI's ω is 2.
SM90 = Family("Sm90", "H200", "sm_90", "dither-2s-licpw-redux-p1-l32", "DS-90", "DC-90-ri",
              {"DS_alu": ("alu", 64), "DS_imad": ("imad", 64), "DS_shfl": ("shfl", 32)},
              (1024, 2048, 4096, 16384), 1980, "the data sheet's boost clock", {"UNI": 2},
              Fraction(1, 100), Fraction(3, 100), honest_menu=True, flat_honest=("DC-90-ri",))
SM100 = Family("Sm100", "B200", "sm_100", "dither-2s-licpw-redux-ldtm0-l32", "DS-100", "DC-100-ri",
               {"DS_alu": ("alu", 64), "DS_imad": ("imad", 64), "DS_shfl": ("shfl", 32)},
               (2048, 4096, 8192, 32768), 1965, "the driver's reported maximum (the data sheet gives no clock)",
               {"UNI": 2}, Fraction(1, 100), Fraction(1, 100))
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
    rows: list[str]
    vec: list[str]
    sheet: dict[str, list]               # the data sheet by class (`_sheet_check`); empty on a measured vector
    writers: list[tuple[str, int]]
    folds: list[dict[str, Fraction]]
    nw: dict[str, Fraction]
    nf: dict[str, Fraction]
    nx: dict[str, Fraction]
    ns: dict[str, Fraction]
    skip: str
    lam: dict[str, Fraction]
    flat: str
    B: Fraction
    bound: Fraction
    Nw: int
    checks: int
    L: int
    form: str                            # honest's MMA form


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


def _sheet_check(fam: Family, pi: E.Machine, c: dict, h: E.Honest) -> dict[str, list]:
    """The data sheet by class (`macs`, `peak`, `pipe`, `smem`), checked to reproduce every DS coefficient on
    `Accounting.TS.Sheet.machine`'s rows: issue 1, tensor `macs/(peak/4)`, a pipe `128/width`, shared memory
    `smem/32`."""
    out: dict[str, list] = {"macs": [], "peak": [], "pipe": [], "smem": []}
    for k, row in c.items():
        if row.get("DS_issue") != 1 or set(row) - set(pi.rows) or set(row) - {"DS_issue", "DS_tensor", "DS_smem",
                                                                                 *fam.pipes}:
            raise SystemExit(f"{fam.sheet}: {k} is not 1 on DS_issue, or has a row outside {pi.rows}")
        macs = peak = 0
        if "DS_tensor" in row:
            s = shape(k, h)
            pk = 4 * Fraction(s[0]) / row["DS_tensor"] if s else None
            if pk is None or pk.denominator != 1 or pk not in fam.peaks:
                raise SystemExit(f"{fam.sheet}: {k}'s {s} MACs at {row['DS_tensor']} SMSP-cycles is no sheet peak")
            macs, peak = s[0], int(pk)
        on = [r for r in fam.pipes if r in row]
        if len(on) > 1 or (on and row[on[0]] != Fraction(128, fam.pipes[on[0]][1])):
            raise SystemExit(f"{fam.sheet}: {k} is on pipes {on} at {row}, not one pipe at 128/width")
        smem = row.get("DS_smem", Fraction(0)) * 32
        if smem.denominator != 1:
            raise SystemExit(f"{fam.sheet}: {k}'s DS_smem {row['DS_smem']} is no whole number of bytes over 32")
        out["macs"].append(macs)
        out["peak"].append(peak)
        out["pipe"].append(fam.pipes[on[0]][0] if on else None)
        out["smem"].append(int(smem))
    return out


def reading(fam: Family, vector: str, full: bool = False, floor_check: bool = True) -> Reading:
    """The family's reading on `vector`; `full` puts every census MMA class on a measured menu `honest_menu` limits.
    `floor_check=False` reads ♭ as honest's program even where `setup`'s floor program mixes word routes (a larger
    credit than the machine model's, so a bound at least the machine model's), for witnesses only."""
    B_ = LH._eps_budget()
    sheet = vector == fam.sheet
    pi = E.machine(vector, fam.data)
    d = E.design(fam.design, fam.data)
    h = E.honest(d.honest, fam.data)
    if (d.delta, d.p, d.checks) != (2.0 ** -41, float(LH.P_OPERAND), 48000):
        raise SystemExit(f"{fam.design}: the statement takes δ′ = 2^-41, operand luck 3/4 and 48000 checks")
    if any(f.p not in (0, 1) for f in d.filler):
        raise SystemExit(f"{fam.design}: a filler enters Y whole at p = 0 (pinned receipts) or not at all at p = 1")
    classes = sorted(pi.c)
    c = _coefs(pi, classes)
    lists = _sheet_check(fam, pi, c, h) if sheet else {}
    load = lambda r, n: sum((c[k].get(r, Fraction(0)) * v for k, v in n.items()), Fraction(0))
    per_mma = Fraction(h.words * h.depth, d.L)
    if per_mma.denominator != 1:
        raise SystemExit(f"{fam.design}: {per_mma} committed words an MMA is not whole")
    Nw = int(per_mma)
    nw = {k: exact(v) for k, v in h.tensor("floor", d.floor).items()}
    nf = {k: exact(v) * Nw / E.LANES for k, v in d.fold.items()}
    nx: dict[str, Fraction] = {}
    for f in (f for f in d.filler if f.p == 0):
        tot = sum(exact(v) for v in f.mix.values())
        for k, v in f.mix.items():
            nx[k] = nx.get(k, Fraction(0)) + exact(f.per_word) * Nw * exact(v) / tot / E.LANES
    n0: dict[str, Fraction] = {}
    for part in (nw, nf, nx):
        for k, v in part.items():
            n0[k] = n0.get(k, Fraction(0)) + v
    s = E.setup(d, pi, fam.data, operands="free", credit="floor")
    if floor_check and (abs(s.baseline.T * Nw - float(max(load(r, n0) for r in pi.rows))) > 1e-6 or any(
            abs(s.baseline.n.get(k, 0.0) * Nw - float(n0.get(k, 0))) > 1e-6 for k in {*s.baseline.n, *n0})):
        raise SystemExit(f"{fam.design} on {vector}: ♭ is not setup's floor program ({s.baseline.n}, T "
                         f"{s.baseline.T})")
    e = E.eps(d, pi, float(Q), fam.data, operands="free", credit="floor")
    B0 = max(load(r, n0) for r in pi.rows)
    flat = next(r for r in pi.rows if load(r, n0) == B0)
    # design_skip's float sums run in string-hash order, so its last bits vary with PYTHONHASHSEED: read it at 1e-12,
    # rounded up (a larger ε_skip only raises the bound), so every run writes the same data.
    eps_skip = Fraction(math.ceil(B_.design_skip(d, data=fam.data, vector=vector).eps_time * 10 ** 12), 10 ** 12)
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
    routes = {k: {r: c[k].get(r, Fraction(0)) / om for r in pi.rows} for k, om in writers}
    routes["vector"] = {r: Q / E.LANES * min(c[k].get(r, Fraction(0)) for k in pi.cuda) for r in pi.rows}
    fold_d = [{r: load(r, f) for r in pi.rows} for f in folds]
    fill_d = [{r: load(r, nx) for r in pi.rows}] if nx else []
    lam = certificate(_best(pi.rows, Nw, B, int(d.checks), routes, fold_d, fill_d))
    price = lambda dem: sum((x * dem[r] for r, x in lam.items()), Fraction(0))
    wk = min(routes, key=lambda k: price(routes[k]))
    w = price(routes[wk])
    phi = min(price(f) for f in fold_d)
    chi = min((price(f) for f in fill_d), default=Fraction(0))
    Y = (Nw * w + Nw * phi + chi) / B
    lp = e.machine_eps * (1 - float(eps_skip)) + float(eps_skip)
    if float(1 - Y) < lp - 1e-6:
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
                   _Exact(classes, c, list(pi.rows), list(pi.cuda), lists, writers, folds, nw, nf, nx, ns, skip, lam,
                          flat, B, bound, Nw, int(d.checks), d.L, h.form))


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


OUT = LH.ROOT / "Security" / "Definitions" / "Accounting" / "HeadlineDc"
SHEET_ROW = {"DS_issue": ".issue", "DS_tensor": ".tensor", "DS_smem": ".smem"}
SHAPES = {"Sm90": "sm_90's wgmma as one warp's m16 slice of its m64 warpgroup instruction (`16·n·k`), its mma.sync as "
                  "m16n8",
          "Sm100": "sm_100's tcgen05 MMA whole (`m·n·k`), its mma.sync as m16n8"}


def _sparse(x: _Exact, n: dict[str, Fraction]) -> str:
    return "[" + ", ".join(f"({x.classes.index(k)}, {v.numerator}, {v.denominator})"
                           for k, v in sorted(n.items(), key=lambda kv: x.classes.index(kv[0]))) + "]"


def _q(v: Fraction) -> str:
    return f"{v.numerator} / {v.denominator}"


def _uniform_words(fam: Family) -> str:
    return "".join(f" and the uniform `{k}` ({om}, a UR pair: RT-HOPPER-7)" for k, om in fam.uniform.items())


def _writer_lines(x: _Exact) -> str:
    return "\n".join(f"    ({x.classes.index(k)}, {om}){',' if j + 1 < len(x.writers) else ''}  -- {k}"
                     for j, (k, om) in enumerate(x.writers))


def _bullets(items: list[str]) -> str:
    return "\n".join(LH._words(b, "* ", "  ") for b in items)


def _header(fam: Family, what: str, imports: str, reads: str) -> str:
    return f'''import {imports}

/-!
# {fam.gpu} ({fam.key})'s {what} (trusted; generated)

Written by `benchmarks/pouw/lean_headline_dc.py` from the machine model's data (`benchmarks/pouw/machine_model/data`),
and checked against it by `benchmarks/pouw/tests/test_lean_headline_dc.py`: edit the data and rerun the exporter, never
this file. `Accounting.Guarantees.TS.{reads}` reads it.
'''


def _model_line(fam: Family, x: _Exact, dvar: str, machine: str, fill: str, rtype: str) -> str:
    nk, nt, nfo = len(x.classes), len(x.writers), len(x.folds)
    body = (f"{dvar}.modelFill {nk} {nt} {nfo} {machine} {fill}" if x.nx else f"{dvar}.model {nk} {nt} {nfo} {machine}")
    return (f"noncomputable def model : Model (Fin {nk}) {rtype} (Fin {nt}) Unit Empty (Fin {nfo}) Unit :=\n"
            f"  {body}")


def render_sheet(fam: Family, r: Reading) -> str:
    x = r.data
    ix = x.classes.index
    lane = [p for p, _ in fam.pipes.values()]
    sv = fam.sheet.replace("-", "").lower()
    nk = len(x.classes)
    filler = bool(x.nx)
    star = {SHEET_ROW.get(row) or f".pipe .{fam.pipes[row][0]}": v for row, v in x.lam.items()}
    lam_cases = "\n".join(f"  | {k} => {_q(v)}" for k, v in star.items())
    if len(star) < 3 + len(lane):
        lam_cases += "\n  | _ => 0"
    honest = f"an {x.form} at L = {x.L}"
    fill_doc = "the filler (ffill on lanes, `flatFill`), " if filler else "no filler, "
    bullets = [
        f"`classes`: the {nk} instruction classes of the {fam.sheet} census, in the order `Fin {nk}` indexes them.",
        f"`{sv}`, `sheet`: the {fam.gpu}'s data sheet ({fam.sheet}): f_max {fam.fmax} MHz ({fam.fmax_src}); each tensor "
        f"class's dense MACs a warp instruction ({SHAPES[fam.ns]}) and its operand type's dense MACs an SM clock; the "
        f"Nsight Compute pipes {', '.join(p.upper() for p in lane[:-1])} and {lane[-1].upper()} with their Programming "
        f"Guide widths; no shared-memory row ({fam.sheet} prices none). Its machine reproduces every {fam.sheet} "
        f"coefficient.",
        f"`d2s`, `model`: `{fam.design}` as a TS model on the sheet's machine: writer kinds every census MMA class "
        f"(ω = m·n, its destination's 32-bit words a warp)" + _uniform_words(fam)
        + f"; the vector classes CUDA's lane classes; the design's fold on lanes and alone on the uniform datapath; "
        f"the table level's bytes counted in {fam.bytes}, which `b_T = 0` prices at nothing (the census has no load "
        f"class); A6's floors `q = q_T = ½`, `b_T = 0`; {x.Nw} committed words a unit of honest work ({honest})"
        + ("; its filler menu the one route ffill runs on (sm_90's uniform datapath has no FFMA)." if filler else "."),
        f"`flat`: the floor program ♭ a unit: the MMA and the design's operand floor (word production), the fold, "
        f"{fill_doc}and ε_skip's skippable work at {x.skip}'s price, {r.eps_skip:.6%} of ♭'s bound "
        f"(`eps_budget.design_skip` on {fam.sheet}).",
        f"`lamStar`: π*, the certificate minimising the bound over this menu ({' + '.join(f'{k} {float(v):.6f}' for k, v in x.lam.items())}).",
        "`checks`: the checks a window, one a round (Theorem S's S).",
    ]
    pipes = [f"some .{p}" if p else "none" for p in x.sheet["pipe"]]
    fold_lines = ",\n".join(f"    {_sparse(x, f)}" for f in x.folds)
    pipe_ctors = "\n".join(f"  | {p}" for p in lane)
    width_lines = "\n".join(f"  | .{p} => {w}" for p, w in fam.pipes.values())
    return _header(fam, f"headline instance: {fam.sheet} and `{fam.design}`", "Definitions.Accounting.HeadlineDc.Encoding",
                   f"{fam.ns}SheetHeadline") + f'''
{_bullets(bullets)}
-/

namespace Accounting.TS.Headline.{fam.ns}

open Accounting.TS

/-- The {fam.sheet} census's instruction classes, in index order. -/
def classes : List String := {LH._list([f'"{k}"' for k in x.classes], "  ")}

/-- The Nsight Compute pipes the sheet rates (A4). -/
inductive Pipe
{pipe_ctors}
  deriving DecidableEq

instance instFintypePipe : Fintype Pipe where
  elems := {{{", ".join(f"Pipe.{p}" for p in lane)}}}
  complete x := by cases x <;> simp

/-- Each pipe's Programming Guide results an SM clock. -/
def Pipe.width : Pipe → ℕ
{width_lines}

/-- The {fam.gpu}'s data sheet, by class index. -/
def {sv} : SheetData Pipe where
  fmax := {fam.fmax}
  macs := {LH._list([str(m) for m in x.sheet["macs"]], "    ")}
  peak := {LH._list([str(p) for p in x.sheet["peak"]], "    ")}
  pipe := {LH._list(pipes, "    ")}
  width := Pipe.width
  smem := {LH._list([str(s) for s in x.sheet["smem"]], "    ")}

/-- The data sheet over the census's classes. -/
noncomputable def sheet : Sheet (Fin {nk}) Pipe := {sv}.sheet {nk}

/-- `{fam.design}`'s design data: writer kinds as (class, ω), vector classes, fold routes. -/
def d2s : DesignData where
  writers := [
{_writer_lines(x)}
  ]
  vec := [{", ".join(str(ix(k)) for k in x.vec)}]  -- {", ".join(x.vec)}
  folds := [
{fold_lines}]
  bytes := {ix(fam.bytes)}  -- {fam.bytes}
  q := {_q(Q)}
  qT := {_q(Q)}
  Nw := {x.Nw}

/-- ♭'s filler a unit{": ffill on lanes" if filler else ": none"}. -/
def flatFill : Sparse := {_sparse(x, x.nx)}

/-- `{fam.design}`'s TS model on the data sheet's machine. -/
{_model_line(fam, x, "d2s", "sheet.machine", "flatFill", "(SheetRow Pipe)")}

/-- ♭'s word production a unit: the MMA and the operand floor. -/
def flatWords : Sparse := {_sparse(x, x.nw)}

/-- ♭'s fold a unit: the lane fold for {x.Nw} words. -/
def flatFold : Sparse := {_sparse(x, x.nf)}

/-- ♭'s skippable work a unit, at {x.skip}'s price. -/
def flatSkip : Sparse := {_sparse(x, x.ns)}

/-- The floor program ♭, labelled. -/
noncomputable def flat : Labelled (Fin {nk}) := labelled {nk} flatWords flatFold flatFill flatSkip

/-- π*: the certificate, its weights exact and summing to 1. -/
noncomputable def lamStar : SheetRow Pipe → ℝ
{lam_cases}

/-- The checks a window. -/
def checks : ℕ := {x.checks}

end Accounting.TS.Headline.{fam.ns}
'''


def render_measured(fam: Family, r: Reading, s: Reading) -> str:
    x, xs = r.data, s.data
    if x.classes != xs.classes or x.vec != xs.vec or x.folds != xs.folds or (x.nw, x.nf, x.nx) != (xs.nw, xs.nf, xs.nx):
        raise SystemExit(f"{fam.measured}: its classes, vector classes or ♭'s parts are not {fam.sheet}'s, which the "
                         f"measured file reads")
    ix = x.classes.index
    nk, m = len(x.classes), len(x.rows)
    sv = fam.measured.replace("-", "").lower()
    rows = []
    for j, row in enumerate(x.rows):
        entries = [f"({ix(k)}, {x.c[k][row].numerator}, {x.c[k][row].denominator})" for k in x.classes if row in x.c[k]]
        body = LH._wrap(entries, "   ", 116)[3:] if entries else ""
        rows.append(f"  -- {j}: {row}\n  [{body}]{',' if j + 1 < m else ''}")
    menu = (f"honest's form{_uniform_words(fam)}: the census prices every other tensor form on the joint rows at "
            "the scalar coefficient, a lower bound on its issue cost, so they are off the menu and `NoSum` carries "
            "them" if r.menu == "honest" else "every census MMA class (ω = m·n)")
    lam = LH._wrap([f"({x.rows.index(k)}, {v.numerator}, {v.denominator})" for k, v in x.lam.items()],
                   " " * (len(f"  cert {m} ["))).lstrip()
    bullets = [
        f"`rowNames`, `{sv}`, `machine`: the measured vector {fam.measured}, {m} rows of SMSP-cycles a warp instruction "
        f"at its epoch's census, each sparse over `{fam.ns}.classes`; every facet SM-clocked, f_max {fam.fmax} MHz.",
        f"`d2s`, `model`: `{fam.design}` on that machine, its writer kinds {menu}; `{fam.ns}.d2s`'s vector classes, "
        f"fold routes, floors and `N_w`" + ("; ♭'s filler `" + fam.ns + ".flatFill` its filler route." if x.nx else "."),
        f"`flatSkip`, `flat`: ♭ is `{fam.ns}.flat`'s word production, fold and filler, with ε_skip's skippable work at "
        f"{fam.measured}'s {r.eps_skip:.6%} (`eps_budget.design_skip` on {fam.measured}), at {x.skip}'s price"
        + (f"; it is honest's program, not `setup`'s floor, which heads some words on honest's lane baseline beside "
           f"the MMA, so B_♭ is larger and the bound at least the machine model's." if fam.measured in fam.flat_honest
           else "."),
        f"`lamStar`: π*, the certificate minimising the bound over this menu "
        f"({' + '.join(f'{k} {float(v):.6f}' for k, v in x.lam.items())}).",
    ]
    nt, nfo = len(x.writers), len(x.folds)
    body = (f"d2s.modelFill {nk} {nt} {nfo} machine {fam.ns}.flatFill" if x.nx else f"d2s.model {nk} {nt} {nfo} machine")
    return _header(fam, f"measured headline instance: {fam.measured} and `{fam.design}`",
                   f"Definitions.Accounting.HeadlineDc.{fam.ns}", f"{fam.ns}MeasuredHeadline") + f'''
{_bullets(bullets)}
-/

namespace Accounting.TS.Headline.{fam.ns}Measured

open Accounting.TS

/-- {fam.measured}'s rows, in index order. -/
def rowNames : List String := {LH._list([f'"{row}"' for row in x.rows], "  ")}

/-- {fam.measured}'s coefficients, row by row: `(class index, numerator, denominator)`. -/
def {sv} : List Sparse := [
{chr(10).join(rows)}]

/-- The measured machine: {fam.measured}'s rows over `{fam.ns}.classes`. -/
noncomputable def machine : Machine (Fin {nk}) (Fin {m}) :=
  rowsMachine {nk} {m} {sv} {fam.fmax}

{LH._words(f"`{fam.design}`'s design data on {fam.measured}: writer kinds as (class, ω), and `{fam.ns}.d2s`'s "
           f"routes. -/", "/-- ", "")}
def d2s : DesignData where
  writers := [
{_writer_lines(x)}
  ]
  vec := {fam.ns}.d2s.vec
  folds := {fam.ns}.d2s.folds
  bytes := {fam.ns}.d2s.bytes
  q := {fam.ns}.d2s.q
  qT := {fam.ns}.d2s.qT
  Nw := {fam.ns}.d2s.Nw

/-- `{fam.design}`'s TS model on the measured machine. -/
noncomputable def model : Model (Fin {nk}) (Fin {m}) (Fin {nt}) Unit Empty (Fin {nfo}) Unit :=
  {body}

/-- ♭'s skippable work a unit on {fam.measured}, at its {x.skip} price. -/
def flatSkip : Sparse := {_sparse(x, x.ns)}

/-- The floor program ♭ on {fam.measured}, labelled. -/
noncomputable def flat : Labelled (Fin {nk}) :=
  labelled {nk} {fam.ns}.flatWords {fam.ns}.flatFold {fam.ns}.flatFill flatSkip

/-- π*: the certificate, its weights exact and summing to 1. -/
noncomputable def lamStar : Fin {m} → ℝ :=
  cert {m} [{lam}]

end Accounting.TS.Headline.{fam.ns}Measured
'''


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="write every reading here as JSON")
    ap.add_argument("--check", action="store_true", help="exit 1 when a committed Lean file is not what it would write")
    ap.add_argument("--numbers", action="store_true", help="print the readings only; write no Lean file")
    a = ap.parse_args(argv)
    out, record, files = [], [], {}
    for fam in FAMILIES:
        for v, full in ((fam.sheet, False), (fam.measured, False), *(((fam.measured, True),) if fam.honest_menu else ())):
            r = reading(fam, v, full, floor_check=v not in fam.flat_honest)
            (record if full else out).append(r)
            if not full:
                files[OUT / (fam.ns + ("" if r.sheet else "Measured") + ".lean")] = (
                    render_sheet(fam, r) if r.sheet else render_measured(fam, r, out[-2]))
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
    above = any(r.bound > r.target for r in out)
    if a.numbers:
        return 1 if above else 0
    if a.check:
        stale = [p for p, text in files.items() if not p.is_file() or p.read_text() != text]
        for p in stale:
            print(f"{p.relative_to(LH.ROOT)} is not what lean_headline_dc.py writes: rerun it", file=sys.stderr)
        return 1 if stale or above else 0
    if above:
        return 1
    for p, text in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f"wrote {p.relative_to(LH.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
