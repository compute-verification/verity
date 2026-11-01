"""`lean_headline_dc`: the H200's and B200's headline instances in `lean_headline.py`'s Lean form, exact.

    python benchmarks/pouw/lean_headline_dc.py            write Security/Definitions/Accounting/HeadlineDc/Sm*.lean
    python benchmarks/pouw/lean_headline_dc.py --check    exit 1 when a committed file is not what it would write
    python benchmarks/pouw/lean_headline_dc.py --numbers  print each instance's bound, its branches and their terms
    python benchmarks/pouw/lean_headline_dc.py --json F   also write the readings as JSON

`lean_headline.py`'s bound (`Whole.headline_of_branches`) for the data-centre families, whose instances its
`instance()` refuses. The bound is the larger of the folded branch, at w* and φ* on π* with p = 7/8, and the unfolded
one, `1 − Y·(1 − 41/S)`, where `Y = (N_w w* + N_w φ* + χ*)/B_♭`. Three things differ from sm_120's and sm_80's instances:

- a design credits only work a check reads: a filler at p = 0 (pinned receipts) would enter Y whole at
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
- the sheet is `sheet.toml` read by `verity_machine_model.sheet` (`Security/Definitions/Accounting/Sheets/`
  states it): each tensor class's MACs and its type's peak are the sheet's integers, a pipe's width its results an SM
  clock (the uniform pipe's the census's measured UP line, a rational), and ω the sheet's `omega`, the destination's
  32-bit words (A6's count). An MMA the sheet rates on no peak (b1) takes its ω from its census name (`shape`: sm_90's
  wgmma as a warp's m16 slice of the m64 warpgroup instruction, sm_100's tcgen05 whole, the mma.sync forms as m16n8),
  and a class the sheet rates on nothing is its issue slot alone (A2).

Each family is read on its data sheet (DS-90, DS-100) and on its measured vector (DC-90-ri, DC-100-ri), both at A6's
floors `q = q_T = ½`, `b_T = 0` (`SheetScalarRouteHalf` proves (I3) there from A6, on any machine). The H200's measured
vector is also read under (I3) `ScalarRoute` at `q = q_T = 3/5` (`Family.twin`, `Sm90MeasuredI3.lean`), an assumed
premise (`accounting-ts-scalar-route`): a twin beside the proved ½ floor, not a replacement. The folded branch reads
ln(8/7) ≥ 1/8 (`LN_P`) on the sheets and DC-100-ri, and ≥ 2/15 (`LN_P_M`, the series' first term, as sm_120's measured
headline does) on DC-90-ri and its twin (`Family.ln_p_m`). A6's vector classes are CUDA's lane classes and both
catch-alls (`CATCH_ALLS`), since an instruction no class names counts in its reading's catch-all. The menu's writer
kinds are every MMA class the reading prices (on a measured vector, the forms it times: an untimed form is its measured
issue slot alone and off the menu, where NoSum carries its words), and on DC-90-ri fewer still: its joint rows price
every tensor form but honest's at the scalar coefficient, a lower bound on its issue cost, and no certificate prices
them above it, so its menu is honest's form and the uniform class, the machine model's, and NoSum carries the rest; the
full menu's bound is printed for the record. The checks are `lean_headline.instance()`'s: the sheet reproduces every DS
coefficient, ♭ is `setup`'s floor program, and the bound is at most the family's target. A uniform writer heads 2 words
(`UIMAD.WIDE.U32` writes a UR pair). The fold menu is the design's fold and its `fold_routes`, every bit-exact
checkpoint loop (CP_SEG4 and CP_SEG4U; CP_ATS, CP_ATS2, CP_ATS8, CP_RED, CP_RED2, CP_RED8 and CP_XT; CP_RED, CP_RED2 and
CP_RED8 through a generic address, ATOM.E, bit-exact on the H200 and the B200: every global and generic red, red2 and
red8 loop at 0 mismatches) on lanes, and alone on the uniform datapath at the least of their counts. That is the
cheater's menu: a design's `counted` routes, which no run has built, are on it too, and ♭ folds by the built ones alone
(`built_routes`), so ε and B_♭ read the same ♭ and the credit (1 − ε)·B_♭ is a program a run has built. Each reading
emits ε and B_♭ as one pair (`Reading.pair`), the statement's ε literal with the B_♭ it is a fraction of. Each reading
declares A2's catch-all (`CATCH_ALL`), as sm_120's sheet does: every warp instruction no class of the reading prices
counts there, at its issue slot, NOROW's DS_issue on a sheet and UNLISTED's measured issue line on DC-90-ri and
DC-100-ri, and on DC-90-ri also the JI90 lines every timed scalar class carries: LD.E and IMAD.WIDE, the instructions it
counts there, were timed beside the IGMMA. A class no run has timed is read at its issue slot alone (A2's reading), and
measured rows replace that once a run times it: on DC-90-ri ATOME32 (the generic-address red's ATOM.E), ATOMS32, REDG32,
LDS128 and REDUX carry the JI90 lines, while the stores (STS32, STS128, STG32, STG128) and ISETP are still the issue row
alone, as every memory and control class is on DC-100-ri. Charging an untimed class another class's rows (SHFL's, say)
asserts lines nobody measured, which inflates ♭'s credit and, where a cheater uses the class, can overstate the floor. ♭
folds by `setup`'s mix of them on each vector and writes by its split of the words between the tensor route and the
design's baseline program (`_flat_shares`), so a measured file states its own `flatFold` or `flatWords` where that mix
or split is not the sheet's.
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
MODEL = HERE / "machine_model"

# The readings are the machine model's data: an installed verity_machine_model from another checkout would give that
# tree's numbers without a word, so this tree's goes first and any other refuses.
sys.path.insert(0, str(MODEL))
import verity_machine_model  # noqa: E402

if Path(verity_machine_model.__file__).resolve().parents[1] != MODEL:
    raise ImportError(f"lean_headline_dc reads {MODEL}'s verity_machine_model, but {verity_machine_model.__file__} "
                      "was imported first: run it in a fresh interpreter, or put that directory first on sys.path")


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

from verity_machine_model import sheet as S  # noqa: E402
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
    fmax: int
    fmax_src: str
    uniform: dict[str, int]              # the uniform writer kinds and ω
    eps: Fraction                        # the sheet's headline target
    eps_m: Fraction                      # the measured reading's
    honest_menu: bool = False            # the measured reading's writer kinds: honest's form and the uniform ones only
    bytes: str = "ALU"                   # the table level's byte class, priced 0 at b_T = 0 (neither census has a load)
    ln_p_m: Fraction = LH.LN_P           # the measured readings' lower bound on ln(1/p) = ln(8/7)
    twin: Fraction | None = None         # (I3)'s q = q_T for the measured reading's twin, at the measured target

    @property
    def data(self) -> Path:
        return device(self.key)

    @property
    def sh(self) -> S.Sheet:
        """The device's data sheet (`sheet.toml`), which `Sheets/H200.lean` and `B200.lean` state too."""
        return S.sheet(self.data)

    @property
    def pipes(self) -> dict[str, tuple[str, Fraction]]:
        """Each DS pipe row: its Lean pipe and the sheet's width, results an SM clock (A4; the uniform datapath's the
        census's measured UP line, which no document states)."""
        return {S.row(q): (q, w.value) for q, w in self.sh.pipes.items()}


# DC-90-ri's joint rows price every tensor form but honest's IGMMA_S8_N256 at the scalar coefficient (rows.csv: "a
# lower bound on their issue cost"), and no certificate prices them otherwise (M gives a GMMA 32 SMSP-cycles to
# IGMMA's 62.7 on JI90_0), so its measured menu is the machine model's: honest's form, the rest off it (NoSum).
# The H200's design credits ffill at p = 1, since no check reads its chains. With UNLISTED,
# ATOME32 and REDUX on the JI90 lines, `setup`'s floor on DC-90-ri
# heads every word on the IGMMA and folds by the generic CP_RED2 and CP_RED8, both built: ♭ never folds by a counted
# route (`counted`), while the cheater's menu holds them all. A uniform instruction heads up to 2 words
# (UIMAD.WIDE.U32 writes a UR pair), so UNI's ω is 2.
# The measured reading and its twin read ln(8/7) ≥ 2/15 (`LN_P_M`), as sm_120's measured headline does. The twin is
# the same reading under (I3) `ScalarRoute` at q = q_T = 3/5 (`accounting-ts-scalar-route`, assumed), the least floor
# of denominator at most 5 at which the bound reaches its floor (from q ≈ 0.5884 up; at 10/17 it is above it): a twin
# beside the ½ floor A6 proves, not a replacement. Both are over 1%, so both are held to the measured target.
SM90 = Family("Sm90", "H200", "sm_90", "dither-2s-licpw-redux-p1-l32", "DS-90", "DC-90-ri", 1980,
              "EPS1 DC's measured f_max at default clocks; the sheet's TOPS are at 1830 MHz", {"UNI": 2},
              Fraction(1, 100), Fraction(3, 100), honest_menu=True, ln_p_m=LH.LN_P_M, twin=Fraction(3, 5))
SM100 = Family("Sm100", "B200", "sm_100", "dither-2s-licpw-redux-ldtm0-l32", "DS-100", "DC-100-ri", 1965,
               "the driver's reported maximum (the data sheet gives no clock)", {"UNI": 2}, Fraction(1, 100),
               Fraction(1, 100))
FAMILIES = (SM90, SM100)
# A2's catch-all by device on its sheet (DS_issue alone) and by vector on its measured reading (the measured issue row,
# with the joint-issue lines once a run times its instructions beside the tensor): NOROW is a class on the measured
# vectors, at SHFL's joint issue rows, so UNLISTED carries the measured one
CATCH_ALL = {"sm_90": "NOROW", "sm_100": "NOROW", "DC-90-ri": "UNLISTED", "DC-100-ri": "UNLISTED"}
# A6 counts a vector word's instructions in the vector classes, and an instruction no class names (IMAD.WIDE, whose low
# word does li's multiply) counts in the reading's catch-all, so both catch-alls are vector classes, as
# sm_120's 64-bit lane classes are (`VEC_64`). The vector classes are one list for a family's sheet and measured files.
CATCH_ALLS = frozenset(CATCH_ALL.values())
# A run that times the catch-all's instructions beside the tensor (LD.E and IMAD.WIDE on the H200)
# gives it the census's joint-issue lines, which every timed scalar class carries at one coefficient a row
SCALAR_REF, JOINT_ISSUE = "FFMA", re.compile(r"JI\d+_\d+")


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


def omega(fam: Family, k: str, h: E.Honest) -> int | None:
    """A writer class's ω, its destination's 32-bit words a warp instruction (A6): the sheet's (`sheet.omega`) for a
    tensor form, and for an MMA the sheet rates on no peak (b1, issue only) the count its census name gives (`shape`);
    None for no MMA."""
    om, named = S.omega(fam.sh).get(k), shape(k, h)
    if om is not None and named is not None and named[1] != om:
        raise SystemExit(f"{fam.key}: {k}'s ω is {om} on the sheet and {named[1]} by its census name")
    om = om if om is not None else (named[1] if named else None)
    if om is not None and Fraction(om).denominator != 1:
        raise SystemExit(f"{fam.key}: {k}'s ω {om} is no whole number of words")
    return None if om is None else int(om)


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
    fold_names: list[str]                # the design's `fold_routes`, `folds`' entries after its fold
    counted: list[str]                   # those no run has built: the cheater's menu holds them, ♭'s fold never does
    fold_mix: str                        # ♭'s fold in words (`_fold_mix`)
    words_mix: str                       # ♭'s word production in words (`_words_mix`)
    untimed: list[str]                   # the sheet's classes the vector has no line for, read at its issue slot alone
    q: Fraction                          # the vector and table routes' floors q = q_T (b_T = 0)
    ln_p: Fraction                       # the lower bound on ln(1/p) the folded branch reads


@dataclass
class Reading:
    family: str
    vector: str
    design: str
    sheet: bool
    menu: str                            # the writer kinds: every census MMA class ("all") or honest's form
    q: str                               # q = q_T: A6's ½, or (I3)'s on a twin
    ln_p: str                            # ln(8/7) ≥ this: 1/8, or 2/15 (the series' first term)
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
    statement: str                       # the guarantee this reading is (Models' HeadlineDc); "" for the record
    pair: dict                           # ε and the B_♭ it is a fraction of, one export (`_pair`)
    exact: dict
    data: _Exact


def _coefs(pi: E.Machine, classes: list[str]) -> dict[str, dict[str, Fraction]]:
    return {k: {r: exact(v) for r, v in pi.c[k].items() if v} for k in classes}


def _fold_at(d: E.Design, z: dict[str, Fraction]) -> dict[str, Fraction]:
    """The fold a word with each of the design's `fold_routes` folding its share z_j of the words."""
    n = {k: exact(v) for k, v in d.fold.items()}
    for j, x in z.items():
        r = d.fold_routes[j]
        for k in {*r, *d.fold}:
            n[k] = n.get(k, Fraction(0)) + x * (exact(r.get(k, 0.0)) - exact(d.fold.get(k, 0.0)))
    return {k: v for k, v in n.items() if v}


#: the key of ♭'s share of the words on the design's baseline program among its fold routes' shares (`_flat_shares`)
BASELINE = "baseline"
#: a statement's ε is the exact bound rounded up at 10^-5, its literal `1 - N / 100000`
EPS_DEN = 10 ** 5


def _pair(statement: str, bound: Fraction, B: Fraction, facet: str) -> dict:
    """ε and B_♭ as one export: the statement's ε (the bound rounded up at 10^-5), the B_♭ it is a fraction of (SMSP-
    cycles a unit, on `facet`), and the credit (1 − ε)·B_♭ they give together. ε from one export and B_♭ from another
    give no credit at all, so a consumer reads both from here."""
    n = math.ceil(bound * EPS_DEN)
    fr = lambda x: f"{x.numerator}/{x.denominator}"
    return {"statement": statement, "eps": f"{n}/{EPS_DEN}", "B_flat": fr(B), "flat_facet": facet,
            "credit": fr((1 - Fraction(n, EPS_DEN)) * B)}


def _flat_shares(d: E.Design, shares: dict[str, float], bound) -> dict[str, Fraction]:
    """♭'s shares (`setup`'s floor): of the words on each of the design's `fold_routes` (`Mix.folds`), the rest on its
    fold, and of the words written by its baseline program (`Mix.words`, at `BASELINE`), the rest on the tensor route.
    The LP's shares are floats, so each is read as the simplest fraction at which ♭'s bound (`bound` of the shares) is
    the float shares' to 1e-9: denominators to 1000 on a sheet's small coefficients, larger where a measured vector's
    facets tie at an irrational-looking share. Any shares fold and write every word, so they only keep ♭ at `setup`'s."""
    if BASELINE in d.fold_routes:
        raise SystemExit(f"{d.name}: a fold route is named {BASELINE!r}, the key of ♭'s words on its baseline program")
    want = bound({j: Fraction(x) for j, x in shares.items()})
    keeps = lambda z: bound({j: Fraction(x) for j, x in z.items()}) <= want * (1 + Fraction(1, 10 ** 9))
    # Where ♭'s facets don't tell two routes apart (DC-90-ri's REDG and ATOM.E reds: AT90 has slack), the LP splits the
    # words between them anyhow; each such route's words go to one that keeps ♭'s bound, so the shares are a vertex of
    # fewest routes, exact at small denominators.
    merged = True
    while merged:
        merged = False
        folds = sorted((j for j in shares if j != BASELINE), key=lambda j: (shares[j], j))
        for j in folds:
            for k in sorted((k for k in folds if k != j), key=lambda k: (-shares[k], k)):
                z = {i: x for i, x in shares.items() if i != j}
                z[k] += shares[j]
                if keeps(z):
                    shares, merged = z, True
                    break
            if merged:
                break
    for den in (*range(1, 1001), *(10 ** k for k in range(4, 16))):
        z = {j: Fraction(x).limit_denominator(den) for j, x in shares.items()}
        if sum(x for j, x in z.items() if j != BASELINE) <= 1 and z.get(BASELINE, 0) <= 1 and bound(z) <= want * (
                1 + Fraction(1, 10 ** 9)):
            return {j: x for j, x in z.items() if x}
    raise SystemExit(f"{d.name}: no shares with denominator at most 10^15 keep ♭'s bound at {shares}")


def _words_mix(d: E.Design, wb: Fraction) -> str:
    """♭'s word production in words: the tensor route, and the design's baseline program on its share of the words."""
    if not wb:
        return "the MMA and the operand floor"
    share = lambda x: f"{x.numerator}/{x.denominator}" if x.denominator <= 100 else f"{float(x):.6f}"
    return (f"the MMA and the operand floor on {share(1 - wb)} of the words, the design's baseline program "
            f"({d.baseline_name}) on {share(wb)}")


def _and(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _fold_mix(d: E.Design, z: dict[str, Fraction]) -> str:
    """♭'s fold in words: each route on its share of the words."""
    if not z:
        return "the design's fold"
    if list(z.values()) == [1]:
        return f"the fold routes' cheapest, {next(iter(z))} on every word"
    share = lambda x: f"{x.numerator}/{x.denominator}" if x.denominator <= 100 else f"{float(x):.4f}"
    rest = 1 - sum(z.values())
    parts = [f"{j} {share(x)}" for j, x in z.items()] + ([f"the design's {d.fold_name} {share(rest)}"] if rest else [])
    return "the fold routes' cheapest mix, by share of the words: " + ", ".join(parts)


def _sheet_check(fam: Family, pi: E.Machine, c: dict) -> dict[str, list]:
    """The data sheet by class (`macs`, `peak`, `pipe`, `smem`), each class's form, peak and pipe the device's
    sheet.toml, checked to reproduce every DS coefficient on `Accounting.TS.Sheet.machine`'s rows: issue 1, tensor
    `macs/(peak/4)`, a pipe `128/width`, shared memory `smem/32`. A class with no form and no pipe (b1, the memory,
    atomic and control classes, the catch-all) is its issue slot alone (A2)."""
    sh = fam.sh
    want = S.lines(sh)
    out: dict[str, list] = {"macs": [], "peak": [], "pipe": [], "smem": []}
    for k, row in c.items():
        if set(row) - set(pi.rows) or set(row) - {"DS_issue", "DS_tensor", "DS_smem", *fam.pipes}:
            raise SystemExit(f"{fam.sheet}: {k} has a row outside {pi.rows}")
        if k not in sh.kinds or row != want[k]:
            raise SystemExit(f"{fam.sheet}: {k}'s lines {row} aren't its sheet.toml kind's, {want.get(k)} "
                             f"(`machine-model --device {fam.key} sheet`)")
        f, macs, peak = sh.kinds[k].form, 0, 0
        if f:
            macs, peak = f.macs, sh.peaks[f.type].macs
            if macs.denominator != 1 or peak.denominator != 1:
                raise SystemExit(f"{fam.sheet}: {k}'s {macs} MACs or its {f.type} peak {peak} is not whole")
            macs, peak = int(macs), int(peak)
        on = [r for r in fam.pipes if r in row]
        smem = row.get("DS_smem", Fraction(0)) * 32
        if smem.denominator != 1:
            raise SystemExit(f"{fam.sheet}: {k}'s DS_smem {row['DS_smem']} is no whole number of bytes over 32")
        out["macs"].append(macs)
        out["peak"].append(peak)
        out["pipe"].append(fam.pipes[on[0]][0] if on else None)
        out["smem"].append(int(smem))
    k = CATCH_ALL.get(fam.key)
    if k is not None and c.get(k) != {"DS_issue": 1}:
        raise SystemExit(f"{fam.sheet}: the catch-all {k} is priced at {c.get(k)}, not at its issue slot alone: give it "
                         f"one DS_issue line in {fam.key}'s census.csv")
    return out


def reading(fam: Family, vector: str, full: bool = False, q: Fraction = Q) -> Reading:
    """The family's reading on `vector`; `full` puts every census MMA class on a measured menu `honest_menu` limits,
    and `q` is the floors `q = q_T` (A6's ½, or (I3)'s on the measured reading's twin)."""
    B_ = LH._eps_budget()
    sheet = vector == fam.sheet
    ln_p = LH.LN_P if sheet else fam.ln_p_m
    pi = E.machine(vector, fam.data)
    d = E.design(fam.design, fam.data)
    h = E.honest(d.honest, fam.data)
    if (d.delta, d.p, d.checks) != (2.0 ** -41, float(LH.P_OPERAND), 48000):
        raise SystemExit(f"{fam.design}: the statement takes δ′ = 2^-41, operand luck 3/4 and 48000 checks")
    if any(f.p not in (0, 1) for f in d.filler):
        raise SystemExit(f"{fam.design}: a filler enters Y whole at p = 0 (pinned receipts) or not at all at p = 1")
    # Both readings index the sheet's classes. A measured vector times fewer: a sheet class it has no line for
    # (DMMA, the sparse and fp4 forms, which no run beside the census timed) is read as its catch-all is, at the
    # measured issue slot alone, and off its writer menu, as it was when the catch-all counted it (NoSum carries it).
    classes = sorted(fam.sh.kinds)
    if set(pi.c) - set(classes) or (sheet and set(pi.c) != set(classes)):
        raise SystemExit(f"{vector}: classes {sorted(set(pi.c) ^ set(classes))} are not {fam.key}'s sheet.toml's")
    c = {**_coefs(pi, [k for k in classes if k in pi.c]),
         **{k: {"issue": Fraction(1)} for k in classes if k not in pi.c}}
    c = {k: c[k] for k in classes}
    if sheet:
        # The census reads a line as a float (the uniform pipe's 1.96110716); the sheet states it exactly.
        want = S.lines(fam.sh)
        for k in classes:
            if set(c[k]) == set(want.get(k, {})) and all(abs(c[k][r] - v) < Fraction(1, 10 ** 12)
                                                          for r, v in want[k].items()):
                c[k] = dict(want[k])
    untimed = [k for k in classes if k not in pi.c]
    if untimed and "issue" not in pi.rows:
        raise SystemExit(f"{vector}: no measured issue row to read {untimed} at")
    lists = _sheet_check(fam, pi, c) if sheet else {}
    k = CATCH_ALL.get(vector)
    joint = {r: v for r, v in c.get(SCALAR_REF, {}).items() if JOINT_ISSUE.fullmatch(r)}
    if not sheet and k is not None and c.get(k) not in ({"issue": 1}, {"issue": 1, **joint}):
        raise SystemExit(f"{vector}: the catch-all {k} is priced at {c.get(k)}, not at the measured issue slot, alone or with "
                         f"the joint-issue lines {SCALAR_REF} carries ({joint}): give it one issue line in {fam.key}'s "
                         f"census.csv, and the census's JI lines only from a run that times its instructions beside the tensor")
    load = lambda r, n: sum((c[k].get(r, Fraction(0)) * v for k, v in n.items()), Fraction(0))
    per_mma = Fraction(h.words * h.depth, d.L)
    if per_mma.denominator != 1:
        raise SystemExit(f"{fam.design}: {per_mma} committed words an MMA is not whole")
    Nw = int(per_mma)
    nw = {k: exact(v) for k, v in h.tensor("floor", d.floor).items()}
    nx: dict[str, Fraction] = {}
    for f in (f for f in d.filler if f.p == 0):
        tot = sum(exact(v) for v in f.mix.values())
        for k, v in f.mix.items():
            nx[k] = nx.get(k, Fraction(0)) + exact(f.per_word) * Nw * exact(v) / tot / E.LANES

    nb = {k: exact(v) * Nw / E.LANES for k, v in d.baseline.items()}

    def words(wb: Fraction) -> dict[str, Fraction]:
        n = {k: (1 - wb) * v for k, v in nw.items()}
        for k, v in nb.items():
            n[k] = n.get(k, Fraction(0)) + wb * v
        return {k: v for k, v in n.items() if v}

    def program(z: dict[str, Fraction]) -> dict[str, Fraction]:
        fold = _fold_at(d, {j: x for j, x in z.items() if j != BASELINE})
        n: dict[str, Fraction] = {}
        for part in (words(z.get(BASELINE, Fraction(0))), {k: v * Nw / E.LANES for k, v in fold.items()}, nx):
            for k, v in part.items():
                n[k] = n.get(k, Fraction(0)) + v
        return n

    s = E.setup(d, pi, fam.data, operands="free", credit="floor")
    others = {r: x for r, x in s.baseline.words.items() if r not in ("tensor", BASELINE) and x > 1e-12}
    if others or s.baseline.short["words"] > 1e-12:
        raise SystemExit(f"{fam.design} on {vector}: ♭ writes words on {others} or leaves {s.baseline.short['words']} "
                         "undone, where the statement's ♭ writes them by the tensor route and the baseline program")
    wb0 = s.baseline.words.get(BASELINE, 0.0)
    shares = {**s.baseline.folds, **({BASELINE: wb0} if wb0 > 1e-12 else {})}
    zs = _flat_shares(d, shares, lambda z: max(load(r, program(z)) for r in pi.rows))
    z = {j: x for j, x in zs.items() if j != BASELINE}
    if set(z) & set(d.counted_routes):
        raise SystemExit(f"{fam.design} on {vector}: ♭ folds by {sorted(set(z) & set(d.counted_routes))}, which no run "
                         "has built: ♭'s menu is the design's fold and its built routes (`Design.built_routes`)")
    wb = zs.get(BASELINE, Fraction(0))
    fold = _fold_at(d, z)
    nf = {k: v * Nw / E.LANES for k, v in fold.items()}
    nwb = words(wb)
    n0 = program(zs)
    # The routes leave the fold's other classes alone; where ♭'s bound ties two shares, the LP's pick is one of them.
    mixed = {k for r in d.fold_routes.values() for k in {*r, *d.fold} if r.get(k, 0.0) != d.fold.get(k, 0.0)}
    if abs(s.baseline.T * Nw - float(max(load(r, n0) for r in pi.rows))) > 1e-6 or any(
            abs(s.baseline.n.get(k, 0.0) * Nw - float(n0.get(k, 0))) > 1e-6 for k in {*s.baseline.n, *n0} - mixed):
        raise SystemExit(f"{fam.design} on {vector}: ♭ is not setup's floor program ({s.baseline.n}, T "
                         f"{s.baseline.T})")
    e = E.eps(d, pi, float(q), fam.data, operands="free", credit="floor")
    B0 = max(load(r, n0) for r in pi.rows)
    # The skip is lane work, so it sits on the lane facet ♭ loads most; where ♭'s facets tie with one no lane class
    # loads (DC-90-ri's AT90), the exact shares can leave that one above by 1e-10, and the skip lifts the lane facet over.
    flat = max((r for r in pi.rows if any(c[k].get(r) for k in pi.cuda)), key=lambda r: load(r, n0))
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
    writers = [(k, om) for k in classes if k in pi.c and (om := omega(fam, k, h)) and (menu == "all" or k == h.form)]
    writers += [(k, om) for k, om in fam.uniform.items() if k in pi.c]
    folds = [{k: exact(v) / E.LANES for k, v in f.items()} for f in (d.fold, *d.fold_routes.values())]
    if d.uniform and E.UNI in pi.c:
        folds.append({E.UNI: min(exact(sum(f.values())) for f in (d.fold, *d.fold_routes.values()))})
    vec = list(pi.cuda) + [k for k in classes if k in CATCH_ALLS and k not in pi.cuda]
    # Each route's demand on every facet, as `Accounting.TS.Model.demand` reads it: a writer kind 1/ω of its class, the
    # vector route (and the table route, at q_T = q, b_T = 0) q/32 of the cheapest vector class on that facet.
    routes = {k: {r: c[k].get(r, Fraction(0)) / om for r in pi.rows} for k, om in writers}
    routes["vector"] = {r: q / E.LANES * min(c[k].get(r, Fraction(0)) for k in vec) for r in pi.rows}
    fold_d = [{r: load(r, f) for r in pi.rows} for f in folds]
    fill_d = [{r: load(r, nx) for r in pi.rows}] if nx else []
    lam = certificate(_best(pi.rows, Nw, B, int(d.checks), routes, fold_d, fill_d, ln_p))
    at = lambda cert: _bound_at(cert, routes, fold_d, fill_d, Nw, B, ln_p, int(d.checks))
    lam = next(({r: Fraction(1)} for r in pi.rows if at({r: Fraction(1)}) <= at(lam)), lam)
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
    folded = 1 - Y + Nw * w / B * 41 * LH.LN2 / (ln_p * d.checks)
    unfolded = 1 - Y * (1 - Fraction(41, d.checks))
    bound = max(folded, unfolded)
    target = fam.eps if sheet else fam.eps_m
    fr = lambda x: f"{x.numerator}/{x.denominator}"
    nprice = sum((x * load(r, n) for r, x in lam.items()), Fraction(0))
    statement = "" if full else f"{fam.ns}{'Sheet' if sheet else 'Measured'}{'I3' if q != Q else ''}Headline"
    return Reading(fam.ns, vector, fam.design, sheet, menu, fr(q), fr(ln_p), {r: float(x) for r, x in lam.items()},
                   {r: fr(x) for r, x in lam.items()}, flat, wk, Nw, int(d.checks), float(eps_skip), skip, float(B),
                   float(1 - nprice / B), float(w), float(phi), float(chi), float(Nw * w / B), float(1 - Y), lp,
                   float(folded), float(unfolded), float(bound), float(target), e.core(), e.machine_eps, e.luck,
                   statement, _pair(statement, bound, B, flat),
                   {"B": fr(B), "w": fr(w), "phi": fr(phi), "chi": fr(chi), "bound": fr(bound)},
                   _Exact(classes, c, list(pi.rows), vec, lists, writers, folds, nwb, nf, nx, ns, skip, lam,
                          flat, B, bound, Nw, int(d.checks), d.L, h.form, list(d.fold_routes),
                          list(d.counted_routes), _fold_mix(d, z), _words_mix(d, wb), untimed, q, ln_p))


def _bound_at(lam: dict[str, Fraction], words: dict, folds: list, fills: list, Nw: int, B: Fraction, ln_p: Fraction,
              checks: int) -> Fraction:
    """The Lean bound at certificate `lam`, exact. The LP's optimum can be a face (DS-90's unfolded branch binds
    across one): where one row alone attains it, `reading` takes that row, whose proof reads one facet."""
    price = lambda dem: sum((x * dem[r] for r, x in lam.items()), Fraction(0))
    w = min(price(v) for v in words.values())
    Y = (Nw * w + Nw * min(price(f) for f in folds) + min((price(f) for f in fills), default=Fraction(0))) / B
    return max(1 - Y + Nw * w / B * 41 * LH.LN2 / (ln_p * checks), 1 - Y * (1 - Fraction(41, checks)))


def _best(rows: list[str], Nw: int, B: Fraction, checks: int, words: dict, folds: list, fills: list,
          ln_p: Fraction) -> dict[str, float]:
    """The certificate that minimises the Lean bound over this menu: min z over `λ ≥ 0`, `Σ λ = 1`, with
    `t_w ≤ λ·v_ρ` for every word route, `t_f ≤ λ·u_φ` for every fold route, `t_x ≤ λ·x_ψ` for the filler, and z at
    least both branches, `1 − Y + c·N_w t_w/B` and `1 − Y·(1 − 41/S)`, `Y = (N_w t_w + N_w t_f + t_x)/B`, with
    `c = 41·ln 2/(ln_p·S)`."""
    from scipy.optimize import linprog

    R = len(rows)
    iw, i_f, ix, iz = R, R + 1, R + 2, R + 3
    nv = R + 4
    cfold = float(Fraction(41) * LH.LN2 / (ln_p * checks))
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
SHAPES = {"Sm90": "sm_90's wgmma as one warp's quarter of its warpgroup instruction (`m·n·k/4`), a 2:4 sparse form at "
                  "its dense twin's, its mma.sync whole, DMMA's at the f64 peak",
          "Sm100": "sm_100's tcgen05 MMA whole (`m·n·k`), a 2:4 sparse form at its dense twin's, its mma.sync whole, "
                   "DMMA's at the f64 peak"}


def _entries(x: _Exact, n: dict[str, Fraction]) -> list[str]:
    return [f"({x.classes.index(k)}, {v.numerator}, {v.denominator})"
            for k, v in sorted(n.items(), key=lambda kv: x.classes.index(kv[0]))]


def _sparse(x: _Exact, n: dict[str, Fraction]) -> str:
    return "[" + ", ".join(_entries(x, n)) + "]"


def _sparse_def(name: str, x: _Exact, n: dict[str, Fraction]) -> str:
    head = f"def {name} : Sparse := ["
    return head + LH._wrap(_entries(x, n), " " * len(head))[len(head):] + "]"


def _q(v: Fraction) -> str:
    return f"{v.numerator} / {v.denominator}"


def _uniform_words(fam: Family) -> str:
    return "".join(f" and the uniform `{k}` ({om}, a UR pair)" for k, om in fam.uniform.items())


MOVES = ("-rt3-wideh", "-rt3", "-wideh")          # the counted moves of a lane route's LI


def _menu_text(names: list[str], counted: list[str]) -> str:
    """The fold menu's programs as the Definitions docstring names them: the built ones (the bit-exact checkpoint loops
    and li-bulk at its measured cost), then the counted ones (the moves of LI on each lane route and li-bulk's
    members), which ♭ never folds by."""
    built = [n for n in names if n not in counted]
    loops = [n for n in built if not n.startswith("li-bulk")]
    bulk = [n for n in built if n not in loops]
    moved = [n for n in counted if n.endswith(MOVES)]
    members = [n for n in counted if n not in moved]
    if any(not n.startswith("li-bulk") for n in members):
        raise SystemExit(f"counted folds {members}: the Definitions docstring names the moves of LI and li-bulk's "
                         "members only; describe the others in `_menu_text`")
    text = f"programs ({', '.join(loops)}: the checkpoint's bit-exact loops"
    if not (bulk or counted):
        return text.replace("programs", "bit-exact programs", 1) + ")"
    if bulk:
        text += (f"; {', '.join(bulk)}: li's XOR through the stores, a bulk XOR with the key slab and a load back, at "
                 "its measured cost")
    if counted:
        text += "; and the counted folds, none built, which ♭ never folds by: "
        text += "; ".join(
            ([f"{', '.join(moved)}, the counted moves: each lane route and the fold with LI's 64 LOP3 a "
              "segment as a memory round trip of three issue slots (ATOM.E.XOR and two LD.E back), its 64 IMAD as "
              "IMAD.WIDE, or both"] if moved else [])
            + ([f"{', '.join(members)}, li-bulk's BT1024 and trimmed W = ∞ members"] if members else []))
    return text + ")"


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
        f"`classes`: the {nk} instruction classes of the {fam.sheet} census, in the order `Fin {nk}` indexes them."
        + (f" `{CATCH_ALL[fam.key]}` is A2's catch-all: every warp instruction no other class names counts there, at its "
           f"issue slot alone." if fam.key in CATCH_ALL else ""),
        f"`{sv}`, `sheet`: the {fam.gpu}'s data sheet ({fam.sheet}), `{S._rel(fam.sh)}` as "
        f"`{fam.sh.lean}` states it: f_max {fam.fmax} MHz ({fam.fmax_src}); each tensor class's dense MACs a warp "
        f"instruction ({SHAPES[fam.ns]}) and its operand type's dense MACs an SM clock; the Nsight Compute pipes "
        f"{', '.join(p.upper() for p in lane[:-1])} and {lane[-1].upper()}, each at its width (the Programming "
        f"Guide's, and the uniform datapath's the census's measured UP line, which no document states); no "
        f"shared-memory row ({fam.sheet} prices none). An MMA class the sheet rates on no peak (b1) and the memory, "
        f"atomic and control classes are their issue slot alone (A2). Its machine reproduces every {fam.sheet} "
        f"coefficient.",
        f"`d2s`, `model`: `{fam.design}` as a TS model on the sheet's machine: writer kinds every MMA class of the "
        f"census (ω its destination's 32-bit words a warp instruction, A6, sheet.toml's `omega`: an f64 element two)"
        + _uniform_words(fam)
        + f"; the vector classes CUDA's lane classes and both readings' catch-alls "
        f"({', '.join(k for k in x.vec if k in CATCH_ALLS)}: A6 counts a vector word's instructions in them, and an "
        "instruction no class names, IMAD.WIDE among them, counts in the catch-all)"
        + "; the fold routes the design's fold"
        + (f" and its {len(x.fold_names)} other {_menu_text(x.fold_names, x.counted)} on lanes, and alone on the uniform datapath "
           "at the least of their counts; "
           if x.fold_names else " on lanes and alone on the uniform datapath; ")
        + f"the table level's bytes counted in {fam.bytes}, which `b_T = 0` prices at nothing (the census has no load "
        f"class); A6's floors `q = q_T = ½`, `b_T = 0`; {x.Nw} committed words a unit of honest work ({honest})"
        + ("; its filler menu the one route ffill runs on (sm_90's uniform datapath has no FFMA)." if filler else "."),
        f"`flat`: the floor program ♭ a unit: the MMA and the design's operand floor (word production), the fold "
        f"({x.fold_mix}), {fill_doc}and ε_skip's skippable work at {x.skip}'s price, {r.eps_skip:.6%} of ♭'s bound "
        f"(`eps_budget.design_skip` on {fam.sheet}).",
        f"`lamStar`: π*, the certificate minimising the bound over this menu ({' + '.join(f'{k} {float(v):.6f}' for k, v in x.lam.items())}).",
        "`checks`: the checks a window, one a round (Theorem S's S).",
    ]
    pipes = [f"some .{p}" if p else "none" for p in x.sheet["pipe"]]
    fold_lines = ",\n".join(f"    {_sparse(x, f)}" for f in x.folds)
    pipe_ctors = "\n".join(f"  | {p}" for p in lane)
    width_lines = "\n".join(f"  | .{p} => {S._num(w)}" for p, w in fam.pipes.values())
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

/-- Each pipe's results an SM clock: the Programming Guide's, the uniform datapath's the census's UP line. -/
def Pipe.width : Pipe → ℚ
{width_lines}

/-- The {fam.gpu}'s data sheet, by class index. -/
def {sv} : SheetDataQ Pipe where
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
  q := {_q(x.q)}
  qT := {_q(x.q)}
  Nw := {x.Nw}

/-- ♭'s filler a unit{": ffill on lanes" if filler else ": none"}. -/
{_sparse_def("flatFill", x, x.nx)}

/-- `{fam.design}`'s TS model on the data sheet's machine. -/
{_model_line(fam, x, "d2s", "sheet.machine", "flatFill", "(SheetRow Pipe)")}

{LH._words(f"♭'s word production a unit: {x.words_mix}. -/", "/-- ", "")}
{_sparse_def("flatWords", x, x.nw)}

{LH._words(f"♭'s fold a unit: the lane fold for {x.Nw} words, {x.fold_mix}. -/", "/-- ", "")}
{_sparse_def("flatFold", x, x.nf)}

/-- ♭'s skippable work a unit, at {x.skip}'s price. -/
{_sparse_def("flatSkip", x, x.ns)}

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
    if x.classes != xs.classes or x.vec != xs.vec or x.folds != xs.folds or x.nx != xs.nx:
        raise SystemExit(f"{fam.measured}: its classes, vector classes, fold routes or ♭'s filler are not "
                         f"{fam.sheet}'s, which the measured file reads")
    own, own_w = x.nf != xs.nf, x.nw != xs.nw
    fold = "flatFold" if own else f"{fam.ns}.flatFold"
    words = "flatWords" if own_w else f"{fam.ns}.flatWords"
    own_fold = ((LH._words(f"♭'s word production a unit on {fam.measured}: {x.words_mix}. -/", "/-- ", "")
                 + f"\n{_sparse_def('flatWords', x, x.nw)}\n\n") if own_w else "") + (
        (LH._words(f"♭'s fold a unit on {fam.measured}: the lane fold for {x.Nw} words, {x.fold_mix}. -/", "/-- ", "")
         + f"\n{_sparse_def('flatFold', x, x.nf)}\n\n") if own else "")
    mine = [p for p, o in (("word production", own_w), ("fold", own)) if o]
    theirs = [p for p, o in (("word production", own_w), ("fold", own)) if not o] + ["filler"]
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
            "them" if r.menu == "honest" else "every MMA class it times (ω its destination's words, A6)")
    lam = LH._wrap([f"({x.rows.index(k)}, {v.numerator}, {v.denominator})" for k, v in x.lam.items()],
                   " " * (len(f"  cert {m} ["))).lstrip()
    joint = fam.measured in CATCH_ALL and set(x.c[CATCH_ALL[fam.measured]]) != {"issue"}
    bullets = [
        f"`rowNames`, `{sv}`, `machine`: the measured vector {fam.measured}, {m} rows of SMSP-cycles a warp instruction "
        f"at its epoch's census, each sparse over `{fam.ns}.classes`; every facet SM-clocked, f_max {fam.fmax} MHz."
        + (f" `{CATCH_ALL[fam.measured]}` is A2's catch-all here: every warp instruction no other class prices on "
           f"{fam.measured} counts there, at the measured issue slot "
           + ("and the joint-issue lines every timed scalar class carries, since a run timed its instructions beside "
              "the tensor." if joint else "alone.") if fam.measured in CATCH_ALL else "")
        + (f" The sheet's classes no run beside {fam.measured}'s census timed ({', '.join(x.untimed)}) are read "
           + ("" if joint else "the same way, ") + "at the measured issue slot alone, and are off the writer menu, as "
           "they were when the catch-all counted them: `NoSum` carries their words." if x.untimed else ""),
        f"`d2s`, `model`: `{fam.design}` on that machine, its writer kinds {menu}; `{fam.ns}.d2s`'s vector classes, "
        f"fold routes, floors and `N_w`" + ("; ♭'s filler `" + fam.ns + ".flatFill` its filler route." if x.nx else "."),
        "".join(f"`{n}`, " for n, o in (("flatWords", own_w), ("flatFold", own)) if o)
        + f"`flatSkip`, `flat`: ♭ is `{fam.ns}.flat`'s {_and(theirs)}"
        + "".join(f", its {p} {fam.measured}'s ({x.words_mix if p == 'word production' else x.fold_mix})"
                  for p in mine)
        + f", with ε_skip's skippable work at {fam.measured}'s {r.eps_skip:.6%} (`eps_budget.design_skip` on "
        f"{fam.measured}), at {x.skip}'s price.",
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

{own_fold}/-- ♭'s skippable work a unit on {fam.measured}, at its {x.skip} price. -/
{_sparse_def("flatSkip", x, x.ns)}

/-- The floor program ♭ on {fam.measured}, labelled. -/
noncomputable def flat : Labelled (Fin {nk}) :=
  labelled {nk} {words} {fold} {fam.ns}.flatFill flatSkip

/-- π*: the certificate, its weights exact and summing to 1. -/
noncomputable def lamStar : Fin {m} → ℝ :=
  cert {m} [{lam}]

end Accounting.TS.Headline.{fam.ns}Measured
'''


def render_twin(fam: Family, t: Reading, r: Reading) -> str:
    """The measured reading's twin under (I3) `ScalarRoute` at `q = q_T = fam.twin`: the measured file's machine, writer
    kinds, routes and ♭, none of which reads `q`, at the twin's floors and its own π*."""
    x, xm = t.data, r.data
    if (x.classes, x.rows, x.vec, x.writers, x.folds, x.nw, x.nf, x.nx, x.ns, x.B) != (
            xm.classes, xm.rows, xm.vec, xm.writers, xm.folds, xm.nw, xm.nf, xm.nx, xm.ns, xm.B):
        raise SystemExit(f"{fam.measured} at q = {t.q}: its machine, menu or ♭ is not the measured file's, which the "
                         "twin reads")
    m, nk, nt, nfo = len(x.rows), len(x.classes), len(x.writers), len(x.folds)
    lam = LH._wrap([f"({x.rows.index(k)}, {v.numerator}, {v.denominator})" for k, v in x.lam.items()],
                   " " * (len(f"  cert {m} ["))).lstrip()
    qd = f"{x.q.numerator}/{x.q.denominator}"
    body = (f"d2s.modelFill {nk} {nt} {nfo} {fam.ns}Measured.machine {fam.ns}.flatFill" if x.nx
            else f"d2s.model {nk} {nt} {nfo} {fam.ns}Measured.machine")
    bullets = [
        f"`d2s`, `model`: `{fam.ns}Measured.d2s` (its writer kinds, and `{fam.ns}.d2s`'s vector classes, fold routes "
        f"and `N_w`) at (I3)'s floors `q = q_T = {qd}`, `b_T = 0`, on `{fam.ns}Measured.machine`. (I3) `ScalarRoute` "
        "there is `accounting-ts-scalar-route`, assumed: A6 gives it only at `q, q_T ≤ ½` (`SheetScalarRouteHalf`). ♭ "
        f"is `{fam.ns}Measured.flat`, since `setup`'s floor program doesn't read `q`.",
        f"`lamStar`: π*, the certificate minimising the bound over this menu at `q = {qd}` "
        f"({' + '.join(f'{k} {float(v):.6f}' for k, v in x.lam.items())}).",
    ]
    return _header(fam, f"(I3) twin: {fam.measured} at q = {qd} and `{fam.design}`",
                   f"Definitions.Accounting.HeadlineDc.{fam.ns}Measured", f"{fam.ns}MeasuredI3Headline") + f'''
{_bullets(bullets)}
-/

namespace Accounting.TS.Headline.{fam.ns}MeasuredI3

open Accounting.TS

/-- `{fam.ns}Measured.d2s` at (I3)'s floors `q = q_T = {qd}`. -/
def d2s : DesignData where
  writers := {fam.ns}Measured.d2s.writers
  vec := {fam.ns}Measured.d2s.vec
  folds := {fam.ns}Measured.d2s.folds
  bytes := {fam.ns}Measured.d2s.bytes
  q := {_q(x.q)}
  qT := {_q(x.q)}
  Nw := {fam.ns}Measured.d2s.Nw

/-- `{fam.design}`'s TS model on the measured machine at (I3)'s floors. -/
noncomputable def model : Model (Fin {nk}) (Fin {m}) (Fin {nt}) Unit Empty (Fin {nfo}) Unit :=
  {body}

/-- π*: the certificate, its weights exact and summing to 1. -/
noncomputable def lamStar : Fin {m} → ℝ :=
  cert {m} [{lam}]

end Accounting.TS.Headline.{fam.ns}MeasuredI3
'''


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="write every reading here as JSON")
    ap.add_argument("--check", action="store_true", help="exit 1 when a committed Lean file is not what it would write")
    ap.add_argument("--numbers", action="store_true", help="print the readings only; write no Lean file")
    a = ap.parse_args(argv)
    out, record, files = [], [], {}
    for fam in FAMILIES:
        for v, full, q in ((fam.sheet, False, Q), (fam.measured, False, Q),
                           *(((fam.measured, True, Q),) if fam.honest_menu else ()),
                           *(((fam.measured, False, fam.twin),) if fam.twin else ())):
            r = reading(fam, v, full, q)
            (record if full else out).append(r)
            if not full and q != Q:
                files[OUT / f"{fam.ns}MeasuredI3.lean"] = render_twin(fam, r, out[-2])
            elif not full:
                files[OUT / (fam.ns + ("" if r.sheet else "Measured") + ".lean")] = (
                    render_sheet(fam, r) if r.sheet else render_measured(fam, r, out[-2]))
            cert = " + ".join(f"{k} {x:.4f}" for k, x in r.certificate.items())
            print(f"{r.design} on {v} ({r.menu} menu{', for the record' if full else ''}, q = q_T = {r.q}, "
                  f"ln(8/7) ≥ {r.ln_p}{', the (I3) twin' if q != Q else ''}): π* {cert}, ♭ on "
                  f"{r.flat_facet} at {r.B:.4f} a unit (ε_cert {r.eps_cert:.4%}), w* {r.wstar}, ε_skip "
                  f"{r.eps_skip:.4%} (skip as {r.skip_class}), κ {r.kappa:.6f}, χ* {r.chi:.4f}; 1 − Y "
                  f"{r.one_minus_Y:.4%} (the LP's {r.lp:.4%}); folded {r.folded:.4%}, unfolded {r.unfolded:.4%}: the "
                  f"Lean bound {r.bound:.4%} {'≤' if r.bound <= r.target else 'ABOVE'} {r.target:.2%} (machine "
                  f"model: core {r.machine_core:.4%}, ε_machine {r.machine_eps:.4%}, ε_luck {r.luck:.4%})"
                  + (f"; {r.statement}: ε {float(Fraction(r.pair['eps'])):.3%} ({r.pair['eps']}) with B_♭ "
                     f"{float(Fraction(r.pair['B_flat'])):.4f} "
                     f"on {r.pair['flat_facet']}, credit (1 − ε)·B_♭ {float(Fraction(r.pair['credit'])):.4f}"
                     if r.statement else ""))
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
