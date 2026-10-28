"""A device's data sheet (`sheet.toml` in its directory): each constant of the data sheet (`Accounting.TS.Sheet`), its
value, the document that states it and its grade, and each class of its DS vector's epoch with its tensor form or its
pipe.

`lines` derives the DS lines from it, exactly: issue 1 for every class (A2, and A2u for the uniform datapath's), a tensor
form's MACs a warp instruction over a quarter of its type's peak (A3), 128 over its pipe's width (A4; the uniform pipe's
is the measured UP line, which no document states), and `omega` each tensor class's A6 ω, its accumulator's words a
warp instruction. `differences` holds the lines to the census file of the DS epoch, `figures` holds each per-clock
peak to the figure the document prints at the clock it names, and `lean` prints the Lean module that states it all
(`lean` in sheet.toml), with a lemma for each figure and each clock. `reading` prices the headline design (headline.toml) on the sheet's lines, beside
headline.py's ε and with A3 read per second at f_max, `moves` finds how much slower the tensor pipe can be before
that ε moves, and `uni` prices it with a uniform instruction at a given share of an issue slot (A2u).
"""
from __future__ import annotations

import csv
import textwrap
import tomllib
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

from verity_machine_model import headline as H
from verity_machine_model.eps import UNI, design, eps, machine
from verity_machine_model.model import model, vector

ROOT = Path(__file__).resolve().parents[4]
SHEET = "sheet.toml"


@dataclass(frozen=True)
class Const:
    value: Fraction
    grade: str
    source: str


@dataclass(frozen=True)
class Peak:
    macs: Fraction      # dense MACs an SM clock (A3)
    sheet: Fraction     # the document's figure, TOPS or TFLOPS
    sparse: bool        # the figure is with 2:4 sparsity, the dense one half of it
    tol: Fraction       # half the figure's last printed digit
    clock: str          # the clock (`[clocks.NAME]`) the figure is quoted at
    grade: str
    source: str
    element_words: int = 1  # 32-bit registers an accumulator element takes: f64's 2, every other type's 1 (A6)

    @property
    def dense(self) -> tuple[Fraction, Fraction]:
        """The dense figure and its tolerance, TOPS."""
        return (self.sheet / 2, self.tol / 2) if self.sparse else (self.sheet, self.tol)


@dataclass(frozen=True)
class Form:
    type: str
    m: int
    n: int
    k: int
    warps: int          # the warps whose instructions issue one MMA: a wgmma's warpgroup 4, mma.sync and tcgen05.mma 1
    sparse: bool = False  # 2:4 sparse, k its logical k

    @property
    def macs(self) -> Fraction:
        """Dense MACs one warp instruction does: a 2:4 sparse form's are half its logical m · n · k (A3)."""
        return Fraction(self.m * self.n * self.k, self.warps * (2 if self.sparse else 1))


@dataclass(frozen=True)
class Kind:
    form: Form | None
    pipe: str | None


@dataclass(frozen=True)
class Sheet:
    name: str
    gpu: str
    vector: str
    lean: str
    fmax: Const         # MHz
    sms: Const
    issue: Const        # A2: the slots a warp instruction takes, 1
    uniform_issue: Const  # A2u: the slots a uniform instruction takes, 1
    clocks: dict[str, Const]
    labels: dict[str, str]  # each clock's name in prose
    peaks: dict[str, Peak]
    pipes: dict[str, Const]
    kinds: dict[str, Kind]
    omega: tuple[str, str]  # A6's ω, a tensor class's accumulator words: its grade and source


def _const(t: dict, key: str) -> Const:
    return Const(Fraction(t[key]), t["grade"], t["source"])


def sheet(data: Path) -> Sheet:
    p = data / SHEET
    if not p.is_file():
        raise SystemExit(f"{data}: no {SHEET}; a device with a data sheet has one (README \"Devices\")")
    t = tomllib.loads(p.read_text())
    kinds = {}
    for k, v in t["kinds"].items():
        form = Form(v["type"], v["m"], v["n"], v["k"], v["warps"], v.get("sparse", False)) if "type" in v else None
        kinds[k] = Kind(form, v.get("pipe"))
        if form and kinds[k].pipe:
            raise SystemExit(f"{p}: {k} has a tensor form and a pipe; a class has one or neither")
    s = Sheet(name=t["name"], gpu=t["gpu"], vector=t["vector"], lean=t["lean"], fmax=_const(t["fmax"], "mhz"),
              sms=_const(t["sms"], "value"), issue=_const(t["issue"], "slots"),
              uniform_issue=_const(t["uniform_issue"], "slots"),
              clocks={c: _const(v, "mhz") for c, v in t["clocks"].items()},
              labels={c: v.get("label", f"`{c}` clock") for c, v in t["clocks"].items()},
              peaks={ty: Peak(Fraction(v["macs"]), Fraction(v["sheet"]), v["sparse"], Fraction(v["tol"]), v["clock"],
                              v["grade"], v["source"], v.get("element_words", 1)) for ty, v in t["peaks"].items()},
              pipes={q: _const(v, "width") for q, v in t["pipes"].items()}, kinds=kinds,
              omega=(t["omega"]["grade"], t["omega"]["source"]))
    if s.issue.value != 1 or s.uniform_issue.value != 1:
        raise SystemExit(f"{p}: [issue] and [uniform_issue] take 1 slot, as `lines`' issue row does; price A2u "
                         f"at another share with `sheet --eps --uni`")
    for c, v in s.clocks.items():
        if v.value > s.fmax.value:
            raise SystemExit(f"{p}: [clocks.{c}] is {_dec(v.value)} MHz, above f_max's {_dec(s.fmax.value)}; A1 bounds "
                             f"the run's clock by f_max, so a figure quoted above it reads a clock no run reaches")
    for k, v in kinds.items():
        if v.form and v.form.type not in s.peaks:
            raise SystemExit(f"{p}: {k}'s type {v.form.type} has no [peaks.{v.form.type}]")
        if v.pipe and v.pipe not in s.pipes:
            raise SystemExit(f"{p}: {k}'s pipe {v.pipe} has no [pipes.{v.pipe}]")
    for ty, v in s.peaks.items():
        if v.clock not in s.clocks:
            raise SystemExit(f"{p}: [peaks.{ty}] is at clock {v.clock!r}, which has no [clocks.{v.clock}]")
    return s


def row(name: str) -> str:
    """The census row of a sheet row: `DS_issue`, `DS_tensor`, `DS_<pipe>`."""
    return f"DS_{name}"


def omega(s: Sheet) -> dict[str, Fraction]:
    """A6's ω for each tensor class: the 32-bit words its accumulator fragment takes a warp instruction, m · n elements
    at their type's registers each over the warps that issue one MMA (a 2:4 sparse form's m · n as its dense twin's)."""
    return {k: Fraction(v.form.m * v.form.n * s.peaks[v.form.type].element_words, v.form.warps)
            for k, v in s.kinds.items() if v.form}


def lines(s: Sheet, peaks: dict[str, Fraction] | None = None) -> dict[str, dict[str, Fraction]]:
    """Each class's SMSP-cycles a warp instruction on each row it holds (a row it holds not at all has no line), at the
    sheet's peaks or at `peaks`."""
    peaks = peaks or {ty: p.macs for ty, p in s.peaks.items()}
    out = {}
    for k, v in s.kinds.items():
        out[k] = {row("issue"): Fraction(1)}
        if v.form:
            out[k][row("tensor")] = v.form.macs / (peaks[v.form.type] / 4)
        if v.pipe:
            out[k][row(v.pipe)] = 128 / s.pipes[v.pipe].value
    return out


def census_lines(s: Sheet, data: Path) -> dict[str, dict[str, Fraction]]:
    """The DS epoch's census file (`census/EPOCH.csv`), each line's cycles exact."""
    e = vector(s.vector, data).epoch
    out: dict[str, dict[str, Fraction]] = {}
    with open(data / "census" / f"{e}.csv", newline="") as f:
        for x in csv.DictReader(f):
            if x["cycles"] == "" or x["rate"] or x["u"]:
                raise SystemExit(f"census/{e}.csv: {x['kind']} {x['row']} gives no cycles; a DS line is in SMSP-cycles")
            out.setdefault(x["kind"], {})[x["row"]] = Fraction(x["cycles"])
    return out


def differences(s: Sheet, data: Path) -> list[str]:
    """Where the sheet's lines and the census file differ, and any line on a row the DS vector doesn't price."""
    v = vector(s.vector, data)
    if v.epoch not in model(data).epochs:
        return [f"{s.vector}'s epoch {v.epoch} isn't in model.toml"]
    have, want = census_lines(s, data), lines(s)
    out = [f"{k} {r}: no row of {s.vector}" for k, rs in want.items() for r in rs if r not in v.rows]
    for k in sorted(have.keys() | want.keys()):
        a, b = have.get(k, {}), want.get(k, {})
        for r in sorted(a.keys() | b.keys()):
            if a.get(r) != b.get(r):
                out.append(f"{k} {r}: census {a.get(r, 'none')}, sheet {b.get(r, 'none')}")
    return out


@dataclass(frozen=True)
class Figure:
    type: str
    value: Fraction     # 2 · macs · SMs · clock, TOPS
    dense: Fraction
    tol: Fraction
    low: Fraction       # the clocks, MHz, at which the figure holds
    high: Fraction

    @property
    def holds(self) -> bool:
        return abs(self.value - self.dense) <= self.tol


def figures(s: Sheet) -> list[Figure]:
    """Each peak's per-second reading at its clock against the document's dense figure."""
    out = []
    for ty, p in s.peaks.items():
        per_mhz = 2 * p.macs * s.sms.value / 10**6   # TOPS a MHz
        dense, tol = p.dense
        out.append(Figure(ty, per_mhz * s.clocks[p.clock].value, dense, tol, (dense - tol) / per_mhz,
                          (dense + tol) / per_mhz))
    return out


def at_fmax(s: Sheet) -> dict[str, Fraction]:
    """Each peak read per second at f_max instead of per clock: the dense figure over 2 · SMs · f_max."""
    return {ty: p.dense[0] * 10**6 / (2 * s.sms.value * s.fmax.value) for ty, p in s.peaks.items()}


@dataclass(frozen=True)
class Reading:
    family: str
    design: str
    q: float
    sheet: float        # the core ε on the sheet's own lines
    headline: float     # headline.py's at its sheet reading
    fmax: float         # with each tensor line at at_fmax's peaks


def _own(pi, s: Sheet, peaks: dict[str, Fraction] | None = None):
    """Π with every line the sheet's own (`lines`), none of the census file's."""
    if set(pi.c) != set(s.kinds):
        raise SystemExit(f"{s.vector}: Π's classes {sorted(set(pi.c) ^ set(s.kinds))} aren't the sheet's, or only its")
    return replace(pi, c={k: {r: float(x) for r, x in rs.items()} for k, rs in lines(s, peaks).items()})


def _setting(s: Sheet, data: Path):
    family = data.name
    t = H.spec()
    f = t["family"][family]
    if f["sheet"]["vector"] != s.vector:
        raise SystemExit(f"headline.toml's {family} sheet reading is {f['sheet']['vector']}, the sheet's {s.vector}")
    pi = _own(machine(s.vector, data), s)
    return family, design(f["design"], data), pi, float(t["reading"]["sheet"]["q"])


def _core(d, pi, q: float, data: Path) -> float:
    return eps(d, pi, q, data, operands=H.OPERANDS, credit=H.CREDIT).core()


def reading(s: Sheet, data: Path) -> Reading:
    """The headline design's core ε at its sheet reading (headline.toml: q* = ½, operands free, credit ♭) on the
    sheet's own lines, beside headline.py's, and with A3 read per second at f_max."""
    family, d, pi, q = _setting(s, data)
    return Reading(family=family, design=d.name, q=q, sheet=_core(d, pi, q, data),
                   headline=H.row(family, "sheet").core, fmax=_core(d, _own(pi, s, at_fmax(s)), q, data))


def moves(s: Sheet, data: Path, top: float = 4.0, steps: int = 30) -> float | None:
    """The least factor on every tensor line (a tensor pipe that much slower than the sheet's) at which the headline ε
    moves, to 1/2^steps of `top`; None if it doesn't move by `top`."""
    _, d, pi, q = _setting(s, data)
    base = _core(d, pi, q, data)
    at = lambda x: _core(d, _own(pi, s, {ty: p.macs / Fraction(x) for ty, p in s.peaks.items()}), q, data)
    if abs(at(top) - base) <= 1e-12:
        return None
    lo, hi = 1.0, top
    for _ in range(steps):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if abs(at(mid) - base) <= 1e-12 else (lo, mid)
    return hi


def uni(s: Sheet, data: Path, cost: float) -> float:
    """The headline design's core ε at its sheet reading with a uniform instruction at `cost` of an issue slot (A2u
    read at `cost`; the sheet's is 1). `eps --uni-issue` leaves a DS vector as it is, since its census charges the slot
    already, so this is the way to sweep A2u on a sheet."""
    _, d, pi, q = _setting(s, data)
    if UNI not in s.kinds:
        raise SystemExit(f"{s.vector}: the sheet has no {UNI} class, so no uniform route to price")
    c = {k: dict(v) for k, v in pi.c.items()}
    c[UNI][row("issue")] = cost
    return _core(d, replace(pi, c=c), q, data)


# --- Lean -----------------------------------------------------------------------------------------------------------

WIDTH = 120


def _stated(c: Const) -> bool:
    """Whether a document states the constant: its grade isn't `none (...)` or `measured only`."""
    return not c.grade.startswith(("none", "measured only"))


def _num(x: Fraction) -> str:
    """A Lean literal."""
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator} / {x.denominator}"


def _dec(x: Fraction) -> str:
    """A number in prose: the decimal, which every value of a sheet has."""
    return f"{float(x):g}"


def _camel(name: str) -> str:
    return name[:1].upper() + name[1:]


def _wrap(text: str, first: str, rest: str = "") -> list[str]:
    return textwrap.wrap(text, WIDTH, initial_indent=first, subsequent_indent=rest, break_long_words=False,
                         break_on_hyphens=False)


def _doc(text: str) -> list[str]:
    if "-/" in text:
        raise SystemExit(f"a docstring can't hold '-/': {text!r}")
    return _wrap(text + " -/", "/-- ")


def _rel(s: Sheet) -> str:
    return f"benchmarks/pouw/machine_model/data/devices/sm_{s.vector.split('-')[1]}/{SHEET}"


def lean(s: Sheet, data: Path) -> str:
    """The Lean module `s.lean`: the sheet's constants with their sources, `sheet : Sheet Kind Pipe` and the figures'
    lemmas."""
    epoch = vector(s.vector, data).epoch
    kinds, pipes, types = sorted(s.kinds), list(s.pipes), list(s.peaks)
    device = f"sm_{s.vector.split('-')[1]}"
    peak = {ty: f"peak{_camel(ty)}" for ty in types}
    width = {q: f"width{_camel(q)}" for q in pipes}
    clock = {c: f"{c}MHz" for c in s.clocks}
    out = ["import Definitions.Accounting.TS.Sheet", "", "/-!", f"# The {s.name}'s data sheet (generated)", ""]
    out += _wrap(f"The data sheet (`Accounting.TS.Sheet`) at the {s.gpu}'s vendor values: each constant with its "
                 "label (A1–A6), its value, its grade and its source. Each tensor peak has a lemma that the per-clock "
                 "peak reproduces the document's per-second figure at the clock that figure is quoted at. A3's MACs "
                 "are device totals over the SMSPs, so a warpgroup's `wgmma` or a single-thread `tcgen05.mma` counts "
                 "its MACs once, spread over the SM's four SMSPs.", "")
    out += [""] + _wrap("Scope: the sheet is one GPU's own SMs. Work a peer GPU on the node does over NVLink or "
                        "NVSwitch and sends back (whole tiles as 64-byte digests) is outside it: the sheet bounds what "
                        "the credited GPU's SMs compute, not which GPU computed a tile, and `Device.lean`'s side "
                        "channels don't list NVLink peer offload.", "")
    out += [""] + _wrap(f"`uv run machine-model --device {device} sheet --lean` prints this file from `{_rel(s)}`, and "
                        "the machine model's tests hold it to that output.", "")
    out += ["-/", "", f"namespace Accounting.TS.Sheets.{s.name}", "", "noncomputable section", ""]
    # Not `deriving Fintype`: Mathlib's handler for an enumeration fails its own `rw [Finset.mem_mk, …]` under this
    # toolchain, so each instance lists the constructors.
    out += [*_doc(f"The classes of the `{epoch}` census (its kinds)."), "inductive Kind", *(f"  | {k}" for k in kinds),
            "  deriving DecidableEq", "", "instance instFintypeKind : Fintype Kind :=",
            *_wrap(f"Fintype.ofList [{', '.join(f'.{k}' for k in kinds)}]", "  ", "    "),
            "    (by intro k; cases k <;> decide)", ""]
    out += [*_doc(f"The Nsight Compute pipes the `{epoch}` classes issue to."), "inductive Pipe",
            *(f"  | {q}" for q in pipes), "  deriving DecidableEq", "", "instance instFintypePipe : Fintype Pipe :=",
            f"  Fintype.ofList [{', '.join(f'.{q}' for q in pipes)}] (by intro p; cases p <;> decide)", ""]
    out += [*_doc("The tensor operand types with a dense peak."), "inductive Ty", *(f"  | {ty}" for ty in types),
            "  deriving DecidableEq", ""]
    out += [*_doc(f"A1, f_max: {_dec(s.fmax.value)} MHz. Grade: {s.fmax.grade}. Source: {s.fmax.source}."),
            f"def fmaxMHz : ℝ := {_num(s.fmax.value)}", ""]
    out += [*_doc(f"The SMs A3's per-SM peaks multiply: {_dec(s.sms.value)}. Grade: {s.sms.grade}. Source: "
                  f"{s.sms.source}."), f"def smCount : ℝ := {_num(s.sms.value)}", ""]
    for c, v in s.clocks.items():
        at = [ty for ty, p in s.peaks.items() if p.clock == c]
        out += [*_doc(f"The {s.labels[c]}, at which the {', '.join(at)} figure{'s are' if len(at) > 1 else ' is'} "
                      f"quoted: {_dec(v.value)} MHz. Grade: {v.grade}. Source: {v.source}."),
                f"def {clock[c]} : ℝ := {_num(v.value)}", ""]
    for ty, p in s.peaks.items():
        c = s.clocks[p.clock]
        out += [*_doc(f"A3, {ty}: {_dec(p.macs)} dense MACs an SM clock, the document's dense figure, "
                      f"{_dec(p.dense[0])} × 10¹² a second, over 2 · {_dec(s.sms.value)} SMs at the "
                      f"{s.labels[p.clock]}, {_dec(c.value)} MHz (A1's f_max is "
                      f"{_dec(s.fmax.value)} MHz). Grade: {p.grade}. Source: {p.source}."),
                f"def {peak[ty]} : ℝ := {_num(p.macs)}", ""]
    out += [*_doc("A3: each type's dense MACs an SM clock."), "def Ty.peak : Ty → ℝ",
            *(f"  | .{ty} => {peak[ty]}" for ty in types), ""]
    for q, w in s.pipes.items():
        out += [*_doc(f"A4, the {q} pipe: {_dec(w.value)} results an SM clock, {_dec(128 / w.value)} SMSP-cycles a "
                      f"warp instruction. Grade: {w.grade}. Source: {w.source}."),
                f"def {width[q]} : ℝ := {_num(w.value)}", ""]
    out += [*_doc("A4: each pipe's results an SM clock."), "def Pipe.width : Pipe → ℝ",
            *(f"  | .{q} => {width[q]}" for q in pipes), ""]
    out += [*_doc("Each class's tensor form: its operand type and its dense MACs a warp instruction, m · n · k over the "
                  "warps that issue one MMA (a `wgmma` warpgroup's 4, 1 for `mma.sync` and the single-thread "
                  "`tcgen05.mma`), and for a 2:4 sparse form, whose k is its logical k, half that (A3: twice the dense "
                  "peak on logical k); `none` off the tensor pipe."), "def Kind.form : Kind → Option (Ty × ℝ)"]
    for k in kinds:
        f = s.kinds[k].form
        half = " / 2" if f and f.sparse else ""
        out.append(f"  | .{k} => " + (f"some (.{f.type}, {f.m} * {f.n} * {f.k}{half} / {f.warps})" if f else "none"))
    out += ["", *_doc("The pipe each class issues to; `none` on the tensor pipe or issue only."),
            "def Kind.pipe : Kind → Option Pipe"]
    for k in kinds:
        q = s.kinds[k].pipe
        out.append(f"  | .{k} => " + (f"some .{q}" if q else "none"))
    out += ["", *_doc("A6's ω for each tensor class: the 32-bit words a warp "
                      "instruction's accumulator fragment takes, m · n elements over the warps that issue one MMA, an "
                      "element one register and an f64 element two (DMMA, `mma.sync .f64`), a 2:4 sparse form's as "
                      "its dense twin's; `none` off the tensor pipe. Grade: "
                      f"{s.omega[0]}. Source: {s.omega[1]}."), "def Kind.omega : Kind → Option ℝ"]
    for k in kinds:
        f = s.kinds[k].form
        e = s.peaks[f.type].element_words if f else 1
        out.append(f"  | .{k} => " + (f"some ({f.m} * {f.n}{f' * {e}' if e != 1 else ''} / {f.warps})" if f
                                       else "none"))
    shared = [k for k in kinds if k.startswith(("LDS", "STS", "ATOMS"))]
    smem = (f"no class on the shared-memory row (the census's shared-memory classes, {', '.join(shared)}, are issue "
            "only, so A5 is 0 for each)" if shared else
            "no shared-memory class (no LDS kind in the census, so A5 is 0 for each class)")
    out += ["", *_doc(f"**The {s.name}'s data sheet**: f_max in Hz, each class's MACs and its type's peak (`0` off the "
                      f"tensor pipe), its pipe, each pipe's width, and {smem}."),
            "def sheet : Sheet Kind Pipe where",
            "  fmax := fmaxMHz * 10 ^ 6",
            "  macs k := match k.form with",
            "    | some (_, m) => m",
            "    | none => 0",
            "  peak k := match k.form with",
            "    | some (t, _) => t.peak",
            "    | none => 0",
            "  pipe := Kind.pipe",
            "  width := Pipe.width",
            "  smem _ := 0",
            ""]
    for c, v in s.clocks.items():
        under = v.value < s.fmax.value
        out += [*_doc(f"The {s.labels[c]} is {'under' if under else 'at most'} f_max."),
                f"theorem {clock[c]}_{'lt' if under else 'le'}_fmaxMHz : {clock[c]} {'<' if under else '≤'} fmaxMHz "
                ":= by", f"  norm_num [{clock[c]}, fmaxMHz]", ""]
    for ty, p in s.peaks.items():
        dense, tol = p.dense
        figure = (f"{_dec(p.sheet)} with sparsity, dense {_dec(dense)} ± {_dec(tol)}" if p.sparse
                  else f"dense {_dec(dense)} ± {_dec(tol)}")
        lhs = f"2 * {peak[ty]} * smCount * ({clock[p.clock]} * 10 ^ 6)"
        c = s.clocks[p.clock]
        unstated = [what for what, x in [(f"the `{p.clock}` clock", c), ("the SM count", s.sms)] if not _stated(x)]
        head = (f"A3's {ty} peak reproduces the document" if not unstated else
                f"A3's {ty} peak is consistent with the document's figure, not confirmed by it")
        tail = ("" if not unstated else
                f" No document states {' or '.join(unstated)} (their grades above), so this holds the per-clock peak, "
                "the SMs and the clock to the figure together, not each to a document.")
        out += [*_doc(f"{head}: 2 · {_dec(p.macs)} MACs · {_dec(s.sms.value)} SMs at the {s.labels[p.clock]}, "
                      f"{_dec(c.value)} MHz, is its figure, {figure}, in 10¹² a second.{tail}"),
                f"theorem {peak[ty]}_figure :",
                f"    ({_num(dense)} - {_num(tol)} : ℝ) * 10 ^ 12 ≤ {lhs} ∧",
                f"      {lhs} ≤ ({_num(dense)} + {_num(tol)} : ℝ) * 10 ^ 12 := by",
                f"  norm_num [{peak[ty]}, smCount, {clock[p.clock]}]", ""]
    out += ["end", "", f"end Accounting.TS.Sheets.{s.name}", ""]
    return "\n".join(out)
