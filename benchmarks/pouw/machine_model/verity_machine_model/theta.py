"""θ, the accelerator model's parameters for one device (`theta.toml` in its directory): every rate, capacity and clock
the model reads, each with the vendor document that states it (`sheet`, `grade`, `source`) and then the census or a
measured run (`measured`, `runs`), or the run that would measure it (`gap`); the card's exact paths; its FP8 atoms'
semantics; and the units that write registers (`[[writer]]`, with the census's joint issue rows, `[[joint]]`).

Each derivation is read twice: on the data sheet (every rate its `sheet` value) and measured (every rate its measured
value, and the sheet's where no run has one, which the result lists as filled).

- `routes`: d, one delivery (a product's value added into a second accumulator) in INT8 MAC units, ρ₈ over the route's
  deliveries a SM-clock, by each route. The shared-memory and shuffle routes price the move alone.
- `strassen`: Strassen one level deep on R·Uᵀ inside each L-segment. A transcript word is one segment's dot product and
  the fold reads each word nonlinearly, so the level splits each segment's k into halves: 7 products of k = L/2 for 8,
  and 5 of the 7 feed two output blocks, 5/(4L) extra deliveries a naive MAC. Its exchange rate is the delivery
  route's MAC-times a tensor MAC-time saved, 5d/(4L) over 1 − (7/8)·ρ_L/ρ_{L/2}, which is 10d/L when the k = L/2 form
  runs at the k = L form's rate; when it runs slower the level saves no tensor time, and there is no exchange. A
  product whose k is a multiple of 32 runs as k32 MMAs (`form`), so at the no-fold receipt's L (`NOFOLD_L`) it's 10d/L.
- `seed`: mma.sync with D ≠ C delivers its C into a new accumulator with the products it computes anyway, at no
  marginal price while those are needed; charged its issue slot, d = ρ₈ / (issue · 128).
- `tick`: a tensor-over-lane ratio per second over per SM-clock, the tensor figures' clock over the lanes'; every d,
  exchange rate and c above times it is that figure per second.
- `bound`: B_RF, the most register bits all units together write a SM-clock, an LP over the writers' warp instructions
  a SM-clock under the issue row, each pipe and, measured, the joint issue rows; c ≈ ρ₈/B_RF, since the transcript is
  one bit a MAC at L = 32 and the fold reads each word from a register. Measured, it is also read with the unmeasured
  writers left out (`measured only`).
- `folds`: for each honest profile `honest` lists (duo's and LI's), its lane instructions a committed word, r_H,
  against the fold's own, r_min, and the slack r_H − r_min in tensor MAC-times a MAC over the cheapest adding route's
  exchange rate: the ε that slack would buy if ε were slack over exchange. A measured profile also gives γ, its
  SMSP-cycles a MMA over the lone INT8 MMA's at the measured rate.
- `units`: every arithmetic or reduction unit RT-UNITS lists (`UNITS`), each with its verdict: a priced rate, an adding
  route d reads, bounded outside the model (its most work a SM-clock over the rate it stands in for, summed by `free`),
  open (with the run that would settle it), none, a clock or bandwidth scale, or absent.
- `sha`: SHA-512, our measured compressions a SM-clock against each public implementation's figure (`[[public]]`) read
  at its card's highest clock, the lowest per-clock reading it can have; ours over it, at most 1, is c_sha's reading.
  `sha_ceiling`: the most a compression of `[sha512_mix]` (the built kernel's SASS) gets from the card's pipes, an LP
  over ALU, IMAD and issue with RT-UNITS' three exact moves to IMAD (a rotation as IMAD.HI + IMAD + IMAD.WIDE, or as
  SHF + IMAD + IMAD.WIDE; a 2-input 64-bit add as IMAD.WIDE + IMAD), with and without IMAD.HI: the rate a `ShaSpeed`
  premise is stated at, since no schedule built from those moves beats it.

`figures` holds each rate's `figure`, the per-second number its document prints at `clock`, to 2 · sheet · SMs · clock.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from scipy.optimize import linprog

from verity_machine_model import eps as E

THETA = "theta.toml"
LANES = 32
WORD = 32                   # bits a register word
SMSPS = 4                   # a joint issue row's SMSP-cycles a SM-clock
TOL = 0.005                 # a figure holds within 0.5%: the documents round, and a DS prints the sparse figure
INT8 = "tensor.s8.k32"
READINGS = ("sheet", "measured")
NOFOLD_L = (1024, 8192, 16384)  # segment lengths of the no-fold receipt's γ figures (L ≤ K, K 16384 to 65536)
FOLD_RATE = {"ALU": "alu.lop3", "IMAD": "imad", "SHFL": "shfl"}   # the lane rate each fold class issues at
ADDING = ("IADD3, two deliveries a lane instruction", "IADD3, one delivery a lane instruction",
          "IMAD a·1 + c, one delivery a lane instruction", "IADD3 on the ALU pipe and IMAD beside it",
          "FFMA a·1 + c, one delivery a lane instruction (exact below 2^24)",
          "ATOMS.ADD into shared memory, one delivery a lane",
          "RED.ADD into global memory at the L2, one delivery a lane",
          "red.shared::cluster.add into a cluster peer's shared memory (DSMEM), one delivery a lane",
          "cp.reduce.async.bulk .add, shared memory into global at the L2 (TMA: one instruction a bulk)",
          "multimem.red .add through the NVSwitch (NVLink SHARP)",
          "IADD3 and IMAD under issue, with TMA bulk reductions beside them")
REDUCTIONS = {"smem.atom": ADDING[5], "l2.red": ADDING[6], "dsmem.red": ADDING[7], "tma.red": ADDING[8],
              "nvls.red": ADDING[9]}
# A 2-input 64-bit add as the kernel was built: its ALU instructions, IMAD instructions and instructions.
ADD_FORMS = {"IADD3 + IADD3.X": (2, 0, 2), "IADD3 + IMAD.X": (1, 1, 2), "IADD.64": (1, 0, 1)}
# Every arithmetic or reduction unit RT-UNITS lists; each θ has a `[[unit]]` for each, with its verdict.
UNITS = ("tensor", "dp4a", "imad", "alu", "fp32", "fp64", "sfu", "uniform", "texture", "smem.atomics", "l2.atomics",
         "tma.reduce", "dsmem", "nvlink.reduce", "copy", "video", "rt", "clocks", "compression", "host.link", "host.cpu")
VERDICTS = {
    "priced": "a θ rate the model prices",
    "route": "an adding route d reads (an upper bound where no run has timed it)",
    "bounded": "outside the model, bounded on documents",
    "open": "outside the model, with a loose bound or none, and the run that would settle it",
    "none": "does no exact arithmetic on a word or a SHA-512 compression",
    "scale": "computes nothing: scales per-second figures, not θ's per-SM-clock ratios",
    "absent": "not on this card",
}


@dataclass(frozen=True)
class Clock:
    name: str
    mhz: float
    grade: str
    source: str
    measured: str
    measured_mhz: float | None = None   # the clock a measured reading runs at, where a run gives one


@dataclass(frozen=True)
class Rate:
    key: str
    what: str
    unit: str
    sheet: float | None
    clock: str | None       # the clock (`[clocks.NAME]`) the document's per-second figure is at
    figure: float | None    # that figure, dense
    grade: str
    source: str
    measured: float | None
    runs: str
    gap: str


@dataclass(frozen=True)
class Capacity:
    key: str
    what: str
    value: float
    unit: str
    grade: str
    source: str
    measured: str
    gap: str


@dataclass(frozen=True)
class Writer:
    name: str
    rate: str               # the rate whose value over `per` is the class's warp instructions a SM-clock
    per: float              # MACs, lanes or bytes a warp instruction, in the rate's unit
    words: int              # 32-bit registers a warp instruction writes
    pipe: str
    joint: dict[str, float]  # SMSP-cycles a warp instruction on each joint row that names the class


@dataclass(frozen=True)
class Joint:
    name: str
    other: float            # SMSP-cycles a warp instruction of every class the row doesn't name
    source: str


@dataclass(frozen=True)
class Unit:
    key: str
    what: str
    verdict: str
    rates: tuple[str, ...]  # the θ rates it is or rests on
    bound: float | None     # most work it adds a SM-clock, in `against`'s unit
    against: str | None     # the θ rate that work would stand in for
    why: str
    gap: str


@dataclass(frozen=True)
class Public:
    impl: str               # the public implementation and its input form
    form: str               # generic (a block the last compression's digest varies) or structured
    gps: float              # 10⁹ SHA-512 compressions a second
    card: str
    sms: int                # the card's SMs, which needn't be θ's
    mhz: float              # the card's highest clock: the per-SM-clock reading at it is the lowest
    source: str


@dataclass(frozen=True)
class Theta:
    gpu: str
    sms: int
    sms_source: str
    honest: tuple[str, ...]  # the honest profiles r_H is read from, the first the one Strassen's L is
    documents: tuple[str, ...]
    clocks: dict[str, Clock]
    rates: dict[str, Rate]
    capacities: dict[str, Capacity]
    paths: tuple[dict, ...]
    fp8: tuple[dict, ...]
    writers: tuple[Writer, ...]
    joints: dict[str, Joint]
    units: dict[str, Unit] = field(default_factory=dict)
    public: tuple[Public, ...] = ()
    sha_mix: dict | None = None     # `[sha512_mix]`: a compression's lane instructions and the ones IMAD can take

    def at(self, key: str, reading: str) -> tuple[float | None, bool]:
        """A rate on the sheet or measured, and whether a measured reading took the sheet's value for want of a run."""
        r = self.rates[key]
        if reading == "sheet" or r.measured is not None:
            return (r.sheet if reading == "sheet" else r.measured), False
        return r.sheet, r.sheet is not None


def has(data: Path) -> bool:
    return (data / THETA).is_file()


def theta(data: Path) -> Theta:
    p = data / THETA
    if not p.is_file():
        raise SystemExit(f"{data}: no {THETA}; the devices with one are sm_120 (data/) and sm_90")
    t = tomllib.loads(p.read_text())
    clocks = {c: Clock(c, v["mhz"], v["grade"], v["source"], v.get("measured", ""), v.get("measured_mhz"))
              for c, v in t["clocks"].items()}
    rates: dict[str, Rate] = {}
    for v in t["rate"]:
        if v["key"] in rates:
            raise SystemExit(f"{p}: two rates {v['key']}")
        rates[v["key"]] = Rate(v["key"], v["what"], v["unit"], v.get("sheet"), v.get("clock"), v.get("figure"),
                               v["grade"], v["source"], v.get("measured"), v.get("runs", ""), v.get("gap", ""))
    caps = {v["key"]: Capacity(v["key"], v["what"], v["value"], v["unit"], v["grade"], v["source"],
                               v.get("measured", ""), v.get("gap", "")) for v in t["capacity"]}
    writers = tuple(Writer(v["name"], v["rate"], v["per"], v["words"], v["pipe"], dict(v.get("joint", {})))
                    for v in t.get("writer", ()))
    joints = {v["name"]: Joint(v["name"], v["other"], v["source"]) for v in t.get("joint", ())}
    units: dict[str, Unit] = {}
    for v in t.get("unit", ()):
        if v["key"] in units:
            raise SystemExit(f"{p}: two units {v['key']}")
        units[v["key"]] = Unit(v["key"], v["what"], v["verdict"], tuple(v.get("rates", ())), v.get("bound"),
                               v.get("against"), v["why"], v.get("gap", ""))
    public = tuple(Public(v["impl"], v["form"], v["gps"], v["card"], v["sms"], v["mhz"], v["source"])
                   for v in t.get("public", ()))
    th = Theta(gpu=t["gpu"], sms=t["sms"], sms_source=t["sms_source"], honest=tuple(t["honest"]),
               documents=tuple(t["documents"]), clocks=clocks, rates=rates, capacities=caps,
               paths=tuple(t.get("path", ())), fp8=tuple(t.get("fp8", ())), writers=writers, joints=joints,
               units=units, public=public, sha_mix=t.get("sha512_mix"))
    if problems := check(th):
        raise SystemExit(f"{p}:\n  " + "\n  ".join(problems))
    return th


def check(t: Theta) -> list[str]:
    """What breaks the file's rules: a rate is on the sheet exactly when its grade isn't `none`, has a source, and has a
    measured value with its runs or a gap; a figure has its clock; a writer reads a rate and joint rows that exist."""
    out = []
    for k, r in t.rates.items():
        if (r.sheet is None) != (r.grade == "none"):
            out.append(f"{k}: a rate with a `sheet` value has its document's grade, and one without has grade 'none'")
        if not r.source:
            out.append(f"{k}: no source; a rate no document states says so in `source`")
        if (r.measured is None) == (not r.gap):
            out.append(f"{k}: give it a measured value with its runs, or a gap naming the run that would measure it")
        if r.measured is not None and not r.runs:
            out.append(f"{k}: measured without runs")
        if r.figure is not None and (r.clock not in t.clocks or r.sheet is None):
            out.append(f"{k}: a figure needs a sheet value and a clock of [clocks]")
    for c in t.capacities.values():
        if c.value <= 0 or not c.source:
            out.append(f"capacity {c.key}: a positive value and a source")
    for w in t.writers:
        if w.rate not in t.rates:
            out.append(f"writer {w.name}: no rate {w.rate}")
        if bad := set(w.joint) - set(t.joints):
            out.append(f"writer {w.name}: no joint row {', '.join(sorted(bad))}")
    for k in ("issue", INT8):
        if k not in t.rates:
            out.append(f"no rate {k}; every derivation reads it")
    if not t.honest:
        out.append("`honest` lists no profile; r_H and Strassen's L read one")
    if missing := [k for k in UNITS if k not in t.units]:
        out.append(f"no [[unit]] for {', '.join(missing)}: every unit RT-UNITS lists has one, `absent` if the card "
                   f"lacks it")
    for u in t.units.values():
        if u.verdict not in VERDICTS:
            out.append(f"unit {u.key}: verdict {u.verdict!r} is none of {', '.join(VERDICTS)}")
        if bad := [r for r in u.rates if r not in t.rates]:
            out.append(f"unit {u.key}: no rate {', '.join(bad)}")
        if u.verdict in ("priced", "route") and not u.rates:
            out.append(f"unit {u.key}: a {u.verdict} unit names its rates")
        if u.verdict == "route" and not set(u.rates) & set(REDUCTIONS):
            out.append(f"unit {u.key}: a route unit names one of {', '.join(REDUCTIONS)}")
        if u.verdict == "bounded" and (u.bound is None or u.against not in t.rates):
            out.append(f"unit {u.key}: a bounded unit gives `bound` and the rate it is `against`")
        if u.verdict == "open" and not u.gap:
            out.append(f"unit {u.key}: an open unit names the run that would settle it in `gap`")
        if u.against is not None and u.against not in t.rates:
            out.append(f"unit {u.key}: no rate {u.against}")
    if t.public and "sha512" not in t.rates:
        out.append("[[public]] SHA-512 figures need the `sha512` rate they're read against")
    if t.sha_mix is not None:
        if "sha512" not in t.rates:
            out.append("[sha512_mix] needs the `sha512` rate its ceiling is read against")
        m = t.sha_mix
        if missing := [k for k in ("alu", "rotates", "adds", "add_form", "overlap", "source") if k not in m]:
            out.append(f"[sha512_mix] lacks {', '.join(missing)}")
        elif m["add_form"] not in ADD_FORMS:
            out.append(f"[sha512_mix] add_form {m['add_form']!r} is none of {', '.join(ADD_FORMS)}")
        else:
            a, i, _ = ADD_FORMS[m["add_form"]]
            if 2 * m["rotates"] + a * m["adds"] > m["alu"]:
                out.append(f"[sha512_mix]: the movable rotates (two SHF each) and adds ({m['add_form']}) take more "
                           f"than `alu`")
            if i * m["adds"] > m.get("imad", 0):
                out.append(f"[sha512_mix]: {m['adds']} adds as {m['add_form']} need that many IMAD in `imad`")
    return out


# --- figures ----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Figure:
    key: str
    clock: str
    mhz: float
    value: float            # 2 · sheet · SMs · clock, 10¹² a second
    figure: float

    @property
    def holds(self) -> bool:
        return abs(self.value - self.figure) <= TOL * self.figure


def figures(t: Theta) -> list[Figure]:
    out = []
    for k, r in t.rates.items():
        if r.figure is not None:
            mhz = t.clocks[r.clock].mhz
            out.append(Figure(k, r.clock, mhz, 2 * r.sheet * t.sms * mhz * 1e6 / 1e12, r.figure))
    return out


# --- d, by route -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Route:
    name: str
    reading: str
    per_clk: float          # deliveries a SM-clock
    rho8: float
    how: str                # the deliveries' arithmetic
    filled: tuple[str, ...] = ()

    @property
    def d(self) -> float:
        return self.rho8 / self.per_clk


def routes(t: Theta, reading: str) -> list[Route]:
    rho8, f8 = t.at(INT8, reading)
    out = []

    def route(name: str, keys: tuple[str, ...], per_clk, how) -> None:
        vals = [t.at(k, reading) if k in t.rates else (None, False) for k in keys]
        if all(v is not None for v, _ in vals):
            xs = [v for v, _ in vals]
            out.append(Route(name, reading, per_clk(*xs), rho8, how(*xs),
                             tuple(k for k, (_, f) in zip(keys, vals) if f) + ((INT8,) if f8 else ())))

    route(ADDING[0], ("alu.iadd3",), lambda a: 2 * a, lambda a: f"2 · alu.iadd3 {a:g}")
    route(ADDING[1], ("alu.iadd3",), lambda a: a, lambda a: f"alu.iadd3 {a:g}")
    route(ADDING[2], ("imad",), lambda i: i, lambda i: f"imad {i:g}")

    def both(a: float, i: float, n: float) -> float:
        xa = min(a / LANES, n)
        return LANES * (2 * xa + min(i / LANES, n - xa))
    route(ADDING[3], ("alu.lop3", "imad", "issue"), both,
          lambda a, i, n: f"32 · (2 · {min(a / LANES, n):.4g} + {min(i / LANES, n - min(a / LANES, n)):.4g}) warp "
                          f"instructions of ALU at alu.lop3 {a:g} and IMAD at {i:g}, issue {n:g}")
    route(ADDING[4], ("ffma",), lambda f: f, lambda f: f"ffma {f:g}")
    for key, name in REDUCTIONS.items():
        route(name, (key,), lambda r: r, lambda r, key=key: f"{key} {r:g}")
    route(ADDING[10], ("alu.lop3", "imad", "issue", "tma.red"), lambda a, i, n, m: both(a, i, n) + m,
          lambda a, i, n, m: f"{both(a, i, n):.4g} under issue {n:g} + tma.red {m:g}, which takes no lane")
    route("shared memory, STS then LDS of a word (the move alone)", ("smem.write", "smem.read"),
          lambda w, r: 1 / (4 / w + 4 / r), lambda w, r: f"words, 1 / (4/smem.write {w:g} + 4/smem.read {r:g})")
    route("SHFL, a word a lane (the move alone)", ("shfl",), lambda s: s, lambda s: f"shfl {s:g}")
    return out


def seed(t: Theta, reading: str) -> float:
    """mma.sync with D ≠ C writes its C, plus the products it computes anyway, into a new accumulator: a delivery of 128
    words (the m16n8 tile) at no marginal price while those products are needed. Charged its issue slot alone, its d."""
    return t.at(INT8, reading)[0] / (t.at("issue", reading)[0] * 128)


def tick(t: Theta, reading: str) -> float:
    """Per second over per SM-clock for a ratio of a tensor rate to a lane rate: the tensor figures' clock over the
    lanes' (`[clocks.tensor]` against `[clocks.fmax]`; measured, the tensor clock a saturated run reaches). 1 on a card
    whose figures share one clock."""
    if "tensor" not in t.clocks:
        return 1.0
    c = t.clocks["tensor"]
    return (c.measured_mhz if reading == "measured" and c.measured_mhz else c.mhz) / t.clocks["fmax"].mhz


# --- Strassen's exchange rate ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Strassen:
    reading: str
    L: int
    rho_l: float            # INT8 MACs a SM-clock at k = L, the honest form's
    rho_half: float         # at k = L/2, the products'
    filled: tuple[str, ...]
    rates: dict[str, float] = field(default_factory=dict)   # each adding route's exchange rate, when the level saves

    @property
    def tensor(self) -> float:
        """The level's tensor time over naive's: 7/8 of the MACs at the k = L/2 form's rate."""
        return 7 / 8 * self.rho_l / self.rho_half

    @property
    def extra(self) -> float:
        """Extra deliveries a naive MAC: 5 of 7 products feed two blocks, each n³/(4L) words of the level's n³/L."""
        return 5 / (4 * self.L)


def form(k: int) -> str:
    """The INT8 rate a k-long product runs at: k32 MMAs end to end once k is a multiple of 32, else its own form."""
    return f"tensor.s8.k{32 if k % 32 == 0 else k}"


def strassen(t: Theta, reading: str, L: int) -> Strassen:
    if L % 2:
        raise SystemExit(f"L = {L}: one Strassen level halves k, so L is even")
    keys = (form(L), form(L // 2))
    if missing := [k for k in keys if k not in t.rates]:
        raise SystemExit(f"{t.gpu}: no rate {', '.join(missing)}, which Strassen at L = {L} reads")
    (rl, fl), (rh, fh) = (t.at(k, reading) for k in keys)
    s = Strassen(reading, L, rl, rh, tuple(k for k, f in zip(keys, (fl, fh)) if f))
    if s.tensor < 1:
        s.rates.update({r.name: s.extra * r.d / (1 - s.tensor) for r in routes(t, reading) if r.name in ADDING})
    return s


# --- B_RF ----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Bound:
    reading: str            # sheet, measured, or measured only
    words: float            # B_RF, register words a SM-clock
    rho8: float
    mix: dict[str, float]   # each writer's warp instructions a SM-clock at the optimum
    filled: tuple[str, ...]
    dropped: tuple[str, ...]

    @property
    def bits(self) -> float:
        return WORD * self.words

    @property
    def c(self) -> float:
        return self.rho8 / self.bits


def bound(t: Theta, reading: str) -> Bound:
    if reading not in (*READINGS, "measured only"):
        raise SystemExit(f"B_RF reads one of sheet, measured, measured only, not {reading}")
    base = "sheet" if reading == "sheet" else "measured"
    ws, caps, filled, dropped = [], [], [], []
    for w in t.writers:
        v, f = t.at(w.rate, base)
        if v is None or (f and reading == "measured only"):
            dropped.append(w.name)
            continue
        ws.append(w)
        caps.append(v / w.per)
        if f:
            filled.append(w.rate)
    issue, _ = t.at("issue", base)
    A, b = [[1.0] * len(ws)], [issue]
    for p in dict.fromkeys(w.pipe for w in ws):
        A.append([1 / c if w.pipe == p else 0.0 for w, c in zip(ws, caps)])
        b.append(1.0)
    if base == "measured":
        for j in t.joints.values():
            A.append([w.joint.get(j.name, j.other) for w in ws])
            b.append(SMSPS)
    s = linprog([-w.words for w in ws], A_ub=A, b_ub=b, bounds=[(0, None)] * len(ws), method="highs")
    if s.status != 0:
        raise SystemExit(f"B_RF's LP ({reading}): {s.message}")
    rho8, f8 = t.at(INT8, base)
    return Bound(reading, -s.fun, rho8, {w.name: x for w, x in zip(ws, s.x) if x > 1e-9},
                 tuple(dict.fromkeys(filled + ([INT8] if f8 else []))), tuple(dropped))


# --- r_H against r_min -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    profile: str
    status: str
    fold: str
    L: int
    per_word: dict[str, float]  # the honest kernel's lane instructions a committed word, by fold class
    minimal: dict[str, float]   # the fold's own: r_min's
    slack: dict[str, float]     # by reading: (r_H − r_min) in tensor MAC-times a MAC
    exchange: dict[str, float | None]  # by reading: the cheapest adding route's exchange rate, None if Strassen saves none
    lone: float                 # the lone INT8 MMA's SMSP-cycles at the measured rate: words · depth · SMSPs / ρ₈
    measured: float | None      # the profile's measured SMSP-cycles a MMA, None if provisional
    measured_source: str

    @property
    def r_h(self) -> float:
        return sum(self.per_word.values())

    @property
    def r_min(self) -> float:
        return sum(self.minimal.values())

    @property
    def gamma(self) -> float | None:
        return None if self.measured is None else self.measured / self.lone

    def eps(self, reading: str) -> float:
        """Slack over exchange: 0 when Strassen frees no tensor time to trade for."""
        x = self.exchange[reading]
        return 0.0 if x is None else self.slack[reading] / x


def folds(t: Theta, data: Path) -> list[Fold]:
    return [fold(t, data, name) for name in t.honest]


def fold(t: Theta, data: Path, name: str) -> Fold:
    h = E.honest(name, data)
    words = h.words * h.depth / h.L
    per_word = {c: h.per_mma.get(c, 0.0) * LANES / words for c in h.fold}
    if missing := set(h.fold) - set(FOLD_RATE):
        raise SystemExit(f"{h.name}: fold class {', '.join(sorted(missing))} has no lane rate in FOLD_RATE")
    slack, exchange = {}, {}
    for reading in READINGS:
        rho8, _ = t.at(INT8, reading)
        slack[reading] = sum((per_word[c] - h.fold[c]) * rho8 / t.at(FOLD_RATE[c], reading)[0] / h.L for c in h.fold)
        rates = strassen(t, reading, h.L).rates
        exchange[reading] = min(rates.values()) if rates else None
    lone = h.words * h.depth * SMSPS / t.at(INT8, "measured")[0]
    return Fold(h.name, h.status, h.fold_name, h.L, per_word, dict(h.fold), slack, exchange, lone, h.measured,
                h.measured_source)


# --- units outside the priced pipes, and SHA-512 -----------------------------------------------------------------------

def share(t: Theta, u: Unit, reading: str) -> float | None:
    """A bounded or open unit's bound over the rate it stands in for, None when either is missing."""
    if u.bound is None or u.against is None:
        return None
    v, _ = t.at(u.against, reading)
    return None if v is None else u.bound / v


def free(t: Theta, reading: str) -> dict[str, float]:
    """The bounded units' shares summed by the rate they stand in for: the most work outside the model adds."""
    out: dict[str, float] = {}
    for u in t.units.values():
        if u.verdict == "bounded" and (s := share(t, u, reading)) is not None:
            out[u.against] = out.get(u.against, 0.0) + s
    return out


@dataclass(frozen=True)
class ShaReading:
    public: Public
    per_clk: float          # the public figure's compressions a SM-clock at its card's highest clock
    c: float                # ours over it, at most 1: c_sha's reading against this figure


def sha(t: Theta) -> list[ShaReading]:
    ours, _ = t.at("sha512", "measured")
    out = []
    for p in t.public:
        per_clk = p.gps * 1e9 / (p.sms * p.mhz * 1e6)
        out.append(ShaReading(p, per_clk, min(1.0, ours / per_clk)))
    return out


@dataclass(frozen=True)
class ShaCeiling:
    reading: str
    built: float            # SM-clocks a compression as the kernel was built (no move)
    clocks: float           # the LP's SM-clocks a compression
    hi_rotates: float       # rotations moved as IMAD.HI + IMAD + IMAD.WIDE at the optimum
    shf_rotates: float      # rotations moved as SHF + IMAD + IMAD.WIDE
    adds: float             # 2-input 64-bit adds moved to IMAD.WIDE + IMAD
    hi: bool                # IMAD.HI allowed

    @property
    def rate(self) -> float:
        return 1 / self.clocks


def sha_ceiling(t: Theta, reading: str, hi: bool = True) -> ShaCeiling | None:
    """The fewest SM-clocks a compression of `[sha512_mix]` takes on the card's pipes: an LP over ALU, IMAD and issue in
    which a rotation or shift (two SHF) may go to IMAD as IMAD.HI + IMAD + IMAD.WIDE (ALU −2, one more instruction) or
    SHF + IMAD + IMAD.WIDE (ALU −1, one more), each a sum of disjoint bit ranges, so no carry, and a 2-input 64-bit add
    (built as `add_form`) as IMAD.WIDE + IMAD. Without `hi` no rotation uses IMAD.HI. None without a mix."""
    m = t.sha_mix
    if m is None:
        return None
    a, i, h, w = (t.at(k, reading)[0] for k in ("alu.lop3", "imad", "imad.hi", "imad.wide"))
    lanes = LANES * t.at("issue", reading)[0]
    alu, imad, n = m["alu"], m.get("imad", 0), m["alu"] + m.get("imad", 0) + m.get("other", 0)
    add_alu, add_imad, add_n = ADD_FORMS[m["add_form"]]
    rows = [[-2 / a, -1 / a, -add_alu / a, -1],                                   # over (R-hi, R-shf, adds, T)
            [1 / h + 1 / i + 1 / w, 1 / i + 1 / w, 1 / w + 1 / i - add_imad / i, -1],
            [1 / lanes, 1 / lanes, (2 - add_n) / lanes, -1],
            [1, 1, 0, 0]]
    rhs = [-alu / a, -imad / i, -n / lanes, m["rotates"]]
    bounds = [(0, m["rotates"] if hi else 0), (0, m["rotates"]), (0, m["adds"]), (0, None)]
    s = linprog([0, 0, 0, 1], A_ub=rows, b_ub=rhs, bounds=bounds, method="highs")
    if s.status != 0:
        raise SystemExit(f"SHA-512's LP ({reading}): {s.message}")
    return ShaCeiling(reading, max(alu / a, imad / i, n / lanes), s.x[3], s.x[0], s.x[1], s.x[2], hi)
