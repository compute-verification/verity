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
  runs at the k = L form's rate; when it runs slower the level saves no tensor time, and there is no exchange.
- `bound`: B_RF, the most register bits all units together write a SM-clock, an LP over the writers' warp instructions
  a SM-clock under the issue row, each pipe and, measured, the joint issue rows; c ≈ ρ₈/B_RF, since the transcript is
  one bit a MAC at L = 32 and the fold reads each word from a register. Measured, it is also read with the unmeasured
  writers left out (`measured only`).
- `fold`: the honest kernel's ALU and IMAD lane instructions a committed word, r_H (its honest profile), against the
  fold's own, r_min, and the ALU slack r_H − r_min in tensor MAC-times a MAC over the cheapest adding route's exchange
  rate: the ε that slack would buy if ε were slack over exchange.

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
FOLD_RATE = {"ALU": "alu.lop3", "IMAD": "imad"}   # the lane rate each fold class issues at: its pipe's width
ADDING = ("IADD3, two deliveries a lane instruction", "IADD3, one delivery a lane instruction",
          "IMAD a·1 + c, one delivery a lane instruction", "IADD3 on the ALU pipe and IMAD beside it")


@dataclass(frozen=True)
class Clock:
    name: str
    mhz: float
    grade: str
    source: str
    measured: str


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
class Theta:
    gpu: str
    sms: int
    sms_source: str
    honest: str
    documents: tuple[str, ...]
    clocks: dict[str, Clock]
    rates: dict[str, Rate]
    capacities: dict[str, Capacity]
    paths: tuple[dict, ...]
    fp8: tuple[dict, ...]
    writers: tuple[Writer, ...]
    joints: dict[str, Joint]

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
    clocks = {c: Clock(c, v["mhz"], v["grade"], v["source"], v.get("measured", "")) for c, v in t["clocks"].items()}
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
    th = Theta(gpu=t["gpu"], sms=t["sms"], sms_source=t["sms_source"], honest=t["honest"],
               documents=tuple(t["documents"]), clocks=clocks, rates=rates, capacities=caps,
               paths=tuple(t.get("path", ())), fp8=tuple(t.get("fp8", ())), writers=writers, joints=joints)
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
    route("shared memory, STS then LDS of a word (the move alone)", ("smem.write", "smem.read"),
          lambda w, r: 1 / (4 / w + 4 / r), lambda w, r: f"words, 1 / (4/smem.write {w:g} + 4/smem.read {r:g})")
    route("SHFL, a word a lane (the move alone)", ("shfl",), lambda s: s, lambda s: f"shfl {s:g}")
    return out


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


def strassen(t: Theta, reading: str, L: int) -> Strassen:
    if L % 2:
        raise SystemExit(f"L = {L}: one Strassen level halves k, so L is even")
    keys = (f"tensor.s8.k{L}", f"tensor.s8.k{L // 2}")
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

    @property
    def r_h(self) -> float:
        return sum(self.per_word.values())

    @property
    def r_min(self) -> float:
        return sum(self.minimal.values())

    def eps(self, reading: str) -> float:
        """Slack over exchange: 0 when Strassen frees no tensor time to trade for."""
        x = self.exchange[reading]
        return 0.0 if x is None else self.slack[reading] / x


def fold(t: Theta, data: Path) -> Fold:
    h = E.honest(t.honest, data)
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
    return Fold(h.name, h.status, h.fold_name, h.L, per_word, dict(h.fold), slack, exchange)
