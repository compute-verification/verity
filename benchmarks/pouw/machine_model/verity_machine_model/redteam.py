"""Red-team audit of a headline PoUW design against the machine model.

ε is proved conditional on every facet of Π being a true ceiling (eps.py, facet_break/). This module reads, from a
device's committed census alone, how much room an adversary has to beat one:

- `audit` solves the cheater's LP at the measured reading (operands free, credit floor, q* = 1.22) and reports the
  binding rows, the rows within `NEAR` of binding (the facets a silicon beat would unseat first), and the margin
  sensitivity of each: how far a row's measured lane coefficient must move in the cheater's favour before the core ε
  rises (`flip`). The `issue` ceiling (one warp instruction a SMSP-cycle, the hardware issue width) is
  flagged: no silicon beats it.
- `margin_audit` runs the joint margin (every near measured facet moved `m` in the cheater's favour at once,
  default 2%) and names the single row that drives it. A facet's *measured lane coefficient* is the one its CUDA lane
  classes share, so the move leaves the tensor-pipe MMA value alone; the issue slot and the exact operand floor are
  axioms, not measured rows, so they are left alone too. Moving a whole row's column (the MMA coefficient included)
  mis-prices the tensor route, so the margin never moves it.
- `least_filler` finds the least verified filler a word (in a class mix) that holds a margin: the cheapest add-on that
  brings the joint margin's worst core back to the design's floor.
- `uniform_gap` bounds a device whose census has no uniform datapath (no UNI class, as sm_80 has none). A cheater that
  runs the fold on the uniform datapath moves it off a bound pipe; crediting the datapath at a reference device's
  measured rates (sm_90's by default) bounds the ε this opens — the gap an adversary with a uniform-datapath filler
  exploits until the device is measured.

No GPU: this reads the committed per-device census (`data/devices/sm_NN/`). It never refits a row or writes data; a beat
found on silicon is adopted the usual way (facet_break's `refit`, then the device's rows.csv and census.csv).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from verity_machine_model import eps as E

NEAR = 0.90                        # a row within this fraction of binding is a near facet
MARGIN_NEAR = 0.95                # the margin audit moves the rows this near
ISSUE_CEILING = "issue"           # 1 warp instruction a SMSP-cycle: the hardware issue width, not a facet


@dataclass(frozen=True)
class Row:
    name: str
    load: float                   # the cheater's load on the row over T (1.0: binding)
    dual: float                   # the certificate's weight on the row (0 off the facet)
    ceiling: bool                 # the issue ceiling: silicon cannot beat it
    flip_pct: float | None        # the move in the cheater's favour (per cent) at which core first rises; None: ≥ cap
    core_at_flip: float | None
    core_at_2pct: float           # core with the row's lane coefficient 2 % in the cheater's favour


@dataclass(frozen=True)
class Audit:
    device: str
    design: str
    q: float
    core: float
    machine_eps: float
    floor: float
    gamma: float
    binding: tuple[str, ...]
    near: tuple[Row, ...]         # the cheater's rows within NEAR of binding, binding first
    uniform_route: bool           # whether the device's census has a uniform datapath (a UNI class)


@dataclass(frozen=True)
class Margin:
    device: str
    design: str
    m: float                      # the margin (fraction) each near facet moves in the cheater's favour
    base: float                   # the core as the design is built
    worst: float                  # the core with every near facet's lane coefficient moved m at once
    driver: str                   # the near row whose move alone raises the core most ("" if none does)
    rows: tuple[tuple[str, float], ...]   # each near row and the core with that one row moved m


@dataclass(frozen=True)
class LeastFill:
    device: str
    design: str
    per_word: float | None        # least filler a word that holds the margin; None if `cap` did not
    mix: dict[str, float]
    core: float                   # the joint margin's worst core at `per_word`
    gamma: float                  # honest's γ at `per_word`
    base_worst: float             # the joint margin's worst core at the design as built


UNIFORM = ("UNI", "UNI64")       # the uniform-datapath classes: a measured coefficient of their own on a joint row


def _lane_coef(pi: E.Machine, row: str) -> tuple[float, tuple[str, ...]]:
    """The small (lane) coefficient a row shares and the classes it moves. The margin moves this one: a joint facet's
    small coefficient is its vector lane cost (the CUDA lane classes, `pi.cuda`); the large one is the saturated tensor
    pipe, which silicon cannot cheapen, so it is left alone. A uniform class's coefficient on the row is measured on its
    own (UNI on JI90_0: 0.965 on DC-90, 0.000186 beside the async wgmma on DC-90-ri), so it never stands for the lane
    coefficient, and it moves with it. A row no lane class is on falls back to the least non-uniform coefficient."""
    cols = {k: pi.c[k][row] for k in pi.c if row in pi.c[k]}
    lanes = ({k: v for k, v in cols.items() if k in pi.cuda}
             or {k: v for k, v in cols.items() if k not in UNIFORM} or cols)
    small = min(lanes.values())
    hit = tuple(k for k, v in lanes.items() if abs(v - small) < 1e-9)
    return small, hit + tuple(k for k in cols if k in UNIFORM and k not in hit)


def _single(pi: E.Machine, row: str, f: float) -> E.Machine:
    """Π with `row`'s lane coefficient (the classes at its minimum) scaled by f (f < 1: cheaper for the cheater)."""
    _, hit = _lane_coef(pi, row)
    return replace(pi, c={k: ({**v, row: v[row] * f} if k in hit else v) for k, v in pi.c.items()})


def _joint(pi: E.Machine, cc: dict[str, dict[str, float]], row: str, hit: tuple[str, ...], f: float) -> E.Machine:
    """Π built from the accumulating census `cc`, with `row`'s lane coefficient moved by f off its original value."""
    return replace(pi, c={k: ({**v, row: pi.c[k][row] * f} if k in hit else v) for k, v in cc.items()})


def _worst_single(core, pi: E.Machine, row: str, m: float) -> float:
    """The higher core from moving `row`'s lane coefficient by (1 − m) or (1 + m): the cheater's favoured direction."""
    return max(core(_single(pi, row, 1 - m)), core(_single(pi, row, 1 + m)))


def _joint_worst(core, pi: E.Machine, near: list[str], m: float) -> float:
    """The margin's joint worst core: every row in `near` moved m in the cheater's favour at once, each in the
    direction that raises the core (greedy)."""
    cc = {k: dict(v) for k, v in pi.c.items()}
    for r in near:
        _, hit = _lane_coef(pi, r)
        f = (1 - m) if core(_joint(pi, cc, r, hit, 1 - m)) >= core(_joint(pi, cc, r, hit, 1 + m)) else (1 + m)
        for k in hit:
            cc[k][r] *= f
    return core(replace(pi, c=cc))


def _near_rows(pi: E.Machine, d: E.Design, data: Path, q: float, thr: float, operands: str, credit: str) -> list[str]:
    c = E.setup(d, pi, data, operands, credit).cheat(q)
    load = pi.load(c.n)
    return [r for r in pi.rows if r != ISSUE_CEILING and c.T and load[r] / c.T >= thr]


def _flip(d: E.Design, pi: E.Machine, data: Path, q: float, row: str, base: float, operands: str, credit: str,
          step: float = 0.005, cap: float = 0.10) -> tuple[float | None, float | None]:
    m = 0.0
    core = lambda p: E.eps(d, p, q, data, operands=operands, credit=credit).core()
    while m < cap - 1e-9:
        m += step
        c = _worst_single(core, pi, row, m)
        if c > base + 5e-4:
            return round(m * 100, 1), c
    return None, None


def audit(design: str, data: Path, q: float = 1.22, operands: str = "free", credit: str = "floor",
          vector: str = "") -> Audit:
    """The red-team audit of `design` on the device directory `data`'s Π (`vector`, default its current) at q*."""
    pi = E.machine(vector, data)
    d = E.design(design, data)
    e = E.eps(d, pi, q, data, operands=operands, credit=credit)
    c = e.cheat
    load, base = pi.load(c.n), e.core()
    core = lambda p: E.eps(d, p, q, data, operands=operands, credit=credit).core()
    rows: list[Row] = []
    for r in pi.rows:
        x = load[r] / c.T if c.T else 0.0
        if x < NEAR:
            continue
        if r == ISSUE_CEILING:
            rows.append(Row(r, x, c.pi.get(r, 0.0), True, None, None, base))
            continue
        flip_pct, core_flip = _flip(d, pi, data, q, r, base, operands, credit)
        rows.append(Row(r, x, c.pi.get(r, 0.0), False, flip_pct, core_flip, _worst_single(core, pi, r, 0.02)))
    rows.sort(key=lambda t: -t.load)
    return Audit(device=pi.vector, design=d.name, q=q, core=base, machine_eps=e.machine_eps, floor=e.floor,
                 gamma=e.gamma, binding=c.binding, near=tuple(rows), uniform_route="UNI" in pi.c)


def margin_audit(design: str, data: Path, m: float = 0.02, near_thr: float = MARGIN_NEAR, q: float = 1.22,
                 operands: str = "free", credit: str = "floor", vector: str = "") -> Margin:
    """The margin on `design`: move each near measured facet's lane coefficient m in the cheater's favour.

    Returns the core as built, the joint worst (every near facet moved at once), and the single row that drives it (the
    one whose move alone raises the core most). The small (lane) coefficient a facet shares moves, not its tensor-pipe
    MMA value (`_lane_coef`); the issue slot and the exact
    operand floor are axioms, not measured rows, so both are left alone."""
    pi = E.machine(vector, data)
    d = E.design(design, data)
    core = lambda p: E.eps(d, p, q, data, operands=operands, credit=credit).core()
    base = core(pi)
    near = _near_rows(pi, d, data, q, near_thr, operands, credit)
    singles = tuple((r, _worst_single(core, pi, r, m)) for r in near)
    driver = max(singles, key=lambda t: t[1])[0] if singles and max(s for _, s in singles) > base + 1e-6 else ""
    return Margin(device=pi.vector, design=d.name, m=m, base=base, worst=_joint_worst(core, pi, near, m),
                  driver=driver, rows=singles)


def least_filler(design: str, data: Path, mix: dict[str, float] | None = None, target: float | None = None,
                 m: float = 0.02, near_thr: float = MARGIN_NEAR, step: float = 0.005, cap: float = 0.30,
                 q: float = 1.22, operands: str = "free", credit: str = "floor", vector: str = "") -> LeastFill:
    """The least verified filler a word, in class mix `mix` (default the design's), that holds the margin:
    its joint worst core is ≤ `target` (default the design's base core, the floor it sits at). The filler replaces the
    design's own and sweeps in `step` up to `cap`, checked as the design's filler is (its p and luck), or by a receipt
    the check recomputes (p = 0) when the design has none. Returns the per_word, the worst core there and honest's γ."""
    pi = E.machine(vector, data)
    d0 = E.design(design, data)
    mix = mix or (d0.filler[0].mix if d0.filler and d0.filler[0].mix else {"FFMA": 1.0})
    check = d0.filler[0] if d0.filler else E.Filler(p=0.0, luck=E.PINNED)
    base_core = E.eps(d0, pi, q, data, operands=operands, credit=credit).core()
    target = base_core if target is None else target
    base_worst = _joint_worst(lambda p: E.eps(d0, p, q, data, operands=operands, credit=credit).core(), pi,
                              _near_rows(pi, d0, data, q, near_thr, operands, credit), m)

    def at(x: float) -> tuple[E.Design, float, float]:
        d = replace(d0, filler=(E.Filler(per_word=x, mix=mix, route="lanes", p=check.p, luck=check.luck,
                                         source="redteam.least_filler"),))
        core = lambda p: E.eps(d, p, q, data, operands=operands, credit=credit).core()
        near = _near_rows(pi, d, data, q, near_thr, operands, credit)
        return d, _joint_worst(core, pi, near, m), E.eps(d, pi, q, data, operands=operands, credit=credit).gamma

    x = step
    while x < cap + 1e-9:
        d, worst, gamma = at(x)
        if worst <= target + 1e-6:
            return LeastFill(pi.vector, d0.name, round(x, 4), mix, worst, gamma, base_worst)
        x += step
    _, worst, gamma = at(cap)
    return LeastFill(pi.vector, d0.name, None, mix, worst, gamma, base_worst)


def uniform_gap(design: str, data: Path, q: float = 1.22, issue_cost: float = 0.9838, pipe_rate: float = 0.4999,
                operands: str = "free", credit: str = "floor", vector: str = "") -> tuple[float, float, tuple[str, ...]]:
    """Bound the ε a uniform datapath would open on a device whose census has none (no UNI class).

    A uniform instruction is credited `issue_cost` of an ALU's slot on every issue-type row (`issue` and the joint
    issue rows), `1 / pipe_rate` SMSP-cycles on a synthesised uniform-pipe row `UP`, and nothing on the
    other pipes (the cheater's best case). The defaults are sm_90's measured uniform datapath (UP 0.4999 instr a
    SMSP-cycle, 0.9838 of an issue slot). Returns (core as built, core with the route open, the cheater's binding);
    where the census already has a UNI class the route is measured and the two cores are equal."""
    pi = E.machine(vector, data)
    d = E.design(design, data)
    base = E.eps(d, pi, q, data, operands=operands, credit=credit).core()
    if "UNI" in pi.c:
        return base, base, ()
    issue_rows = [r for r in pi.rows if r == ISSUE_CEILING or r.startswith("JI") or r == "JIM"]
    c = {k: {**col, "UP": 0.0} for k, col in pi.c.items()}
    c["UNI"] = {r: issue_cost * pi.c["ALU"].get(r, 0.0) for r in issue_rows}
    c["UNI"]["UP"] = 1.0 / pipe_rate
    pi_u = replace(pi, rows=(*pi.rows, "UP"), c=c, cuda=pi.cuda)
    e = E.eps(d, pi_u, q, data, operands=operands, credit=credit, uniform=True)
    return base, e.core(), e.cheat.binding
