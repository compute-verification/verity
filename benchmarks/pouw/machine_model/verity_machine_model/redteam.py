"""Red-team audit of a headline PoUW design against the machine model (RT-DC, campaign cmp-13).

ε is proved conditional on every facet of Π being a true ceiling (eps.py, facet_break/). This module reads, from a
device's committed census alone, how much room an adversary has to beat one:

- `audit` solves the cheater's LP at decision 89's reading (operands free, credit floor, q* = 1.22) and reports the
  binding rows, the rows within `NEAR` of binding (the facets a silicon beat would unseat first), and the decision-76
  sensitivity of each: how far a row's coefficients must move in the cheater's favour before the core ε rises (`flip`).
  The `issue` ceiling (one warp instruction a SMSP-cycle, the hardware issue width, decision 73) is flagged: no silicon
  beats it.
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
ISSUE_CEILING = "issue"           # 1 warp instruction a SMSP-cycle: the hardware issue width (decision 73), not a facet


@dataclass(frozen=True)
class Row:
    name: str
    load: float                   # the cheater's load on the row over T (1.0: binding)
    dual: float                   # the certificate's weight on the row (0 off the facet)
    ceiling: bool                 # the issue ceiling: silicon cannot beat it
    flip_pct: float | None        # the move in the cheater's favour (per cent) at which core first rises; None: ≥ cap
    core_at_flip: float | None
    core_at_2pct: float           # core with the row 2 % in the cheater's favour (decision 76's margin)


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


def _scaled(pi: E.Machine, rows: set[str], f: float) -> E.Machine:
    """Π with every coefficient of each row in `rows` scaled by f (f < 1: cheaper for the cheater)."""
    c = {k: {r: (v * f if r in rows else v) for r, v in col.items()} for k, col in pi.c.items()}
    return replace(pi, c=c)


def _flip(d: E.Design, pi: E.Machine, data: Path, q: float, row: str, base: float, operands: str, credit: str,
          step: float = 0.005, cap: float = 0.10) -> tuple[float | None, float | None]:
    m = 0.0
    while m < cap - 1e-9:
        m += step
        e = E.eps(d, _scaled(pi, {row}, 1 - m), q, data, operands=operands, credit=credit)
        if e.core() > base + 5e-4:
            return round(m * 100, 1), e.core()
    return None, None


def audit(design: str, data: Path, q: float = 1.22, operands: str = "free", credit: str = "floor") -> Audit:
    """The red-team audit of `design` on the device directory `data`'s Π at q*."""
    pi = E.machine("", data)
    d = E.design(design, data)
    e = E.eps(d, pi, q, data, operands=operands, credit=credit)
    c = e.cheat
    load, base = pi.load(c.n), e.core()
    rows: list[Row] = []
    for r in pi.rows:
        x = load[r] / c.T if c.T else 0.0
        if x < NEAR:
            continue
        if r == ISSUE_CEILING:
            rows.append(Row(r, x, c.pi.get(r, 0.0), True, None, None, base))
            continue
        flip_pct, core_flip = _flip(d, pi, data, q, r, base, operands, credit)
        core2 = E.eps(d, _scaled(pi, {r}, 0.98), q, data, operands=operands, credit=credit).core()
        rows.append(Row(r, x, c.pi.get(r, 0.0), False, flip_pct, core_flip, core2))
    rows.sort(key=lambda t: -t.load)
    return Audit(device=pi.vector, design=d.name, q=q, core=base, machine_eps=e.machine_eps, floor=e.floor,
                 gamma=e.gamma, binding=c.binding, near=tuple(rows), uniform_route="UNI" in pi.c)


def uniform_gap(design: str, data: Path, q: float = 1.22, issue_cost: float = 0.9838, pipe_rate: float = 0.4999,
                operands: str = "free", credit: str = "floor") -> tuple[float, float, tuple[str, ...]]:
    """Bound the ε a uniform datapath would open on a device whose census has none (no UNI class).

    A uniform instruction is credited `issue_cost` of an ALU's slot on every issue-type row (`issue` and the joint
    issue rows, decision 73), `1 / pipe_rate` SMSP-cycles on a synthesised uniform-pipe row `UP`, and nothing on the
    other pipes (the cheater's best case). The defaults are sm_90's measured uniform datapath (UP 0.4999 instr a
    SMSP-cycle, 0.9838 of an issue slot). Returns (core as built, core with the route open, the cheater's binding);
    where the census already has a UNI class the route is measured and the two cores are equal."""
    pi = E.machine("", data)
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
