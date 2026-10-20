"""The per-family headline (`machine-model headline`): each GPU family's design of record (data/headline.toml) at each
reading, operands free and credit ♭ (decision 89), with every number computed from the design files and the vectors.

A reading is a vector per family and a q*: `sheet`, the data sheet's vectors at q* = ½, and `measured`, decision 89's
q* = 1.22 (the 1.22-slot issue floor, Theorem C, unproved in Lean) on the measured vectors, the -ri ones where they
exist (decision 98). ε is relative to the declared capacity, the reading's Π (decision 100), and per node (ASK-65 = A,
data/headline.toml's `scope`): an accepted window's node, its GPUs, host CPU and peer GPUs together, performed at
least 1 − ε of the credited work, and a per-GPU reading is the attribution layer's (README, "The headline is per
node"). A row is ε_skip at that vector (the design's `[skip.served]` there, which must be recorded: ♭ moves with Π),
ε_mand, ε_machine and ε_luck, their sum (the core ε, Corollary D3's with ε_sched = ε_clock = 0 under ♭) and γ. Beside
them are the conditions it is stated under:

- the floor it needs: the ½-slot floor where ε_machine is at its floor by q* = ½, else the 1.22-slot issue floor;
- each filler's credit, the theorem its check meets (decision 99), and the family's p = 1 sibling's ε where it has one;
- the reading's and the family's stated conditions, from data/headline.toml;
- A2u, the reading's `a2u`, wherever Π charges a uniform instruction on an issue row: no data sheet gives a uniform
  rate, so there the uniform route is priced by A2u alone (without it the sheet rows' ε is unbounded).

`TERMS` holds the terms' names, the one place they are spelled; the JSON's keys are TERMS' keys.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from verity_machine_model.eps import Design, design, eps, machine, setup, threshold, uniform_issue_charged
from verity_machine_model.model import DATA, device, model

TERMS = {"skip": "ε_skip", "mand": "ε_mand", "machine": "ε_machine", "luck": "ε_luck", "core": "ε", "gamma": "γ",
         "threshold": "q*₀"}
READINGS = ("sheet", "measured")
OPERANDS, CREDIT = "free", "floor"


def spec(data: Path = DATA) -> dict:
    return tomllib.loads((data / "headline.toml").read_text())


def skip_at(d: Design, vector: str, data: Path) -> float:
    """ε_skip at `vector`: each part's at the device's current vector, or its `at` entry; unrecorded, refused."""
    current = model(data).current
    missing = [k for k, p in d.skip.items() if vector != current and vector not in p.at]
    if missing:
        raise SystemExit(f"{d.name}: no ε_skip at {vector} (current {current}): record eps_budget's `design_skip(d, "
                         f"data=..., vector={vector!r})` as [skip.{missing[0]}] at = {{ \"{vector}\" = … }}")
    return sum(p.eps if vector == current else p.at[vector] for p in d.skip.values())


def _sibling(d: Design, p1: Design) -> None:
    """`p1` must be `d` with every filler at p = 1 and no luck: what else differs is its name and ε_skip."""
    free = lambda x: replace(x, name="", title="", summary="", aliases=(), base="", skip={},
                             filler=tuple(replace(f, p=1.0, luck="", source="") for f in x.filler))
    if not d.filler or free(d) != free(p1) or any(f.p < 1 for f in p1.filler):
        raise SystemExit(f"{p1.name}: a family's p1 design is its design ({d.name}) with every filler at p = 1, no "
                         "luck; only its name, title, summary and ε_skip differ")


@dataclass(frozen=True)
class Row:
    family: str
    gpu: str
    design: str
    reading: str
    vector: str
    q: float
    skip: float
    mand: float
    machine: float
    luck: float
    core: float
    gamma: float
    threshold: float | None          # the least q* at which ε_machine reaches its floor
    binding: tuple[str, ...]         # the cheater's binding rows
    floor: str                       # the floor the row needs: a key of headline.toml's [floor]
    fillers: tuple[str, ...]         # each filler's credit
    p1: dict | None                  # the p = 1 sibling's design and terms
    conditions: tuple[str, ...]      # every condition, computed and stated, in print order


def row(family: str, reading: str, data: Path = DATA) -> Row:
    t = spec(data)
    f, r = t["family"][family], t["reading"][reading]
    dd = device(family)
    d = design(f["design"], dd)
    vec, q = f[reading]["vector"], float(r["q"])
    pi = machine(vec, dd)
    skip = skip_at(d, pi.vector, dd)
    e = eps(d, pi, q, dd, operands=OPERANDS, credit=CREDIT)
    assert abs(e.skip - skip) < 1e-12
    thr = threshold(setup(d, pi, dd, OPERANDS, CREDIT))
    floor = "half" if q <= 0.5 or (thr is not None and thr <= 0.5) else "issue"
    fillers = tuple(
        f"{x.per_word:g} {'+'.join(x.mix or {})} a word credited at p = {x.p:g} ({x.luck.split(',')[0].split()[0]})"
        if x.p < 1 else f"{x.per_word:g} {'+'.join(x.mix or {})} a word uncredited, p = 1 (decision 99)"
        for x, _ in e.fillers)
    p1 = None
    if f.get("p1"):
        d1 = design(f["p1"], dd)
        _sibling(d, d1)
        skip_at(d1, pi.vector, dd)
        e1 = eps(d1, pi, q, dd, operands=OPERANDS, credit=CREDIT)
        p1 = {"design": d1.name, "skip": e1.skip, "mand": e1.mand, "machine": e1.machine_eps, "luck": e1.luck,
              "core": e1.core(), "gamma": e1.gamma}
    computed = (t["floor"][floor], *fillers,
                *([f"at p = 1 ({p1['design']}): {TERMS['core']} {100 * p1['core']:.3f}%"] if p1 else []))
    a2u = (r["a2u"],) if r.get("a2u") and uniform_issue_charged(pi) else ()
    return Row(family=family, gpu=f["gpu"], design=d.name, reading=reading, vector=pi.vector, q=q, skip=e.skip,
               mand=e.mand, machine=e.machine_eps, luck=e.luck, core=e.core(), gamma=e.gamma, threshold=thr,
               binding=e.cheat.binding, floor=floor, fillers=fillers, p1=p1,
               conditions=(*computed, *f.get("conditions", ()), *f[reading].get("conditions", ()), *a2u))


def rows(reading: str, data: Path = DATA) -> list[Row]:
    return [row(fam, reading, data) for fam in spec(data)["family"]]


def record(readings: tuple[str, ...] = READINGS, data: Path = DATA) -> dict:
    t = spec(data)
    return {"terms": TERMS, "operands": OPERANDS, "credit": CREDIT, "capacity": t["capacity"], "scope": t["scope"],
            "readings": {r: {"title": t["reading"][r]["title"], "q": t["reading"][r]["q"],
                             "conditions": t["reading"][r]["conditions"],
                             "rows": [asdict(x) for x in rows(r, data)]} for r in readings}}


def _pct(x: float) -> str:
    return f"{100 * x:.3f}%"


def text(readings: tuple[str, ...] = READINGS, data: Path = DATA) -> str:
    """Each reading as a Markdown table, a row a family, and its conditions beneath, a line a family."""
    t, T = spec(data), TERMS
    out = []
    for r in readings:
        rd = t["reading"][r]
        rs = rows(r, data)
        out += [f"## {rd['title']} (`{r}`): q* = {rd['q']:g}, operands {OPERANDS}, credit ♭", "",
                *(f"- {c}" for c in (t["scope"], t["capacity"], *rd["conditions"])), "",
                f"| GPU | design | Π | {T['skip']} | {T['mand']} | {T['machine']} | {T['luck']} | "
                f"{T['core']} = their sum | {T['gamma']} | {T['threshold']} | cheater binds |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for x in rs:
            thr = "none ≤ 64" if x.threshold is None else f"{x.threshold:.3f}"
            out.append(f"| {x.gpu} | {x.design} | {x.vector} | {_pct(x.skip)} | {_pct(x.mand)} | {_pct(x.machine)} | "
                       f"{_pct(x.luck)} | **{_pct(x.core)}** | {x.gamma:.3f} | {thr} | {', '.join(x.binding)} |")
        out += ["", "Conditions:", *(f"- {x.gpu}: " + "; ".join(x.conditions) for x in rs), ""]
    return "\n".join(out).rstrip() + "\n"
