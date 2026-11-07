"""Lean's check of the planner's one-stage designs: `SampledProofs.PlanCheck` (the `sampled-proofs-plancheck` executable)
reads a design exactly, in rationals, and answers what it certifies, and `Properties.SampledProofs` proves each answer
from the proved guarantees (`LeakCheckSound`, `LeakCountSound`, `WorkCheckSound`).

A leak design is `units` proof blocks each exporting at most `width` bits, each sampled independently at `rate`
(`Law.bernoulli`), planned so that more than `count` incorrect ones escape with probability below `delta`.  Lean answers
the bits it certifies (`CertifiesLeak`), or None when it can't certify the count: at least the planner's
`b K + loc_bits(n, K)` in exact arithmetic, which the planner's float can exceed by its rounding.  A work design is strata of proof blocks, each a block's `work` and its rate `num / den`, planned so that incorrect
blocks of total work at least `target` escape with probability at most `delta` (`MeetsTarget`, however many blocks each
stratum holds), and Lean answers whether they do.

The planner computes in floats.  `leak_design` and `work_design` state its design exactly, each rate rounded up by
`SLACK` (which raises the design's expected samples by that share and no more), so that a float rounded below the rate
the bound needs is not a refusal; `delta` is the decimal it prints.  `check` runs the executable once over many designs.
"""
from __future__ import annotations

import json
import math
import subprocess
from fractions import Fraction
from typing import Iterable, Sequence

from .one_stage.verifier import LeanError, executable
from .optimize import Grain, leak_count

EXE = "sampled-proofs-plancheck"
#: the share by which a design's float rate is rounded up before Lean checks it
SLACK = Fraction(1, 2**20)


def exact(x: float | int | Fraction) -> Fraction:
    """A float as the decimal Python prints for it (0.01 as 1/100); an int or a Fraction as itself."""
    return Fraction(repr(x)) if isinstance(x, float) else Fraction(x)


def _q(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


def upper(x: Fraction) -> float:
    """The least float at least `x`."""
    f = float(x)
    return f if Fraction(f) >= x else math.nextafter(f, math.inf)


def rate_up(q: float) -> Fraction:
    """`q`, a rate in [0, 1], rounded up by `SLACK`, at most 1."""
    if not 0 <= q <= 1:
        raise ValueError(f"a sampling rate is in [0, 1], not {q}")
    return min(Fraction(1), Fraction(q) * (1 + SLACK))


def leak(units: int, width: int, rate: Fraction, count: int, delta: float | Fraction) -> dict:
    """A leak design as the executable reads it."""
    if units < 1 or width < 0 or count < 0 or not 0 <= rate <= 1:
        raise ValueError("a leak design needs units >= 1, width >= 0, count >= 0 and a rate in [0, 1]")
    return {"kind": "leak", "units": str(units), "width": str(width), "q": _q(Fraction(rate)), "count": str(count),
            "delta": _q(exact(delta))}


def work(strata: Iterable[tuple[float | Fraction, int]], den: int, target: float | Fraction,
         delta: float | Fraction) -> dict:
    """A work design as the executable reads it: each stratum a block's work and its rate's numerator over `den`."""
    return {"kind": "work", "strata": [[_q(Fraction(w)), str(n)] for w, n in strata], "den": str(den),
            "target": _q(Fraction(target)), "delta": _q(exact(delta))}


def leak_design(g: Grain, ell: float, delta: float) -> dict:
    """The planner's one-stage design on `g` at a leak target of `ell` bits (`optimize.leak_draws`): the count
    `leak_count` allows, every proof block sampled at min(1, Λ/(K + 1)), its width `out_bits` rounded up."""
    k = leak_count(g, ell)
    q = 0.0 if k >= g.units else min(1.0, math.log(1 / delta) / (k + 1))
    return leak(g.units, math.ceil(g.out_bits), rate_up(q), k, delta)


def work_design(g: Grain, target: float, delta: float) -> dict:
    """The planner's one-stage design on `g` at a free-compute target of `target` (`optimize.work_draws`): every proof
    block, of work `g.work`, sampled at min(1, Λ s/C)."""
    r = rate_up(min(1.0, math.log(1 / delta) * g.work / target))
    return work([(Fraction(g.work), r.numerator)], r.denominator, target, delta)


def check(designs: Sequence[dict]) -> list[Fraction | bool | None]:
    """Lean's answer to each design, in order: a leak design's certified bits (None when refused), a work design's
    verdict."""
    if not designs:
        return []
    p = subprocess.run([str(executable(EXE))], input="".join(json.dumps(d) + "\n" for d in designs),
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise LeanError(f"{EXE} exited {p.returncode}: {p.stderr[-2000:]}")
    out = [json.loads(line) for line in p.stdout.splitlines()]
    if len(out) != len(designs):
        raise LeanError(f"{EXE} answered {len(out)} of {len(designs)} designs")
    answers: list[Fraction | bool | None] = []
    for d, o in zip(designs, out):
        if "error" in o:
            raise LeanError(f"{EXE} can't read {d}: {o['error']}")
        if d["kind"] == "leak":
            answers.append(None if o["bound"] is None else Fraction(o["bound"]))
        else:
            answers.append(o["ok"])
    return answers
