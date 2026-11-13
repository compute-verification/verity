"""What a Lean run reserves and how many jobs it runs, never what it checks.

Each step of an audit (`check.py`: `build`, the package's `lake build`, and `replay`, its kernel replay) holds at its
largest the GiB `need` gives it, by what the package pins of the heavy dependencies (``pins``: the names of Mathlib and
ArkLib its manifest pins, empty for a package that pins neither). A run asks the largest need among the steps it takes
of every package it audits (`ask`, in GB as `research run --mem-gb` takes it), refuses below the largest `floor`, and
plans a heavy build's jobs and a heavy replay's threads within what it reserved (`reserved`, `build_jobs`,
`replay_threads`). A step whose peak would give a larger figure than its own warns, naming the line to raise (`warning`).

The rules are here, in every audit's key: a changed rule audits every package again. The figures are `reserve.json`'s,
which no audit's key reads (`lean_audit.NOT_READ`), so a changed figure reruns no audit and no records run. This file
takes numbers and reads nothing but that file: what a run reserved is read by `check.py`.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

DATA = Path(__file__).with_name("reserve.json")
HERE = "tools/verity/lean/reserve.json"
FIGURES = json.loads(DATA.read_text())
#: a build of a package that pins Mathlib or ArkLib, every job of it at once and the python beside them
BUILD_HEAVY_GIB = FIGURES["BUILD_HEAVY_GIB"]
#: a build of a package that pins neither, and the replay of one that doesn't pin ArkLib
LIGHT_GIB = FIGURES["LIGHT_GIB"]
#: the whole kernel replay of a package that pins ArkLib, one process run alone: a figure of the tree that grows with what
#: the replay checks, so raise it when a replay warns
REPLAY_ARKLIB_GIB = FIGURES["REPLAY_ARKLIB_GIB"]
#: a heavy replay's threads: one per this many GiB of what its run reserved, since nothing else bounds what it holds
REPLAY_PER_THREAD_GIB = FIGURES["REPLAY_PER_THREAD_GIB"]
#: the least a heavy build runs in: its largest job alone and the python beside it
ONE_JOB_GIB = FIGURES["ONE_JOB_GIB"]
#: each job a heavy build runs beside that one, below `BUILD_HEAVY_GIB`
PER_JOB_GIB = FIGURES["PER_JOB_GIB"]
#: the share a step's figure is set above the largest peak the step has shown
MARGIN = FIGURES["MARGIN"]
ARKLIB = "Arklib"
#: what says a run reserved, each in its unit: a check slot's in GiB (slot.py), a `research run --mem-gb` run's in GB
RESERVATIONS = (("CHECK_SLOT_MEM_GB", "GiB"), ("CHECK_RUN_MEM_GB", "GB"), ("RESEARCH_RUN_MEM_GB", "GB"))


def figure(step: str, pins: Iterable[str]) -> str:
    """The name of the figure ``step`` of a package pinning ``pins`` holds at its largest."""
    pins = set(pins)
    if step == "replay" and ARKLIB in pins:
        return "REPLAY_ARKLIB_GIB"
    return "BUILD_HEAVY_GIB" if step == "build" and pins else "LIGHT_GIB"


def need(step: str, pins: Iterable[str]) -> int:
    """The GiB ``step`` of a package pinning ``pins`` holds at its largest."""
    return FIGURES[figure(step, pins)]


def floor(step: str, pins: Iterable[str]) -> int:
    """The least GiB a run taking ``step`` of such a package must have reserved: a heavy build's one job, since it runs
    fewer jobs down to one; an ArkLib replay's whole need; none for the rest."""
    pins = set(pins)
    if step == "replay" and ARKLIB in pins:
        return REPLAY_ARKLIB_GIB
    return ONE_JOB_GIB if step == "build" and pins else 0


def largest(of, pins_of: Mapping[str, Iterable[str]], steps: Iterable[str]) -> tuple[int, str | None, str | None]:
    """(the largest ``of`` (`need` or `floor`) among ``steps`` of the packages of ``pins_of`` ({package: its pins}), the
    step and the package it is of); (0, None, None) for none."""
    return max(((of(s, p), s, q) for q, p in pins_of.items() for s in steps), key=lambda t: t[0], default=(0, None, None))


def gb(gib: float) -> int:
    """``gib`` GiB as the whole GB (10^9 bytes) at or above it, as `research run --mem-gb` takes it."""
    return math.ceil(round(gib * 2**30 / 10**9, 6))


def gib(gb: float) -> float:
    """``gb`` GB (10^9 bytes) in GiB."""
    return gb * 10**9 / 2**30


def ask(pins_of: Mapping[str, Iterable[str]], steps: Iterable[str]) -> tuple[int, int, str | None, str | None]:
    """(the GB a run asks, the GiB that holds, the step and the package it is of): the largest `need` among ``steps`` of
    the packages it audits (``pins_of``)."""
    n, step, pkg = largest(need, pins_of, steps)
    return gb(n), n, step, pkg


def reserved(env: Mapping[str, str], machine_gib: float, machine: str) -> tuple[float | None, str]:
    """(the GiB a run reserved, what says so): the least of the `RESERVATIONS` set in ``env``, each converted to GiB, and
    what this machine lets the run hold (``machine_gib``, which ``machine`` names: its cgroup's cap or its memory), which
    is all a run that declares none of them has; None wherever one of them is no number."""
    got = []
    for name, unit in RESERVATIONS:
        try:
            v = float(env.get(name) or 0)
        except ValueError:
            return None, f"{name}={env[name]!r} is no number of {unit}"
        if v > 0:
            got.append((v if unit == "GiB" else gib(v), f"{name}={env[name]}, in {unit}"))
    if not got:
        return machine_gib, f"none of {', '.join(n for n, _ in RESERVATIONS)} declared, so {machine}"
    return min([*got, (machine_gib, machine)], key=lambda t: t[0])


def build_jobs(reserved_gib: float, jobs: int) -> int:
    """How many of ``jobs`` a heavy build runs at once within ``reserved_gib``: all of them from `BUILD_HEAVY_GIB`, else
    one in `ONE_JOB_GIB` and one more for each `PER_JOB_GIB` above it, and never none."""
    if reserved_gib >= BUILD_HEAVY_GIB:
        return jobs
    return max(1, min(jobs, 1 + int((reserved_gib - ONE_JOB_GIB) // PER_JOB_GIB)))


def replay_threads(reserved_gib: float, jobs: int) -> int:
    """How many of ``jobs`` threads a heavy replay runs within ``reserved_gib``: one per `REPLAY_PER_THREAD_GIB`, never
    none."""
    return max(1, min(jobs, int(reserved_gib // REPLAY_PER_THREAD_GIB)))


def where(name: str) -> str:
    """``name``'s line in `reserve.json`, as `tools/verity/lean/reserve.json:N`."""
    lines = DATA.read_text().splitlines()
    return f"{HERE}:{next(i for i, line in enumerate(lines, 1) if line.strip().startswith(json.dumps(name) + ':'))}"


def warning(step: str, pins: Iterable[str], peak_gib: float) -> str | None:
    """What a run prints when ``step``'s peak, ``peak_gib``, would give a larger figure than the one it is given; None
    while it wouldn't."""
    n, name = need(step, pins), figure(step, pins)
    # a figure is its step's largest peak with MARGIN above it, to the nearest GiB, so the peak it was set from never warns
    if peak_gib * (1 + MARGIN) < n + 0.5:
        return None
    return (f"{step}: its peak, {peak_gib:g} GiB, with {MARGIN:.0%} above it passes the {n} GiB it is given ({name}): "
            f"raise {name} at {where(name)}, whose ask a run reserves, before a run that reserved it runs out")
