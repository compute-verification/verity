"""What a Lean run reserves and how many jobs it runs, never what it checks.

Each step of an audit (`check.py`: `build`, the package's `lake build`, and `replay`, its kernel replay) holds at its
largest the GiB `need` gives it, by what the package pins of the heavy dependencies (``pins``: the names of Mathlib and
ArkLib its manifest pins, empty for a package that pins neither). A run asks the largest need among the steps it takes
of every package it audits (`ask`, in GB as `research run --mem-gb` takes it), refuses below the largest `floor`, and
plans a heavy build's jobs and a heavy replay's threads within what it reserved (`reserved`, `build_jobs`,
`replay_threads`). A step whose peak passes `WARN_SHARE` of
its need warns, naming the line here to raise (`warning`). No audit's key reads this file (`lean_audit.NOT_READ`): a
figure changed here reruns no audit and no records run. Standard library only.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path

#: a build of a package that pins Mathlib or ArkLib, every job of it at once and the python beside them
BUILD_HEAVY_GIB = 86
#: a build of a package that pins neither, and the replay of one that doesn't pin ArkLib
LIGHT_GIB = 14
#: the whole kernel replay of a package that pins ArkLib, one process run alone: a figure of the tree that grows with what
#: the replay checks, so raise it when a replay's report warns
REPLAY_ARKLIB_GIB = 240
#: a heavy replay's threads: one per this many GiB of what its run reserved, since nothing else bounds what it holds
REPLAY_PER_THREAD_GIB = 4
#: the least a heavy build runs in: its largest job alone and the python beside it
ONE_JOB_GIB = 39
#: each job a heavy build runs beside that one, below `BUILD_HEAVY_GIB`
PER_JOB_GIB = 8
#: the share of its need past which a step's peak warns
WARN_SHARE = 0.95
ARKLIB = "Arklib"
STEPS = ("build", "replay")
#: what says a run reserved, each in its unit: a check slot's in GiB (slot.py), a `research run --mem-gb` run's in GB
RESERVATIONS = (("CHECK_SLOT_MEM_GB", "GiB"), ("CHECK_RUN_MEM_GB", "GB"), ("RESEARCH_RUN_MEM_GB", "GB"))
#: set inside a research run, where none of `RESERVATIONS` set means the run reserved nothing
IN_RUN = "RESEARCH_RUN_ID"
HERE = "tools/verity/lean/reserve.py"


def figure(step: str, pins: Iterable[str]) -> str:
    """The name here of what ``step`` of a package pinning ``pins`` holds at its largest."""
    pins = set(pins)
    if step == "replay" and ARKLIB in pins:
        return "REPLAY_ARKLIB_GIB"
    return "BUILD_HEAVY_GIB" if step == "build" and pins else "LIGHT_GIB"


def need(step: str, pins: Iterable[str]) -> int:
    """The GiB ``step`` of a package pinning ``pins`` holds at its largest."""
    return globals()[figure(step, pins)]


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


def ask(pins_of: Mapping[str, Iterable[str]], steps: Iterable[str] = STEPS) -> tuple[int, int, str | None, str | None]:
    """(the GB a run asks, the GiB that holds, the step and the package it is of): the largest `need` among ``steps`` of
    the packages it audits (``pins_of``)."""
    n, step, pkg = largest(need, pins_of, steps)
    return gb(n), n, step, pkg


def reserved(env: Mapping[str, str], machine_gib: float) -> tuple[float | None, str]:
    """(the GiB a run reserved, what says so): the least of the `RESERVATIONS` set in ``env``, each converted to GiB, and
    the machine's memory (``machine_gib``); None inside a research run (`IN_RUN`) that sets none of them, and wherever
    one of them is no number."""
    got = []
    for name, unit in RESERVATIONS:
        try:
            v = float(env.get(name) or 0)
        except ValueError:
            return None, f"{name}={env[name]!r} is no number of {unit}"
        if v > 0:
            got.append((v if unit == "GiB" else gib(v), f"{name}={env[name]}, in {unit}"))
    if not got and env.get(IN_RUN):
        return None, f"research run {env[IN_RUN]} sets none of {', '.join(n for n, _ in RESERVATIONS)}"
    return min([*got, (machine_gib, "this machine's memory and swap, within its cgroup")], key=lambda t: t[0])


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
    """``name``'s line in this file, as `tools/verity/lean/reserve.py:N`."""
    lines = Path(__file__).read_text().splitlines()
    return f"{HERE}:{next(i for i, line in enumerate(lines, 1) if line.startswith(f'{name} = '))}"


def warning(step: str, pins: Iterable[str], peak_gib: float) -> str | None:
    """What ``step``'s report says when its peak, ``peak_gib``, passes `WARN_SHARE` of its `need`; None below that."""
    n, name = need(step, pins), figure(step, pins)
    if peak_gib <= WARN_SHARE * n:
        return None
    return (f"{step}: its peak, {peak_gib:g} GiB, passes {WARN_SHARE:.0%} of the {n} GiB it is given ({name}): raise "
            f"{name} at {where(name)}, whose ask a run reserves, before a run that reserved it runs out")
