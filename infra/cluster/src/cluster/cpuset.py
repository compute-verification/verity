"""CPU sets in the kernel's list format: `0-7,16,18-19`.

`parse` reads one into a frozenset of CPU numbers and `format` writes the shortest list back, so `format(parse(s))` is the
canonical spelling. An empty string is the empty set.
"""
from __future__ import annotations

from collections.abc import Iterable


class CpuSetError(ValueError):
    pass


def parse(text: str) -> frozenset[int]:
    cpus: set[int] = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        if not lo.isdigit() or (sep and not hi.isdigit()):
            raise CpuSetError(f"bad CPU list item {part!r} in {text!r}")
        a, b = int(lo), int(hi) if sep else int(lo)
        if a > b:
            raise CpuSetError(f"descending CPU range {part!r} in {text!r}")
        cpus.update(range(a, b + 1))
    return frozenset(cpus)


def format(cpus: Iterable[int]) -> str:
    out: list[str] = []
    run: list[int] = []
    for c in sorted(set(cpus)):
        if run and c == run[-1] + 1:
            run.append(c)
            continue
        if run:
            out.append(_run(run))
        run = [c]
    if run:
        out.append(_run(run))
    return ",".join(out)


def _run(run: list[int]) -> str:
    return str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}"
