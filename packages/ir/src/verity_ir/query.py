"""Query language (IR §7): `/`, `/Name`, `//Name`, `//A/B`, `//A//B`, `//A ! groups(F, k)`.

Results are ordered collections of `FunctionCall`s (path queries) or `Region`s (`groups`), in canonical
call-preorder, zero-based. `groups(F, k)` is evaluated once per scope selected by the left side and
concatenated; a block whose calls are not adjacent is rejected (`GapError`), never absorbed or re-cut.
"""

from __future__ import annotations

import re
from typing import Iterator

from .layout import FunctionCall, Circuit, Region


class QueryError(ValueError):
    pass


class GapError(QueryError):
    pass


_STEP = re.compile(r"(//|/)([A-Za-z_][A-Za-z0-9_]*)?")
_GROUPS = re.compile(r"^\s*groups\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)\s*$")


def _parse_path(path: str) -> list[tuple[str, str | None]]:
    path = path.strip()
    if not path:
        raise QueryError("empty query")
    steps: list[tuple[str, str | None]] = []
    pos = 0
    while pos < len(path):
        m = _STEP.match(path, pos)
        if not m:
            raise QueryError(f"cannot parse query at {path[pos:]!r}")
        steps.append((m.group(1), m.group(2)))
        pos = m.end()
    if path == "/":
        return [("/", None)]
    for i, (op, name) in enumerate(steps):
        if name is None and not (i == 0 and len(steps) == 1):
            raise QueryError("a step needs a name")
    return steps


def _children(circuit: Circuit, scope_call: FunctionCall | None, name: str | None = None) -> Iterator[FunctionCall]:
    if scope_call is None:
        yield from circuit.root_calls(primitive_batches=name)
    else:
        yield from scope_call.children(primitive_batches=name)


def _descendants(circuit: Circuit, scope_call: FunctionCall | None, name: str | None = None) -> Iterator[FunctionCall]:
    for c in _children(circuit, scope_call, name):
        yield c
        if not c.is_primitive:
            yield from _descendants(circuit, c, name)


def query(circuit: Circuit, q: str) -> list:
    """Evaluate a query against the circuit. Returns a list of `FunctionCall` (path queries) or `Region`
    (`groups`) objects in canonical order."""
    if "!" in q:
        left, right = q.split("!", 1)
        m = _GROUPS.match(right)
        if not m:
            raise QueryError(f"unsupported right side {right.strip()!r} (only groups(F, k))")
        fname, k = m.group(1), int(m.group(2))
        if k <= 0:
            raise QueryError("groups(F, k) needs a positive k")
        scopes = _path(circuit, left)
        out: list[Region] = []
        for sc in scopes:
            out.extend(groups(circuit, sc, fname, k))
        return out
    return _path(circuit, q)


def _path(circuit: Circuit, path: str) -> list:
    steps = _parse_path(path)
    if steps == [("/", None)]:
        return [Region(circuit, 0, circuit.size)]
    # frontier: None stands for the root scope
    frontier: list[FunctionCall | None] = [None]
    for op, name in steps:
        nxt: list[FunctionCall] = []
        seen: set[tuple[int, int]] = set()
        for sc in frontier:
            it = _children(circuit, sc, name) if op == "/" else _descendants(circuit, sc, name)
            for c in it:
                if c.matches(name):  # type: ignore[arg-type]
                    key = (c.lo, c.hi)
                    if key in seen and op == "//":
                        continue  # the same call reached through two nested matching ancestors
                    seen.add(key)
                    nxt.append(c)
        nxt.sort(key=lambda c: (c.lo, -c.hi))
        frontier = list(nxt)
    return frontier  # type: ignore[return-value]


def groups(circuit: Circuit, scope: FunctionCall | Region | None, fname: str, k: int) -> list[Region]:
    """Within one scope: direct child calls to `fname` in canonical order, cut from the beginning into blocks
    of at most `k`; each block must be adjacent, nonempty, nonoverlapping; returns one region per block."""
    if k <= 0:
        raise QueryError("groups(F, k) needs a positive k")
    if isinstance(scope, Region) and not isinstance(scope, FunctionCall):
        if scope.lo == 0 and scope.hi == circuit.size:
            scope = None
        else:
            raise QueryError("groups scope must be a call (or the root)")
    calls = [c for c in _children(circuit, scope, fname) if c.matches(fname)]
    out: list[Region] = []
    for b in range(0, len(calls), k):
        block = calls[b:b + k]
        lo, hi = block[0].lo, block[0].hi
        for c in block[1:]:
            if c.lo != hi:
                raise GapError(f"groups({fname},{k}): calls {block[0]!r} .. {c!r} are not adjacent (gap at [{hi}:{c.lo}])")
            hi = c.hi
        out.append(Region(circuit, lo, hi))
    return out
