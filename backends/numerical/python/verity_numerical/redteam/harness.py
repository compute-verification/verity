"""The checker-agnostic red-team harness.

A *target* is ``check(x, y, h) -> bool`` (does the checker accept output ``y`` for public input ``x`` under auxiliary
witness ``h``), ``gen(x) -> h`` (the checker's own witness generator), ``reference(x) -> y`` (the function being proved)
and a description of the witness shape: every auxiliary column with the range the checker *declares* for it.  Given
that, the harness runs

    completeness      every case: check(x, reference(x), gen(x)) must hold
    mutations         every column of gen(x), perturbed within its declared range (+-1, bounds, random, sign flip,
                      Boolean flip, power-of-two step): accepted with a different output is a SAT finding, accepted
                      with the same output is a free (unconstrained but unobservable) column
    coordinated       random multi-column perturbations and an output-neighbourhood search (two outputs, one input)
    z3                a driver for target-specific gadget queries with a bounded timeout: SAT / UNSAT / UNKNOWN

and emits a JSON findings report.  Plugging in another checker (WP2's ``check/gen/contract()``) is one ``Target``.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Column", "Finding", "Report", "Target", "run_completeness", "run_coordinated", "run_mutations",
           "tree_get", "tree_set", "z3_query"]

Path_ = tuple[Any, ...]

# Column kinds drive the perturbation set.
BOOL, SIGN, EXPONENT, SHIFT, SELECTOR, MAGNITUDE, REMAINDER, POWER, OTHER = (
    "bool", "sign", "exponent", "shift", "selector", "magnitude", "remainder", "power", "other")


@dataclass(frozen=True)
class Column:
    """One auxiliary witness value: where it lives in ``h`` and the half-open range ``[lo, hi)`` the checker declares."""

    path: Path_
    lo: int
    hi: int
    kind: str = OTHER

    @property
    def name(self) -> str:
        return ".".join(str(k) for k in self.path)


@dataclass
class Target:
    """A checker under attack.  ``output_of(h)`` is the output the witness itself asserts, when the checker has one."""

    name: str
    check: Callable[[Any, Any, Any], bool]
    gen: Callable[[Any], Any]
    reference: Callable[[Any], Any]
    columns: Callable[[Any], list[Column]] = lambda h: []
    output_of: Callable[[Any], Any] | None = None
    output_neighbours: Callable[[Any, Any], Iterable[Any]] = lambda x, y: ()
    fmt_x: Callable[[Any], Any] = lambda x: repr(x)
    fmt_y: Callable[[Any], Any] = lambda y: repr(y)
    fmt_h: Callable[[Any], Any] = lambda h: repr(h)

    def accepts(self, x: Any, y: Any, h: Any) -> bool:
        try:
            return bool(self.check(x, y, h))
        except Exception:      # a perturbed column may break the checker's own arithmetic; that is a rejection
            return False


@dataclass
class Finding:
    """One row of the findings table.  ``status``: SAT (accepted-but-wrong exists), UNSAT (excluded within the
    timeout), UNKNOWN (timeout), PASS / FAIL (completeness), NOTE (observation, not a soundness break)."""

    id: str
    target: str
    attack: str
    status: str
    seconds: float
    case: str = ""
    detail: dict = field(default_factory=dict)
    reproducer: str = ""

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def add(self, f: Finding) -> Finding:
        self.findings.append(f)
        return f

    def to_json(self) -> dict:
        return {"meta": self.meta, "findings": [f.to_json() for f in self.findings],
                "counts": dict(sorted(_count(f.status for f in self.findings).items()))}

    def write(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=1, default=str) + "\n")

    def table(self) -> str:
        rows = [("id", "target", "attack", "status", "seconds", "case", "reproducer")]
        rows += [(f.id, f.target, f.attack, f.status, f"{f.seconds:.2f}", f.case[:40], f.reproducer) for f in self.findings]
        widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
        return "\n".join("  ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in rows)


def _count(items: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        out[it] = out.get(it, 0) + 1
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Witness trees: dicts, lists/tuples and (frozen) dataclasses, addressed by a path of keys / indices / field names
# ---------------------------------------------------------------------------------------------------------------------

def tree_get(h: Any, path: Path_) -> Any:
    for k in path:
        h = getattr(h, k) if dataclasses.is_dataclass(h) else h[k]
    return h


def tree_set(h: Any, path: Path_, value: Any) -> Any:
    """A deep copy of ``h`` with the value at ``path`` replaced (frozen dataclasses are rebuilt with ``replace``)."""
    h = copy.deepcopy(h)
    return _set(h, path, value)


def _set(node: Any, path: Path_, value: Any) -> Any:
    k, rest = path[0], path[1:]
    if dataclasses.is_dataclass(node):
        child = getattr(node, k)
        return dataclasses.replace(node, **{k: value if not rest else _set(child, rest, value)})
    child = node[k]
    new = value if not rest else _set(child, rest, value)
    if isinstance(node, tuple):
        lst = list(node)
        lst[k] = new
        return tuple(lst)
    node[k] = new
    return node


# ---------------------------------------------------------------------------------------------------------------------
# (i) completeness
# ---------------------------------------------------------------------------------------------------------------------

def run_completeness(target: Target, cases: Sequence[tuple[str, Any]], *, fid: str = "C") -> Finding:
    """``check(x, reference(x), gen(x))`` on every case; also that the witness's own output is the reference."""
    t0 = time.time()
    failures = []
    for name, x in cases:
        try:
            y = target.reference(x)
            h = target.gen(x)
            ok = target.check(x, y, h)
            own = target.output_of(h) if target.output_of else y
            if not ok or own != y:
                failures.append({"case": name, "x": target.fmt_x(x), "reference": target.fmt_y(y),
                                 "witness_output": target.fmt_y(own), "accepted": ok})
        except Exception as e:      # noqa: BLE001 -- any exception on an honest case is a completeness failure
            failures.append({"case": name, "x": target.fmt_x(x), "error": repr(e)})
    return Finding(fid, target.name, "completeness", "FAIL" if failures else "PASS", time.time() - t0,
                   case=f"{len(cases)} cases", detail={"failures": failures, "cases": len(cases)})


# ---------------------------------------------------------------------------------------------------------------------
# (ii) single-column mutations within the declared range
# ---------------------------------------------------------------------------------------------------------------------

def perturbations(col: Column, v: int, rng: random.Random) -> list[int]:
    """Candidate replacement values for ``v`` in column ``col``, all inside ``[lo, hi)`` except the sign flip."""
    lo, hi = col.lo, col.hi
    cands: list[int] = []
    if col.kind in (BOOL, SIGN):
        cands.append(1 - v)
    if col.kind == POWER:
        cands += [v * 2, v // 2]
    cands += [v + 1, v - 1, lo, hi - 1]
    if hi - lo > 4:
        cands += [rng.randrange(lo, hi) for _ in range(2)]
    if col.kind in (MAGNITUDE, REMAINDER, OTHER, EXPONENT):
        cands.append(-v)                                   # sign flip: outside the range on purpose
    out, seen = [], {v}
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        if c == -v or lo <= c < hi:
            out.append(c)
    return out


def run_mutations(target: Target, cases: Sequence[tuple[str, Any]], *, rng: random.Random | None = None,
                  fid: str = "M", limit_per_case: int | None = None) -> Finding:
    """Every column of every case's honest witness, every perturbation.  Accepted with another output -> SAT."""
    rng = rng or random.Random(0)
    t0 = time.time()
    sat, free, tried = [], [], 0
    for name, x in cases:
        y = target.reference(x)
        h = target.gen(x)
        cols = target.columns(h)
        if limit_per_case is not None and len(cols) > limit_per_case:
            cols = rng.sample(cols, limit_per_case)
        for col in cols:
            v = tree_get(h, col.path)
            if not isinstance(v, int):
                continue
            for v2 in perturbations(col, v, rng):
                tried += 1
                h2 = tree_set(h, col.path, v2)
                y2 = target.output_of(h2) if target.output_of else y
                if not target.accepts(x, y2, h2):
                    continue
                rec = {"case": name, "column": col.name, "from": v, "to": v2, "x": target.fmt_x(x)}
                if y2 != y:
                    rec.update(reference=target.fmt_y(y), accepted_output=target.fmt_y(y2))
                    sat.append(rec)
                else:
                    free.append(rec)
    status = "SAT" if sat else ("NOTE" if free else "PASS")
    return Finding(fid, target.name, "mutation", status, time.time() - t0, case=f"{len(cases)} cases, {tried} mutations",
                   detail={"accepted_wrong": sat, "free_columns": _dedupe(free), "mutations": tried})


def _dedupe(recs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in recs:
        key = (r["case"], r["column"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# (iv) two outputs, one input: coordinated random perturbations and the output neighbourhood
# ---------------------------------------------------------------------------------------------------------------------

def run_coordinated(target: Target, cases: Sequence[tuple[str, Any]], *, rounds: int = 200, width: int = 3,
                    rng: random.Random | None = None, fid: str = "X") -> Finding:
    """Per case: ``rounds`` random perturbations of ``width`` columns at once, plus every output neighbour with the
    honest witness (the only attack available against a checker with an empty witness)."""
    rng = rng or random.Random(1)
    t0 = time.time()
    sat, tried = [], 0
    for name, x in cases:
        y = target.reference(x)
        h = target.gen(x)
        for y2 in target.output_neighbours(x, y):
            tried += 1
            if y2 != y and target.accepts(x, y2, h):
                sat.append({"case": name, "x": target.fmt_x(x), "reference": target.fmt_y(y),
                            "accepted_output": target.fmt_y(y2), "witness": "honest"})
        cols = [c for c in target.columns(h) if isinstance(tree_get(h, c.path), int)]
        if not cols:
            continue
        for _ in range(rounds):
            tried += 1
            h2 = h
            picked = rng.sample(cols, min(width, len(cols)))
            changes = {}
            for col in picked:
                v = tree_get(h, col.path)
                opts = perturbations(col, v, rng)
                if not opts:
                    continue
                v2 = rng.choice(opts)
                h2 = tree_set(h2, col.path, v2)
                changes[col.name] = (v, v2)
            y2 = target.output_of(h2) if target.output_of else y
            if y2 != y and target.accepts(x, y2, h2):
                sat.append({"case": name, "x": target.fmt_x(x), "reference": target.fmt_y(y),
                            "accepted_output": target.fmt_y(y2), "changes": changes})
    return Finding(fid, target.name, "two-outputs-one-input (coordinated random)", "SAT" if sat else "PASS",
                   time.time() - t0, case=f"{len(cases)} cases, {tried} attempts", detail={"accepted_wrong": sat})


# ---------------------------------------------------------------------------------------------------------------------
# (iii) the z3 driver
# ---------------------------------------------------------------------------------------------------------------------

def z3_query(build: Callable[[Any], dict[str, Any]], *, timeout_ms: int = 20_000) -> tuple[str, dict | None, float]:
    """``build(solver)`` adds the constraints and returns the named terms to report.  Returns
    ``(status, model_values | None, seconds)`` with status SAT / UNSAT / UNKNOWN."""
    import z3

    s = z3.Solver()
    s.set("timeout", timeout_ms)
    report = build(s)
    t0 = time.time()
    res = s.check()
    dt = time.time() - t0
    if res == z3.sat:
        m = s.model()
        vals = {}
        for k, v in report.items():
            try:
                vals[k] = m.eval(v, model_completion=True).as_long()
            except Exception:  # noqa: BLE001
                vals[k] = str(m.eval(v, model_completion=True))
        return "SAT", vals, dt
    return ("UNSAT" if res == z3.unsat else "UNKNOWN"), None, dt
