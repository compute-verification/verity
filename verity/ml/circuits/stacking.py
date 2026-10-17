"""The stacking property (`verity/stacking-property` v1): P(c, R_in, L), a check of a Program's census
(`verity.ml.circuits.census`) that bounds what an audited block of its dynamic memory can carry.

A carrier is an audited block whose contents were chosen after the seed and hold another block's encoding. Such an encoding
reaches the circuit's values only as input bytes or through an encode inside the circuit, which needs roundings. P bounds
both, and how long a value can wait:

1. ``unclassified``: every gate's primitive is classified (`census.kind`), so no gate is free by omission;
2. ``fp-mma``: every matmul step is an exact s8 IMMA step (no FP tensor-core step: its accumulator rounds every product);
3. ``s8-may-overflow``: the interval pass proved every s8 IMMA step's sum within s32 from its operands' ranges and its
   reduction length, so no accumulation wraps or clamps;
4. ``uncovered``: every step lies in a window;
5. ``roundings``: in each window, roundings (casts, special functions, FP32 arithmetic, integer and bit operations,
   comparisons and selections, gathers) are at most c per byte the audit certified as read in that window;
6. ``products``: in each window, products of two run values (``rr`` and ``rsel``, scalar and inside matmul steps) are at
   most c_rr per certified byte read, when c_rr is given (None: not bounded, and the verdict says so in its numbers);
7. ``inputs``: in each window, the run-input bytes the Program first reads there are at most R_in;
8. ``lifetime``: no dynamic value, and no run input held past its first read, is read more than L steps after it is
   written.

The certified reads are the audit's (a parameter, `Window.reads`), not the Program's. Bits become bytes exactly (bits / 8),
and every comparison is exact (`fractions.Fraction`).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from . import census as C

TAG = "verity/stacking-property"
VERSION = 1
SCHEMA = f"{TAG}/v{VERSION}"


@dataclass(frozen=True)
class Window:
    """A window: a range of the census's steps, and the bytes the audit certified as read in it."""

    steps: range
    reads: int


@dataclass(frozen=True)
class Verdict:
    """Whether P holds, why not (``reasons``: each a ``code`` with its numbers) and the numbers it read."""

    ok: bool
    reasons: tuple[dict, ...]
    numbers: dict

    def to_json(self) -> dict:
        return {"schema": SCHEMA, "ok": self.ok, "reasons": list(self.reasons), "numbers": self.numbers}


def roundings(counts: dict) -> int:
    """The rounding-class gates of a census's counts."""
    return sum(counts["roundings"].values()) + counts["compare_select"] + counts["gathers"]


def products(counts: dict) -> int:
    """The products of two run (or selected) values of a census's counts, scalar and inside matmul steps."""
    return counts["products"]["rr"] + counts["products"]["rsel"]


def _q(x: Fraction) -> str:
    return str(x)


def check(census: C.Census | dict, *, c: Any, R_in: Any, L: int, windows: list[Window] | tuple[Window, ...],
          c_rr: Any = None) -> Verdict:
    """P(c, R_in, L) with an optional products budget c_rr, on `census` (a `Census` or its JSON) under the audit's `windows`.
    c and c_rr are per certified byte read, R_in is bytes per window, L is steps. A census of another schema or class table,
    or a window outside its steps, is refused (ValueError)."""
    j = census.to_json() if isinstance(census, C.Census) else census
    if j.get("schema") != C.SCHEMA or j.get("classes") != C.digest():
        raise ValueError(f"a {C.SCHEMA} census under class table {C.digest()[:16]}..., not {j.get('schema')!r} "
                         f"under {str(j.get('classes'))[:16]}...")
    c, R_in = Fraction(c), Fraction(R_in)
    c_rr = None if c_rr is None else Fraction(c_rr)
    steps, total = j["steps"], j["total"]
    n = len(steps)
    for w in windows:
        if not isinstance(w.steps, range) or w.steps.step != 1 or not 0 <= w.steps.start < w.steps.stop <= n or w.reads < 0:
            raise ValueError(f"a window is a non-empty range of the census's {n} steps with reads >= 0, not {w}")
    reasons: list[dict] = []
    if total["unclassified"]:
        reasons.append({"code": "unclassified", "primitives": total["unclassified"]})
    fp = {fid: m["steps"] for fid, m in total["mma"].items() if m["kind"] != "s8-imma"}
    if fp:
        reasons.append({"code": "fp-mma", "steps": fp})
    unproven = {fid: m["steps"] - m["exact"] for fid, m in total["mma"].items() if m["kind"] == "s8-imma" and m["exact"] < m["steps"]}
    if unproven:
        reasons.append({"code": "s8-may-overflow", "steps": unproven, "sites": j["may_overflow"]})
    covered = set().union(*(set(w.steps) for w in windows)) if windows else set()
    if len(covered) < n:
        reasons.append({"code": "uncovered", "steps": sorted(set(range(n)) - covered)})
    rows = []
    for w in windows:
        sl = [steps[i] for i in w.steps]
        r = sum(roundings(s) for s in sl)
        p = sum(products(s) for s in sl)
        inb = Fraction(sum(s["memory"]["inputs"]["run_bits"] for s in sl), 8)
        row = {"steps": [w.steps.start, w.steps.stop], "reads": w.reads, "roundings": r, "products": p, "input_bytes": _q(inb)}
        rows.append(row)
        if r > c * w.reads:
            reasons.append({"code": "roundings", **row, "c": _q(c)})
        if c_rr is not None and p > c_rr * w.reads:
            reasons.append({"code": "products", **row, "c_rr": _q(c_rr)})
        if inb > R_in:
            reasons.append({"code": "inputs", **row, "R_in": _q(R_in)})
    life = j["memory"]["max_lifetime"]
    if life > L:
        reasons.append({"code": "lifetime", "max_lifetime": life, "L": L})
    numbers = {"c": _q(c), "c_rr": None if c_rr is None else _q(c_rr), "R_in": _q(R_in), "L": L, "windows": rows,
               "max_lifetime": life, "macs": total["macs"], "roundings": roundings(total), "products": products(total)}
    return Verdict(not reasons, tuple(reasons), numbers)


__all__ = ["SCHEMA", "TAG", "VERSION", "Verdict", "Window", "check", "products", "roundings"]
