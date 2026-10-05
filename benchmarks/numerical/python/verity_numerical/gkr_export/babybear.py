"""The BabyBear export of the WP2 checker for A-GKR (``archive/gkr --features babybear``).

Same machinery as :mod:`.circuit` (the gadgets run once on symbolic values), but the gadgets are the *embedded*
ones of :mod:`verity_numerical.redteam.babybear` under the six rules of ``reports/babybear_rules.md`` (``NORM_VBND``,
proven redundant, dropped):

* ``LIMB16``: the FP32 accumulator in (``x.c_lo``, ``x.c_hi``), out (``y_lo``, ``y_hi``) and the epilogue's input
  (``y32_lo``, ``y32_hi``) are two public 16-bit limb columns each; the fraction hints ``acc.f_lo/f_hi`` and
  ``pack.low_lo/low_hi`` replace ``acc.f`` / ``pack.low``; the epilogue's ``hi``/``lo`` hints disappear.
* ``LT_DIRECT`` / ``QUOT_TIGHT``: the alignment bounds and the leading-bit window are range checks on *linear*
  wires against the table's bound column (POW ``pow``/``ipow``, ALIGN ``sr``/``isr``, LEAD's new ``span``): no
  product wire, two 16-bit chunk lookups per 25/30-bit range.
* ``NORM_MBND``: NORM gains ``mbnd``; one 30-bit range per group.
* ``V_PREMISE``: a premise of the composition (nothing to emit).

Every predicate is read modulo ``p = 2^31 - 2^27 + 1``: coefficients and witness values are reduced mod p, range
chunks and looked-up rows are computed from the canonical representative (what a forger would do), so the
circuit's accepted set is exactly :class:`~verity_numerical.redteam.babybear.BabyBearCtx`'s.  The text format and
the witness file format are the ones ``archive/gkr/src/circuit.rs`` parses (values < p in u64 slots).
"""
from __future__ import annotations

import copy
import random
from collections.abc import Sequence
from pathlib import Path

from verity_numerical.checker import REAL, Params, Reject
from verity_numerical.checker.ctx import IntCtx, Table
from verity_numerical.redteam.babybear import (
    ALL_RULES,
    LIMB,
    Embedding,
    check_step_bb,
    gen_step_bb,
    limbs,
)

from .circuit import (
    CHUNK,
    CircuitCtx,
    _get,
    _leaves,
    _set,
    export_circuit,
    range_table_width,
    write_witness,
)

__all__ = [
    "BABYBEAR",
    "EMBEDDING",
    "BabyBearCircuitCtx",
    "build_epilogue_circuit_bb",
    "build_unit_circuit_bb",
    "epilogue_witness_bb",
    "export_circuit_bb",
    "limb_hints",
    "unit_witness_bb",
    "verify_unit_bb",
    "write_witness_bb",
]

BABYBEAR = 2**31 - 2**27 + 1
#: The six rules with the redundant ``NORM_VBND`` dropped (``babybear_rules.md`` section 6).
EMBEDDING = Embedding(ALL_RULES - {"NORM_VBND"})
#: Hint leaves the embedded unit never reads (their limbs replace them).
UNUSED_HINTS = ("h.acc.f", "h.pack.low")


class BabyBearCircuitCtx(CircuitCtx):
    """:class:`CircuitCtx` with the BabyBear window discipline: a non-power-of-two range ``[lo, hi)`` of ``n = hi - lo``
    values is two one-sided ``w``-bit chunks (``x - lo`` and ``hi - 1 - x``), sound iff ``2^(w+1) - n <= p``: an
    ``x - lo = t`` in ``[n, 2^w)`` makes ``hi - 1 - x = -(t - n + 1)`` read as ``p - (t - n + 1) >= p - (2^w - n)``,
    which the second chunk rejects exactly when that is ``>= 2^w``.  A range without a sound form is refused.
    The 5-column NORM table of the embedding loses its unused ``vbnd`` column."""

    modulus = BABYBEAR

    def rng(self, name: str, x, lo: int, hi: int) -> None:
        if not (isinstance(x, int) or x.is_const):
            n = hi - lo
            w = max(1, (n - 1).bit_length())
            if n != 1 << w and 2 ** (w + 1) - n > self.modulus:
                raise Reject(f"{name}: a {w}-bit non-power-of-two range of {n} values has no sound two-sided form mod p")
            if 1 << w > self.modulus:
                raise Reject(f"{name}: {w}-bit range wider than the field")
        super().rng(name, x, lo, hi)

    def lookup(self, name: str, table: Table, key) -> tuple:
        if table.name == "NORM" and table.outs[-1] == "vbnd" and not EMBEDDING.has("NORM_VBND"):
            base = Table("NORM", table.lo, table.hi, table.outs[:-1], lambda k, t=table: t.fn(k)[:-1])
            return (*super().lookup(name, base, key), 0)
        return super().lookup(name, table, key)


def _paths(sample: dict) -> list[str]:
    paths: list[str] = []
    _leaves(sample, "h", paths)
    return [q for q in paths if q not in UNUSED_HINTS]


def build_unit_circuit_bb(p: Params = REAL) -> tuple[BabyBearCircuitCtx, list[str]]:
    """The embedded transition unit as a circuit; columns ``one, x.c_lo, x.c_hi, x.a[..], x.b[..], y_lo, y_hi``,
    then the hint paths returned (``h.acc.s``, ..., ``h.acc.f_lo``, ...)."""
    ctx = BabyBearCircuitCtx()
    k = sum(p.groups)
    c_lo = ctx.column("x.c_lo")
    c_hi = ctx.column("x.c_hi")
    a = [ctx.column(f"x.a[{i}]") for i in range(k)]
    b = [ctx.column(f"x.b[{i}]") for i in range(k)]
    y_lo = ctx.column("y_lo")
    y_hi = ctx.column("y_hi")
    sample = gen_step_bb(p, 0, [0] * k, [0] * k)
    paths = _paths(sample)
    h_sym = copy.deepcopy(sample)
    for path in paths:
        _set(h_sym, path, ctx.column(path))
    got_lo, got_hi = check_step_bb(ctx, p, (c_lo, c_hi), a, b, h_sym, EMBEDDING)
    ctx.eq("y_lo", got_lo, y_lo)
    ctx.eq("y_hi", got_hi, y_hi)
    return ctx, paths


def build_epilogue_circuit_bb(p: Params = REAL) -> tuple[BabyBearCircuitCtx, list[str]]:
    """``Embedding.epilogue`` as a circuit: columns ``one, y32_lo, y32_hi, y16, hints`` (``hi2, lsb, carry, rem``)."""
    from verity_numerical.checker.gadgets import Epilogue

    ctx = BabyBearCircuitCtx()
    y32_lo = ctx.column("y32_lo")
    y32_hi = ctx.column("y32_hi")
    y16 = ctx.column("y16")
    sample = {k: v for k, v in Epilogue.gen(p, 0).items() if k not in ("hi", "lo")}
    paths: list[str] = []
    _leaves(sample, "h", paths)
    h_sym = copy.deepcopy(sample)
    for path in paths:
        _set(h_sym, path, ctx.column(path))
    ctx.rng("y32_lo", y32_lo, 0, 2**LIMB)
    ctx.rng("y32_hi", y32_hi, 0, 2**LIMB)
    got = EMBEDDING.epilogue(ctx, p, (y32_lo, y32_hi), h_sym)
    ctx.eq("y16", got, y16)
    return ctx, paths


def limb_hints(h: dict) -> dict:
    """The WP2 hint tree completed with the limb hints when absent (canonical mod p, as a forger would split)."""
    h = copy.deepcopy(h)
    if "f_lo" not in h["acc"]:
        h["acc"]["f_lo"], h["acc"]["f_hi"] = limbs(int(h["acc"]["f"]) % BABYBEAR)
    if "low_lo" not in h["pack"]:
        h["pack"]["low_lo"], h["pack"]["low_hi"] = limbs(int(h["pack"]["low"]) % BABYBEAR)
    return h


def _derive(ctx: CircuitCtx, vals: list[int | None]) -> list[int]:
    """Derived columns from the canonical representatives mod p (range chunks of ``v mod p``, rows at the canonical key)."""
    P = BABYBEAR
    cache: dict[int, int] = {}
    for col, d in enumerate(ctx.derived):
        if d is None:
            continue
        kind = d[0]
        if kind == "const":
            vals[col] = d[1]
        elif kind == "lin":
            vals[col] = ctx.eval_lin(d[1], vals, cache) % P
        elif kind == "lo16":
            vals[col] = (ctx.eval_lin(d[1], vals, cache) % P) & (2**CHUNK - 1)
        elif kind == "hi16":
            vals[col] = (ctx.eval_lin(d[1], vals, cache) % P) >> CHUNK
        elif kind == "row":
            _, table, key, idx = d
            kv = (ctx.eval_lin(key, vals, cache) - table.lo) % P + table.lo
            vals[col] = table.row(kv)[idx] if table.lo <= kv < table.hi else 0
    return [int(v) % P for v in vals]  # type: ignore[arg-type]


def unit_witness_bb(ctx: CircuitCtx, paths: list[str], c: int, a: Sequence[int], b: Sequence[int], y: int, h: dict) -> list[int]:
    """Every column of one unit (mod p): the words as limbs, the hints (limb hints completed), the derived columns."""
    k = len(a)
    h = limb_hints(h)
    vals: list[int | None] = [None] * len(ctx.cols)
    vals[0] = 1
    vals[1], vals[2] = limbs(int(c) % 2**32)
    for i in range(k):
        vals[3 + i] = int(a[i])
        vals[3 + k + i] = int(b[i])
    vals[3 + 2 * k], vals[4 + 2 * k] = limbs(int(y) % 2**32)
    base = 5 + 2 * k
    for i, path in enumerate(paths):
        vals[base + i] = int(_get(h, path))
    return _derive(ctx, vals)


def epilogue_witness_bb(ctx: CircuitCtx, paths: list[str], y32: int, y16: int, h: dict) -> list[int]:
    vals: list[int | None] = [None] * len(ctx.cols)
    vals[0] = 1
    vals[1], vals[2] = limbs(int(y32) % 2**32)
    vals[3] = int(y16)
    for i, path in enumerate(paths):
        vals[4 + i] = int(_get(h, path))
    return _derive(ctx, vals)


def verify_unit_bb(ctx: CircuitCtx, vals: list[int]) -> str | None:
    """The circuit's own check on one row modulo p: the first failing assertion / query, or None."""
    P = BABYBEAR
    cache: dict[int, int] = {}
    for name, x in ctx.asserts:
        if ctx.eval_lin(x, vals, cache) % P:
            return f"assert {name}"
    for i, (tname, lins) in enumerate(ctx.queries):
        row = tuple(ctx.eval_lin(x, vals, cache) % P for x in lins)
        if (w := range_table_width(tname)) is not None:
            if row[0] >= 2**w:
                return f"query {i} {tname} {row[0]}"
        else:
            t = ctx.tables[tname]
            key = (row[0] - t.lo) % P + t.lo
            if not t.lo <= key < t.hi or tuple(v % P for v in t.row(key)) != row[1:]:
                return f"query {i} {tname} {row}"
    return None


def export_circuit_bb(ctx: CircuitCtx, path: Path) -> None:
    export_circuit(ctx, path, modulus=BABYBEAR)


def write_witness_bb(units: list[list[int]], path: Path) -> None:
    write_witness(units, path, modulus=BABYBEAR)


def check_embedded(p: Params, c: int, a: Sequence[int], b: Sequence[int], y: int, h: dict) -> None:
    """The embedded checker on Python ints (integer semantics): raises ``Reject``."""
    ctx = IntCtx()
    got = check_step_bb(ctx, p, limbs(c), list(a), list(b), limb_hints(h), EMBEDDING)
    y_lo, y_hi = limbs(y)
    ctx.eq("y_lo", got[0], y_lo)
    ctx.eq("y_hi", got[1], y_hi)


def random_finite_unit_bb(rng: random.Random, p: Params = REAL):
    from .circuit import random_finite_unit

    c, a, b, y, h = random_finite_unit(rng, p)
    return c, a, b, y, limb_hints(h)
