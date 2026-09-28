"""Adapters: the WP2 checker (``verity_numerical.checker``) as a harness target, over the integers or over F_p.

The public input is ``x = (c, a, b)`` (one FP32 word and the 2k operand words); the witness ``h`` is the hint tree
``gen_step`` produces, every leaf a hint column with the range its gadget's ``contract()`` declares; the claimed
output ``y`` is the FP32 word.  ``check`` runs every predicate of ``check_step_expr`` and accepts iff none is
violated and the word the predicates form equals ``y``.

Two evaluators of the five predicates on Python ints:

    IntCtx (the checker's own)   integers: the reference semantics
    FieldCtx(p)                  F_p as a backend embeds it: every ``eq`` a congruence, every ``rng(x, lo, hi)`` the
                                 check ``(x - lo) mod p < hi - lo`` on the canonical representative, ``boolean`` mod p,
                                 lookup keys canonical (offset by the table's ``lo``), ``lt_pow`` the product realisation
                                 ``x * ipow mod p < 2^B`` (checker F1: sound iff p > 2^(2B)).  Hints may be any residue.

``LaxCtx`` evaluates the same expressions without checking anything: it yields the word a (possibly forged) witness
asserts, which the harness compares to the reference.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from verity_numerical.checker import REAL, Params, Reject
from verity_numerical.checker.ctx import IntCtx, Table
from verity_numerical.checker.gadgets import (
    AlignProduct,
    AlignState,
    Bf16Decode,
    Epilogue,
    Fp32Decode,
    GroupMax,
    LeadingBit,
    Normalise,
    Pack,
    Product,
    SignedSum,
    StateOut,
)
from verity_numerical.checker.step import (
    check_step_expr,
    check_vu_expr,
    gen_step,
    gen_vu,
)

from .harness import (
    BOOL,
    MAGNITUDE,
    OTHER,
    REMAINDER,
    SELECTOR,
    SHIFT,
    SIGN,
    Column,
    Target,
)
from .relation_target import GOLDILOCKS, M61, P40, fmt_x, fp32_neighbours

__all__ = [
    "GOLDILOCKS",
    "M31",
    "M61",
    "P40",
    "FieldCtx",
    "LaxCtx",
    "checker_columns",
    "checker_target",
    "checker_vu_target",
    "field_check",
    "step_cases",
    "vu_columns",
]

M31 = (1 << 31) - 1


# ---------------------------------------------------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------------------------------------------------

class FieldCtx(IntCtx):
    """The checker's predicates read over F_p on Python ints (see the module docstring)."""

    def __init__(self, p: int):
        self.p = p

    def rng(self, name: str, x: int, lo: int, hi: int) -> None:
        if type(x) is not int or not (x - lo) % self.p < hi - lo:
            raise Reject(f"{name}: {x!r} not in [{lo}, {hi}) mod p")

    def eq(self, name: str, a: int, b: int) -> None:
        if (a - b) % self.p:
            raise Reject(f"{name}: {a!r} != {b!r} mod p")

    def boolean(self, name: str, b: int) -> None:
        if type(b) is not int or b % self.p not in (0, 1):
            raise Reject(f"{name}: {b!r} is not a bit mod p")

    def lookup(self, name: str, table: Table, key: int) -> tuple[int, ...]:
        k = (key - table.lo) % self.p + table.lo
        try:
            return table.row(k)
        except KeyError:
            raise Reject(f"{name}: key {key!r} is off table {table.name} mod p") from None

    def lt_pow(self, name: str, x: int, ipow: int, bits: int) -> None:
        self.rng(name, (x * ipow) % self.p, 0, 2**bits)


class LaxCtx(IntCtx):
    """Checks nothing; forms the output expression.  An off-table key yields the table's first row."""

    def rng(self, name, x, lo, hi):
        pass

    def eq(self, name, a, b):
        pass

    def boolean(self, name, b):
        pass

    def lookup(self, name, table: Table, key):
        try:
            return table.row(key)
        except KeyError:
            return table.fn(table.lo)

    def lt_pow(self, name, x, ipow, bits):
        pass


def field_check(p: Params, field: int | None, c: int, a: Sequence[int], b: Sequence[int], y: int, h: dict) -> tuple[bool, str]:
    """``(accepted, first violated predicate or '')`` for the unit, over the integers (``field=None``) or F_p."""
    ctx = FieldCtx(field) if field else IntCtx()
    try:
        got = check_step_expr(ctx, p, c, a, b, h)
        ctx.eq("y", got, y)
    except Reject as e:
        return False, str(e)
    except Exception as e:  # noqa: BLE001 -- a forged hint may break the arithmetic itself
        return False, f"raised {e!r}"
    return True, ""


def _step_output(p: Params, field: int | None, x, h: dict) -> int:
    c, a, b = x
    try:
        y = check_step_expr(LaxCtx(), p, c, a, b, h)
    except Exception:  # noqa: BLE001
        return -1
    return y % field if field else y


# ---------------------------------------------------------------------------------------------------------------------
# The witness shape from the contracts
# ---------------------------------------------------------------------------------------------------------------------

_KIND = {"s": SIGN, "sgn": SIGN, "nz": BOOL, "z": BOOL, "clamp": BOOL, "ovf": BOOL, "hidden": BOOL, "lsb": BOOL,
         "carry": BOOL, "sel": SELECTOR, "E": SHIFT, "l": SHIFT, "d": SHIFT, "q": MAGNITUDE, "M": MAGNITUDE,
         "r": REMAINDER, "rem": REMAINDER, "t": MAGNITUDE, "f": MAGNITUDE, "low": MAGNITUDE, "hi": MAGNITUDE,
         "lo": MAGNITUDE, "hi2": MAGNITUDE}


def _cols_of(contract, base: tuple, node: dict, out: list[Column]) -> None:
    for name, v in node.items():
        if isinstance(v, list):
            for i, _ in enumerate(v):
                lo, hi = contract.hints[f"{name}[{i}]"]
                out.append(Column(base + (name, i), lo, hi, _KIND.get(name, OTHER)))
        else:
            lo, hi = contract.hints[name]
            out.append(Column(base + (name,), lo, hi, _KIND.get(name, OTHER)))


def checker_columns(p: Params, h: dict, base: tuple = ()) -> list[Column]:
    """Every hint of a unit witness with the range its gadget contract declares."""
    cols: list[Column] = []
    _cols_of(Fp32Decode.contract(p), base + ("acc",), h["acc"], cols)
    for side in ("a", "b"):
        for i, hw in enumerate(h[side]):
            _cols_of(Bf16Decode.contract(p), base + (side, i), hw, cols)
    for i, hp in enumerate(h["prod"]):
        _cols_of(Product.contract(p), base + ("prod", i), hp, cols)
    for g, hg in enumerate(h["groups"]):
        gb = base + ("groups", g)
        _cols_of(GroupMax.contract(p, p.groups[g] + 1), gb + ("max",), hg["max"], cols)
        _cols_of(AlignState.contract(p), gb + ("acc_align",), hg["acc_align"], cols)
        for i, hq in enumerate(hg["prod_align"]):
            _cols_of(AlignProduct.contract(p), gb + ("prod_align", i), hq, cols)
        _cols_of(SignedSum.contract(p, p.groups[g]), gb + ("sum",), hg["sum"], cols)
        _cols_of(LeadingBit.contract(p), gb + ("lead",), hg["lead"], cols)
        _cols_of(Normalise.contract(p), gb + ("norm",), hg["norm"], cols)
        _cols_of(StateOut.contract(p), gb + ("out",), hg["out"], cols)
    _cols_of(Pack.contract(p), base + ("pack",), h["pack"], cols)
    return cols


def vu_columns(p: Params, h: dict) -> list[Column]:
    cols: list[Column] = []
    for s, hs in enumerate(h["steps"]):
        cols += checker_columns(p, hs, ("steps", s))
    _cols_of(Epilogue.contract(p), ("epilogue",), h["epilogue"], cols)
    return cols


# ---------------------------------------------------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------------------------------------------------

def _label(field: int | None) -> str:
    return "Z" if field is None else {P40: "p40", M31: "m31", M61: "m61", GOLDILOCKS: "goldilocks"}.get(field, f"p={field}")


def checker_target(p: Params = REAL, field: int | None = None) -> Target:
    """The transition unit ``check_step`` as a harness target."""
    from verity_numerical.checker.reference import dot

    return Target(
        name=f"checker[unit,{_label(field)}]",
        check=lambda x, y, h: field_check(p, field, x[0], list(x[1]), list(x[2]), y, h)[0],
        gen=lambda x: gen_step(p, x[0], list(x[1]), list(x[2])),
        reference=lambda x: dot(p, x[0], list(x[1]), list(x[2])) % field if field else dot(p, x[0], list(x[1]), list(x[2])),
        columns=lambda h: checker_columns(p, h),
        output_of=lambda h: None,       # replaced below (needs x)
        output_neighbours=fp32_neighbours,
        fmt_x=fmt_x, fmt_y=lambda y: f"{y:#010x}" if isinstance(y, int) and y >= 0 else repr(y), fmt_h=lambda h: "<unit hints>",
    )


class _BoundOutput:
    """The harness calls ``output_of(h)`` without ``x``; the checker's word is an expression of both, so the target
    remembers the case under test (the harness is single-threaded and sets it through ``gen``)."""

    def __init__(self, p: Params, field: int | None):
        self.p, self.field, self.x = p, field, None

    def gen(self, x):
        self.x = x
        return gen_step(self.p, x[0], list(x[1]), list(x[2]))

    def output_of(self, h):
        return _step_output(self.p, self.field, self.x, h)


def bound_checker_target(p: Params = REAL, field: int | None = None) -> Target:
    """``checker_target`` with ``output_of`` bound to the last generated case (what the mutation runs need)."""
    t = checker_target(p, field)
    bo = _BoundOutput(p, field)
    t.gen = bo.gen
    t.output_of = bo.output_of
    return t


def checker_vu_target(p: Params = REAL, field: int | None = None) -> Target:
    """The VU ``check_vu`` (word-chained units + epilogue) as a target; ``x = (a, b)``, ``y`` the BF16 word."""
    from verity_numerical.checker.reference import vu

    def ref(x):
        y = vu(p, list(x[0]), list(x[1]))
        return y % field if field else y

    def check(x, y, h):
        ctx = FieldCtx(field) if field else IntCtx()
        try:
            ctx.eq("y16", check_vu_expr(ctx, p, list(x[0]), list(x[1]), h), y)
        except Exception:  # noqa: BLE001
            return False
        return True

    state = {"x": None}

    def gen(x):
        state["x"] = x
        return gen_vu(p, list(x[0]), list(x[1]))

    def output_of(h):
        try:
            y = check_vu_expr(LaxCtx(), p, list(state["x"][0]), list(state["x"][1]), h)
        except Exception:  # noqa: BLE001
            return -1
        return y % field if field else y

    return Target(name=f"checker[vu,{_label(field)}]", check=check, gen=gen, reference=ref,
                  columns=lambda h: vu_columns(p, h), output_of=output_of,
                  fmt_x=lambda x: {"a": "".join(f"{w:04x}" for w in x[0]), "b": "".join(f"{w:04x}" for w in x[1])},
                  fmt_y=lambda y: f"{y:#06x}" if isinstance(y, int) and y >= 0 else repr(y))


# ---------------------------------------------------------------------------------------------------------------------
# checker v2 (verity_numerical.checker.v2): the same relation, the lookup-centred witness
# ---------------------------------------------------------------------------------------------------------------------

_KIND_V2 = {"hdr": MAGNITUDE, "M_hi": MAGNITUDE, "M_lo": MAGNITUDE, "v_hi": MAGNITUDE, "v_lo": MAGNITUDE}


def checker_columns_v2(p: Params, h: dict, base: tuple = ()) -> list[Column]:
    """Every hint of a v2 unit witness with its contract range (``acc`` only for the word-in unit; ``acc_split``
    is ``None`` for a unit on the constant zero state)."""
    from verity_numerical.checker.v2 import gadgets as G

    kinds = {**_KIND, **_KIND_V2}
    cols: list[Column] = []

    def add(contract, b, node):
        for name, v in node.items():
            lo, hi = contract.hints[name]
            cols.append(Column(b + (name,), lo, hi, kinds.get(name, OTHER)))

    if h.get("acc") is not None:
        add(G.StateDecode.contract(p), base + ("acc",), h["acc"])
    for g, hg in enumerate(h["groups"]):
        gb = base + ("groups", g)
        lo, hi = G.GroupMax.contract(p, p.groups[g] + 1).hints["E"]
        cols.append(Column(gb + ("E",), lo, hi, SHIFT))
        if hg.get("acc_split") is not None:
            add(G.StateShift.contract(p), gb + ("acc_split",), hg["acc_split"])
        add(G.SignedSum.contract(p, p.groups[g]), gb + ("sum",), hg["sum"])
    return cols


def vu_columns_v2(p: Params, h: dict) -> list[Column]:
    cols: list[Column] = []
    for s, hs in enumerate(h["steps"]):
        cols += checker_columns_v2(p, hs, ("steps", s))
    _cols_of(Epilogue.contract(p), ("epilogue",), h["epilogue"], cols)
    return cols


def bound_checker_target_v2(p: Params = REAL, field: int | None = None) -> Target:
    """The v2 transition unit as a harness target (``output_of`` bound to the last generated case)."""
    from verity_numerical.checker.reference import dot
    from verity_numerical.checker.v2 import step as S2

    state = {"x": None}

    def check(x, y, h):
        ctx = FieldCtx(field) if field else IntCtx()
        try:
            ctx.eq("y", S2.check_step_expr(ctx, p, x[0], list(x[1]), list(x[2]), h), y)
        except Exception:  # noqa: BLE001
            return False
        return True

    def gen(x):
        state["x"] = x
        return S2.gen_step(p, x[0], list(x[1]), list(x[2]))

    def output_of(h):
        x = state["x"]
        try:
            y = S2.check_step_expr(LaxCtx(), p, x[0], list(x[1]), list(x[2]), h)
        except Exception:  # noqa: BLE001
            return -1
        return y % field if field else y

    return Target(
        name=f"checker-v2[unit,{_label(field)}]", check=check, gen=gen,
        reference=lambda x: dot(p, x[0], list(x[1]), list(x[2])) % field if field else dot(p, x[0], list(x[1]), list(x[2])),
        columns=lambda h: checker_columns_v2(p, h), output_of=output_of, output_neighbours=fp32_neighbours,
        fmt_x=fmt_x, fmt_y=lambda y: f"{y:#010x}" if isinstance(y, int) and y >= 0 else repr(y), fmt_h=lambda h: "<unit hints>",
    )


def checker_vu_target_v2(p: Params = REAL, field: int | None = None) -> Target:
    """The v2 VU (state-chained units, one Pack, the epilogue) as a target; ``x = (a, b)``, ``y`` the BF16 word."""
    from verity_numerical.checker.reference import vu
    from verity_numerical.checker.v2 import step as S2

    def ref(x):
        y = vu(p, list(x[0]), list(x[1]))
        return y % field if field else y

    def check(x, y, h):
        ctx = FieldCtx(field) if field else IntCtx()
        try:
            ctx.eq("y16", S2.check_vu_expr(ctx, p, list(x[0]), list(x[1]), h), y)
        except Exception:  # noqa: BLE001
            return False
        return True

    state = {"x": None}

    def gen(x):
        state["x"] = x
        return S2.gen_vu(p, list(x[0]), list(x[1]))

    def output_of(h):
        try:
            y = S2.check_vu_expr(LaxCtx(), p, list(state["x"][0]), list(state["x"][1]), h)
        except Exception:  # noqa: BLE001
            return -1
        return y % field if field else y

    return Target(name=f"checker-v2[vu,{_label(field)}]", check=check, gen=gen, reference=ref,
                  columns=lambda h: vu_columns_v2(p, h), output_of=output_of,
                  fmt_x=lambda x: {"a": "".join(f"{w:04x}" for w in x[0]), "b": "".join(f"{w:04x}" for w in x[1])},
                  fmt_y=lambda y: f"{y:#06x}" if isinstance(y, int) and y >= 0 else repr(y))


def step_cases(n_golden: int = 12, seed: int = 0) -> list[tuple[str, Any]]:
    """Structured Ampere cases plus a sample of golden records, as ``(name, (c, a, b))``."""
    import random

    from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16 as AMP

    from . import cases as C

    golden, _ = C.golden_cases()
    rng = random.Random(seed)
    return C.structured_cases(AMP) + rng.sample(golden, n_golden)
