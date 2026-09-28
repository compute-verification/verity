"""Adapters: ``verity.ml.circuits.silicon.relation`` (over the integers or a prime field) and ``silicon.tc_dot`` as harness targets.

The relation's acceptance predicate, as attacked here: the public input is ``(acc, a, b)``; the witness ``h`` is the
dict tree ``step_witness`` produces, with ``h["in"]`` bound to ``decode_state(acc)`` and ``h["a"], h["b"]`` to the
operands (public); the claimed output ``y`` is ``pack_state`` of the last group's ``out`` state.  ``check`` runs every
predicate of ``check_step`` and accepts iff none is violated.

``FieldCensus`` evaluates the same predicates over F_p exactly as the module's docstring says they are meant: ``mul``
and ``lin`` identities modulo ``p``, lookups and range chunks on canonical representatives in ``[0, p)``.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from verity.ml.circuits.silicon import relation as R
from verity.ml.circuits.silicon.silicon import Pipeline, tc_dot
from verity.ml.circuits.silicon.total import tc_dot_total

from .harness import BOOL, EXPONENT, MAGNITUDE, OTHER, POWER, REMAINDER, SELECTOR, SHIFT, SIGN, Column, Target

__all__ = ["FieldCensus", "GOLDILOCKS", "M61", "P40", "WeakenedCensus", "fp32_neighbours", "relation_columns",
           "relation_target", "silicon_target"]

P40 = 1099511627791                 # the first prime above 2^40 -- the docstring's "field of at least 40 bits"
M61 = (1 << 61) - 1                 # Mersenne 61
GOLDILOCKS = (1 << 64) - (1 << 32) + 1


class FieldCensus(R.Census):
    """``Census`` over F_p: ``a*b = c`` and ``sum = 0`` modulo ``p``; lookup keys, outputs and ranged values canonical."""

    def __init__(self, p: int, strict: bool = True):
        super().__init__(strict=strict)
        self.p = p

    def MUL(self, name: str, a: int, b: int, c: int) -> None:
        self.mul += 1
        if (a * b - c) % self.p:
            self._fail(f"{name}: {a} * {b} != {c} (mod p)")

    def LIN(self, name: str, residual: int) -> None:
        self.lin += 1
        if residual % self.p:
            self._fail(f"{name}: linear residual {residual} (mod p)")

    def LOOKUP(self, table: R.Table, key: tuple[int, ...], outs: tuple[int, ...]) -> None:
        """Keys are canonical representatives; a table row's (possibly negative) integer outputs are field elements."""
        self.lookups[table.name] += 1
        key = tuple(k % self.p for k in key)
        try:
            got = table(*key)
        except KeyError:
            self._fail(f"{table.name}: key {key} off the table")
            return
        if tuple(g % self.p for g in got) != tuple(o % self.p for o in outs):
            self._fail(f"{table.name}{key}: {tuple(got)} != {tuple(outs)} (mod p)")

    def RANGE(self, name: str, x: int, bits: int) -> None:
        super().RANGE(name, x % self.p, bits)

    def ISZERO(self, name: str, x: int, bits: int, z: int) -> None:
        super().ISZERO(name, x % self.p, bits, z % self.p)


class WeakenedCensus(R.Census):
    """A ``Census`` that skips the predicates whose name starts with one of ``dropped`` (to show the harness notices)."""

    def __init__(self, dropped: tuple[str, ...], strict: bool = True):
        super().__init__(strict=strict)
        self.dropped = dropped

    def _skip(self, name: str) -> bool:
        return any(name.startswith(d) for d in self.dropped)

    def MUL(self, name: str, a: int, b: int, c: int) -> None:
        if not self._skip(name):
            super().MUL(name, a, b, c)

    def LIN(self, name: str, residual: int) -> None:
        if not self._skip(name):
            super().LIN(name, residual)

    def RANGE(self, name: str, x: int, bits: int) -> None:
        if not self._skip(name):
            super().RANGE(name, x, bits)


@dataclass(frozen=True)
class RelationSemantics:
    pipe: Pipeline
    align: str = R.ALIGN_DIV
    field: int | None = None            # None: the integers
    dropped: tuple[str, ...] = ()       # predicates removed (harness self-test only)

    def census(self, strict: bool = False) -> R.Census:
        if self.field is not None:
            return FieldCensus(self.field, strict=strict)
        if self.dropped:
            return WeakenedCensus(self.dropped, strict=strict)
        return R.Census(strict=strict)

    @property
    def label(self) -> str:
        f = "Z" if self.field is None else f"F_p(p={self.field})"
        return f"relation[{self.pipe.name},{self.align},{f}{',dropped=' + ','.join(self.dropped) if self.dropped else ''}]"


def relation_check(sem: RelationSemantics, x, y: int, h: dict) -> tuple[bool, list[str]]:
    """Every predicate of ``check_step`` plus the public-input binding.  Returns ``(accepted, violations)``."""
    acc, a, b = x
    viol: list[str] = []
    if list(h["a"]) != list(a) or list(h["b"]) != list(b):
        viol.append("public operands differ")
    if h["in"] != R.decode_state(acc, sem.pipe):
        viol.append("input state is not decode_state(acc)")
    if len(h["groups"]) != len(sem.pipe.groups):
        viol.append("group count")
    c = sem.census(strict=False)
    try:
        out = R.check_step(sem.pipe, h, c)
    except Exception as e:  # noqa: BLE001
        viol.append(f"checker raised {e!r}")
        return False, viol + c.violations
    viol += c.violations
    claimed = R.pack_state(out)
    if sem.field is not None:
        claimed %= sem.field
    if claimed != y:
        viol.append(f"claimed output {claimed:#x} != {y:#x}")
    return not viol, viol


def relation_target(pipe: Pipeline, align: str = R.ALIGN_DIV, field: int | None = None,
                    dropped: tuple[str, ...] = ()) -> Target:
    sem = RelationSemantics(pipe, align, field, dropped)
    return Target(
        name=sem.label,
        check=lambda x, y, h: relation_check(sem, x, y, h)[0],
        gen=lambda x: R.step_witness(pipe, R.decode_state(x[0], pipe), list(x[1]), list(x[2]), align),
        reference=lambda x: tc_dot_total(pipe, x[0], list(x[1]), list(x[2])),
        columns=lambda h: relation_columns(pipe, h),
        output_of=lambda h: R.pack_state(h["groups"][-1]["out"]),
        output_neighbours=fp32_neighbours,
        fmt_x=fmt_x, fmt_y=lambda y: f"{y:#010x}", fmt_h=lambda h: "<step witness>",
    )


def silicon_target(pipe: Pipeline) -> Target:
    """``tc_dot`` as a checker with an empty witness: ``check(x, y, ()) := tc_dot(x) == y`` (trivially sound)."""
    def check(x, y, h):
        acc, a, b = x
        return h == () and tc_dot(pipe, acc, list(a), list(b)) == y
    return Target(name=f"silicon.tc_dot[{pipe.name}]", check=check, gen=lambda x: (),
                  reference=lambda x: tc_dot(pipe, x[0], list(x[1]), list(x[2])), output_neighbours=fp32_neighbours,
                  fmt_x=fmt_x, fmt_y=lambda y: f"{y:#010x}")


def fmt_x(x) -> dict:
    acc, a, b = x
    return {"acc": f"{acc:#010x}", "a": "".join(f"{w:04x}" for w in a), "b": "".join(f"{w:04x}" for w in b)}


def fp32_neighbours(x, y: int) -> Iterable[int]:
    """Nearby FP32 words: +-1 ulp, sign flip, exponent +-1, the zeros, the infinities, the canonical NaN."""
    out = {(y + 1) & 0xFFFFFFFF, (y - 1) & 0xFFFFFFFF, y ^ 0x80000000, (y + (1 << 23)) & 0xFFFFFFFF,
           (y - (1 << 23)) & 0xFFFFFFFF, 0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FFFFFFF, y ^ 1, y ^ (1 << 22)}
    out.discard(y)
    return sorted(out)


# ---------------------------------------------------------------------------------------------------------------------
# The witness shape: every column of a step witness with the range the relation declares for it
# ---------------------------------------------------------------------------------------------------------------------

def relation_columns(pipe: Pipeline, h: dict) -> list[Column]:
    p = R.params(pipe)
    A, B, W, K = p.A, p.B, p.W, p.K
    E_LO, E_HI = -252, 255              # every masked exponent of a product or state lies here
    cols: list[Column] = []

    def col(path, lo, hi, kind=OTHER):
        cols.append(Column(tuple(path), lo, hi, kind))

    for j, q in enumerate(h["prod"]):
        base = ("prod", j)
        for k_, (lo, hi, kind) in {
            "sa": (0, 2, SIGN), "sb": (0, 2, SIGN), "sp": (0, 2, SIGN),
            "ea": (0, 256, EXPONENT), "eb": (0, 256, EXPONENT),
            "SA": (0, 256, MAGNITUDE), "SB": (0, 256, MAGNITUDE), "P": (0, 1 << 16, MAGNITUDE),
            "ca": (0, 4, SELECTOR), "cb": (0, 4, SELECTOR),
            "nan": (0, 2, BOOL), "pinf": (0, 2, BOOL), "ninf": (0, 2, BOOL), "zero": (0, 2, BOOL),
            "ep": (E_LO, E_HI, EXPONENT), "em": (E_LO, E_HI, EXPONENT),
        }.items():
            if k_ in q:
                col(base + (k_,), lo, hi, kind)
    for gi, g in enumerate(h["groups"]):
        base = ("groups", gi)
        size = pipe.groups[gi]
        col(base + ("Estar",), E_LO, E_HI, EXPONENT)
        for i, t in enumerate(g["terms"]):
            tb = base + ("terms", i)
            col(tb + ("d",), 0, 512, SHIFT)
            col(tb + ("pow",), 1, (1 << A) + 1, POWER)
            col(tb + ("invpow",), 1, (1 << A) + 1, POWER)
            col(tb + ("aligned",), 0, 1 << A, MAGNITUDE)
            col(tb + ("r",), 0, 1 << A, REMAINDER)
            col(tb + ("rinv",), 0, 1 << A, REMAINDER)
            col(tb + ("signed",), 0, 1 << A, MAGNITUDE)
        n_terms = size + 1
        col(base + ("total",), -(n_terms << A), n_terms << A, OTHER)
        col(base + ("neg_t",), 0, 2, BOOL)
        col(base + ("mag_t",), 0, 1 << B, MAGNITUDE)
        col(base + ("L",), 0, B + 1, SHIFT)
        col(base + ("normpow",), 1, (1 << B) + 1, POWER)
        col(base + ("z_t",), 0, 2, BOOL)
        col(base + ("normB",), 0, 1 << B, MAGNITUDE)
        col(base + ("magW",), 0, 1 << W, MAGNITUDE)
        col(base + ("r_n",), 0, 1 << (B - W), REMAINDER)
        col(base + ("e_out",), -R.E_OFF, 512 - R.E_OFF, EXPONENT)
        col(base + ("e_fin",), R.FP32_MIN_E, 512 - R.E_OFF, EXPONENT)
        col(base + ("pow_s",), 1, (1 << K) + 1, POWER)
        col(base + ("invpow_s",), 1, (1 << K) + 1, POWER)
        col(base + ("ovf",), 0, 2, BOOL)
        col(base + ("mag24",), 0, 1 << 24, MAGNITUDE)
        col(base + ("r_s",), 0, 1 << K, REMAINDER)
        col(base + ("rsinv",), 0, 1 << K, REMAINDER)
        col(base + ("z_out",), 0, 2, BOOL)
        for k_ in ("pinf0", "ninf0", "any_nan", "any_pinf", "any_ninf", "ovf_eff", "any_inf", "nan_out", "inf_out",
                   "neg_inf", "fin", "neg_fin"):
            col(base + (k_,), 0, 2, BOOL)
        for k_ in ("S_nan", "S_pinf", "S_ninf"):
            col(base + (k_,), 0, size + 2, SELECTOR)
        col(base + ("e_state",), E_LO, E_HI, EXPONENT)
        for k_, (lo, hi, kind) in {"nan": (0, 2, BOOL), "inf": (0, 2, BOOL), "neg": (0, 2, SIGN),
                                   "e": (E_LO, E_HI, EXPONENT), "M": (0, 1 << 24, MAGNITUDE)}.items():
            col(base + ("out", k_), lo, hi, kind)
    return cols
