"""Alternative-field cost models for the frozen target's checker: Goldilocks (baseline, reproduced), Mersenne31 and
BabyBear with degree-4/5 extensions, and a binary tower (Binius-style, GF(2) bit columns under GF(2^128) challenges).

Nothing here is a decision.  Every number is an ESTIMATE from an explicit cost model over the WP2 checker census
(``verity_numerical.security.census``: the gadgets are executed once on symbolic values, so no count is copied), and
the soundness/latency/communication columns come from ``verity_numerical.security.accounting`` with the per-unit
figures swapped for the field's.

Method.

* **Small primes** (31-bit).  :class:`SmallFieldCtx` re-runs the checker with the red team's wrap rule: an
  ``lt_pow(x, ipow, B)`` whose product ``x ipow`` can reach ``2^(2B) >= p`` (checker docstring F1) is realised
  *directly* as the range check ``bound - 1 - x in [0, 2^B)`` on the table's bound column (``x`` is already
  range-checked to ``[0, 2^B)`` by its gadget, so the difference of a violating ``x`` has representative
  ``>= p - 2^B + 1``, which is ``>= 2^B`` iff ``p >= 2^(B+1) - 1``: exactly true for Mersenne31 at ``B = 30``,
  false for BabyBear, which therefore tightens the signed sum to its premise bound: ``BABYBEAR_EXTRA``).  The
  direct realisation drops the multiplication and keeps one B-bit range check.  The three relation.py wrap
  products (``aligned*pow`` 2^52, ``mag24*pow_s`` 2^59, ``mag_t*normpow`` 2^60) are, in the checker, the tightly
  bounded ``q*pow`` / ``q*sr`` (below 2^26 / 2^17 once the quotient bound holds) and ``div*M + r = v*mul``
  (F3, reaching 2^31 - 1), so only the last must be limb-split; the 32-bit words (F4) become two 16-bit limbs.
  Those two repairs are not re-derivable from the gadget code without rewriting it, so they are explicit
  per-group / per-unit deltas (:data:`F3_SPLIT_PER_GROUP`, :data:`F4_LIMBS_PER_UNIT`) with the derivation in
  their docstrings.
* **Binary tower.**  :class:`BinaryCtx` re-runs the checker with GF(2) semantics: every w-bit value is w
  committed bit columns (so a power-of-two range check on a column is free and booleanity is free); every
  integer identity ``eq`` over k non-constant terms is ``k - 1`` ripple-carry adders (Binius Protocol 5.3: one
  committed carry bit and one degree-2 constraint ``x y + x cin + y cin = cout`` per bit position, counted as 3
  GF(2) multiplications); a product with a 1-bit factor is w ANDs; a product by a looked-up power of two is a
  barrel shifter (``ceil(log2 w_shift)`` mux stages of w ANDs, each stage committed); a product of two multi-bit
  values of total width <= 16 is one lookup into a 2^16 byte-product table (Binius Example 5.7); wider ones would
  be schoolbook (Protocol 5.5) -- the checker has none.  ``lt_pow`` is one comparison adder.  The carry/adder
  cost is the crux of the binary option and is reported on its own.

Soundness.  ``accounting`` is reused unchanged for the prime fields (``first_configuration`` over the small
field's extension degrees).  For the binary tower ``accounting.logup_identity_error`` refuses (logUp needs
char(F) > N, Haboeck Lemma 5: multiplicities vanish in characteristic 2), so the lookup is Lasso's multiset
check (Binius 4.4) whose error is the Schwartz-Zippel bound ``(N + |T|)/|F|`` over GF(2^128) -- and that is why
2^-128 is out of reach there without a 2^256 tower level or repetition.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache

from verity_numerical.checker import REAL, Params
from verity_numerical.checker.ctx import Table
from verity_numerical.checker.step import check_step_expr, gen_step
from verity_numerical.security import accounting as acc
from verity_numerical.security import census
from verity_numerical.security.census import RANGE_CHUNK, CensusCtx, Sym, UnitCensus

__all__ = [
    "BABYBEAR_EXTRA",
    "F3_SPLIT_PER_GROUP",
    "F4_LIMBS_PER_UNIT",
    "OPTIONS",
    "BinaryCensus",
    "BinaryCtx",
    "FieldOption",
    "SmallFieldCensus",
    "SmallFieldCtx",
    "WSym",
    "binary_census",
    "binary_row",
    "binary_soundness",
    "compare",
    "first_small_field_configuration",
    "main",
    "prime_row",
    "report",
    "small_field_census",
]

# ---------------------------------------------------------------------------------------------------------------------
# Width-tracking symbolic values
# ---------------------------------------------------------------------------------------------------------------------


class WSym(Sym):
    """A :class:`census.Sym` that also carries an upper bound ``vmax`` on the magnitude of the value (``bits`` is
    its bit length), the number of non-constant summands (``terms``) and whether the value is a looked-up power of
    two (``pow2``).  Sums and differences bound the magnitude by the sum of the bounds (signed values are read by
    magnitude), constant coefficients multiply it, products multiply the bounds."""

    __slots__ = ("bitlike", "pow2", "terms", "vmax")

    def __init__(self, ctx, const=None, atom=None, depth=0, vmax=0, terms=0, pow2=False, bitlike=False):
        super().__init__(ctx, const=const, atom=atom, depth=depth)
        self.vmax = abs(const) if const is not None else vmax
        self.terms = 0 if const is not None else terms
        self.pow2 = pow2
        self.bitlike = bitlike          # a single bit, possibly scaled by a constant (a bit in GF(2) terms)

    @property
    def bits(self) -> int:
        return max(1, int(self.vmax).bit_length())

    def _lift(self, other):
        if isinstance(other, WSym):
            return other
        if isinstance(other, int):
            return WSym(self.ctx, const=other)
        return NotImplemented

    def _linear(self, other):
        if self.const is not None and other.const is not None:
            return WSym(self.ctx, const=0)
        c, x = (self, other) if self.const is not None else (other, self)
        bitlike = c.const is not None and x.bitlike and c.vmax <= 1 and x.vmax <= 1   # 1 - b, b - 1: a bit
        return WSym(self.ctx, depth=max(self.depth, other.depth), vmax=self.vmax + other.vmax,
                    terms=self.terms + other.terms, bitlike=bitlike)

    def __neg__(self):
        if self.const is not None:
            return WSym(self.ctx, const=0)
        return WSym(self.ctx, depth=self.depth, vmax=self.vmax, terms=self.terms, atom=None, bitlike=self.bitlike)

    def __mul__(self, other):
        o = self._lift(other)
        if o is NotImplemented:
            return o
        if self.const is not None and o.const is not None:
            return WSym(self.ctx, const=0)
        if self.const is not None or o.const is not None:
            nz, c = (self, o) if self.const is None else (o, self)
            return WSym(self.ctx, depth=nz.depth, vmax=nz.vmax * c.vmax, terms=nz.terms, bitlike=nz.bitlike)
        return self.ctx.product(self, o)

    __rmul__ = __mul__


def _vmax_of_bits(bits: int) -> int:
    return 2**bits - 1


@cache
def _table_maxes(table: Table) -> tuple[tuple[int, bool], ...]:
    """Per output column: (max value over the rows, every row value is 0 or a power of two)."""
    cols = list(zip(*(table.fn(k) for k in range(table.lo, table.hi))))
    return tuple((max(col), all(v == 0 or v & (v - 1) == 0 for v in col)) for col in cols)


def _table_widths(table: Table) -> tuple[tuple[int, bool], ...]:
    return tuple((m.bit_length(), p2) for m, p2 in _table_maxes(table))


def _is_pow2_range(lo: int, hi: int) -> bool:
    return lo == 0 and hi > 0 and hi & (hi - 1) == 0


class _WidthCtx(CensusCtx):
    """CensusCtx producing :class:`WSym` values."""

    def const(self, v: int) -> WSym:
        return WSym(self, const=v)

    def input(self, bits: int) -> WSym:
        self.c.columns_inputs += 1
        return WSym(self, atom=self.column(), vmax=_vmax_of_bits(bits), terms=1, bitlike=bits == 1)

    def hint(self, bits: int) -> WSym:
        self.c.columns_hints += 1
        self.c.hint_bits[bits] += 1
        return WSym(self, atom=self.column(), vmax=_vmax_of_bits(bits), terms=1, bitlike=bits == 1)

    def product(self, a, b) -> WSym:
        s = super().product(a, b)
        return WSym(self, atom=s.atom, depth=s.depth, vmax=a.vmax * b.vmax, terms=1, bitlike=a.bitlike and b.bitlike)

    def lookup(self, name: str, table: Table, key) -> tuple:
        if isinstance(key, int) or key.const is not None:
            return tuple(self.const(v) for v in table.row(key if isinstance(key, int) else 0))
        outs = super().lookup(name, table, key)
        return tuple(WSym(self, atom=o.atom, depth=o.depth, vmax=m, terms=1, pow2=p2)
                     for o, (m, p2) in zip(outs, _table_maxes(table)))


def _run_unit(ctx: CensusCtx, p: Params = REAL) -> UnitCensus:
    """``census.census_unit`` with a caller-supplied context (same construction, the census's own helpers); the
    active ``census.CHECKER_VARIANT`` (v2: the lookup-centred unit; its small-field deltas below are v1's and are
    an approximation there -- v2 has no ``lt_pow`` and one wide product, ``v_hi A``, to split in a 31-bit field)."""
    c0, a0, b0 = census._shape_inputs(p)
    if census.CHECKER_VARIANT == "v2":
        from verity_numerical.checker.v2 import step as step2
        shape = step2.gen_step(p, c0, a0, b0)
        h = census._sym_step_hints_v2(ctx, p, shape)
        run, gadgets = step2.check_step_expr, census.V2_GADGETS
    else:
        shape = gen_step(p, c0, a0, b0)
        h = census._sym_step_hints(ctx, p, shape)
        run, gadgets = check_step_expr, census.V1_GADGETS
    c = ctx.input(p.acc_word_bits)
    a = [ctx.input(p.word_bits) for _ in a0]
    b = [ctx.input(p.word_bits) for _ in b0]
    y = ctx.input(p.acc_word_bits)
    with census._sectioned(gadgets):
        got = run(ctx, p, c, a, b, h)
    ctx.eq("y", got, y)
    return ctx.c


# ---------------------------------------------------------------------------------------------------------------------
# Small prime fields
# ---------------------------------------------------------------------------------------------------------------------


class SmallFieldCtx(_WidthCtx):
    """The census under the wrap rule of a ``p_bits``-bit prime: ``lt_pow`` whose product can reach ``2^(2B) >= p``
    is realised directly (no multiplication, one B-bit range check on ``bound - 1 - x``)."""

    def __init__(self, variant: str, p: int):
        super().__init__(variant)
        self.p = p
        self.lt_pow_by_width: Counter = Counter()
        self.lt_pow_direct: Counter = Counter()

    def lt_pow(self, name: str, x, ipow, bits: int) -> None:
        self.lt_pow_by_width[bits] += 1
        if 2 ** (2 * bits) >= self.p:
            self.lt_pow_direct[bits] += 1
            self.rng(name, ipow - 1 - x, 0, 2**bits)          # direct: a materialised linear expression
        else:
            super().lt_pow(name, x, ipow, bits)


@dataclass(frozen=True)
class Delta:
    """Extra predicates a field repair costs, per occurrence: hint columns (by width), range checks (by width),
    linear identities, multiplications, input columns."""

    hints: dict[int, int] = field(default_factory=dict)
    ranges: dict[int, int] = field(default_factory=dict)
    lin: int = 0
    mul: int = 0
    inputs: int = 0
    note: str = ""
    fixed: dict[str, tuple[int, int, int]] = field(default_factory=dict)   # table -> (queries, rows, columns)

    def scaled(self, k: int) -> Delta:
        return Delta({w: n * k for w, n in self.hints.items()}, {w: n * k for w, n in self.ranges.items()},
                     self.lin * k, self.mul * k, self.inputs * k, self.note,
                     {t: (q * k, r, c) for t, (q, r, c) in self.fixed.items()})

    @property
    def fixed_lookups(self) -> int:
        return sum(q for q, _, _ in self.fixed.values())

    @property
    def range_lookups(self) -> int:
        return sum(-(-w // RANGE_CHUNK) * n for w, n in self.ranges.items())

    @property
    def range_splits(self) -> int:
        return sum(n for w, n in self.ranges.items() if w > RANGE_CHUNK)


def _f3_split(p: Params) -> Delta:
    """F3: ``div M' + r = v mul`` reaches ``(2^24 - 1) 2^7 + 2^7 - 1 = 2^31 - 1``.  Split M' = M1 2^12 + M0 and
    v = v1 2^15 + v0 (4 hints, 4 ranges, 2 recompositions), state the identity on the low and high halves with one
    small signed carry hint (1 hint, 1 range, 2 identities); ``div M0``, ``div M1``, ``mul v0``, ``mul v1`` are four
    products where the original had two (+2 mul).  Per group."""
    half_m, half_v = p.acc_sig // 2, p.s_bits // 2
    cols = Counter([half_m, p.acc_sig - half_m, half_v, p.s_bits - half_v, 8])      # M0, M1, v0, v1, carry
    return Delta(hints=dict(cols), ranges=dict(cols),
                 lin=4, mul=2, note="F3 normalise division split into two halves with a carry")


def _f4_limbs(p: Params) -> Delta:
    """F4: a 32-bit word is two 16-bit limbs.  ``c`` and ``y`` gain one input column each; the 32-bit range check
    (2 chunks + 1 split) becomes two 16-bit checks (2 chunks, no split: ``ranges`` books -1 at width 32, +2 at 16);
    the decode identity splits ``f`` into ``f_lo`` (16) and ``f_hi`` (7): +1 hint, its 23-bit range becomes 16 + 7,
    +1 identity; the pack expression is written per limb: ``low`` split likewise (+1 hint, +1 identity), the output
    identity doubles (+1 lin), the three word-selecting products are formed per limb (+3 mul).  Per unit."""
    L = RANGE_CHUNK
    hi = p.acc_frac - L
    return Delta(hints={hi: 2}, ranges={p.acc_word_bits: -1, L: 2 + 2, p.acc_frac: -2, hi: 2},
                 lin=3, mul=3, inputs=2, note="F4 32-bit words as two 16-bit limbs")


def _signed_sum_tighten(p: Params) -> Delta:
    """The signed-sum range ``v in [0, 2^30)`` is tightened to the premise bound ``(2^25 - 1) + 8 (2^26 - 1)`` when
    ``p < 2^31 - 1`` (BabyBear: ``v = S + p`` for a negative ``S`` can land in ``[0, 2^30)``): one more comparison
    range check of ``s_bits`` bits per group (``bound - 1 - v in [0, 2^30)``)."""
    return Delta(ranges={p.s_bits: 1}, note="signed sum tightened to its premise bound")


def _v2_norm_lookup(p: Params) -> Delta:
    """checker v2 in a field below ``2^(vh + vb + k_hi)`` (every 31-bit prime): the normalisation product
    ``v_hi A(k)`` (A up to 2^39 at REAL) does not fit; it becomes one more TNORM-style lookup on
    ``v_hi + 2^vh koff`` (2^vh k_rows rows: 1.3M at REAL) and the product disappears.  Per group."""
    from verity_numerical.checker.v2.params import V2
    rows = 2 ** V2.vh(p) * V2.k_rows(p)
    return Delta(mul=-1, note="v2 normalise: v_hi A(k) as a TNORM_HI lookup instead of a product",
                 fixed={"TNORM_HI": (1, rows, 2)})


F3_SPLIT_PER_GROUP: Delta = _f3_split(REAL)
F4_LIMBS_PER_UNIT: Delta = _f4_limbs(REAL)
BABYBEAR_EXTRA: Delta = _signed_sum_tighten(REAL)


@dataclass
class SmallFieldCensus:
    """The unit census under a small prime: the re-run counts plus the explicit repairs."""

    field: acc.Field
    base: UnitCensus
    lt_pow_by_width: dict[int, int]
    lt_pow_direct: dict[int, int]
    deltas: list[Delta]

    @property
    def mul(self) -> int:
        return self.base.mul + sum(d.mul for d in self.deltas)

    @property
    def lin_total(self) -> int:
        return self.base.lin_total + sum(d.lin + d.range_splits for d in self.deltas)

    @property
    def range_by_width(self) -> Counter:
        out = Counter(self.base.range_by_width)
        for d in self.deltas:
            out.update(d.ranges)
        return Counter({w: n for w, n in out.items() if n})

    @property
    def range_lookups(self) -> int:
        return sum(-(-w // RANGE_CHUNK) * n for w, n in self.range_by_width.items())

    @property
    def lookup(self) -> int:
        return self.range_lookups + self.base.fixed_lookups + sum(d.fixed_lookups for d in self.deltas)

    @property
    def extra_hints(self) -> int:
        return sum(sum(d.hints.values()) for d in self.deltas)

    @property
    def extra_inputs(self) -> int:
        return sum(d.inputs for d in self.deltas)

    @property
    def columns_committed_a(self) -> int:
        return self.base.columns_committed_a + self.extra_hints + self.extra_inputs

    @property
    def queries_on_computed_wires(self) -> int:
        # the direct lt_pow realisations are queries on linear expressions, as the product ones were on products
        return self.base.queries_on_computed_wires + sum(d.range_lookups and 1 for d in self.deltas)

    @property
    def columns_committed_b(self) -> int:
        """Same layout ESTIMATE as ``UnitCensus.columns_committed_b`` on the adjusted counts."""
        extra_cols = self.extra_hints + self.extra_inputs + sum(sum(d.ranges.values()) for d in self.deltas)
        return 3 * self.mul + (self.base.columns_all - self.base.columns_in_mul) + extra_cols

    def tables(self) -> tuple[acc.LookupTable, ...]:
        out = Counter(self.base.lookups)
        for w, n in self.range_by_width.items():
            out[f"T_range{RANGE_CHUNK}"] += -(-w // RANGE_CHUNK) * n
        ft = dict(census.fixed_tables())
        for d in self.deltas:
            for name, (q, rows, cols) in d.fixed.items():
                out[name] += q
                ft.setdefault(name, (rows, cols))
        return tuple(acc.LookupTable(name, ft[name][0], q, ft[name][1]) for name, q in sorted(out.items()))

    def summary(self) -> dict:
        return {
            "field": self.field.name,
            "mul": self.mul, "lin": self.lin_total, "lookup": self.lookup,
            "range_by_width": dict(sorted(self.range_by_width.items())),
            "lt_pow_by_width": dict(sorted(self.lt_pow_by_width.items())),
            "lt_pow_direct": dict(sorted(self.lt_pow_direct.items())),
            "extra_hints": self.extra_hints, "extra_inputs": self.extra_inputs,
            "columns_committed_A": self.columns_committed_a, "columns_committed_B_estimate": self.columns_committed_b,
            "repairs": [d.note for d in self.deltas],
        }


def small_field_census(F: acc.Field, p: Params = REAL) -> SmallFieldCensus:
    """Unit census under the prime ``F`` with F1 (direct ``lt_pow``), F3 (split normalisation, when the field is
    below ``2^31 - 1``... i.e. always for 31-bit primes), F4 (word limbs) and, when ``p < 2^31 - 1``, the signed-sum
    tightening.  For Goldilocks nothing changes and the result equals ``census.census_unit()``."""
    ctx = SmallFieldCtx(f"unit@{F.name}", F.characteristic)
    base = _run_unit(ctx, p)
    deltas: list[Delta] = []
    groups = len(p.groups)
    if census.CHECKER_VARIANT == "v2":
        # v2: no lt_pow, no division identity (F3 is moot); the one wide product v_hi A(k) < 2^(vh + vb + k_hi)
        # becomes a lookup below that; F4 and the BabyBear sign tightening apply as for v1 (F4's +3 mul per limb is
        # v1's pack; v2's pack has one word-selecting product: an overcount of 2 mul, kept)
        from verity_numerical.checker.v2.params import V2
        if F.characteristic < 2 ** (V2.vh(p) + V2.vb(p) + p.k_hi):
            deltas.append(_v2_norm_lookup(p).scaled(groups))
    elif F.characteristic <= 2**31 - 1:                        # F3: the division identity reaches 2^31 - 1
        deltas.append(_f3_split(p).scaled(groups))
    if F.characteristic < 2**p.acc_word_bits:                  # F4: words do not fit
        deltas.append(_f4_limbs(p))
    if F.characteristic < 2**31 - 1:                           # BabyBear: the two-sided 30-bit checks need p >= 2^31 - 1
        deltas.append(_signed_sum_tighten(p).scaled(groups))
    return SmallFieldCensus(F, base, dict(ctx.lt_pow_by_width), dict(ctx.lt_pow_direct), deltas)


# ---------------------------------------------------------------------------------------------------------------------
# Binary tower
# ---------------------------------------------------------------------------------------------------------------------

ADDER_MULS_PER_BIT = 3        # x*y + x*cin + y*cin  (Binius Protocol 5.3), one committed carry bit per position
MUL_TABLE_BITS = 16           # byte-by-byte product table (Binius Example 5.7)


@dataclass
class BinaryCensus:
    """GF(2) counts per unit: committed bits by origin, adders (with their carry bits and AND gates), ANDs from
    bit-times-value products and barrel shifts, lookups (fixed tables + the byte-product table), free range checks."""

    bits_inputs: int = 0
    bits_hints: int = 0
    bits_lookup_outputs: int = 0
    bits_products: int = 0          # committed outputs of shifts and table products
    bits_materialised: int = 0      # expressions that must exist as bit columns (lookup keys, range operands)
    bits_carries: int = 0
    adders: int = 0
    adder_bits: int = 0             # total bit positions across adders (= carries)
    ands_adders: int = 0
    ands_bit_products: int = 0
    ands_shifts: int = 0
    ands_schoolbook: int = 0
    schoolbook: int = 0             # small-by-wide products the tracker cannot see as bits (e.g. the XOR-formed sign)
    shift_stages: int = 0
    lookups_fixed: int = 0
    lookups_mul_table: int = 0
    range_free: int = 0
    range_compare: int = 0
    booleans_free: int = 0
    lt_pow_direct: int = 0
    adders_by_width: Counter = field(default_factory=Counter)

    @property
    def bits_committed(self) -> int:
        return (self.bits_inputs + self.bits_hints + self.bits_lookup_outputs + self.bits_products
                + self.bits_materialised + self.bits_carries)

    @property
    def ands(self) -> int:
        return self.ands_adders + self.ands_bit_products + self.ands_shifts + self.ands_schoolbook

    @property
    def lookups(self) -> int:
        return self.lookups_fixed + self.lookups_mul_table

    def summary(self) -> dict:
        return {
            "bits": {"inputs": self.bits_inputs, "hints": self.bits_hints, "lookup_outputs": self.bits_lookup_outputs,
                     "products": self.bits_products, "materialised": self.bits_materialised,
                     "carries": self.bits_carries, "committed": self.bits_committed},
            "adders": self.adders, "adder_bit_positions": self.adder_bits,
            "adders_by_width": dict(sorted(self.adders_by_width.items())),
            "ands": {"adders": self.ands_adders, "bit_products": self.ands_bit_products, "shifts": self.ands_shifts,
                     "schoolbook": self.ands_schoolbook, "total": self.ands},
            "schoolbook_products": self.schoolbook,
            "shift_stages": self.shift_stages,
            "lookups": {"fixed": self.lookups_fixed, "mul_table": self.lookups_mul_table, "total": self.lookups},
            "range_checks": {"free_on_bit_columns": self.range_free, "comparison_adders": self.range_compare},
            "booleans_free": self.booleans_free, "lt_pow_direct": self.lt_pow_direct,
        }


class BinaryCtx(_WidthCtx):
    """The checker in GF(2) semantics (module docstring)."""

    def __init__(self, variant: str):
        super().__init__(variant)
        self.b = BinaryCensus()

    # -- helpers -----------------------------------------------------------------------------------------------------
    def _adders(self, k: int, width: int) -> None:
        if k <= 0:
            return
        self.b.adders += k
        self.b.adders_by_width[width] += k
        self.b.adder_bits += k * width
        self.b.bits_carries += k * width
        self.b.ands_adders += ADDER_MULS_PER_BIT * k * width

    def _materialise_bits(self, x: WSym) -> None:
        """A non-column value that must exist as bits: commit its bits and bind them with the adders of its sum."""
        if x.atom is None and x.const is None:
            self.b.bits_materialised += x.bits
            self._adders(max(0, x.terms - 1), x.bits)
            x.atom, x.terms = self.column(), 1          # it is a column from here on (shared reference)

    # -- columns ------------------------------------------------------------------------------------------------------
    def input(self, bits: int) -> WSym:
        self.b.bits_inputs += bits
        return super().input(bits)

    def hint(self, bits: int) -> WSym:
        self.b.bits_hints += bits
        return super().hint(bits)

    def lookup(self, name: str, table: Table, key) -> tuple:
        if isinstance(key, int) or key.const is not None:
            return super().lookup(name, table, key)
        self._materialise_bits(key)
        self.b.lookups_fixed += 1
        self.b.bits_lookup_outputs += sum(w for w, _ in _table_widths(table))
        return super().lookup(name, table, key)

    def product(self, a: WSym, b: WSym) -> WSym:
        s = super().product(a, b)                      # keeps the prime-field census counts for reference
        if a.bitlike or b.bitlike:
            bit, other = (a, b) if a.bitlike else (b, a)
            self._materialise_bits(other)
            if not bit.bitlike or bit.atom is None:
                self._materialise_bits(bit)
            self.b.ands_bit_products += other.bits
            return WSym(self, atom=s.atom, depth=s.depth, vmax=a.vmax * b.vmax, terms=1, bitlike=other.bitlike)
        if a.pow2 or b.pow2:
            val, sh = (a, b) if b.pow2 else (b, a)
            self._materialise_bits(val)
            stages = max(1, math.ceil(math.log2(max(2, sh.bits))))
            width = val.bits + sh.bits
            self.b.shift_stages += stages
            self.b.ands_shifts += stages * width
            self.b.bits_products += stages * width
            return WSym(self, atom=s.atom, depth=s.depth, vmax=a.vmax * b.vmax, terms=1)
        if a.bits + b.bits <= MUL_TABLE_BITS:
            self._materialise_bits(a)
            self._materialise_bits(b)
            self.b.lookups_mul_table += 1
            self.b.bits_products += a.bits + b.bits
            return WSym(self, atom=s.atom, depth=s.depth, vmax=a.vmax * b.vmax, terms=1)
        # schoolbook by bits (Binius Protocol 5.5 at limb size 1): one AND row per bit of the narrower factor, rows added
        narrow, wide = (a, b) if a.bits <= b.bits else (b, a)
        self._materialise_bits(narrow)
        self._materialise_bits(wide)
        self.b.schoolbook += 1
        self.b.ands_schoolbook += narrow.bits * wide.bits
        self._adders(narrow.bits - 1, narrow.bits + wide.bits)
        self.b.bits_products += narrow.bits + wide.bits
        return WSym(self, atom=s.atom, depth=s.depth, vmax=a.vmax * b.vmax, terms=1)

    # -- predicates ---------------------------------------------------------------------------------------------------
    def rng(self, name: str, x, lo: int, hi: int) -> None:
        if isinstance(x, int) or x.const is not None:
            return
        super().rng(name, x, lo, hi)
        w = census._width(lo, hi)
        self._materialise_bits(x)
        if x.atom is not None and _is_pow2_range(lo, hi) and x.bits <= w:
            self.b.range_free += 1
        else:
            self.b.range_compare += 1
            self._adders(1, max(w, x.bits))

    def eq(self, name: str, a, b) -> None:
        super().eq(name, a, b)
        ta = 0 if isinstance(a, int) or a.const is not None else a.terms
        tb = 0 if isinstance(b, int) or b.const is not None else b.terms
        wa = 0 if isinstance(a, int) else a.bits
        wb = 0 if isinstance(b, int) else b.bits
        self._adders(max(0, ta + tb - 1), max(wa, wb))

    def boolean(self, name: str, b) -> None:
        super().boolean(name, b)
        if not isinstance(b, int) and b.const is None:
            self.b.booleans_free += 1

    def lt_pow(self, name: str, x, ipow, bits: int) -> None:
        self.b.lt_pow_direct += 1
        self._materialise_bits(x)
        self.b.range_compare += 1
        self._adders(1, bits)


def binary_census(p: Params = REAL) -> tuple[UnitCensus, BinaryCensus]:
    ctx = BinaryCtx("unit@GF(2)")
    base = _run_unit(ctx, p)
    return base, ctx.b


# ---------------------------------------------------------------------------------------------------------------------
# Field options and the comparison
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldOption:
    name: str
    trace: acc.Field
    element_bytes: float
    extension_degrees: tuple[int, ...]
    mult_vs_goldilocks: float        # ESTIMATE: one base-field multiplication relative to one Goldilocks multiplication
    tooling: str


GOLDILOCKS = FieldOption("goldilocks", acc.Field.prime("Goldilocks", acc.GOLDILOCKS), 8, (2, 3, 4), 1.0,
                         "plonky2 / plonky3 (MIT OR Apache-2.0); libzk/Longfellow (Apache-2.0)")
MERSENNE31 = FieldOption("m31", acc.Field.prime("Mersenne31", acc.MERSENNE31), 4, (4, 5, 6), 0.5,
                         "plonky3 p3-mersenne-31 0.5.x (MIT OR Apache-2.0; QM31 degree 4, no degree 5), stwo (Apache-2.0)")
BABYBEAR = FieldOption("babybear", acc.Field.prime("BabyBear", acc.BABYBEAR), 4, (4, 5, 6), 0.5,
                       "plonky3 p3-baby-bear (MIT OR Apache-2.0; BinomialExtensionField degrees 4, 5, 8)")
OPTIONS: tuple[FieldOption, ...] = (GOLDILOCKS, MERSENNE31, BABYBEAR)

N_CODE, K_CODE, L_PAYLOAD = 2**14, 2**12, 3840       # the protocol documents' Ligero parameters
HASH_QUERIES = 2**60
K_TARGET = 1536


def _candidates(sfc: SmallFieldCensus, K: int, B: int) -> tuple[Callable, Callable]:
    """Budget builders in ``first_configuration``'s calling convention, with the field's per-unit figures."""
    tables = sfc.tables()

    def mk_a(Ft, Fc, lig, hash_queries, **kw):
        return acc.account_candidate_a(acc.CandidateA(
            K=K, B=B, trace_field=Ft, challenge_field=Fc, ligero=lig, hash_queries=hash_queries, tables=tables,
            inputs_per_unit=sfc.columns_committed_a, query_wires_per_unit=sfc.queries_on_computed_wires, **kw))

    def mk_b(Ft, Fc, lig, hash_queries, **kw):
        return acc.account_candidate_b(acc.CandidateB(
            K=K, B=B, trace_field=Ft, challenge_field=Fc, ligero=lig, hash_queries=hash_queries, tables=tables,
            witness_per_unit=sfc.columns_committed_b, linear_per_unit=sfc.lin_total + 2 * sfc.mul,
            quadratic_per_unit=sfc.mul, **kw))

    return mk_a, mk_b


def first_small_field_configuration(target_log2: int, opt: FieldOption, candidate: str = "A", K: int = K_TARGET,
                                    B: int = 4096, extension_degrees: Sequence[int] | None = None,
                                    **kw) -> tuple[int, int, acc.Budget]:
    """``accounting.first_configuration`` with the field's census and extension degrees (``kw`` such as
    ``lookup_repetitions`` go to the candidate).  Raises ``AccountingError`` (with the flooring component) when no
    degree reaches the target."""
    sfc = small_field_census(opt.trace)
    mk_a, mk_b = _candidates(sfc, K, B)
    return acc.first_configuration(target_log2, mk_a if candidate == "A" else mk_b, opt.trace, N_CODE, K_CODE,
                                   L_PAYLOAD, extension_degrees=extension_degrees or opt.extension_degrees,
                                   hash_queries=HASH_QUERIES, **kw)


def _rescale_bytes(est: dict, ratio: float) -> dict:
    """``census_estimate_*`` count 8 bytes per trace element; a 4-byte field encodes half the bytes."""
    if ratio == 1:
        return dict(est)
    out = dict(est)
    enc = est["census.bytes_encoded"]
    out["census.bytes_encoded"] = enc * ratio
    out["census.hash_bytes"] = est["census.hash_bytes"] - enc + enc * ratio
    out["census.bytes_moved"] = est["census.bytes_moved"] * ratio
    out["estimate.peak_bytes"] = est["estimate.peak_bytes"] * ratio
    out["estimate.transcript_p2v_bytes"] = est["estimate.transcript_p2v_bytes"] * ratio
    out["transcript.bytes"] = out["estimate.transcript_p2v_bytes"] + est["estimate.transcript_v2p_bytes"]
    return out


def prime_row(opt: FieldOption, target_log2: int, candidate: str = "A", K: int = K_TARGET, B: int = 4096,
              extension_degrees: Sequence[int] | None = None, **kw) -> dict:
    """The contract's ``census.*`` columns (and the rest of ``census_estimate_*``) for one prime-field option at one
    operating point, plus the unit census and the first configuration.  ``soundness.reachable`` is False with the
    flooring reason when no extension degree in the option reaches the target."""
    sfc = small_field_census(opt.trace)
    row: dict = {"field": opt.name, "candidate": candidate, "B": B, "K": K, "target_log2": target_log2,
                 "unit": sfc.summary(), "tooling": opt.tooling}
    try:
        d, t, bud = first_small_field_configuration(target_log2, opt, candidate, K, B, extension_degrees, **kw)
    except acc.AccountingError as e:
        row["soundness"] = {"reachable": False, "reason": str(e)}
        return row
    Fc = opt.trace.extension(d)
    lig = acc.LigeroParams(n=N_CODE, k=K_CODE, l=L_PAYLOAD, t=t)
    tables = sfc.tables()
    if candidate == "A":
        cand = acc.CandidateA(K=K, B=B, trace_field=opt.trace, challenge_field=Fc, ligero=lig, hash_queries=HASH_QUERIES,
                              tables=tables, inputs_per_unit=sfc.columns_committed_a,
                              query_wires_per_unit=sfc.queries_on_computed_wires, **kw)
        est = acc.census_estimate_a(cand)
    else:
        cand = acc.CandidateB(K=K, B=B, trace_field=opt.trace, challenge_field=Fc, ligero=lig, hash_queries=HASH_QUERIES,
                              tables=tables, witness_per_unit=sfc.columns_committed_b,
                              linear_per_unit=sfc.lin_total + 2 * sfc.mul, quadratic_per_unit=sfc.mul, **kw)
        est = acc.census_estimate_b(cand)
    est = _rescale_bytes(est, opt.element_bytes / 8)
    est["estimate.field_ops_goldilocks_equiv"] = est["census.field_ops"] * opt.mult_vs_goldilocks
    est["estimate.committed_bytes"] = est["census.committed_elements"] * opt.element_bytes
    row.update(est)
    row["soundness"] = {"reachable": True, "extension_degree": d, "challenge_field": Fc.name, "opened_columns": t,
                        "total_log2": acc.log2(bud.total()),
                        "by_component_log2": {c: acc.log2(v) for c, v in bud.by_component().items() if v}}
    return row


def binary_soundness(units: int, tables: Sequence[acc.LookupTable], top_bits: int = 128) -> dict:
    """Terms over GF(2^top_bits): Lasso multiset checks per table (Schwartz-Zippel, degree ``N + |T|``), the
    zerocheck over all constraints (degree-2 constraints on ``log2(rows)`` variables), no logUp (``accounting``
    refuses characteristic 2 by Haboeck Lemma 5).  ESTIMATE of the structure, exact for the stated terms."""
    F = acc.Field(f"GF(2^{top_bits})", 2**top_bits, 2)
    terms = {}
    for tbl in tables:
        N = tbl.queries_per_unit * units
        terms[f"{tbl.name}: multiset check, N={N}, |T|={tbl.rows}"] = acc.schwartz_zippel_error(N + tbl.rows, F)
    rows_log2 = max(1, math.ceil(math.log2(units)))
    terms["zerocheck: degree-2 sumcheck over log2(rows) + log2(constraints/unit) variables"] = \
        acc.sumcheck_error([2] * (rows_log2 + 12), F)
    terms["Merkle collisions q^2/2^h, q=2^60, h=256"] = acc.hash_collision_bound(256, HASH_QUERIES)
    total = sum(terms.values(), Fraction(0))
    logup_refused = None
    try:
        acc.logup_identity_error(2, 1, F)
    except acc.AccountingError as e:
        logup_refused = str(e)
    return {"field": F.name, "terms_log2": {k: acc.log2(v) for k, v in terms.items()}, "total_log2": acc.log2(total),
            "meets_-80": total <= Fraction(2) ** -80, "meets_-128": total <= Fraction(2) ** -128,
            "logup_refused": logup_refused}


def binary_row(target_log2: int, K: int = K_TARGET, B: int = 4096, top_bits: int = 128, rate: float = 4.0) -> dict:
    """The binary tower's row in the same columns: committed *bits* (GF(2) elements), bytes at Reed-Solomon rate
    ``1/rate`` (Binius uses 1/4), GF(2) ANDs as field operations, lookups, a Lasso-style grand-product depth, and
    the soundness terms.  ESTIMATES throughout; the carry/adder cost is separated."""
    base, bc = binary_census()
    units = B * (K // 16)
    ft = census.fixed_tables()
    tables = [acc.LookupTable(n, ft[n][0], q, ft[n][1]) for n, q in sorted(base.lookups.items())]
    tables.append(acc.LookupTable("MUL8x8", 2**MUL_TABLE_BITS, bc.lookups_mul_table, 3))
    committed_bits = units * bc.bits_committed + sum(t.rows * 16 for t in tables)
    bytes_committed = committed_bits / 8
    bytes_encoded = bytes_committed * rate
    n_dim = max(math.ceil(math.log2(t.queries_per_unit * units + t.rows)) for t in tables)
    slots, depth_lookup = acc._fractional_chain(n_dim, 1)          # a grand-product GKR chain of the same shape
    zerocheck_rounds = math.ceil(math.log2(units)) + 12
    depth = 1 + max(depth_lookup, zerocheck_rounds) + 2
    sound = binary_soundness(units, tables, top_bits)
    t_open = acc.ligero_queries_for(target_log2 - 1, N_CODE, K_CODE, L_PAYLOAD)
    rows = math.ceil(committed_bits / (L_PAYLOAD * top_bits)) + 3
    p2v = rows * t_open * (top_bits // 8) + t_open * (int(math.log2(N_CODE)) * acc.DIGEST_BYTES) + slots * (top_bits // 8) * 4
    v2p = (slots + zerocheck_rounds + 2) * (top_bits // 8)
    gf2_ops = units * bc.ands
    ext_mults = 12 * (units * (bc.bits_committed + bc.ands)) // top_bits      # sumcheck work on packed 128-bit words
    return {
        "field": f"binary tower GF(2) / GF(2^{top_bits})", "candidate": "Binius-style (flat constraints + Lasso)",
        "B": B, "K": K, "target_log2": target_log2, "unit": bc.summary(),
        "unit_prime_census_for_reference": {"mul": base.mul, "lin": base.lin_total, "lookup": base.lookup},
        "census.witness_elements": units * (bc.bits_committed + bc.ands),
        "census.committed_elements": committed_bits,
        "estimate.committed_bytes": bytes_committed,
        "census.bytes_encoded": bytes_encoded,
        "census.hash_bytes": bytes_encoded + 2 * N_CODE * acc.DIGEST_BYTES,
        "census.field_ops": gf2_ops,
        "estimate.gf2_128_mults": ext_mults,
        "estimate.field_ops_goldilocks_equiv": ext_mults * 4 + gf2_ops / 64,   # 128-bit tower mult ~ 4 Goldilocks; ANDs 64/word
        "census.memory_passes": 3 + 2 + 2 * len(tables),
        "census.bytes_moved": 3 * bytes_encoded + 2 * units * (bc.bits_committed + bc.ands) / 8 * 2,
        "rounds.protocol": slots + zerocheck_rounds + 3,
        "rounds.sequential_depth": depth,
        "transcript.bytes": p2v + v2p,
        "estimate.transcript_p2v_bytes": p2v,
        "estimate.transcript_v2p_bytes": v2p,
        "estimate.peak_bytes": bytes_encoded + bytes_committed + units * bc.ands / 8,
        "soundness": {"reachable": sound.get(f"meets_{target_log2}", None),
                      **sound},
        "carry_cost_per_unit": {"adders": bc.adders, "carry_bits": bc.bits_carries, "ands": bc.ands_adders,
                                "share_of_committed_bits": bc.bits_carries / bc.bits_committed,
                                "share_of_ands": bc.ands_adders / bc.ands},
        "tooling": ("binius crate 0.1.1 (crates.io, Apache-2.0, 2025-05-23; repository IrreducibleOSS/binius ARCHIVED "
                    "2025-09-09); successor binius64 (binius-zk/binius64, Apache-2.0 OR MIT, git-only crates "
                    "binius-prover/-verifier, 64-bit word constraint system, ZK)"),
        **acc.e2e_sweep(depth),
    }


def compare(B_values: Sequence[int] = (1, 64, 4096), targets: Sequence[int] = (-128, -80), K: int = K_TARGET) -> dict:
    """Every option x candidate x B x target: the rows of :func:`prime_row` and :func:`binary_row`."""
    out: dict = {"K": K, "rows": []}
    for target in targets:
        for B in B_values:
            for opt in OPTIONS:
                for cand in ("A", "B"):
                    out["rows"].append(prime_row(opt, target, cand, K, B))
            out["rows"].append(binary_row(target, K, B))
    return out


CITATIONS: dict[str, dict] = {
    "DP23": {"title": "Diamond, Posen: Succinct Arguments over Towers of Binary Fields (Binius)",
             "url": "https://eprint.iacr.org/2023/1784",
             "used_for": "Protocol 5.3 ripple-carry adder (x y + x cin + y cin = cout, shifted carry column); Protocol 5.5 "
                         "schoolbook multiplication with limb products by lookup, Example 5.7 (2^16 byte-product table); "
                         "4.4 binary-field Lasso lookup (multiset check, no logUp); 4.3 shift virtual polynomials (constant "
                         "shifts only); Tables 5/6: 8,192 Keccak-f proved in 65.3 s single-threaded / 5.90 s multithreaded "
                         "with Groestl (vs Plonky3 71.2 s / 13.8 s with Keccak-256), hardware as stated in the paper's 6; no "
                         "zero-knowledge variant is given in the paper"},
    "DP24": {"title": "Diamond, Posen: Polylogarithmic Proofs for Multilinears over Binary Towers (FRI-Binius)",
             "url": "https://eprint.iacr.org/2024/504",
             "used_for": "the PCS a binary option would use; the Irreducible release note reports commit+prove for 2^24 "
                         "coefficients: FRI-Binius 0.91 s (1-bit), 7.6 s (8-bit), 28.9 s (32-bit) vs Plonky3 FRI 9.55 s "
                         "(31-bit) -- https://www.irreducible.com/posts/better-faster-smaller-binius"},
    "BINIUS_CRATE": {"title": "binius crate 0.1.1 (crates.io) / IrreducibleOSS/binius",
                     "url": "https://crates.io/crates/binius",
                     "used_for": "tooling: Apache-2.0, published 2025-05-23, repository archived 2025-09-09 and succeeded by "
                                 "Binius64 (https://github.com/binius-zk/binius64, Apache-2.0 OR MIT, git-only crates "
                                 "binius-prover / binius-verifier, 64-bit-word constraint system with shifted value "
                                 "indices, zk_config modules); README of the archived crate discourages security-critical use"},
    "BINIUS64_BENCH": {"title": "binius.xyz benchmarks (Binius64 vs Plonky3, 1,365 Keccak-f)",
                       "url": "https://www.binius.xyz/benchmarks/",
                       "used_for": "Binius64 142.27 ms multithreaded / 499.26 ms single-threaded on AWS c7i.16xlarge (x86) vs "
                                   "Plonky3 265 ms / 3,920 ms; proof 304 KiB vs 1,861 KiB"},
    "BINARYSPARTAN": {"title": "BinarySpartan: Spartan over binary fields (2026)", "url": "https://eprint.iacr.org/2026/1656",
                      "used_for": "Table 1 hashes/s at 12 threads: Binius64 45,786 Keccak-f/s vs Plonky3 7,604; states that "
                                  "Binius64 is zero-knowledge while BinarySpartan and Flock are not"},
    "HAB22": {"title": "Haboeck: Multivariate lookups based on logarithmic derivatives", "url": "https://eprint.iacr.org/2022/1530",
              "used_for": "Theorem 4 / Lemma 5 (char(F) > N): the reason logUp is unavailable in characteristic 2 and the "
                          "identity error floor (N + |T| - 1)/(|F| - |T|) used for the 31-bit extensions"},
    "STW24": {"title": "Setty, Thaler, Wahby: Unlocking the lookup singularity with Lasso", "url": "https://eprint.iacr.org/2023/1216",
              "used_for": "the multiset-check lookup a binary option falls back to; its error is the Schwartz-Zippel bound"},
    "P3_M31": {"title": "Plonky3 p3-mersenne-31 (crates.io) and commit 05cc5ac (QM31 for Circle proofs, 2026-06-12)",
               "url": "https://github.com/Plonky3/Plonky3/commit/05cc5ac5948ae32f4cea4484c7e5f6de66593cde",
               "used_for": "M31 tooling: MIT OR Apache-2.0, 0.5.x releases in 2026; degree-4 (QM31) challenge field, cubic "
                           "before; no degree-5/6 M31 extension shipped"},
    "P3_BB": {"title": "Plonky3 baby-bear/src/baby_bear.rs", "url": "https://github.com/Plonky3/Plonky3/blob/main/baby-bear/src/baby_bear.rs",
              "used_for": "BinomialExtensionData<4>, <5>, <8> for BabyBear (degrees 4, 5, 8 shipped)"},
    "STWO": {"title": "StarkWare S-two book: Mersenne prime, CM31/QM31", "url": "https://docs.starknet.io/learn/S-two-book/how-it-works/mersenne-prime",
             "used_for": "stwo draws challenges from QM31 (2^124); Circle STARKs over M31"},
    "AHIV22": {"title": "Ames, Hazay, Ishai, Venkitasubramaniam: Ligero (journal version)", "url": "https://eprint.iacr.org/2022/1608",
               "used_for": "the encoding/opening terms reused through accounting for every prime option"},
}

ASSUMPTIONS: tuple[str, ...] = (
    ("Every figure is an ESTIMATE from the cost model in this module's docstring over the WP2 checker census; nothing is "
    "measured.  Widths in the binary model are upper bounds from value-magnitude tracking (a sum's bound is the sum of "
    "bounds), so adder widths exceed the true ones by 1-3 bits in places (the signed sum is tracked at 33-34 bits, true 30)."),
    ("Small primes: F1 direct lt_pow as bound - 1 - x in [0, 2^B) is sound iff p >= 2^(B+1) - 1 given x in [0, 2^B); true for "
    "Mersenne31 at B = 30 exactly, false for BabyBear, which pays the signed-sum tightening (BABYBEAR_EXTRA).  Not verified "
    "with z3 in this lane; the checker's search.modular_queries is the place to do it."),
    ("Small primes: F3 (split normalisation) and F4 (word limbs) are hand-derived deltas (F3_SPLIT_PER_GROUP, F4_LIMBS_PER_UNIT), "
    "not re-derived by executing rewritten gadgets; the gadgets were not edited."),
    ("Small primes: accounting's census_estimate_* is reused with the field's per-unit figures; bytes are rescaled to 4-byte "
    "elements, field_ops stay in base-field multiplication equivalents (EXT_MULT_COST: degree 4 -> 9, 5 -> 13, 6 -> 17) and "
    "a Goldilocks-equivalent column assumes one 31-bit multiplication costs half a Goldilocks one (ESTIMATE); the layered "
    "shape and the internal-wire count of Candidate A are Goldilocks' (the small-field checker has 31 fewer products)."),
    ("Binary tower: committed GF(2) elements are bits; bytes at Reed-Solomon rate 1/4 (Binius default), packed 128 bits per "
    "symbol; barrel shifts commit every mux stage; products of a bit and a value are w ANDs and stay virtual; 19 small-by-"
    "wide products the tracker cannot recognise as bits (the XOR-formed product sign) are costed as bit-schoolbook (an "
    "overcount of about 2,000 ANDs per unit)."),
    ("Binary tower soundness: Lasso multiset check at (N + |T|)/2^128 per table plus a degree-2 zerocheck; the tower's top "
    "field in the crates is GF(2^128), so 2^-128 needs GF(2^256) (defined by the tower, not implemented) or repetition."),
    ("Latency and communication for the binary tower are modelled as a grand-product GKR chain of the LogUp-GKR shape plus a "
    "zerocheck; Ligero-style column openings at the same t as the Goldilocks operating point."),
)

KILL_CRITERIA: dict[str, str] = {
    "goldilocks": "nothing found here; the cubic extension reaches 2^-128 at every B, and it has the fewest field operations.",
    "m31 / babybear": "2^-128 at B = 4096 needs the degree-6 extension (or degree 5 with two lookup repetitions for Candidate "
                      "A only: Candidate B's powers-combination term (n_lin - 1)/|F| floors at 2^-126.6 in 2^155); neither "
                      "Plonky3 nor stwo ships a degree-5/6 M31 extension; the F1/F3/F4 repairs are unverified by z3; "
                      "field_ops in Goldilocks equivalents are 1.4-2.7x the baseline at 2^-128 -- the halved bytes do not pay.",
    "binary tower": "the checker is integer arithmetic: 712 adders per unit put 11,534 carry bits (50% of all committed bits) "
                    "and 34,602 of 43,592 ANDs on the table, so committed bits (23,243/unit) exceed Goldilocks A's 321 x 64 = "
                    "20,544; logUp is unavailable (char 2); 2^-128 unreachable at GF(2^128); the original crate is archived "
                    "and Binius64's 64-bit-word constraint system is a different arithmetisation than the one costed here.",
}


def report(K: int = K_TARGET) -> dict:
    """Everything ``backends/numerical/reports/fields.json`` carries: tables, per-unit censuses, citations, assumptions."""
    rows = compare(K=K)["rows"]
    unit = {opt.name: small_field_census(opt.trace).summary() for opt in OPTIONS}
    _base, bc = binary_census()
    unit["binary_tower"] = bc.summary()
    extra = {}
    for opt in (MERSENNE31, BABYBEAR):
        for cand in ("A", "B"):
            r = prime_row(opt, -128, cand, K, 4096, extension_degrees=(5,), lookup_repetitions=2)
            extra[f"{opt.name}/{cand}: degree 5 with 2 lookup repetitions, B=4096"] = r["soundness"]
    return {
        "meta": {"lane": "explore-fields", "K": K, "goldilocks_baseline_reproduced": {
            "A_committed_elements_B4096_2^-128": next(r for r in rows if r["field"] == "goldilocks" and r["candidate"] == "A"
                                                      and r["B"] == 4096 and r["target_log2"] == -128)["census.committed_elements"],
            "B_committed_elements_B4096_2^-128": next(r for r in rows if r["field"] == "goldilocks" and r["candidate"] == "B"
                                                      and r["B"] == 4096 and r["target_log2"] == -128)["census.committed_elements"]}},
        "unit_census": unit,
        "small_field_repairs": {"F3_per_group": F3_SPLIT_PER_GROUP.__dict__, "F4_per_unit": F4_LIMBS_PER_UNIT.__dict__,
                                "babybear_signed_sum_per_group": BABYBEAR_EXTRA.__dict__},
        "carry_cost_binary_per_unit": next(r for r in rows if r["field"].startswith("binary"))["carry_cost_per_unit"],
        "rows": rows,
        "degree5_two_repetitions": extra,
        "citations": CITATIONS,
        "assumptions": list(ASSUMPTIONS),
        "what_would_kill_each_option": KILL_CRITERIA,
    }


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.3g}" if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


COLUMNS = ("census.committed_elements", "estimate.committed_bytes", "census.bytes_encoded", "census.field_ops",
           "estimate.field_ops_goldilocks_equiv", "rounds.sequential_depth", "transcript.bytes", "estimate.peak_bytes")


def main(argv: Sequence[str] | None = None) -> None:
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    if args and args[0] == "--json":
        print(json.dumps(compare(), indent=1, default=str))
        return
    if args and args[0] == "--report":
        from pathlib import Path

        Path(args[1]).write_text(json.dumps(report(), indent=1, default=str) + "\n")
        return
    for target in (-128, -80):
        for B in (1, 64, 4096):
            print(f"\n== target 2^{target}, B = {B}, K = {K_TARGET} ==")
            head = f"{'field':10} {'cand':5} {'ext':>4} {'t':>4} " + " ".join(f"{c.split('.')[-1][:18]:>18}" for c in COLUMNS)
            print(head)
            for opt in OPTIONS:
                for cand in ("A", "B"):
                    r = prime_row(opt, target, cand, K_TARGET, B)
                    s = r["soundness"]
                    if not s["reachable"]:
                        print(f"{opt.name:10} {cand:5} {'-':>4} {'-':>4}  unreachable: {s['reason'][:100]}")
                        continue
                    print(f"{opt.name:10} {cand:5} {s['extension_degree']:>4} {s['opened_columns']:>4} "
                          + " ".join(f"{_fmt(r[c]):>18}" for c in COLUMNS))
            r = binary_row(target, K_TARGET, B)
            print(f"{'gf2 tower':10} {'Bin':5} {'128':>4} {'-':>4} " + " ".join(f"{_fmt(r[c]):>18}" for c in COLUMNS)
                  + f"   soundness 2^{r['soundness']['total_log2']:.1f} reachable={r['soundness']['reachable']}")
    _base, bc = binary_census()
    print("\nbinary tower per unit:", json.dumps(bc.summary()))
    for opt in OPTIONS:
        print(f"\n{opt.name} per unit:", json.dumps(small_field_census(opt.trace).summary()))


if __name__ == "__main__":       # pragma: no cover
    main()
