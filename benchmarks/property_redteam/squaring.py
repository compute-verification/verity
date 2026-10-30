"""Squaring mod p1024 in 6b's circuit model, measured in budgeted and nonlinear units: the attack on `SquaringFloor`.

**The target** (`Security/Definitions/Accounting/Assumptions/SquaringChain.lean`). The encode bound rests on
`SquaringFloor F e ε`, a squaring costs `e` nonlinear units (`Gates.nonlinear`: budgeted units plus data products, an
`mma` lane whose two multiplicands both depend on run or coin data, one unit whatever their range), and `ChainSum`, a
word's `k·S` chained squarings cost their sum, so `E = k·S·e`. `e = (2L − 1) + (L − d(ε))`, `L = n/8 = 128`,
`d = d₀ + ⌈log₂(1/ε)/8⌉`, `d₀ = 4`: `e = 374` at `ε = 2^−40` (`E = 1,145,562` a word), `e = 376` at `ε = 2^−20`
(`E = 1,151,688`). A squaring circuit under `e` nonlinear units on more than `ε` of its inputs falsifies
`SquaringFloor`. They replace `EncodeChainHardness` (`EncodeChain.lean`), the two at once in budgeted units, which
`bitwise_codec.py` falsified.

**What this module ports.** A scalar (Python-int, so p1024-sized) port of `Security/Definitions/Accounting/Gates/Model.lean`
(at 8225774747): `Op` (a gather with its table-read `low`),
`Gate`, `gatherAt`, `raw`, `out`, `Ok`/`InWindow`, `derive`/`intervals` (a gather's interval the hull of its column's and
0), `Proven`, `DeclaredHold`, `OperandBits`, `dynamic`, `indexBits`, `units` (fresh index bits per index gate and kind)
and `budgeted`, plus `Restricted` (`Codec.lean`). `units(..., fixed=True)` is that count; `fixed=False` keeps the earlier
count, `Nat.clog 2 (args.length − 1)` index bits a range gather.

**Two Lean-only holes, both closed.** (A) Under the earlier count a gather whose column is one constant word cost
`Nat.clog 2 1 = 0` units and returns `c·[idx ≤ 0]`, a free comparison of any s32 value, so carries are free:
`free_square` builds a whole squaring mod p1024 at **0 units** as that count reads, and `free_codec.py` the whole codec
word (no budgeted gate, right on every draw). The current count charges it one unit, and `free_square` costs 6,014. (B) While `IntGate.derive`
returned a gather's declared interval, a gather at a fixed index (wiring, 0 units) re-declared a free affine
remainder `T_m − 256·⌊T_m/256⌋` into `[0, 255]`, and `holeb_square` squared at 128 units, one carry a limb. `IntGate.derive`
now derives the hull of the column's intervals and 0, and `Proven` refuses it: the gather's derived interval leaves its
declared one, and the next squaring's `mma` operands leave u8.

**The constructions,** measured here (units, all `Restricted 128`, `Proven`, `DeclaredHold` on their inputs, and right):

* `ripple_square` 512 and `lookahead_square` 342 reproduce 6b's falsification test in this port (two exact passes; a carry
  a limb, or a carry a 3-limb group read as its own operand).
* `merged_square` 297: the same two passes with the carry merged into the group's first column (`T = c + f_m`, a free
  affine), so a group reads `T` and three more ≈30-bit columns, 120 operand bits: four limbs a carry.
* `onepass_square` 168: one merged pass (128 byte gates + 32 carries = 160), the `256^128 ≡ 105` fold, and an 8-byte
  fixup group with no carry; wrong with probability ≈ `105·c_top / 256^8` ≈ 2^−44.
* `seam_square` **160**, the cheapest: group 0's carry is computed first, the pass over limbs 4..127 ends in the top carry
  `c_top`, and limbs 0..3 are emitted last from `f_0 + 105·c_top`, so the fold needs no fixup. Wrong only when
  `105·c_top` bumps group 0's carry (`seam_fails`): measured 25 in 131,072 inputs on a node (2^−12.4;
  2^−12.3 modeled from `c_top`'s mean 8,193), so a 3,063-squaring word is right ≈55% of the time.
* `holeb_square` 128 under a `derive` that returns the declared interval, refused under the hull.

**Why 160 holds.** 160 = 128 byte gates + 32 group carries. Each output byte is an s8-family operand of the next
squaring, so a gate whose derived interval lies in a byte: a budgeted gate (one unit), or a free gate over narrower
budgeted ones, or a gather, whose hull is its column's and which pays each index bit it reads fresh (modeled, not a
theorem). The carries: `group_plan` searches every one-pass plan of byte gates and one carry gate a group whose
operands are columns or adjacent pairs `f_p + 256·f_{p+1}` within s32, the carry merged into the first, in u8 or s8
(balanced) digits and from any seam limb, under `OperandBits 128`: none uses fewer than 32 groups (u8 columns are 30 bits
at limb 0 down to 23 at limb 127, s8 a bit narrower, and five columns fit only near the top, where merges are too
few to save a group). The margin is small but real: a 31-group plan first appears at `OperandBits 130` (s8, merged) to
136 (u8, unmerged), so the 128-bit rule is what holds 160. Lazy reduction can't skip bytes, since each product's operands must be bytes. Nothing beats
the renormalization floor `L − d(ε) = 119` (121 at 2^−20): the 128 bytes alone exceed it, by 41 units a squaring. In
nonlinear units every construction also pays its column sums, `L² = 16,384` data products (`column_sums`), so
`seam_square` costs 16,544 against `e = 374`. Numbers are *measured* by building and running the circuits here unless
marked modeled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from verity.core.protocols.remote_bandwidth import codec

S32_MIN, S32_MAX = -(2 ** 31), 2 ** 31 - 1

F = codec.P1024
P = F.p
N_LIMBS = F.word_bytes  # 128 at p1024
DELTA = (1 << F.n) - P  # 256^128 − p = 105: the fold constant, 256^128 ≡ DELTA (mod p)

# An s8-family operand's format, as `Int8Fmt` (Model.lean).
S8 = (-128, 127)
U8 = (0, 255)


def clog2(n: int) -> int:
    """`Nat.clog 2 n`: the least `k` with `n ≤ 2^k` (`clog2 0 = clog2 1 = 0`)."""
    return 0 if n <= 1 else (n - 1).bit_length()


# ---- the model: Model.lean's gates, values, window, interval analysis and units ------------------------------------------


@dataclass(frozen=True)
class Op:
    """`Op ι`: an `input` slot by its binding, a `const`, an `affine` map `c + Σ coeff·v`, an s8-family `mma` coordinate
    (`C + Σ a_i·b_i`, clamped or wrapped to s32), a `gather` (`col.getD (gatherAt low idx) 0`: at `idx.toNat`, or for a
    table read, `low = n`, at `idx mod 2^n`), or a `budgeted` primitive of class `cls` (any function of its operands)."""

    kind: str
    binding: str = "registered"  # input: registered | coin | run
    slot: int = -1
    c: int = 0  # const
    coeff: tuple[int, ...] = ()  # affine
    fa: tuple[int, int] = S8  # mma operand formats
    fb: tuple[int, int] = S8
    k: int = 0  # mma products
    sat: bool = True  # mma: satfinite clamp, else wrap
    cls: str = "rounding"  # budgeted class
    low: int | None = None  # gather: a table read of the index's low `low` bits
    f: Callable[[list[int]], int] | None = field(default=None, compare=False)

    @property
    def is_derived(self) -> bool:
        """`Op.IsDerived`: a const, affine map, s8-family step or gather, whose interval analysis derives."""
        return self.kind in ("const", "affine", "mma", "gather")

    @property
    def charged(self) -> bool:
        """`Op.charged`: a run input, a gather or a budgeted gate."""
        return self.kind in ("gather", "budgeted") or (self.kind == "input" and self.binding == "run")

    @property
    def budget(self) -> str | None:
        """`Op.budget`: its budgeted class, if any."""
        return "gather" if self.kind == "gather" else self.cls if self.kind == "budgeted" else None


@dataclass(frozen=True)
class Gate:
    """`Gate ι`: an operation, its operands (earlier gates, by index), and its declared interval `[lo, hi]`."""

    op: Op
    args: tuple[int, ...]
    lo: int
    hi: int

    @property
    def size(self) -> int:
        """`Gate.size`: the values its declared interval holds."""
        return max(0, self.hi + 1 - self.lo)

    @property
    def bits(self) -> int:
        """`Gate.bits = Nat.clog 2 size`."""
        return clog2(self.size)

    def index_bits(self, fixed: bool = False) -> int:
        """`Gate.indexBits`: `n` for a table read of the index's low `n` bits, else `Nat.clog 2 args.length`,
        `⌈log₂(V + 1)⌉` for a column of `V` words (the `V` words and the out-of-range 0). `fixed=False` is the earlier
        count, `Nat.clog 2 (args.length − 1)`, under which a one-word column cost 0."""
        if self.op.low is not None:
            return self.op.low
        return clog2(len(self.args)) if fixed else clog2(len(self.args) - 1)


def clamp_s32(z: int) -> int:
    return max(S32_MIN, min(S32_MAX, z))


def wrap_s32(z: int) -> int:
    return (z - S32_MIN) % 2 ** 32 + S32_MIN


def raw(op: Op, x: dict[int, int], v: list[int]) -> int:
    """`Op.raw`: the exact value before an s8-family step's clamp or wrap."""
    if op.kind == "input":
        return x[op.slot]
    if op.kind == "const":
        return op.c
    if op.kind == "affine":
        return op.c + sum(cf * val for cf, val in zip(op.coeff, v))
    if op.kind == "mma":
        head = v[0] if v else 0
        tail = v[1:]
        return head + sum(tail[i] * tail[op.k + i] for i in range(op.k))
    if op.kind == "gather":
        idx = v[0] if v else 0
        col = v[1:]
        t = gather_at(op.low, idx)
        return col[t] if t < len(col) else 0
    assert op.f is not None
    return op.f(v)


def gather_at(low: int | None, z: int) -> int:
    """`gatherAt`: the column position a gather reads at index `z`, `z.toNat` or a table read's `(z mod 2^n).toNat`."""
    return max(z, 0) if low is None else z % 2 ** low


def out(op: Op, z: int) -> int:
    """`Op.out`: an s8-family step clamps (`satfinite`) or wraps to s32; every other op passes its exact value."""
    if op.kind != "mma":
        return z
    return clamp_s32(z) if op.sat else wrap_s32(z)


def operands(g: Gate, vs: list[int]) -> list[int]:
    return [vs[j] if j < len(vs) else 0 for j in g.args]


def value(g: Gate, x: dict[int, int], vs: list[int]) -> int:
    return out(g.op, raw(g.op, x, operands(g, vs)))


def values(C: list[Gate], x: dict[int, int]) -> list[int]:
    vs: list[int] = []
    for g in C:
        vs.append(value(g, x, vs))
    return vs


def val(C: list[Gate], x: dict[int, int], i: int) -> int:
    return values(C, x)[i]


def operands_ok(g: Gate, vs: list[int]) -> bool:
    """`Op.OperandsOk`: an s8-family step's `2k + 1` operands lie in their formats; other ops ask nothing."""
    if g.op.kind != "mma":
        return True
    o = operands(g, vs)
    if len(o) != 2 * g.op.k + 1:
        return False
    a, b = o[1:1 + g.op.k], o[1 + g.op.k:1 + 2 * g.op.k]
    return all(g.op.fa[0] <= z <= g.op.fa[1] for z in a) and all(g.op.fb[0] <= z <= g.op.fb[1] for z in b)


def ok(g: Gate, x: dict[int, int], vs: list[int]) -> bool:
    """`Gate.Ok`: the exact value in its declared interval, an s8-family step's operands in their formats and its
    declared interval within s32."""
    r = raw(g.op, x, operands(g, vs))
    if not (g.lo <= r <= g.hi and operands_ok(g, vs)):
        return False
    if g.op.kind == "mma":
        return S32_MIN <= g.lo and g.hi <= S32_MAX
    return True


def in_window(C: list[Gate], x: dict[int, int]) -> bool:
    """`InWindow`: every gate `Ok` on `x`."""
    vs: list[int] = []
    good = True
    for g in C:
        good = good and ok(g, x, vs)
        vs.append(value(g, x, vs))
    return good


def declared_hold(C: list[Gate], x: dict[int, int]) -> bool:
    """`DeclaredHold`: every non-derived gate's raw value lies in its declared interval."""
    vs: list[int] = []
    good = True
    for g in C:
        if not g.op.is_derived:
            r = raw(g.op, x, operands(g, vs))
            good = good and g.lo <= r <= g.hi
        vs.append(value(g, x, vs))
    return good


# ---- interval analysis (Model.lean's `derive`, `intervals`, `Proven`) ----------------------------------------------------

Iv = tuple[int, int]


def iv_add(a: Iv, b: Iv) -> Iv:
    return (a[0] + b[0], a[1] + b[1])


def iv_scale(c: int, a: Iv) -> Iv:
    return (c * a[0], c * a[1]) if c >= 0 else (c * a[1], c * a[0])


def iv_mul(a: Iv, b: Iv) -> Iv:
    ps = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (min(ps), max(ps))


def iv_within(a: Iv, lo: int, hi: int) -> bool:
    return lo <= a[0] and a[1] <= hi


def iv_join(a: Iv, b: Iv) -> Iv:
    return (min(a[0], b[0]), max(a[1], b[1]))


def derive(g: Gate, ivs: list[Iv]) -> Iv:
    """`Gate.derive`: the interval of a const, affine map or s8-family step from its operands' intervals, a gather's
    hull of its column's intervals and 0, and every other gate's declared interval."""
    a = [ivs[j] if j < len(ivs) else (0, 0) for j in g.args]
    op = g.op
    if op.kind == "const":
        return (op.c, op.c)
    if op.kind == "affine":
        acc = (op.c, op.c)
        for cf, iv in zip(op.coeff, a):
            acc = iv_add(acc, iv_scale(cf, iv))
        return acc
    if op.kind == "mma":
        acc = a[0] if a else (0, 0)
        tail = a[1:]
        for i in range(op.k):
            acc = iv_add(acc, iv_mul(tail[i], tail[op.k + i]))
        return acc
    if op.kind == "gather":
        acc = (0, 0)
        for iv in a[1:]:
            acc = iv_join(acc, iv)
        return acc
    return (g.lo, g.hi)


def intervals(C: list[Gate]) -> list[Iv]:
    ivs: list[Iv] = []
    for g in C:
        ivs.append(derive(g, ivs))
    return ivs


def proven(C: list[Gate]) -> list[str]:
    """`Proven`: each derived gate's (a gather's among them) derived interval within its declared one, each affine map's
    declared interval within s32, and each s8-family step's `2k + 1` operands within their formats and its declared
    interval within s32. Returns the violations, so `[]` means proven."""
    errs: list[str] = []
    ivs: list[Iv] = []
    for i, g in enumerate(C):
        d = derive(g, ivs)
        if g.op.is_derived and not iv_within(d, g.lo, g.hi):
            errs.append(f"gate {i} ({g.op.kind}): derived {d} outside declared [{g.lo}, {g.hi}]")
        if g.op.kind == "affine" and not (S32_MIN <= g.lo and g.hi <= S32_MAX):
            errs.append(f"gate {i} (affine): declared [{g.lo}, {g.hi}] outside s32")
        if g.op.kind == "mma":
            if len(g.args) != 2 * g.op.k + 1:
                errs.append(f"gate {i} (mma): {len(g.args)} operands, not 2k+1={2 * g.op.k + 1}")
            if not (S32_MIN <= g.lo and g.hi <= S32_MAX):
                errs.append(f"gate {i} (mma): declared [{g.lo}, {g.hi}] outside s32")
            a = g.args[1:1 + g.op.k]
            b = g.args[1 + g.op.k:1 + 2 * g.op.k]
            for j in a:
                if j < len(ivs) and not iv_within(ivs[j], g.op.fa[0], g.op.fa[1]):
                    errs.append(f"gate {i} (mma): operand {j} {ivs[j]} outside format {g.op.fa}")
            for j in b:
                if j < len(ivs) and not iv_within(ivs[j], g.op.fb[0], g.op.fb[1]):
                    errs.append(f"gate {i} (mma): operand {j} {ivs[j]} outside format {g.op.fb}")
        ivs.append(d)
    return errs


def operand_bits(B: int, C: list[Gate]) -> bool:
    """`OperandBits B`: every budgeted gate's operands' declared bits sum to at most `B` (a gather is exempt)."""
    return all(sum(C[j].bits for j in g.args if j < len(C)) <= B for g in C if g.op.kind == "budgeted")


def restricted(B: int, C: list[Gate]) -> bool:
    """`Restricted B = Proven ∧ OperandBits B`."""
    return not proven(C) and operand_bits(B, C)


def dynamic(C: list[Gate]) -> list[bool]:
    """`dynamic`: whether each gate's value depends on a run or coin input."""
    ds: list[bool] = []
    for g in C:
        if g.op.kind == "input":
            ds.append(g.op.binding != "registered")
        else:
            ds.append(any(ds[j] for j in g.args if j < len(ds)))
    return ds


def units(C: list[Gate], i: int, *, fixed: bool = False, dyn: list[bool] | None = None) -> int:
    """`units`: a budgeted gate is one unit when it reads a dynamic value; a gather is its fresh index bits (its
    `indexBits` less the most any earlier gather of the same index gate and kind paid) plus one when its column is
    dynamic too."""
    if dyn is None:
        dyn = dynamic(C)
    g = C[i]
    if g.op.kind == "budgeted":
        return 1 if any(dyn[j] for j in g.args) else 0
    if g.op.kind == "gather":
        j = g.args[0]
        if not dyn[j]:
            return 0
        kind = g.op.low is not None
        prior = max((C[h].index_bits(fixed) for h in range(i)
                     if C[h].op.kind == "gather" and (C[h].op.low is not None) == kind
                     and C[h].args and C[h].args[0] == j), default=0)
        col_dyn = any(dyn[a] for a in g.args[1:])
        return max(0, g.index_bits(fixed) - prior) + (1 if col_dyn else 0)
    return 0


def budgeted(C: list[Gate], *, fixed: bool = False) -> int:
    """`budgeted`: the circuit's total units."""
    dyn = dynamic(C)
    return sum(units(C, i, fixed=fixed, dyn=dyn) for i in range(len(C)))


def data_products(C: list[Gate], i: int, *, dyn: list[bool] | None = None) -> int:
    """`dataProducts`: the lanes of an s8-family step whose two multiplicands both depend on a run or coin input, one
    unit each whatever their range. A lane with a constant or registered multiplicand is a linear map, and free."""
    g = C[i]
    if g.op.kind != "mma":
        return 0
    dyn = dynamic(C) if dyn is None else dyn
    k = g.op.k
    return sum(1 for a, b in zip(g.args[1:1 + k], g.args[1 + k:1 + 2 * k]) if dyn[a] and dyn[b])


def products(C: list[Gate]) -> int:
    """`products`: the circuit's data products."""
    dyn = dynamic(C)
    return sum(data_products(C, i, dyn=dyn) for i in range(len(C)))


def nonlinear(C: list[Gate]) -> int:
    """`nonlinear`: the budgeted units (the count on main, `fixed=True`) and the data products, what the property's caps
    bound: every operation but a linear map with coefficients fixed before the coins."""
    return budgeted(C, fixed=True) + products(C)


# ---- a builder that keeps intervals, dynamic and the gather dedup incrementally (O(1) a gate) ----------------------------


class Builder:
    """Appends gates in canonical order, deriving a free gate's declared interval as `Proven` requires, and keeping the
    intervals, the `dynamic` flags and the gather dedup so a large free circuit builds and measures in linear time."""

    def __init__(self) -> None:
        self.gates: list[Gate] = []
        self.ivs: list[Iv] = []
        self.dyn: list[bool] = []
        self._lean: list[int] = []  # each gate's Lean units
        self._fixed: list[int] = []
        self._products = 0
        self._gmax: dict[tuple[int, bool], int] = {}  # (index gate, kind) -> max earlier-count indexBits so far
        self._gmaxf: dict[tuple[int, bool], int] = {}
        self._consts: dict[int, int] = {}

    def _append(self, op: Op, args: tuple[int, ...], lo: int, hi: int) -> int:
        i = len(self.gates)
        if op.kind == "input":
            self.dyn.append(op.binding != "registered")
        else:
            self.dyn.append(any(self.dyn[j] for j in args))
        g = Gate(op, args, lo, hi)
        self.ivs.append(derive(g, self.ivs))
        self.gates.append(g)
        lean = fixed = 0
        if op.kind == "budgeted":
            lean = fixed = 1 if any(self.dyn[j] for j in args) else 0
        elif op.kind == "gather":
            j = args[0]
            if self.dyn[j]:
                key = (j, op.low is not None)
                col_dyn = any(self.dyn[a] for a in args[1:])
                ib, ibf = g.index_bits(False), g.index_bits(True)
                lean = max(0, ib - self._gmax.get(key, 0)) + (1 if col_dyn else 0)
                fixed = max(0, ibf - self._gmaxf.get(key, 0)) + (1 if col_dyn else 0)
                self._gmax[key] = max(self._gmax.get(key, 0), ib)
                self._gmaxf[key] = max(self._gmaxf.get(key, 0), ibf)
        elif op.kind == "mma":
            self._products += data_products(self.gates, i, dyn=self.dyn)
        self._lean.append(lean)
        self._fixed.append(fixed)
        return i

    def inp(self, slot: int, lo: int, hi: int, binding: str = "coin") -> int:
        return self._append(Op("input", binding=binding, slot=slot), (), lo, hi)

    def const(self, c: int) -> int:
        if c not in self._consts:
            self._consts[c] = self._append(Op("const", c=c), (), c, c)
        return self._consts[c]

    def affine(self, c: int, terms: list[tuple[int, int]]) -> int:
        """`c + Σ coeff·gate`, its declared interval the derived one (so `Proven`); asserts it within s32."""
        args = tuple(j for _, j in terms)
        op = Op("affine", c=c, coeff=tuple(cf for cf, _ in terms))
        lo, hi = derive(Gate(op, args, 0, 0), self.ivs)
        assert S32_MIN <= lo and hi <= S32_MAX, f"affine [{lo}, {hi}] outside s32"
        return self._append(op, args, lo, hi)

    def mma(self, head: int, a: list[int], b: list[int], fa: tuple[int, int], fb: tuple[int, int]) -> int:
        """`head + Σ a_i·b_i`, its declared interval the derived one; asserts within s32."""
        assert len(a) == len(b)
        op = Op("mma", fa=fa, fb=fb, k=len(a))
        args = (head, *a, *b)
        lo, hi = derive(Gate(op, args, 0, 0), self.ivs)
        assert S32_MIN <= lo and hi <= S32_MAX, f"mma [{lo}, {hi}] outside s32"
        return self._append(op, args, lo, hi)

    def gather(self, idx: int, column: list[int], lo: int | None = None, hi: int | None = None,
               low: int | None = None) -> int:
        """`column[gatherAt low idx]`, declared `[lo, hi]`, or by default its derived hull (so `Proven`)."""
        op = Op("gather", low=low)
        args = (idx, *column)
        if lo is None or hi is None:
            lo, hi = derive(Gate(op, args, 0, 0), self.ivs)
        return self._append(op, args, lo, hi)

    def budgeted(self, f: Callable[[list[int]], int], args: list[int], lo: int, hi: int, cls: str = "rounding") -> int:
        return self._append(Op("budgeted", cls=cls, f=f), tuple(args), lo, hi)

    # -- measures, from the incremental caches --
    @property
    def lean_units(self) -> int:
        return sum(self._lean)

    @property
    def fixed_units(self) -> int:
        return sum(self._fixed)

    @property
    def products(self) -> int:
        return self._products

    @property
    def nonlinear_units(self) -> int:
        """The nonlinear units: the budgeted units on main and the data products."""
        return self.fixed_units + self._products

    def evaluate(self, x: dict[int, int]) -> tuple[list[int], bool, bool]:
        """Run on input `x`: the gates' values, and whether the window (`InWindow`) and `DeclaredHold` hold."""
        vs: list[int] = []
        window = hold = True
        for g in self.gates:
            window = window and ok(g, x, vs)
            if not g.op.is_derived:
                r = raw(g.op, x, operands(g, vs))
                hold = hold and g.lo <= r <= g.hi
            vs.append(value(g, x, vs))
        return vs, window, hold


# ---- the costed knobs ----------------------------------------------------------------------------------------------------


def renorm_floor(log2_inv_eps: int) -> int:
    """Lemma R's floor on a squaring's budgeted units, `L − d(ε)`, `d = d₀ + ⌈log₂(1/ε)/8⌉`, `d₀ = 4`."""
    return N_LIMBS - 4 - -(-log2_inv_eps // 8)


def e_per_squaring(log2_inv_eps: int) -> int:
    """`SquaringFloor`'s conjectured price in nonlinear units, `e = (2L − 1) + (L − d(ε))`: a data product for each of
    the square's `2L − 1` column sums, and Lemma R's carries (SquaringChain.lean)."""
    return 2 * N_LIMBS - 1 + renorm_floor(log2_inv_eps)


def counting_floor(log2_inv_eps: int) -> int:
    """The byte-gate floor on a squaring's budgeted units, modeled (not a theorem): if every output byte is a budgeted
    gate or a constant guess (the gather hull's premise, with hole B closed: a derived value of width 255 reads only
    gates that narrow, and a gather's hull is its column's while it pays each index bit it reads fresh), a squaring right on `ε` of
    inputs guesses at most `⌊log₂(1/ε)/8⌋` bytes, so `U ≥ ⌈(8L − log₂(1/ε))/8⌉`: 126 at `ε = 2^−20`, 123 at `2^−40`."""
    return -(-(8 * N_LIMBS - log2_inv_eps) // 8)


SQUARINGS = ((P + 1) // 4).bit_length() - 1  # 1021 at p1024
ROUNDS = codec.DEFAULT.rounds  # 3


# ---- the honest column sums, shared by every construction ----------------------------------------------------------------


def column_sums(b: Builder, a: list[int], fmt: tuple[int, int] = U8) -> list[int]:
    """The 255 columns of `a²` for 128 limbs `a` of format `fmt` (u8, or s8 balanced digits), each an s8-family step from
    `C = 0`, then the top 127 folded into the bottom by `256^128 ≡ DELTA`, all within s32: no budgeted unit, and
    `L² = 16,384` data products, a lane each. Returns the 128 folded column gates `f_0 … f_127`."""
    zero = b.const(0)
    cols: list[int] = []
    for m in range(2 * N_LIMBS - 1):
        idx = [i for i in range(max(0, m - N_LIMBS + 1), min(m, N_LIMBS - 1) + 1)]
        if idx:
            cols.append(b.mma(zero, [a[i] for i in idx], [a[m - i] for i in idx], fmt, fmt))
        else:
            cols.append(zero)
    folded = []
    for m in range(N_LIMBS):
        top = m + N_LIMBS
        if top < 2 * N_LIMBS - 1:
            folded.append(b.affine(0, [(1, cols[m]), (DELTA, cols[top])]))
        else:
            folded.append(b.affine(0, [(1, cols[m])]))
    return folded


# ---- construction A: the free squaring (0 units as Lean reads, thousands under the fix) -----------------------------------


def free_bit(b: Builder, r: int, k: int) -> int:
    """`[value(r) ≥ 2^k]` as a free one-word-column gather: `gather [2^k − r, const 1]`, `0 units` (Lean), `1` (fixed)."""
    idx = b.affine(2 ** k, [(-1, r)])
    one = b.const(1)
    return b.gather(idx, [one], 0, 1)


def free_split(b: Builder, u: int, nbits: int, hi_bits: int) -> tuple[int, int]:
    """Peel the low `hi_bits` bits of a non-negative value `u < 2^nbits` as a byte and the rest as the carry, all free: a
    gather a bit (MSB first, each off a fresh index, so no dedup), and affine sums for the byte and the carry."""
    bits: list[tuple[int, int]] = []
    r = u
    for k in range(nbits - 1, -1, -1):
        bk = free_bit(b, r, k)
        bits.append((k, bk))
        r = b.affine(0, [(1, r), (-(2 ** k), bk)])
    low = b.affine(0, [(2 ** k, bk) for k, bk in bits if k < hi_bits] or [(0, u)])
    carry = b.affine(0, [(2 ** (k - hi_bits), bk) for k, bk in bits if k >= hi_bits] or [(0, u)])
    return low, carry


def _carry_pass_free(b: Builder, f: list[int]) -> tuple[list[int], int]:
    out: list[int] = []
    carry = b.const(0)
    for m in range(N_LIMBS):
        u = b.affine(0, [(1, f[m]), (1, carry)])
        nbits = max(8, b.gates[u].hi.bit_length())
        low, carry = free_split(b, u, nbits, 8)
        out.append(low)
    return out, carry


def free_square(b: Builder, a: list[int], passes: int = 3) -> tuple[list[int], int]:
    """A whole squaring mod p1024 at **0 budgeted units** (Lean's count): the honest free column sums, then carry
    propagation done with free comparators. Folds the top carry by `256^128 ≡ DELTA` between passes and re-canonicalizes;
    the seam reopens limb 0 only, so three passes converge and the last pass's carry is 0 (asserted by the caller). No
    fold after the final pass — that would reopen limb 0 and break byte format. Returns 128 u8 limbs congruent to `a²`,
    and the final carry (0 in window)."""
    out = column_sums(b, a)
    carry = b.const(0)
    for p in range(passes):
        out, carry = _carry_pass_free(b, out)
        if p < passes - 1:
            out[0] = b.affine(0, [(1, out[0]), (DELTA, carry)])
    return out, carry


# ---- constructions B/C: the honest, counted carry passes (one budgeted gate a byte and a carry) --------------------------
#
# Each counted limb/carry gate reads an accumulator (a carry in, or the fold-merged first limb `T`) and `g` column words,
# value `acc + Σ_j g_j·256^j`. A byte gate returns `(value >> 8i) % 256`, declared `[0, 255]`; a carry gate returns
# `value >> 8w`, declared `[0, ⌊max value / 256^w⌋]`, tight so the `256^128 ≡ DELTA` fold's `out[0] += 105·carry` stays in
# s32 and OperandBits sees the real carry bits. Columns are ≈30 bits at the low end, so a merged group fits `w = 4` of them
# in `B = 128` operand bits (`T` + 3 columns ≈ 120), against `w = 3` when the carry is a separate operand (6b's form).


def _poly(weights: tuple[int, ...]) -> Callable[[list[int]], int]:
    return lambda v: sum(w * x for w, x in zip(weights, v))


def _byte_fn(weights: tuple[int, ...], i: int) -> Callable[[list[int]], int]:
    p = _poly(weights)
    return lambda v: (p(v) >> (8 * i)) % 256


def _carry_fn(weights: tuple[int, ...], w: int) -> Callable[[list[int]], int]:
    p = _poly(weights)
    return lambda v: p(v) >> (8 * w)


def _fhi(b: Builder, g: int) -> int:
    return b.gates[g].hi


def _emit_group(b: Builder, out: list[int], base: int, carry: int, carry_hi: int, cols: list[int],
                merge: bool) -> tuple[int, int]:
    """Emit one carry group over limbs `base … base + len(cols) − 1`: a budgeted byte gate a limb (into `out`) and a
    budgeted carry gate, declared tight. With `merge`, the carry in is folded into the first column by a free affine
    `T = carry + cols[0]`, so the carry isn't a separate operand and a wider group fits `B`; without it, the carry is a
    budgeted operand (6b's form). Returns the outgoing carry gate and its `hi`."""
    g = len(cols)
    if merge:
        t = b.affine(0, [(1, carry), (1, cols[0])])
        operands = [t, *cols[1:]]
        weights = tuple(256 ** j for j in range(g))
    else:
        operands = [carry, *cols]
        weights = (1,) + tuple(256 ** j for j in range(g))
    vmax = sum(wt * _fhi(b, o) for wt, o in zip(weights, operands))
    for i in range(g):
        out[base + i] = b.budgeted(_byte_fn(weights, i), operands, 0, 255)
    hi = vmax >> (8 * g)
    return b.budgeted(_carry_fn(weights, g), operands, 0, hi), hi


def _carry_pass(b: Builder, f: list[int], start: int, stop: int, w: int, merge: bool) -> tuple[list[int], int]:
    """One left-to-right carry pass over limbs `[start, stop)` in fixed groups of `w`. Returns the limbs and the final
    carry gate."""
    out = list(f)
    carry, hi = b.const(0), 0
    for base in range(start, stop, w):
        carry, hi = _emit_group(b, out, base, carry, hi, f[base:min(base + w, stop)], merge)
    return out, carry


def _greedy_pass(b: Builder, f: list[int], start: int, stop: int, carry: int | None = None, carry_hi: int = 0,
                 cap: int = 128) -> tuple[list[int], int, list[int]]:
    """One merged carry pass whose group widths are greedy: each group takes columns while `T`'s bits plus the extra
    columns' bits stay `≤ cap` (OperandBits). Starts from `carry` (default 0). Returns the limbs, the final carry gate
    and the group widths."""
    out = list(f)
    if carry is None:
        carry = b.const(0)
    base, widths = start, []
    while base < stop:
        bits = clog2(carry_hi + _fhi(b, f[base]) + 1)
        g = 1
        while base + g < stop and bits + clog2(_fhi(b, f[base + g]) + 1) <= cap:
            bits += clog2(_fhi(b, f[base + g]) + 1)
            g += 1
        carry, carry_hi = _emit_group(b, out, base, carry, carry_hi, f[base:base + g], merge=True)
        widths.append(g)
        base += g
    return out, carry, widths


def _fixup(b: Builder, out: list[int], t: int) -> None:
    """Re-canonicalize the low `t` limbs after the fold reopened limb 0: one merged group of byte gates over
    `[out[0], out[1], …, out[t−1]]` (limb 0 wide, the rest bytes, so `t` up to ≈12 fits `B`), its top carry dropped
    (no gate). Wrong only when the fold's `105·c_top` ripples past limb `t − 1`, probability ≈ `105·c_top / 256^t`."""
    cols = out[:t]
    weights = tuple(256 ** j for j in range(t))
    out[:t] = [b.budgeted(_byte_fn(weights, i), cols, 0, 255) for i in range(t)]


def ripple_square(b: Builder, a: list[int]) -> tuple[list[int], int]:
    """6b's ripple construction in this port: two full ripple passes (`w = 1`), 512 budgeted units."""
    f = column_sums(b, a)
    out, carry = _carry_pass(b, f, 0, N_LIMBS, 1, merge=False)
    out[0] = b.affine(0, [(1, out[0]), (DELTA, carry)])
    return _carry_pass(b, out, 0, N_LIMBS, 1, merge=False)


def lookahead_square(b: Builder, a: list[int], w: int = 3) -> tuple[list[int], int]:
    """6b's 3-limb look-ahead construction (carry a separate operand): two passes, 342 budgeted units at `w = 3`."""
    f = column_sums(b, a)
    out, carry = _carry_pass(b, f, 0, N_LIMBS, w, merge=False)
    out[0] = b.affine(0, [(1, out[0]), (DELTA, carry)])
    return _carry_pass(b, out, 0, N_LIMBS, w, merge=False)


def merged_square(b: Builder, a: list[int]) -> tuple[list[int], int]:
    """6b's two-pass form with the carry merged into `T` and greedy group widths: two exact passes."""
    f = column_sums(b, a)
    out, carry, _ = _greedy_pass(b, f, 0, N_LIMBS)
    out[0] = b.affine(0, [(1, out[0]), (DELTA, carry)])
    out, carry, _ = _greedy_pass(b, out, 0, N_LIMBS)
    return out, carry


def onepass_square(b: Builder, a: list[int], fixup: int = 8) -> tuple[list[int], int]:
    """One merged greedy pass over the 128 wide columns, the `256^128 ≡ DELTA` fold, then a `fixup`-limb re-canonicalizing
    group (`fixup` byte gates, no carry): `160 + fixup` units, wrong with probability ≈ `105·c_top / 256^fixup`
    (below 2^−40 at `fixup = 8`). Returns the limbs and the pass's top carry."""
    f = column_sums(b, a)
    out, carry, _ = _greedy_pass(b, f, 0, N_LIMBS)
    out[0] = b.affine(0, [(1, out[0]), (DELTA, carry)])
    _fixup(b, out, fixup)
    return out, carry


SEAM_W = 4


def seam_square(b: Builder, a: list[int]) -> tuple[list[int], int]:
    """The **zero-overhead seam**, the cheapest squaring reached. Group 0's carry `c_4` is computed first from
    `(f_0, f_1, f_2, f_3)` with carry in 0; a merged greedy pass over limbs 4..127 starts from it and ends in the top
    carry `c_top`; then `T_0 = f_0 + 105·c_top` (free) and limbs 0..3 are byte gates over `(T_0, f_1, f_2, f_3)`. The
    pass's own top carry is the fold, so no fixup: 128 byte gates and one carry a group. Wrong only when `105·c_top`
    bumps group 0's carry, i.e. `(X_0 mod 2^32) + 105·c_top ≥ 2^32` for `X_0 = Σ_{m<4} f_m 256^m` (`seam_fails`).
    Returns the limbs and `c_top`."""
    f = column_sums(b, a)
    out = list(f)
    g0 = f[:SEAM_W]
    weights = tuple(256 ** j for j in range(SEAM_W))
    vmax = sum(wt * _fhi(b, o) for wt, o in zip(weights, g0))
    c4 = b.budgeted(_carry_fn(weights, SEAM_W), g0, 0, vmax >> (8 * SEAM_W))
    out, c_top, _ = _greedy_pass(b, f, SEAM_W, N_LIMBS, carry=c4, carry_hi=vmax >> (8 * SEAM_W))
    t0 = b.affine(0, [(1, f[0]), (DELTA, c_top)])
    cols = [t0, *f[1:SEAM_W]]
    for i in range(SEAM_W):
        out[i] = b.budgeted(_byte_fn(weights, i), cols, 0, 255)
    return out, c_top


# ---- the group plan: the fewest carry groups one pass can use under `B` --------------------------------------------------


def _within_s32(iv: Iv) -> bool:
    return S32_MIN <= iv[0] and iv[1] <= S32_MAX


def _iv_bits(iv: Iv) -> int:
    return clog2(iv[1] + 1 - iv[0])


def _carry_iv(x: Iv, w: int, signed: bool) -> Iv:
    """A group's carry out over `w` limbs of value interval `x`: `⌊x/256^w⌋`, or for s8 digits `⌊(x + K)/256^w⌋`,
    `K = 128·Σ_{i<w} 256^i` (each digit is `u − 128` for `u` the byte of `x + K`)."""
    k = 128 * (256 ** w - 1) // 255 if signed else 0
    return ((x[0] + k) // 256 ** w, (x[1] + k) // 256 ** w)


Plan = tuple[tuple[tuple[int, int], ...], ...]


def group_plan(cols: list[Iv], signed: bool = False, cap: int = 128, merge: bool = True,
               start: int = 0) -> tuple[int, Plan]:
    """The fewest carry groups one pass can cover the 128 folded columns with, given their derived intervals `cols`: a
    lower bound on `seam_square`'s family, whose units are 128 byte gates plus one carry gate a group.

    A group over limbs `[m, m + w)` reads `T = c + o_0` (its carry in merged into its first operand by a free affine
    within s32) and operands `o_1 …`, each a column `f_p` or, with `merge`, an adjacent pair `f_p + 256·f_{p+1}` (a free
    affine within s32); its carry gate reads them all, so their declared bits sum to at most `cap` (OperandBits). The
    pass starts at limb `start` with carry in 0 (the seam: its group's bytes, emitted last, read the wrapped carry too,
    which only adds bits, so this relaxes it) and runs cyclically: a carry out of limb 127 enters limb 0 times
    `DELTA` (`256^128 ≡ DELTA`), and no group straddles that fold. Exact over the family by memoized search on
    (position, carry interval). Returns the group count and each group's operands as `(limb, span)`."""
    n = len(cols)
    order = [(start + i) % n for i in range(n)]
    wrap = n - start  # the index in `order` where limb 0 falls; a group ends at or before it

    def operand(i: int, span: int) -> Iv | None:
        if i + span > n or (span == 2 and (order[i + 1] != order[i] + 1 or i + 1 == wrap)):
            return None
        p = order[i]
        iv = cols[p] if span == 1 else iv_add(cols[p], iv_scale(256, cols[p + 1]))
        return iv if _within_s32(iv) else None

    def groups(m: int, c: Iv) -> list[tuple[tuple[tuple[int, int], ...], int, Iv]]:
        found: list[tuple[tuple[tuple[int, int], ...], int, Iv]] = []
        stop = wrap if m < wrap else n

        def extend(i: int, ops: tuple[tuple[int, int], ...], bits: int, x: Iv) -> None:
            if ops:
                found.append((ops, i, x))
            for span in ((1, 2) if merge else (1,)):
                iv = operand(i, span) if i + span <= stop else None
                if iv is None:
                    continue
                if ops:
                    nb = _iv_bits(iv)
                else:
                    t = iv_add(c, iv)
                    if not _within_s32(t):
                        continue
                    nb = _iv_bits(t)
                if bits + nb <= cap:
                    extend(i + span, (*ops, (order[i], span)), bits + nb, iv_add(x, iv_scale(256 ** (i - m), iv)))

        extend(m, (), 0, c)
        return found

    memo: dict[tuple[int, Iv], tuple[int, Plan] | None] = {}

    def best(m: int, c: Iv) -> tuple[int, Plan] | None:
        if m == n:
            return 0, ()
        key = (m, c)
        if key not in memo:
            res: tuple[int, Plan] | None = None
            for ops, end, x in groups(m, c):
                out = _carry_iv(x, end - m, signed)
                if end == wrap and 0 < wrap < n:
                    out = iv_scale(DELTA, out)
                sub = best(end, out)
                if sub is not None and (res is None or sub[0] + 1 < res[0]):
                    res = (sub[0] + 1, (ops, *sub[1]))
            memo[key] = res
        return memo[key]

    res = best(0, (0, 0))
    assert res is not None, "no plan fits the cap"
    return res


def column_intervals(fmt: tuple[int, int] = U8) -> list[Iv]:
    """The 128 folded columns' derived intervals for limbs of format `fmt`, read off `column_sums`'s gates."""
    b = Builder()
    f = column_sums(b, [b.inp(i, fmt[0], fmt[1]) for i in range(N_LIMBS)], fmt)
    return [b.ivs[g] for g in f]


def _holeb_limb(b: Builder, t: int, carry_out: int | None = None) -> tuple[int, int]:
    """One limb under hole B: the carry `h = ⌊t/256⌋` a budgeted gate (or `carry_out`, already paid), and the byte
    `t − 256·h`, a free affine map, re-declared `[0, 255]` by a gather at the fixed index 0 (wiring: 0 units). A `derive`
    that returns a gather's declared interval lets the next squaring's `mma` take it as a u8 operand; Model.lean
    derives the hull of its column and 0, the affine map's wide interval, and `Proven` refuses it."""
    if carry_out is None:
        carry_out = b.budgeted(lambda v: v[0] >> 8, [t], 0, _fhi(b, t) >> 8)
    rem = b.affine(0, [(1, t), (-256, carry_out)])
    return b.gather(b.const(0), [rem], 0, 255), carry_out


def holeb_square(b: Builder, a: list[int]) -> tuple[list[int], int]:
    """**Hole B** (Lean-only, closed by the gather hull): one budgeted unit a limb, 128 a
    squaring, and refused by `Proven` under the hull. Every byte is its limb's `T_m − 256·⌊T_m/256⌋`, free, re-declared
    into `[0, 255]` by a fixed-index gather, so only the carries pay. The seam
    is `seam_square`'s: group 0's carry `c_4` first (one gate over `f_0 … f_3`), a ripple over limbs 4..127 to `c_top`,
    then limbs 0..3 from `f_0 + 105·c_top`, the last of them closed by `c_4` itself (no gate). Wrong exactly when
    `seam_fails`, where limb 3's re-declared byte leaves `[0, 255]` (DeclaredHold fails). Returns the limbs and `c_top`."""
    f = column_sums(b, a)
    out = list(f)
    weights = tuple(256 ** j for j in range(SEAM_W))
    vmax = sum(wt * _fhi(b, o) for wt, o in zip(weights, f[:SEAM_W]))
    carry = b.budgeted(_carry_fn(weights, SEAM_W), f[:SEAM_W], 0, vmax >> (8 * SEAM_W))
    c4 = carry
    for m in range(SEAM_W, N_LIMBS):
        out[m], carry = _holeb_limb(b, b.affine(0, [(1, f[m]), (1, carry)]))
    c_top = carry
    t = b.affine(0, [(1, f[0]), (DELTA, c_top)])
    for m in range(SEAM_W):
        out[m], h = _holeb_limb(b, t, c4 if m == SEAM_W - 1 else None)
        if m < SEAM_W - 1:
            t = b.affine(0, [(1, f[m + 1]), (1, h)])
    return out, c_top


def folded_columns(a: int) -> list[int]:
    """The 128 folded column values `f_m` of `a²` for `a`'s byte limbs, by one big-integer product (the columns are
    below `2^32`, so packing limbs `2^32` apart keeps them apart). Matches `column_sums`'s gates on every input."""
    packed = sum(d << (32 * i) for i, d in enumerate(limbs(a)))
    sq = packed * packed
    cols = [(sq >> (32 * m)) & 0xFFFFFFFF for m in range(2 * N_LIMBS - 1)]
    return [cols[m] + (DELTA * cols[m + N_LIMBS] if m + N_LIMBS < len(cols) else 0) for m in range(N_LIMBS)]


def seam_fails(a: int) -> bool:
    """Whether `seam_square` is wrong on `a`: the fold's `105·c_top` crosses a `2^32` boundary of group 0's value."""
    f = folded_columns(a)
    x0 = sum(f[m] << (8 * m) for m in range(SEAM_W))
    c_top = sum(v << (8 * m) for m, v in enumerate(f)) >> (8 * N_LIMBS)
    return (x0 % 2 ** (8 * SEAM_W)) + DELTA * c_top >= 2 ** (8 * SEAM_W)


def limbs(a: int) -> list[int]:
    """`a`'s 128 u8 byte limbs, little-endian."""
    return list(a.to_bytes(N_LIMBS, "little"))


def output_value(out_vals: list[int]) -> int:
    return sum(d << (8 * m) for m, d in enumerate(out_vals))


CONSTRUCTIONS: dict[str, Callable[[Builder, list[int]], tuple[list[int], int]]] = {
    "free_square": free_square, "ripple_square": ripple_square, "lookahead_square": lookahead_square,
    "merged_square": merged_square, "onepass_square": onepass_square, "seam_square": seam_square,
    "holeb_square": holeb_square,
}


def measure(name: str, inputs: list[int]) -> dict:
    """One construction's units (Lean's and fixed), gate count, restriction and its outputs on `inputs`."""
    b = Builder()
    a = [b.inp(i, 0, 255) for i in range(N_LIMBS)]
    out, _ = CONSTRUCTIONS[name](b, a)
    right = window = hold = 0
    for x in inputs:
        vals, w, h = b.evaluate(dict(enumerate(limbs(x))))
        o = [vals[g] for g in out]
        right += all(0 <= v <= 255 for v in o) and output_value(o) % P == x * x % P
        window += w
        hold += h
    budgeted_gates = [g for g in b.gates if g.op.kind == "budgeted"]
    return {"lean_units": b.lean_units, "fixed_units": b.fixed_units, "products": b.products,
            "nonlinear_units": b.nonlinear_units, "gates": len(b.gates),
            "budgeted_gates": len(budgeted_gates), "gathers": sum(g.op.kind == "gather" for g in b.gates),
            "proven": proven(b.gates) == [], "operand_bits_128": operand_bits(128, b.gates),
            "max_operand_bits": max((sum(b.gates[j].bits for j in g.args) for g in budgeted_gates), default=0),
            "inputs": len(inputs), "right": right, "in_window": window, "declared_hold": hold}


def main() -> None:
    import argparse
    import json
    import random

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the JSON file to write")
    ap.add_argument("--inputs", type=int, default=64, help="random inputs each construction is evaluated on")
    ap.add_argument("--seam-samples", type=int, default=1 << 17, help="random inputs `seam_fails` is evaluated on")
    args = ap.parse_args()
    rng = random.Random(20261018)
    inputs = [rng.randrange(1 << 1024) for _ in range(args.inputs)] + [P - 1, 0, 1 << 1023]
    rows = {name: measure(name, inputs) for name in CONSTRUCTIONS}
    fails, ctops = 0, []
    for _ in range(args.seam_samples):
        x = rng.randrange(1 << 1024)
        ctops.append(sum(v << (8 * m) for m, v in enumerate(folded_columns(x))) >> (8 * N_LIMBS))
        fails += seam_fails(x)
    b = Builder()
    f = column_sums(b, [b.inp(i, 0, 255) for i in range(N_LIMBS)])
    _, _, widths = _greedy_pass(b, f, 0, N_LIMBS)
    plans = {}
    for name, fmt in (("u8", U8), ("s8", S8)):
        cols = column_intervals(fmt)
        for merge in (False, True):
            def fewest(cap: int) -> tuple[int, int]:
                return min((group_plan(cols, fmt == S8, cap, merge, s)[0], s) for s in range(N_LIMBS))
            groups, start = fewest(128)
            cap = 128
            while fewest(cap)[0] >= groups:
                cap += 1
            plans[f"{name}{'_merge' if merge else ''}"] = {
                "groups": groups, "units": N_LIMBS + groups, "seam_limb": start, "cap_for_one_fewer": cap,
                "column_bits": [_iv_bits(iv) for iv in cols]}
    best = rows["seam_square"]["fixed_units"]
    result = {
        "model": "Security/Definitions/Accounting/Gates/Model.lean @ 8225774747, ported scalar; lean_units is "
                 "the count before the gather fix",
        "constructions": rows,
        "seam": {"samples": args.seam_samples, "fails": fails, "c_top_mean": sum(ctops) / len(ctops),
                 "c_top_max": max(ctops), "model_rate": DELTA * (sum(ctops) / len(ctops)) / 2 ** 32},
        "column_bits": [clog2(b.gates[g].hi + 1) for g in f], "greedy_widths": widths,
        "group_plans": plans,
        "conjecture": {f"2^-{t}": {"e": e_per_squaring(t), "E": ROUNDS * SQUARINGS * e_per_squaring(t),
                                   "renorm_floor": renorm_floor(t), "counting_floor": counting_floor(t)}
                       for t in (20, 40)},
        "best": {"per_squaring": best, "per_word_k_S_e": ROUNDS * SQUARINGS * best},
        "hole_b_before_2459": {"per_squaring": rows["holeb_square"]["fixed_units"],
                               "proven": rows["holeb_square"]["proven"]},
        "squarings": SQUARINGS, "rounds": ROUNDS,
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps({k: (v["lean_units"], v["fixed_units"], v["right"], v["proven"]) for k, v in rows.items()}))
    print(json.dumps(result["seam"]), json.dumps(result["best"]), json.dumps(result["hole_b_before_2459"]))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "column_bits"} for k, v in plans.items()}))


if __name__ == "__main__":
    main()
