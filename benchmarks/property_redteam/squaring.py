"""Squaring mod p1024 in 6b's circuit model, measured in budgeted units: the attack on `EncodeChainHardness`.

**The target** (`Security/Models/Accounting/Assumptions/EncodeChain.lean`, 6b's branch cursor/circuit-encode-6cd3 at
a949c3f5ed, unchanged by the proofs PR #2363). The encode bound rests on `EncodeChainHardness F k B E ε`: a restricted
circuit computing `q` words of the codec, less parity, on more than `ε` of the draws costs `q·E` budgeted units less a
word-per-run-bit discount. Its headline is `E = k·S·e`, `e = L − d(ε)` a squaring, `L = n/8 = 128`, `d = d₀ + ⌈log₂(1/ε)/8⌉`,
`d₀ = 4`: `e = 119` at `ε = 2^−40` (`E = 364,497` a word), `e = 121` at `ε = 2^−20` (`E = 370,623`). A squaring circuit under
`e` budgeted units on more than `ε` of its inputs falsifies it.

**What this module ports.** A scalar (Python-int, so p1024-sized) port of `Security/Definitions/Accounting/Gates/Model.lean`:
`Op`, `Gate`, `raw`, `out`, `Ok`/`InWindow`, `derive`/`intervals`, `Proven`, `DeclaredHold`, `OperandBits`, `dynamic`,
`units` and `budgeted`, plus `Restricted` (`Codec.lean`). It carries the units count two ways: `units(..., fixed=False)` is
Model.lean's as written, `Nat.clog 2 (args.length − 1)` fresh index bits a gather; `fixed=True` is the one-line fix below,
`Nat.clog 2 args.length`. The difference is the attack.

**The model bug, and the squarings it buys.** Model.lean's `Gate.indexBits := Nat.clog 2 (g.args.length − 1)`. A gather
whose column is one word (`args = [idx, c]`, `args.length = 2`) has `indexBits = Nat.clog 2 1 = 0`, and when that word is a
constant its `units` are `0 − 0 + 0 = 0`: it is free. Its value is `c·[idx ≤ 0]` (`raw = col.getD idx.toNat 0`, and
`idx.toNat = 0` exactly when `idx ≤ 0`), a comparison of an arbitrary s32 value against 0, exempt from `OperandBits`
(gathers are). So a bit of any value `v ∈ [0, 2^b)` costs nothing: `b_k = gather [2^k − r, const 1]` is `[r ≥ 2^k]`, and
`r ← r − 2^k·b_k` (affine, free) peels it off, MSB first. From the bits, bytes and carries are affine, so carry propagation
— the one step the conjecture argues no free gate can do (`Renormalization`: a byte can't be made from a wide value) — is
free. `free_square` builds a whole squaring mod p1024 at **0 budgeted units** (Lean's count), congruent to `a²` and in
byte format for the next squaring. So `EncodeChainHardness` and `Renormalization` are **false as the Lean reads today**, for
every `E > 0` and every `ε`, not only below a floor. property/v1's Python already charges a one-word range gather one unit
(`stacking._BITS["range"] = n.bit_length()`), so the gap is Lean-only; the fix is `indexBits := Nat.clog 2 g.args.length`
(or `Proven` requiring the index in `[0, V−1]`), which this module measures as `fixed=True`.

**Under the fix.** The free comparator costs one unit a bit, so `free_square` costs 6,014 — far above `e`. The honest
constructions, measured here (units, all `Restricted 128`, `Proven`, `DeclaredHold` on their inputs, and right):

* `ripple_square` 512 and `lookahead_square` 342 reproduce 6b's falsification test in this port (two exact passes; a carry
  a limb, or a carry a 3-limb group read as its own operand).
* `merged_square` 297: the same two passes with the carry merged into the group's first column (`T = c + f_m`, a free
  affine), so a group reads `T` and three more ≈30-bit columns, 120 operand bits: four limbs a carry.
* `onepass_square` 168: one merged pass (128 byte gates + 32 carries = 160), the `256^128 ≡ 105` fold, and an 8-byte
  fixup group with no carry; wrong with probability ≈ `105·c_top / 256^8` ≈ 2^−44.
* `seam_square` **160**, the cheapest reached: group 0's carry is computed first, the pass over limbs 4..127 ends in the
  top carry `c_top`, and limbs 0..3 are emitted last from `f_0 + 105·c_top`, so the fold needs no fixup. Wrong only when
  `105·c_top` bumps group 0's carry (`seam_fails`): measured 4 in 32,768 inputs (2^−13.0; 2^−12.3 modeled from
  `c_top`'s mean 8,185), so a 3,063-squaring word is right ≈55% of the time.

160 = 128 byte gates (each output byte of a squaring is an s8-family operand of the next, so a non-derived gate of
`[0, 255]`, and a budgeted gate is the cheapest such: a gather pays ≥ 1 unit an index bit) + 32 group carries (a carry
gate reads `T` and three ≈30-bit columns; a fifth column exceeds `B = 128`). Signed limbs shave a bit off each column
and still need 32 groups (modeled); lazy reduction across squarings can't skip the bytes, since each product's operands
must be bytes, and the seam already leaves the result non-canonical in `[0, 256^128)`. All sit above `e = 119` and the
counting floor `⌈(8L − log₂(1/ε))/8⌉` (126 at 2^−20, 123 at 2^−40): the fixed conjecture survives, its `e` about 41
units below the best construction. Numbers are *measured* by building and running the circuits here unless marked
modeled.
"""

from __future__ import annotations

import math
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
    (`C + Σ a_i·b_i`, clamped or wrapped to s32), a `gather` (`col.getD idx.toNat 0`), or a `budgeted` primitive of class
    `cls` (any function of its operands)."""

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
    f: Callable[[list[int]], int] | None = field(default=None, compare=False)

    @property
    def is_derived(self) -> bool:
        """`Op.IsDerived`: a const, affine map or s8-family step, whose interval analysis derives."""
        return self.kind in ("const", "affine", "mma")

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
        """A gather's `⌈log₂ V⌉` for its column of `V = args.length − 1` words. Model.lean writes `Nat.clog 2 (V)` as
        `Nat.clog 2 (args.length − 1)`, which is the bug (one word costs 0); `fixed` is `Nat.clog 2 args.length`,
        `⌈log₂(V + 1)⌉`, the `V + 1` outcomes (the `V` words and the out-of-range 0)."""
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
        t = max(idx, 0)  # (v.headD 0).toNat
        return col[t] if t < len(col) else 0
    assert op.f is not None
    return op.f(v)


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


def derive(g: Gate, ivs: list[Iv]) -> Iv:
    """`Gate.derive`: the interval of a const, affine map or s8-family step from its operands' intervals; a declared
    interval for every other gate (a gather among them)."""
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
    return (g.lo, g.hi)


def intervals(C: list[Gate]) -> list[Iv]:
    ivs: list[Iv] = []
    for g in C:
        ivs.append(derive(g, ivs))
    return ivs


def proven(C: list[Gate]) -> list[str]:
    """`Proven`: each derived gate's derived interval within its declared one, each affine map's declared interval within
    s32, and each s8-family step's `2k + 1` operands within their formats and its declared interval within s32. Returns
    the violations, so `[]` means proven."""
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
    `indexBits` less the most any earlier gather of the same index gate paid) plus one when its column is dynamic too."""
    if dyn is None:
        dyn = dynamic(C)
    g = C[i]
    if g.op.kind == "budgeted":
        return 1 if any(dyn[j] for j in g.args) else 0
    if g.op.kind == "gather":
        j = g.args[0]
        if not dyn[j]:
            return 0
        prior = max((C[h].index_bits(fixed) for h in range(i)
                     if C[h].op.kind == "gather" and C[h].args and C[h].args[0] == j), default=0)
        col_dyn = any(dyn[a] for a in g.args[1:])
        return max(0, g.index_bits(fixed) - prior) + (1 if col_dyn else 0)
    return 0


def budgeted(C: list[Gate], *, fixed: bool = False) -> int:
    """`budgeted`: the circuit's total units."""
    dyn = dynamic(C)
    return sum(units(C, i, fixed=fixed, dyn=dyn) for i in range(len(C)))


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
        self._gmax: dict[int, int] = {}  # index gate -> max Lean indexBits of its gathers so far
        self._gmaxf: dict[int, int] = {}
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
                col_dyn = any(self.dyn[a] for a in args[1:])
                ib, ibf = g.index_bits(False), g.index_bits(True)
                lean = max(0, ib - self._gmax.get(j, 0)) + (1 if col_dyn else 0)
                fixed = max(0, ibf - self._gmaxf.get(j, 0)) + (1 if col_dyn else 0)
                self._gmax[j] = max(self._gmax.get(j, 0), ib)
                self._gmaxf[j] = max(self._gmaxf.get(j, 0), ibf)
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

    def gather(self, idx: int, column: list[int], lo: int, hi: int) -> int:
        return self._append(Op("gather"), (idx, *column), lo, hi)

    def budgeted(self, f: Callable[[list[int]], int], args: list[int], lo: int, hi: int, cls: str = "rounding") -> int:
        return self._append(Op("budgeted", cls=cls, f=f), tuple(args), lo, hi)

    # -- measures, from the incremental caches --
    @property
    def lean_units(self) -> int:
        return sum(self._lean)

    @property
    def fixed_units(self) -> int:
        return sum(self._fixed)

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


def e_per_squaring(log2_inv_eps: int) -> int:
    """6b's conjectured floor `e = L − d(ε)`, `d = d₀ + ⌈log₂(1/ε)/8⌉`, `d₀ = 4` (EncodeChain.lean)."""
    return N_LIMBS - 4 - -(-log2_inv_eps // 8)


def counting_floor(log2_inv_eps: int) -> int:
    """The red team's information floor on a squaring's budgeted units, modeled: the 128 output bytes carry `8L` bits, the
    circuit's dynamic information is its budgeted units' output bits (free affine/mma are deterministic and a fixed gather
    adds nothing), so on a fraction `ε` of inputs a byte-output squaring needs `U` with `8U ≥ 8L − log₂(1/ε)`, i.e.
    `U ≥ ⌈(8L − log₂(1/ε))/8⌉`. At `ε = 2^−20` this is 126, at `2^−40` it is 123; both above `e`, so `e` is conservative."""
    return -(-(8 * N_LIMBS - log2_inv_eps) // 8)


SQUARINGS = ((P + 1) // 4).bit_length() - 1  # 1021 at p1024
ROUNDS = codec.DEFAULT.rounds  # 3


# ---- the honest column sums, shared by every construction ----------------------------------------------------------------


def column_sums(b: Builder, a: list[int]) -> list[int]:
    """The 255 columns of `a²` for 128 u8 limbs `a`, each an s8-family step (u8×u8 from `C = 0`), then the top 127 folded
    into the bottom by `256^128 ≡ DELTA`: 128 free `mma`/`affine` gates a column, all non-negative and within s32. Returns
    the 128 folded column gates `f_0 … f_127`."""
    zero = b.const(0)
    cols: list[int] = []
    for m in range(2 * N_LIMBS - 1):
        idx = [i for i in range(max(0, m - N_LIMBS + 1), min(m, N_LIMBS - 1) + 1)]
        if idx:
            cols.append(b.mma(zero, [a[i] for i in idx], [a[m - i] for i in idx], U8, U8))
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
