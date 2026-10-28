"""Model.lean's gates on the canonical circuit: the one Python statement of the gate model the red team's studies build in.

A circuit of `Security/Definitions/Accounting/Gates/Model.lean` is a `verity.ml.circuits.program.Program` whose root body holds
its gates in the order they were built, and its values on an input are the `Trace` that
`verity.ml.circuits.evaluation.reference.evaluate_region` computes. What the IR has no place for stays in the `Builder` that
makes the Program: each gate's declared interval `[lo, hi]`, each input's slot and binding, and the running counts a large
circuit is measured with.

* **Values.** A gate's value is a `Value<64>` word read as two's complement (`signed`). It is Model.lean's integer exactly
  while it lies in s64, and every value of the studies' circuits does (`Proven` puts their free gates within s32).
* **Operations.** Each of `Op`'s forms but an input is an unregistered primitive, one per distinct operation (`op_const`,
  `op_affine`, `op_mma`, `op_gather`, `op_budgeted`). Its parameters are its attributes, `raw` is `Op.raw` on integers, and its
  evaluator is `Op.out ∘ Op.raw` on encodings. An input is the Program's `RegisteredInput64_v1` gate (a registered input) or
  `Input64_v1` gate (a coin or run input), whose value the trace prescribes. A gather reads its column as the studies' Model.lean
  did: a negative index reads entry 0, and an index past the column reads 0.
* **The window and the analyses** are functions of the Builder's gates: `ok`, `InWindow` and `DeclaredHold` read the trace too
  (`Builder.evaluate`); `derive`/`intervals` (a gather's interval is the hull of its column's and 0), `Proven`, `OperandBits`,
  `dynamic`, `indexBits`, `units` (fresh index bits per index gate and kind) and `budgeted`, `dataProducts`, `products` and
  `nonlinear` read the declarations alone. `units(..., fixed=True)` is the count on main; `fixed=False` keeps the earlier count,
  `Nat.clog 2 (args.length − 1)` index bits a range gather, under which a one-word column cost 0.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.evaluation.reference import Trace, evaluate_region
from verity.ml.circuits.program import REGISTRATION, RUN, Program, input_prim
from verity.ml.circuits.types import Array, Value

WIDTH = 64
WORD = Value(WIDTH)
_MASK = (1 << WIDTH) - 1

S32_MIN, S32_MAX = -(2 ** 31), 2 ** 31 - 1

# An s8-family operand's format as its range, `Int8Fmt` (Model.lean); a toy base 2^b's digit format is (0, 2^b − 1).
S8 = (-128, 127)
U8 = (0, 255)

Iv = tuple[int, int]


def enc(z: int) -> int:
    """An integer's `Value<64>` encoding."""
    return z & _MASK


def signed(u: int) -> int:
    """A `Value<64>` encoding read as two's complement."""
    return u - (1 << WIDTH) if u >> (WIDTH - 1) else u


def clog2(n: int) -> int:
    """`Nat.clog 2 n`: the least `k` with `n ≤ 2^k` (`clog2 0 = clog2 1 = 0`)."""
    return 0 if n <= 1 else (n - 1).bit_length()


def clamp_s32(z: int) -> int:
    return max(S32_MIN, min(S32_MAX, z))


def wrap_s32(z: int) -> int:
    return (z - S32_MIN) % 2 ** 32 + S32_MIN


def gather_at(low: int | None, z: int) -> int:
    """`gatherAt`: the column position a gather reads at index `z`, `z.toNat` or a table read's `(z mod 2^n).toNat`."""
    return max(z, 0) if low is None else z % 2 ** low


# ---- the operations: one unregistered primitive per distinct `Op` ---------------------------------------------------------

_OPS: dict[Hashable, PrimitiveDefinition] = {}


def _op(key: Hashable, name: str, arity: int, kind: str, raw: Callable[[list[int]], int], **attrs: Any) -> PrimitiveDefinition:
    p = _OPS.get(key)
    if p is not None:
        return p
    sat = attrs.get("sat", True)

    def evaluate(*v: int) -> int:
        z = raw([signed(u) for u in v])
        if kind == "mma":
            z = clamp_s32(z) if sat else wrap_s32(z)
        return enc(z)

    p = PrimitiveDefinition(name, 1, tuple((f"a{i}", WORD) for i in range(arity)), WORD, evaluate, register=False,
                            doc=f"Model.lean's `Op.{kind}` on Value<{WIDTH}> words read signed")
    p.kind, p.raw = kind, raw  # type: ignore[attr-defined]
    for k, v in attrs.items():
        setattr(p, k, v)
    if kind != "budgeted":
        _OPS[key] = p
    return p


def op_const(c: int) -> PrimitiveDefinition:
    """`Op.const c`."""
    return _op(("const", c), f"GateConst[{c}]", 0, "const", lambda v: c, c=c)


def op_affine(c: int, coeff: tuple[int, ...]) -> PrimitiveDefinition:
    """`Op.affine c coeff`: `c + Σ coeff·v`."""
    coeff = tuple(coeff)
    return _op(("affine", c, coeff), f"GateAffine[{c};{','.join(map(str, coeff))}]", len(coeff), "affine",
               lambda v: c + sum(cf * x for cf, x in zip(coeff, v)), c=c, coeff=coeff)


def op_mma(fa: Iv, fb: Iv, k: int, sat: bool = True, arity: int | None = None) -> PrimitiveDefinition:
    """`Op.mma fa fb k sat`: `C + Σ_{i<k} a_i·b_i` over operands `(C, a_0 … a_{k−1}, b_0 … b_{k−1})`, clamped (`satfinite`)
    or wrapped to s32. `arity` is its operand count, `2k + 1` unless a test asks for another."""
    n = 2 * k + 1 if arity is None else arity

    def raw(v: list[int]) -> int:
        tail = v[1:]
        return (v[0] if v else 0) + sum(a * b for a, b in zip(tail[:k], tail[k:]))

    return _op(("mma", fa, fb, k, sat, n), f"GateMma[{fa[0]}..{fa[1]}x{fb[0]}..{fb[1]};{k};{'sat' if sat else 'wrap'};{n}]",
               n, "mma", raw, fa=fa, fb=fb, k=k, sat=sat)


def op_gather(arity: int, low: int | None = None) -> PrimitiveDefinition:
    """`Op.gather low`: operands `(index, column…)`, `column[gatherAt low index]`, or 0 past the column."""

    def raw(v: list[int]) -> int:
        col = v[1:]
        t = gather_at(low, v[0] if v else 0)
        return col[t] if t < len(col) else 0

    return _op(("gather", arity, low), f"GateGather[{arity};{'range' if low is None else low}]", arity, "gather", raw, low=low)


_BUDGETED = [0]


def op_budgeted(f: Callable[[list[int]], int], arity: int, cls: str = "rounding") -> PrimitiveDefinition:
    """`Op.budgeted cls f`: any function of its operands. Each is its own primitive, numbered in the order they are made."""
    _BUDGETED[0] += 1
    return _op(None, f"GateBudgeted[{cls};{_BUDGETED[0]}]", arity, "budgeted", lambda v: int(f(v)), cls=cls, f=f)


def is_derived(kind: str) -> bool:
    """`Op.IsDerived`: a const, affine map, s8-family step or gather, whose interval analysis derives."""
    return kind in ("const", "affine", "mma", "gather")


def is_charged(kind: str, binding: str | None) -> bool:
    """`Op.charged`: a run input, a gather or a budgeted gate."""
    return kind in ("gather", "budgeted") or (kind == "input" and binding == "run")


# ---- the circuit -----------------------------------------------------------------------------------------------------------


class Builder:
    """Appends the model's gates in canonical order, and makes the canonical Program they form.

    Gate `i` is `prims[i]` over the gates `args[i]`, declared `[lo[i], hi[i]]`; an input's slot and binding are `slots[i]`
    and `bindings[i]`. The helpers (`const`, `affine`, `mma`, `gather`) give a free gate its derived interval as `Proven`
    requires, and the Builder keeps the derived intervals (`ivs`), the `dynamic` flags and the units, the data products and
    the gather dedup as it goes, so a large circuit builds and measures in linear time."""

    def __init__(self) -> None:
        self.prims: list[PrimitiveDefinition] = []
        self.kinds: list[str] = []
        self.args: list[tuple[int, ...]] = []
        self.lo: list[int] = []
        self.hi: list[int] = []
        self.slots: dict[int, Hashable] = {}
        self.bindings: dict[int, str] = {}
        self.ivs: list[Iv] = []
        self.dyn: list[bool] = []
        self._lean: list[int] = []
        self._fixed: list[int] = []
        self._products = 0
        self._gmax: dict[tuple[int, bool], int] = {}  # (index gate, kind) -> the most indexBits an earlier gather paid
        self._gmaxf: dict[tuple[int, bool], int] = {}
        self._consts: dict[int, int] = {}
        self._program: tuple[int, Program, list[int], Any] | None = None

    def __len__(self) -> int:
        return len(self.prims)

    def gate(self, prim: PrimitiveDefinition, args: tuple[int, ...] | list[int], lo: int | None = None,
             hi: int | None = None) -> int:
        """Append `prim` over `args`, declared `[lo, hi]`; a derived gate left undeclared takes its derived interval."""
        args = tuple(args)
        if len(args) != len(prim.params):
            raise ValueError(f"{prim.id} takes {len(prim.params)} operands, not {len(args)}")
        if any(not 0 <= j < len(self.prims) for j in args):
            raise ValueError(f"{prim.id}: an operand names no earlier gate ({args})")
        kind = prim.kind  # type: ignore[attr-defined]
        i = len(self.prims)
        if not is_derived(kind) and (lo is None or hi is None):
            raise ValueError(f"{prim.id} is not derived: it needs a declared interval")
        d = derive_op(prim, [self.ivs[j] for j in args], lo, hi)  # type: ignore[arg-type]
        if lo is None or hi is None:
            lo, hi = d
        self.prims.append(prim)
        self.kinds.append(kind)
        self.args.append(args)
        self.lo.append(int(lo))
        self.hi.append(int(hi))
        self.ivs.append(d)
        self.dyn.append(any(self.dyn[j] for j in args))
        self._count(i)
        return i

    def inp(self, slot: Hashable, lo: int, hi: int, binding: str = "coin") -> int:
        """An input gate of slot `slot` and binding `registered`, `coin` or `run`, declared `[lo, hi]`."""
        if binding not in ("registered", "coin", "run"):
            raise ValueError(f"an input is registered, coin or run, not {binding!r}")
        i = len(self.prims)
        self.prims.append(input_prim(WIDTH, REGISTRATION if binding == "registered" else RUN))
        self.kinds.append("input")
        self.args.append(())
        self.lo.append(int(lo))
        self.hi.append(int(hi))
        self.slots[i], self.bindings[i] = slot, binding
        self.ivs.append((int(lo), int(hi)))
        self.dyn.append(binding != "registered")
        self._count(i)
        return i

    def const(self, c: int) -> int:
        if c not in self._consts:
            self._consts[c] = self.gate(op_const(c), ())
        return self._consts[c]

    def _s32(self, prim: PrimitiveDefinition, args: tuple[int, ...]) -> int:
        """Append a derived gate at its derived interval, refusing it (and appending nothing) outside s32."""
        lo, hi = derive_op(prim, [self.ivs[j] for j in args], 0, 0)
        assert S32_MIN <= lo and hi <= S32_MAX, f"{prim.kind} [{lo}, {hi}] outside s32"  # type: ignore[attr-defined]
        return self.gate(prim, args, lo, hi)

    def affine(self, c: int, terms: list[tuple[int, int]]) -> int:
        """`c + Σ coeff·gate`, its declared interval the derived one (so `Proven`); asserts it within s32."""
        return self._s32(op_affine(c, tuple(cf for cf, _ in terms)), tuple(j for _, j in terms))

    def mma(self, head: int, a: list[int], b: list[int], fa: Iv, fb: Iv, sat: bool = True) -> int:
        """`head + Σ a_i·b_i`, its declared interval the derived one; asserts within s32."""
        assert len(a) == len(b)
        return self._s32(op_mma(fa, fb, len(a), sat), (head, *a, *b))

    def gather(self, idx: int, column: list[int], lo: int | None = None, hi: int | None = None,
               low: int | None = None) -> int:
        """`column[gatherAt low idx]`, declared `[lo, hi]`, or by default its derived hull (so `Proven`)."""
        return self.gate(op_gather(1 + len(column), low), (idx, *column), lo, hi)

    def budgeted(self, f: Callable[[list[int]], int], args: list[int], lo: int, hi: int, cls: str = "rounding") -> int:
        return self.gate(op_budgeted(f, len(args), cls), tuple(args), lo, hi)

    def bits(self, i: int) -> int:
        """`Gate.bits = Nat.clog 2 size`, `size` the values its declared interval holds."""
        return clog2(max(0, self.hi[i] + 1 - self.lo[i]))

    def index_bits(self, i: int, fixed: bool = False) -> int:
        """`Gate.indexBits` of gather `i`: `n` for a table read of the index's low `n` bits, else `Nat.clog 2 args.length`,
        `⌈log₂(V + 1)⌉` for a column of `V` words (the `V` words and the out-of-range 0). `fixed=False` is the earlier
        count, `Nat.clog 2 (args.length − 1)`, under which a one-word column cost 0."""
        low = self.prims[i].low  # type: ignore[attr-defined]
        if low is not None:
            return low
        n = len(self.args[i])
        return clog2(n) if fixed else clog2(n - 1)

    def _count(self, i: int) -> None:
        kind, args = self.kinds[i], self.args[i]
        lean = fixed = 0
        if kind == "budgeted":
            lean = fixed = 1 if any(self.dyn[j] for j in args) else 0
        elif kind == "gather":
            j = args[0]
            if self.dyn[j]:
                key = (j, self.prims[i].low is not None)  # type: ignore[attr-defined]
                col_dyn = any(self.dyn[a] for a in args[1:])
                ib, ibf = self.index_bits(i, False), self.index_bits(i, True)
                lean = max(0, ib - self._gmax.get(key, 0)) + (1 if col_dyn else 0)
                fixed = max(0, ibf - self._gmaxf.get(key, 0)) + (1 if col_dyn else 0)
                self._gmax[key] = max(self._gmax.get(key, 0), ib)
                self._gmaxf[key] = max(self._gmaxf.get(key, 0), ibf)
        elif kind == "mma":
            self._products += data_products(self, i, dyn=self.dyn)
        self._lean.append(lean)
        self._fixed.append(fixed)

    # -- measures, from the running counts --
    @property
    def lean_units(self) -> int:
        """The units under the count before the gather fix."""
        return sum(self._lean)

    @property
    def fixed_units(self) -> int:
        """The units on main."""
        return sum(self._fixed)

    @property
    def products(self) -> int:
        return self._products

    @property
    def nonlinear_units(self) -> int:
        """The nonlinear units: the budgeted units on main and the data products."""
        return self.fixed_units + self._products

    # -- the canonical circuit --
    def program(self) -> tuple[Program, list[int]]:
        """The Program of the gates so far and, for each gate, its index in the Program's circuit. The root's parameters
        are the registered inputs, then the coin and run inputs, each in the order they were built; its body is the other
        gates in the order they were built, and it returns the last gate."""
        n = len(self.prims)
        if self._program is not None and self._program[0] == n:
            return self._program[1], self._program[2]
        if not n:
            raise ValueError("an empty circuit has no Program")
        groups = [("registered", [i for i in self.slots if self.bindings[i] == "registered"]),
                  ("run", [i for i in self.slots if self.bindings[i] != "registered"])]
        groups = [(name, g) for name, g in groups if g]
        prims, kinds, args = self.prims[:n], self.kinds[:n], self.args[:n]

        def signature(S: Any):
            return tuple((name, Array(len(g), WORD)) for name, g in groups), WORD

        def body(B: Any, S: Any, *colls: Any):
            ref: list[Any] = [None] * n
            for (_name, g), coll in zip(groups, colls):
                for j, i in enumerate(g):
                    ref[i] = coll[j]
            for i in range(n):
                if kinds[i] != "input":
                    ref[i] = B.call(prims[i], *[ref[j] for j in args[i]])
            return ref[n - 1]

        fn = bind(CompositeDefinition("GateCircuit", 1, (), signature, body, register=False))
        prog = Program(fn, registered=tuple(name for name, _ in groups if name == "registered"))
        index = [0] * n
        g = 0
        for _name, grp in groups:
            for i in grp:
                index[i] = g
                g += 1
        for i in range(n):
            if kinds[i] != "input":
                index[i] = g
                g += 1
        assert g == prog.gates
        circuit = prog.circuit
        self._program = (n, prog, index, list(circuit[0:circuit.size]))  # the gates, resolved once for every run
        return prog, index

    def trace(self, x: dict[Hashable, int]) -> Trace:
        """The canonical evaluator's trace on input `x` (slot -> integer): every gate's encoding, by Program gate index."""
        prog, index = self.program()
        assert self._program is not None
        t: Trace = {index[i]: enc(x[slot]) for i, slot in self.slots.items()}
        return evaluate_region(prog.circuit, self._program[3], t)

    def values(self, x: dict[Hashable, int]) -> list[int]:
        """`values`: each gate's value on `x`, read off the canonical evaluator's trace."""
        _prog, index = self.program()
        t = self.trace(x)
        return [signed(t[g]) for g in index]

    def evaluate(self, x: dict[Hashable, int]) -> tuple[list[int], bool, bool]:
        """Run on input `x`: the gates' values, and whether the window (`InWindow`) and `DeclaredHold` hold."""
        vs = self.values(x)
        return vs, in_window(self, vs), declared_hold(self, vs)


# ---- the window (Model.lean's `Ok`, `InWindow`, `DeclaredHold`) over a run's values ------------------------------------


def raw(C: Builder, i: int, vs: list[int]) -> int:
    """`Gate.raw`: gate `i`'s exact value before an s8-family step's clamp or wrap, from its operands' values."""
    if C.kinds[i] != "mma":
        return vs[i]
    return C.prims[i].raw([vs[j] for j in C.args[i]])  # type: ignore[attr-defined]


def operands_ok(C: Builder, i: int, vs: list[int]) -> bool:
    """`Op.OperandsOk`: an s8-family step's `2k + 1` operands lie in their formats; other ops ask nothing."""
    if C.kinds[i] != "mma":
        return True
    p, o = C.prims[i], [vs[j] for j in C.args[i]]
    k = p.k  # type: ignore[attr-defined]
    if len(o) != 2 * k + 1:
        return False
    fa, fb = p.fa, p.fb  # type: ignore[attr-defined]
    return all(fa[0] <= z <= fa[1] for z in o[1:1 + k]) and all(fb[0] <= z <= fb[1] for z in o[1 + k:])


def ok(C: Builder, i: int, vs: list[int]) -> bool:
    """`Gate.Ok`: the exact value in its declared interval, an s8-family step's operands in their formats and its
    declared interval within s32."""
    r = raw(C, i, vs)
    if not (C.lo[i] <= r <= C.hi[i] and operands_ok(C, i, vs)):
        return False
    return C.kinds[i] != "mma" or (S32_MIN <= C.lo[i] and C.hi[i] <= S32_MAX)


def in_window(C: Builder, vs: list[int]) -> bool:
    """`InWindow`: every gate `Ok` on the run whose values are `vs`."""
    return all(ok(C, i, vs) for i in range(len(vs)))


def declared_hold(C: Builder, vs: list[int]) -> bool:
    """`DeclaredHold`: every non-derived gate's raw value lies in its declared interval."""
    return all(C.lo[i] <= vs[i] <= C.hi[i] for i in range(len(vs)) if not is_derived(C.kinds[i]))


# ---- interval analysis (Model.lean's `derive`, `intervals`, `Proven`) --------------------------------------------------


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


def derive(C: Builder, i: int, ivs: list[Iv]) -> Iv:
    """`Gate.derive` of gate `i` from the earlier gates' intervals `ivs`."""
    return derive_op(C.prims[i], [ivs[j] for j in C.args[i]], C.lo[i], C.hi[i])


def derive_op(p: PrimitiveDefinition, a: list[Iv], lo: int, hi: int) -> Iv:
    """`Gate.derive` of a gate `p` over operands of intervals `a`, declared `[lo, hi]`: a const's value, an affine map's or
    an s8-family step's interval image, a gather's hull of its column's intervals and 0, and every other gate's declared
    interval."""
    kind = getattr(p, "kind", "input")
    if kind == "const":
        return (p.c, p.c)  # type: ignore[attr-defined]
    if kind == "affine":
        acc = (p.c, p.c)  # type: ignore[attr-defined]
        for cf, iv in zip(p.coeff, a):  # type: ignore[attr-defined]
            acc = iv_add(acc, iv_scale(cf, iv))
        return acc
    if kind == "mma":
        k = p.k  # type: ignore[attr-defined]
        acc = a[0] if a else (0, 0)
        tail = a[1:]
        for x, y in zip(tail[:k], tail[k:]):
            acc = iv_add(acc, iv_mul(x, y))
        return acc
    if kind == "gather":
        acc = (0, 0)
        for iv in a[1:]:
            acc = iv_join(acc, iv)
        return acc
    return (lo, hi)


def intervals(C: Builder) -> list[Iv]:
    ivs: list[Iv] = []
    for i in range(len(C)):
        ivs.append(derive(C, i, ivs))
    return ivs


def proven(C: Builder) -> list[str]:
    """`Proven`: each derived gate's (a gather's among them) derived interval within its declared one, each affine map's
    declared interval within s32, and each s8-family step's `2k + 1` operands within their formats and its declared
    interval within s32. Returns the violations, so `[]` means proven."""
    errs: list[str] = []
    ivs: list[Iv] = []
    for i in range(len(C)):
        kind, lo, hi, args = C.kinds[i], C.lo[i], C.hi[i], C.args[i]
        d = derive(C, i, ivs)
        if is_derived(kind) and not iv_within(d, lo, hi):
            errs.append(f"gate {i} ({kind}): derived {d} outside declared [{lo}, {hi}]")
        if kind == "affine" and not (S32_MIN <= lo and hi <= S32_MAX):
            errs.append(f"gate {i} (affine): declared [{lo}, {hi}] outside s32")
        if kind == "mma":
            p = C.prims[i]
            k, fa, fb = p.k, p.fa, p.fb  # type: ignore[attr-defined]
            if len(args) != 2 * k + 1:
                errs.append(f"gate {i} (mma): {len(args)} operands, not 2k+1={2 * k + 1}")
            if not (S32_MIN <= lo and hi <= S32_MAX):
                errs.append(f"gate {i} (mma): declared [{lo}, {hi}] outside s32")
            for j in args[1:1 + k]:
                if not iv_within(ivs[j], fa[0], fa[1]):
                    errs.append(f"gate {i} (mma): operand {j} {ivs[j]} outside format {fa}")
            for j in args[1 + k:1 + 2 * k]:
                if not iv_within(ivs[j], fb[0], fb[1]):
                    errs.append(f"gate {i} (mma): operand {j} {ivs[j]} outside format {fb}")
        ivs.append(d)
    return errs


def operand_bits(B: int, C: Builder) -> bool:
    """`OperandBits B`: every budgeted gate's operands' declared bits sum to at most `B` (a gather is exempt)."""
    return all(sum(C.bits(j) for j in C.args[i]) <= B for i in range(len(C)) if C.kinds[i] == "budgeted")


def restricted(B: int, C: Builder) -> bool:
    """`Restricted B = Proven ∧ OperandBits B`."""
    return not proven(C) and operand_bits(B, C)


def dynamic(C: Builder) -> list[bool]:
    """`dynamic`: whether each gate's value depends on a run or coin input."""
    ds: list[bool] = []
    for i in range(len(C)):
        if C.kinds[i] == "input":
            ds.append(C.bindings[i] != "registered")
        else:
            ds.append(any(ds[j] for j in C.args[i]))
    return ds


def units(C: Builder, i: int, *, fixed: bool = False, dyn: list[bool] | None = None) -> int:
    """`units`: a budgeted gate is one unit when it reads a dynamic value; a gather is its fresh index bits (its
    `indexBits` less the most any earlier gather of the same index gate and kind paid) plus one when its column is
    dynamic too."""
    if dyn is None:
        dyn = dynamic(C)
    kind, args = C.kinds[i], C.args[i]
    if kind == "budgeted":
        return 1 if any(dyn[j] for j in args) else 0
    if kind == "gather":
        j = args[0]
        if not dyn[j]:
            return 0
        table = C.prims[i].low is not None  # type: ignore[attr-defined]
        prior = max((C.index_bits(h, fixed) for h in range(i)
                     if C.kinds[h] == "gather" and (C.prims[h].low is not None) == table  # type: ignore[attr-defined]
                     and C.args[h][0] == j), default=0)
        col_dyn = any(dyn[a] for a in args[1:])
        return max(0, C.index_bits(i, fixed) - prior) + (1 if col_dyn else 0)
    return 0


def budgeted(C: Builder, *, fixed: bool = False) -> int:
    """`budgeted`: the circuit's total units."""
    dyn = dynamic(C)
    return sum(units(C, i, fixed=fixed, dyn=dyn) for i in range(len(C)))


def data_products(C: Builder, i: int, *, dyn: list[bool] | None = None) -> int:
    """`dataProducts`: the lanes of an s8-family step whose two multiplicands both depend on a run or coin input, one
    unit each whatever their range. A lane with a constant or registered multiplicand is a linear map, and free."""
    if C.kinds[i] != "mma":
        return 0
    dyn = dynamic(C) if dyn is None else dyn
    k, args = C.prims[i].k, C.args[i]  # type: ignore[attr-defined]
    return sum(1 for a, b in zip(args[1:1 + k], args[1 + k:1 + 2 * k]) if dyn[a] and dyn[b])


def products(C: Builder) -> int:
    """`products`: the circuit's data products."""
    dyn = dynamic(C)
    return sum(data_products(C, i, dyn=dyn) for i in range(len(C)))


def nonlinear(C: Builder) -> int:
    """`nonlinear`: the budgeted units (the count on main, `fixed=True`) and the data products, what the property's caps
    bound: every operation but a linear map with coefficients fixed before the coins."""
    return budgeted(C, fixed=True) + products(C)
