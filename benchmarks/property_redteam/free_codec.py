"""The whole codec word at 0 budgeted units, as Model.lean once counted: `EncodeChainHardness` false for every `E > 0`.

`squaring.free_square` shows a squaring mod p1024 at 0 units through the one-word gather (`gather [idx, const c]` is
`c·[idx ≤ 0]`, and `Gate.indexBits = Nat.clog 2 (args.length − 1) = 0` for it). This module completes the argument at the
fields 6b's `codecRows` (`Security/Proofs/Accounting/Gates.lean`) use, `⟨8, 5⟩` and `⟨16, 17⟩`: a circuit of the model
that computes `byte (lessParity (encodeWord F.p d.key (d.tweak F i) m)) l` for every `l`, on every draw, from the
`CodecInputs` slots (payload bytes registered, round constants', mask's and tweak's bytes coin), with its window
`Proven`, no budgeted gate at all (so `OperandBits B` for every `B`), no run input, and `budgeted C = 0`. Then
`words.length · E · (n − 1) ≤ budgeted C · (n − 1) + runBits C · E` reads `E · (n − 1) ≤ 0`, false for every `E ≥ 1`
and every `ε < 1`.

Every gadget is a free comparator or a free affine/s8 step: a bit `[x ≥ 2^k]` (`squaring.free_bit`), so bit
decompositions; AND as a u8 s8-family product of two bits; XOR and selections as affine sums re-narrowed to a bit by
`[x ≥ 1]`; `[x ≥ y]` as `gather [y − x, const 1]`; `x mod p` by binary long division (`[V ≥ 2^k·p]`, subtract); and the
product `x·y mod p` from byte limbs (s8-family columns, the `2^n ≡ c` fold) then long division. At `n ≤ 16` an element
and a folded product fit s32, which is what keeps this module small; at p1024 the same gadgets run on 128 limbs
(`free_square`), so the encode is free there too, only too large to build here (≈3 × 1,040 products of ≈26k gates).

Under the fix (`Nat.clog 2 args.length`, property/v1's `ceil(log2(V + 1))`), every comparator is a unit, and the same
circuit costs its comparator count (`Builder.fixed_units`). In nonlinear units each AND of two run-time bits is also a
data product, a unit (`Builder.nonlinear_units`).
"""

from __future__ import annotations

from squaring import U8, Builder, free_bit


def rho_ref(p: int, a: int) -> int:
    """Codec.lean's `rho`."""
    s = pow(a, (p + 1) // 4, p)
    if s % 2 == (0 if s * s % p == a else 1):
        return s
    return 0 if s == 0 else p - s


def encode_word_ref(p: int, cs: list[int], mask: int, t: int, m: int) -> int:
    """Codec.lean's `encodeWord p ⟨cs, mask⟩ t m`."""
    x = m
    for r, c in enumerate(cs):
        if r:
            x = x ^ mask if x ^ mask < p else x
        x = rho_ref(p, (x + c + t) % p)
    return x


class FreeField:
    """Free arithmetic over `p = 2^n − c` (`n ≤ 16`) on a `Builder`: an element is one gate whose value is in `[0, p)`."""

    def __init__(self, b: Builder, n: int, c: int):
        assert n % 8 == 0 and 8 <= n <= 16, "an element and a folded product must fit s32"
        self.b, self.n, self.c, self.p, self.L = b, n, c, (1 << n) - c, n // 8

    # -- bits --
    def bits(self, v: int, nbits: int) -> list[int]:
        """`v`'s low `nbits` bits, LSB first (its value must lie in `[0, 2^nbits)`): MSB-first free comparators."""
        b, out, r = self.b, [0] * nbits, v
        for k in range(nbits - 1, -1, -1):
            out[k] = free_bit(b, r, k)
            r = b.affine(0, [(1, r), (-(2 ** k), out[k])])
        return out

    def value(self, bits: list[int], shift: int = 0) -> int:
        return self.b.affine(0, [(2 ** (k - shift), g) for k, g in enumerate(bits) if k >= shift] or [(0, bits[0])])

    def narrow(self, x: int) -> int:
        """A value that is 0 or 1, re-declared as a bit by the free comparator `[x ≥ 1]`."""
        return free_bit(self.b, x, 0)

    def and_(self, u: int, v: int) -> int:
        return self.b.mma(self.b.const(0), [u], [v], U8, U8)

    def not_(self, u: int) -> int:
        return self.b.affine(1, [(-1, u)])

    def xor(self, u: int, v: int) -> int:
        return self.narrow(self.b.affine(0, [(1, u), (1, v), (-2, self.and_(u, v))]))

    def select(self, cond: int, xs: list[int], ys: list[int]) -> list[int]:
        """`cond ? xs : ys`, bitwise."""
        nc = self.not_(cond)
        return [self.narrow(self.b.affine(0, [(1, self.and_(cond, x)), (1, self.and_(nc, y))])) for x, y in zip(xs, ys)]

    def geq(self, x: int, y: int | None = None, const: int = 0) -> int:
        """`[x ≥ y]` (or `[x ≥ const]`): `gather [y − x, const 1]`."""
        b = self.b
        idx = b.affine(const, [(-1, x)] + ([(1, y)] if y is not None else []))
        return b.gather(idx, [b.const(1)], 0, 1)

    def eq(self, x: int, y: int) -> int:
        return self.and_(self.geq(x, y), self.geq(y, x))

    # -- arithmetic mod p --
    def reduce(self, v: int) -> int:
        """`v mod p` for `v ≥ 0` by binary long division: `[v ≥ 2^k·p]` then subtract, from the top quotient bit down."""
        b, p = self.b, self.p
        q = b.gates[v].hi // p
        for k in range(max(q.bit_length() - 1, 0), -1, -1):
            bit = self.geq(v, const=(2 ** k) * p)
            v = b.affine(0, [(1, v), (-(2 ** k) * p, bit)])
        return v

    def limbs(self, x: int) -> list[int]:
        bits = self.bits(x, self.n)
        return [self.b.affine(0, [(2 ** j, bits[8 * l + j]) for j in range(8)]) for l in range(self.L)]

    def mul(self, x: int, y: int) -> int:
        """`x·y mod p`: byte limbs, s8-family columns, the `2^n ≡ c` fold, then long division."""
        b, L = self.b, self.L
        xl = self.limbs(x)
        yl = xl if y == x else self.limbs(y)
        zero = b.const(0)
        cols = []
        for m in range(2 * L - 1):
            idx = [i for i in range(L) if 0 <= m - i < L]
            cols.append(b.mma(zero, [xl[i] for i in idx], [yl[m - i] for i in idx], U8, U8))
        terms = [(256 ** m, cols[m]) for m in range(L)] + [(self.c * 256 ** (m - L), cols[m]) for m in range(L, 2 * L - 1)]
        return self.reduce(b.affine(0, terms))

    def pow(self, x: int, e: int) -> int:
        r = x
        for bit in bin(e)[3:]:
            r = self.mul(r, r)
            if bit == "1":
                r = self.mul(r, x)
        return r

    def rho(self, a: int) -> list[int]:
        """`rho p a`'s bits: the root `s = a^((p+1)/4)`, kept when its parity is `a`'s non-residuosity, else `p − s`
        (0 at `s = 0`)."""
        p, n = self.p, self.n
        s = self.pow(a, (p + 1) // 4)
        square = self.eq(self.mul(s, s), a)
        sb = self.bits(s, n)
        keep = self.not_(self.xor(sb[0], self.not_(square)))
        nonzero = self.geq(s, const=1)
        neg = [self.and_(nonzero, g) for g in self.bits(self.b.affine(p, [(-1, s)]), n)]
        return self.select(keep, sb, neg)

    def sigma(self, xb: list[int], maskb: list[int]) -> list[int]:
        """`σ_K(x)`'s bits: `x xor K` when below `p`, else `x`."""
        z = [self.xor(x, k) for x, k in zip(xb, maskb)]
        below = self.not_(self.geq(self.value(z), const=self.p))
        return self.select(below, z, xb)


def free_encode_word(n: int, c: int, rounds: int) -> tuple[Builder, list[int], dict]:
    """The circuit computing one word's encoding less its parity at field `⟨n, c⟩` and `rounds` rounds, from the
    `CodecInputs` slots: `("payload", l)` registered, `("const", r, l)`, `("mask", l)` and `("tweak", l)` coin. Returns
    the builder, the `n/8` output byte gates, and the input gates by slot."""
    b = Builder()
    F = FreeField(b, n, c)
    slots: dict = {}

    def bytes_in(kind: tuple, binding: str) -> list[int]:
        gs = [b.inp(kind + (l,), 0, 255, binding=binding) for l in range(F.L)]
        for l, g in enumerate(gs):
            slots[kind + (l,)] = g
        return gs

    def val(gs: list[int]) -> int:
        return b.affine(0, [(256 ** l, g) for l, g in enumerate(gs)])

    x = val(bytes_in(("payload",), "registered"))
    t = val(bytes_in(("tweak",), "coin"))
    maskb = [bit for g in bytes_in(("mask",), "coin") for bit in F.bits(g, 8)]
    xb: list[int] | None = None
    for r in range(rounds):
        if r:
            x = F.value(F.sigma(xb, maskb))
        cr = val(bytes_in(("const", r), "coin"))
        xb = F.rho(F.reduce(b.affine(0, [(1, x), (1, cr), (1, t)])))
        x = F.value(xb)
    out = [b.affine(0, [(2 ** j, xb[8 * l + j]) for j in range(8) if 8 * l + j >= 1]) for l in range(F.L)]
    return b, out, slots


def draw_inputs(n: int, c: int, cs: list[int], mask: int, tau: int, i: int, m: int) -> dict:
    """The slot values of a draw `(cs, mask, τ)` for word `i` with payload `m` (`Slot.value`)."""
    p, L = (1 << n) - c, n // 8
    t = tau * i % p
    x: dict = {}
    for l in range(L):
        x[("payload", l)] = m >> (8 * l) & 255
        x[("tweak", l)] = t >> (8 * l) & 255
        x[("mask", l)] = mask >> (8 * l) & 255
        for r, cr in enumerate(cs):
            x[("const", r, l)] = cr >> (8 * l) & 255
    return x
