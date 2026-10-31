"""Any number of codec words for 8 units a key byte, as Model.lean counts on main: `EncodeChainHardness` is false for
every `E ≥ 1`.

`free_codec.py`'s word was free only under the earlier count: its XORs and long division narrow through one-word
gathers, a unit each under the fix. This module needs no narrowing at all. Every intermediate value is a bit, and the
model keeps a bit a bit for free:

* AND is an s8-family step `0 + u·v` with u8 operands (`mma`, `k = 1`): its derived interval is `[0,1]·[0,1] = [0, 1]`;
* NOT is the affine map `1 − u`, derived `[0, 1]`;

so OR, XOR and selection (De Morgan over those two) stay in `[0, 1]` too, `Proven` holds with every declared interval
`[0, 1]`, and a Boolean circuit of any size costs 0 units (`units` charges a budgeted gate or a gather, and these are
neither). The draw's bits enter through table reads: eight `gather (some 8) 0` over constant 0/1 columns (bit `b` of
the index) at one coin byte pay 8 fresh index bits for the first and 0 for the rest (same index gate, same kind). A
payload byte is registered, so its table reads pay nothing.

The circuit computes the codec with the field's arithmetic built from those gates (ripple-carry adders, the
conditional subtraction of `p`, double-and-add products, square-and-multiply for `ρ`'s exponent), and every word's tweak
from word 1's, `t_i = i·t_1 mod p` (`Draw.tweak`: `t_i = τ·i mod p`, so `t_1 = τ`). It reads the coin slots
`(const, r, l)`, `(mask, l)` and `(tweak, 1, l)` once: `8·(k + 2)·n/8` units for the whole block, whatever its number
of words. At p1024 and `k = 3` that is 5,120 units, so `EncodeChainHardness p1024 3 B E ε`'s conclusion
`q·E·(n − 1) ≤ budgeted·(n − 1)` fails at every `q > 5120/E`, for every `B` (there is no budgeted gate) and every
`ε < 1` (it is right on every draw). The stacking property then capped no MAC (`Property` capped rounding,
compare-select, gather and fold units; property/v1's census reads its MACs as informational), so these circuits also
satisfied it at any caps that allowed the key's gather units.

The cost has moved to MACs: one a gate-level AND. A field product here is ≈ 17n² of them (measured: 36,214 MACs a
word at n = 8 and 336,688 at n = 16, three rounds), so at p1024 a word, ≈ 3 · 2,040 products (`(p+1)/4`'s 1,021
squarings and 1,018 multiplications a root), is ≈ 1.1·10^11 MACs (modeled from those two fields).

**Nonlinear units.** The model's only free operations are now linear maps whose coefficients are fixed before the
coins, so every product of two values that both depend on run or coin data is a data product, one unit whatever its
operands' ranges (`dataProducts`, `nonlinear`), and `Property` caps the data products per certified byte read. Every
AND here on two run-time bits is a data product, so the block costs its key's 40 units and 36,152 data products a word
at n = 8 (`Builder.nonlinear_units`), and a squaring ≈ 17n² of them; `EncodeChainHardness` is replaced by
`SquaringFloor` and `ChainSum`, priced in nonlinear units (`test_bitwise_codec.py`).
"""

from __future__ import annotations

from squaring import U8, Builder


class Bits:
    """Boolean gates on a `Builder`, each value a gate in `{0, 1}` with declared interval `[0, 1]` (or a constant),
    folding constants and sharing repeated ANDs and NOTs."""

    def __init__(self, b: Builder):
        self.b = b
        self.zero, self.one = b.const(0), b.const(1)
        self._and: dict[tuple[int, int], int] = {}
        self._not: dict[int, int] = {self.zero: self.one, self.one: self.zero}

    def const(self, v: int) -> int:
        return self.one if v else self.zero

    def and_(self, u: int, v: int) -> int:
        if u == self.zero or v == self.zero:
            return self.zero
        if u == self.one or u == v:
            return v
        if v == self.one:
            return u
        if self._not.get(u) == v:
            return self.zero
        key = (min(u, v), max(u, v))
        if key not in self._and:
            self._and[key] = self.b.mma(self.zero, [u], [v], U8, U8)
        return self._and[key]

    def not_(self, u: int) -> int:
        if u not in self._not:
            g = self.b.affine(1, [(-1, u)])
            self._not[u], self._not[g] = g, u
        return self._not[u]

    def or_(self, u: int, v: int) -> int:
        return self.not_(self.and_(self.not_(u), self.not_(v)))

    def xor(self, u: int, v: int) -> int:
        return self.and_(self.or_(u, v), self.not_(self.and_(u, v)))

    def mux(self, c: int, a: int, b: int) -> int:
        """`c ? a : b`."""
        if a == b:
            return a
        return self.or_(self.and_(c, a), self.and_(self.not_(c), b))

    def all_(self, us: list[int]) -> int:
        while len(us) > 1:
            us = [self.and_(us[i], us[i + 1]) if i + 1 < len(us) else us[i] for i in range(0, len(us), 2)]
        return us[0] if us else self.one

    def byte_bits(self, g: int) -> list[int]:
        """A byte gate's eight bits, LSB first: table reads of its low 8 bits over constant 0/1 columns, one index gate
        and kind, so they pay 8 fresh index bits together if `g` is dynamic and nothing if it isn't."""
        return [self.b.gather(g, [self.const(v >> j & 1) for v in range(256)], low=8) for j in range(8)]

    def value(self, bits: list[int], shift: int = 0) -> int:
        """The integer `Σ_{j ≥ shift} 2^j·bit_j`, one affine gate (within s32 for the bytes it is used on)."""
        terms = [(1 << j, g) for j, g in enumerate(bits) if j >= shift]
        return self.b.affine(0, terms or [(0, self.zero)])


class Field:
    """Arithmetic mod `p = 2^n − c` on `n`-bit little-endian vectors of `Bits`."""

    def __init__(self, bits: Bits, n: int, c: int):
        assert 0 < c < 1 << (n - 1)
        self.B, self.n, self.p = bits, n, (1 << n) - c

    def consts(self, v: int, width: int) -> list[int]:
        return [self.B.const(v >> j & 1) for j in range(width)]

    def add(self, xs: list[int], ys: list[int], carry: int | None = None) -> list[int]:
        """`xs + ys (+ carry)`, one bit wider than the wider operand."""
        B = self.B
        w = max(len(xs), len(ys))
        xs, ys = xs + [B.zero] * (w - len(xs)), ys + [B.zero] * (w - len(ys))
        c, out = B.zero if carry is None else carry, []
        for x, y in zip(xs, ys):
            h = B.xor(x, y)
            out.append(B.xor(h, c))
            c = B.or_(B.and_(x, y), B.and_(c, h))
        return out + [c]

    def geq_sub(self, xs: list[int], v: int) -> tuple[list[int], int]:
        """`xs − v` (mod `2^len`) and `[xs ≥ v]`, as `xs + ¬v + 1` and its carry."""
        s = self.add(xs, [self.B.not_(g) for g in self.consts(v, len(xs))], self.B.one)
        return s[:-1], s[-1]

    def reduce(self, xs: list[int]) -> list[int]:
        """`xs mod p` for `xs < 2p`: subtract `p` when `xs ≥ p`."""
        d, ge = self.geq_sub(xs, self.p)
        return [self.B.mux(ge, a, b) for a, b in zip(d[:self.n], xs[:self.n])]

    def add_mod(self, xs: list[int], ys: list[int]) -> list[int]:
        return self.reduce(self.add(xs, ys))

    def mul(self, xs: list[int], ys: list[int]) -> list[int]:
        """`xs·ys mod p`, double-and-add from `ys`'s top bit."""
        B, r = self.B, [self.B.zero] * self.n
        for y in reversed(ys):
            r = self.reduce([B.zero] + r)
            r = self.add_mod(r, [B.and_(y, x) for x in xs])
        return r

    def pow(self, xs: list[int], e: int) -> list[int]:
        r = xs
        for bit in bin(e)[3:]:
            r = self.mul(r, r)
            if bit == "1":
                r = self.mul(r, xs)
        return r

    def eq(self, xs: list[int], ys: list[int]) -> int:
        return self.B.all_([self.B.not_(self.B.xor(x, y)) for x, y in zip(xs, ys)])

    def rho(self, a: list[int]) -> list[int]:
        """`Codec.rho`: `s = a^((p+1)/4)`, kept when its parity is `a`'s non-residuosity, else `p − s` (0 at `s = 0`)."""
        B, p = self.B, self.p
        s = self.pow(a, (p + 1) // 4)
        keep = B.xor(s[0], self.eq(self.mul(s, s), a))
        nonzero = B.not_(self.eq(s, self.consts(0, self.n)))
        neg = self.add(self.consts(p, self.n), [B.not_(g) for g in s], B.one)[:self.n]
        return [B.mux(keep, x, B.and_(nonzero, y)) for x, y in zip(s, neg)]

    def sigma(self, xs: list[int], mask: list[int]) -> list[int]:
        """`σ_K(x)`: `x xor K` when that is below `p`, else `x`."""
        z = [self.B.xor(x, k) for x, k in zip(xs, mask)]
        _, ge = self.geq_sub(z, self.p)
        return [self.B.mux(ge, x, y) for x, y in zip(xs, z)]


def encode_block(n: int, c: int, rounds: int, words: list[int]) -> tuple[Builder, dict[int, list[int]], dict]:
    """The circuit computing words `words`' encodings less parity at field `⟨n, c⟩` and `rounds` rounds, from the
    `CodecInputs` slots: `("payload", i, l)` registered, `("const", r, l)`, `("mask", l)` and `("tweak", 1, l)` coin
    (word 1's tweak only; word `i`'s is `i·t_1 mod p`). Returns the builder, each word's `n/8` output byte gates, and
    the input gates by slot."""
    b = Builder()
    B = Bits(b)
    F = Field(B, n, c)
    L = n // 8
    slots: dict = {}

    def read(slot: tuple, binding: str) -> list[int]:
        out = []
        for l in range(L):
            g = b.inp(slot + (l,), 0, 255, binding=binding)
            slots[slot + (l,)] = g
            out += B.byte_bits(g)
        return out

    cs = [read(("const", r), "coin") for r in range(rounds)]
    mask = read(("mask",), "coin")
    t1 = read(("tweak", 1), "coin")
    outs = {}
    for i in words:
        t = F.mul(t1, F.consts(i % F.p, n))
        x = F.reduce(read(("payload", i), "registered") + [B.zero])
        for r in range(rounds):
            if r:
                x = F.sigma(x, mask)
            x = F.rho(F.add_mod(F.add_mod(x, cs[r]), t))
        outs[i] = [B.value(x[8 * l:8 * l + 8], 1 if l == 0 else 0) for l in range(L)]
    return b, outs, slots


def draw_inputs(n: int, c: int, cs: list[int], mask: int, tau: int, payload: dict[int, int]) -> dict:
    """The slot values of a draw `(cs, mask, τ)` with payload words `payload` (`Slot.value`)."""
    p, L = (1 << n) - c, n // 8
    x: dict = {}
    for l in range(L):
        x[("mask", l)] = mask >> (8 * l) & 255
        x[("tweak", 1, l)] = tau % p >> (8 * l) & 255
        for r, cr in enumerate(cs):
            x[("const", r, l)] = cr >> (8 * l) & 255
        for i, m in payload.items():
            x[("payload", i, l)] = m >> (8 * l) & 255
    return x
