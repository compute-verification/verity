"""[R8 C-codec] Bit-level I/O and integer codes.  All codes are canonical: for a given (phi, model) the decoder reads exactly the
bits the encoder wrote, no padding.  Lengths are reported in bits.

  BitWriter / BitReader     MSB-first bit strings
  fixed(n, k) / gamma(n) / rice(n, k)  integer codes (Elias-gamma codes n>=1; use gamma(n+1) for n>=0)
  ArithEncoder / ArithDecoder          exact arithmetic coder over big-integer fractions with a *static* (or context-static) frequency
                                       model; code length = the shortest dyadic interval inside the final interval
                                       (<= ceil(-log2 P) + 1 bits).  The decoder is told the symbol alphabets by the caller (the
                                       hint grammar is self-delimiting), never by the bit string.
"""
from __future__ import annotations

from fractions import Fraction
from math import log2


class BitWriter:
    def __init__(self) -> None:
        self.bits: list[int] = []
        self.marks: dict[str, int] = {}  # category -> bits attributed

    def write(self, value: int, width: int, cat: str = "other") -> None:
        if width < 0 or (width < 64 and value >= (1 << width)) or value < 0:
            raise ValueError(f"value {value} does not fit in {width} bits")
        for i in range(width - 1, -1, -1):
            self.bits.append((value >> i) & 1)
        self.marks[cat] = self.marks.get(cat, 0) + width

    def write_bits(self, bits: list[int], cat: str = "other") -> None:
        self.bits.extend(bits)
        self.marks[cat] = self.marks.get(cat, 0) + len(bits)

    def gamma(self, n: int, cat: str = "other") -> None:
        """Elias-gamma for n >= 1: (len-1) zeros, then n in binary."""
        if n < 1:
            raise ValueError("gamma needs n >= 1")
        L = n.bit_length()
        self.write(0, L - 1, cat)
        self.write(n, L, cat)

    def rice(self, n: int, k: int, cat: str = "other") -> None:
        """Rice code: unary quotient (q ones + 0), k-bit remainder."""
        q, r = n >> k, n & ((1 << k) - 1)
        self.write((1 << q) - 1, q, cat)
        self.write(0, 1, cat)
        self.write(r, k, cat)

    def __len__(self) -> int:
        return len(self.bits)

    def tobytes(self) -> bytes:
        b = self.bits + [0] * (-len(self.bits) % 8)
        return bytes(int("".join(map(str, b[i:i + 8])), 2) for i in range(0, len(b), 8))


class BitReader:
    def __init__(self, bits: list[int]) -> None:
        self.bits = bits
        self.pos = 0

    def read(self, width: int) -> int:
        if self.pos + width > len(self.bits):
            raise EOFError(f"read past end: pos {self.pos} width {width} len {len(self.bits)}")
        v = 0
        for _ in range(width):
            v = (v << 1) | self.bits[self.pos]
            self.pos += 1
        return v

    def gamma(self) -> int:
        z = 0
        while self.read(1) == 0:
            z += 1
        return (1 << z) | self.read(z)

    def rice(self, k: int) -> int:
        q = 0
        while self.read(1) == 1:
            q += 1
        return (q << k) | self.read(k)

    def done(self) -> bool:
        return self.pos == len(self.bits)


def bits_for(cardinality: int) -> int:
    """Fixed width for a field with `cardinality` admissible values (0 bits when there is only one)."""
    if cardinality <= 1:
        return 0
    return (cardinality - 1).bit_length()


# ---------------------------------------------------------------- exact arithmetic coder (static model)

class ArithEncoder:
    """Interval [lo, lo+width) over Fraction; symbols given as (cum_lo, cum_hi, total) integer frequency intervals."""

    def __init__(self) -> None:
        self.lo = Fraction(0)
        self.width = Fraction(1)
        self.ideal_bits = 0.0
        self.marks: dict[str, float] = {}
        self.n_symbols = 0

    def encode(self, cum_lo: int, cum_hi: int, total: int, cat: str = "other") -> None:
        if not (0 <= cum_lo < cum_hi <= total):
            raise ValueError("bad frequency interval")
        self.lo += self.width * Fraction(cum_lo, total)
        self.width *= Fraction(cum_hi - cum_lo, total)
        b = -log2((cum_hi - cum_lo) / total)
        self.ideal_bits += b
        self.marks[cat] = self.marks.get(cat, 0.0) + b
        self.n_symbols += 1

    def finish(self) -> list[int]:
        """Shortest bit string b (length k) with [b/2^k, (b+1)/2^k) inside [lo, lo+width)."""
        lo, hi = self.lo, self.lo + self.width
        k = 0
        while True:
            scale = 1 << k
            # smallest integer b with b/scale >= lo
            b = -((-lo.numerator * scale) // lo.denominator)  # ceil(lo*scale)
            if Fraction(b + 1, scale) <= hi:
                return [(b >> i) & 1 for i in range(k - 1, -1, -1)]
            k += 1
            if k > 1 << 20:
                raise RuntimeError("arith finish runaway")


class ArithDecoder:
    def __init__(self, bits: list[int]) -> None:
        k = len(bits)
        b = 0
        for x in bits:
            b = (b << 1) | x
        # any point inside the codeword's dyadic interval decodes the same message; take its midpoint
        self.x = Fraction(2 * b + 1, 1 << (k + 1)) if k else Fraction(1, 2)
        self.lo = Fraction(0)
        self.width = Fraction(1)

    def decode(self, cum: list[int], total: int) -> int:
        """cum: cumulative frequencies c_0=0 < ... c_n = total; returns symbol index s with c_s <= target < c_{s+1}."""
        target = (self.x - self.lo) / self.width  # in [0,1)
        # find s: cum[s]/total <= target < cum[s+1]/total
        lo, hi = 0, len(cum) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if Fraction(cum[mid], total) <= target:
                lo = mid
            else:
                hi = mid
        s = lo
        self.lo += self.width * Fraction(cum[s], total)
        self.width *= Fraction(cum[s + 1] - cum[s], total)
        return s

    def decode_uniform(self, card: int) -> int:
        """Uniform symbol in [0, card) (cum[s] = s) without materialising the table: s = floor(target * card)."""
        target = (self.x - self.lo) / self.width
        s = (target.numerator * card) // target.denominator
        s = min(max(int(s), 0), card - 1)
        self.lo += self.width * Fraction(s, card)
        self.width *= Fraction(1, card)
        return s
