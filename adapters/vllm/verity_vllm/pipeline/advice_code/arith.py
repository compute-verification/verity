"""The advice code's coding primitives: bits, Witten–Neal–Cleary's 32-bit integer arithmetic coder, and the registered
distributions it codes integers with, all exact integer arithmetic with no hash.

`BitWriter` and `BitReader` hold bits most significant first. `ArithEncoder` codes one terminated segment
(`ArithEncoder.finish`: the pending bits and two more, a dyadic interval inside the message's final interval, so no
segment is a prefix of another's), and `ArithDecoder`, its mirror, tracks the encoder's interval so it knows where the
segment ends; its `finish` re-encodes the decoded intervals and refuses a segment unless its bits are the encoder's, so
each message has one string. `Table` is a registered distribution over an integer range, `Binned` one over values ≥ 0
as an Exp-Golomb bin (`bin_of`) and the value within it, and `put_uniform` / `get_uniform` a value uniform over a range;
`model_from_json` reads a registered model back. `zigzag` maps the integers onto values ≥ 0.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

__all__ = ["ArithDecoder", "ArithEncoder", "bin_of", "bin_range", "Binned", "BitReader", "BitWriter", "get_uniform",
    "in_range", "MAX_TOTAL", "model_from_json", "put_uniform", "QUARTER", "Table", "THREE_QUARTERS", "uniform_bits",
    "unzigzag", "width", "zigzag"]


class BitWriter:
    """Bits, most significant first, one 0/1 per entry."""

    __slots__ = ("bits",)

    def __init__(self) -> None:
        self.bits = bytearray()

    def bit(self, b: int) -> None:
        self.bits.append(b & 1)

    def uint(self, value: int, width: int) -> None:
        if value < 0 or value >> width:
            raise ValueError(f"{value} does not fit {width} bits")
        for i in range(width - 1, -1, -1):
            self.bits.append((value >> i) & 1)

    def __len__(self) -> int:
        return len(self.bits)

    def to_bytes(self) -> bytes:
        """The bits packed most significant first, the last byte padded with zeros."""
        out = bytearray((len(self.bits) + 7) // 8)
        for i, b in enumerate(self.bits):
            out[i >> 3] |= b << (7 - (i & 7))
        return bytes(out)


class BitReader:
    __slots__ = ("bits", "pos")

    def __init__(self, bits: bytes | bytearray, pos: int = 0) -> None:
        self.bits, self.pos = bits, pos

    @classmethod
    def from_bytes(cls, data: bytes, nbits: int) -> BitReader:
        return cls(bytearray((data[i >> 3] >> (7 - (i & 7))) & 1 for i in range(nbits)))

    def bit(self) -> int:
        if self.pos >= len(self.bits):
            raise ValueError("advice ends inside a codeword")
        self.pos += 1
        return self.bits[self.pos - 1]

    def uint(self, width: int) -> int:
        v = 0
        for _ in range(width):
            v = (v << 1) | self.bit()
        return v

    def at(self, i: int) -> int:
        """The bit at absolute position i, 0 past the end: the arithmetic decoder's lookahead reads past a segment's last
        codeword bit into the next segment or the padding, and any continuation decodes the same symbols."""
        return self.bits[i] if i < len(self.bits) else 0


def zigzag(v: int) -> int:
    return 2 * v if v >= 0 else -2 * v - 1


def unzigzag(z: int) -> int:
    return z // 2 if z % 2 == 0 else -(z + 1) // 2


def width(span: int) -> int:
    """Bits of a fixed-width field with `span` values."""
    return max(span - 1, 0).bit_length()


# --------------------------------------------------------------------------------------------- arithmetic coder

_P = 32
_TOP = (1 << _P) - 1
_HALF = 1 << (_P - 1)
QUARTER = 1 << (_P - 2)
THREE_QUARTERS = 3 * QUARTER
#: The largest frequency total. After renormalization the range exceeds `QUARTER = 2^30`, so every symbol of frequency ≥ 1
#: keeps an interval of at least 2^14 values, and distinct messages keep disjoint intervals.
MAX_TOTAL = 1 << 16


class ArithEncoder:
    """Witten–Neal–Cleary's integer arithmetic coder at 32 bits, one terminated segment.

    `finish` emits two bits (and the pending ones) naming a dyadic interval inside the final interval, `[Q1, HALF)` or
    `[HALF, Q3)`. Distinct messages of one model have disjoint intervals, so their codewords are prefix-free, and any
    continuation of a codeword decodes to the same message. The segment is `shifts + 2` bits, or none when no symbol
    needed coding (its message was the only one possible)."""

    def __init__(self, out: BitWriter) -> None:
        self.out, self.start = out, len(out)
        self.low, self.high, self.pending, self.used = 0, _TOP, 0, False

    def _emit(self, b: int) -> None:
        self.out.bit(b)
        for _ in range(self.pending):
            self.out.bit(1 - b)
        self.pending = 0

    def encode(self, cum: int, freq: int, total: int) -> None:
        if not (0 <= cum and freq >= 1 and cum + freq <= total <= MAX_TOTAL):
            raise ValueError(f"bad interval {cum}+{freq}/{total}")
        if freq == total:  # the only possible symbol: the interval is unchanged
            return
        self.used = True
        r = self.high - self.low + 1
        self.high = self.low + (r * (cum + freq)) // total - 1
        self.low = self.low + (r * cum) // total
        while True:
            if self.high < _HALF:
                self._emit(0)
            elif self.low >= _HALF:
                self._emit(1)
                self.low -= _HALF
                self.high -= _HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTERS:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low, self.high = 2 * self.low, 2 * self.high + 1

    def finish(self) -> int:
        if self.used:
            self.pending += 1
            self._emit(0 if self.low < QUARTER else 1)
        return len(self.out) - self.start


class ArithDecoder:
    """The encoder's mirror. It tracks the encoder's interval, so it knows the segment's length (`shifts + 2`) without
    reading it, and `finish` moves the reader to the next segment.

    Alone it would take 2 to 4 strings for one message: any whose interval meets the final interval, after reading 32 bits
    past the segment (the certifier PROTOCOL's attack 10). So `finish` re-encodes the intervals it consumed and refuses the
    segment unless its bits are the encoder's, bit for bit, and lie inside the input. The segments it accepts are then
    exactly the encoder's, and the bits read past an accepted segment decide nothing."""

    def __init__(self, inp: BitReader) -> None:
        self.inp, self.start = inp, inp.pos
        self.low, self.high, self.shifts, self.used = 0, _TOP, 0, False
        self.log: list[tuple[int, int, int]] = []
        self.value = 0
        for i in range(_P):
            self.value = (self.value << 1) | inp.at(self.start + i)
        self.cursor = self.start + _P

    def target(self, total: int) -> int:
        r = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // r

    def consume(self, cum: int, freq: int, total: int) -> None:
        if freq == total:
            return
        self.used = True
        self.log.append((cum, freq, total))
        r = self.high - self.low + 1
        self.high = self.low + (r * (cum + freq)) // total - 1
        self.low = self.low + (r * cum) // total
        while True:
            if self.high < _HALF:
                pass
            elif self.low >= _HALF:
                self.low -= _HALF
                self.high -= _HALF
                self.value -= _HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTERS:
                self.low -= QUARTER
                self.high -= QUARTER
                self.value -= QUARTER
            else:
                break
            self.low, self.high = 2 * self.low, 2 * self.high + 1
            self.value = 2 * self.value + self.inp.at(self.cursor)
            self.cursor += 1
            self.shifts += 1

    def finish(self) -> int:
        n = self.shifts + 2 if self.used else 0
        if self.start + n > len(self.inp.bits):
            raise ValueError("advice ends inside an arithmetic segment")
        w = BitWriter()
        enc = ArithEncoder(w)
        for t in self.log:
            enc.encode(*t)
        enc.finish()
        if w.bits != self.inp.bits[self.start:self.start + n]:
            raise ValueError("an arithmetic segment that isn't the encoder's codeword for what it decodes to")
        self.inp.pos = self.start + n
        return n


_CHUNK = 15


def put_uniform(enc: ArithEncoder, v: int, n: int) -> None:
    """v uniform over [0, n): one symbol when n ≤ 2^16, else 15 low bits under the high part, at most a bit over log2 n."""
    if not 0 <= v < n:
        raise ValueError(f"{v} outside [0, {n})")
    if n <= MAX_TOTAL:
        if n > 1:
            enc.encode(v, 1, n)
        return
    put_uniform(enc, v >> _CHUNK, ((n - 1) >> _CHUNK) + 1)
    enc.encode(v & ((1 << _CHUNK) - 1), 1, 1 << _CHUNK)


def get_uniform(dec: ArithDecoder, n: int) -> int:
    if n <= MAX_TOTAL:
        if n <= 1:
            return 0
        v = dec.target(n)
        dec.consume(v, 1, n)
        return v
    hi = get_uniform(dec, ((n - 1) >> _CHUNK) + 1)
    lo = dec.target(1 << _CHUNK)
    dec.consume(lo, 1, 1 << _CHUNK)
    return (hi << _CHUNK) | lo


def uniform_bits(n: int) -> float:
    """What `put_uniform` costs over n values, at most."""
    if n <= 1:
        return 0.0
    if n <= MAX_TOTAL:
        return math.log2(n)
    return uniform_bits(((n - 1) >> _CHUNK) + 1) + _CHUNK


# --------------------------------------------------------------------------------------------------------- models


@dataclass(frozen=True)
class Table:
    """A registered distribution over the integers `lo … lo + len(freqs) − 1` (the core), each of frequency ≥ 1, and an
    escape of frequency `escape ≥ 1` for every feasible value outside the core, coded uniform over them. Coding a value
    restricts the table to a feasible interval `[flo, fhi]`: the core outside it and, when nothing feasible lies outside
    the core, the escape drop out, and the rest is renormalized."""

    lo: int
    freqs: tuple[int, ...]
    escape: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "freqs", tuple(int(f) for f in self.freqs))
        if not self.freqs or min(self.freqs) < 1 or self.escape < 1 or sum(self.freqs) + self.escape > MAX_TOTAL:
            raise ValueError("a table's frequencies are ≥ 1 and total at most 2^16 with the escape")
        cum = [0]
        for f in self.freqs:
            cum.append(cum[-1] + f)
        object.__setattr__(self, "_cum", tuple(cum))

    @property
    def hi(self) -> int:
        return self.lo + len(self.freqs) - 1

    def _layout(self, flo: int, fhi: int) -> tuple[int, int, int, int, int, int]:
        """The feasible core `[a, b]`, the feasible values left and right of the core, the core's mass and the total."""
        a, b = max(self.lo, flo), min(self.hi, fhi)
        nl = max(0, min(self.lo - 1, fhi) - flo + 1)
        nr = max(0, fhi - max(self.hi + 1, flo) + 1)
        core = self._cum[b - self.lo + 1] - self._cum[a - self.lo] if a <= b else 0
        return a, b, nl, nr, core, core + (self.escape if nl + nr else 0)

    def put(self, enc: ArithEncoder, v: int, flo: int, fhi: int) -> None:
        if not flo <= v <= fhi:
            raise ValueError(f"{v} outside its feasible range [{flo}, {fhi}]")
        a, b, nl, nr, core, total = self._layout(flo, fhi)
        if a <= v <= b:
            if total > 1:
                enc.encode(self._cum[v - self.lo] - self._cum[a - self.lo], self.freqs[v - self.lo], total)
            return
        if core:
            enc.encode(core, self.escape, total)
        put_uniform(enc, v - flo if v < self.lo else nl + v - max(self.hi + 1, flo), nl + nr)

    def get(self, dec: ArithDecoder, flo: int, fhi: int) -> int:
        a, b, nl, nr, core, total = self._layout(flo, fhi)
        if core:
            if total == 1:
                return a
            base = self._cum[a - self.lo]
            t = dec.target(total)
            if t < core:
                i = bisect.bisect_right(self._cum, base + t) - 1
                dec.consume(self._cum[i] - base, self.freqs[i], total)
                return self.lo + i
            dec.consume(core, self.escape, total)
        u = get_uniform(dec, nl + nr)
        return flo + u if u < nl else max(self.hi + 1, flo) + u - nl

    def cost(self, v: int, flo: int, fhi: int) -> float:
        """The model's bits for v, before the coder's rounding."""
        a, b, nl, nr, core, total = self._layout(flo, fhi)
        if a <= v <= b:
            return math.log2(total / self.freqs[v - self.lo])
        return (math.log2(total / self.escape) if core else 0.0) + uniform_bits(nl + nr)

    def worst(self, flo: int, fhi: int) -> float:
        """The most bits one value of `[flo, fhi]` costs, before the coder's rounding."""
        a, b, nl, nr, core, total = self._layout(flo, fhi)
        out = math.log2(total / min(self.freqs[a - self.lo:b - self.lo + 1])) if a <= b else 0.0
        if nl + nr:
            out = max(out, (math.log2(total / self.escape) if core else 0.0) + uniform_bits(nl + nr))
        return out

    def to_json(self) -> dict:
        return {"lo": self.lo, "freqs": list(self.freqs), "escape": self.escape}


def bin_of(v: int) -> int:
    """The Exp-Golomb bin of v ≥ 0: bin 0 is {0}, bin j ≥ 1 is [2^(j−1), 2^j)."""
    return v.bit_length()


def bin_range(j: int) -> tuple[int, int]:
    return (0, 0) if j == 0 else (1 << (j - 1), (1 << j) - 1)


@dataclass(frozen=True)
class Binned:
    """A registered distribution over values ≥ 0: an Exp-Golomb bin from `bins` (a Table over bin indices), then the
    offset within the bin, uniform over its feasible part."""

    bins: Table

    def _part(self, j: int, flo: int, fhi: int) -> tuple[int, int]:
        a, b = bin_range(j)
        return max(a, flo), min(b, fhi)

    def put(self, enc: ArithEncoder, v: int, flo: int, fhi: int) -> None:
        if not 0 <= flo <= v <= fhi:
            raise ValueError(f"{v} outside its feasible range [{flo}, {fhi}]")
        j = bin_of(v)
        self.bins.put(enc, j, bin_of(flo), bin_of(fhi))
        a, b = self._part(j, flo, fhi)
        put_uniform(enc, v - a, b - a + 1)

    def get(self, dec: ArithDecoder, flo: int, fhi: int) -> int:
        j = self.bins.get(dec, bin_of(flo), bin_of(fhi))
        a, b = self._part(j, flo, fhi)
        return a + get_uniform(dec, b - a + 1)

    def cost(self, v: int, flo: int, fhi: int) -> float:
        j = bin_of(v)
        a, b = self._part(j, flo, fhi)
        return self.bins.cost(j, bin_of(flo), bin_of(fhi)) + uniform_bits(b - a + 1)

    def worst(self, flo: int, fhi: int) -> float:
        jl, jh = bin_of(flo), bin_of(fhi)
        return max(self.bins.cost(j, jl, jh) + uniform_bits(self._part(j, flo, fhi)[1] - self._part(j, flo, fhi)[0] + 1)
                   for j in range(jl, jh + 1))

    def to_json(self) -> dict:
        return {"binned": self.bins.to_json()}


def model_from_json(d: dict) -> Table | Binned:
    if "binned" in d:
        b = d["binned"]
        return Binned(Table(b["lo"], tuple(b["freqs"]), b["escape"]))
    return Table(d["lo"], tuple(d["freqs"]), d["escape"])


def in_range(name: str, v: int, lo: int, hi: int) -> int:
    if v is None or not lo <= v <= hi:
        raise ValueError(f"{name} {v} outside its feasible range [{lo}, {hi}]")
    return int(v)
