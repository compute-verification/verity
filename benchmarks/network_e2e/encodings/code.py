"""ENC-CODE: the advice coded as tightly as the leak accounting allows.

The advice (I1 v0, `verity/core/protocols/certifier/advice.py` once ADV-SPEC lands it) is everything serving made
unpredictable, and its length is the leak, charged as `A_c` in the certifier's accepted-runs bound
(`SecurityProofs.EncardAccSeqsLeConstantRate`: at most `#Ac · #Σ_sync^W · D^n` observation sequences). This module codes
two subjects, each with four codes, all prefix-free and all fixed before the run by a registered `Codebook`:

- **v0**: I1 v0's fields as drafted (`requests`, `steps` with their rows, `release`), the request ids aside (they are the
  certified ingress's `user` fields);
- **admit**: the residual left once the fields that the certified I/O already determines are dropped (ENC-DERIVED)
  and the scheduler is replayed (ENC-REPLAY): each request's admission step, the engine step before which it joined.

The codes:
- `fixed`: every field at the width of its registered cap (v0's fixed-width code);
- `gamma`: Elias-gamma of each field's natural delta (gaps, positions against history), with no registered model;
- `static`: the same symbols through an arithmetic coder with registered static tables, fit to historical traffic;
- `cond`: an arithmetic coder whose tables are conditioned on public data. For v0, the step's rows are predicted from
  the previous step's (the public schedule: a running request continues, a prefill chunk fills the token budget, new rows
  are first come, first served); for admit, the step count between two arrivals is predicted from their certified
  arrival buckets and a registered step-time model run over the replayed schedule. `cond+adapt` also rescales the
  step-time model online from the decoded past (a public, adaptive model).

**What a window is charged.** Window w's segment codes the requests whose certified arrival bucket is in w and the steps
whose release bucket is in w, conditioned only on earlier segments and public data. Each segment is prefix-free (the
arithmetic coder's segments are terminated), so the concatenation of the first m segments is, and Kraft's inequality
gives `#{advice prefixes whose first m segments total at most L bits} ≤ 2^L`. So either charge is an instance of the
existing bound, with `Ac` the set of codewords it admits:
- the **worst case**: `Ac = {codewords with segment w at most L_w bits, every w}`, `log2 #Ac ≤ Σ L_w`, fixed before the run
  from the caps (`worst_bits`), the audit refusing a longer segment;
- the **realized length**: `Ac_L = {codewords of at most L bits}`, `log2 #Ac_L ≤ L`, for L the run's realized total; an
  accepted run charged L bits is one of at most `2^L · #Σ_sync^W · D^n` sequences. A per-window realized charge `ℓ_w` is
  cumulative, as the timing charge is: the first m windows carry at most `Σ_{w<m} ℓ_w` bits of advice.

**The decoder** (`decode_v0`, `decode_admit`) is a deterministic public function of the bits, the public data
(`Public`) and the codebook, with exact integer arithmetic and no hash. For `admit` it also reads the replayed schedule
of the requests already decoded (`Replay`); `TraceReplay` stands in for ENC-REPLAY's replay by reading a trace's own
rows of those requests.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

FAMILIES = ("fixed", "gamma", "static", "cond", "cond+adapt")

# ------------------------------------------------------------------------------------------------------------- bits


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


# ------------------------------------------------------------------------------------------------- universal codes


def gamma_len(n: int) -> int:
    """Elias-gamma's length for n ≥ 1: `2·⌊log2 n⌋ + 1`."""
    return 2 * n.bit_length() - 1


def put_gamma(w: BitWriter, n: int) -> None:
    if n < 1:
        raise ValueError("Elias-gamma codes n ≥ 1")
    k = n.bit_length()
    w.uint(0, k - 1)
    w.uint(n, k)


def get_gamma(r: BitReader) -> int:
    k = 1
    while r.bit() == 0:
        k += 1
        if k > 64:
            raise ValueError("Elias-gamma prefix past 64 bits")
    return (1 << (k - 1)) | r.uint(k - 1)


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
_Q1 = 1 << (_P - 2)
_Q3 = 3 * _Q1
#: The largest frequency total. After renormalization the range exceeds `_Q1 = 2^30`, so every symbol of frequency ≥ 1
#: keeps an interval of at least 2^14 values, and distinct messages keep disjoint intervals.
MAX_TOTAL = 1 << 16
#: Per symbol, the integer interval loses less than a `MAX_TOTAL / _Q1 = 2^-14` share to rounding.
OVERHEAD_PER_SYMBOL = -math.log2(1 - MAX_TOTAL / _Q1)


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
            elif self.low >= _Q1 and self.high < _Q3:
                self.pending += 1
                self.low -= _Q1
                self.high -= _Q1
            else:
                break
            self.low, self.high = 2 * self.low, 2 * self.high + 1

    def finish(self) -> int:
        if self.used:
            self.pending += 1
            self._emit(0 if self.low < _Q1 else 1)
        return len(self.out) - self.start


class ArithDecoder:
    """The encoder's mirror. It tracks the encoder's interval, so it knows the segment's length (`shifts + 2`) without
    reading it, and `finish` moves the reader to the next segment."""

    def __init__(self, inp: BitReader) -> None:
        self.inp, self.start = inp, inp.pos
        self.low, self.high, self.shifts, self.used = 0, _TOP, 0, False
        self.value = 0
        for i in range(_P):
            self.value = (self.value << 1) | inp.at(self.start + i)
        self.cursor = self.start + _P

    def target(self, total: int) -> int:
        r = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // r

    def consume(self, cum: int, freq: int, total: int) -> None:
        self.used = True
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
            elif self.low >= _Q1 and self.high < _Q3:
                self.low -= _Q1
                self.high -= _Q1
                self.value -= _Q1
            else:
                break
            self.low, self.high = 2 * self.low, 2 * self.high + 1
            self.value = 2 * self.value + self.inp.at(self.cursor)
            self.cursor += 1
            self.shifts += 1

    def finish(self) -> int:
        n = self.shifts + 2 if self.used else 0
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


def fit_table(samples: Sequence[int], lo: int, hi: int, escape_share: float = 2 ** -12, total: int = MAX_TOTAL) -> Table:
    """A table over `[lo, hi]` fit to samples: counts plus one, scaled into `total`, every value at least 1, and the escape
    at least `escape_share` of the total (the uniform floor that bounds a code's worst case)."""
    counts = [1] * (hi - lo + 1)
    out = 0
    for s in samples:
        if lo <= s <= hi:
            counts[s - lo] += 1
        else:
            out += 1
    esc = max(1, round(total * max(escape_share, out / max(len(samples), 1))))
    budget = total - esc
    if budget < len(counts):
        raise ValueError("a table's core is wider than its total")
    n = sum(counts)
    freqs = [max(1, (c * budget) // n) for c in counts]
    while sum(freqs) > budget:
        freqs[max(range(len(freqs)), key=freqs.__getitem__)] -= 1
    return Table(lo, tuple(freqs), esc)


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


# ------------------------------------------------------------------------------------------------------ the advice


@dataclass(frozen=True, slots=True)
class RequestAdvice:
    id: str
    prompt_len: int
    output_len: int
    finish: str  # "eos" | "length"


@dataclass(frozen=True, slots=True)
class Row:
    request: int
    kind: str  # "prefill" | "decode"
    start: int
    n: int
    cache_hit: int = 0
    preempted: bool = False


@dataclass(frozen=True, slots=True)
class StepAdvice:
    rows: tuple[Row, ...]


@dataclass(frozen=True, slots=True)
class Advice:
    """I1 v0 as drafted (the plan, 04:29Z). ADV-SPEC's classes, read by attribute name, work in its place (`as_advice`)."""

    requests: tuple[RequestAdvice, ...]
    steps: tuple[StepAdvice, ...]
    release: tuple[int, ...]


def as_advice(a: Any) -> Advice:
    """This module's `Advice` from any object with I1 v0's attribute names (ADV-SPEC's dataclasses, say)."""
    if isinstance(a, Advice):
        return a
    return Advice(
        tuple(RequestAdvice(str(r.id), int(r.prompt_len), int(r.output_len), str(getattr(r.finish, "value", r.finish)))
              for r in a.requests),
        tuple(StepAdvice(tuple(Row(int(x.request), str(getattr(x.kind, "value", x.kind)), int(x.start), int(x.n),
                                   int(x.cache_hit), bool(x.preempted)) for x in s.rows)) for s in a.steps),
        tuple(int(b) for b in a.release))


@dataclass(frozen=True)
class Public:
    """What the decoder may read besides the advice: each request's certified arrival bucket (its anchor, in request
    order, non-decreasing), the buckets per window T, and the number of windows W the audit covers."""

    anchors: tuple[int, ...]
    T: int
    W: int

    def requests_in(self, w: int) -> range:
        a = bisect.bisect_left(self.anchors, w * self.T)
        return range(a, bisect.bisect_left(self.anchors, (w + 1) * self.T))


@dataclass(frozen=True)
class Caps:
    """The registered caps: the widths of the fixed code, and the ranges every other code is feasible over."""

    prompt_max: int = 1024
    output_max: int = 1024
    requests_max: int = 1 << 20  # requests per run
    seqs_max: int = 256  # rows per step (max_num_seqs)
    tokens_max: int = 8192  # tokens per step (max_num_batched_tokens)
    steps_per_bucket: int = 15  # s_max
    arrivals_per_bucket: int = 34  # r_in: at most one request per ingress frame
    windows_max: int = 1440  # windows per run (a day of one-minute windows)

    def steps_window(self, T: int) -> int:
        return self.steps_per_bucket * T

    def steps_run(self, T: int) -> int:
        return self.steps_per_bucket * T * self.windows_max


@dataclass(frozen=True)
class StepTime:
    """The registered step-time model, in microseconds: `a + b·tokens + c·context/1024`, all integers, and the mean
    delay from a request's delivery to its admission step's start (`phase`)."""

    a: int
    b: int
    c: int
    phase: int = 0

    def us(self, tokens: int, context: int) -> int:
        return self.a + self.b * tokens + (self.c * context) // 1024


@dataclass(frozen=True)
class Codebook:
    """Everything registered before the run: the caps, the bucket length, the step-time model and the tables (by
    prefix and symbol name: `s:` static, `c:` conditional, `a:` conditional with the adaptive step-time model). Its digest
    (SHA-512 of its canonical JSON) is what a deployment registers."""

    caps: Caps
    tau_us: int
    steptime: StepTime
    tables: dict[str, Table | Binned] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"caps": asdict(self.caps), "tau_us": self.tau_us, "steptime": asdict(self.steptime),
                "tables": {k: v.to_json() for k, v in sorted(self.tables.items())}}

    @classmethod
    def from_json(cls, d: dict) -> Codebook:
        return cls(Caps(**d["caps"]), d["tau_us"], StepTime(**d["steptime"]),
                   {k: model_from_json(v) for k, v in d["tables"].items()})

    def digest(self) -> str:
        return hashlib.sha512(json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------------- streams


class Stream:
    """One window's symbols, in either direction: `sym(name, v, lo, hi)` writes v (encoding) or reads and returns a
    value (decoding, v is None), feasible over `[lo, hi]`. Each decomposition below is written once against it, so the
    encoder and the decoder share every rule."""

    decoding = False

    def sym(self, name: str, v: int | None, lo: int, hi: int) -> int:
        raise NotImplementedError

    def finish(self) -> None:
        pass


def _check(name: str, v: int, lo: int, hi: int) -> int:
    if v is None or not lo <= v <= hi:
        raise ValueError(f"{name} {v} outside its feasible range [{lo}, {hi}]")
    return int(v)


class FixedStream(Stream):
    """Every symbol at the width of its range."""

    def __init__(self, w: BitWriter | None = None, r: BitReader | None = None) -> None:
        self.w, self.r, self.decoding = w, r, r is not None

    def sym(self, name: str, v: int | None, lo: int, hi: int) -> int:
        k = width(hi - lo + 1)
        if self.decoding:
            return _check(name, lo + self.r.uint(k), lo, hi)
        self.w.uint(_check(name, v, lo, hi) - lo, k)
        return int(v)


class GammaStream(Stream):
    """Elias-gamma of `v − lo + 1`, or of `zigzag(v) + 1` for a value whose range spans 0 (a signed delta)."""

    def __init__(self, w: BitWriter | None = None, r: BitReader | None = None) -> None:
        self.w, self.r, self.decoding = w, r, r is not None

    def sym(self, name: str, v: int | None, lo: int, hi: int) -> int:
        if self.decoding:
            g = get_gamma(self.r)
            return _check(name, unzigzag(g - 1) if lo < 0 else lo + g - 1, lo, hi)
        v = _check(name, v, lo, hi)
        put_gamma(self.w, zigzag(v) + 1 if lo < 0 else v - lo + 1)
        return v


class ArithStream(Stream):
    """The arithmetic coder with the codebook's table for each symbol name; one terminated segment per window."""

    def __init__(self, tables: dict[str, Table | Binned], w: BitWriter | None = None, r: BitReader | None = None) -> None:
        self.tables, self.decoding = tables, r is not None
        self.enc = ArithEncoder(w) if w is not None else None
        self.dec = ArithDecoder(r) if r is not None else None

    def sym(self, name: str, v: int | None, lo: int, hi: int) -> int:
        t = self.tables.get(name)
        if t is None:
            raise KeyError(f"the codebook has no table {name!r}")
        if self.decoding:
            return t.get(self.dec, lo, hi)
        v = _check(name, v, lo, hi)
        t.put(self.enc, v, lo, hi)
        return v

    def finish(self) -> None:
        self.dec.finish() if self.decoding else self.enc.finish()


class CountStream(Stream):
    """Records each symbol's name, value and feasible range, for fitting tables and bounding the worst case."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, int, int, int]] = []

    def sym(self, name: str, v: int | None, lo: int, hi: int) -> int:
        v = _check(name, v, lo, hi)
        self.seen.append((name, v, lo, hi))
        return v


def _tables_for(cb: Codebook, family: str) -> dict:
    pre = {"static": "s:", "cond": "c:", "cond+adapt": "a:"}[family]
    return {k[len(pre):]: v for k, v in cb.tables.items() if k.startswith(pre)}


def _stream(family: str, cb: Codebook, w: BitWriter | None = None, r: BitReader | None = None) -> Stream:
    if family == "fixed":
        return FixedStream(w, r)
    if family == "gamma":
        return GammaStream(w, r)
    return ArithStream(_tables_for(cb, family), w, r)


# ------------------------------------------------------------------------------------------- v0: the decompositions

PREFILL, DECODE = "prefill", "decode"
FINISH = ("eos", "length")
#: New-row candidates beyond the running requests: the first this many not yet started, in request order.
CANDIDATES = 64


@dataclass
class _V0State:
    """The advice decoded so far, which every later symbol may condition on."""

    requests: list[RequestAdvice] = field(default_factory=list)
    steps: list[StepAdvice] = field(default_factory=list)
    release: list[int] = field(default_factory=list)
    pos: list[int] = field(default_factory=list)  # each request's next position
    started: list[bool] = field(default_factory=list)
    active: set[int] = field(default_factory=set)  # started and not done
    first_unstarted: int = 0
    phase: int = 0  # the release clock: µs from the last release bucket's start to the predicted end of its step

    def done(self, r: int) -> bool:
        q = self.requests[r]
        return self.pos[r] >= q.prompt_len + q.output_len - 1

    def unstarted(self, k: int) -> list[int]:
        while self.first_unstarted < len(self.requests) and self.started[self.first_unstarted]:
            self.first_unstarted += 1
        out, r = [], self.first_unstarted
        while r < len(self.requests) and len(out) < k:
            if not self.started[r]:
                out.append(r)
            r += 1
        return out

    def apply(self, rows: Sequence[Row]) -> None:
        for x in rows:
            self.pos[x.request] = x.start + x.n
            self.started[x.request] = True
            if self.done(x.request):
                self.active.discard(x.request)
            else:
                self.active.add(x.request)


def _requests_window(st: Stream, family: str, caps: Caps, idx: range, adv: Advice | None, s: _V0State) -> None:
    for i in idx:
        q = adv.requests[i] if adv else None
        P = st.sym("prompt_len", q.prompt_len if q else None, 1, caps.prompt_max)
        O = st.sym("output_len", q.output_len if q else None, 1, caps.output_max)
        name = "finish" if family in ("fixed", "gamma", "static") else (
            "finish_at_cap" if O == caps.output_max else "finish_below_cap")
        F = st.sym(name, FINISH.index(q.finish) if q else None, 0, 1)
        s.requests.append(RequestAdvice(q.id if q else f"#{i}", P, O, FINISH[F]))
        s.pos.append(0)
        s.started.append(False)


def _row_raw(st: Stream, caps: Caps, x: Row | None, s: _V0State) -> Row:
    """A row at its fields' registered widths."""
    r = st.sym("request", x.request if x else None, 0, caps.requests_max - 1)
    k = st.sym("kind", int(x.kind == PREFILL) if x else None, 0, 1)
    start = st.sym("start", x.start if x else None, 0, caps.prompt_max + caps.output_max - 1)
    n = st.sym("n", x.n if x else None, 0, caps.tokens_max)
    c = st.sym("cache_hit", x.cache_hit if x else None, 0, caps.prompt_max)
    p = st.sym("preempted", int(x.preempted) if x else None, 0, 1)
    if r >= len(s.requests):
        raise ValueError(f"a row names request {r}, which has not arrived")
    return Row(r, PREFILL if k else DECODE, start, n, c, bool(p))


def _row_delta(st: Stream, caps: Caps, x: Row | None, s: _V0State, pred: int) -> Row:
    """A row as deltas: its request from the previous row's plus one, its start from the request's next position."""
    nreq = len(s.requests)
    pred = min(pred, nreq - 1)
    r = pred + st.sym("request_delta", x.request - pred if x else None, -pred, nreq - 1 - pred)
    k = st.sym("kind", int(x.kind == PREFILL) if x else None, 0, 1)
    p0, span = s.pos[r], caps.prompt_max + caps.output_max - 1
    start = p0 + st.sym("start_delta", x.start - p0 if x else None, -p0, span - p0)
    n = st.sym("n_prefill" if k else "n_decode", x.n if x else None, 0, caps.tokens_max)
    c = st.sym("cache_hit", x.cache_hit if x else None, 0, caps.prompt_max)
    p = st.sym("preempted", int(x.preempted) if x else None, 0, 1)
    return Row(r, PREFILL if k else DECODE, start, n, c, bool(p))


def _rows_delta(st: Stream, caps: Caps, x: StepAdvice | None, s: _V0State) -> tuple[Row, ...]:
    m = st.sym("rows", len(x.rows) if x else None, 0, caps.seqs_max)
    rows, pred = [], s.steps[-1].rows[0].request if s.steps and s.steps[-1].rows else 0
    for i in range(m):
        y = _row_delta(st, caps, x.rows[i] if x else None, s, pred)
        rows.append(y)
        pred = y.request + 1
    return tuple(rows)


def _step_cond(st: Stream, caps: Caps, x: StepAdvice | None, s: _V0State, prev: Sequence[int], waiting: int) -> tuple[Row, ...]:
    """A step predicted from the public schedule: the previous step's unfinished rows continue in order, each at its
    next position, a decode one token and a prefill chunk as much of its prompt as the token budget leaves (a flag says
    whether each is present, and a chunk's shortfall is coded); then new rows, each a pick among the started requests
    not running (in request order) and the first `CANDIDATES` not yet started (first come, first served is pick 0), at
    its next position or, not yet started, at its cache hit. Raises ValueError for a step outside this form."""
    cont = [r for r in prev if not s.done(r)]
    running = set(cont)
    by_req = {y.request: y for y in x.rows} if x else {}
    rows: list[Row] = []
    budget = caps.tokens_max
    for r in cont:
        if budget < 1:
            break
        q = s.requests[r]
        prefill = s.pos[r] < q.prompt_len
        if not st.sym("cont_prefill" if prefill else "cont_decode", int(r in by_req) if x else None, 0, 1):
            continue
        n = 1
        if prefill:
            room = min(q.prompt_len - s.pos[r], budget)
            n = room + st.sym("chunk_cont", by_req[r].n - room if x else None, 1 - room, 0)
        rows.append(Row(r, PREFILL if prefill else DECODE, s.pos[r], n))
        budget -= n
    cands = sorted(s.active - running) + s.unstarted(CANDIDATES)
    new = [y for y in x.rows if y.request not in running] if x else None
    m = st.sym(f"new_rows_{waiting}", len(new) if x else None, 0, min(caps.seqs_max - len(rows), len(cands), budget))
    for j in range(m):
        y = new[j] if x else None
        if budget < 1:
            raise ValueError("new rows past the token budget")
        if y is not None and y.request not in cands:
            raise ValueError("a new row outside the candidates")
        r = cands.pop(st.sym("new_request", cands.index(y.request) if x else None, 0, len(cands) - 1))
        q = s.requests[r]
        c = 0
        if s.started[r]:
            start = s.pos[r]
        else:
            c = st.sym("cache_hit_new", y.cache_hit if x else None, 0, q.prompt_len - 1)
            start = c
        prefill, n = start < q.prompt_len, 1
        if prefill:
            room = min(q.prompt_len - start, budget)
            n = room + st.sym("chunk_new", y.n - room if x else None, 1 - room, 0)
        rows.append(Row(r, PREFILL if prefill else DECODE, start, n, c))
        budget -= n
    if x is not None and tuple(rows) != tuple(x.rows):
        raise ValueError("the step is not in the predicted form")
    return tuple(rows)


def _canonical(x: StepAdvice, s: _V0State, prev: Sequence[int], caps: Caps, waiting: int) -> bool:
    try:
        _step_cond(CountStream(), caps, x, s, prev, waiting)
    except (ValueError, KeyError):
        return False
    return True


#: The release clock's contexts: where a step's predicted end falls in its bucket, in sixteenths, and one for a prediction
#: behind the window's start.
PHASES = 16


def _release_clock(cb: Codebook, pub: Public, s: _V0State, rows: Sequence[Row], idle: bool, prev_rel: int) -> tuple[int, int, int]:
    """The step's predicted release from the public schedule: it starts where the previous step ended (the tracked
    phase `s.phase` in the previous release bucket) or, after an idle engine, at its first request's delivery (the end
    of that request's anchor bucket, phase 0), and runs the registered step time of its rows. Returns the predicted gap
    from `prev_rel`, the context, and the predicted end in µs from the start of bucket `base` with `base`."""
    base, start = (s.release[-1], s.phase) if s.release else (0, 0)
    if idle and rows:
        base, start = max((base, start), (pub.anchors[rows[0].request] + 1, 0))
    end = start + cb.steptime.us(sum(x.n for x in rows), sum(x.start + x.n for x in rows))
    pred = base + end // cb.tau_us
    ctx = PHASES if pred < prev_rel else ((end % cb.tau_us) * PHASES) // cb.tau_us
    return max(pred - prev_rel, 0), ctx, end - (prev_rel - base) * cb.tau_us


def _tokens_bin(rows: Sequence[Row]) -> int:
    t = sum(x.n for x in rows)
    return 0 if t <= 1 else 1 if t <= 8 else 2 if t <= 64 else 3 if t <= 512 else 4


def _v0_window(st: Stream, family: str, cb: Codebook, pub: Public, w: int, adv: Advice | None, s: _V0State,
               steps_idx: range | None) -> None:
    caps, T = cb.caps, pub.T
    _requests_window(st, family, caps, pub.requests_in(w), adv, s)
    nsteps = st.sym("steps", len(steps_idx) if adv else None, 0, caps.steps_window(T))
    for j in range(nsteps):
        k = steps_idx[j] if adv else None
        x = adv.steps[k] if adv else None
        prev_rel = max(s.release[-1], w * T) if s.release else w * T
        if family == "fixed":
            m = st.sym("rows", len(x.rows) if x else None, 0, caps.seqs_max)
            rows = tuple(_row_raw(st, caps, x.rows[i] if x else None, s) for i in range(m))
            rel = w * T + st.sym("release", adv.release[k] - w * T if adv else None, 0, T - 1)
            if rel < prev_rel:
                raise ValueError("releases go back in time")
        elif family in ("gamma", "static"):
            rows = _rows_delta(st, caps, x, s)
            rel = prev_rel + st.sym("release_gap", adv.release[k] - prev_rel if adv else None, 0, (w + 1) * T - 1 - prev_rel)
        else:
            prev = [y.request for y in s.steps[-1].rows] if s.steps else []
            unst = s.unstarted(1)
            idle = not s.active
            # a request is delivered at its anchor bucket's end, so it waits for this step once the previous step ended
            # in a later bucket
            waiting = 2 if idle else int(bool(unst) and bool(s.release) and pub.anchors[unst[0]] < s.release[-1])
            if st.sym("canon", int(_canonical(x, s, prev, caps, waiting)) if x else None, 0, 1):
                rows = _step_cond(st, caps, x, s, prev, waiting)
            else:
                rows = _rows_delta(st, caps, x, s)
            hi = (w + 1) * T - 1 - prev_rel
            ghat, ctx, end = _release_clock(cb, pub, s, rows, idle, prev_rel)
            ghat = min(ghat, hi)
            gap = ghat + st.sym(f"release_res_{ctx}", adv.release[k] - prev_rel - ghat if adv else None, -ghat, hi - ghat)
            rel = prev_rel + gap
            s.phase = min(max(end - gap * cb.tau_us, 0), cb.tau_us - 1)
        if x is not None and rows != tuple(x.rows):
            raise ValueError(f"step {k} does not round-trip under {family}")
        s.steps.append(StepAdvice(rows))
        s.release.append(rel)
        s.apply(rows)


def _steps_by_window(adv: Advice, pub: Public) -> list[range]:
    out, k = [], 0
    for w in range(pub.W):
        a = k
        while k < len(adv.release) and adv.release[k] < (w + 1) * pub.T:
            k += 1
        out.append(range(a, k))
    if k != len(adv.release):
        raise ValueError("a step is released after the last window")
    return out


def _check_public(adv: Advice, pub: Public) -> None:
    if len(pub.anchors) != len(adv.requests) or len(adv.release) != len(adv.steps):
        raise ValueError("one anchor per request and one release per step")
    if any(b < a for a, b in zip(pub.anchors, pub.anchors[1:])) or any(b < a for a, b in zip(adv.release, adv.release[1:])):
        raise ValueError("anchors and releases are non-decreasing")
    if pub.anchors and pub.anchors[-1] >= pub.W * pub.T:
        raise ValueError("a request arrives after the last window")


def _v0_family(family: str) -> str:
    """v0 has no step-time model, so `cond+adapt` codes it as `cond` does."""
    return "cond" if family == "cond+adapt" else family


def encode_v0(adv: Any, pub: Public, cb: Codebook, family: str) -> tuple[BitWriter, list[int]]:
    """The advice's v0 fields under `family`: the bits, and each window's segment length (the bits charged to it)."""
    adv, family = as_advice(adv), _v0_family(family)
    _check_public(adv, pub)
    out, lens, s = BitWriter(), [], _V0State()
    for w, idx in enumerate(_steps_by_window(adv, pub)):
        n0 = len(out)
        st = _stream(family, cb, w=out)
        _v0_window(st, family, cb, pub, w, adv, s, idx)
        st.finish()
        lens.append(len(out) - n0)
    return out, lens


def decode_v0(r: BitReader, pub: Public, cb: Codebook, family: str) -> Advice:
    family = _v0_family(family)
    s = _V0State()
    for w in range(pub.W):
        st = _stream(family, cb, r=r)
        _v0_window(st, family, cb, pub, w, None, s, None)
        st.finish()
    return Advice(tuple(s.requests), tuple(s.steps), tuple(s.release))


def symbols_v0(adv: Any, pub: Public, cb: Codebook, family: str) -> list[list[tuple[str, int, int, int]]]:
    """Each window's symbols (name, value, feasible lo, hi) under `family`'s decomposition."""
    adv, family = as_advice(adv), _v0_family(family)
    s, out = _V0State(), []
    for w, idx in enumerate(_steps_by_window(adv, pub)):
        st = CountStream()
        _v0_window(st, family, cb, pub, w, adv, s, idx)
        out.append(st.seen)
    return out


# --------------------------------------------------------------------------------------- admit: the residual


class Replay(Protocol):
    """The replayed public schedule of the requests already decoded: for request i, the last step at which any request
    before i runs (`end(i)`, −1 for none), and step k's load from those requests (`load(i, k)`: tokens, context)."""

    def end(self, i: int) -> int: ...

    def load(self, i: int, k: int) -> tuple[int, int]: ...


class TraceReplay:
    """ENC-REPLAY's replay, stood in for by a trace's own rows of the requests before i. Exact while a later arrival
    doesn't change an earlier request's rows, which vLLM's scheduler gives (running requests first, waiting ones first
    come, first served) unless a preemption intervenes."""

    def __init__(self, adv: Any) -> None:
        adv = as_advice(adv)
        last: dict[int, int] = {}
        self.rows = [[(x.request, x.n, x.start + x.n) for x in s.rows] for s in adv.steps]
        for k, s in enumerate(adv.steps):
            for x in s.rows:
                last[x.request] = k
        self._end, m = [], -1
        for i in range(len(adv.requests)):
            self._end.append(m)
            m = max(m, last.get(i, -1))

    def end(self, i: int) -> int:
        return self._end[i]

    def load(self, i: int, k: int) -> tuple[int, int]:
        t = c = 0
        for r, n, ctx in self.rows[k]:
            if r < i:
                t += n
                c += ctx
        return t, c


def _gap_bin(g: int) -> int:
    return 0 if g == 0 else 1 if g <= 3 else 2 if g <= 15 else 3 if g <= 63 else 4


ADMIT_CONTEXTS = ("same", "idle", "g0", "g1", "g2", "g3", "g4")


@dataclass
class _Adapt:
    """The public adaptive rescaling of the step-time model: elapsed bucket time over predicted time across the last
    `H` busy gaps, an integer ratio clamped to [1/4, 4]."""

    H: int = 64
    hist: deque = field(default_factory=deque)
    num: int = 0
    den: int = 0

    def scale(self, us: int) -> int:
        if self.den <= 0:
            return us
        return (us * max(min(self.num, 4 * self.den), self.den // 4)) // self.den

    def add(self, elapsed: int, predicted: int) -> None:
        self.hist.append((elapsed, predicted))
        self.num += elapsed
        self.den += predicted
        if len(self.hist) > self.H:
            e, p = self.hist.popleft()
            self.num -= e
            self.den -= p


def predict_gap(cb: Codebook, replay: Replay, i: int, s_prev: int, dbuckets: int, gmax: int,
                adapt: _Adapt | None) -> int:
    """The predicted steps from the previous admission to request i's: the first g whose predicted start,
    `phase + Σ_{j<g} d̂(s_prev + j)`, reaches the delivery `dbuckets · τ` later; at most gmax (an idle engine admits at
    once)."""
    if dbuckets == 0:
        return 0
    target, t, g = dbuckets * cb.tau_us, cb.steptime.phase, 0
    while g < gmax and t < target:
        d = cb.steptime.us(*replay.load(i, s_prev + g))
        t += adapt.scale(d) if adapt else d
        g += 1
    return g


def _admit_window(st: Stream, family: str, cb: Codebook, pub: Public, w: int, admit: Sequence[int] | None,
                  out: list[int], replay: Replay | None, adapt: _Adapt | None) -> None:
    cap = cb.caps.steps_run(pub.T)
    for i in pub.requests_in(w):
        if i == 0:
            out.append(0)  # request 0 joins before step 0
            continue
        s_prev = out[-1]
        v = admit[i] - s_prev if admit is not None else None
        if family == "fixed":
            g = st.sym("admit", v, 0, cap)
        elif family in ("gamma", "static"):
            g = st.sym("admit_gap", v, 0, cap)
        else:
            gmax = max(replay.end(i) + 1 - s_prev, 0)
            db = pub.anchors[i] - pub.anchors[i - 1]
            ghat = predict_gap(cb, replay, i, s_prev, db, gmax, adapt)
            ctx = "same" if db == 0 else "idle" if ghat >= gmax else f"g{_gap_bin(ghat)}"
            g = ghat + st.sym(f"admit_res_{ctx}", v - ghat if v is not None else None, -ghat, gmax - ghat)
            if adapt is not None and db and 0 < g < gmax:
                adapt.add(db * cb.tau_us - cb.steptime.phase,
                          sum(cb.steptime.us(*replay.load(i, s_prev + j)) for j in range(g)))
        out.append(s_prev + g)


def _check_admit(admit: Sequence[int], pub: Public) -> None:
    if len(admit) != len(pub.anchors) or any(b < a for a, b in zip(admit, admit[1:])) or (admit and admit[0] != 0):
        raise ValueError("admission steps are one per request, non-decreasing, from step 0")


def encode_admit(admit: Sequence[int], pub: Public, cb: Codebook, family: str,
                 replay: Replay | None = None) -> tuple[BitWriter, list[int]]:
    """The residual, each request's admission step, under `family`: the bits and each window's segment length."""
    _check_admit(admit, pub)
    out, lens, got = BitWriter(), [], []
    adapt = _Adapt() if family == "cond+adapt" else None
    for w in range(pub.W):
        n0 = len(out)
        st = _stream(family, cb, w=out)
        _admit_window(st, family, cb, pub, w, admit, got, replay, adapt)
        st.finish()
        lens.append(len(out) - n0)
    if got != list(admit):
        raise ValueError(f"the admission steps do not round-trip under {family}")
    return out, lens


def decode_admit(r: BitReader, pub: Public, cb: Codebook, family: str, replay: Replay | None = None) -> list[int]:
    got: list[int] = []
    adapt = _Adapt() if family == "cond+adapt" else None
    for w in range(pub.W):
        st = _stream(family, cb, r=r)
        _admit_window(st, family, cb, pub, w, None, got, replay, adapt)
        st.finish()
    return got


def symbols_admit(admit: Sequence[int], pub: Public, cb: Codebook, family: str,
                  replay: Replay | None = None) -> list[list[tuple[str, int, int, int]]]:
    _check_admit(admit, pub)
    got: list[int] = []
    adapt = _Adapt() if family == "cond+adapt" else None
    out = []
    for w in range(pub.W):
        st = CountStream()
        _admit_window(st, family, cb, pub, w, admit, got, replay, adapt)
        out.append(st.seen)
    return out


# ---------------------------------------------------------------------------------------- schema, fit, worst case


def schema(caps: Caps, T: int) -> dict[str, tuple]:
    """Every table a codebook holds, by prefixed name: `("binned", cap)` for a value from 0 to cap, `("table", lo, hi)`
    for a core over `[lo, hi]` with the escape for the rest."""
    span = caps.prompt_max + caps.output_max - 1
    rows = {"rows": ("binned", caps.seqs_max), "request_delta": ("table", -64, 64), "kind": ("table", 0, 1),
            "start_delta": ("table", -64, 64), "n_prefill": ("binned", caps.tokens_max),
            "n_decode": ("binned", caps.tokens_max), "cache_hit": ("binned", caps.prompt_max),
            "preempted": ("table", 0, 1)}
    reqs = {"prompt_len": ("binned", caps.prompt_max), "output_len": ("binned", caps.output_max),
            "steps": ("binned", caps.steps_window(T))}
    out = {f"s:{k}": v for k, v in {**reqs, **rows, "finish": ("table", 0, 1), "release_gap": ("binned", T - 1),
                                     "admit_gap": ("binned", caps.steps_run(T))}.items()}
    cond = {**reqs, **rows, "finish_at_cap": ("table", 0, 1), "finish_below_cap": ("table", 0, 1),
            "canon": ("table", 0, 1), "cont_prefill": ("table", 0, 1), "cont_decode": ("table", 0, 1),
            "chunk_cont": ("table", -64, 0), "chunk_new": ("table", -64, 0), "new_rows_0": ("table", 0, 16),
            "new_rows_1": ("table", 0, 16), "new_rows_2": ("table", 0, 16), "new_request": ("table", 0, 16),
            "cache_hit_new": ("binned", caps.prompt_max),
            **{f"release_res_{q}": ("table", -8, 8) for q in range(PHASES + 1)}}
    out.update({f"c:{k}": v for k, v in cond.items()})
    for pre in ("c:", "a:"):
        out.update({f"{pre}admit_res_{ctx}": ("table", -64, 64) for ctx in ADMIT_CONTEXTS})
    assert span > 0
    return out


def fit_codebook(caps: Caps, T: int, tau_us: int, steptime: StepTime,
                 symbols: dict[str, Sequence[Sequence[tuple[str, int, int, int]]]], escape_share: float = 2 ** -12,
                 admit_escape_share: float = 2 ** -8) -> Codebook:
    """A codebook fit to historical traffic: `symbols[family]` is that family's windows of symbols on the training
    traces (`symbols_v0`, `symbols_admit`). Every table of the schema is made, a name never seen as near uniform. The
    admission residual's escape is wider (`admit_escape_share`): its uniform floor is what bounds an arrival's worst
    case, at about `log2(1/share)` bits over the fixed code's."""
    pre = {"static": "s:", "cond": "c:", "cond+adapt": "a:"}
    seen: dict[str, list[int]] = {}
    for family, wins in symbols.items():
        for win in wins:
            for name, v, _, _ in win:
                seen.setdefault(pre[family] + name, []).append(v)
    tables: dict[str, Table | Binned] = {}
    for name, spec in schema(caps, T).items():
        vs = seen.get(name, [])
        share = admit_escape_share if "admit" in name else escape_share
        if spec[0] == "binned":
            tables[name] = Binned(fit_table([bin_of(v) for v in vs], 0, bin_of(spec[1]), share))
        else:
            tables[name] = fit_table(vs, spec[1], spec[2], share)
    return Codebook(caps, tau_us, steptime, tables)


def worst_bits(symbols: Sequence[tuple[str, int, int, int]], cb: Codebook, family: str) -> float:
    """An upper bound on a segment's length over every value its symbols could take, each over its feasible range: the
    cap a worst-case charge registers for a window of these symbols (`fixed` and `gamma` exactly; the arithmetic coder's
    within its rounding, plus its two terminating bits)."""
    if family == "fixed":
        return float(sum(width(hi - lo + 1) for _, _, lo, hi in symbols))
    if family == "gamma":
        return float(sum(gamma_len(max(zigzag(lo), zigzag(hi)) + 1) if lo < 0 else gamma_len(hi - lo + 1)
                         for _, _, lo, hi in symbols))
    tables = _tables_for(cb, family)
    return sum(tables[name].worst(lo, hi) for name, _, lo, hi in symbols) + len(symbols) * OVERHEAD_PER_SYMBOL + 2


def model_bits(symbols: Sequence[tuple[str, int, int, int]], cb: Codebook, family: str) -> float:
    """The model's bits for these symbols (the ideal code length the arithmetic coder comes within rounding of)."""
    tables = _tables_for(cb, family)
    return sum(tables[name].cost(v, lo, hi) for name, v, lo, hi in symbols)
