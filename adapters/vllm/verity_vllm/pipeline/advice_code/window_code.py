"""The advice code's window coder: the residual's coder, canonical, prefix-free and self-delimiting per window.

The advice code (`codec`) charges nothing for what is certified (the network certifier's admission order, the request
records, the release buckets), replayed (the scheduler replay, `replay`) or derived (`derived`'s rules), and sends only the
residual, one `Window` (the certifier's `window_charge.Window`) per sampling block, each field's values in the
order its reader asks for them:
- `arrived`: the replay's decisions, one per open step: whether the next request had arrived (`codec.Decision`);
- `hits`: each admitting row's prefix-cache hit in hit units, from the step where the no-eviction bound fails
  (`derived.Residual.hits`);
- `resumes`: which later rows resume their request, by position among those rows, increasing over the run
  (`Residual.resumes`);
- `order`: a step's new requests' certified ingress indices where the walk's prediction misses, steps increasing over
  the run (`Residual.order`);
- `lag`: async mode's bit per request (`Residual.lag`).

Which window a value is charged to is the caller's, by I1's rule: no later than the first window whose rows it can change
(the certifier PROTOCOL's A5).

**The code.** Each window's part is one segment of `arith`'s 32-bit arithmetic coder: the window's count of each field,
then the field's values, under the registered `Book`. Each count and integer is coded with an Exp-Golomb bin table
(`arith.Binned`), and a field that increases over the run is coded as its gap. Each bit is coded with a public adaptive
Krichevsky–Trofimov estimate per context (`KT`): counts start from the book's prior, are updated as the bits are decoded,
and are halved past 2^16, so the model is public. `Encoder` and `Decoder` take the same symbols online, for a reader that
codes with contexts of its own (the replay's egress contexts). `encode` and `decode` drive them over `Window`s, where a
window's `arrived` bits take as context the run of decisions since the last arrival.

**Prefix-free, canonical, self-delimiting.** The sampled leak bound's count (`hAs`: at most 2^c advice values charged c
or less) holds when the charge is at least the length of a prefix-free encoding.
- A segment's symbols are fixed by its counts, so the decoder knows where the segment's message ends. The coder's
  termination (`arith.ArithEncoder.finish`: the pending bits and two more) names a dyadic interval inside the message's
  final interval. Distinct messages have disjoint final intervals, so no encoder output is a prefix of another's, and
  Kraft's sum over them is at most 1. A run's string is its windows' segments in order, each prefix-free given the earlier
  ones, so the whole string is too.
- The WNC decoder alone takes 2 to 4 strings for one message (the certifier PROTOCOL's attack 10): any whose interval
  meets the final interval, after reading 32 bits past the segment. `arith.ArithDecoder.finish` makes it canonical. It
  re-encodes each segment's decoded intervals and refuses the segment unless the bits are the encoder's, bit for bit.
  `Decoder` also refuses:
  - a decoded value outside its range (the uniform coder's top chunk can name one);
  - a segment that runs past the end of the string;
  - bits after the last window, and nonzero padding in the last byte.
- So `decode` accepts exactly `encode`'s outputs, a prefix-free set. Each window's part ends where its decoder says, and
  the bits read past it decide nothing in an accepted string.
- Delimiting costs the termination's 2 bits per window that codes anything (`Encoded.lens`). A window whose message is
  the only one possible costs nothing.
- A window may carry a tail after its segment: plain fixed-width fields whose widths the reader knows before it reads
  them (`encode`'s ``tails``, `decode_tailed`), charged to the window. Every value of a width is a distinct string of
  it, so the string stays prefix-free; a window with no tail codes as before, bit for bit.
- A window may carry a second coded segment after its tail (`Encoder.segment`; `encode`'s and `decode_tailed`'s
  ``coded``): symbols a reader codes with contexts of its own (`kept`), under the same adaptive estimates, whose number
  the reader knows from what it has decoded. It ends as the first does, so the string stays prefix-free and canonical,
  and its bits, the termination's 2 among them, are charged to the window; a window with none codes as before, bit
  for bit.

**The registered codebook.** `Book` holds the caps and every model, and is fit on historical traces, never on the
run: a codebook fit to the run would be advice. Its digest (`Book.digest`, SHA-512 of its canonical JSON) is
registered with the codec, `compile` and the cap c before the audit's first window (the certifier PROTOCOL's A7). The
decoder reads the book that digest names. It is integer only and uses no hash.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

from verity.protocols.certifier.prover.window_charge import Window
from verity_vllm.pipeline.advice_code import arith as C

__all__ = ["Book", "Caps", "code_window", "CoderState", "context_key", "decode", "decode_tailed", "Decoder", "DEFAULT_CAPS",
    "encode", "Encoded", "Encoder", "ints", "KT", "pack", "unpack"]


@dataclass(frozen=True)
class Caps:
    """The registered caps: each field's count a window, and each value's range."""

    arrived: int = 1 << 16          # decisions a window
    hits: int = 1 << 12             # hits a window
    hit_units: int = 1 << 16        # a hit, in hit units (max_model_len over the unit)
    resumes: int = 1 << 12          # resumed rows a window
    order: int = 1 << 10            # order entries a window
    order_len: int = 1 << 10        # new requests in one step (max_num_seqs)
    gap: int = 1 << 24              # a row position's or a step's gap from the previous one
    index: int = 1 << 24            # a certified ingress index
    lag: int = 1 << 12              # lag bits a window


DEFAULT_CAPS = Caps()

def ints(caps: Caps) -> dict[str, tuple[int, int]]:
    """Every integer symbol and its range."""
    return {"n_arrived": (0, caps.arrived), "n_hits": (0, caps.hits), "hit": (0, caps.hit_units),
            "n_resumes": (0, caps.resumes), "resume_gap": (0, caps.gap), "n_order": (0, caps.order),
            "order_gap": (0, caps.gap), "order_len": (1, caps.order_len), "order_index": (0, 2 * caps.index + 1),
            "n_lag": (0, caps.lag)}


@dataclass(frozen=True)
class Book:
    """The registered codebook: the caps, an Exp-Golomb bin table per integer symbol, and a Krichevsky–Trofimov prior
    `(n0, n1)` per bit symbol and context (`"name/ctx"`), all fixed before the run."""

    caps: Caps
    models: dict[str, Any]
    priors: dict[str, tuple[int, int]]

    def __post_init__(self) -> None:
        missing = set(ints(self.caps)) - set(self.models)
        if missing:
            raise ValueError(f"the book has no model for {sorted(missing)}")
        for k, (n0, n1) in self.priors.items():
            if n0 < 0 or n1 < 0 or 2 * (n0 + n1) + 2 > C.MAX_TOTAL:
                raise ValueError(f"prior {k} {(n0, n1)}: counts ≥ 0 whose estimate totals at most 2^16")

    @classmethod
    def uniform(cls, caps: Caps = DEFAULT_CAPS) -> Book:
        """Every bin alike and no prior: the book before any history."""
        return cls(caps, {n: C.Binned(C.Table(0, (1,) * (C.bin_of(hi) + 1))) for n, (_, hi) in ints(caps).items()}, {})

    def to_json(self) -> dict:
        return {"caps": asdict(self.caps), "models": {k: v.to_json() for k, v in sorted(self.models.items())},
                "priors": {k: list(v) for k, v in sorted(self.priors.items())}}

    @classmethod
    def from_json(cls, d: dict) -> Book:
        return cls(Caps(**d["caps"]), {k: C.model_from_json(v) for k, v in d["models"].items()},
                   {k: (int(v[0]), int(v[1])) for k, v in d["priors"].items()})

    def digest(self) -> str:
        """What the codec's registration holds: SHA-512 of the canonical JSON."""
        return hashlib.sha512(json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class KT:
    """Krichevsky–Trofimov estimates per key, from the book's priors: frequencies `(2·n0 + 1, 2·n1 + 1)`, the counts
    halved (rounding up) whenever their total would pass the coder's 2^16."""

    def __init__(self, priors: dict[str, tuple[int, int]]) -> None:
        self.n = {k: list(v) for k, v in priors.items()}

    def freqs(self, key: str) -> tuple[int, int]:
        n0, n1 = self.n.get(key, (0, 0))
        return 2 * n0 + 1, 2 * n1 + 1

    def update(self, key: str, b: bool) -> None:
        n = self.n.setdefault(key, [0, 0])
        n[int(b)] += 1
        if 2 * (n[0] + n[1]) + 2 > C.MAX_TOTAL:
            n[0], n[1] = (n[0] + 1) // 2, (n[1] + 1) // 2


def context_key(name: str, ctx: Any) -> str:
    return f"{name}/{ctx}"


# ------------------------------------------------------------------------------------------------- bits and bytes


class Encoded(NamedTuple):
    """A run's residual: its bits packed most significant first, the last byte padded with zeros, and each window's
    charge (`lens`, bits, summing to `nbits`)."""

    bits: bytes
    nbits: int
    lens: tuple[int, ...]

    def bitlist(self) -> list[int]:
        return unpack(self.bits, self.nbits)


def pack(bits: Sequence[int]) -> tuple[bytes, int]:
    w = C.BitWriter()
    for b in bits:
        w.bit(b)
    return w.to_bytes(), len(w)


def unpack(data: bytes, nbits: int) -> list[int]:
    return list(C.BitReader.from_bytes(data, nbits).bits)


# ------------------------------------------------------------------------------------------------------- the coder


class Encoder:
    """The residual's symbols, one arithmetic segment per window: `window()` opens the next window's segment (closing the
    last), then `value` and `bit` code its symbols."""

    def __init__(self, book: Book) -> None:
        self.book, self.w, self.kt = book, C.BitWriter(), KT(book.priors)
        self.seg: C.ArithEncoder | None = None
        self.lens: list[int] = []
        self.base = 0               # the open window's bits before its open segment, when that is its second

    def window(self) -> None:
        self._close()
        self.seg = C.ArithEncoder(self.w)

    def segment(self) -> None:
        """Opens the open window's second segment, after its first and its tail: `value` and `bit` code its symbols, and
        its bits are charged to that window."""
        self._close()
        if not self.lens:
            raise ValueError("a second segment outside any window: call window() first")
        self.base, self.seg = self.lens.pop(), C.ArithEncoder(self.w)

    def _close(self) -> None:
        if self.seg is not None:
            self.lens.append(self.base + self.seg.finish())
            self.base, self.seg = 0, None

    def _open(self) -> C.ArithEncoder:
        if self.seg is None:
            raise ValueError("a symbol outside any window: call window() first")
        return self.seg

    def value(self, name: str, v: int, lo: int, hi: int) -> int:
        v = C.in_range(name, v, lo, hi)
        self.book.models[name].put(self._open(), v, lo, hi)
        return v

    def bit(self, name: str, b: bool, ctx: Any = 0) -> bool:
        key, b = context_key(name, ctx), bool(b)
        f0, f1 = self.kt.freqs(key)
        self._open().encode(f0 if b else 0, f1 if b else f0, f0 + f1)
        self.kt.update(key, b)
        return b

    def raw(self, v: int, width: int) -> None:
        """`width` plain bits of v, most significant first, after the open window's segment, charged to that window."""
        self._close()
        if not self.lens:
            raise ValueError("a plain field outside any window: call window() first")
        self.w.uint(v, width)
        self.lens[-1] += width

    def finish(self) -> Encoded:
        self._close()
        return Encoded(self.w.to_bytes(), len(self.w), tuple(self.lens))

    _sym = value
    _flag = bit


class Decoder:
    """The encoder's mirror. Each window's segment ends where its decoder says, and `window()` and `finish()` refuse it
    unless re-encoding what it decoded gives its bits exactly (`code.ArithDecoder.finish`)."""

    def __init__(self, bits: bytes, nbits: int, book: Book) -> None:
        if nbits < 0 or len(bits) != (nbits + 7) // 8:
            raise ValueError(f"{len(bits)} bytes for {nbits} bits")
        if nbits % 8 and bits[-1] & ((1 << (8 - nbits % 8)) - 1):
            raise ValueError("nonzero padding after the last bit")
        self.book, self.kt, self.nbits = book, KT(book.priors), nbits
        self.r = C.BitReader.from_bytes(bits, nbits)
        self.seg: C.ArithDecoder | None = None
        self.lens: list[int] = []
        self.base = 0               # as `Encoder.base`

    def window(self) -> None:
        self._close()
        self.seg = C.ArithDecoder(self.r)

    def segment(self) -> None:
        """`Encoder.segment`'s mirror: the open window's second segment, read from where its tail ends."""
        self._close()
        if not self.lens:
            raise ValueError("a second segment outside any window: call window() first")
        self.base, self.seg = self.lens.pop(), C.ArithDecoder(self.r)

    def _close(self) -> None:
        if self.seg is None:
            return
        try:
            self.lens.append(self.base + self.seg.finish())
        except ValueError as e:
            raise ValueError(f"window {len(self.lens)}: {e}") from None
        self.base, self.seg = 0, None

    def _open(self) -> C.ArithDecoder:
        if self.seg is None:
            raise ValueError("a symbol outside any window: call window() first")
        return self.seg

    def value(self, name: str, lo: int, hi: int) -> int:
        return C.in_range(name, self.book.models[name].get(self._open(), lo, hi), lo, hi)

    def bit(self, name: str, ctx: Any = 0) -> bool:
        key, seg = context_key(name, ctx), self._open()
        f0, f1 = self.kt.freqs(key)
        b = seg.target(f0 + f1) >= f0
        seg.consume(f0 if b else 0, f1 if b else f0, f0 + f1)
        self.kt.update(key, b)
        return b

    def raw(self, width: int) -> int:
        self._close()
        if not self.lens:
            raise ValueError("a plain field outside any window: call window() first")
        v = self.r.uint(width)
        self.lens[-1] += width
        return v

    def finish(self) -> None:
        self._close()
        if self.r.pos != self.nbits:
            raise ValueError(f"{self.nbits - self.r.pos} bits after the last window")

    def _sym(self, name: str, v: int | None, lo: int, hi: int) -> int:
        return self.value(name, lo, hi)

    def _flag(self, name: str, b: bool | None, ctx: Any = 0) -> bool:
        return self.bit(name, ctx)


class CoderState:
    """What the windows carry from one to the next: the run of decisions since the last arrival, and the last resumed
    position, order step and ingress index."""

    def __init__(self) -> None:
        self.run, self.resume, self.step, self.index = 0, -1, -1, -1


def code_window(c: Any, win: Window | None, st: CoderState) -> Window:
    """One window's symbols, written once for every direction: `win` is None when decoding."""
    caps = c.book.caps

    def count(name: str, xs: Sequence) -> int:
        return c._sym(name, None if win is None else len(xs), 0, getattr(caps, name[2:]))

    arrived = []
    for j in range(count("n_arrived", win.arrived if win else ())):
        b = c._flag("arrived", None if win is None else win.arrived[j], min(st.run, 2))
        st.run = 0 if b else st.run + 1
        arrived.append(b)
    hits = [c._sym("hit", None if win is None else win.hits[j], 0, caps.hit_units)
            for j in range(count("n_hits", win.hits if win else ()))]
    resumes = []
    for j in range(count("n_resumes", win.resumes if win else ())):
        st.resume += c._sym("resume_gap", None if win is None else win.resumes[j] - st.resume - 1, 0, caps.gap) + 1
        resumes.append(st.resume)
    order = []
    for j in range(count("n_order", win.order if win else ())):
        step, idx = (None, None) if win is None else win.order[j]
        st.step += c._sym("order_gap", None if win is None else step - st.step - 1, 0, caps.gap) + 1
        got = []
        for m in range(c._sym("order_len", None if win is None else len(idx), 1, caps.order_len)):
            if win is not None and not 0 <= idx[m] <= caps.index:
                raise ValueError(f"order index {idx[m]} outside [0, {caps.index}]")
            z = c._sym("order_index", None if win is None else C.zigzag(idx[m] - st.index - 1), 0, 2 * caps.index + 1)
            i = st.index + 1 + C.unzigzag(z)
            if not 0 <= i <= caps.index:
                raise ValueError(f"order index {i} outside [0, {caps.index}]")
            st.index = i
            got.append(i)
        order.append((st.step, tuple(got)))
    lag = [c._flag("lag", None if win is None else win.lag[j], 0) for j in range(count("n_lag", win.lag if win else ()))]
    return Window(tuple(arrived), tuple(hits), tuple(resumes), tuple(order), tuple(lag))


# ------------------------------------------------------------------------------------------------------------ API


def encode(windows: Sequence[Window], book: Book, tails: Sequence[Sequence[tuple[int, int]]] | None = None,
           coded: Any = None) -> Encoded:
    """The run's residual, window by window; `lens[w]` is window w's charge. `tails[w]`, (value, width) pairs, follow
    window w's segment as plain fixed-width fields (`Encoder.raw`), charged to it. `coded(encoder, w)`, where given,
    then codes window w's second segment (`Encoder.segment`), charged to it too."""
    e, st = Encoder(book), CoderState()
    for w, win in enumerate(windows):
        e.window()
        code_window(e, win, st)
        for v, n in tails[w] if tails else ():
            e.raw(v, n)
        if coded is not None:
            e.segment()
            coded(e, w)
    return e.finish()


def decode(bits: bytes, nbits: int, book: Book, n_windows: int) -> list[Window]:
    """The windows back, refusing (ValueError) any string `encode` wouldn't produce."""
    return decode_tailed(bits, nbits, book, [()] * n_windows)[0]


def decode_tailed(bits: bytes, nbits: int, book: Book, widths: Sequence[Sequence[int]],
                  coded: Any = None) -> tuple[list[Window], list[list[int]]]:
    """`decode` of a string `encode` wrote with tails of these widths, one list a window: the windows and the tails'
    values. `coded(decoder, w)`, where given, then reads window w's second segment, as `encode`'s wrote it."""
    d, st = Decoder(bits, nbits, book), CoderState()
    out, tails = [], []
    for w, ws in enumerate(widths):
        d.window()
        out.append(code_window(d, None, st))
        tails.append([d.raw(n) for n in ws])
        if coded is not None:
            d.segment()
            coded(d, w)
    d.finish()
    return out, tails
