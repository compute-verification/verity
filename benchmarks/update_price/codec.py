"""Exact descriptions of an updated tensor given what a decoder holds: the price attacks' codecs.

Every codec here is lossless at the representation's precision: the decoder rebuilds the target's bit patterns from
the bytes and a reference it computes itself (the registered value, or a prediction from it), and `decode(encode(...))`
returns them bit for bit. The length of the bytes is a description length; the decoder's work is a few dozen array
operations per slot.

- `ulp_encode` / `ulp_decode`: the target's ulp distance from a reference (`keys`: the float's order as an integer), each
  distance as a modeled symbol (sign, bit length and the two bits below the leading one) plus its remaining bits raw.
  The symbols go through an interleaved rANS coder (`rans_encode`) with one frequency table per context, the context being
  the reference's exponent and sign within a group (a tensor); contexts with few slots merge (`contexts`), and the decoder
  derives the same merge from the reference alone.
- `lowrank_encode` / `lowrank_decode`: a rank-r integer factorization of the update, its product added to the reference
  in float64 and rounded to the format, and the residual through `ulp_encode`.
- `affine_encode` / `affine_decode`: the reference plus a float32-coefficient combination of float64 bases the decoder
  computes (the registered value, a replayed prefix's change, its Adam direction), and the residual through `ulp_encode`.
- `lattice_encode` / `lattice_decode`: that affine prediction plus a whole number of float32 steps per slot, the step
  counts under the reference's contexts, and the residual through `ulp_encode`: an update that sits near a lattice
  (AdamW's first step moves nearly every weight by ±lr) costs its step counts and not the low bits of each distance.
  `lattice_encode_base` is the same on a base prediction and per-slot steps the decoder already holds (a selection
  across tensors).

numpy and the standard library only.
"""
from __future__ import annotations

import io
import lzma
import struct
from dataclasses import dataclass

import numpy as np

#: storage dtype, sign-bit position and exponent shift of each representation
FORMATS = {"fp32": (np.uint32, 31, 23), "bf16": (np.uint16, 15, 7)}

TOP = 2                       # bits below the leading one that the model codes; the rest go raw
ALPHABET = 1 + 2 * (33 << TOP)  # zero, then (bit length 1..33, top bits, sign)
PROB_BITS = 15
M = 1 << PROB_BITS
L = 1 << 16                   # rANS state lower bound; states live in [L, 2^32), words are 16 bits
LANE_MIN = 1 << 12            # symbols per interleaved lane at least, so a lane's 32-bit final state costs < 0.01 bit/symbol
LANES_MAX = 1 << 16
MIN_CONTEXT = 4096            # a context with fewer slots merges into a coarser one


class CodecError(ValueError):
    """A stream that doesn't decode: truncated, trailing bytes, or a final state that isn't the initial one."""


def check_format(fmt: str) -> tuple:
    if fmt not in FORMATS:
        raise CodecError(f"unknown format {fmt!r}: one of {sorted(FORMATS)}")
    return FORMATS[fmt]


def to_bits(x: np.ndarray, fmt: str) -> np.ndarray:
    """The format's bit patterns of float values: fp32 exactly, bf16 by round-to-nearest-even from float32."""
    dt, _, _ = check_format(fmt)
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    b = x32.view(np.uint32)
    if fmt == "fp32":
        return b.copy()
    return ((b + np.uint32(0x7FFF) + ((b >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(dt)


def from_bits(bits: np.ndarray, fmt: str) -> np.ndarray:
    """float32 values of the format's bit patterns."""
    check_format(fmt)
    b = np.ascontiguousarray(bits).astype(np.uint32)
    return (b if fmt == "fp32" else b << np.uint32(16)).view(np.float32)


def keys(bits: np.ndarray, fmt: str) -> np.ndarray:
    """Each pattern's position in the float order, as int64: +x -> its magnitude bits, -x -> -(magnitude bits) - 1."""
    _, sb, _ = check_format(fmt)
    b = bits.astype(np.int64)
    mag = b & ((1 << sb) - 1)
    return np.where((b >> sb) & 1 == 1, -mag - 1, mag)


def from_keys(k: np.ndarray, fmt: str) -> np.ndarray:
    dt, sb, _ = check_format(fmt)
    k = k.astype(np.int64)
    neg = k < 0
    mag = np.where(neg, -k - 1, k)
    if mag.size and (mag.min() < 0 or mag.max() >= (1 << sb)):
        raise CodecError("a decoded key lies outside the format")
    return (mag | (neg.astype(np.int64) << sb)).astype(dt)


def round_to(x64: np.ndarray, fmt: str) -> np.ndarray:
    """The format's bit patterns nearest float64 values (through float32, a fixed deterministic rounding)."""
    return to_bits(np.asarray(x64, dtype=np.float64).astype(np.float32), fmt)


def bit_length(a: np.ndarray) -> np.ndarray:
    """Bit length of non-negative int64 values below 2^53 (frexp is exact there)."""
    return np.frexp(a.astype(np.float64))[1].astype(np.int64)


def symbolize(d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed integers -> (modeled symbol, raw bits, raw width)."""
    d = d.astype(np.int64)
    a = np.abs(d)
    b = bit_length(a)
    t = np.clip(b - 1, 0, TOP)
    w = np.maximum(b - 1 - t, 0)
    top = (a >> w) & ((np.int64(1) << t) - 1)
    sym = np.where(b == 0, 0, 1 + 2 * (((b - 1) << TOP) + top) + (d < 0))
    raw = a & ((np.int64(1) << w) - 1)
    return sym.astype(np.int64), raw, w


def widths(sym: np.ndarray) -> np.ndarray:
    s = np.maximum(sym - 1, 0) >> 1
    b = (s >> TOP) + 1
    return np.where(sym == 0, 0, np.maximum(b - 1 - np.clip(b - 1, 0, TOP), 0))


def desymbolize(sym: np.ndarray, raw: np.ndarray) -> np.ndarray:
    sym = sym.astype(np.int64)
    s = np.maximum(sym - 1, 0)
    neg = s & 1
    q = s >> 1
    b = (q >> TOP) + 1
    top = q & ((1 << TOP) - 1)
    t = np.clip(b - 1, 0, TOP)
    w = np.maximum(b - 1 - t, 0)
    a = (np.int64(1) << (b - 1)) | (top << w) | raw.astype(np.int64)
    return np.where(sym == 0, 0, np.where(neg == 1, -a, a))


def contexts(ref_bits: np.ndarray, group: np.ndarray, fmt: str, min_count: int = MIN_CONTEXT) -> tuple[np.ndarray, int]:
    """Context of each slot: (group, reference exponent, reference sign), merged to (exponent, sign) and then to one
    global context where fewer than `min_count` slots share it. A function of the reference and groups only."""
    _, sb, es = check_format(fmt)
    r = ref_bits.astype(np.int64)
    e = (r >> es) & 0xFF
    sg = (r >> sb) & 1
    fine = (group.astype(np.int64) * 256 + e) * 2 + sg
    _, inv, cnt = np.unique(fine, return_inverse=True, return_counts=True)
    lvl = np.where(cnt[inv] < min_count, -(1 + e * 2 + sg), fine)
    _, inv, cnt = np.unique(lvl, return_inverse=True, return_counts=True)
    lvl = np.where(cnt[inv] < min_count, -(1 << 20), lvl)
    _, ctx = np.unique(lvl, return_inverse=True)
    ctx = ctx.astype(np.int64).reshape(-1)
    return ctx, int(ctx.max()) + 1 if ctx.size else 0


def quantize(counts: np.ndarray) -> np.ndarray:
    """Frequencies summing to M, at least 1 for every symbol that occurs."""
    counts = counts.astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        raise CodecError("an empty context")
    f = np.where(counts > 0, np.maximum(1, (counts * M) // total), 0)
    diff = M - int(f.sum())
    while diff:
        i = int(np.argmax(f))
        step = diff if diff > 0 else max(diff, 1 - int(f[i]))
        if step == 0:
            raise CodecError("too many symbols for the frequency precision")
        f[i] += step
        diff -= step
    return f


def lanes(n: int) -> int:
    return int(min(LANES_MAX, max(1, n // LANE_MIN)))


def rans_encode(sym: np.ndarray, ctx: np.ndarray, freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interleaved rANS: symbol i in lane i % K. -> (K final states, uint32; the 16-bit words in decoding order)."""
    n = len(sym)
    k = lanes(n)
    f_all = freq.astype(np.uint64)
    c_all = (np.cumsum(freq, axis=1) - freq).astype(np.uint64)
    x = np.full(k, L, dtype=np.uint64)
    chunks = []
    for r in range(-(-n // k) - 1, -1, -1):
        lo = r * k
        hi = min(lo + k, n)
        s, c = sym[lo:hi], ctx[lo:hi]
        f, cm = f_all[c, s], c_all[c, s]
        xs = x[: hi - lo]
        big = xs >= (f << np.uint64(17))
        chunks.append((xs[big] & np.uint64(0xFFFF)).astype(np.uint16))
        xs = np.where(big, xs >> np.uint64(16), xs)
        x[: hi - lo] = ((xs // f) << np.uint64(PROB_BITS)) + (xs % f) + cm
    words = np.concatenate(chunks[::-1]) if chunks else np.zeros(0, np.uint16)
    return x.astype(np.uint32), words


def rans_decode(states: np.ndarray, words: np.ndarray, ctx: np.ndarray, freq: np.ndarray) -> np.ndarray:
    n = len(ctx)
    k = len(states)
    n_ctx, alpha = freq.shape
    f_all = freq.astype(np.uint64)
    cum = np.cumsum(freq, axis=1) - freq
    c_all = cum.astype(np.uint64)
    flat = (np.arange(n_ctx, dtype=np.int64)[:, None] * M + cum).reshape(-1)
    x = states.astype(np.uint64)
    out = np.empty(n, dtype=np.int64)
    pos = 0
    for r in range(-(-n // k) if n else 0):
        lo = r * k
        hi = min(lo + k, n)
        c = ctx[lo:hi]
        xs = x[: hi - lo]
        slot = (xs & np.uint64(M - 1)).astype(np.int64)
        s = np.searchsorted(flat, c * M + slot, side="right") - 1 - c * alpha
        f, cm = f_all[c, s], c_all[c, s]
        xs = f * (xs >> np.uint64(PROB_BITS)) + slot.astype(np.uint64) - cm
        need = np.nonzero(xs < np.uint64(L))[0]
        if pos + len(need) > len(words):
            raise CodecError("the rANS words ran out")
        xs[need] = (xs[need] << np.uint64(16)) | words[pos:pos + len(need)].astype(np.uint64)
        pos += len(need)
        x[: hi - lo] = xs
        out[lo:hi] = s
    if pos != len(words) or (n and not np.all(x == np.uint64(L))):
        raise CodecError("the rANS stream doesn't end in its initial state")
    return out


def pack_raw(raw: np.ndarray, w: np.ndarray) -> bytes:
    """Variable-width raw bits, grouped by width (ascending), positional order within a width, MSB first."""
    parts = []
    for width in np.unique(w[w > 0]):
        v = raw[w == width].astype(np.uint64)
        sh = np.arange(width - 1, -1, -1, dtype=np.uint64)
        for i in range(0, len(v), 1 << 22):
            parts.append(((v[i:i + (1 << 22), None] >> sh) & np.uint64(1)).astype(np.uint8).reshape(-1))
    return np.packbits(np.concatenate(parts)).tobytes() if parts else b""


def unpack_raw(buf: bytes, w: np.ndarray) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8))
    raw = np.zeros(len(w), dtype=np.int64)
    pos = 0
    for width in np.unique(w[w > 0]):
        idx = np.nonzero(w == width)[0]
        sh = np.arange(width - 1, -1, -1, dtype=np.int64)
        for i in range(0, len(idx), 1 << 22):
            j = idx[i:i + (1 << 22)]
            b = bits[pos:pos + len(j) * width]
            if len(b) != len(j) * width:
                raise CodecError("the raw bits ran out")
            raw[j] = (b.reshape(len(j), width).astype(np.int64) << sh).sum(axis=1)
            pos += len(j) * width
    if (len(bits) - pos) >= 8:
        raise CodecError("trailing raw bytes")
    return raw


def _varint(out: io.BytesIO, v: int) -> None:
    if v < 0:
        raise CodecError("varints are non-negative")
    while True:
        b = v & 0x7F
        v >>= 7
        out.write(bytes([b | (0x80 if v else 0)]))
        if not v:
            return


def _read_varint(buf: memoryview, pos: int) -> tuple[int, int]:
    v = shift = 0
    while True:
        if pos >= len(buf):
            raise CodecError("truncated header")
        b = buf[pos]
        pos += 1
        v |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return v, pos


def encode_ints(d: np.ndarray, ctx: np.ndarray, n_ctx: int) -> bytes:
    """Signed integers under per-context tables: header (tables), rANS states and words, raw bits."""
    sym, raw, w = symbolize(d)
    counts = np.bincount(ctx * ALPHABET + sym, minlength=max(n_ctx, 1) * ALPHABET).reshape(-1, ALPHABET)
    out = io.BytesIO()
    _varint(out, len(d))
    freq = np.zeros_like(counts)
    for c in range(n_ctx):
        freq[c] = quantize(counts[c])
        nz = np.nonzero(freq[c])[0]
        _varint(out, len(nz))
        prev = 0
        for s in nz:
            _varint(out, int(s) - prev)
            _varint(out, int(freq[c, s]) - 1)
            prev = int(s)
    states, words = rans_encode(sym, ctx, freq) if len(d) else (np.zeros(0, np.uint32), np.zeros(0, np.uint16))
    _varint(out, len(words))
    out.write(states.astype("<u4").tobytes())
    out.write(words.astype("<u2").tobytes())
    out.write(pack_raw(raw, w))
    return out.getvalue()


def decode_ints(data: bytes, ctx: np.ndarray, n_ctx: int) -> np.ndarray:
    buf = memoryview(data)
    n, pos = _read_varint(buf, 0)
    if n != len(ctx):
        raise CodecError(f"the stream holds {n} values, the decoder expects {len(ctx)}")
    freq = np.zeros((max(n_ctx, 1), ALPHABET), dtype=np.int64)
    for c in range(n_ctx):
        nnz, pos = _read_varint(buf, pos)
        prev = 0
        for _ in range(nnz):
            ds, pos = _read_varint(buf, pos)
            f, pos = _read_varint(buf, pos)
            prev += ds
            if prev >= ALPHABET:
                raise CodecError("a symbol outside the alphabet")
            freq[c, prev] = f + 1
        if freq[c].sum() != M:
            raise CodecError("a frequency table that doesn't sum to M")
    n_words, pos = _read_varint(buf, pos)
    k = lanes(n) if n else 0
    states = np.frombuffer(data, dtype="<u4", count=k, offset=pos)
    pos += 4 * k
    words = np.frombuffer(data, dtype="<u2", count=n_words, offset=pos)
    pos += 2 * n_words
    sym = rans_decode(states, words, ctx, freq) if n else np.zeros(0, np.int64)
    raw = unpack_raw(bytes(buf[pos:]), widths(sym))
    return desymbolize(sym, raw)


def ulp_encode(target_bits: np.ndarray, ref_bits: np.ndarray, group: np.ndarray, fmt: str) -> bytes:
    """The target's ulp distances from the reference, coded under the reference's contexts."""
    ctx, n_ctx = contexts(ref_bits, group, fmt)
    return encode_ints(keys(target_bits, fmt) - keys(ref_bits, fmt), ctx, n_ctx)


def ulp_decode(data: bytes, ref_bits: np.ndarray, group: np.ndarray, fmt: str) -> np.ndarray:
    ctx, n_ctx = contexts(ref_bits, group, fmt)
    return from_keys(keys(ref_bits, fmt) + decode_ints(data, ctx, n_ctx), fmt)


def ideal_bits(target_bits: np.ndarray, ref_bits: np.ndarray, group: np.ndarray, fmt: str) -> float:
    """`ulp_encode`'s length to within its header and lane states: each context's empirical entropy plus raw bits. For
    choosing among candidates before coding the chosen one."""
    ctx, n_ctx = contexts(ref_bits, group, fmt)
    return ideal_ints(keys(target_bits, fmt) - keys(ref_bits, fmt), ctx, n_ctx)


def ideal_ints(d: np.ndarray, ctx: np.ndarray, n_ctx: int) -> float:
    """`encode_ints`'s length to within its lane states: entropy per context, raw bits, 3 bytes per table entry."""
    if not len(d):
        return 0.0
    sym, _, w = symbolize(d)
    counts = np.bincount(ctx * ALPHABET + sym, minlength=n_ctx * ALPHABET).reshape(n_ctx, ALPHABET).astype(np.float64)
    tot = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(counts > 0, counts * np.log2(counts / np.maximum(tot, 1)), 0.0).sum()
    return float(h + w.sum() + 24 * np.count_nonzero(counts))


def slot_bits(target_bits: np.ndarray, ref_bits: np.ndarray, group: np.ndarray, fmt: str) -> np.ndarray:
    """Each slot's own share of `ideal_bits` (its symbol's -log2 probability in its context, plus its raw bits)."""
    ctx, n_ctx = contexts(ref_bits, group, fmt)
    sym, _, w = symbolize(keys(target_bits, fmt) - keys(ref_bits, fmt))
    flat = ctx * ALPHABET + sym
    counts = np.bincount(flat, minlength=n_ctx * ALPHABET).astype(np.float64)
    tot = np.bincount(ctx, minlength=n_ctx).astype(np.float64)
    return -np.log2(counts[flat] / tot[ctx]) + w


# --- low rank: integer factors, their product added in float64, the residual coded -------------------------------------

@dataclass(frozen=True)
class Factors:
    a: np.ndarray       # int64 (m, r)
    b: np.ndarray       # int64 (n, r)
    shift: int          # the product's scale: (a @ b.T) * 2^-shift


def factorize(delta64: np.ndarray, rank: int, qbits: int, svd=None) -> Factors:
    """Rank-`rank` factors of a float64 matrix, each scaled by a power of two to `qbits` bits of magnitude and rounded."""
    m, n = delta64.shape
    if rank == 0:
        return Factors(np.zeros((m, 0), np.int64), np.zeros((n, 0), np.int64), 0)
    u, s, vt = svd if svd is not None else np.linalg.svd(delta64, full_matrices=False)
    r = min(rank, len(s))
    a = u[:, :r] * np.sqrt(s[:r])
    b = vt[:r].T * np.sqrt(s[:r])
    ea = qbits - int(np.ceil(np.log2(max(np.abs(a).max(), 1e-300))))
    eb = qbits - int(np.ceil(np.log2(max(np.abs(b).max(), 1e-300))))
    return Factors(np.rint(np.ldexp(a, ea)).astype(np.int64), np.rint(np.ldexp(b, eb)).astype(np.int64), ea + eb)


def lowrank_predict(ref_bits: np.ndarray, fac: Factors, fmt: str) -> np.ndarray:
    """The reference plus the factors' product, rounded to the format. Integer product: exact and the same everywhere."""
    p = fac.a @ fac.b.T
    if np.abs(p).max(initial=0) >= 1 << 53:
        raise CodecError("the factors' product exceeds float64's exact integers")
    x = from_bits(ref_bits, fmt).astype(np.float64).reshape(p.shape) + np.ldexp(p.astype(np.float64), -fac.shift)
    return round_to(x, fmt).reshape(-1)


def lowrank_encode(target_bits: np.ndarray, ref_bits: np.ndarray, fac: Factors, fmt: str) -> bytes:
    """Rank, factor shapes and scales, the factors (as integers, one context per factor), then the residual."""
    m, r = fac.a.shape
    n = fac.b.shape[0]
    head = struct.pack("<IIIi", m, n, r, fac.shift)
    fa = encode_ints(np.concatenate([fac.a.reshape(-1), fac.b.reshape(-1)]),
                     np.repeat(np.arange(2, dtype=np.int64), [m * r, n * r]), 2) if r else b""
    pred = lowrank_predict(ref_bits, fac, fmt) if r else ref_bits.reshape(-1)
    res = ulp_encode(target_bits.reshape(-1), pred, np.zeros(pred.size, np.int64), fmt)
    return head + struct.pack("<Q", len(fa)) + fa + res


def lowrank_decode(data: bytes, ref_bits: np.ndarray, fmt: str) -> np.ndarray:
    m, n, r, shift = struct.unpack_from("<IIIi", data, 0)
    (nf,) = struct.unpack_from("<Q", data, 16)
    pos = 24
    if r:
        ab = decode_ints(data[pos:pos + nf], np.repeat(np.arange(2, dtype=np.int64), [m * r, n * r]), 2)
        fac = Factors(ab[: m * r].reshape(m, r), ab[m * r:].reshape(n, r), shift)
        pred = lowrank_predict(ref_bits, fac, fmt)
    else:
        pred = ref_bits.reshape(-1)
    return ulp_decode(data[pos + nf:], pred, np.zeros(pred.size, np.int64), fmt)


# --- affine predictors: float32 coefficients over float64 bases the decoder computes ------------------------------------

def affine64(ref_bits: np.ndarray, bases: list[np.ndarray], coef: np.ndarray, fmt: str) -> np.ndarray:
    x = from_bits(ref_bits, fmt).astype(np.float64)
    for c, b in zip(coef.astype(np.float32).astype(np.float64), bases):
        x = x + c * b
    return x


def affine_predict(ref_bits: np.ndarray, bases: list[np.ndarray], coef: np.ndarray, fmt: str) -> np.ndarray:
    return round_to(affine64(ref_bits, bases, coef, fmt), fmt)


def fit_affine(target_bits: np.ndarray, ref_bits: np.ndarray, bases: list[np.ndarray], fmt: str) -> np.ndarray:
    """Least-squares float32 coefficients for the target's change from the reference over the bases."""
    if not bases:
        return np.zeros(0, np.float32)
    y = from_bits(target_bits, fmt).astype(np.float64) - from_bits(ref_bits, fmt).astype(np.float64)
    g = np.array([[float(np.dot(a, b)) for b in bases] for a in bases])
    h = np.array([float(np.dot(a, y)) for a in bases])
    coef = np.linalg.lstsq(g, h, rcond=None)[0]
    return np.nan_to_num(coef).astype(np.float32)


def affine_encode(target_bits: np.ndarray, ref_bits: np.ndarray, bases: list[np.ndarray], coef: np.ndarray,
                  group: np.ndarray, fmt: str) -> bytes:
    pred = affine_predict(ref_bits, bases, coef, fmt)
    return struct.pack("<I", len(coef)) + coef.astype("<f4").tobytes() + ulp_encode(target_bits, pred, group, fmt)


def affine_decode(data: bytes, ref_bits: np.ndarray, bases: list[np.ndarray], group: np.ndarray, fmt: str) -> np.ndarray:
    (nc,) = struct.unpack_from("<I", data, 0)
    coef = np.frombuffer(data, dtype="<f4", count=nc, offset=4)
    if nc != len(bases):
        raise CodecError(f"{nc} coefficients for {len(bases)} bases")
    return ulp_decode(data[4 + 4 * nc:], affine_predict(ref_bits, bases, coef, fmt), group, fmt)


# --- lattice predictors: a base prediction plus a whole number of float32 steps, then the residual ---------------------

def _steps64(step, n: int) -> np.ndarray:
    s = np.broadcast_to(np.asarray(step, dtype=np.float32), (n,)).astype(np.float64)
    if s.size and not (np.all(s > 0) and np.all(np.isfinite(s))):
        raise CodecError("a lattice step must be positive and finite")
    return s


def lattice_steps(target_bits: np.ndarray, base64: np.ndarray, step, fmt: str) -> np.ndarray:
    """Per slot, the nearest whole number of its `step` (float32, a scalar or one per slot) from the float64 base
    prediction to the target."""
    q = np.rint((from_bits(target_bits, fmt).astype(np.float64) - base64) / _steps64(step, len(base64)))
    if q.size and np.abs(q).max() >= 2.0 ** 33:
        raise CodecError("a lattice step this small needs more than 33-bit step counts")
    return q.astype(np.int64)


def lattice_predict(base64: np.ndarray, step, q: np.ndarray, fmt: str) -> np.ndarray:
    return round_to(base64 + _steps64(step, len(base64)) * q.astype(np.float64), fmt)


def lattice_encode_base(target_bits: np.ndarray, ref_bits: np.ndarray, base64: np.ndarray, step, group: np.ndarray,
                        fmt: str) -> bytes:
    """The step counts under the reference's contexts, then the residual from the lattice point. The decoder computes
    `base64` and holds `step` itself."""
    q = lattice_steps(target_bits, base64, step, fmt)
    ctx, n_ctx = contexts(ref_bits, group, fmt)
    qs = encode_ints(q, ctx, n_ctx)
    return struct.pack("<Q", len(qs)) + qs + ulp_encode(target_bits, lattice_predict(base64, step, q, fmt), group, fmt)


def lattice_decode_base(data: bytes, ref_bits: np.ndarray, base64: np.ndarray, step, group: np.ndarray,
                        fmt: str) -> np.ndarray:
    (nq,) = struct.unpack_from("<Q", data, 0)
    ctx, n_ctx = contexts(ref_bits, group, fmt)
    q = decode_ints(data[8:8 + nq], ctx, n_ctx)
    return ulp_decode(data[8 + nq:], lattice_predict(base64, step, q, fmt), group, fmt)


def lattice_slot_bits(target_bits: np.ndarray, ref_bits: np.ndarray, base64: np.ndarray, step, group: np.ndarray,
                      fmt: str) -> np.ndarray:
    """Each slot's share of `lattice_ideal`: its step count's -log2 probability and raw bits, and its residual's."""
    q = lattice_steps(target_bits, base64, step, fmt)
    ctx, n_ctx = contexts(ref_bits, group, fmt)
    sym, _, w = symbolize(q)
    flat = ctx * ALPHABET + sym
    counts = np.bincount(flat, minlength=n_ctx * ALPHABET).astype(np.float64)
    tot = np.bincount(ctx, minlength=n_ctx).astype(np.float64)
    pred = lattice_predict(base64, step, q, fmt)
    return -np.log2(counts[flat] / tot[ctx]) + w + slot_bits(target_bits, pred, group, fmt)


def lattice_ideal(target_bits: np.ndarray, ref_bits: np.ndarray, base64: np.ndarray, step, group: np.ndarray,
                  fmt: str) -> float:
    """`lattice_encode_base`'s length to within its headers and lane states, for choosing the step."""
    q = lattice_steps(target_bits, base64, step, fmt)
    ctx, n_ctx = contexts(ref_bits, group, fmt)
    return ideal_ints(q, ctx, n_ctx) + ideal_bits(target_bits, lattice_predict(base64, step, q, fmt), group, fmt)


def fit_lattice(target_bits: np.ndarray, ref_bits: np.ndarray, bases: list[np.ndarray], coef: np.ndarray,
                group: np.ndarray, fmt: str, scales=(1.0, 0.5, 2.0),
                sample: int = 1 << 22) -> tuple[np.ndarray, float] | None:
    """Coefficients and a step for `lattice_encode`. Each candidate step is one of `scales` times the median distance from
    the rounded affine prediction to the target where they differ; its step counts give a least-squares refit of the
    coefficients and the step together (a weight decay fitted before the steps are known is swamped by them). The pair
    with the shortest `lattice_ideal` on a strided sample of at most `sample` slots wins, the earlier on a tie. None
    where the prediction is already exact on the sample."""
    sl = slice(None, None, max(1, len(ref_bits) // sample))
    t, r, g = target_bits[sl], ref_bits[sl], group[sl]
    bs = [b[sl] for b in bases]
    coef = np.asarray(coef, np.float32)
    base = affine64(r, bs, coef, fmt)
    tv = from_bits(t, fmt).astype(np.float64)
    y = np.abs(tv - from_bits(round_to(base, fmt), fmt).astype(np.float64))
    y = y[y > 0]
    if not y.size:
        return None
    s0 = float(np.median(y))
    dy = tv - from_bits(r, fmt).astype(np.float64)
    best = None
    for c in scales:
        s = float(np.float32(s0 * c))
        if not (s > 0 and np.isfinite(s)):
            continue
        try:
            q = lattice_steps(t, base, s, fmt).astype(np.float64)
        except CodecError:
            continue
        cols = [*bs, q]
        gm = np.array([[float(np.dot(u, v)) for v in cols] for u in cols])
        h = np.array([float(np.dot(u, dy)) for u in cols])
        sol = np.nan_to_num(np.linalg.lstsq(gm, h, rcond=None)[0])
        cands = [(coef, s)]
        if np.float32(sol[-1]) > 0 and np.isfinite(np.float32(sol[-1])):
            cands.append((sol[:-1].astype(np.float32), float(np.float32(sol[-1]))))
        for cf, st in cands:
            try:
                bits = lattice_ideal(t, r, affine64(r, bs, cf, fmt), st, g, fmt)
            except CodecError:
                continue
            if best is None or bits < best[0]:
                best = (bits, cf, st)
    return None if best is None else (best[1], best[2])


def lattice_encode(target_bits: np.ndarray, ref_bits: np.ndarray, bases: list[np.ndarray], coef: np.ndarray,
                   step: float, group: np.ndarray, fmt: str) -> bytes:
    """One tensor: its affine coefficients and step (float32), then `lattice_encode_base` on the affine prediction."""
    head = struct.pack("<I", len(coef)) + coef.astype("<f4").tobytes() + np.float32(step).astype("<f4").tobytes()
    return head + lattice_encode_base(target_bits, ref_bits, affine64(ref_bits, bases, coef, fmt), step, group, fmt)


def lattice_decode(data: bytes, ref_bits: np.ndarray, bases: list[np.ndarray], group: np.ndarray, fmt: str) -> np.ndarray:
    (nc,) = struct.unpack_from("<I", data, 0)
    if nc != len(bases):
        raise CodecError(f"{nc} coefficients for {len(bases)} bases")
    coef = np.frombuffer(data, dtype="<f4", count=nc, offset=4)
    step = np.frombuffer(data, dtype="<f4", count=1, offset=4 + 4 * nc)[0]
    return lattice_decode_base(data[8 + 4 * nc:], ref_bits, affine64(ref_bits, bases, coef, fmt), step, group, fmt)


# --- general-purpose baselines -----------------------------------------------------------------------------------------

def planes(v: np.ndarray) -> bytes:
    """Byte planes of an array of unsigned words, most significant first."""
    b = v.view(np.uint8).reshape(len(v), v.dtype.itemsize)
    return b[:, ::-1].T.copy().tobytes()


def unplanes(data: bytes, dtype) -> np.ndarray:
    w = np.dtype(dtype).itemsize
    b = np.frombuffer(data, dtype=np.uint8).reshape(w, -1)
    return b.T[:, ::-1].copy().view(dtype).reshape(-1)


def xor_planes(target_bits: np.ndarray, ref_bits: np.ndarray) -> bytes:
    return planes(np.ascontiguousarray(target_bits ^ ref_bits))


def xz(data: bytes) -> bytes:
    return lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)
