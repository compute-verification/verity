"""N3's proof unit for one port-window, zero-knowledge: the frames a certifier recorded on one link direction in one
window are the committed crossing values, framed as the layout committed before the run says. A Boolean Definition,
which a proof unit of the proof service runs; its one output is the ok bit.

    N3Window_v2{FMT, N, M, SALTED}
        (layout, layout_domain: Digest, at: Array<DL>, record, record_domain: Digest, direction: Array<1>,
         roots, domains: Array<U, Digest>, [salt: Array<72, Byte>],
         frames: Array<N, Array<B, Byte>>,
         descriptors: Array<n, Array<LW>>, descriptor_salts: Array<n, Salt>,
         descriptor_paths: Array<n, Array<DL, Digest>>,
         rows: Array<M, Row>, row_salts: Array<M, Salt>, row_paths: Array<M, Array<DB, Digest>>,
         record_salt: Salt, record_sibling: Array<1, Digest>, [slot: Array<SB>]) -> Value<1>
        Digest = Array<8, Array<64, Value<1>>> (SHA-512 words, `sha512.words_be`),
        Salt = Array<24, Array<64, Value<1>>>, Byte = Array<8, Value<1>>, Row = Array<256, Value<1>>,
        n = N - SALTED, SB = max(1, bitlen(N - 1))

**Statics: the size class.** `FMT` (`Format`) is the run's descriptor format: the frame size B, the widest prefix P
and suffix Q, the most rows a frame draws from MF, each unit's value-tree depth (`DEPTHS`, U units) and the layout
tree's depth DL. N is the window's recorded frame count, M its pool of rows (a power of two at least the rows its
frames draw from), SALTED whether one of the frames is the certifier's salt frame. Nothing else about the layout is
static.

**Public inputs.** `layout` is the layout's root, `layout_domain` its domain id, and `at` the position of the window's
first descriptor: the port's index times the port's capacity, plus the frames the port recorded in earlier windows
(less their salt frames), all public occupancy. `record` is the (link, window) record commitment's root,
`record_domain` its domain id and `direction` the port's end of the link. `roots` and `domains` are the units' value
commitments and domain ids, `salt` the window number (u64, big-endian) and the revealed salt commitment.

**Witness.** The frames, the n descriptors (the layout's rows at at, at + 1, ...) with their hiding salts and paths, the
pool's rows with their salts and paths, the record commitment's salt and sibling, and the salt frame's slot.

**What 1 means.** (1) Each descriptor opens at its position in the layout (`N3HidingRead`: its sha512/row/v2 digest,
hm96's hiding leaf, the frame-v3-sha512 read). (2) With m_i the rows descriptor i draws from and o_i = m_0 + ... +
m_(i-1), the m's fit the pool (sum <= M); pool row o_i + j (j < m_i) is the row descriptor i names in its slot j, and
every other pool row is zero. (3) Every pool row opens in its unit's commitment at the position its slot names, or is
zero for a zero slot (`N3RowRead`). (4) Frame i (the salt frame's slot skipped) is what descriptor i puts on rows
o_i ... o_i + m_i - 1 (`N3Frame`, `assemble`). (5) The salt frame, when SALTED, is `salt_frame` at the slot. (6) The
record digest of the N frames (`record_digest`) opens at `direction` in the record commitment.

**A descriptor** (`Descriptor`, LW bits, `encode`): a mode (RAW 1, BASE64 2; 0 and 3 are invalid, and `NONE` pads the
tree), m, the text's byte offset p, its data length r, the suffix offset s, the prefix (P bytes) and suffix (Q bytes),
and per slot j < MF an identity (kind 1: unit u's row at position pos; kind 0: a zero row) and a 256-bit keep mask. The
frame is prefix XOR text << 8p XOR suffix << 8s over B bytes, where the stream is the kept bits of the slots' rows in
(slot, bit) order, the data its first min(r, B) bytes, and the text the data (RAW) or its base64 (BASE64; `assemble`).

**Commitments and who registers them.** Each unit registers its values (`values`). The prover registers the layout
before the run (`register_layout`): one advice tree of the descriptors, one per (port, frame index) at port * CAP +
index, under the binding SHA-512(`LAYOUT_LABEL` || program || placement root). The certifiers register each window's
record commitment (`register_record`): a two-row advice tree of the link's two directions' record digests, under
SHA-512(`RECORD_LABEL` || program || link || window), publishing its root instead of the digests; the salts go to the
proof service behind the firewall. The network_sim README ("N3 in zero knowledge") has what that asks of the
certifier, and the open questions.

    python -m workload.n3_unit sim --out FILE [--tamper UNIT:MODE] [--buckets K] [--frame-bytes B]
    python -m workload.n3_unit lab RUN_DIR --out FILE     every window a lab run's certifiers recorded (lab_run --out)
    python -m workload.n3_unit circuit-check --out FILE   circuit-check's checks on `demo()`'s window and its parts
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, NamedTuple, Sequence

import numpy as np

from verity.core.protocols.certifier.active import DATA, HEADER, LAST, encode_frame
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import ONE, ZERO, add, and_reduce, const_bits, shl_window, sub
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array
from verity.ml.commitments import advice, hm96
from verity.ml.commitments.gates.frame_v3 import FrameV3Leaf, FrameV3Node, FrameV3Read, index_bytes, root_ref
from verity.ml.commitments.gates.hm96 import Hm96Sha512Leaf
from verity.ml.commitments.gates.lms import Sha512Compress
from verity.ml.commitments.gates.sha512 import IV, midstate, pad_tail, words_be
from verity.ml.commitments.merkle import merkle_depth, tree_levels
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.ml.commitments.rowleaf import sha512_row_v2_digest, sha512_row_v2_prefix

from . import framing as F
from . import model as M
from . import placement as P
from . import values as V
from .hardware import Hardware

BIT = BL.BIT
BYTE = Array(8, BIT)
WORD = Array(64, BIT)
DIGEST = Array(8, WORD)
ROW = Array(V.ROW_BITS, BIT)
SALT_T = Array(hm96.SHA512.salt_bytes // 8, WORD)
LEVEL = Array(8, BIT)
DOMAIN = b"network_sim/certifier-record/v1\0"
LAYOUT_LABEL = b"verity/netsim/n3-layout/v1/"
RECORD_LABEL = b"verity/netsim/n3-record/v1/"
RAW, B64 = 1, 2
B64_ALPHABET = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
# What base64 adds to a 6-bit value v, mod 256, for v below 26, below 52, below 62, 62 and 63.
B64_OFFSETS = (65, 71, 252, 237, 240)


def record_digest(frames: Sequence[bytes], reason: str = "") -> bytes:
    h = hashlib.sha512(DOMAIN + reason.encode() + b"\0")
    for f in frames:
        h.update(f)
    return h.digest()


def prefix(reason: str) -> bytes:
    return DOMAIN + reason.encode() + b"\0"


SALT, SALT_STREAM = 3, 0xFFFF_FFFF
SALT_PAYLOAD = 72


def salt_frame(window: int, commitment: bytes, frame_bytes: int) -> bytes:
    """Window `window`'s salt frame as the ingress certifier records it: kind `SALT` on `SALT_STREAM`, its payload the
    window (u64, big-endian) and the salt's 64-byte commitment, which is public once the salt is revealed."""
    return encode_frame(SALT, SALT_STREAM, window.to_bytes(8, "big") + commitment, frame_bytes)


def _is_salt(frame: bytes) -> bool:
    kind, stream, _ = HEADER.unpack_from(frame)
    return kind == SALT and stream == SALT_STREAM


def hiding_value(row: bytes, n_bits: int, salt: bytes) -> bytes:
    """The 64 bytes an advice row's frame-v3-sha512 leaf binds: hm96's tree leaf over the row's sha512/row/v2 digest."""
    inst, key = hm96.SHA512, hm96.SHA512.default_key
    return inst.tree_leaf(key, inst.commit_string(key, sha512_row_v2_digest(row, n_bits), salt))


# --- pieces -----------------------------------------------------------------------------------------------------------
# A frame as the statement declares it: pieces whose widths add up to 8B bits, least significant first from bit 0 of
# byte 0. ("k", w, v): w constant bits, v's; ("r", key, o, w): bits o ... o + w - 1 of the row `key` names (unit, row);
# ("b", pieces, L): the base64 text (RFC 4648, padded) of the L bytes `pieces` give, 32 ceil(L / 3) bits.

def const(data: bytes) -> tuple:
    return ("k", 8 * len(data), int.from_bytes(data, "little"))


def zeros(nbytes: int) -> tuple:
    return ("k", 8 * nbytes, 0)


def width(p: Sequence) -> int:
    if p[0] == "k":
        return p[1]
    if p[0] == "r":
        return p[3]
    return 32 * -(-p[2] // 3)


def _slice(pieces: Sequence[tuple], lo: int, hi: int) -> list[tuple]:
    """Bits lo ... hi - 1 of the concatenation of constant and row pieces."""
    out, at = [], 0
    for p in pieces:
        w = width(p)
        a, b = max(lo, at) - at, min(hi, at + w) - at
        if a < b and p[0] == "k":
            out.append(("k", b - a, (p[2] >> a) & ((1 << (b - a)) - 1)))
        elif a < b:
            out.append(("r", p[1], p[2] + a, b - a))
        at += w
    return out


def _merge(pieces: Sequence[tuple]) -> list[tuple]:
    """Adjacent constants joined, adjacent runs of one row joined, empty pieces dropped."""
    out: list[tuple] = []
    for p in pieces:
        if p[0] == "b":
            p = ("b", tuple(_merge(p[1])), p[2])
        elif width(p) == 0:
            continue
        q = out[-1] if out else None
        if q is not None and q[0] == p[0] == "k":
            out[-1] = ("k", q[1] + p[1], q[2] | (p[2] << q[1]))
        elif q is not None and q[0] == p[0] == "r" and q[1] == p[1] and q[2] + q[3] == p[2]:
            out[-1] = ("r", q[1], q[2], q[3] + p[3])
        else:
            out.append(p)
    return out


def _bits(pieces: Sequence[tuple], row: Callable[[tuple], bytes]) -> tuple[int, int]:
    acc, at = 0, 0
    for p in pieces:
        if p[0] == "k":
            v, w = p[2], p[1]
        elif p[0] == "r":
            v, w = (int.from_bytes(row(p[1]), "little") >> p[2]) & ((1 << p[3]) - 1), p[3]
        else:
            inner, n = _bits(p[1], row)
            text = base64.b64encode(inner.to_bytes(n // 8, "little"))
            v, w = int.from_bytes(text, "little"), 8 * len(text)
        acc |= v << at
        at += w
    return acc, at


def expand(pieces: Sequence[tuple], B: int, row: Callable[[tuple], bytes]) -> bytes:
    """The frame the pieces declare over the rows `row(key)` gives."""
    return _bits(pieces, row)[0].to_bytes(B, "little")


# --- the descriptor format --------------------------------------------------------------------------------------------

class Format(NamedTuple):
    """The run's descriptor format, a static of the unit: frame bytes, prefix and suffix bytes, slots, unit depths, the
    layout tree's depth. The derived widths and the field offsets (bits, least significant first) follow."""

    B: int
    P: int
    Q: int
    MF: int
    DEPTHS: tuple
    DL: int

    @property
    def U(self) -> int:
        return len(self.DEPTHS)

    @property
    def DB(self) -> int:
        return max(1, *self.DEPTHS)

    @property
    def UB(self) -> int:
        return max(1, (self.U - 1).bit_length())

    @property
    def MB(self) -> int:
        return self.MF.bit_length()

    @property
    def PB(self) -> int:
        return self.P.bit_length()

    @property
    def LB(self) -> int:
        return self.B.bit_length()

    @property
    def IDW(self) -> int:
        return 1 + self.UB + self.DB

    @property
    def W(self) -> int:
        return V.ROW_BITS * self.MF

    @property
    def Z(self) -> int:
        return max(1, (self.W - 1).bit_length())

    @property
    def P_OFF(self) -> int:
        return 2 + self.MB

    @property
    def R_OFF(self) -> int:
        return self.P_OFF + self.PB

    @property
    def S_OFF(self) -> int:
        return self.R_OFF + self.LB

    @property
    def PRE_OFF(self) -> int:
        return self.S_OFF + self.LB

    @property
    def SUF_OFF(self) -> int:
        return self.PRE_OFF + 8 * self.P

    @property
    def HW(self) -> int:
        return self.SUF_OFF + 8 * self.Q

    @property
    def KEEP_OFF(self) -> int:
        return self.HW + self.MF * self.IDW

    @property
    def LW(self) -> int:
        return self.KEEP_OFF + V.ROW_BITS * self.MF


def _format(fmt) -> Format:
    try:
        f = Format(*fmt)
    except TypeError:
        raise ValueError(f"FMT is (B, P, Q, MF, DEPTHS, DL), not {fmt!r}") from None
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (f.B, f.P, f.Q, f.MF, f.DL)):
        raise ValueError(f"FMT's B, P, Q, MF and DL are ints, not {fmt!r}")
    if f.B <= HEADER.size or not 1 <= f.P <= f.B or not 1 <= f.Q <= f.B or f.MF < 1 or not 1 <= f.DL <= 64:
        raise ValueError(f"a format has B > {HEADER.size}, 1 <= P, Q <= B, MF >= 1 and 1 <= DL <= 64, not {fmt!r}")
    depths = f.DEPTHS
    if not isinstance(depths, tuple) or not depths or any(not isinstance(d, int) or not 0 <= d <= 63 for d in depths):
        raise ValueError(f"DEPTHS is a nonempty tuple of depths 0 ... 63, not {f.DEPTHS!r}")
    return f


@dataclass(frozen=True)
class Descriptor:
    """One frame's layout: mode, text offset p, data bytes r, suffix offset s, prefix, suffix, and per slot its identity
    (kind, unit index, position) and keep mask."""

    mode: int = 0
    p: int = 0
    r: int = 0
    s: int = 0
    prefix: bytes = b""
    suffix: bytes = b""
    ids: tuple = ()
    keep: tuple = ()

    @property
    def m(self) -> int:
        return len(self.ids)


NONE = Descriptor()


def encode(f: Format, d: Descriptor) -> bytes:
    """The descriptor as a layout row: LW bits, least significant first (`rowleaf.bit_row`'s bytes)."""
    if (d.m > f.MF or len(d.keep) != d.m or len(d.prefix) > f.P or len(d.suffix) > f.Q or not 0 <= d.mode < 4
            or not 0 <= d.p < 1 << f.PB or not 0 <= d.r < 1 << f.LB or not 0 <= d.s < 1 << f.LB):
        raise ValueError(f"the descriptor doesn't fit the format {f}")
    x = d.mode | d.m << 2 | d.p << f.P_OFF | d.r << f.R_OFF | d.s << f.S_OFF
    x |= int.from_bytes(d.prefix, "little") << f.PRE_OFF | int.from_bytes(d.suffix, "little") << f.SUF_OFF
    for j, ((kind, u, pos), keep) in enumerate(zip(d.ids, d.keep)):
        if kind not in (0, 1) or not 0 <= u < 1 << f.UB or not 0 <= pos < 1 << f.DB or not 0 <= keep < 1 << V.ROW_BITS:
            raise ValueError(f"slot {j}'s identity or keep mask doesn't fit the format")
        x |= (kind | u << 1 | pos << (1 + f.UB)) << (f.HW + j * f.IDW)
        x |= keep << (f.KEEP_OFF + V.ROW_BITS * j)
    return x.to_bytes(-(-f.LW // 8), "little")


class Fields(NamedTuple):
    """A layout row's fields, whatever its bits: every slot, past m too."""

    mode: int
    m: int
    p: int
    r: int
    s: int
    prefix: int
    suffix: int
    ids: tuple
    keep: tuple


def fields(f: Format, row: bytes) -> Fields:
    x = int.from_bytes(row, "little")

    def get(off: int, w: int) -> int:
        return (x >> off) & ((1 << w) - 1)

    ids = []
    for j in range(f.MF):
        v = get(f.HW + j * f.IDW, f.IDW)
        ids.append((v & 1, (v >> 1) & ((1 << f.UB) - 1), v >> (1 + f.UB)))
    keep = tuple(get(f.KEEP_OFF + V.ROW_BITS * j, V.ROW_BITS) for j in range(f.MF))
    return Fields(get(0, 2), get(2, f.MB), get(f.P_OFF, f.PB), get(f.R_OFF, f.LB), get(f.S_OFF, f.LB),
                  get(f.PRE_OFF, 8 * f.P), get(f.SUF_OFF, 8 * f.Q), tuple(ids), keep)


def decode(f: Format, row: bytes) -> Descriptor:
    """The descriptor a layout row holds (its first m slots; the prefix and suffix without trailing zero bytes)."""
    x = fields(f, row)
    m = min(x.m, f.MF)
    return Descriptor(x.mode, x.p, x.r, x.s, x.prefix.to_bytes(f.P, "little").rstrip(b"\0"),
                      x.suffix.to_bytes(f.Q, "little").rstrip(b"\0"), x.ids[:m], x.keep[:m])


def assemble(f: Format, row: bytes, rows: Sequence[bytes]) -> bytes:
    """The frame the layout row puts on `rows` (its MF slots' rows, zero past those given): `N3Frame`'s reference."""
    x = fields(f, row)
    stream, n = 0, 0
    for j in range(f.MF):
        value = int.from_bytes(rows[j], "little") if j < len(rows) else 0
        keep = x.keep[j]
        while keep:
            b = (keep & -keep).bit_length() - 1
            stream |= ((value >> b) & 1) << n
            n += 1
            keep &= keep - 1
    data = (stream & ((1 << 8 * min(x.r, f.B)) - 1)).to_bytes(f.B, "little")
    if x.mode == RAW:
        text = data
    elif x.mode == B64:
        g = f.B // 4
        enc = base64.b64encode(data[:3 * g])
        chars = bytearray(f.B)
        for j in range(4 * g):
            if x.r >= 3 * j // 4 + 1:
                chars[j] = enc[j]
            elif x.r >= 3 * (j // 4) + 1:
                chars[j] = ord("=")
        text = bytes(chars)
    else:
        text = bytes(f.B)
    out = x.prefix ^ (int.from_bytes(text, "little") << 8 * x.p) ^ (x.suffix << 8 * x.s)
    return (out & ((1 << 8 * f.B) - 1)).to_bytes(f.B, "little")


def describe(pieces: Sequence[tuple], B: int, index: Callable[[tuple], tuple[int, int]]) -> Descriptor:
    """The descriptor of a frame declared as pieces, each row piece's key named (unit index, position) by `index`.

    The frame's constant bits give the prefix (bytes before p) and the suffix (bytes from s, trailing zeros dropped);
    the non-constant pieces, one base64 piece (BASE64) or row pieces (RAW), give the text between. Consecutive row
    pieces of one row at increasing offsets share a slot; a zero constant inside the text is read from zero slots of 256
    bits each, and a nonzero one has no descriptor."""
    data, first, last, at, walk = 0, None, None, 0, []
    for q in pieces:
        w = width(q)
        walk.append((q, at))
        if q[0] == "k":
            data |= q[2] << at
        else:
            first = at if first is None else first
            last = at + w
        at += w
    if at != 8 * B:
        raise ValueError(f"the pieces are {at} bits, not {8 * B}")
    raw = data.to_bytes(B, "little")
    if first is None:
        pre = raw.rstrip(b"\0")
        return Descriptor(RAW, len(pre), 0, len(pre), pre)
    p = first // 8
    inside = [q for q, start in walk if start >= first and start + width(q) <= last]
    if len(inside) == 1 and inside[0][0] == "b":
        if first % 8:
            raise ValueError("a base64 text starts on a byte")
        mode, r, run = B64, inside[0][2], list(inside[0][1])
        length = 4 * -(-r // 3)
    else:
        if any(q[0] == "b" for q in inside):
            raise ValueError("a base64 text shares its frame with other row bits")
        if (data >> 8 * p) & ((1 << (first - 8 * p)) - 1):
            raise ValueError("the text's first byte holds nonzero constant bits")
        run = ([("k", first - 8 * p, 0)] if first > 8 * p else []) + inside
        mode, r = RAW, -(-(last - 8 * p) // 8)
        length = r
    run = _merge(run)
    while run and run[-1][0] == "k" and run[-1][2] == 0:
        run.pop()
    s = p + length
    if s > B or (data >> 8 * p) & ((1 << 8 * length) - 1):
        raise ValueError("the text's bytes hold nonzero constant bits")
    ids: list[tuple[int, int, int]] = []
    keep: list[int] = []
    cur: tuple = (None, None, 0)
    for q in run:
        if q[0] == "r":
            key, o, w = q[1], q[2], q[3]
            if cur[0] == 1 and cur[1] == key and o >= cur[2]:
                keep[-1] |= ((1 << w) - 1) << o
            else:
                ids.append((1, *index(key)))
                keep.append(((1 << w) - 1) << o)
            cur = (1, key, o + w)
        elif q[0] == "k":
            if q[2]:
                raise ValueError("a nonzero constant inside a frame's text")
            w = q[1]
            while w:
                if cur[0] != 0 or cur[2] == V.ROW_BITS:
                    ids.append((0, 0, 0))
                    keep.append(0)
                    cur = (0, None, 0)
                t = min(w, V.ROW_BITS - cur[2])
                keep[-1] |= ((1 << t) - 1) << cur[2]
                cur, w = (0, None, cur[2] + t), w - t
        else:
            raise ValueError("a base64 text inside a base64 text")
    return Descriptor(mode, p, r, s, raw[:p], raw[s:].rstrip(b"\0"), tuple(ids), tuple(keep))


def fit(descriptors: Sequence[Descriptor], B: int, depths: Sequence[int], DL: int) -> Format:
    """The narrowest format that holds the descriptors."""
    return _format((B, max([1] + [max(d.p, len(d.prefix)) for d in descriptors]),
                    max([1] + [len(d.suffix) for d in descriptors]), max([1] + [d.m for d in descriptors]),
                    tuple(depths), DL))


# --- the circuit: traced parts ----------------------------------------------------------------------------------------

def _part(name: str, statics: tuple[str, ...], signature: Callable, build: Callable, doc: str) -> CompositeDefinition:
    """A Definition traced from ``build(C, S, *elements)`` once per binding of ``statics``."""

    @lru_cache(maxsize=None)
    def traced(*key) -> TR.Traced:
        S = SimpleNamespace(**dict(zip(statics, key)))
        params, ret = signature(S)
        return TR.trace(lambda C, *e: build(C, S, *e), [t for _, t in params], ret)

    def body(B, S, *colls):
        params, ret = signature(S)
        return TR.emit(traced(*(getattr(S, k) for k in statics)), [t for _, t in params], ret, B, *colls)

    d = CompositeDefinition(name, 1, statics, signature, body, conformance="exact-model-tested", register=False,
                            doc=doc)
    d.traced = traced  # type: ignore[attr-defined]
    return d


def _thresholds(C, x: Sequence) -> Callable[[int], object]:
    """c -> (x >= c) for constants c, x unsigned and least significant bit first, each computed once."""
    cache: dict[int, object] = {}

    def ge(c: int):
        if c <= 0:
            return ONE
        if c >= 1 << len(x):
            return ZERO
        if c not in cache:
            cache[c] = sub(C, list(x), const_bits(c, len(x)), len(x))[1].inv()
        return cache[c]

    return ge


def _onehot(C, bits: Sequence, n: int) -> list:
    return [and_reduce(C, [b if k >> i & 1 else b.inv() for i, b in enumerate(bits)]) for k in range(n)]


def _eq(C, a: Sequence, b: Sequence):
    return and_reduce(C, [(x ^ y).inv() for x, y in zip(a, b, strict=True)])


def b64_char(C, v: Sequence) -> list:
    """The base64 character of a 6-bit value (bits least significant first), as 8 bits."""
    v = list(v)
    lt = [sub(C, v, const_bits(k, 6), 6)[1] for k in (26, 52, 62)]
    top = and_reduce(C, v)
    flags = (lt[0], lt[1] ^ lt[0], lt[2] ^ lt[1], lt[2].inv() ^ top, top)
    offset = []
    for j in range(8):
        bit = ZERO
        for f, o in zip(flags, B64_OFFSETS):
            if o >> j & 1:
                bit = bit ^ f
        offset.append(bit)
    return add(C, v + [ZERO, ZERO], offset, width=8)


def b64(C, data: Sequence, nbytes: int) -> list:
    """The base64 text of `nbytes` bytes (8 bits each, least significant first, in `data` one after another)."""
    out = []
    for g in range(0, nbytes, 3):
        k = min(3, nbytes - g)
        group = [list(data[8 * (g + i):8 * (g + i + 1)]) if i < k else [ZERO] * 8 for i in range(3)]
        value = group[2] + group[1] + group[0]
        for c in range(4):
            out += b64_char(C, value[18 - 6 * c:24 - 6 * c]) if c <= k else const_bits(ord("="), 8)
    return out


def _sig_place(S):
    f = _format(S.FMT)
    return (("head", Array(f.HW, BIT)), ("stream", Array(8 * f.B, BIT)), ("frame", Array(f.B, BYTE))), BIT


def _build_place(C, S, head, stream, *frame):
    f = _format(S.FMT)
    nb = 8 * f.B
    mode, p = head[0:2], list(head[f.P_OFF:f.P_OFF + f.PB])
    r, s = head[f.R_OFF:f.R_OFF + f.LB], list(head[f.S_OFF:f.S_OFF + f.LB])
    ge = _thresholds(C, r)
    data = [C.AND(stream[k], ge(k // 8 + 1)) for k in range(nb)]
    g = f.B // 4
    chars = b64(C, data, 3 * g)
    text64 = []
    for j in range(4 * g):
        real = ge(3 * j // 4 + 1)
        pad = ge(3 * (j // 4) + 1) ^ real
        text64 += [C.AND(real, chars[8 * j + t]) ^ (pad if ord("=") >> t & 1 else ZERO) for t in range(8)]
    text64 += [ZERO] * (nb - len(text64))
    is_raw, is_b64 = C.AND(mode[0], mode[1].inv()), C.AND(mode[1], mode[0].inv())
    text = [C.AND(is_raw, x) ^ C.AND(is_b64, y) for x, y in zip(data, text64)]
    moved = shl_window(C, text, [ZERO] * 3 + p, 0, nb)
    tail = shl_window(C, list(head[f.SUF_OFF:f.SUF_OFF + 8 * f.Q]), [ZERO] * 3 + s, 0, nb)
    pre = list(head[f.PRE_OFF:f.PRE_OFF + 8 * f.P])[:nb]
    pre += [ZERO] * (nb - len(pre))
    got = [a ^ b ^ c for a, b, c in zip(pre, moved, tail)]
    return [C.AND(_eq(C, got, [x for byte in frame for x in byte]), mode[0] ^ mode[1])]


N3Place = _part(
    "N3Place", ("FMT",), _sig_place, _build_place,
    "1 iff the mode is RAW or BASE64 and the frame is prefix XOR text << 8p XOR suffix << 8s, the text the stream's "
    "first min(r, B) bytes (RAW) or their base64 (BASE64), from the descriptor's head")


def _sig_local(S):
    return (("keep", ROW), ("row", ROW)), Array(V.ROW_BITS, Array(10, BIT))


def _build_local(C, S, keep, row):
    z, out = [ZERO] * 8, []
    for b in range(V.ROW_BITS):
        out.append([C.AND(keep[b], row[b]), keep[b]] + z)
        carry, nz = keep[b].inv(), []
        for k in range(8):
            nz.append(z[k] ^ carry)
            carry = C.AND(carry, z[k])
        z = nz
    return out


N3Local = _part("N3Local", (), _sig_local, _build_local,
                "per bit of a slot: the kept value, the keep bit, and the number of unkept bits before it (8 bits)")


def _sig_bases(S):
    return (("last", Array(S.MF, Array(9, BIT))),), Array(S.MF, Array(S.Z, BIT))


def _build_bases(C, S, *last):
    acc, out = [ZERO] * S.Z, []
    for e in last:
        out.append(acc)
        acc = add(C, acc, add(C, list(e[1:9]), [e[0].inv()], width=9), width=S.Z)
    return out


N3Bases = _part("N3Bases", ("MF", "Z"), _sig_bases, _build_bases,
                "each slot's base: the unkept bits of the slots before it, from each slot's last bit (keep, count)")


def _sig_addbase(S):
    cells = Array(V.ROW_BITS, Array(S.Z + 1, BIT))
    return (("local", Array(V.ROW_BITS, Array(10, BIT))), ("base", Array(S.Z, BIT))), cells


def _build_addbase(C, S, *e):
    out = []
    for cell in e[:V.ROW_BITS]:
        out.append([cell[0]] + [C.AND(cell[1], x) for x in add(C, list(cell[2:10]), list(e[V.ROW_BITS]), width=S.Z)])
    return out


N3AddBase = _part("N3AddBase", ("Z",), _sig_addbase, _build_addbase,
                  "per bit of a slot: the kept value and, for a kept bit, its shift (the unkept bits before it)")


def _sig_step(S):
    return (("here", Array(S.K + 1, BIT)), ("there", Array(S.K + 1, BIT))), Array(S.K, BIT)


def _build_step(C, S, here, there):
    stay, move = here[1].inv(), there[1]
    return [C.AND(stay, a) ^ C.AND(move, b) for a, b in zip([here[0]] + list(here[2:]), [there[0]] + list(there[2:]))]


N3Step = _part("N3Step", ("K",), _sig_step, _build_step,
               "one compaction stage at a cell: the cell if it stays (its shift's low bit 0), XOR the cell 2^t on if "
               "that one moves; the value and the shift's remaining bits")


def _sig_stay(S):
    return (("here", Array(S.K + 1, BIT)),), Array(S.K, BIT)


N3Stay = _part("N3Stay", ("K",), _sig_stay,
               lambda C, S, here: [C.AND(here[1].inv(), a) for a in [here[0]] + list(here[2:])],
               "one compaction stage at a cell nothing moves to")


def _sig_mux(S):
    return (("a", Array(S.W, BIT)), ("b", Array(S.W, BIT)), ("s", BIT)), Array(S.W, BIT)


N3Mux = _part("N3Mux", ("W",), _sig_mux, lambda C, S, a, b, s: [x ^ C.AND(s[0], x ^ y) for x, y in zip(a, b)],
              "b if s, else a")


def _sig_gate(S):
    return (("x", Array(S.W, BIT)), ("s", BIT)), Array(S.W, BIT)


N3Gate = _part("N3Gate", ("W",), _sig_gate, lambda C, S, x, s: [C.AND(s[0], v) for v in x], "x if s, else 0")


def _sig_xor(S):
    return (("xs", Array(S.N, Array(S.W, BIT))),), Array(S.W, BIT)


def _build_xor(C, S, *xs):
    out = list(xs[0])
    for x in xs[1:]:
        out = [a ^ b for a, b in zip(out, x)]
    return out


N3Xor = _part("N3Xor", ("N", "W"), _sig_xor, _build_xor, "the XOR of N bit strings")


def _sig_below(S):
    return (("m", Array(S.MB, BIT)),), Array(S.MF, BIT)


N3Below = _part("N3Below", ("MF", "MB"), _sig_below,
                lambda C, S, m: [_thresholds(C, m)(j + 1) for j in range(S.MF)], "bit j: j < m")


def _sig_positions(S):
    return (("at", Array(S.DL, BIT)),), Array(S.N, Array(S.DL, BIT))


N3Positions = _part("N3Positions", ("DL", "N"), _sig_positions,
                    lambda C, S, at: [add(C, list(at), const_bits(i, S.DL), width=S.DL) for i in range(S.N)],
                    "at + i mod 2^DL for i < N")


def _ob(M_: int) -> int:
    return M_.bit_length()


def _sig_offsets(S):
    return (("m", Array(S.N, Array(S.MB, BIT))),), Array(S.N * _ob(S.M) + 1, BIT)


def _build_offsets(C, S, *ms):
    ob = _ob(S.M)
    sw = max(ob, (S.N * ((1 << S.MB) - 1)).bit_length())
    acc, out = [ZERO] * sw, []
    for m in ms:
        out += acc[:ob]
        acc = add(C, acc, list(m), width=sw)
    return out + [sub(C, const_bits(S.M, sw), acc, sw)[1].inv()]


N3Offsets = _part("N3Offsets", ("MB", "N", "M"), _sig_offsets, _build_offsets,
                  "each frame's pool offset, the m's before it (bitlen(M) bits each), then whether all N m's fit M")


def _sig_scatter(S):
    return (("ids", Array(S.MF, Array(S.IDW, BIT))), ("m", Array(S.MB, BIT)), ("o", Array(_ob(S.M), BIT))), \
        Array(S.M, Array(S.IDW, BIT))


def _build_scatter(C, S, *e):
    ids, m, o = e[:S.MF], e[S.MF], e[S.MF + 1]
    ge = _thresholds(C, m)
    cur = [[C.AND(ge(j + 1), x) for x in ids[j]] for j in range(min(S.MF, S.M))]
    cur += [[ZERO] * S.IDW for _ in range(S.M - len(cur))]
    for k, bit in enumerate(o):
        sh = 1 << k
        cur = [[x ^ C.AND(bit, x ^ y) for x, y in zip(cur[q], cur[q - sh] if q >= sh else [ZERO] * S.IDW)]
               for q in range(S.M)]
    return cur


N3Scatter = _part("N3Scatter", ("MF", "M", "IDW", "MB"), _sig_scatter, _build_scatter,
                  "a frame's slot identities placed in the pool: slot j < m at pool row o + j, every other row zero")


def _sig_rootat(S):
    f = SimpleNamespace(U=len(S.DEPTHS), DB=max(1, *S.DEPTHS))
    idw = 1 + max(1, (f.U - 1).bit_length()) + f.DB
    return ((("id", Array(idw, BIT)), ("row", ROW), ("roots", Array(f.U, DIGEST)),
             ("levels", Array(f.DB + 1, DIGEST))), BIT)


def _build_rootat(C, S, idb, row, *rest):
    U = len(S.DEPTHS)
    ub = max(1, (U - 1).bit_length())
    roots = [rest[8 * k:8 * k + 8] for k in range(U)]
    levels = [rest[8 * U + 8 * k:8 * U + 8 * k + 8] for k in range((len(rest) - 8 * U) // 8)]
    e = _onehot(C, idb[1:1 + ub], U)
    hit = ZERO
    for k in range(U):
        hit = hit ^ C.AND(e[k], _eq(C, [x for w in levels[S.DEPTHS[k]] for x in w], [x for w in roots[k] for x in w]))
    zero = and_reduce(C, [x.inv() for x in row])
    return [zero ^ C.AND(idb[0], hit ^ zero)]


N3RootAt = _part("N3RootAt", ("DEPTHS",), _sig_rootat, _build_rootat,
                 "kind 1: the read's level D_u is unit u's root (0 for a u past the units); kind 0: the row is zero")


def _sig_pick(S):
    return (("u", Array(max(1, (S.U - 1).bit_length()), BIT)), ("table", Array(S.U, DIGEST))), DIGEST


def _build_pick(C, S, u, *table):
    e = _onehot(C, u, S.U)
    out = []
    for w in range(8):
        word = [ZERO] * 64
        for k in range(S.U):
            word = [a ^ C.AND(e[k], b) for a, b in zip(word, table[8 * k + w])]
        out.append(word)
    return out


N3Pick = _part("N3Pick", ("U",), _sig_pick, _build_pick, "table[u], or zero for a u past the table")


def _sig_slot(S):
    return (("slot", Array(max(1, (S.N - 1).bit_length()), BIT)),), Array(2 * S.N, BIT)


def _build_slot(C, S, slot):
    ge = _thresholds(C, slot)
    return [ge(j + 1).inv() for j in range(S.N)] + _onehot(C, slot, S.N)


N3Slot = _part("N3Slot", ("N",), _sig_slot, _build_slot, "slot <= j for j < N, then slot == j for j < N")


def _sig_salteq(S):
    return (("frame", Array(8 * S.B, BIT)), ("salt", Array(SALT_PAYLOAD, BYTE))), BIT


def _build_salteq(C, S, frame, *salt):
    head = HEADER.pack(SALT, SALT_STREAM, SALT_PAYLOAD)
    want = [x for b in head for x in const_bits(b, 8)] + [x for byte in salt for x in byte]
    want += [ZERO] * (8 * S.B - len(want))
    return [_eq(C, frame, want)]


N3SaltEq = _part("N3SaltEq", ("B",), _sig_salteq, _build_salteq,
                 "1 iff the frame is the salt frame of the payload salt (the window and the salt's commitment)")


# --- the circuit: composite parts -------------------------------------------------------------------------------------

class _Consts:
    """The constant bits a body reads, one node each."""

    def __init__(self, B):
        self.B, self.ref = B, {}

    def bits(self, value: int, w: int) -> list:
        for b in sorted({value >> j & 1 for j in range(w)} - self.ref.keys()):
            self.ref[b] = self.B.call(BL.constant(b)).refs[0]
        return [self.ref[value >> j & 1] for j in range(w)]


def _refs(c: Coll) -> list:
    return [c.refs[k] for k in range(c.refs.count)]


def _coll(t, refs: list) -> Coll:
    return Coll(t, TR._refs(list(refs)))


def _cat(t, *colls: Coll) -> Coll:
    return Coll(t, concat([c.refs for c in colls]))


def _words(data: Sequence[Sequence]) -> list[list]:
    """Bytes (8 bits each, least significant first) as SHA-512 words: 64 bits least significant first, bytes
    big-endian."""
    return [[x for b in range(8) for x in data[8 * k + 7 - b]] for k in range(len(data) // 8)]


def _bytes(words: Sequence[Sequence]) -> list[list]:
    return [list(w[8 * (7 - b):8 * (8 - b)]) for w in words for b in range(8)]


def _sha512(B, K: _Consts, cv: Sequence[int], data: list[list]) -> Coll:
    """Sha512Compress calls from the chaining value `cv` over `data` (byte refs, whole blocks)."""
    words = _words(data)
    out = _coll(DIGEST, [x for v in cv for x in K.bits(v, 64)])
    for k in range(0, len(words), 16):
        out = B.call(Sha512Compress, out, _coll(Array(16, WORD), [x for w in words[k:k + 16] for x in w]))
    return out


def _sig_rowdigest(S):
    return (("row", Array(S.N_BITS, BIT)),), DIGEST


def _body_rowdigest(B, S, row):
    K = _Consts(B)
    nbytes = -(-S.N_BITS // 8)
    bits = _refs(row) + K.bits(0, 8 * nbytes - S.N_BITS)
    data = [bits[8 * k:8 * k + 8] for k in range(nbytes)] + [K.bits(b, 8) for b in pad_tail(128 + nbytes)]
    return _sha512(B, K, midstate(sha512_row_v2_prefix(S.N_BITS)), data)


N3RowDigest = CompositeDefinition(
    "N3RowDigest", 1, ("N_BITS",), _sig_rowdigest, _body_rowdigest, conformance="exact-model-tested", register=False,
    doc="the sha512/row/v2 digest of an N_BITS-bit row: SHA-512 of its 128-byte prefix block and the row's bytes")


def _sig_hread(S):
    return (("row", Array(S.N_BITS, BIT)), ("salt", SALT_T), ("root", DIGEST), ("domain", DIGEST),
            ("pos", Array(S.D, BIT)), ("path", Array(S.D, DIGEST))), BIT


def _body_hread(B, S, row, salt, root, domain, pos, path):
    x = B.call(bind(N3RowDigest, N_BITS=S.N_BITS), row)
    leaf = B.call(Hm96Sha512Leaf, x, salt)[2]
    return B.call(bind(FrameV3Read, D=S.D, SCHEMA=SCHEMA), root, domain, leaf, pos, path)


N3HidingRead = CompositeDefinition(
    "N3HidingRead", 1, ("N_BITS", "D"), _sig_hread, _body_hread, conformance="exact-model-tested", register=False,
    doc="1 iff the N_BITS-bit row with its salt opens at pos in the depth-D advice tree (hm96-sha512/row/v2 leaves) of "
        "root and domain id domain")


def _sig_rowread(S):
    f = SimpleNamespace(U=len(S.DEPTHS), DB=max(1, *S.DEPTHS))
    idw = 1 + max(1, (f.U - 1).bit_length()) + f.DB
    return (("id", Array(idw, BIT)), ("row", ROW), ("salt", SALT_T), ("path", Array(f.DB, DIGEST)),
            ("roots", Array(f.U, DIGEST)), ("domains", Array(f.U, DIGEST))), BIT


def _body_rowread(B, S, idb, row, salt, path, roots, domains):
    U, DB = len(S.DEPTHS), max(1, *S.DEPTHS)
    ub = max(1, (U - 1).bit_length())
    K = _Consts(B)
    ib = _refs(idb)
    pos = ib[1 + ub:1 + ub + DB]
    x = B.call(bind(N3RowDigest, N_BITS=V.ROW_BITS), row)
    value = B.call(Hm96Sha512Leaf, x, salt)[2]
    domain = B.call(bind(N3Pick, U=U), _coll(Array(ub, BIT), ib[1:1 + ub]), domains)
    nb = index_bytes(DB)
    cur = B.call(bind(FrameV3Leaf, B=nb, SCHEMA=SCHEMA), domain, value,
                 _coll(Array(8 * nb, BIT), pos + K.bits(0, 8 * nb - DB)))
    levels = [cur]
    for k in range(DB):
        hi = pos[k + 1:]
        n = index_bytes(len(hi))
        cur = B.call(bind(FrameV3Node, B=n), domain, _coll(LEVEL, K.bits(k, 8)),
                     _coll(Array(8 * n, BIT), hi + K.bits(0, 8 * n - len(hi))), _coll(BIT, [pos[k]]), cur, path[k])
        levels.append(cur)
    return B.call(bind(N3RootAt, DEPTHS=S.DEPTHS), idb, row, roots, _cat(Array(DB + 1, DIGEST), *levels))


N3RowRead = CompositeDefinition(
    "N3RowRead", 1, ("DEPTHS",), _sig_rowread, _body_rowread, conformance="exact-model-tested", register=False,
    doc="1 iff a pool row is what its slot identity names: kind 1, the row with its salt opens at position pos in unit "
        "u's value tree (depth DEPTHS[u]); kind 0, the row is zero")


def _sig_gather(S):
    return (("pool", Array(S.M, ROW)), ("o", Array(_ob(S.M), BIT)), ("m", Array(S.MB, BIT))), Array(S.MF, ROW)


def _body_gather(B, S, pool, o, m):
    K = _Consts(B)
    mux, gate = bind(N3Mux, W=V.ROW_BITS), bind(N3Gate, W=V.ROW_BITS)
    cur, n = pool, S.M
    for k in reversed(range(_ob(S.M))):
        sh = 1 << k
        n_out = min(n, S.MF + sh - 1)
        a = max(0, min(n_out, n - sh))
        parts = []
        if a:
            parts.append(B.batch(mux, cur[0:a], cur[sh:sh + a], o[k], axes=(0, 0, None)))
        if n_out > a:
            parts.append(B.batch(gate, cur[a:n_out], B.call(BL.NOT, o[k]), axes=(0, None)))
        cur, n = _cat(Array(n_out, ROW), *parts), n_out
    if n < S.MF:
        cur = _cat(Array(S.MF, ROW), cur, _coll(Array(S.MF - n, ROW), K.bits(0, V.ROW_BITS * (S.MF - n))))
    elif n > S.MF:
        cur = cur[0:S.MF]
    below = B.call(bind(N3Below, MF=S.MF, MB=S.MB), m)
    return B.batch(gate, cur, below, axes=(0, 0))


N3Gather = CompositeDefinition(
    "N3Gather", 1, ("MF", "M", "MB"), _sig_gather, _body_gather, conformance="exact-model-tested", register=False,
    doc="a frame's rows: row j is pool row o + j if j < m and o + j < M, else zero")


def _sig_frame(S):
    f = _format(S.FMT)
    return (("descriptor", Array(f.LW, BIT)), ("rows", Array(f.MF, ROW)), ("frame", Array(f.B, BYTE))), BIT


def _body_frame(B, S, desc, rows, frame):
    f = _format(S.FMT)
    keep = Coll(Array(f.MF, ROW), desc.refs.slice(f.KEEP_OFF, f.LW))
    local = B.batch(N3Local, keep, rows, axes=(0, 0))
    lr = local.refs
    cell = 10 * V.ROW_BITS
    last = Coll(Array(f.MF, Array(9, BIT)),
                concat([lr.slice(j * cell + cell - 9, (j + 1) * cell) for j in range(f.MF)]))
    bases = B.call(bind(N3Bases, MF=f.MF, Z=f.Z), last)
    cells = B.batch(bind(N3AddBase, Z=f.Z), local, bases, axes=(0, 0))
    nb = 8 * f.B
    cur, n = cells.reshape(Array(f.W, Array(f.Z + 1, BIT))), f.W
    for t in range(f.Z):
        sh, k = 1 << t, f.Z - t
        n_out = min(n, nb + (1 << f.Z) - (1 << (t + 1)))
        a = max(0, min(n_out, n - sh))
        parts = []
        if a:
            parts.append(B.batch(bind(N3Step, K=k), cur[0:a], cur[sh:sh + a], axes=(0, 0)))
        if n_out > a:
            parts.append(B.batch(bind(N3Stay, K=k), cur[a:n_out], axes=(0,)))
        cur, n = _cat(Array(n_out, Array(k, BIT)), *parts), n_out
    stream = Coll(Array(min(n, nb), BIT), cur.refs.slice(0, min(n, nb)))
    if n < nb:
        stream = _cat(Array(nb, BIT), stream, _coll(Array(nb - n, BIT), _Consts(B).bits(0, nb - n)))
    return B.call(bind(N3Place, FMT=S.FMT), Coll(Array(f.HW, BIT), desc.refs.slice(0, f.HW)), stream, frame)


N3Frame = CompositeDefinition(
    "N3Frame", 1, ("FMT",), _sig_frame, _body_frame, conformance="exact-model-tested", register=False,
    doc="1 iff the frame is what the descriptor puts on the rows: the kept bits compacted into the stream (per slot, "
        "the unkept bits before each bit; then shifts by their binary digits), placed by N3Place")


# --- the window -------------------------------------------------------------------------------------------------------

def _window_params(S) -> list[tuple[str, object]]:
    f = _format(S.FMT)
    if S.SALTED not in (0, 1) or not isinstance(S.N, int) or isinstance(S.N, bool) or S.N < 1:
        raise ValueError(f"N3Window_v2 takes N >= 1 frames and SALTED 0 or 1, not N={S.N!r}, SALTED={S.SALTED!r}")
    if not isinstance(S.M, int) or S.M < 1 or S.M & (S.M - 1):
        raise ValueError(f"M, the pool's size class, is a power of two, not {S.M!r}")
    if S.SALTED and f.B < HEADER.size + SALT_PAYLOAD:
        raise ValueError(f"a salt frame takes {HEADER.size + SALT_PAYLOAD} bytes, more than B = {f.B}")
    n = S.N - S.SALTED
    params = [("layout", DIGEST), ("layout_domain", DIGEST), ("at", Array(f.DL, BIT)), ("record", DIGEST),
              ("record_domain", DIGEST), ("direction", Array(1, BIT)), ("roots", Array(f.U, DIGEST)),
              ("domains", Array(f.U, DIGEST))]
    if S.SALTED:
        params.append(("salt", Array(SALT_PAYLOAD, BYTE)))
    params.append(("frames", Array(S.N, Array(f.B, BYTE))))
    if n:
        params += [("descriptors", Array(n, Array(f.LW, BIT))), ("descriptor_salts", Array(n, SALT_T)),
                   ("descriptor_paths", Array(n, Array(f.DL, DIGEST)))]
    params += [("rows", Array(S.M, ROW)), ("row_salts", Array(S.M, SALT_T)),
               ("row_paths", Array(S.M, Array(f.DB, DIGEST))), ("record_salt", SALT_T),
               ("record_sibling", Array(1, DIGEST))]
    if S.SALTED:
        params.append(("slot", Array(max(1, (S.N - 1).bit_length()), BIT)))
    return params


def _sig_window(S):
    return tuple(_window_params(S)), BIT


def _body_window(B, S, *args):
    f = _format(S.FMT)
    N, Mp, n = S.N, S.M, S.N - S.SALTED
    a = dict(zip([k for k, _ in _window_params(S)], args))
    K = _Consts(B)
    oks: list[Coll] = []
    frames = a["frames"]
    data = frames
    if S.SALTED:
        flat = frames.reshape(Array(N, Array(8 * f.B, BIT)))
        slot = B.call(bind(N3Slot, N=N), a["slot"])
        picked = B.batch(bind(N3Gate, W=8 * f.B), flat, slot[N:2 * N], axes=(0, 0))
        chosen = B.call(bind(N3Xor, N=N, W=8 * f.B), picked) if N > 1 else picked[0]
        oks.append(B.call(bind(N3SaltEq, B=f.B), chosen, a["salt"]))
        if n:
            routed = B.batch(bind(N3Mux, W=8 * f.B), flat[0:n], flat[1:N], slot[0:n], axes=(0, 0, 0))
            data = routed.reshape(Array(n, Array(f.B, BYTE)))
    ob = _ob(Mp)
    if n:
        d = a["descriptors"]
        pos = B.call(bind(N3Positions, DL=f.DL, N=n), a["at"])
        oks.append(B.batch(bind(N3HidingRead, N_BITS=f.LW, D=f.DL), d, a["descriptor_salts"], a["layout"],
                           a["layout_domain"], pos, a["descriptor_paths"], axes=(0, 0, None, None, 0, 0)))
        ms = Coll(Array(n, Array(f.MB, BIT)),
                  concat([d.refs.slice(i * f.LW + 2, i * f.LW + 2 + f.MB) for i in range(n)]))
        offsets = B.call(bind(N3Offsets, MB=f.MB, N=n, M=Mp), ms)
        os_ = offsets[0:n * ob].reshape(Array(n, Array(ob, BIT)))
        oks.append(offsets[n * ob])
        ids = Coll(Array(n, Array(f.MF, Array(f.IDW, BIT))),
                   concat([d.refs.slice(i * f.LW + f.HW, i * f.LW + f.HW + f.MF * f.IDW) for i in range(n)]))
        placed = B.batch(bind(N3Scatter, MF=f.MF, M=Mp, IDW=f.IDW, MB=f.MB), ids, ms, os_, axes=(0, 0, 0))
        if n > 1:
            pool_ids = B.call(bind(N3Xor, N=n, W=Mp * f.IDW), placed.reshape(Array(n, Array(Mp * f.IDW, BIT))))
            pool_ids = pool_ids.reshape(Array(Mp, Array(f.IDW, BIT)))
        else:
            pool_ids = placed[0]
    else:
        pool_ids = _coll(Array(Mp, Array(f.IDW, BIT)), K.bits(0, Mp * f.IDW))
    oks.append(B.batch(bind(N3RowRead, DEPTHS=f.DEPTHS), pool_ids, a["rows"], a["row_salts"], a["row_paths"],
                       a["roots"], a["domains"], axes=(0, 0, 0, 0, None, None)))
    if n:
        got = B.batch(bind(N3Gather, MF=f.MF, M=Mp, MB=f.MB), a["rows"], os_, ms, axes=(None, 0, 0))
        oks.append(B.batch(bind(N3Frame, FMT=S.FMT), d, got, data, axes=(0, 0, 0)))
    fb = _refs(frames)
    msg = [K.bits(b, 8) for b in prefix("")] + [fb[i:i + 8] for i in range(0, len(fb), 8)]
    msg += [K.bits(b, 8) for b in pad_tail(len(msg))]
    digest = _sha512(B, K, IV, msg)
    words = _refs(digest)
    row = _coll(Array(512, BIT), [x for byte in _bytes([words[64 * w:64 * w + 64] for w in range(8)]) for x in byte])
    oks.append(B.call(bind(N3HidingRead, N_BITS=512, D=1), row, a["record_salt"], a["record"], a["record_domain"],
                      a["direction"], a["record_sibling"]))
    bits = concat([o.refs for o in oks])
    return B.call(BL.and_gate(bits.count), Coll(Array(bits.count, BIT), bits))


N3Window = CompositeDefinition(
    "N3Window", 2, ("FMT", "N", "M", "SALTED"), _sig_window, _body_window, conformance="exact-model-tested",
    register=False,
    doc="1 iff the descriptors open at at, at + 1, ... in the layout, their slots fit the pool and name its rows, "
        "every pool row opens in its unit (or is zero), every frame (the salt frame's slot skipped) is its "
        "descriptor's over its rows, the salt frame is the salt's, and the record digest of the frames opens at "
        "direction in the record commitment")


def unit(f: Format, N: int, M_: int, salted: bool):
    return bind(N3Window, FMT=tuple(f), N=N, M=M_, SALTED=int(salted))


# --- the reference and the arguments ----------------------------------------------------------------------------------

@dataclass
class Instance:
    """One window's unit and its arguments as bytes and ints, by parameter name (paths as lists of digests)."""

    fmt: Format
    N: int
    M: int
    salted: bool
    args: dict

    @property
    def unit(self):
        return unit(self.fmt, self.N, self.M, self.salted)

    @property
    def key(self) -> tuple:
        return (tuple(self.fmt), self.N, self.M, self.salted)


def reference(inst: Instance) -> bool:
    """`N3Window_v2`'s output on the instance, computed from its arguments as the circuit does."""
    f, a, N, Mp = inst.fmt, inst.args, inst.N, inst.M
    n = N - inst.salted
    frames = list(a["frames"])
    if inst.salted:
        slot = a["slot"]
        head = HEADER.pack(SALT, SALT_STREAM, SALT_PAYLOAD)
        if slot >= N or frames[slot] != (head + a["salt"]).ljust(f.B, b"\0"):
            return False
        frames = frames[:slot] + frames[slot + 1:]
    descs = a.get("descriptors", [])
    for i in range(n):
        value = hiding_value(descs[i], f.LW, a["descriptor_salts"][i])
        pos = (a["at"] + i) % (1 << f.DL)
        if root_ref(a["layout_domain"], SCHEMA, value, pos, a["descriptor_paths"][i]) != a["layout"]:
            return False
    xs = [fields(f, x) for x in descs]
    offsets, total = [], 0
    for x in xs:
        offsets.append(total)
        total += x.m
    if total > Mp:
        return False
    ids = [(0, 0, 0)] * Mp
    for x, o in zip(xs, offsets):
        for j in range(min(x.m, f.MF)):
            if o + j < Mp:
                ids[o + j] = x.ids[j]
    for q, (kind, u, pos) in enumerate(ids):
        row = a["rows"][q]
        if kind == 0:
            if any(row):
                return False
            continue
        if u >= f.U:
            return False
        value = hiding_value(row, V.ROW_BITS, a["row_salts"][q])
        if root_ref(a["domains"][u], SCHEMA, value, pos, a["row_paths"][q][:f.DEPTHS[u]]) != a["roots"][u]:
            return False
    zero = bytes(V.ROW_BITS // 8)
    for i, (x, o) in enumerate(zip(xs, offsets)):
        rows = [a["rows"][o + j] if j < x.m and o + j < Mp else zero for j in range(f.MF)]
        if x.mode not in (RAW, B64) or assemble(f, descs[i], rows) != frames[i]:
            return False
    value = hiding_value(record_digest(a["frames"]), 512, a["record_salt"])
    return root_ref(a["record_domain"], SCHEMA, value, a["direction"], [a["record_sibling"]]) == a["record"]


def _digest_bits(d: bytes) -> list[int]:
    return [(w >> j) & 1 for w in words_be(d) for j in range(64)]


def _int_bits(x: int, w: int) -> list[int]:
    return [(x >> j) & 1 for j in range(w)]


def _bytes_bits(data: bytes, nbits: int) -> list[int]:
    return _int_bits(int.from_bytes(data, "little"), nbits)


def arguments(inst: Instance) -> list[np.ndarray]:
    """The unit's arguments for one instance in its signature's order, each (1, leaves) of 0/1."""
    f, a = inst.fmt, inst.args
    zero_digest = bytes(64)
    flat: dict[str, list[int]] = {
        "layout": _digest_bits(a["layout"]), "layout_domain": _digest_bits(a["layout_domain"]),
        "at": _int_bits(a["at"], f.DL), "record": _digest_bits(a["record"]),
        "record_domain": _digest_bits(a["record_domain"]), "direction": [a["direction"]],
        "roots": [b for r in a["roots"] for b in _digest_bits(r)],
        "domains": [b for r in a["domains"] for b in _digest_bits(r)],
        "frames": [b for fr in a["frames"] for b in _bytes_bits(fr, 8 * f.B)],
        "rows": [b for r in a["rows"] for b in _bytes_bits(r, V.ROW_BITS)],
        "row_salts": [b for s in a["row_salts"] for b in _digest_bits(s)],
        "row_paths": [b for p in a["row_paths"] for s in list(p) + [zero_digest] * (f.DB - len(p))
                      for b in _digest_bits(s)],
        "record_salt": _digest_bits(a["record_salt"]), "record_sibling": _digest_bits(a["record_sibling"]),
    }
    if inst.salted:
        flat["salt"] = _bytes_bits(a["salt"], 8 * SALT_PAYLOAD)
        flat["slot"] = _int_bits(a["slot"], max(1, (inst.N - 1).bit_length()))
    if inst.N - inst.salted:
        flat["descriptors"] = [b for x in a["descriptors"] for b in _bytes_bits(x, f.LW)]
        flat["descriptor_salts"] = [b for s in a["descriptor_salts"] for b in _digest_bits(s)]
        flat["descriptor_paths"] = [b for p in a["descriptor_paths"] for s in p for b in _digest_bits(s)]
    params, _ = inst.unit.signature()
    out = []
    for name, t in params:
        if len(flat[name]) != t.leaves:
            raise ValueError(f"argument {name} has {len(flat[name])} bits, the unit takes {t.leaves}")
        out.append(np.asarray(flat[name], dtype=np.uint8).reshape(1, -1))
    return out


def evaluate(*insts: Instance) -> list[int]:
    """The unit's output on each instance, all of one unit, in one evaluation."""
    if len({i.key for i in insts}) != 1:
        raise ValueError("the instances are of one unit")
    args = [np.concatenate(x) for x in zip(*(arguments(i) for i in insts))]
    return [int(b) for b in np.asarray(evaluate_bits(insts[0].unit, *args)).reshape(-1)]


_GROUPS = {"N3RowRead": "row_reads", "N3Gather": "gather", "N3Frame": "frames", "Sha512Compress": "record_digest"}


def _times(nd) -> int:
    return 1 if nd.form in ("primitive", "call") else nd.n


def _ands(fn, memo: dict) -> int:
    key = fn.id
    if key not in memo:
        if isinstance(fn, PrimitiveDefinition):
            kind = BL.gate_kind(fn)
            memo[key] = kind[1] - 1 if kind is not None and kind[0] == "and" else 0
        else:
            memo[key] = sum(_times(nd) * _ands(nd.fn, memo) for nd in fn.body.nodes)
    return memo[key]


def size(inst_or_unit, memo: dict | None = None) -> dict:
    """The window unit's ANDs, in total and by part: row reads, descriptor reads, the record's opening, the frames, the
    rows' gather, the record digest and the rest (the salt, the offsets, the slot identities, the final AND)."""
    target = inst_or_unit.unit if isinstance(inst_or_unit, Instance) else inst_or_unit
    memo = {} if memo is None else memo
    out = dict.fromkeys(("row_reads", "descriptor_reads", "record_opening", "frames", "gather", "record_digest",
                         "other"), 0)
    for nd in target.body.nodes:
        name = getattr(nd.fn, "name", "")
        group = _GROUPS.get(name, "other")
        if name == "N3HidingRead":
            group = "record_opening" if nd.form == "call" else "descriptor_reads"
        out[group] += _times(nd) * _ands(nd.fn, memo)
    out["ands"] = sum(out.values())
    return out


# --- registration -----------------------------------------------------------------------------------------------------

def _salts(seed: bytes, n: int) -> list[bytes]:
    return [hashlib.shake_256(seed + i.to_bytes(8, "big")).digest(hm96.SHA512.salt_bytes) for i in range(n)]


@dataclass
class Tree:
    """A registered advice tree with its levels kept, so a path is a lookup."""

    reg: advice.Registration
    levels: tuple = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.levels = tree_levels(advice.domain(self.reg.binding, len(self.reg.rows)), self.reg.leaves)

    @property
    def root(self) -> bytes:
        return self.reg.root

    @property
    def domain_id(self) -> bytes:
        return advice.domain(self.reg.binding, len(self.reg.rows)).domain_id

    @property
    def depth(self) -> int:
        return merkle_depth(len(self.reg.rows))

    def path(self, i: int) -> list[bytes]:
        return [level[(i >> k) ^ 1] for k, level in enumerate(self.levels[:-1])]


def register(rows: Sequence[bytes], n_bits: int, binding: bytes, seed: bytes) -> Tree:
    return Tree(advice.register(rows, n_bits, binding, _salts(seed, len(rows))))


def layout_binding(program: bytes, placement: bytes) -> bytes:
    return hashlib.sha512(LAYOUT_LABEL + program + placement).digest()


def record_binding(program: bytes, link: str, window: int) -> bytes:
    return hashlib.sha512(RECORD_LABEL + program + link.encode() + b"\0" + window.to_bytes(8, "big")).digest()


def register_record(program: bytes, link: str, window: int, digests: Sequence[bytes], seed: bytes) -> Tree:
    """The certifiers' record commitment for (link, window): the two directions' record digests, 512-bit hiding
    rows."""
    return register(digests, 512, record_binding(program, link, window), seed + b"/record/" + link.encode()
                    + window.to_bytes(8, "big"))


@dataclass
class UnitTable:
    """The units in canonical order: names, value-tree depths, roots (zero for a unit that committed none), domain ids,
    and each committed unit's tree."""

    names: list[str]
    depths: tuple
    roots: list[bytes]
    domains: list[bytes]
    trees: dict[str, Tree]

    def index(self, key: tuple[str, int]) -> tuple[int, int]:
        return self.names.index(key[0]), key[1]


@dataclass
class Layout:
    """The prover's layout: the format, the ports in order, each port's capacity, its declared frames' descriptors and
    pieces, and the advice tree of every position's descriptor."""

    fmt: Format
    ports: list[str]
    cap: int
    descriptors: dict[str, list[Descriptor]]
    tree: Tree

    def position(self, port: str, j: int) -> int:
        return self.ports.index(port) * self.cap + j

    def descriptor(self, port: str, j: int) -> Descriptor:
        d = self.descriptors.get(port, [])
        return d[j] if j < len(d) else NONE


def register_layout(declared: dict[str, list[list[tuple]]], ports: list[str], B: int, units: UnitTable, program: bytes,
                    placement: bytes, row: Callable[[tuple], bytes], seed: bytes) -> Layout:
    """Describe every declared frame, fit the format, and register the layout: one advice tree, CAP positions a port."""
    descs = {p: [describe(pieces, B, units.index) for pieces in declared.get(p, [])] for p in ports}
    most = max([len(v) for v in descs.values()] + [1])
    cap = 1 << max(1, most.bit_length())
    fmt = fit([d for v in descs.values() for d in v], B, units.depths, max(1, merkle_depth(len(ports) * cap)))
    zero = bytes(V.ROW_BITS // 8)
    for p in ports:
        for pieces, d in zip(declared.get(p, []), descs[p]):
            rows = [row(units.names[u], pos) if kind else zero for kind, u, pos in d.ids]
            if assemble(fmt, encode(fmt, d), rows) != expand(pieces, B, lambda k: row(*k)):
                raise AssertionError(f"{p}: a descriptor doesn't put its declared frame on its rows")
    rows = [encode(fmt, descs[p][j] if j < len(descs[p]) else NONE) for p in ports for j in range(cap)]
    tree = register(rows, fmt.LW, layout_binding(program, placement), seed + b"/layout")
    return Layout(fmt, ports, cap, descs, tree)


# --- the statement ----------------------------------------------------------------------------------------------------

@dataclass
class Statement:
    """What the layout is derived from: the Program, the hardware, φ, the placement, every unit's values (by the
    placement) and the units' committed values."""

    model: M.Model
    hw: Hardware
    phi: F.Phi
    placement: P.Placement
    units: dict[str, V.UnitValues]
    gates: dict[str, tuple[int, ...]]
    program: bytes
    placement_root: bytes

    def rows(self, unit: str) -> int:
        return max(1, -(-len(self.gates[unit]) // (V.ROW_BITS // self.model.value_bits)))

    def locate(self, unit: str, gate: int) -> tuple[int, int]:
        """(row, bit offset) of a value the unit holds by the placement, whether or not the unit committed it."""
        width = self.model.value_bits
        i = V._index(self.gates[unit], gate)
        return i // (V.ROW_BITS // width), (i % (V.ROW_BITS // width)) * width

    def row(self, unit: str, pos: int) -> bytes:
        """The unit's committed row, or the zero row for a unit that committed none."""
        u = self.units.get(unit)
        return u.registration.rows[pos] if u is not None and pos < len(u.registration.rows) else bytes(V.ROW_BITS // 8)

    def message(self, m: F.Message) -> list[tuple]:
        """The message's bits as pieces: each crossing value's bits in its source unit's row, then the last byte's
        zeros."""
        out = []
        for e in m.edges:
            row, o = self.locate(e.src, e.value)
            out.append(("r", (e.src, row), o, e.width))
        return out + [("k", -m.bits % 8, 0)]

    def table(self) -> UnitTable:
        names, depths, roots, domains, trees = [], [], [], [], {}
        for u in self.hw.unit_names():
            n = self.rows(u)
            names.append(u)
            depths.append(merkle_depth(n))
            domains.append(advice.domain(V.binding(u), n).domain_id)
            if u in self.units:
                trees[u] = Tree(self.units[u].registration)
                roots.append(trees[u].root)
            else:
                roots.append(bytes(64))
        return UnitTable(names, tuple(depths), roots, domains, trees)


def statement(public: dict, witness: dict) -> Statement:
    """The statement a run's public and witness files give, each unit's rows registered against its public root."""
    from . import n3

    report = n3.Report()
    model = M.build(public.get("workload", "mlp"), public["spec"])
    hw = Hardware.from_json(public["hardware"])
    placement = n3.placement_of(public, witness, model, hw, report)
    if placement is None or report.violations:
        raise ValueError(f"the placement doesn't hold: {report.violations[:1]}")
    edges = P.crossings(model, placement, hw)
    gates = {u: tuple(sorted(placement.at(hw.leaves_of(u)))) for u in hw.unit_names()}
    units = {}
    for u, pub in public["values"].items():
        w = witness["values"][u]
        reg = advice.register([bytes.fromhex(r) for r in w["rows"]], V.ROW_BITS, V.binding(u),
                              [bytes.fromhex(s) for s in w["salts"]])
        if reg.root.hex() != pub["root"] or pub["count"] != len(gates[u]):
            raise ValueError(f"unit {u}'s rows don't register to its public root")
        units[u] = V.UnitValues(u, gates[u], model.value_bits, reg)
    return Statement(model, hw, F.phi(hw, edges, public["frame_bytes"]), placement, units, gates,
                     bytes.fromhex(public.get("program", "")), bytes.fromhex(public.get("placement", "")))


def _data_frame(kind: int, stream: int, payload: list[tuple], n: int, B: int) -> list[tuple]:
    return [const(HEADER.pack(kind, stream, n)), *payload, zeros(B - HEADER.size - n)]


def sim_declared(S: Statement) -> dict[str, list[tuple[int, list[tuple]]]]:
    """Per certified link direction, φ's frames as (bucket, pieces) in grid order: send order, stably by bucket."""
    phi, B = S.phi, S.phi.frame_bytes
    room = B - HEADER.size
    streams = phi.by_stream()
    out: dict[str, list[tuple[int, list[tuple]]]] = {}
    for key, slots in phi.slots().items():
        frames = []
        for s in slots:
            if s.kind == "hello":
                frames.append((s.bucket, [const(F.hello(S.hw, s.src, s.dst, B))]))
                continue
            m = streams[s.stream]
            message = S.message(m)
            nbytes, last = (m.bits + 7) // 8, s.index == F.frame_count(m, B) - 1
            lo, hi = s.index * room, min((s.index + 1) * room, nbytes)
            frames.append((s.bucket, _data_frame(LAST if last else DATA, m.stream, _slice(message, 8 * lo, 8 * hi),
                                                 hi - lo, B)))
        out[key] = sorted(frames, key=lambda x: x[0])
    return out


def lab_declared(S: Statement, block: str) -> tuple[list, list]:
    """The block's egress frames (its events in position order) and ingress frames (its requests in session order) as
    pieces."""
    from . import lab as L

    B = S.phi.frame_bytes
    lab = L.Lab(S.model, S.hw, S.placement, S.phi.messages, B)
    messages = {stream: S.message(m) for stream, m in lab.messages.items()}
    egress = []
    for p, (j, m, c) in enumerate(lab.events[block]):
        nbytes = (m.bits + 7) // 8
        lo, hi = c * lab.chunk, min((c + 1) * lab.chunk, nbytes)
        head = b"id: %d\nevent: %d\ndata: " % (p, m.stream)
        n = len(head) + 4 * -(-(hi - lo) // 3) + 2
        payload = [const(head), ("b", tuple(_slice(messages[m.stream], 8 * lo, 8 * hi)), hi - lo), const(b"\n\n")]
        egress.append(_data_frame(DATA, j + 1, payload, n, B))
    ingress = []
    room = B - HEADER.size
    for s in lab.sessions[block]:
        n_frames = -(-lab.request_length(s) // room)
        body = sum((m.bits + 7) // 8 for m in s.inbound)
        request = [const(L.REQUEST.pack(s.index, body))]
        for m in s.inbound:
            request += messages[m.stream]
        total = L.REQUEST.size + body
        for k in range(n_frames):
            lo, hi = k * room, min((k + 1) * room, total)
            ingress.append(_data_frame(LAST if k == n_frames - 1 else DATA, s.stream, _slice(request, 8 * lo, 8 * hi),
                                       hi - lo, B))
    return egress, ingress


# --- windows ----------------------------------------------------------------------------------------------------------

EMPTY = "nothing recorded"


@dataclass
class Window:
    """One port-window: the frames the record holds, its public counts and status, the revealed salt commitment of a
    salted window, and the unit's instance; `why` says why no unit runs (refused from public data alone, or `EMPTY`)."""

    port: str
    window: int
    frames: list[bytes]
    counts: tuple[int, ...] = ()
    reason: str = ""
    salt: bytes | None = None
    instance: Instance | None = None
    why: str = ""

    def check(self, circuit: bool = True) -> dict:
        out = {"port": self.port, "window": self.window, "recorded": len(self.frames)}
        if self.instance is None:
            return out | {"by": "public data", "ok": self.why == EMPTY and not self.frames, "why": self.why}
        inst = self.instance
        ref = reference(inst)
        out |= {"by": "unit", "ok": ref, "N": inst.N, "M": inst.M, "salted": inst.salted}
        if circuit:
            t0 = time.perf_counter()
            (out["circuit"],) = evaluate(inst)
            out["seconds"] = round(time.perf_counter() - t0, 3)
            if out["circuit"] != ref:
                raise AssertionError(f"{self.port}[{self.window}]: the circuit says {out['circuit']}, "
                                     f"the reference {ref}")
        return out


@dataclass
class Setup:
    """A run's statement, units, registered layout and the ports' links: what every window's instance is built from."""

    S: Statement
    units: UnitTable
    layout: Layout
    links: dict[str, tuple[str, int]]
    seed: bytes

    @classmethod
    def build(cls, S: Statement, declared: dict[str, list[list[tuple]]], ports: list[str],
              links: dict[str, tuple[str, int]], seed: bytes = b"n3-unit") -> "Setup":
        units = S.table()
        layout = register_layout(declared, ports, S.phi.frame_bytes, units, S.program, S.placement_root, S.row, seed)
        return cls(S, units, layout, links, seed)

    @property
    def fmt(self) -> Format:
        return self.layout.fmt

    def attach(self, windows: list[Window]) -> list[Window]:
        """Each window's instance, its port's earlier frames giving its first descriptor's position; or why none
        runs."""
        digests: dict[tuple[str, int, int], bytes] = {}
        for w in windows:
            link, d = self.links[w.port]
            digests[(link, w.window, d)] = record_digest(w.frames, w.reason)
        trees: dict[tuple[str, int], Tree] = {}
        seen: dict[str, int] = defaultdict(int)
        for w in sorted(windows, key=lambda w: (self.layout.ports.index(w.port), w.window)):
            n = len(w.frames) - (w.salt is not None and len(w.frames) > 0)
            a = seen[w.port]
            seen[w.port] += n
            if not w.frames:
                w.why = EMPTY
            elif w.reason:
                w.why = f"the record's status is {w.reason}"
            elif a + n > self.layout.cap:
                w.why = f"frames past the port's capacity, {self.layout.cap}"
            else:
                link, d = self.links[w.port]
                if (link, w.window) not in trees:
                    pair = [digests.get((link, w.window, k), record_digest([])) for k in (0, 1)]
                    trees[(link, w.window)] = register_record(self.S.program, link, w.window, pair, self.seed)
                w.instance = self.instance(w, a, n, trees[(link, w.window)], d)
        return windows

    def instance(self, w: Window, a: int, n: int, record: Tree, direction: int) -> Instance:
        f, L, U = self.fmt, self.layout, self.units
        positions = [L.position(w.port, a + i) for i in range(n)]
        descs = [L.descriptor(w.port, a + i) for i in range(n)]
        zero_row, zero_salt = bytes(V.ROW_BITS // 8), bytes(hm96.SHA512.salt_bytes)
        rows, salts, paths = [], [], []
        for d in descs:
            for kind, u, pos in d.ids:
                tree = U.trees.get(U.names[u]) if kind else None
                if tree is None:
                    rows.append(zero_row)
                    salts.append(zero_salt)
                    paths.append([])
                else:
                    rows.append(tree.reg.rows[pos])
                    salts.append(tree.reg.salts[pos])
                    paths.append(tree.path(pos))
        Mp = 1 << max(0, (len(rows) - 1).bit_length()) if rows else 1
        rows += [zero_row] * (Mp - len(rows))
        salts += [zero_salt] * (Mp - len(salts))
        paths += [[]] * (Mp - len(paths))
        args = {"layout": L.tree.root, "layout_domain": L.tree.domain_id, "at": positions[0] if n else
                L.position(w.port, min(a, L.cap - 1)), "record": record.root, "record_domain": record.domain_id,
                "direction": direction, "roots": U.roots, "domains": U.domains, "frames": w.frames,
                "descriptors": [L.tree.reg.rows[p] for p in positions],
                "descriptor_salts": [L.tree.reg.salts[p] for p in positions],
                "descriptor_paths": [L.tree.path(p) for p in positions],
                "rows": rows, "row_salts": salts, "row_paths": paths, "record_salt": record.reg.salts[direction],
                "record_sibling": record.path(direction)[0]}
        salted = w.salt is not None
        if salted:
            args["salt"] = w.window.to_bytes(8, "big") + w.salt
            args["slot"] = next((i for i, fr in enumerate(w.frames) if _is_salt(fr)), 0)
        return Instance(f, len(w.frames), Mp, salted, args)


def sim_ports(hw: Hardware) -> dict[str, tuple[str, int]]:
    """Every certified link direction's key, with its link and the end it delivers to."""
    out = {}
    for e, link in enumerate(hw.links):
        if hw.certified(e):
            for d, to in enumerate(link.ends):
                out[hw.key(e, to)] = (link.name, d)
    return out


def sim_windows(public: dict, witness: dict, records: dict[str, list[tuple[int, bytes]]], buckets: int,
                seed: bytes = b"n3-unit") -> tuple[Setup, list[Window]]:
    """The simulator's windows: per certified link direction, `buckets` buckets a window, the recorded frames in grid
    order (stably by bucket), against the layout of φ's declared frames, registered before the run."""
    S = statement(public, witness)
    links = sim_ports(S.hw)
    declared = sim_declared(S)
    setup = Setup.build(S, {k: [p for _, p in v] for k, v in declared.items()}, list(links), links, seed)
    windows = []
    for key in links:
        rec = sorted(records.get(key, []), key=lambda x: x[0])
        last = max([b for b, _ in rec] + [-1])
        for w in range(last // buckets + 1):
            lo, hi = w * buckets, (w + 1) * buckets
            frames = [f for b, f in rec if lo <= b < hi]
            windows.append(Window(key, w, frames, tuple(sum(1 for b, _ in rec if b == t) for t in range(lo, hi))))
    return setup, setup.attach(windows)


def lab_windows(public: dict, witness: dict, published: dict[str, Sequence],
                salts: dict[tuple[str, int], bytes] | None = None,
                seed: bytes = b"n3-unit") -> tuple[Setup, list[Window]]:
    """The lab's windows: per block and link (`lab.EGRESS`, `lab.INGRESS`), each record's released frames, against the
    layout of the declared frames. With `salts`, each salted ingress window's revealed salt commitment by (block,
    window): one of its frames is then the certifier's salt frame, and the rest are the port's next declared frames."""
    from verity.core.protocols.certifier.grid import occupancy

    from . import lab as L

    S = statement(public, witness)
    blocks = sorted(S.hw.units)
    links = {f"{b}:{k}": (b, d) for b in blocks for d, k in enumerate((L.EGRESS, L.INGRESS))}
    declared = {}
    for b in blocks:
        declared[f"{b}:{L.EGRESS}"], declared[f"{b}:{L.INGRESS}"] = lab_declared(S, b)
    setup = Setup.build(S, declared, list(links), links, seed)
    windows = []
    for block in sorted(published):
        for link in (L.EGRESS, L.INGRESS):
            for rec in sorted((r for r in published[block] if r.link == link), key=lambda r: r.window):
                salt = (salts or {}).get((block, rec.window)) if link == L.INGRESS else None
                windows.append(Window(f"{block}:{link}", rec.window, list(L.released([rec])), occupancy(rec.grid),
                                      rec.status.reason.value if rec.status.reason else "", salt))
    return setup, setup.attach(windows)


# --- the demo ---------------------------------------------------------------------------------------------------------

DEMO_FRAMES = [
    [const(encode_frame(LAST, F.HELLO_STREAM, F.HELLO.pack(0, 1), 16))],
    [const(HEADER.pack(DATA, 1, 3)), ("r", ("u0", 0), 3, 13), ("r", ("u1", 0), 250, 6), ("k", 5, 0), zeros(4)],
    [const(HEADER.pack(DATA, 1, 6)), ("b", (("r", ("u1", 0), 0, 8), ("r", ("u0", 2), 100, 8)), 2), const(b"\n\n"),
     zeros(1)],
    [const(HEADER.pack(DATA, 2, 3)), ("r", ("u0", 1), 10, 5), ("k", 3, 0), ("r", ("u0", 1), 200, 16), zeros(4)],
]
"""Four frames of every kind, B = 16, on port p0: a HELLO-like constant frame, a frame of two units' row bits, an
event-like frame of base64, and a frame reading one row twice around a zero constant."""


@dataclass
class Demo:
    units: UnitTable
    layout: Layout
    record: Tree
    window: Window


def demo(seed: bytes = b"n3-demo") -> Demo:
    """A small run of two units (three rows, depth 2; one row, depth 0) and two ports, and p0's window of DEMO_FRAMES:
    N = 4, seven pool rows (M = 8)."""
    B = 16
    rows = {u: [hashlib.shake_256(seed + u.encode() + bytes([i])).digest(32) for i in range(k)]
            for u, k in (("u0", 3), ("u1", 1))}
    trees = {u: register(r, V.ROW_BITS, V.binding(u), seed + u.encode()) for u, r in rows.items()}
    units = UnitTable(["u0", "u1"], tuple(t.depth for t in trees.values()), [t.root for t in trees.values()],
                      [t.domain_id for t in trees.values()], trees)
    layout = register_layout({"p0": DEMO_FRAMES}, ["p0", "p1"], B, units, b"demo", b"",
                             lambda u, pos: rows[u][pos], seed)
    frames = [expand(p, B, lambda k: rows[k[0]][k[1]]) for p in DEMO_FRAMES]
    setup = Setup(None, units, layout, {"p0": ("demo", 0), "p1": ("demo", 1)}, seed)  # type: ignore[arg-type]
    record = register_record(b"demo", "demo", 0, [record_digest(frames), record_digest([])], seed)
    window = Window("p0", 0, frames)
    window.instance = setup.instance(window, 0, len(frames), record, 0)
    return Demo(units, layout, record, window)


# --- the command ------------------------------------------------------------------------------------------------------

def summarize(results: list[dict]) -> dict:
    units = [r for r in results if r["by"] == "unit"]
    refused = [f"{r['port']}[{r['window']}]: {r['why']}" for r in results if not r["ok"] and r["by"] != "unit"]
    return {"ok": all(r["ok"] for r in results), "windows": len(results), "units": len(units),
            "units_ok": sum(r["ok"] for r in units), "refused_without_a_unit": refused,
            "not_ok": [f"{r['port']}[{r['window']}]" for r in results if not r["ok"]],
            "ands_max": max((r.get("ands", 0) for r in units), default=0),
            "seconds": round(sum(r.get("seconds", 0) for r in units), 3)}


def _sized(windows: list[Window], circuit: bool) -> list[dict]:
    memo: dict = {}
    out = []
    for w in windows:
        if w.why == EMPTY:
            continue
        r = w.check(circuit=circuit)
        if w.instance is not None:
            r |= size(w.instance, memo)
        out.append(r)
    return out


def _sim(args) -> dict:
    from .block import Context
    from .n3 import merge
    from .run import run_memory
    from .run import statement as run_statement

    config = {"workload": "mlp", "spec": {"d": args.d, "layers": args.layers, "requests": args.requests, "groups": 1},
              "blocks": args.blocks, "devices": args.devices, "frame_bytes": args.frame_bytes, "seed": args.seed}
    if args.tamper:
        unit_, _, mode = args.tamper.partition(":")
        config["tamper"] = {"unit": unit_, "mode": mode}
    ctx = Context(config)
    results = run_memory(ctx)
    public, witness = run_statement(ctx, results)
    records, _ = merge(results)
    setup, windows = sim_windows(public, witness, records, args.buckets)
    checked = _sized(windows, not args.reference_only)
    return {"config": config, "buckets": args.buckets, "format": setup.fmt._asdict(), "summary": summarize(checked),
            "windows": checked}


def _lab(args) -> dict:
    run = Path(args.run)
    here = Path(__file__).resolve().parents[1]
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import records  # noqa: PLC0415  (puts network_traces on the path, for active_replay)

    import active_replay as ar  # noqa: PLC0415

    from . import lab as L

    public = json.loads((run / "public.json").read_text())
    witness = json.loads((run / "witness.json").read_text())
    B = public["frame_bytes"]
    published = {}
    for block in sorted(p.name for p in run.iterdir() if (p / f"{L.EGRESS}.records").exists()):
        published[block] = [ar.unpack_record(x, B) for f in (L.EGRESS, L.INGRESS)
                            for x in ar.read_blobs(run / block / f"{f}.records")]
        for r in published[block]:
            reason = r.status.reason.value if r.status.reason else ""
            if records.digest(r) != record_digest(list(L.released([r])), reason):
                raise AssertionError(f"records.digest and record_digest differ on {block} {r.link}[{r.window}]")
    setup, windows = lab_windows(public, witness, published)
    results = _sized(windows, not args.reference_only)
    return {"run": str(run), "format": setup.fmt._asdict(), "summary": summarize(results), "windows": results}


def _circuit_check(args) -> dict:
    """circuit-check's checks on the demo's window and its SHA-free parts, with the CLI's options and pins. The
    window's C-Flock lowering is its parts' (`lower_whole=False`): lowered whole it would carry the SHA-512 compression
    that `checks.UNLOWERED` records as not fitting in 64 GiB. The unit is a Program of its own, never a served
    Program's Call, so the partition invariant doesn't apply (`as_call=False`)."""
    from circuit_check import checks as CK

    d = demo()
    f, inst = d.layout.fmt, d.window.instance
    opts = CK.Options()
    entries = []
    for target, whole in ((bind(N3Frame, FMT=tuple(f)), True), (bind(N3Gather, MF=f.MF, M=inst.M, MB=f.MB), True),
                          (bind(N3Scatter, MF=f.MF, M=inst.M, IDW=f.IDW, MB=f.MB), True), (inst.unit, False)):
        t0 = time.time()
        try:
            r = CK.check_definition(target, opts, lower_whole=whole, as_call=False)
        except Exception as e:  # noqa: BLE001  (a target the checks can't run on is a failure by name, as the CLI's)
            r = {"definition": target.id, "failures": [{"name": f"error/raised {target.id}: {type(e).__name__}: {e}"}],
                 "warnings": []}
        entries.append({**r, "kind": "definition", "target": target.id, "seconds": round(time.time() - t0, 3)})
    rep = CK.report(entries, opts)
    rep["size"] = size(inst)
    rep["format"] = f._asdict()
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub_ = ap.add_subparsers(dest="cmd", required=True)
    s = sub_.add_parser("sim", help="serve the MLP on the simulator in one process and run the unit on every window")
    s.add_argument("--tamper", help="UNIT:MODE, MODE one of extra-frame, flip-bit, fill")
    s.add_argument("--buckets", type=int, default=4, help="buckets a window")
    s.add_argument("--frame-bytes", type=int, default=64)
    s.add_argument("--blocks", type=int, default=3)
    s.add_argument("--devices", type=int, default=2)
    s.add_argument("--d", type=int, default=4)
    s.add_argument("--layers", type=int, default=3)
    s.add_argument("--requests", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=_sim)
    q = sub_.add_parser("lab", help="run the unit on every window a lab run's certifiers recorded")
    q.add_argument("run", help="the run's directory (lab_run's --out)")
    q.set_defaults(fn=_lab)
    c = sub_.add_parser("circuit-check", help="circuit-check's checks on the demo's window and its parts")
    c.set_defaults(fn=_circuit_check)
    for p in (s, q):
        p.add_argument("--reference-only", action="store_true", help="the Python reference alone, no circuit")
    for p in (s, q, c):
        p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    result = args.fn(args)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result.get("summary", {k: result.get(k) for k in ("ok", "size")}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
