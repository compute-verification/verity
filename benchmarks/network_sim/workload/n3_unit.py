"""N3's proof unit for one port-window, zero-knowledge: the frames a certifier recorded on one link direction in one
window are the committed crossing values, framed as the layout committed before the run says. A Boolean Definition,
which a proof unit of the proof service runs; its one output is the ok bit.

    N3Window_v3{FMT, N, G, SALTED, REC}
        (layout, layout_domain: Digest, at: Array<DL>, record, record_domain: Digest, direction: Array<1>,
         roots, domains: Array<U, Digest>, [salt: Array<72, Byte>],
         frames: Array<N, Array<B, Byte>>,
         descriptors: Array<n, Array<LW>>, descriptor_salts: Array<n, Salt>,
         descriptor_paths: Array<n, Array<DL, Digest>>,
         rows: Array<n, G, Row>, row_salts: Array<n, G, Salt>, row_paths: Array<n, G, Array<DB, Digest>>,
         head: Array<HL, Byte>, record_salt: Salt, record_path: Array<2, Digest>, [slot: Array<SB>]) -> Value<1>
        Digest = Array<8, Array<64, Value<1>>> (SHA-512 words, `sha512.words_be`),
        Salt = Array<24, Array<64, Value<1>>>, Byte = Array<8, Value<1>>, Row = Array<2B, Byte> (an outbox row),
        n = N - SALTED, SB = max(1, bitlen(N - 1)), HL = 93 + L + 4T

**Statics: the size class.** `FMT` (`Format`) is the run's descriptor format: the frame size B, the widest prefix P
and suffix Q, the most segments a frame reads, each unit's outbox-tree depth (`DEPTHS`, U units) and the layout tree's
depth DL. N is the window's recorded frame count, G the segments its port's frames read (at most the format's),
SALTED whether one of the frames is the certifier's salt frame. `REC` (`Shape`) is the certified record's shape: the
UTF-8 length L of the link's name as its certifier writes it, and the grid's T buckets of r slots, fixed before the run
(N <= T r). Nothing else about the layout or the record is static.

**Public inputs.** `layout` is the layout's root, `layout_domain` its domain id, and `at` the position of the window's
first descriptor: the port's index times the port's capacity, plus the frames the port recorded in earlier windows
(less their salt frames), all public occupancy. `record` is the root of N2's commitment to the link-window's certified
records (`LinkWindows`, `verity.core.protocols.certifier.service`), `record_domain` its domain id (one-stage's served
domain of the run, the link and the window) and `direction` the port's end of the link (0 egress, 1 ingress). `roots`
and `domains` are the units' outbox commitments and domain ids, `salt` the window number (u64, big-endian) and the
revealed salt commitment.

**Witness.** The frames, the n descriptors (the layout's rows at at, at + 1, ...) with their hiding salts and paths, for
each descriptor's segment g < G the outbox row it reads with its salt and path, the record head's bytes before its
frames root (`RecordHead.encode`) with its leaf's salt and two siblings in N2's tree, and the salt frame's slot.

**What 1 means.** (1) Each descriptor opens at its position in the layout (`N3HidingRead`: its sha512/row/v2 digest,
hm96's hiding leaf, the frame-v3-sha512 read). (2) Each segment g < G of each descriptor reads one row: of kind 1, row q
of unit u's outbox, which opens there in its commitment; of kind 0, a zero row (`N3RowRead`). Every segment from G on is
of kind 0. (3) Frame i (the salt frame's slot skipped) is what descriptor
i puts on its segments' rows (`N3Frame`, `assemble`). (4) The salt frame, when SALTED, is `salt_frame` at the slot. (5)
The head's bytes and N2's frames root of the N frames (`certificate.frames_root`, depth D = bitlen(T r - 1)) are a
head whose status byte is 0 (complete), and that head opens at leaf 2 + direction of N2's commitment (`N3HidingRead`,
depth 2), the leaf N2 commits beside each certified record (`service.HEADS`). The head's other fields are N2's to
check: N2's accepted outcome on the same root says that leaf holds the certified head of (link, direction, window).

**A descriptor** (`Descriptor`, LW bits, `encode`): a mode (RAW 1, BASE64 2; 0 and 3 are invalid, and `NONE` pads the
tree), the text's byte offset p, its data length r, the suffix offset s, the prefix (P bytes) and suffix (Q bytes), and
per segment an identity (kind, unit u, row q), a source offset o < 2^OB, a length n and a destination d. Segment g puts
bytes o ... o + n - 1 of row q (2B bytes) at bytes d ... d + n - 1 of a B-byte stream (`N3Seg`), and the
stream is the segments' XOR. The frame is prefix XOR text << 8p XOR suffix << 8s over B bytes, the data the stream's
first min(r, B) bytes, and the text the data (RAW) or its base64 (BASE64; `assemble`).

**The outbox** (`outbox_rows`): a unit's outgoing messages in φ's send order (`Phi.send_order`), back to back, each
message the bytes `framing.encode` packs its crossing values into. Its rows overlap: row q holds bytes
qB ... qB + 2B - 1 (zero past the end) for q < k = max(1, ceil(bytes / B)), and zero rows follow up to a power of two.
A message is then one contiguous range of its source's outbox, and any run of at most B of its bytes lies in one row,
the row of the run's first byte, so a frame reads it as one segment, one row.

**Commitments and who registers them.** Each producing unit registers its outbox (`register_outbox`), beside its values,
under the binding SHA-512(`OUTBOX_LABEL` || program || unit). Its domain id makes its row count public, 2^depth with
depth = bitlen(k - 1): that is all it discloses of the unit's crossing bytes. That the outbox holds the bytes of the
unit's committed values, each byte the same in every row that holds it, is not proved here; it needs a proof unit of
its own. The prover registers the layout before the run (`register_layout`): one advice tree of the descriptors, one
per (port, frame index) at port * CAP + index, under the binding SHA-512(`LAYOUT_LABEL` || program || placement
root). The record commitment is N2's, which the firewall registers (`LinkWindows.commit`, `registration`): its root
alone is public, and the heads, certificates and salts stay behind the firewall. Here `Certifier` stands in for the
certifier and the firewall: a small LMS key signs each head in one certifier session, and the four salts come from a
seed. The network_sim README ("N3 in zero knowledge") has the sizes and the open questions.

    python -m workload.n3_unit sim --out FILE [--tamper UNIT:MODE] [--buckets K] [--frame-bytes B]
    python -m workload.n3_unit lab RUN_DIR --out FILE     every window a lab run's certifiers recorded (lab_run --out)
    python -m workload.n3_unit circuit-check --out FILE [--parts]   circuit-check on `demo()`'s window and its parts
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, NamedTuple, Sequence

import numpy as np

from verity.core.protocols.certifier import certificate as CT
from verity.core.protocols.certifier import service as N2
from verity.core.protocols.certifier.active import DATA, HEADER, LAST, encode_frame
from verity.core.protocols.certifier.grid import COMPLETE, GridParams, Reason, Status, fill
from verity.core.protocols.certifier.schedule import SyncSet
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
from verity.ml.commitments.gates.sha512 import midstate, pad_tail, words_be
from verity.ml.commitments.merkle import merkle_depth, tree_levels
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.ml.commitments.rowleaf import sha512_row_v2_digest, sha512_row_v2_prefix
from verity.ml.crypto import lms

from . import framing as F
from . import model as M
from . import placement as P
from . import values as V
from .hardware import Hardware

BIT = BL.BIT
BYTE = Array(8, BIT)
WORD = Array(64, BIT)
DIGEST = Array(8, WORD)
SALT_T = Array(hm96.SHA512.salt_bytes // 8, WORD)
LEVEL = Array(8, BIT)
LAYOUT_LABEL = b"verity/netsim/n3-layout/v1/"
OUTBOX_LABEL = b"verity/netsim/n3-outbox/v1/"
RAW, B64 = 1, 2
B64_ALPHABET = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
# What base64 adds to a 6-bit value v, mod 256, for v below 26, below 52, below 62, 62 and 63.
B64_OFFSETS = (65, 71, 252, 237, 240)

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
# byte 0. ("k", w, v): w constant bits, v's; ("r", key, o, w): bits o ... o + w - 1 of the bytes `key` names (a unit's
# outbox, (unit,)); ("b", pieces, L): the base64 text (RFC 4648, padded) of the L bytes `pieces` give, 32 ceil(L / 3)
# bits.

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
    """The run's descriptor format, a static of the unit: frame bytes, prefix and suffix bytes, segments, the units'
    outbox depths, the layout tree's depth. The derived widths and the field offsets (bits, least significant first)
    follow; a segment is (kind 1, u UB, q DB, o OB, n LB, d LB)."""

    B: int
    P: int
    Q: int
    G: int
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
    def PB(self) -> int:
        return self.P.bit_length()

    @property
    def LB(self) -> int:
        return self.B.bit_length()

    @property
    def OB(self) -> int:
        return (self.B - 1).bit_length()

    @property
    def IDW(self) -> int:
        return 1 + self.UB + self.DB

    @property
    def SW(self) -> int:
        return self.IDW + self.OB + 2 * self.LB

    @property
    def P_OFF(self) -> int:
        return 2

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
    def LW(self) -> int:
        return self.HW + self.G * self.SW


def _format(fmt) -> Format:
    try:
        f = Format(*fmt)
    except TypeError:
        raise ValueError(f"FMT is (B, P, Q, G, DEPTHS, DL), not {fmt!r}") from None
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (f.B, f.P, f.Q, f.G, f.DL)):
        raise ValueError(f"FMT's B, P, Q, G and DL are ints, not {fmt!r}")
    if f.B <= HEADER.size or not 1 <= f.P <= f.B or not 1 <= f.Q <= f.B or f.G < 1 or not 1 <= f.DL <= 64:
        raise ValueError(f"a format has B > {HEADER.size}, 1 <= P, Q <= B, G >= 1 and 1 <= DL <= 64, not {fmt!r}")
    depths = f.DEPTHS
    if not isinstance(depths, tuple) or not depths or any(not isinstance(d, int) or not 0 <= d <= 63 for d in depths):
        raise ValueError(f"DEPTHS is a nonempty tuple of depths 0 ... 63, not {f.DEPTHS!r}")
    return f


class Shape(NamedTuple):
    """The certified record's shape, a static of the unit: the UTF-8 length L of the link's name in the head, and the
    window's T buckets of R slots. The head's bytes before its frames root (HL), its status byte's offset (ST) and the
    frames tree's depth D follow (`RecordHead.encode`, `certificate.depth`)."""

    L: int
    T: int
    R: int

    @property
    def HL(self) -> int:
        return 93 + self.L + 4 * self.T

    @property
    def ST(self) -> int:
        return 27 + self.L

    @property
    def D(self) -> int:
        return (self.T * self.R - 1).bit_length()


def _shape(rec) -> Shape:
    try:
        s = Shape(*rec)
    except TypeError:
        raise ValueError(f"REC is (L, T, R), not {rec!r}") from None
    if (not all(isinstance(x, int) and not isinstance(x, bool) for x in s) or not 0 <= s.L < 1 << 16
            or min(s.T, s.R) < 1):
        raise ValueError(f"a record's shape has 0 <= L < 2^16, T >= 1 and R >= 1, not {rec!r}")
    return s


class Segment(NamedTuple):
    """A run of n bytes of a unit's outbox: kind 1 reads unit u's row q (kind 0, a zero row), and bytes o ... o + n - 1
    of it go to bytes d ... d + n - 1 of the frame's stream."""

    kind: int = 0
    u: int = 0
    q: int = 0
    o: int = 0
    n: int = 0
    d: int = 0


@dataclass(frozen=True)
class Descriptor:
    """One frame's layout: mode, text offset p, data bytes r, suffix offset s, prefix, suffix, and its segments."""

    mode: int = 0
    p: int = 0
    r: int = 0
    s: int = 0
    prefix: bytes = b""
    suffix: bytes = b""
    segs: tuple = ()


NONE = Descriptor()


def encode(f: Format, d: Descriptor) -> bytes:
    """The descriptor as a layout row: LW bits, least significant first (`rowleaf.bit_row`'s bytes)."""
    if (len(d.segs) > f.G or len(d.prefix) > f.P or len(d.suffix) > f.Q or not 0 <= d.mode < 4
            or not 0 <= d.p < 1 << f.PB or not 0 <= d.r < 1 << f.LB or not 0 <= d.s < 1 << f.LB):
        raise ValueError(f"the descriptor doesn't fit the format {f}")
    x = d.mode | d.p << f.P_OFF | d.r << f.R_OFF | d.s << f.S_OFF
    x |= int.from_bytes(d.prefix, "little") << f.PRE_OFF | int.from_bytes(d.suffix, "little") << f.SUF_OFF
    for g, (kind, u, q, o, n, dst) in enumerate(d.segs):
        if (kind not in (0, 1) or not 0 <= u < 1 << f.UB or not 0 <= q < 1 << f.DB or not 0 <= o < 1 << f.OB
                or not 0 <= n < 1 << f.LB or not 0 <= dst < 1 << f.LB):
            raise ValueError(f"segment {g} doesn't fit the format")
        v = kind | u << 1 | q << (1 + f.UB) | o << f.IDW | n << (f.IDW + f.OB) | dst << (f.IDW + f.OB + f.LB)
        x |= v << (f.HW + g * f.SW)
    return x.to_bytes(-(-f.LW // 8), "little")


class Fields(NamedTuple):
    """A layout row's fields, whatever its bits: every segment of the format."""

    mode: int
    p: int
    r: int
    s: int
    prefix: int
    suffix: int
    segs: tuple


def fields(f: Format, row: bytes) -> Fields:
    x = int.from_bytes(row, "little")

    def get(off: int, w: int) -> int:
        return (x >> off) & ((1 << w) - 1)

    segs = []
    for g in range(f.G):
        at = f.HW + g * f.SW
        segs.append(Segment(get(at, 1), get(at + 1, f.UB), get(at + 1 + f.UB, f.DB), get(at + f.IDW, f.OB),
                            get(at + f.IDW + f.OB, f.LB), get(at + f.IDW + f.OB + f.LB, f.LB)))
    return Fields(get(0, 2), get(f.P_OFF, f.PB), get(f.R_OFF, f.LB), get(f.S_OFF, f.LB), get(f.PRE_OFF, 8 * f.P),
                  get(f.SUF_OFF, 8 * f.Q), tuple(segs))


def decode(f: Format, row: bytes) -> Descriptor:
    """The descriptor a layout row holds (its segments up to the last nonzero one; the prefix and suffix without
    trailing zero bytes)."""
    x = fields(f, row)
    segs = list(x.segs)
    while segs and segs[-1] == Segment():
        segs.pop()
    return Descriptor(x.mode, x.p, x.r, x.s, x.prefix.to_bytes(f.P, "little").rstrip(b"\0"),
                      x.suffix.to_bytes(f.Q, "little").rstrip(b"\0"), tuple(segs))


def segment(B: int, seg: Sequence[int], row: bytes) -> bytes:
    """The B-byte stream one segment puts its 2B-byte row on: `N3Seg`'s reference."""
    _, _, _, o, n, d = seg
    out = bytearray(B)
    for j in range(d, min(B, d + n)):
        if o + j - d < 2 * B:
            out[j] = row[o + j - d]
    return bytes(out)


def assemble(f: Format, row: bytes, rows: Sequence[bytes], G: int | None = None) -> bytes:
    """The frame the layout row puts on `rows` (each of its first G segments' row, zero past those given): `N3Frame`'s
    reference, less its check that the segments from G on are of kind 0."""
    x = fields(f, row)
    stream = 0
    for g in range(f.G if G is None else G):
        src = rows[g] if g < len(rows) else bytes(2 * f.B)
        stream ^= int.from_bytes(segment(f.B, x.segs[g], src), "little")
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


def describe(pieces: Sequence[tuple], B: int, index: Callable[[tuple], int]) -> Descriptor:
    """The descriptor of a frame declared as pieces, each outbox piece's key named by its unit index `index(key)`.

    The frame's constant bits give the prefix (bytes before p) and the suffix (bytes from s, trailing zeros dropped);
    the non-constant pieces, one base64 piece (BASE64) or outbox pieces (RAW), give the text between. Each run of one
    outbox, whole bytes, is a segment; a zero constant inside the text is a gap no segment writes, and a nonzero one has
    no descriptor."""
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
    if first % 8 or last % 8:
        raise ValueError("a frame's text is whole bytes")
    p = first // 8
    inside = [q for q, start in walk if start >= first and start + width(q) <= last]
    if len(inside) == 1 and inside[0][0] == "b":
        mode, r, run = B64, inside[0][2], list(inside[0][1])
        length = 4 * -(-r // 3)
    else:
        if any(q[0] == "b" for q in inside):
            raise ValueError("a base64 text shares its frame with other outbox bytes")
        mode, r, run = RAW, (last - first) // 8, inside
        length = r
    s = p + length
    if s > B or (data >> 8 * p) & ((1 << 8 * length) - 1):
        raise ValueError("the text's bytes hold nonzero constant bits")
    segs: list[Segment] = []
    at = 0
    for q in _merge(run):
        if q[0] == "r":
            if at % 8 or q[2] % 8 or q[3] % 8:
                raise ValueError("a segment is whole bytes of its outbox")
            o = q[2] // 8
            segs.append(Segment(1, index(q[1]), o // B, o % B, q[3] // 8, at // 8))
        elif q[0] == "k" and q[2]:
            raise ValueError("a nonzero constant inside a frame's text")
        elif q[0] == "b":
            raise ValueError("a base64 text inside a base64 text")
        at += width(q)
    return Descriptor(mode, p, r, s, raw[:p], raw[s:].rstrip(b"\0"), tuple(segs))


def fit(descriptors: Sequence[Descriptor], B: int, depths: Sequence[int], DL: int) -> Format:
    """The narrowest format that holds the descriptors."""
    return _format((B, max([1] + [max(d.p, len(d.prefix)) for d in descriptors]),
                    max([1] + [len(d.suffix) for d in descriptors]), max([1] + [len(d.segs) for d in descriptors]),
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


def _seg_widths(B: int) -> tuple[int, int]:
    return (B - 1).bit_length(), B.bit_length()


def _sig_seg(S):
    ob, lb = _seg_widths(S.B)
    return ((("row", Array(16 * S.B, BIT)), ("o", Array(ob, BIT)), ("n", Array(lb, BIT)), ("d", Array(lb, BIT))),
            Array(8 * S.B, BIT))


def _build_seg(C, S, row, o, n, d):
    ob, lb = _seg_widths(S.B)
    K, ew = 1 << ob, max(ob, lb) + 1
    # Byte j of the window [K, K + B) of the row shifted left by e = K + d - o bytes is byte o + j - d of the row;
    # K >= B > o keeps e nonnegative.
    e = sub(C, add(C, const_bits(K, ew), list(d), width=ew), list(o), ew)[0]
    moved = shl_window(C, list(row), [ZERO] * 3 + e, 8 * K, 8 * K + 8 * S.B)
    ge_d, ge_end = _thresholds(C, d), _thresholds(C, add(C, list(d), list(n), width=lb + 1))
    out = []
    for j in range(S.B):
        inside = C.AND(ge_d(j + 1).inv(), ge_end(j + 1))
        out += [C.AND(inside, x) for x in moved[8 * j:8 * j + 8]]
    return out


N3Seg = _part("N3Seg", ("B",), _sig_seg, _build_seg,
              "the B-byte stream of one segment: bytes o ... o + n - 1 of its 2B-byte row at bytes d ... d + n - 1, "
              "every other byte zero")


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


def _sig_positions(S):
    return (("at", Array(S.DL, BIT)),), Array(S.N, Array(S.DL, BIT))


N3Positions = _part("N3Positions", ("DL", "N"), _sig_positions,
                    lambda C, S, at: [add(C, list(at), const_bits(i, S.DL), width=S.DL) for i in range(S.N)],
                    "at + i mod 2^DL for i < N")


def _idw(depths: tuple) -> tuple[int, int, int, int]:
    """U, DB, UB and IDW of a tuple of unit depths, as `Format` derives them."""
    U, DB = len(depths), max(1, *depths)
    UB = max(1, (U - 1).bit_length())
    return U, DB, UB, 1 + UB + DB


def _sig_rootat(S):
    U, DB, _, idw = _idw(S.DEPTHS)
    return ((("id", Array(idw, BIT)), ("row", Array(S.N_BITS, BIT)), ("roots", Array(U, DIGEST)),
             ("levels", Array(DB + 1, DIGEST))), BIT)


def _build_rootat(C, S, idb, row, *rest):
    U, _, ub, _ = _idw(S.DEPTHS)
    roots = [rest[8 * k:8 * k + 8] for k in range(U)]
    levels = [rest[8 * U + 8 * k:8 * U + 8 * k + 8] for k in range((len(rest) - 8 * U) // 8)]
    e = _onehot(C, idb[1:1 + ub], U)
    hit = ZERO
    for k in range(U):
        hit = hit ^ C.AND(e[k], _eq(C, [x for w in levels[S.DEPTHS[k]] for x in w], [x for w in roots[k] for x in w]))
    zero = and_reduce(C, [x.inv() for x in row])
    return [zero ^ C.AND(idb[0], hit ^ zero)]


N3RootAt = _part("N3RootAt", ("DEPTHS", "N_BITS"), _sig_rootat, _build_rootat,
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


N3IsZero = _part("N3IsZero", ("W",), lambda S: ((("x", Array(S.W, BIT)),), BIT),
                 lambda C, S, x: [and_reduce(C, [b.inv() for b in x])], "1 iff the W bits are all zero")


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


def _digest_bytes(d: Coll) -> list[list]:
    w = _refs(d)
    return _bytes([w[64 * k:64 * k + 64] for k in range(8)])


def _sig_froot(S):
    return (("frames", Array(S.N, Array(S.B, BYTE))),), DIGEST


def _body_froot(B, S, frames):
    K = _Consts(B)
    fb = _refs(frames)
    leaf_cv = midstate((CT.FRAME_TAG + S.B.to_bytes(4, "big")).ljust(128, b"\0"))
    node_cv = midstate(CT.NODE_TAG.ljust(128, b"\0"))
    leaf_pad = [K.bits(b, 8) for b in pad_tail(128 + S.B)]
    node_pad = [K.bits(b, 8) for b in pad_tail(256)]
    level = [_sha512(B, K, leaf_cv, [fb[8 * k:8 * k + 8] for k in range(i * S.B, (i + 1) * S.B)] + leaf_pad)
             for i in range(S.N)]
    for lv in range(S.D):
        if len(level) % 2:
            level.append(_coll(DIGEST, [x for w in words_be(CT.empty(lv)) for x in K.bits(w, 64)]))
        level = [_sha512(B, K, node_cv, _digest_bytes(level[i]) + _digest_bytes(level[i + 1]) + node_pad)
                 for i in range(0, len(level), 2)]
    return level[0]


N3FramesRoot = CompositeDefinition(
    "N3FramesRoot", 1, ("B", "N", "D"), _sig_froot, _body_froot, conformance="exact-model-tested", register=False,
    doc="N2's frames root of N B-byte frames in a depth-D tree (`certificate.frames_root`): each frame's tagged leaf, "
        "each level's odd node paired with the empty subtree's root E_level")


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
    U, DB, _, idw = _idw(S.DEPTHS)
    return (("id", Array(idw, BIT)), ("row", Array(S.N_BITS, BIT)), ("salt", SALT_T), ("path", Array(DB, DIGEST)),
            ("roots", Array(U, DIGEST)), ("domains", Array(U, DIGEST))), BIT


def _body_rowread(B, S, idb, row, salt, path, roots, domains):
    U, DB, ub, _ = _idw(S.DEPTHS)
    K = _Consts(B)
    ib = _refs(idb)
    pos = ib[1 + ub:1 + ub + DB]
    x = B.call(bind(N3RowDigest, N_BITS=S.N_BITS), row)
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
    return B.call(bind(N3RootAt, DEPTHS=S.DEPTHS, N_BITS=S.N_BITS), idb, row, roots,
                  _cat(Array(DB + 1, DIGEST), *levels))


N3RowRead = CompositeDefinition(
    "N3RowRead", 1, ("DEPTHS", "N_BITS"), _sig_rowread, _body_rowread, conformance="exact-model-tested",
    register=False,
    doc="1 iff an N_BITS-bit row is what its identity names: kind 1, the row with its salt opens at position q in unit "
        "u's tree (depth DEPTHS[u]); kind 0, the row is zero")


def _sig_frame(S):
    f = _format(S.FMT)
    if not isinstance(S.G, int) or not 1 <= S.G <= f.G:
        raise ValueError(f"G is 1 ... the format's {f.G} segments, not {S.G!r}")
    return (("descriptor", Array(f.LW, BIT)), ("rows", Array(S.G, Array(16 * f.B, BIT))),
            ("frame", Array(f.B, BYTE))), BIT


def _body_frame(B, S, desc, rows, frame):
    f = _format(S.FMT)
    nb, d = 8 * f.B, desc.refs

    def each(off: int, w: int) -> Coll:
        return Coll(Array(S.G, Array(w, BIT)),
                    concat([d.slice(f.HW + g * f.SW + off, f.HW + g * f.SW + off + w) for g in range(S.G)]))

    segs = B.batch(bind(N3Seg, B=f.B), rows, each(f.IDW, f.OB), each(f.IDW + f.OB, f.LB),
                   each(f.IDW + f.OB + f.LB, f.LB), axes=(0, 0, 0, 0))
    stream = B.call(bind(N3Xor, N=S.G, W=nb), segs) if S.G > 1 else segs[0]
    ok = B.call(bind(N3Place, FMT=S.FMT), Coll(Array(f.HW, BIT), d.slice(0, f.HW)), stream, frame)
    if S.G == f.G:
        return ok
    kinds = _coll(Array(f.G - S.G, BIT), [d[f.HW + g * f.SW] for g in range(S.G, f.G)])
    return B.call(BL.and_gate(2), _cat(Array(2, BIT), ok, B.call(bind(N3IsZero, W=f.G - S.G), kinds)))


N3Frame = CompositeDefinition(
    "N3Frame", 1, ("FMT", "G"), _sig_frame, _body_frame, conformance="exact-model-tested", register=False,
    doc="1 iff the frame is what the descriptor puts on its first G segments' rows, one each: the segments' streams "
        "(N3Seg) XORed, placed by N3Place; and every segment from G on is of kind 0")


# --- the window -------------------------------------------------------------------------------------------------------

def _window_params(S) -> list[tuple[str, object]]:
    f, rec = _format(S.FMT), _shape(S.REC)
    if S.SALTED not in (0, 1) or not isinstance(S.N, int) or isinstance(S.N, bool) or S.N < 1:
        raise ValueError(f"N3Window_v3 takes N >= 1 frames and SALTED 0 or 1, not N={S.N!r}, SALTED={S.SALTED!r}")
    if S.N > rec.T * rec.R:
        raise ValueError(f"N = {S.N} frames don't fit a record of {rec.T} buckets of {rec.R} slots")
    if not isinstance(S.G, int) or isinstance(S.G, bool) or not 1 <= S.G <= f.G:
        raise ValueError(f"G is 1 ... the format's {f.G} segments, not {S.G!r}")
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
        def each(t):
            return Array(n, Array(S.G, t))

        params += [("descriptors", Array(n, Array(f.LW, BIT))), ("descriptor_salts", Array(n, SALT_T)),
                   ("descriptor_paths", Array(n, Array(f.DL, DIGEST))), ("rows", each(Array(16 * f.B, BIT))),
                   ("row_salts", each(SALT_T)), ("row_paths", each(Array(f.DB, DIGEST)))]
    params += [("head", Array(rec.HL, BYTE)), ("record_salt", SALT_T), ("record_path", Array(2, DIGEST))]
    if S.SALTED:
        params.append(("slot", Array(max(1, (S.N - 1).bit_length()), BIT)))
    return params


def _sig_window(S):
    return tuple(_window_params(S)), BIT


def _body_window(B, S, *args):
    f = _format(S.FMT)
    N, G, n = S.N, S.G, S.N - S.SALTED
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
    if n:
        d = a["descriptors"]
        pos = B.call(bind(N3Positions, DL=f.DL, N=n), a["at"])
        oks.append(B.batch(bind(N3HidingRead, N_BITS=f.LW, D=f.DL), d, a["descriptor_salts"], a["layout"],
                           a["layout_domain"], pos, a["descriptor_paths"], axes=(0, 0, None, None, 0, 0)))
        ids = Coll(Array(n * G, Array(f.IDW, BIT)),
                   concat([d.refs.slice(i * f.LW + f.HW + g * f.SW, i * f.LW + f.HW + g * f.SW + f.IDW)
                           for i in range(n) for g in range(G)]))
        rows = a["rows"].reshape(Array(n * G, Array(16 * f.B, BIT)))
        salts = a["row_salts"].reshape(Array(n * G, SALT_T))
        paths = a["row_paths"].reshape(Array(n * G, Array(f.DB, DIGEST)))
        oks.append(B.batch(bind(N3RowRead, DEPTHS=f.DEPTHS, N_BITS=16 * f.B), ids, rows, salts, paths, a["roots"],
                           a["domains"], axes=(0, 0, 0, 0, None, None)))
        oks.append(B.batch(bind(N3Frame, FMT=S.FMT, G=G), d, a["rows"], data, axes=(0, 0, 0)))
    rec = _shape(S.REC)
    root = B.call(bind(N3FramesRoot, B=f.B, N=N, D=rec.D), frames)
    hb = _refs(a["head"])
    oks.append(B.call(bind(N3IsZero, W=8), _coll(Array(8, BIT), hb[8 * rec.ST:8 * rec.ST + 8])))
    nh = 8 * (rec.HL + 64)
    head = _coll(Array(nh, BIT), hb + [x for byte in _digest_bytes(root) for x in byte])
    # The head of the record at leaf d is leaf N2.HEADS + d = 2 + d: position bits (d, 1).
    pos = _coll(Array(2, BIT), _refs(a["direction"]) + K.bits(1, 1))
    oks.append(B.call(bind(N3HidingRead, N_BITS=nh, D=2), head, a["record_salt"], a["record"], a["record_domain"],
                      pos, a["record_path"]))
    bits = concat([o.refs for o in oks])
    return B.call(BL.and_gate(bits.count), Coll(Array(bits.count, BIT), bits))


N3Window = CompositeDefinition(
    "N3Window", 3, ("FMT", "N", "G", "SALTED", "REC"), _sig_window, _body_window, conformance="exact-model-tested",
    register=False,
    doc="1 iff the descriptors open at at, at + 1, ... in the layout, each of their first G segments' row opens in "
        "its unit's outbox (or is zero), every frame (the salt frame's slot skipped) is its descriptor's over "
        "those rows, the salt frame is the salt's, and the head over the frames' root is complete and opens at leaf "
        "2 + direction of N2's commitment")


def unit(f: Format, N: int, G: int, salted: bool, rec: Shape):
    return bind(N3Window, FMT=tuple(f), N=N, G=G, SALTED=int(salted), REC=tuple(rec))


# --- the reference and the arguments ----------------------------------------------------------------------------------

@dataclass
class Instance:
    """One window's unit and its arguments as bytes and ints, by parameter name (paths as lists of digests)."""

    fmt: Format
    N: int
    G: int
    salted: bool
    rec: Shape
    args: dict

    @property
    def unit(self):
        return unit(self.fmt, self.N, self.G, self.salted, self.rec)

    @property
    def key(self) -> tuple:
        return (tuple(self.fmt), self.N, self.G, self.salted, tuple(self.rec))


def read_ok(f: Format, seg: Sequence[int], row: bytes, salt: bytes, path: Sequence[bytes], roots: Sequence[bytes],
            domains: Sequence[bytes]) -> bool:
    """`N3RowRead`'s reference: kind 1, the row opens at q in unit u's outbox; kind 0, it is zero."""
    kind, u, q = seg[:3]
    if kind == 0:
        return not any(row)
    if u >= f.U:
        return False
    return root_ref(domains[u], SCHEMA, hiding_value(row, 16 * f.B, salt), q, path[:f.DEPTHS[u]]) == roots[u]


def reference(inst: Instance) -> bool:
    """`N3Window_v3`'s output on the instance, computed from its arguments as the circuit does."""
    f, a, N, G = inst.fmt, inst.args, inst.N, inst.G
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
    for i, x in enumerate(fields(f, d) for d in descs):
        rows = a["rows"][i]
        if not all(read_ok(f, x.segs[g], rows[g], a["row_salts"][i][g], a["row_paths"][i][g], a["roots"],
                           a["domains"]) for g in range(G)):
            return False
        if any(s.kind for s in x.segs[G:]) or x.mode not in (RAW, B64) or assemble(f, descs[i], rows, G) != frames[i]:
            return False
    rec = inst.rec
    head = a["head"] + CT.frames_root(a["frames"], GridParams(rec.T, rec.R, f.B))
    if head[rec.ST]:
        return False
    value = hiding_value(head, 8 * len(head), a["record_salt"])
    return root_ref(a["record_domain"], SCHEMA, value, N2.HEADS + a["direction"], a["record_path"]) == a["record"]


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
        "head": _bytes_bits(a["head"], 8 * inst.rec.HL), "record_salt": _digest_bits(a["record_salt"]),
        "record_path": [b for s in a["record_path"] for b in _digest_bits(s)],
    }
    if inst.salted:
        flat["salt"] = _bytes_bits(a["salt"], 8 * SALT_PAYLOAD)
        flat["slot"] = _int_bits(a["slot"], max(1, (inst.N - 1).bit_length()))
    if inst.N - inst.salted:
        flat["descriptors"] = [b for x in a["descriptors"] for b in _bytes_bits(x, f.LW)]
        flat["descriptor_salts"] = [b for s in a["descriptor_salts"] for b in _digest_bits(s)]
        flat["descriptor_paths"] = [b for p in a["descriptor_paths"] for s in p for b in _digest_bits(s)]
        flat["rows"] = [b for per in a["rows"] for r in per for b in _bytes_bits(r, 16 * f.B)]
        flat["row_salts"] = [b for per in a["row_salts"] for s in per for b in _digest_bits(s)]
        flat["row_paths"] = [b for per in a["row_paths"] for p in per
                             for s in list(p) + [zero_digest] * (f.DB - len(p)) for b in _digest_bits(s)]
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


_GROUPS = {"N3RowRead": "row_reads", "N3Frame": "frames", "N3FramesRoot": "frames_root",
           "N3IsZero": "record_opening"}


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
    """The window unit's ANDs, in total and by part: outbox row reads, descriptor reads, the head's status and opening,
    the segments' shifts (`N3Seg`), the rest of the frames (their XOR and placement), the frames root and the rest (the
    salt, the positions, the final AND)."""
    target = inst_or_unit.unit if isinstance(inst_or_unit, Instance) else inst_or_unit
    memo = {} if memo is None else memo
    out = dict.fromkeys(("row_reads", "descriptor_reads", "record_opening", "segments", "frames", "frames_root",
                         "other"), 0)
    for nd in target.body.nodes:
        name = getattr(nd.fn, "name", "")
        group = _GROUPS.get(name, "other")
        if name == "N3HidingRead":
            group = "record_opening" if nd.form == "call" else "descriptor_reads"
        ands = _times(nd) * _ands(nd.fn, memo)
        if name == "N3Frame":
            seg = _times(nd) * sum(_times(x) * _ands(x.fn, memo) for x in nd.fn.body.nodes
                                   if getattr(x.fn, "name", "") == "N3Seg")
            out["segments"] += seg
            ands -= seg
        out[group] += ands
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


def outbox_binding(program: bytes, unit: str) -> bytes:
    return hashlib.sha512(OUTBOX_LABEL + program + unit.encode()).digest()


def outbox_rows(data: bytes, B: int) -> list[bytes]:
    """The outbox's rows of 2B bytes: row q holds bytes qB ... qB + 2B - 1, zero past the end, for q < max(1,
    ceil(len / B)); then zero rows up to a power of two."""
    k = max(1, -(-len(data) // B))
    rows = [data[q * B:(q + 2) * B].ljust(2 * B, b"\0") for q in range(k)]
    return rows + [bytes(2 * B)] * ((1 << (k - 1).bit_length()) - k)


def register_outbox(program: bytes, unit: str, data: bytes, B: int, seed: bytes) -> Tree:
    return register(outbox_rows(data, B), 16 * B, outbox_binding(program, unit), seed + b"/outbox/" + unit.encode())


@dataclass
class Committed:
    """N2's commitment to one link-window (`LinkWindows`) and the firewall's opening, with the tree's levels kept."""

    lw: N2.LinkWindows
    opening: N2.Opening
    levels: tuple = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.levels = tuple(tree_levels(self.lw.domain(self.opening.window), self.lw.leaves(self.opening)))

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    @property
    def domain_id(self) -> bytes:
        return self.lw.domain(self.opening.window).domain_id

    @property
    def rows(self) -> list[bytes]:
        return N2.rows(self.opening.records)

    def path(self, i: int) -> list[bytes]:
        return [level[(i >> k) ^ 1] for k, level in enumerate(self.levels[:-1])]

    def head(self, d: int) -> bytes:
        """The head of the record at leaf d (0 egress, 1 ingress) less its frames root: the unit's `head`."""
        return self.opening.records[d].head.encode()[:-CT.DIGEST_BYTES]


@lru_cache(maxsize=None)
def _lms_key(H: int, seed: bytes) -> lms.PrivateKey:
    return lms.keygen(lms.Params(64, 1, H), hashlib.sha512(seed).digest())


@dataclass
class Certifier:
    """A stand-in for the network certifier and the firewall: an LMS key of 2^H one-time keys (the deployed N = 64,
    W = 1) that signs each record head in one certifier session, and N2's commitment to each link-window
    (`LinkWindows.commit`), its salts drawn from a seed."""

    key: lms.PrivateKey
    seed: bytes
    run: str = "n3-unit"
    session: bytes = bytes(32)
    used: int = 0

    @classmethod
    def toy(cls, signatures: int, seed: bytes) -> "Certifier":
        return cls(_lms_key(max(4, (signatures - 1).bit_length()), seed + b"/lms"), seed)

    @property
    def public(self) -> CT.LmsKey:
        return CT.LmsKey(self.key.params, lms.public_key(self.key))

    def certify(self, head: CT.RecordHead) -> CT.CertifiedRecord:
        digest, key_id = head.digest(), self.public.key_id
        message = CT.certificate_message(key_id, self.session, head.link, head.direction, head.window, digest)
        signature = lms.sign_at(self.key, self.used, message).encode()
        self.used += 1
        return CT.CertifiedRecord(head, CT.Certificate(key_id, head.link, head.direction, head.window, digest,
                                                       signature))

    def commit(self, link: str, window: int, heads: Sequence[CT.RecordHead], egress: GridParams, ingress: GridParams,
               sync_set: SyncSet) -> Committed:
        lw = N2.LinkWindows(link, egress, sync_set, self.public, self.run, self.session, ingress=ingress)
        salts = _salts(self.seed + b"/n2/" + link.encode() + b"\0" + window.to_bytes(8, "big"), N2.LEAVES)
        return Committed(lw, lw.commit(window, *(self.certify(h) for h in heads), salts))


@dataclass
class UnitTable:
    """The units in canonical order: names, outbox-tree depths, outbox roots and domain ids, and each unit's outbox
    tree and bytes."""

    names: list[str]
    depths: tuple
    roots: list[bytes]
    domains: list[bytes]
    trees: dict[str, Tree]
    data: dict[str, bytes]

    @classmethod
    def register(cls, outboxes: dict[str, bytes], B: int, program: bytes, seed: bytes) -> "UnitTable":
        """Each unit's outbox registered, the units in the order given."""
        trees = {u: register_outbox(program, u, x, B, seed) for u, x in outboxes.items()}
        return cls(list(trees), tuple(t.depth for t in trees.values()), [t.root for t in trees.values()],
                   [t.domain_id for t in trees.values()], trees, dict(outboxes))

    def index(self, key: tuple) -> int:
        return self.names.index(key[0])

    def outbox(self, key: tuple) -> bytes:
        return self.data[key[0]]

    def read(self, seg: Sequence[int], B: int) -> tuple[bytes, bytes, list[bytes]]:
        """The row a segment reads, its salt and its path: a zero row and salt for a segment of kind 0."""
        kind, u, q = seg[:3]
        if not kind:
            return bytes(2 * B), bytes(hm96.SHA512.salt_bytes), []
        t = self.trees[self.names[u]]
        return t.reg.rows[q], t.reg.salts[q], t.path(q)


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

    def segments(self, port: str) -> int:
        """G of the port's windows: the most segments its declared frames read, at least one."""
        return max([1] + [len(d.segs) for d in self.descriptors.get(port, [])])


def register_layout(declared: dict[str, list[list[tuple]]], ports: list[str], B: int, units: UnitTable, program: bytes,
                    placement: bytes, seed: bytes) -> Layout:
    """Describe every declared frame, fit the format, and register the layout: one advice tree, CAP positions a port."""
    descs = {p: [describe(pieces, B, units.index) for pieces in declared.get(p, [])] for p in ports}
    most = max([len(v) for v in descs.values()] + [1])
    cap = 1 << max(1, most.bit_length())
    fmt = fit([d for v in descs.values() for d in v], B, units.depths, max(1, merkle_depth(len(ports) * cap)))
    for p in ports:
        for pieces, d in zip(declared.get(p, []), descs[p]):
            rows = [units.read(s, B)[0] for s in d.segs]
            if assemble(fmt, encode(fmt, d), rows) != expand(pieces, B, units.outbox):
                raise AssertionError(f"{p}: a descriptor doesn't put its declared frame on its outbox rows")
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

    def locate(self, unit: str, gate: int) -> tuple[int, int]:
        """(row, bit offset) of a value the unit holds by the placement, whether or not the unit committed it."""
        width = self.model.value_bits
        i = V._index(self.gates[unit], gate)
        return i // (V.ROW_BITS // width), (i % (V.ROW_BITS // width)) * width

    def row(self, unit: str, pos: int) -> bytes:
        """The unit's committed row, or the zero row for a unit that committed none."""
        u = self.units.get(unit)
        return u.registration.rows[pos] if u is not None and pos < len(u.registration.rows) else bytes(V.ROW_BITS // 8)

    def payload(self, m: F.Message) -> bytes:
        """The message's bytes (`framing.encode`), from its crossing values' bits in its source's committed rows."""
        acc, at = 0, 0
        for e in m.edges:
            row, o = self.locate(e.src, e.value)
            acc |= ((int.from_bytes(self.row(e.src, row), "little") >> o) & ((1 << e.width) - 1)) << at
            at += e.width
        return acc.to_bytes((m.bits + 7) // 8, "little")

    @cached_property
    def outboxes(self) -> tuple[dict[str, bytes], dict[int, int]]:
        """Each unit's outbox, its outgoing messages' bytes in send order, and each message's byte offset in its
        source's."""
        data: dict[str, bytes] = dict.fromkeys(self.hw.unit_names(), b"")
        base = {}
        for m in self.phi.send_order():
            base[m.stream] = len(data[m.src])
            data[m.src] += self.payload(m)
        return data, base

    def message(self, m: F.Message) -> list[tuple]:
        """The message's bits as pieces: one run of its source's outbox."""
        return [("r", (m.src,), 8 * self.outboxes[1][m.stream], 8 * ((m.bits + 7) // 8))]

    def table(self, seed: bytes) -> UnitTable:
        """Every unit's outbox, registered."""
        return UnitTable.register(self.outboxes[0], self.phi.frame_bytes, self.program, seed)


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
    salted window, the record's grid (when not the frames filled by the counts), and the unit's instance; `why` says
    why no unit runs (refused from public data alone, or `EMPTY`)."""

    port: str
    window: int
    frames: list[bytes]
    counts: tuple[int, ...] = ()
    reason: str = ""
    salt: bytes | None = None
    grid: tuple | None = None
    instance: Instance | None = None
    why: str = ""

    def check(self, circuit: bool = True) -> dict:
        out = {"port": self.port, "window": self.window, "recorded": len(self.frames)}
        if self.instance is None:
            return out | {"by": "public data", "ok": self.why == EMPTY and not self.frames, "why": self.why}
        inst = self.instance
        ref = reference(inst)
        out |= {"by": "unit", "ok": ref, "N": inst.N, "G": inst.G, "salted": inst.salted}
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
    """A run's statement, units, registered layout, the ports' links (the link-window commitment each records into,
    and its end), each port's grid, Σ_sync, each port's link as its certifier names it in its heads (when not its
    commitment's), the certifier, and every link-window's commitment: what every window's instance is built from."""

    S: Statement
    units: UnitTable
    layout: Layout
    links: dict[str, tuple[str, int]]
    grids: dict[str, GridParams]
    seed: bytes
    sync_set: SyncSet = SyncSet((0,), (1,))
    named: dict[str, str] = field(default_factory=dict)
    certifier: Certifier | None = None
    committed: dict[tuple[str, int], Committed] = field(default_factory=dict)

    @classmethod
    def build(cls, S: Statement, declared: dict[str, list[list[tuple]]], ports: list[str],
              links: dict[str, tuple[str, int]], grids: dict[str, GridParams], seed: bytes = b"n3-unit",
              **kw) -> "Setup":
        units = S.table(seed)
        layout = register_layout(declared, ports, S.phi.frame_bytes, units, S.program, S.placement_root, seed)
        return cls(S, units, layout, links, grids, seed, **kw)

    @property
    def fmt(self) -> Format:
        return self.layout.fmt

    def shape(self, port: str) -> Shape:
        g = self.grids[port]
        return Shape(len(self.named.get(port, self.links[port][0]).encode()), g.buckets, g.allocation)

    def head(self, port: str, window: int, w: Window | None) -> CT.RecordHead:
        """The record head the port's certifier signs for the window: an empty complete grid when it recorded
        nothing."""
        link, d = self.links[port]
        g = self.grids[port]
        if w is None:
            grid, status = fill([], [0] * g.buckets, g.allocation), COMPLETE
        else:
            grid = w.grid if w.grid is not None else fill(w.frames, w.counts, g.allocation)
            status = Status(Reason(w.reason)) if w.reason else COMPLETE
        direction = CT.Direction.INGRESS if d else CT.Direction.EGRESS
        return CT.record_head(self.named.get(port, link), direction, window, grid, status, g,
                              CT.parameters_digest(g, None if d else self.sync_set))

    def commit(self, link: str, window: int, recorded: dict[tuple[str, int, int], Window]) -> Committed:
        """N2's commitment to the link-window: both ends' certified records and their heads."""
        if (link, window) not in self.committed:
            ports = {v: k for k, v in self.links.items()}
            heads = [self.head(ports[(link, d)], window, recorded.get((link, d, window))) for d in (0, 1)]
            self.committed[(link, window)] = self.certifier.commit(  # type: ignore[union-attr]
                link, window, heads, self.grids[ports[(link, 0)]], self.grids[ports[(link, 1)]], self.sync_set)
        return self.committed[(link, window)]

    def attach(self, windows: list[Window]) -> list[Window]:
        """Each window's instance, its port's earlier frames giving its first descriptor's position; or why none
        runs."""
        recorded = {(*self.links[w.port], w.window): w for w in windows}
        if self.certifier is None:
            self.certifier = Certifier.toy(2 * len({(link, w) for link, _, w in recorded}), self.seed)
        seen: dict[str, int] = defaultdict(int)
        for w in sorted(windows, key=lambda w: (self.layout.ports.index(w.port), w.window)):
            n = len(w.frames) - (w.salt is not None and len(w.frames) > 0)
            a = seen[w.port]
            seen[w.port] += n
            if w.reason:
                w.why = f"the record's status is {w.reason}"
            elif not w.frames:
                w.why = EMPTY
            elif a + n > self.layout.cap:
                w.why = f"frames past the port's capacity, {self.layout.cap}"
            else:
                link, d = self.links[w.port]
                w.instance = self.instance(w, a, n, self.commit(link, w.window, recorded), d)
        return windows

    def instance(self, w: Window, a: int, n: int, record: Committed, direction: int) -> Instance:
        f, L, U = self.fmt, self.layout, self.units
        G = L.segments(w.port)
        positions = [L.position(w.port, a + i) for i in range(n)]
        descs = [L.descriptor(w.port, a + i) for i in range(n)]
        reads = [[U.read(d.segs[g] if g < len(d.segs) else Segment(), f.B) for g in range(G)] for d in descs]
        args = {"layout": L.tree.root, "layout_domain": L.tree.domain_id, "at": positions[0] if n else
                L.position(w.port, min(a, L.cap - 1)), "record": record.root, "record_domain": record.domain_id,
                "direction": direction, "roots": U.roots, "domains": U.domains, "frames": w.frames,
                "descriptors": [L.tree.reg.rows[p] for p in positions],
                "descriptor_salts": [L.tree.reg.salts[p] for p in positions],
                "descriptor_paths": [L.tree.path(p) for p in positions],
                "rows": [[r[0] for r in per] for per in reads], "row_salts": [[r[1] for r in per] for per in reads],
                "row_paths": [[r[2] for r in per] for per in reads], "head": record.head(direction),
                "record_salt": record.opening.salts[N2.HEADS + direction],
                "record_path": record.path(N2.HEADS + direction)}
        salted = w.salt is not None
        if salted:
            args["salt"] = w.window.to_bytes(8, "big") + w.salt
            args["slot"] = next((i for i, fr in enumerate(w.frames) if _is_salt(fr)), 0)
        return Instance(f, len(w.frames), G, salted, self.shape(w.port), args)


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
    order (stably by bucket), against the layout of φ's declared frames, registered before the run. Every port's grid
    is `buckets` buckets of r slots, r the most frames any port declared or recorded in one bucket."""
    S = statement(public, witness)
    links = sim_ports(S.hw)
    declared = sim_declared(S)
    per_bucket = [n for v in (*declared.values(), *records.values()) for n in Counter(b for b, _ in v).values()]
    grid = GridParams(buckets, max([1] + per_bucket), S.phi.frame_bytes)
    setup = Setup.build(S, {k: [p for _, p in v] for k, v in declared.items()}, list(links), links,
                        dict.fromkeys(links, grid), seed)
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
                salts: dict[tuple[str, int], bytes] | None = None, seed: bytes = b"n3-unit",
                grids: tuple[GridParams, GridParams] | None = None,
                sync_set: SyncSet | None = None) -> tuple[Setup, list[Window]]:
    """The lab's windows: per block and link (`lab.EGRESS`, `lab.INGRESS`), each record's released frames, against the
    layout of the declared frames. With `salts`, each salted ingress window's revealed salt commitment by (block,
    window): one of its frames is then the certifier's salt frame, and the rest are the port's next declared frames.
    `grids` are the egress and ingress grids (default: each link's first record's shape) and `sync_set` the egress
    Σ_sync, as the calibration report gives them. A block's two links share one N2 commitment a window, named for the
    block, its egress record at leaf 0 and its ingress record at leaf 1."""
    from verity.core.protocols.certifier.grid import occupancy

    from . import lab as L

    S = statement(public, witness)
    B = S.phi.frame_bytes
    blocks = sorted(S.hw.units)
    kinds = (L.EGRESS, L.INGRESS)
    links = {f"{b}:{k}": (b, d) for b in blocks for d, k in enumerate(kinds)}
    if grids is None:
        first = [next((r for rs in published.values() for r in rs if r.link == k), None) for k in kinds]
        grids = tuple(GridParams(len(r.grid), len(r.grid[0]), B) if r else GridParams(1, 1, B)
                      for r in first)  # type: ignore[assignment]
    declared = {}
    for b in blocks:
        declared[f"{b}:{L.EGRESS}"], declared[f"{b}:{L.INGRESS}"] = lab_declared(S, b)
    setup = Setup.build(S, declared, list(links), links, {p: grids[d] for p, (_, d) in links.items()}, seed,
                        named={p: p.split(":")[1] for p in links}, **({"sync_set": sync_set} if sync_set else {}))
    windows = []
    for block in sorted(published):
        for link in kinds:
            for rec in sorted((r for r in published[block] if r.link == link), key=lambda r: r.window):
                salt = (salts or {}).get((block, rec.window)) if link == L.INGRESS else None
                windows.append(Window(f"{block}:{link}", rec.window, list(L.released([rec])), occupancy(rec.grid),
                                      rec.status.reason.value if rec.status.reason else "", salt, rec.grid))
    return setup, setup.attach(windows)


# --- the demo ---------------------------------------------------------------------------------------------------------

DEMO_FRAMES = [
    [const(encode_frame(LAST, F.HELLO_STREAM, F.HELLO.pack(0, 1), 16))],
    [const(HEADER.pack(DATA, 1, 5)), ("r", ("u0",), 8 * 14, 8 * 5), zeros(2)],
    [const(HEADER.pack(DATA, 1, 6)), ("b", (("r", ("u1",), 0, 8), ("r", ("u0",), 8 * 20, 8)), 2), const(b"\n\n"),
     zeros(1)],
    [const(HEADER.pack(DATA, 2, 6)), ("r", ("u0",), 8 * 33, 8 * 2), zeros(1), ("r", ("u0",), 8 * 36, 8 * 3),
     zeros(1)],
]
"""Four frames of every kind, B = 16, on port p0: a HELLO-like constant frame, a frame of u0's bytes 14 ... 18 across
its rows 0 and 1, an event-like frame of base64 over two units' bytes, and a frame of two runs of u0's row 2 around a
zero byte."""


@dataclass
class Demo:
    units: UnitTable
    layout: Layout
    record: Committed
    window: Window
    setup: Setup


def demo(seed: bytes = b"n3-demo") -> Demo:
    """A small run of two units (an outbox of 40 bytes, four rows of depth 2; one of 10 bytes, one row of depth 0) and
    two ports, the two ends of link "demo", and p0's window of DEMO_FRAMES: N = 4 in two buckets of two slots, G = 2."""
    B = 16
    outboxes = {u: hashlib.shake_256(seed + u.encode()).digest(k) for u, k in (("u0", 40), ("u1", 10))}
    units = UnitTable.register(outboxes, B, b"demo", seed)
    layout = register_layout({"p0": DEMO_FRAMES}, ["p0", "p1"], B, units, b"demo", b"", seed)
    frames = [expand(p, B, units.outbox) for p in DEMO_FRAMES]
    links = {"p0": ("demo", 0), "p1": ("demo", 1)}
    setup = Setup(None, units, layout, links, dict.fromkeys(links, GridParams(2, 2, B)), seed)  # type: ignore[arg-type]
    (window,) = setup.attach([Window("p0", 0, frames, (2, 2))])
    return Demo(units, layout, setup.committed[("demo", 0)], window, setup)


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
    import records  # noqa: F401, PLC0415  (puts network_traces on the path, for active_replay)

    import active_replay as ar  # noqa: PLC0415

    from . import lab as L

    public = json.loads((run / "public.json").read_text())
    witness = json.loads((run / "witness.json").read_text())
    report = ar.Setup(json.loads((run / "report.json").read_text()))
    B = public["frame_bytes"]
    published = {}
    for block in sorted(p.name for p in run.iterdir() if (p / f"{L.EGRESS}.records").exists()):
        published[block] = [ar.unpack_record(x, B) for f in (L.EGRESS, L.INGRESS)
                            for x in ar.read_blobs(run / block / f"{f}.records")]
    setup, windows = lab_windows(public, witness, published, grids=(report.egress_params, report.ingress_params),
                                 sync_set=report.sync_set)
    results = _sized(windows, not args.reference_only)
    return {"run": str(run), "format": setup.fmt._asdict(), "summary": summarize(results), "windows": results}


def _circuit_check(args) -> dict:
    """circuit-check's checks on the demo's window and its SHA-free parts (with `--parts`, the parts alone), with the
    CLI's options and pins. The window's C-Flock lowering is its parts' (`lower_whole=False`): lowered whole it would
    carry the SHA-512 compression that `checks.UNLOWERED` records as not fitting in 64 GiB. The unit is a Program of its
    own, never a served Program's Call, so the partition invariant doesn't apply (`as_call=False`)."""
    from circuit_check import checks as CK

    d = demo()
    f, inst = d.layout.fmt, d.window.instance
    opts = CK.Options()
    entries = []
    targets = [(bind(N3Frame, FMT=tuple(f), G=inst.G), True), (bind(N3Seg, B=f.B), True)]
    for target, whole in targets + ([] if args.parts else [(inst.unit, False)]):
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
    c.add_argument("--parts", action="store_true", help="the SHA-free parts alone, not the window")
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
