"""N3's proof unit for one port-window: the frames a certifier recorded on one link direction in one window are
exactly the committed crossing values, framed by φ. A Boolean Definition, which a proof unit of the proof service runs.

    N3Window_v1{LAYOUT}
        (digest: Array<8, Word>, counts: Array<T, Count>, frames: Array<n, Array<B, Byte>>, rows: Array<m, Row>)
        -> Value<1>
        Word = Array<64, Value<1>>, Count = Array<16, Value<1>>, Byte = Array<8, Value<1>>, Row = Array<256, Value<1>>

1 iff (1) `counts`, the record's frames per bucket, are the layout's, (2) `digest` is the record's digest of
`frames` (`record_digest`, the format of `../records.py`'s `digest`: SHA-512 of `DOMAIN`, the window's status reason, a
0 byte and the frames in grid order) and (3) every frame is the layout's: each of its bits is a constant or a bit of
one of the committed rows `rows`. The digest and the counts are the record's, public; the layout is protocol
structure, public; the frames and the rows are the witness. A digest is Array<8, Word> as
`verity.ml.commitments.gates.lms` lays SHA-512 words out (bytes big-endian in a word, each word's bits least
significant first); a count, a byte and a row are least significant bit first, a row's bit i being bit i % 8 of its
byte i // 8 as `values.rows` packs it.

**The layout** (`LAYOUT`, a dict): `B` the frame size, `reason` the window's status reason ("" when complete),
`counts` frames per bucket (T of them, n = their sum, 1 <= n), `rows` the number m of row inputs, and `frames`, n lists
of pieces whose widths add up to 8B bits, the frame's bits least significant first from bit 0 of byte 0:

- `("k", w, v)`: w constant bits, v's;
- `("r", i, o, w)`: bits o … o + w − 1 of row i;
- `("b", pieces, L)`: the base64 text (RFC 4648, padded) of the L bytes `pieces` give, 32⌈L/3⌉ bits.

`sim_windows` and `lab_windows` derive each window's layout from the public statement alone: φ's slots (the
simulator) or the lab's events and requests at the record's counts (the lab), with each crossing value's bits named by
its source unit's row and offset (`values.UnitValues.locate`). The rows are the units' committed rows (`values`):
binding them to the units' roots is the proof service's commitment port, not this unit (the network_sim README, "N3 in
zero knowledge").

**The parts.** `N3Match_v1{LAYOUT}` (counts, frames, rows) -> Value<1> is (1) and (3), traced from `forms`: one AND
per compared bit and 22 per base64 character. The digest is `Sha512Compress_v1` calls over the frames' bytes behind
the constant prefix, and `LmsEqual_v1{N=64}` compares it, as `LmsVerify_v1` wires its public key's hash.

`expand` and `reference` are the Python reference; `evaluate` runs the Definition (`evaluate_bits`).

    python -m workload.n3_unit sim --out FILE [--tamper UNIT:MODE] [--buckets K] [--frame-bytes B]
    python -m workload.n3_unit lab RUN_DIR --out FILE     every window a lab run's certifiers recorded (lab_run --out)
    python -m workload.n3_unit circuit-check --out FILE   circuit-check's checks on `DEMO`'s binding and its part
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
from pathlib import Path
from typing import Sequence

import numpy as np

from verity.core.protocols.certifier.active import DATA, HEADER, LAST, encode_frame
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import ZERO, add, and_reduce, const_bits, sub
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array
from verity.ml.commitments import advice
from verity.ml.commitments.gates.lms import LmsEqual, Sha512Compress
from verity.ml.commitments.gates.sha512 import IV, pad_tail, words_be

from . import framing as F
from . import model as M
from . import placement as P
from . import values as V
from .hardware import Hardware

BIT = BL.BIT
BYTE = Array(8, BIT)
WORD = Array(64, BIT)
DIGEST = Array(8, WORD)
COUNT_BITS = 16
COUNT = Array(COUNT_BITS, BIT)
ROW = Array(V.ROW_BITS, BIT)
DOMAIN = b"network_sim/certifier-record/v1\0"
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


def salt_frame(window: int, commitment: bytes, frame_bytes: int) -> bytes:
    """Window `window`'s salt frame as the ingress certifier records it: kind `SALT` on `SALT_STREAM`, its payload the
    window (u64, big-endian) and the salt's 64-byte commitment, which is public once the salt is revealed."""
    return encode_frame(SALT, SALT_STREAM, window.to_bytes(8, "big") + commitment, frame_bytes)


def _is_salt(frame: bytes) -> bool:
    kind, stream, _ = HEADER.unpack_from(frame)
    return kind == SALT and stream == SALT_STREAM


# --- pieces -----------------------------------------------------------------------------------------------------------

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
    """Bits lo … hi − 1 of the concatenation of constant and row pieces."""
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


def why(layout: dict) -> str | None:
    """Why `layout` is no layout `N3Window_v1` covers, or None."""
    try:
        B, counts, m, frames, reason = layout["B"], layout["counts"], layout["rows"], layout["frames"], layout["reason"]
    except (KeyError, TypeError):
        return "a layout is a dict of B, reason, counts, rows and frames"
    if not isinstance(B, int) or B <= HEADER.size or not isinstance(m, int) or m < 0 or not isinstance(reason, str):
        return f"a frame has room past its {HEADER.size}-byte header, and there are m >= 0 rows"
    if not counts or any(not isinstance(c, int) or not 0 <= c < 1 << COUNT_BITS for c in counts):
        return f"the counts are one or more integers below 2^{COUNT_BITS}"
    if sum(counts) < 1 or sum(counts) != len(frames):
        return "the counts add up to the number of frames, at least one"

    def bad(pieces, inner: bool) -> str | None:
        for p in pieces:
            if p[0] == "k" and len(p) == 3 and p[1] >= 1 and 0 <= p[2] < 1 << p[1]:
                continue
            if p[0] == "r" and len(p) == 4 and 0 <= p[1] < m and p[2] >= 0 and p[3] >= 1 and p[2] + p[3] <= V.ROW_BITS:
                continue
            if p[0] == "b" and len(p) == 3 and not inner and p[2] >= 1:
                if (w := bad(p[1], True)) is not None:
                    return w
                if sum(width(q) for q in p[1]) == 8 * p[2]:
                    continue
            return f"piece {p!r:.80} is none of ('k', w, v), ('r', row, offset, w), ('b', pieces, bytes)"
        return None

    for j, pieces in enumerate(frames):
        if (w := bad(pieces, False)) is not None:
            return f"frame {j}: {w}"
        if sum(width(p) for p in pieces) != 8 * B:
            return f"frame {j}'s pieces are {sum(width(p) for p in pieces)} bits, not {8 * B}"
    return None


# --- the Python reference ---------------------------------------------------------------------------------------------

def _bits(pieces: Sequence[tuple], rows: Sequence[bytes]) -> tuple[int, int]:
    acc, at = 0, 0
    for p in pieces:
        if p[0] == "k":
            v, w = p[2], p[1]
        elif p[0] == "r":
            v, w = (int.from_bytes(rows[p[1]], "little") >> p[2]) & ((1 << p[3]) - 1), p[3]
        else:
            inner, n = _bits(p[1], rows)
            text = base64.b64encode(inner.to_bytes(n // 8, "little"))
            v, w = int.from_bytes(text, "little"), 8 * len(text)
        acc |= v << at
        at += w
    return acc, at


def expand(layout: dict, rows: Sequence[bytes]) -> list[bytes]:
    """The frames the layout puts in the window over the rows."""
    return [_bits(pieces, rows)[0].to_bytes(layout["B"], "little") for pieces in layout["frames"]]


def reference(layout: dict, digest: bytes, counts: Sequence[int], frames: Sequence[bytes],
              rows: Sequence[bytes]) -> bool:
    return (tuple(counts) == tuple(layout["counts"]) and record_digest(frames, layout["reason"]) == digest
            and list(frames) == expand(layout, rows))


# --- the circuit ------------------------------------------------------------------------------------------------------

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


def _expect(C, pieces: Sequence[tuple], rows: Sequence[Sequence]) -> list:
    out = []
    for p in pieces:
        if p[0] == "k":
            out += const_bits(p[2], p[1])
        elif p[0] == "r":
            out += list(rows[p[1]][p[2]:p[2] + p[3]])
        else:
            out += b64(C, _expect(C, p[1], rows), p[2])
    return out


def _sig_match(S):
    lay = S.LAYOUT
    return ((("counts", Array(len(lay["counts"]), COUNT)), ("frames", Array(sum(lay["counts"]), Array(lay["B"], BYTE))),
             ("rows", Array(lay["rows"], ROW))), BIT)


def _build_match(C, lay: dict, *e) -> list:
    T, n, B = len(lay["counts"]), sum(lay["counts"]), lay["B"]
    counts, frames, rows = e[:T], e[T:T + n * B], e[T + n * B:]
    same = []
    for got, want in zip(counts, lay["counts"]):
        same += [(x ^ y).inv() for x, y in zip(got, const_bits(want, COUNT_BITS))]
    for j, pieces in enumerate(lay["frames"]):
        got = [x for byte in frames[j * B:(j + 1) * B] for x in byte]
        same += [(x ^ y).inv() for x, y in zip(got, _expect(C, pieces, rows), strict=True)]
    return [and_reduce(C, same)]


_TRACED: dict[str, TR.Traced] = {}


def traced(layout: dict) -> TR.Traced:
    """`N3Match_v1{LAYOUT}`'s gate circuit, traced once per layout."""
    key = json.dumps(layout, sort_keys=True)
    if key not in _TRACED:
        params, ret = _sig_match(_S(layout))
        _TRACED[key] = TR.trace(lambda C, *e: _build_match(C, layout, *e), [t for _, t in params], ret)
    return _TRACED[key]


@dataclass(frozen=True)
class _S:
    LAYOUT: dict


def _body_match(B, S, *colls):
    params, ret = _sig_match(S)
    return TR.emit(traced(S.LAYOUT), [t for _, t in params], ret, B, *colls)


def _refuse(bindings: dict) -> str | None:
    return why(bindings["LAYOUT"]) if "LAYOUT" in bindings else None


N3Match = CompositeDefinition(
    "N3Match", 1, ("LAYOUT",), _sig_match, _body_match, conformance="exact-model-tested", register=False,
    refuse=_refuse,
    doc="1 iff the counts are the layout's frames per bucket and every frame is the layout's over the rows")


def _sig_window(S):
    params, ret = _sig_match(S)
    return (("digest", DIGEST), *params), ret


def _refs(c: Coll) -> list:
    return [c.refs[k] for k in range(c.refs.count)]


def _coll(t, refs: list) -> Coll:
    return Coll(t, TR._refs(list(refs)))


def _words(data: Sequence[Sequence]) -> list[list]:
    """Bytes (8 bits each, least significant first) as SHA-512 words: 64 bits least significant first, bytes
    big-endian."""
    return [[x for b in range(8) for x in data[8 * k + 7 - b]] for k in range(len(data) // 8)]


def _body_window(B, S, digest, counts, frames, rows):
    lay = S.LAYOUT
    bit = [B.call(BL.constant(b)).refs[0] for b in (0, 1)]

    def bits(value: int, w: int) -> list:
        return [bit[(value >> k) & 1] for k in range(w)]

    matched = B.call(bind(N3Match, LAYOUT=lay), counts, frames, rows)
    fb = _refs(frames)
    data = [bits(b, 8) for b in prefix(lay["reason"])] + [fb[i:i + 8] for i in range(0, len(fb), 8)]
    data += [bits(b, 8) for b in pad_tail(len(data))]
    words = _words(data)
    cv = _coll(DIGEST, [b for v in IV for b in bits(v, 64)])
    for k in range(0, len(words), 16):
        cv = B.call(Sha512Compress, cv, _coll(Array(16, WORD), [x for w in words[k:k + 16] for x in w]))
    hashed = B.call(bind(LmsEqual, N=64), cv, digest)
    return B.call(BL.AND, BL.operands(matched, hashed))


N3Window = CompositeDefinition(
    "N3Window", 1, ("LAYOUT",), _sig_window, _body_window, conformance="exact-model-tested", register=False,
    refuse=_refuse,
    doc="1 iff the counts are the layout's frames per bucket, the digest is the record's digest of the frames "
        "(SHA-512 of the domain, the reason, a 0 byte and the frames) and every frame is the layout's over the rows")


def unit(layout: dict):
    return bind(N3Window, LAYOUT=layout)


def compressions(layout: dict) -> int:
    n = len(prefix(layout["reason"])) + sum(layout["counts"]) * layout["B"]
    return (n + len(pad_tail(n))) // 128


def size(layout: dict) -> dict:
    """The unit's gates: its ANDs, the compressions among them, and the frame bits it compares."""
    match = traced(layout).counts.get("and", 0)
    sha = Sha512Compress.traced().counts.get("and", 0)
    eq = LmsEqual.traced(64).counts.get("and", 0)
    c = compressions(layout)
    return {"frames": sum(layout["counts"]), "frame_bytes": layout["B"], "rows": layout["rows"], "compressions": c,
            "ands": match + c * sha + eq + 1, "ands_match": match, "ands_sha512": c * sha}


def arguments(digest: bytes, counts: Sequence[int], frames: Sequence[bytes], rows: Sequence[bytes]) -> list[np.ndarray]:
    """`N3Window_v1`'s four arguments for one instance, each (1, leaves) of 0/1."""
    if any(not 0 <= c < 1 << COUNT_BITS for c in counts):
        raise ValueError(f"a count is below 2^{COUNT_BITS}")
    flat = [
        [(w >> b) & 1 for w in words_be(digest) for b in range(64)],
        [(c >> b) & 1 for c in counts for b in range(COUNT_BITS)],
        [(x >> b) & 1 for f in frames for x in f for b in range(8)],
        [(int.from_bytes(r, "little") >> b) & 1 for r in rows for b in range(V.ROW_BITS)],
    ]
    return [np.asarray(a, dtype=np.uint8).reshape(1, -1) for a in flat]


def evaluate(layout: dict, *instances: tuple[bytes, Sequence[int], Sequence[bytes], Sequence[bytes]]) -> list[int]:
    """The unit's output on each instance (digest, counts, frames, rows), all in one evaluation."""
    for _, _, frames, rows in instances:
        if len(frames) != sum(layout["counts"]) or len(rows) != layout["rows"]:
            raise ValueError(f"the layout takes {sum(layout['counts'])} frames and {layout['rows']} rows")
    args = [np.concatenate(a) for a in zip(*(arguments(*x) for x in instances))]
    return [int(b) for b in np.asarray(evaluate_bits(unit(layout), *args)).reshape(-1)]


# --- windows ----------------------------------------------------------------------------------------------------------

@dataclass
class Window:
    """One port-window: the record's public digest and counts, the frames it holds, and the unit's layout and rows.
    `why` says why no unit runs: the window is refused from public data alone, or (`EMPTY`) nothing is declared or
    recorded."""

    port: str
    window: int
    digest: bytes
    counts: tuple[int, ...]
    frames: list[bytes]
    layout: dict | None = None
    sources: tuple[tuple[str, int], ...] = ()
    rows: list[bytes] = field(default_factory=list)
    why: str = ""

    def witness(self) -> list[bytes]:
        """The frames the prover feeds the unit: the recorded ones, or φ's when the record holds another number."""
        if len(self.frames) == sum(self.layout["counts"]):
            return self.frames
        return expand(self.layout, self.rows)

    def check(self, circuit: bool = True) -> dict:
        out = {"port": self.port, "window": self.window, "recorded": len(self.frames)}
        if self.layout is None:
            ok = self.why == EMPTY and not self.frames
            return out | {"by": "public data", "ok": ok, "why": self.why}
        instance = (self.digest, self.counts, self.witness(), self.rows)
        ref = reference(self.layout, *instance)
        out |= {"by": "unit", "ok": ref, "rows": len(self.rows), "compressions": compressions(self.layout)}
        if circuit:
            t0 = time.perf_counter()
            (out["circuit"],) = evaluate(self.layout, instance)
            out["seconds"] = round(time.perf_counter() - t0, 3)
            if out["circuit"] != ref:
                raise AssertionError(f"{self.port}[{self.window}]: the circuit says {out['circuit']}, "
                                     f"the reference {ref}")
        return out


EMPTY = "nothing declared or recorded"


def _window(port: str, w: int, B: int, reason: str, counts: Sequence[int], frames: list[bytes],
            declared: list[list[tuple]], committed: dict[str, V.UnitValues]) -> Window:
    """The window's layout from the declared frames' pieces, whose rows are named (unit, row), indexed in order."""
    keys: set[tuple[str, int]] = set()

    def collect(pieces):
        for p in pieces:
            if p[0] == "r":
                keys.add(p[1])
            elif p[0] == "b":
                collect(p[1])

    for pieces in declared:
        collect(pieces)
    sources = tuple(sorted(keys))
    index = {k: i for i, k in enumerate(sources)}

    def rewrite(pieces):
        return [("r", index[p[1]], p[2], p[3]) if p[0] == "r" else
                ("b", tuple(rewrite(p[1])), p[2]) if p[0] == "b" else p for p in pieces]

    layout = {"B": B, "reason": reason, "counts": list(counts), "rows": len(sources),
              "frames": [_merge(rewrite(pieces)) for pieces in declared]}
    rows = [committed[u].registration.rows[r] for u, r in sources]
    return Window(port, w, record_digest(frames, reason), tuple(counts), frames, layout, sources, rows)


@dataclass
class Statement:
    """What a window's layout is derived from: the Program, the hardware, φ, and each unit's committed values."""

    model: M.Model
    hw: Hardware
    phi: F.Phi
    placement: P.Placement
    units: dict[str, V.UnitValues]

    def message(self, m: F.Message) -> list[tuple]:
        """The message's bits as pieces: each crossing value's bits in its source unit's row, then the last byte's
        zeros."""
        out = []
        for e in m.edges:
            if e.src not in self.units:
                raise KeyError(f"unit {e.src} committed no values")
            row, o = self.units[e.src].locate(e.value)
            out.append(("r", (e.src, row), o, e.width))
        return out + [("k", -m.bits % 8, 0)]


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
    units = {}
    for u, pub in public["values"].items():
        w = witness["values"][u]
        reg = advice.register([bytes.fromhex(r) for r in w["rows"]], V.ROW_BITS, V.binding(u),
                              [bytes.fromhex(s) for s in w["salts"]])
        gates = tuple(placement.at(hw.leaves_of(u)))
        if reg.root.hex() != pub["root"] or pub["count"] != len(gates):
            raise ValueError(f"unit {u}'s rows don't register to its public root")
        units[u] = V.UnitValues(u, gates, model.value_bits, reg)
    return Statement(model, hw, F.phi(hw, edges, public["frame_bytes"]), placement, units)


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
            try:
                message = S.message(m)
            except KeyError as e:
                frames.append((s.bucket, e))
                continue
            nbytes, last = (m.bits + 7) // 8, s.index == F.frame_count(m, B) - 1
            lo, hi = s.index * room, min((s.index + 1) * room, nbytes)
            frames.append((s.bucket, _data_frame(LAST if last else DATA, m.stream, _slice(message, 8 * lo, 8 * hi),
                                                 hi - lo, B)))
        out[key] = sorted(frames, key=lambda x: x[0])
    return out


def sim_windows(public: dict, witness: dict, records: dict[str, list[tuple[int, bytes]]], buckets: int) -> list[Window]:
    """The simulator's windows: per certified link direction, `buckets` buckets a window, the recorded frames in grid
    order (stably by bucket), and the layout φ declares there, at φ's counts."""
    S = statement(public, witness)
    B = public["frame_bytes"]
    declared = sim_declared(S)
    out = []
    for key in sorted(set(declared) | set(records)):
        decl, rec = declared.get(key, []), sorted(records.get(key, []), key=lambda x: x[0])
        last = max([b for b, _ in decl] + [b for b, _ in rec] + [0])
        for w in range(last // buckets + 1):
            lo, hi = w * buckets, (w + 1) * buckets
            d = [(b, p) for b, p in decl if lo <= b < hi]
            frames = [f for b, f in rec if lo <= b < hi]
            counts = tuple(sum(1 for b, _ in rec if b == t) for t in range(lo, hi))
            win = Window(key, w, record_digest(frames), counts, frames)
            if not d:
                win.why = EMPTY if not frames else "frames recorded where φ declares none"
            elif bad := [p for _, p in d if isinstance(p, KeyError)]:
                win.why = str(bad[0]).strip("'\"")
            else:
                want = [sum(1 for b, _ in d if b == t) for t in range(lo, hi)]
                win = _window(key, w, B, "", want, frames, [p for _, p in d], S.units)
                win.counts, win.digest = counts, record_digest(frames)
            out.append(win)
    return out


def lab_declared(S: Statement, block: str) -> tuple[list, list]:
    """The block's egress frames (its events in position order) and ingress frames (its requests in session order) as
    pieces, or a KeyError where a unit committed no values."""
    from . import lab as L

    B = S.phi.frame_bytes
    lab = L.Lab(S.model, S.hw, S.placement, S.phi.messages, B)
    messages: dict[int, list[tuple] | KeyError] = {}
    for stream, m in lab.messages.items():
        try:
            messages[stream] = S.message(m)
        except KeyError as e:
            messages[stream] = e
    egress = []
    for p, (j, m, c) in enumerate(lab.events[block]):
        if isinstance(messages[m.stream], KeyError):
            egress.append(messages[m.stream])
            continue
        nbytes = (m.bits + 7) // 8
        lo, hi = c * lab.chunk, min((c + 1) * lab.chunk, nbytes)
        head = b"id: %d\nevent: %d\ndata: " % (p, m.stream)
        n = len(head) + 4 * -(-(hi - lo) // 3) + 2
        payload = [const(head), ("b", tuple(_slice(messages[m.stream], 8 * lo, 8 * hi)), hi - lo), const(b"\n\n")]
        egress.append(_data_frame(DATA, j + 1, payload, n, B))
    ingress = []
    room = B - HEADER.size
    for s in lab.sessions[block]:
        bad = [messages[m.stream] for m in s.inbound if isinstance(messages[m.stream], KeyError)]
        n_frames = -(-lab.request_length(s) // room)
        if bad:
            ingress += bad[:1] * n_frames
            continue
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


def lab_windows(public: dict, witness: dict, published: dict[str, Sequence],
                salts: dict[tuple[str, int], bytes] | None = None) -> list[Window]:
    """The lab's windows: per block and link (`lab.EGRESS`, `lab.INGRESS`), each record's released frames, and the
    layout of the declared frames at the positions the earlier windows' counts reach, at the record's counts.

    With `salts`, each salted ingress window's salt commitment by (block, window): the layout puts the certifier's salt
    frame (`salt_frame`), a constant, in the slot the record holds a salt frame in (the first, behind any queue carried
    from the window before; the first slot when the record holds none), and the declared frames in the others."""
    from verity.core.protocols.certifier.grid import occupancy

    from . import lab as L

    S = statement(public, witness)
    B = public["frame_bytes"]
    out = []
    for block in sorted(published):
        egress, ingress = lab_declared(S, block)
        for link, declared in ((L.EGRESS, egress), (L.INGRESS, ingress)):
            at = 0
            for rec in sorted((r for r in published[block] if r.link == link), key=lambda r: r.window):
                frames = list(L.released([rec]))
                counts, reason = occupancy(rec.grid), rec.status.reason.value if rec.status.reason else ""
                win = Window(f"{block}:{link}", rec.window, record_digest(frames, reason), counts, frames)
                salt = (salts or {}).get((block, rec.window)) if link == L.INGRESS else None
                n = len(frames) - (salt is not None and len(frames) > 0)
                d, at = declared[at:at + n], at + n
                if not frames:
                    win.why = EMPTY
                elif len(d) < n:
                    win.why = f"frames past the Program's {len(declared)}"
                elif bad := [p for p in d if isinstance(p, KeyError)]:
                    win.why = str(bad[0]).strip("'\"")
                else:
                    if salt is not None:
                        slot = next((i for i, f in enumerate(frames) if _is_salt(f)), 0)
                        d = d[:slot] + [[const(salt_frame(rec.window, salt, B))]] + d[slot:]
                    win = _window(win.port, rec.window, B, reason, counts, frames, d, S.units)
                out.append(win)
    return out


# --- the demo binding -------------------------------------------------------------------------------------------------

DEMO = {
    "B": 16, "reason": "", "counts": [1, 0, 2], "rows": 2,
    "frames": [
        [const(encode_frame(LAST, F.HELLO_STREAM, F.HELLO.pack(0, 1), 16))],
        [const(HEADER.pack(DATA, 1, 3)), ("r", 0, 3, 13), ("r", 1, 250, 6), ("k", 5, 0), zeros(4)],
        [const(HEADER.pack(DATA, 1, 6)), ("b", (("r", 1, 0, 8), ("r", 0, 100, 8)), 2), const(b"\n\n"), zeros(1)],
    ],
}
"""A small layout of every piece kind, one compression: a HELLO-like constant frame, a frame of row bits, an event-like
frame of base64."""


# --- the command ------------------------------------------------------------------------------------------------------

def summarize(results: list[dict]) -> dict:
    by_port: dict[str, list[int]] = defaultdict(list)
    for r in results:
        if r["by"] == "unit" or not r["ok"]:
            by_port[r["port"]].append(r["window"])
    units = [r for r in results if r["by"] == "unit"]
    refused = [f"{r['port']}[{r['window']}]: {r['why']}" for r in results if not r["ok"] and r["by"] != "unit"]
    return {"ok": all(r["ok"] for r in results), "windows": len(results), "units": len(units),
            "units_ok": sum(r["ok"] for r in units), "refused_without_a_unit": refused,
            "not_ok": [f"{r['port']}[{r['window']}]" for r in results if not r["ok"]],
            "compressions_max": max((r["compressions"] for r in units), default=0),
            "seconds": round(sum(r.get("seconds", 0) for r in units), 3)}


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
    checked = [w.check() for w in sim_windows(public, witness, records, args.buckets) if w.why != EMPTY]
    return {"config": config, "buckets": args.buckets, "summary": summarize(checked), "windows": checked}


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
    windows = lab_windows(public, witness, published)
    results = [w.check(circuit=not args.reference_only) for w in windows if w.why != EMPTY]
    return {"run": str(run), "summary": summarize(results), "windows": results}


def _circuit_check(args) -> dict:
    """circuit-check's checks on DEMO's two Definitions, with the CLI's options and pins. The window's C-Flock lowering
    is its parts' (N3Match's here, Sha512Compress's and LmsEqual's in their own entries): lowered whole it would carry
    the SHA-512 compression that `checks.UNLOWERED` records as not fitting in 64 GiB, a family check a composite that
    calls it doesn't inherit. The unit is a Program of its own, never a served Program's Call, so the partition
    invariant doesn't apply (`as_call=False`, which also spares building the CLI's whole catalog to find that out)."""
    from circuit_check import checks as CK

    opts = CK.Options()
    entries = []
    for target, whole in ((bind(N3Match, LAYOUT=DEMO), True), (unit(DEMO), False)):
        t0 = time.time()
        try:
            r = CK.check_definition(target, opts, lower_whole=whole, as_call=False)
        except Exception as e:  # noqa: BLE001  (a target the checks can't run on is a failure by name, as the CLI's)
            r = {"definition": target.id, "failures": [{"name": f"error/raised {target.id}: {type(e).__name__}: {e}"}],
                 "warnings": []}
        entries.append({**r, "kind": "definition", "target": target.id, "seconds": round(time.time() - t0, 3)})
    rep = CK.report(entries, opts)
    rep["size"] = size(DEMO)
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
    q.add_argument("--reference-only", action="store_true", help="the Python reference alone, no circuit")
    q.set_defaults(fn=_lab)
    c = sub_.add_parser("circuit-check", help="circuit-check's checks on DEMO's binding and its N3Match part")
    c.set_defaults(fn=_circuit_check)
    for p in (s, q, c):
        p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    result = args.fn(args)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result.get("summary"), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
