"""N2's check of one certified record head, zero-knowledge: the head committed at leaf 2 + direction of N2's root
(`service.LinkWindows`) is the link-window's, complete, under the declared grid, and, where its end has an occupancy
profile, its counts are the profile's. A Boolean Definition, the gated part of N2's proof unit (`service`'s
`NetworkLinkWindow` v2), which runs on every link-window; its one output is the ok bit. That unit's Program holds it as
one gate (`gate`), and N2's verifier of record runs its `reference`, which the tests hold to the circuit.

    NetworkHeadCheck_v1{L, T, COUNTS}
        (record, record_domain: Digest, link: Array<L, Byte>, direction: Array<1>, window: Array<64>,
         grid: Array<80, Byte>, [profile: Array<T, Array<32>>], head: Array<HL, Byte>, salt: Salt,
         path: Array<2, Digest>) -> Value<1>
        Digest = Array<8, Array<64, Value<1>>> (SHA-512 words, `sha512.words_be`),
        Salt = Array<24, Array<64, Value<1>>>, Byte = Array<8, Value<1>>, HL = 157 + L + 4T

**Statics: the size class.** L is the UTF-8 length of the link's name, T the end's buckets a window, and COUNTS 1 when
the end has an occupancy profile (`LinkWindows.occupancy`), so `profile` is a parameter. All three are public: the
link's name and its grids are, and so is whether an end has a profile.

**Public inputs.** `record` is the root of N2's commitment to the link-window, `record_domain` its domain id
(`LinkWindows.domain`), `link` the link's name, `direction` the end (0 egress, 1 ingress), `window` the window's index
(u64, least significant bit first), `grid` the declared grid as the head encodes it (`grid_bytes`: T, r, B and τ_b as
u32 big-endian, then the parameters digest), and `profile` the profile's row for the window: T counts, each u32, least
significant bit first.

**Witness.** The head's bytes (`RecordHead.encode`, its frames root included), its leaf's hiding salt and its two
siblings in N2's tree.

**What 1 means.** (1) The head opens at leaf 2 + direction of N2's commitment (`NetworkHidingRead`: its sha512/row/v2
digest, hm96's hiding leaf, the frame-v3-sha512 read at depth 2), the leaf N3 opens for the same head
(`N3HidingRead`). (2) Its fields are the ones the public inputs give (`NetworkHeadFields`): its link's length and name,
its direction and window, the grid's T, r, B and τ_b, status 0 (complete) and the grid's parameters digest. (3) When
COUNTS, its T counts are the profile's row (`NetworkHeadCounts`). Two fields are free: the deadline byte, which the
certificate's host-side check reads (`LinkWindows.verify`), and the frames root, which N3 holds the window's frames to
on the windows it draws.

**Cost.** Beyond the opening, the fields are an equality of 736 + 8L bits and the counts one of 32·T bits, so
32·T − 1 ANDs (19,199 at T = 600), and two ANDs join the three. The opening is SHA-512 of the head's 157 + L + 4T bytes
after its row prefix's block, hm96's leaf and two frame-v3 nodes, and it dominates: at L = 9 it is 775,077 ANDs at T = 4
and 1,876,070 at T = 600, where the fields are 807 (`size` counts each part's ANDs).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import Callable, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import and_reduce
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array
from verity.ml.commitments import hm96
from verity.ml.commitments.gates.frame_v3 import FrameV3Read, root_ref
from verity.ml.commitments.gates.hm96 import Hm96Sha512Leaf
from verity.ml.commitments.gates.lms import Sha512Compress
from verity.ml.commitments.gates.sha512 import midstate, pad_tail, words_be
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.ml.commitments.rowleaf import sha512_row_v2_digest, sha512_row_v2_prefix
from verity.core.protocols.certifier.certificate import DIGEST_BYTES, STATUS_CODES
from verity.core.protocols.certifier.grid import GridParams

BIT = BL.BIT
BYTE = Array(8, BIT)
WORD = Array(64, BIT)
DIGEST = Array(8, WORD)
SALT = Array(hm96.SHA512.salt_bytes // 8, WORD)
#: the head's leaf in N2's tree is HEADS + direction (`service.HEADS`), so its position bits are (direction, 1)
HEADS = 2
GRID_BYTES = 16 + DIGEST_BYTES
COMPLETE = STATUS_CODES[None]


class Shape(SimpleNamespace):
    """A head's byte offsets (`RecordHead.encode`) for a link name of L bytes and T buckets: the direction, the window,
    the grid's four u32s, the status, the deadline, the parameters digest, the counts and the frames root."""

    def __init__(self, L: int, T: int) -> None:
        if not all(type(x) is int for x in (L, T)) or not 0 <= L < 1 << 16 or T < 1:
            raise ValueError(f"a head has 0 <= L < 2^16 and T >= 1, not L={L!r}, T={T!r}")
        super().__init__(L=L, T=T, DIR=2 + L, WIN=3 + L, GRID=11 + L, ST=27 + L, DL=28 + L, PAR=29 + L, CNT=93 + L,
                         ROOT=93 + L + 4 * T, HL=157 + L + 4 * T)


def grid_bytes(params: GridParams, parameters: bytes) -> bytes:
    """The declared grid as a head encodes it: T, r, B and τ_b (u32 big-endian), then the parameters digest."""
    if len(parameters) != DIGEST_BYTES:
        raise ValueError(f"a parameters digest is {DIGEST_BYTES} bytes")
    return struct.pack(">IIII", params.buckets, params.allocation, params.frame_bytes, params.bucket_ms) + parameters


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

    d = CompositeDefinition(name, 1, statics, signature, body, conformance="exact-model-tested", doc=doc)
    d.traced = traced  # type: ignore[attr-defined]
    return d


def _same(x, want) -> object:
    """1 iff bit x is `want`: a bit of the circuit, or the constant 0 or 1."""
    if isinstance(want, int):
        return x if want else x.inv()
    return (x ^ want).inv()


def _sig_counts(S):
    return (("counts", Array(S.T, Array(4, BYTE))), ("profile", Array(S.T, Array(32, BIT)))), BIT


def _build_counts(C, S, *e):
    counts, profile = e[:4 * S.T], e[4 * S.T:]
    # count t is u32 big-endian at bytes 4t ... 4t + 3: its bit j is bit j % 8 of byte 4t + 3 - j // 8
    return [and_reduce(C, [_same(counts[4 * t + 3 - j // 8][j % 8], profile[t][j]) for t in range(S.T)
                           for j in range(32)])]


NetworkHeadCounts = _part(
    "NetworkHeadCounts", ("T",), _sig_counts, _build_counts,
    "1 iff the head's T counts (u32 big-endian) are the profile's (u32, least significant bit first): 32·T − 1 ANDs")


def _sig_fields(S):
    sh = Shape(S.L, 1)
    return ((("head", Array(sh.CNT, BYTE)), ("link", Array(S.L, BYTE)), ("direction", Array(1, BIT)),
             ("window", Array(64, BIT)), ("grid", Array(GRID_BYTES, BYTE))), BIT)


def _build_fields(C, S, *e):
    sh = Shape(S.L, 1)
    head, link, grid = e[:sh.CNT], e[sh.CNT:sh.CNT + S.L], e[sh.CNT + S.L + 2:]
    direction, window = e[sh.CNT + S.L], e[sh.CNT + S.L + 1]

    def const(v: int) -> list[int]:
        return [v >> b & 1 for b in range(8)]

    want = [(0, const(S.L >> 8)), (1, const(S.L & 0xFF)), *((2 + i, link[i]) for i in range(S.L)),
            (sh.DIR, [direction[0]] + [0] * 7), *((sh.WIN + i, window[8 * (7 - i):8 * (8 - i)]) for i in range(8)),
            *((sh.GRID + i, grid[i]) for i in range(16)), (sh.ST, const(COMPLETE)),
            *((sh.PAR + i, grid[16 + i]) for i in range(DIGEST_BYTES))]
    return [and_reduce(C, [_same(head[at][b], byte[b]) for at, byte in want for b in range(8)])]


NetworkHeadFields = _part(
    "NetworkHeadFields", ("L",), _sig_fields, _build_fields,
    "1 iff the head's bytes before its counts, its deadline byte aside, are u16 L, the link, the direction, the window "
    "(u64 big-endian), the grid's T, r, B and τ_b, status 0 and the grid's parameters digest")


# --- the circuit: the opening -----------------------------------------------------------------------------------------

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


def _words(data: Sequence[Sequence]) -> list[list]:
    """Bytes (8 bits each, least significant first) as SHA-512 words: 64 bits least significant first, bytes
    big-endian."""
    return [[x for b in range(8) for x in data[8 * k + 7 - b]] for k in range(len(data) // 8)]


def _sig_rowdigest(S):
    return (("row", Array(S.N_BITS, BIT)),), DIGEST


def _body_rowdigest(B, S, row):
    K = _Consts(B)
    nbytes = -(-S.N_BITS // 8)
    bits = _refs(row) + K.bits(0, 8 * nbytes - S.N_BITS)
    data = [bits[8 * k:8 * k + 8] for k in range(nbytes)] + [K.bits(b, 8) for b in pad_tail(128 + nbytes)]
    words = _words(data)
    out = _coll(DIGEST, [x for v in midstate(sha512_row_v2_prefix(S.N_BITS)) for x in K.bits(v, 64)])
    for k in range(0, len(words), 16):
        out = B.call(Sha512Compress, out, _coll(Array(16, WORD), [x for w in words[k:k + 16] for x in w]))
    return out


NetworkRowDigest = CompositeDefinition(
    "NetworkRowDigest", 1, ("N_BITS",), _sig_rowdigest, _body_rowdigest, conformance="exact-model-tested",
    doc="the sha512/row/v2 digest of an N_BITS-bit row: SHA-512 from its 128-byte prefix block's midstate over the "
        "row's bytes")


def _sig_hread(S):
    return (("row", Array(S.N_BITS, BIT)), ("salt", SALT), ("root", DIGEST), ("domain", DIGEST),
            ("pos", Array(S.D, BIT)), ("path", Array(S.D, DIGEST))), BIT


def _body_hread(B, S, row, salt, root, domain, pos, path):
    x = B.call(bind(NetworkRowDigest, N_BITS=S.N_BITS), row)
    leaf = B.call(Hm96Sha512Leaf, x, salt)[2]
    return B.call(bind(FrameV3Read, D=S.D, SCHEMA=SCHEMA), root, domain, leaf, pos, path)


NetworkHidingRead = CompositeDefinition(
    "NetworkHidingRead", 1, ("N_BITS", "D"), _sig_hread, _body_hread, conformance="exact-model-tested",
    doc="1 iff the N_BITS-bit row with its salt opens at pos in the depth-D tree (hm96-sha512/row/v2 leaves) of root "
        "and domain id domain")


# --- the head check ---------------------------------------------------------------------------------------------------

def _check_statics(S) -> Shape:
    if S.COUNTS not in (0, 1) or isinstance(S.COUNTS, bool):
        raise ValueError(f"COUNTS is 0 or 1, not {S.COUNTS!r}")
    return Shape(S.L, S.T)


def _params(S) -> list[tuple[str, object]]:
    sh = _check_statics(S)
    params = [("record", DIGEST), ("record_domain", DIGEST), ("link", Array(S.L, BYTE)), ("direction", Array(1, BIT)),
              ("window", Array(64, BIT)), ("grid", Array(GRID_BYTES, BYTE))]
    if S.COUNTS:
        params.append(("profile", Array(S.T, Array(32, BIT))))
    return params + [("head", Array(sh.HL, BYTE)), ("salt", SALT), ("path", Array(2, DIGEST))]


def _sig_check(S):
    return tuple(_params(S)), BIT


def _body_check(B, S, *args):
    sh = _check_statics(S)
    a = dict(zip([k for k, _ in _params(S)], args))
    K = _Consts(B)
    hb = _refs(a["head"])
    oks = [B.call(bind(NetworkHidingRead, N_BITS=8 * sh.HL, D=2), a["head"].reshape(Array(8 * sh.HL, BIT)), a["salt"],
                  a["record"], a["record_domain"], _coll(Array(2, BIT), _refs(a["direction"]) + K.bits(1, 1)),
                  a["path"]),
           B.call(bind(NetworkHeadFields, L=S.L), _coll(Array(sh.CNT, BYTE), hb[:8 * sh.CNT]), a["link"],
                  a["direction"], a["window"], a["grid"])]
    if S.COUNTS:
        oks.append(B.call(bind(NetworkHeadCounts, T=S.T), _coll(Array(S.T, Array(4, BYTE)), hb[8 * sh.CNT:8 * sh.ROOT]),
                          a["profile"]))
    bits = concat([o.refs for o in oks])
    return B.call(BL.and_gate(bits.count), Coll(Array(bits.count, BIT), bits))


NetworkHeadCheck = CompositeDefinition(
    "NetworkHeadCheck", 1, ("L", "T", "COUNTS"), _sig_check, _body_check, conformance="exact-model-tested",
    doc="1 iff the head opens at leaf 2 + direction of N2's commitment, its fields are the link-window's, complete and "
        "the declared grid's, and (COUNTS) its counts are the profile's row; its deadline byte and frames root are free")


def unit(L: int, T: int, counts: bool):
    return bind(NetworkHeadCheck, L=L, T=T, COUNTS=int(counts))


def _gate_only(fid: str) -> Callable[..., int]:
    def evaluate(*_args: int) -> int:
        raise NotImplementedError(f"the gate stands for {fid}: evaluate that Definition (`evaluate`) or its `reference`")
    return evaluate


@lru_cache(maxsize=None)
def gate(L: int, T: int, counts: bool) -> PrimitiveDefinition:
    """`unit(L, T, counts)` as one gate of N2's proof unit (`service.LinkWindows.unit`): its signature, under an id that
    names it, `Gate[NetworkHeadCheck_v1{L=..,T=..,COUNTS=..}]_v1`. The unit's Program holds the gate rather than the
    Definition's body because the commitment gadgets' Boolean bodies (`Hm96Sha512Leaf`, `FrameV3Node`, `FrameV3Leaf`,
    `Sha512Compress`) encode to about 140 MB, and the proof service encodes the Program on every registration."""
    d = unit(L, T, counts)
    params, ret = d.signature()
    return PrimitiveDefinition(f"Gate[{d.id}]", 1, params, ret, _gate_only(d.id), register=False,
                               doc=f"{d.id}, N2's head check, as one gate: the Definition is registered and circuit-checked")


# --- the reference and the arguments ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Instance:
    """One head's check: its size class and its arguments as bytes and ints (`profile` None when the end has no
    profile; `path` the head leaf's two siblings)."""

    record: bytes
    record_domain: bytes
    link: bytes
    direction: int
    window: int
    grid: bytes
    profile: tuple[int, ...] | None
    head: bytes
    salt: bytes
    path: tuple[bytes, bytes]
    T: int

    @property
    def counts(self) -> bool:
        return self.profile is not None

    @property
    def unit(self):
        return unit(len(self.link), self.T, self.counts)

    @property
    def key(self) -> tuple:
        return (len(self.link), self.T, self.counts)


def hiding_value(row: bytes, salt: bytes) -> bytes:
    """The 64 bytes a row's frame-v3-sha512 leaf binds: hm96's tree leaf over the row's sha512/row/v2 digest."""
    inst, key = hm96.SHA512, hm96.SHA512.default_key
    return inst.tree_leaf(key, inst.commit_string(key, sha512_row_v2_digest(row, 8 * len(row)), salt))


def reference_parts(inst: Instance) -> dict[str, bool]:
    """Each part's output on the instance, from its arguments as the circuit reads them: the opening
    (`NetworkHidingRead`), the fields (`NetworkHeadFields`) and the counts (`NetworkHeadCounts`, 1 without a profile)."""
    sh, h = Shape(len(inst.link), inst.T), inst.head
    if len(h) != sh.HL:
        raise ValueError(f"the head is {len(h)} bytes, the unit's {sh.HL}")
    fields = (h[:2] == sh.L.to_bytes(2, "big") and h[2:sh.DIR] == inst.link and h[sh.DIR] == inst.direction
              and h[sh.WIN:sh.GRID] == inst.window.to_bytes(8, "big") and h[sh.GRID:sh.ST] == inst.grid[:16]
              and h[sh.ST] == COMPLETE and h[sh.PAR:sh.CNT] == inst.grid[16:])
    counts = inst.profile is None or h[sh.CNT:sh.ROOT] == b"".join(n.to_bytes(4, "big") for n in inst.profile)
    opens = root_ref(inst.record_domain, SCHEMA, hiding_value(h, inst.salt), HEADS + inst.direction,
                     inst.path) == inst.record
    return {"opening": bool(opens), "fields": bool(fields), "counts": bool(counts)}


def reference(inst: Instance) -> bool:
    """`NetworkHeadCheck_v1`'s output on the instance: every part's."""
    return all(reference_parts(inst).values())


def _digest_bits(d: bytes) -> list[int]:
    return [(w >> j) & 1 for w in words_be(d) for j in range(64)]


def _int_bits(x: int, w: int) -> list[int]:
    return [(x >> j) & 1 for j in range(w)]


def _bytes_bits(data: bytes) -> list[int]:
    return _int_bits(int.from_bytes(data, "little"), 8 * len(data))


def arguments(inst: Instance) -> list[list[int]]:
    """The unit's arguments for one instance in its signature's order, each its leaves' bits, 0 or 1."""
    flat = {"record": _digest_bits(inst.record), "record_domain": _digest_bits(inst.record_domain),
            "link": _bytes_bits(inst.link), "direction": [inst.direction], "window": _int_bits(inst.window, 64),
            "grid": _bytes_bits(inst.grid), "profile": [b for n in inst.profile or () for b in _int_bits(n, 32)],
            "head": _bytes_bits(inst.head), "salt": _digest_bits(inst.salt),
            "path": [b for s in inst.path for b in _digest_bits(s)]}
    params, _ = inst.unit.signature()
    out = []
    for name, t in params:
        if len(flat[name]) != t.leaves:
            raise ValueError(f"argument {name} has {len(flat[name])} bits, the unit takes {t.leaves}")
        out.append(flat[name])
    return out


def evaluate(*insts: Instance) -> list[int]:
    """The unit's output on each instance, all of one unit, in one evaluation."""
    if len({i.key for i in insts}) != 1:
        raise ValueError("the instances are of one unit")
    args = [list(rows) for rows in zip(*(arguments(i) for i in insts))]
    return [int(b) for b in evaluate_bits(insts[0].unit, *args).reshape(-1).tolist()]


def _times(nd) -> int:
    return 1 if nd.form in ("primitive", "call") else nd.n


def ands(fn, memo: dict | None = None) -> int:
    """A Boolean Definition's AND gates, an n-input AND counted as n − 1."""
    memo = {} if memo is None else memo
    if fn.id not in memo:
        if isinstance(fn, PrimitiveDefinition):
            kind = BL.gate_kind(fn)
            memo[fn.id] = kind[1] - 1 if kind is not None and kind[0] == "and" else 0
        else:
            memo[fn.id] = sum(_times(nd) * ands(nd.fn, memo) for nd in fn.body.nodes)
    return memo[fn.id]


_GROUPS = {"NetworkHidingRead": "opening", "NetworkHeadFields": "fields", "NetworkHeadCounts": "counts"}


def size(target, memo: dict | None = None) -> dict:
    """The head check's ANDs, in total and by part: the opening, the fields, the counts and the rest (the final AND)."""
    memo = {} if memo is None else memo
    out = dict.fromkeys(("opening", "fields", "counts", "other"), 0)
    for nd in target.body.nodes:
        out[_GROUPS.get(getattr(nd.fn, "name", ""), "other")] += _times(nd) * ands(nd.fn, memo)
    out["ands"] = sum(out.values())
    return out


__all__ = ["GRID_BYTES", "HEADS", "Instance", "NetworkHeadCheck", "NetworkHeadCounts", "NetworkHeadFields",
           "NetworkHidingRead", "NetworkRowDigest", "Shape", "ands", "arguments", "evaluate", "gate", "grid_bytes",
           "reference", "reference_parts", "size", "unit"]
