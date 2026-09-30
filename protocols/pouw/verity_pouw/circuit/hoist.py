"""Hoisting: a per-row function's call-wide part, computed once.

`schemes.fp8_is` hashes long call-wide prefixes into every row's message: the salt, the matmul index, root_A and the weight
id in each E₁ row and F₁ column stream, and the commitment domain in every Merkle leaf and node.  Computed in each row's unit,
those gates would compute the same function of the same values in many units, which `validate_unit_cut` rejects
(``gate-recomputed``).  So a row's function is built once as a small DAG over named inputs, each marked call-wide (shared)
or the row's own (local), and split:

* the **shared** part (every gate whose operands are constants or shared values) goes into the call's one unit, which
  returns the **frontier**, the shared values some local gate reads;
* the **local** part (every gate with a local operand) is a Definition of its own that reads the frontier as one committed
  array, so each of its gates depends on the row and no two rows compute the same value.

The call-wide units are cut once more (`public_part`): their gates over the salt and the weight id alone would compute the
same values in every call of a run, or of a weight, which a window's cut rejects.  Those are functions of the verifier's
anchors, so the verifier evaluates them natively (`evaluate_cone`) and the unit reads the values it needs as an anchored input.

Constants fold as the DAG is built.  A hash block whose inputs are all shared, or all local, is one call of the hash's
Definition (`hashes.KeccakF1600`, `Sha256Compress`); a block that mixes them is expanded into primitives, so that only its
shared gates are hoisted.  The message builders pack bytes exactly as `verity.randomness.frame` and
`verity.commitments.merkle` lay them out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, Union

from verity.ir.defs import CompositeDefinition, bind
from verity.ir.refs import Coll, array_of
from verity.ir.types import Array, Value
from verity.ml.prims import const

from . import fp8_words as FW
from . import words as W
from .hashes import KECCAK_RC, KECCAK_ROT, KeccakF1600, RATE_LANES

M32 = W.M32
U32 = W.U32


# ---- the DAG --------------------------------------------------------------------------------------------------------------

class Node:
    __slots__ = ("i", "kind", "fn", "args", "shared", "name", "src", "k", "py")

    def __init__(self, i: int, kind: str, shared: bool, fn=None, args=(), name=None, src=None, k: int = 0, py=None):
        self.i, self.kind, self.shared, self.fn, self.args, self.name, self.src, self.k, self.py = (
            i, kind, shared, fn, args, name, src, k, py)

    def __repr__(self) -> str:
        return f"<{self.kind} {self.i}{' S' if self.shared else ''} {self.name or getattr(self.fn, 'id', '')}>"


Val = Union[Node, int]


def _shared(args) -> bool:
    return all(isinstance(a, int) or a.shared for a in args)


class Graph:
    """A straight-line program over named inputs, constants folded as it is built."""

    def __init__(self) -> None:
        self.nodes: list[Node] = []

    def _add(self, **kw) -> Node:
        n = Node(len(self.nodes), **kw)
        self.nodes.append(n)
        return n

    def input(self, name: Any, shared: bool) -> Node:
        return self._add(kind="in", shared=shared, name=name)

    def op(self, fn, *args: Val) -> Val:
        if all(isinstance(a, int) for a in args):
            return fn.evaluate(*args)
        return self._add(kind="op", shared=_shared(args), fn=fn, args=args)

    def call(self, fn, groups: Sequence[Sequence[Val]], outs: int, py: Callable[..., list[int]]) -> list[Val]:
        """A composite call: ``groups`` are its parameters' leaves, ``py`` its semantics on ints (for folding and for
        `evaluate`)."""
        flat = [a for grp in groups for a in grp]
        if all(isinstance(a, int) for a in flat):
            return list(py(*groups))
        c = self._add(kind="call", shared=_shared(flat), fn=fn, args=tuple(tuple(grp) for grp in groups), py=py)
        return [self._add(kind="out", shared=c.shared, src=c, k=j) for j in range(outs)]


def evaluate(g: Graph, outputs: Sequence[Val], env: dict) -> list[int]:
    """The outputs' values with the inputs named in ``env``."""
    val: dict[int, Any] = {}
    get = lambda a: a if isinstance(a, int) else val[a.i]  # noqa: E731
    for n in g.nodes:
        if n.kind == "in":
            val[n.i] = env[n.name]
        elif n.kind == "op":
            val[n.i] = n.fn.evaluate(*(get(a) for a in n.args))
        elif n.kind == "call":
            val[n.i] = n.py(*[[get(a) for a in grp] for grp in n.args])
        else:
            val[n.i] = val[n.src.i][n.k]
    return [get(o) for o in outputs]


def _deps(n: Node) -> list[Node]:
    if n.kind == "op":
        return [a for a in n.args if not isinstance(a, int)]
    if n.kind == "call":
        return [a for grp in n.args for a in grp if not isinstance(a, int)]
    if n.kind == "out":
        return [n.src]
    return []


@dataclass
class Split:
    """A function's DAG cut into its shared and local parts."""

    graph: Graph
    outputs: list[Val]
    frontier: list[Node] = field(default_factory=list)         # shared computed values a local gate or output reads
    direct: list[Node] = field(default_factory=list)           # shared inputs a local gate reads
    hoisted: list[Node] = field(default_factory=list)          # the shared gates the frontier needs, in order
    local: list[Node] = field(default_factory=list)            # the local gates the outputs need, in order
    local_inputs: list[Node] = field(default_factory=list)


def split(g: Graph, outputs: Sequence[Val]) -> Split:
    live: set[int] = set()
    stack = [o for o in outputs if not isinstance(o, int)]
    while stack:
        n = stack.pop()
        if n.i in live:
            continue
        live.add(n.i)
        stack.extend(_deps(n))
    s = Split(g, list(outputs))
    fset: dict[int, Node] = {}
    dset: dict[int, Node] = {}
    for n in g.nodes:
        if n.i not in live:
            continue
        if n.kind == "in":
            (s.local_inputs if not n.shared else []).append(n)
            continue
        if n.shared:
            continue
        s.local.append(n)
        for a in _deps(n):
            if a.shared:
                (dset if a.kind == "in" else fset).setdefault(a.i, a)
    for o in outputs:
        if not isinstance(o, int) and o.shared:
            raise ValueError("a hoisted function's outputs depend on its local inputs")
    s.frontier = sorted(fset.values(), key=lambda n: n.i)
    s.direct = sorted(dset.values(), key=lambda n: n.i)
    need: set[int] = set()
    stack = list(s.frontier)
    while stack:
        n = stack.pop()
        if n.i in need or n.kind == "in":
            continue
        need.add(n.i)
        stack.extend(_deps(n))
    s.hoisted = [n for n in g.nodes if n.i in need]
    return s


@dataclass
class Public:
    """A call-wide unit's gates cut at its **public** part: every gate whose operands are constants, public inputs or public
    gates is a function of the verifier's anchors alone, so the verifier evaluates it natively (`evaluate_cone`) and the
    unit reads the public values its other gates need (``frontier``) as one anchored input."""

    graph: Graph
    frontier: list[Node]                       # public gates a remaining gate reads, in graph order
    rest: list[Node]                           # the gates the unit still computes, in order


def public_part(g: Graph, nodes: Sequence[Node], public: Callable[[Any], bool], outputs: Sequence[Val] = ()) -> Public:
    """Cut ``nodes`` (a unit's gates of ``g``, in order) at the gates over inputs named ``public`` alone.  A public input read
    by a remaining gate, or a public value among the unit's ``outputs``, would need the unit to return an anchor as its own
    value, and is refused."""
    pub: set[int] = set()
    for n in g.nodes:
        if n.kind == "in":
            if public(n.name):
                pub.add(n.i)
        elif (ds := _deps(n)) and all(d.i in pub for d in ds):
            pub.add(n.i)
    rest = [n for n in nodes if n.i not in pub]
    front: dict[int, Node] = {}
    for n in rest:
        for d in _deps(n):
            if d.i in pub:
                if d.kind == "in":
                    raise AssertionError(f"a remaining gate reads the public input {d.name} directly")
                front.setdefault(d.i, d)
    if any(not isinstance(o, int) and o.i in pub for o in outputs):
        raise AssertionError("a unit's output is public")
    return Public(g, sorted(front.values(), key=lambda n: n.i), rest)


def _live(outputs: Sequence[Val]) -> set[int]:
    live: set[int] = set()
    stack = [o for o in outputs if not isinstance(o, int)]
    while stack:
        n = stack.pop()
        if n.i not in live:
            live.add(n.i)
            stack.extend(_deps(n))
    return live


def cone(g: Graph, outputs: Sequence[Val]) -> list[Node]:
    """The gates ``outputs`` need, in graph order."""
    live = _live(outputs)
    return [n for n in g.nodes if n.i in live and n.kind != "in"]


def evaluate_cone(g: Graph, outputs: Sequence[Val], env: dict) -> list[int]:
    """`evaluate` over only the gates ``outputs`` need, so ``env`` names only the inputs they read."""
    live = _live(outputs)
    val: dict[int, Any] = {}
    get = lambda a: a if isinstance(a, int) else val[a.i]  # noqa: E731
    for n in g.nodes:
        if n.i not in live:
            continue
        if n.kind == "in":
            val[n.i] = env[n.name]
        elif n.kind == "op":
            val[n.i] = n.fn.evaluate(*(get(a) for a in n.args))
        elif n.kind == "call":
            val[n.i] = n.py(*[[get(a) for a in grp] for grp in n.args])
        else:
            val[n.i] = val[n.src.i][n.k]
    return [get(o) for o in outputs]


def _leaf_widths(t) -> list[int]:
    if isinstance(t, Value):
        return [t.w]
    if isinstance(t, Array):
        return _leaf_widths(t.elem) * t.n
    raise TypeError(f"unsupported parameter type {t}")


class Emitter:
    """Emits DAG nodes into a body; constants deduplicated per (width, value)."""

    def __init__(self, B, env: dict[int, Coll]) -> None:
        self.B, self.env = B, env
        self._consts: dict[tuple[int, int], Coll] = {}

    def const(self, w: int, v: int) -> Coll:
        key = (w, v & ((1 << w) - 1))
        got = self._consts.get(key)
        if got is None:
            got = self._consts[key] = self.B.call(const(*key))
        return got

    def val(self, a: Val, w: int) -> Coll:
        return self.const(w, a) if isinstance(a, int) else self.env[a.i]

    def emit(self, nodes: Sequence[Node]) -> None:
        for n in nodes:
            if n.kind == "op":
                params, _ = n.fn.signature()
                self.env[n.i] = self.B.call(n.fn, *(self.val(a, t.w) for a, (_, t) in zip(n.args, params)))
            elif n.kind == "call":
                params, _ = n.fn.signature()
                colls = []
                for grp, (_, t) in zip(n.args, params):
                    ws = _leaf_widths(t)
                    colls.append(array_of([self.val(a, w) for a, w in zip(grp, ws)]) if isinstance(t, Array)
                                 else self.val(grp[0], ws[0]))
                self.env[n.i] = self.B.call(n.fn, *colls)
            elif n.kind == "out":
                self.env[n.i] = self.env[n.src.i][n.k]


# ---- SHA-256 ----------------------------------------------------------------------------------------------------------------

SHA256_K = (
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
)
SHA256_IV = (0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19)


class _Ops:
    """The word operations over a `Graph` (or, with a fresh one and ints, over ints)."""

    def __init__(self, g: Graph) -> None:
        self.g = g

    def add(self, a: Val, b: Val) -> Val:
        return self.g.op(FW.PouwAdd32, a, b)

    def xor(self, a: Val, b: Val) -> Val:
        if isinstance(a, int) and a == 0:
            return b
        if isinstance(b, int) and b == 0:
            return a
        return self.g.op(W.PouwXor32, a, b)


def _sha256_rounds(o: _Ops, st: Sequence[Val], blk: Sequence[Val]) -> list[Val]:
    g = o.g
    w = list(blk)
    for t in range(16, 64):
        s = o.add(g.op(FW.PouwSha256SmallSigma1, w[t - 2]), w[t - 7])
        w.append(o.add(o.add(s, g.op(FW.PouwSha256SmallSigma0, w[t - 15])), w[t - 16]))
    a, b, c, d, e, f, gg, h = st
    for t in range(64):
        t1 = o.add(o.add(o.add(o.add(h, g.op(FW.PouwSha256BigSigma1, e)), g.op(W.PouwCh32, e, f, gg)), SHA256_K[t]), w[t])
        t2 = o.add(g.op(FW.PouwSha256BigSigma0, a), g.op(W.PouwMaj32, a, b, c))
        h, gg, f, e, d, c, b, a = gg, f, e, o.add(d, t1), c, b, a, o.add(t1, t2)
    return [o.add(x, y) for x, y in zip(st, (a, b, c, d, e, f, gg, h))]


def _sha256_py(st, blk) -> list[int]:
    return _sha256_rounds(_Ops(Graph()), list(st), list(blk))


def _compress_body(B, S, st, blk):
    g = Graph()
    ins = [g.input(("s", i), False) for i in range(8)] + [g.input(("b", i), False) for i in range(16)]
    outs = _sha256_rounds(_Ops(g), ins[:8], ins[8:])
    env = {n.i: c for n, c in zip(ins, [st[i] for i in range(8)] + [blk[i] for i in range(16)])}
    e = Emitter(B, env)
    e.emit([n for n in g.nodes if n.kind != "in"])
    return array_of([e.val(o, 32) for o in outs])


Sha256Compress = bind(CompositeDefinition(
    "PouwSha256Compress", 1, (), lambda S: ((("state", Array(8, U32)), ("block", Array(16, U32))), Array(8, U32)),
    _compress_body, doc="One SHA-256 compression (FIPS 180-4) over big-endian words.", conformance="protocol-defined"))


def _uniform(vals: Sequence[Val]) -> bool:
    """Whether a block's inputs are all gates of one kind (all shared or all local): then it is one call."""
    return all(not isinstance(v, int) for v in vals) and len({v.shared for v in vals}) == 1


def sha256_compress(g: Graph, st: Sequence[Val], blk: Sequence[Val]) -> list[Val]:
    if _uniform(list(st) + list(blk)):
        return g.call(Sha256Compress, [st, blk], 8, _sha256_py)
    return _sha256_rounds(_Ops(g), list(st), list(blk))


# ---- Keccak-f[1600] and SHAKE256 --------------------------------------------------------------------------------------------

def _keccak_rounds(o: _Ops, s: Sequence[Val]) -> list[Val]:
    g = o.g

    def x2(p, q):
        return o.xor(p[0], q[0]), o.xor(p[1], q[1])

    def rotl(p, c):
        if c % 64 == 0:
            return p
        return g.op(W.PouwRotl64Lo, p[0], p[1], c), g.op(W.PouwRotl64Hi, p[0], p[1], c)

    def chi(a, b, c):
        return g.op(W.PouwChi32, a[0], b[0], c[0]), g.op(W.PouwChi32, a[1], b[1], c[1])

    a = [[(s[2 * (x + 5 * y)], s[2 * (x + 5 * y) + 1]) for y in range(5)] for x in range(5)]
    for r in range(24):
        col = []
        for x in range(5):
            p = a[x][0]
            for y in range(1, 5):
                p = x2(p, a[x][y])
            col.append(p)
        d = [x2(col[(x - 1) % 5], rotl(col[(x + 1) % 5], 1)) for x in range(5)]
        a = [[x2(a[x][y], d[x]) for y in range(5)] for x in range(5)]
        b = [[(0, 0)] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                b[y][(2 * x + 3 * y) % 5] = rotl(a[x][y], KECCAK_ROT[x][y])
        a = [[chi(b[x][y], b[(x + 1) % 5][y], b[(x + 2) % 5][y]) for y in range(5)] for x in range(5)]
        lo, hi = KECCAK_RC[r] & M32, KECCAK_RC[r] >> 32
        a[0][0] = (o.xor(a[0][0][0], lo), o.xor(a[0][0][1], hi))
    return [h for y in range(5) for x in range(5) for h in a[x][y]]


def _keccak_py(s) -> list[int]:
    return _keccak_rounds(_Ops(Graph()), list(s))


def keccak(g: Graph, s: Sequence[Val]) -> list[Val]:
    if _uniform(s):
        return g.call(KeccakF1600, [s], 50, _keccak_py)
    return _keccak_rounds(_Ops(g), list(s))


# ---- messages as bytes --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Byte:
    """Byte ``(v >> shift) & 0xFF`` of a 32-bit value ``v``."""

    v: Val
    shift: int


@dataclass(frozen=True)
class Byte8:
    """A U8 value."""

    v: Val


Item = Union[int, Byte, Byte8]


def const_bytes(b: bytes) -> list[Item]:
    return list(b)


def be_words(ws: Sequence[Val]) -> list[Item]:
    """The bytes of big-endian 32-bit words (a SHA-256 digest)."""
    return [Byte(w, 24 - 8 * j) for w in ws for j in range(4)]


def uint_bytes(hi_lo: Sequence[Val], n: int) -> list[Item]:
    """The last ``n`` big-endian bytes of the 64-bit value ``(hi, lo)``: `verity.randomness.frame`'s and
    `verity.commitments.merkle`'s minimal integer encoding when the value needs exactly ``n`` bytes."""
    hi, lo = hi_lo
    out: list[Item] = []
    for t in range(n):
        o = 8 * (n - 1 - t)
        out.append(Byte(lo, o) if o < 32 else Byte(hi, o - 32))
    return out


def u32_bytes(v: Val, n: int) -> list[Item]:
    """The last ``n`` big-endian bytes of a 32-bit value."""
    return [Byte(v, 8 * (n - 1 - t)) for t in range(n)]


def frame(kind: bytes, body: Sequence[Item]) -> list[Item]:
    """`verity.randomness.frame`: the kind byte, the body's 8-byte big-endian length, the body."""
    return list(kind) + list(len(body).to_bytes(8, "big")) + list(body)


MERKLE_FRAME = b"veritor/protocol/merkle/frame/v3\0"


def merkle_message(tag: bytes, *parts: Sequence[Item]) -> list[Item]:
    """`verity.commitments.merkle._hash`'s message: the frame, the tag's 4-byte length and the tag, then each part's 8-byte
    length and the part."""
    out = list(MERKLE_FRAME) + list(len(tag).to_bytes(4, "big")) + list(tag)
    for p in parts:
        out += list(len(p).to_bytes(8, "big")) + list(p)
    return out


def _pack(g: Graph, items: Sequence[Item], big: bool) -> Val:
    """One 32-bit word from four byte items (big-endian: item 0 is bits 24..31; little-endian: bits 0..7)."""
    pos = [24 - 8 * j if big else 8 * j for j in range(4)]
    if big and all(isinstance(it, Byte8) for it in items):
        return g.op(FW.PouwPackBE4U8, *(it.v for it in items))
    c, parts = 0, []
    j = 0
    while j < 4:
        it = items[j]
        if isinstance(it, int):
            c |= (it & 0xFF) << pos[j]
            j += 1
        elif isinstance(it, Byte8):
            if isinstance(it.v, int):
                c |= (it.v & 0xFF) << pos[j]
            else:
                parts.append(g.op(FW.PouwPlace8, it.v, pos[j]))
            j += 1
        else:
            r = 1                                            # a run of bytes of one value, in the word's own byte order
            step = -8 if big else 8
            while (j + r < 4 and isinstance(items[j + r], Byte) and items[j + r].v is it.v
                   and items[j + r].shift == it.shift + step * r):
                r += 1
            s_lo = min(it.shift, it.shift + step * (r - 1))
            d_lo = min(pos[j], pos[j + r - 1])
            if isinstance(it.v, int):
                c |= (((it.v & M32) >> s_lo) & ((1 << 8 * r) - 1)) << d_lo
            elif s_lo == d_lo == 0 and r == 4:
                parts.append(it.v)
            else:
                parts.append(g.op(FW.PouwMove32, it.v, s_lo, 8 * r, d_lo))
            j += r
    o = _Ops(g)
    acc: Val = 0
    for p in parts:
        acc = o.xor(acc, p)
    return o.xor(acc, c) if c else acc


def sha256(g: Graph, msg: Sequence[Item], iv: Sequence[Val] = SHA256_IV) -> list[Val]:
    """SHA-256 of the message's bytes: the digest as 8 big-endian words."""
    n = len(msg)
    m = list(msg) + [0x80]
    while len(m) % 64 != 56:
        m.append(0)
    m += list((8 * n).to_bytes(8, "big"))
    st = list(iv)
    for blk in range(0, len(m), 64):
        words = [_pack(g, m[blk + 4 * t:blk + 4 * t + 4], True) for t in range(16)]
        st = sha256_compress(g, st, words)
    return st


def shake256(g: Graph, msg: Sequence[Item], n: int) -> list[Val]:
    """The first ``n`` bytes of SHAKE256 of the message, as ``ceil(n / 4)`` little-endian 32-bit half-lanes."""
    rate = 8 * RATE_LANES
    m = list(msg) + [0x1F]
    while len(m) % rate:
        m.append(0)
    m[-1] = (m[-1] if isinstance(m[-1], int) else 0) | 0x80
    o = _Ops(g)
    st: list[Val] = [0] * 50
    for blk in range(0, len(m), rate):
        halves = [_pack(g, m[blk + 4 * t:blk + 4 * t + 4], False) for t in range(2 * RATE_LANES)]
        st = [o.xor(st[t], halves[t]) if t < 2 * RATE_LANES else st[t] for t in range(50)]
        st = keccak(g, st)
    out: list[Val] = []
    need = -(-n // 4)
    while True:
        out += st[:2 * RATE_LANES]
        if len(out) >= need:
            return out[:need]
        st = keccak(g, st)


__all__ = ["Byte", "Byte8", "Emitter", "Graph", "MERKLE_FRAME", "Node", "Public", "SHA256_IV", "Sha256Compress", "Split",
           "be_words", "cone", "const_bytes", "evaluate", "evaluate_cone", "frame", "keccak", "merkle_message", "public_part", "sha256",
           "sha256_compress", "shake256", "split", "u32_bytes", "uint_bytes"]
