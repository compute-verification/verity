"""NCP-FP8 (D-3s, #295's `fp8-is-h100-d3s-v0[-f16]`) as a Program, cut into its proof units under layout (a).

A weight's h and amplitudes read only its codes, so they are formed once per run and weight: `NcpFp8Weight{K,N,SUMS}`'s root
body holds its `NcpFp8Amp` units.  One linear call ``A·Bᵀ`` (A: m × k activation rows as the model's E4M3 codes X_q) is
`NcpFp8Linear{M,K,N,SUMS,IL,OL,WL,BIAS}`, which reads them as committed inputs.  Each root body holds only instances of
the unit templates below, so each partition is `Q_template_instances` over them (`circuit.partition`), and every value is
computed in exactly one of them.  `NcpFp8Window{…,C}` is C calls on one weight in one root, the weight's units once.

    template                    one instance per            reads                                      returns
    NcpFp8Amp{K,SUMS}           16-column weight strip      its 16 weight rows                         h and the amplitudes
      (NcpFp8Weight: once per run and weight)
    NcpFp8Domain{M,K,IL,OL}     call                        its prefix, index                          root_A's domain id, the tree's
                                                                                                       frontiers
    NcpFp8Digest{K,RL,L0..L3}   16-row strip of A           its 16 rows, row ids, frontiers            its 16 leaves' subtree root
    NcpFp8Pad{RL,L0..L3}        16 pad leaves               pad ranks, frontiers                       their subtree root
    NcpFp8Node{L}               tree node above the strips  its two children, its (level, index)       the parent; the top is root_A
    NcpFp8Key{K,SUMS,IL,WL,M,N} call                        its prefix, index, root_A                  E₁'s and F₁'s frontiers
    NcpFp8FormY{K,SUMS,..,LL}   16-column weight strip      its h and amplitudes, column ids, F₁'s     the Ỹ strip (16 × 3k codes)
                                                            frontier
    NcpFp8FormX{K,SUMS,..,LL}   16-row strip of A           16 rows, row ids, E₁'s frontier            the X̃ strip (16 × 3k codes)
    NcpFp8Tile{K}               16 × 16 block               its X̃ strip, its Ỹ strip                   the checked words, and Z
    NcpFp8Dequant{BIAS}         output entry                Z, the row scale, the column scale, bias   y (BF16), in vLLM's
                                                                                                       epilogue order

**Semantics** are #295's (`schemes.fp8_is`), bit for bit (`fp8_reference`):
- root_A is the protocol's commitment to A's rows (`audit.commit_rows`: `verity.commitments.merkle`'s SHA-256 frame-v3 tree,
  one leaf per row, its domain bound to the salt and the index, padded to a power of two);
- E₁'s row i and F₁'s column j are `D3Family.stream`'s SHAKE256 streams over (salt, index, root_A, weight id, line);
- X̃'s rows and Ỹ's columns are `D3Family.side`'s D-3s forming (the rms statistic, partner-max amplitudes, the three NCP
  blocks and their E4M3 casts, in f32 sums or f16 sums); a Ỹ column's part over its codes alone (h, and the amplitudes)
  is its weight's `NcpFp8Amp`, the part over F₁ its call's `NcpFp8FormY`;
- a cell's checked words are `cell_words`': T = 3k/32 fresh E4M3 k32 atoms I_τ from +0 (`HopperE4m3QgmmaDot32`) and the
  running FP32 sums R_2 … R_T; the tile returns I_1 … I_T and R_2 … R_(T−1) as its words, and R_T, the useful output, as Z.

**Hoisting** (`hoist`): the streams' and the tree's messages share call-wide prefixes, so each per-row function is split,
its call-wide gates computed once in `NcpFp8Domain` (the tree's) and `NcpFp8Key` (the streams'), each returning its frontier, and
its row gates in the row's unit.  Their gates over the salt and the weight id alone would repeat in every call of a run, or
of a weight, so they are the verifier's: evaluated natively (`prefix`) and read as the call's anchored **prefix**.
**Index binding**: the index, the row ranks, the rows' stream lines, the column lines, the pad ranks and the upper nodes'
(level, index) are Program inputs, which the verifier builds (`anchors.Fp8Anchors`).  A row's rank
(the tree's) and its line (E₁'s) are equal but separate inputs, so that no two units place the same input's bytes alike.  Their encodings are minimal big-endian
(`verity.randomness.frame`, `verity.commitments.merkle._uint`), so byte lengths are template parameters: the index's (IL),
index + 2's, the domain's owner (OL), and the weight id's (WL) per call; a strip's ranks and its nodes' indices
(`strip_lengths`), an upper node's index, and a row's or column's line per binding.  Strips and forming strips are batched
in runs of one byte-length signature, and the per-call units return one frontier per variant (`domain_layout`,
`key_layout`).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Sequence

from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import merkle_depth
from verity.ir.defs import REGISTRY, CompositeDefinition, SpecializedDefinition, bind
from verity.ir.refs import Coll, array_of, concat, tuple_of
from verity.ir.types import Array, Tuple, Value

import verity.ml.fp32  # noqa: F401  (registers F32Add_v1, F32Mul_v1, F32Div_v1, F32Mul_v2, Bf16Add_v1)
import verity.ml.prims  # noqa: F401  (registers F32ToE4m3Sat_v1, F32ToBf16Rn_v1, HopperE4m3QgmmaDot32_v1)

from ..protocol import DomainError
from ..schemes import fp8_is as S8
from . import fp8_words as FW
from . import hoist as H
from . import words as W

U32, U16, U8, F32, BF16 = W.U32, FW.U16, FW.U8, Value(32), Value(16)
D = 16
SLICE = S8.SLICE
DIGEST = Array(8, U32)                                     # a SHA-256 digest as big-endian words
PAIR = Array(2, U32)                                       # a 64-bit value as (hi, lo)
ROW_SCHEMA = b"verity/pouw/row/v1"                         # `audit.ROW_SCHEMA`
ACTIVATION_TAG = b"verity/pouw/activation-rows/v1"         # `audit.activation_binding`'s tag
LAW = "d3s"
SUMS = ("f32", "f16")
MAX_LINES = 1 << 24                                        # row, column and rank ids are u32 Program inputs


def _prim(name: str):
    return REGISTRY[name]


def tag(sums: str) -> str:
    """The forming layer's tag (`D3Family.tag` on X_q): ``d3s-v0`` or ``d3s-v0-f16``."""
    if sums not in SUMS:
        raise ValueError(f"SUMS is one of {SUMS}")
    return "d3s-v0" + ("-f16" if sums == "f16" else "")


@lru_cache(maxsize=16)
def perm(k: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(π, π⁻¹) at k: `fp8_is.ncp_perm` under D-3s's seed, refused unless admissible at d = 32."""
    return S8.ncp_perm(k, f"verity/pouw/fp8-is/{LAW}/P".encode())


# ---- hoisted functions --------------------------------------------------------------------------------------------------

@dataclass
class Hoisted:
    """A per-row function split by `hoist.split`, with its local part as a Definition."""

    split: H.Split
    shared: dict[Any, H.Node]                     # shared input name -> its node
    local: list[tuple[str, Any, list[Any]]]       # (parameter, type, its input names, leaf order)
    out_type: Any
    definition: SpecializedDefinition | None = None

    @property
    def width(self) -> int:
        return len(self.split.frontier)

    def frontier(self, B, env: dict[Any, Coll]) -> Coll:
        """Emit the shared gates into a call-wide unit's body (its shared inputs by name) and return the frontier."""
        e = H.Emitter(B, {n.i: env[name] for name, n in self.shared.items() if name in env})
        e.emit(self.split.hoisted)
        return array_of([e.val(n, 32) for n in self.split.frontier])


def _width(n: H.Node) -> int:
    if n.kind == "op":
        return n.fn.signature()[1].w
    if n.kind == "out":
        return 32
    raise AssertionError(n)


def _split(name: str, build: Callable[[H.Graph], tuple[dict, list, list, Any]]) -> Hoisted:
    """``build(g)`` returns (shared inputs by name, local parameters [(name, type, input nodes)], outputs, output type)."""
    g = H.Graph()
    shared, local, outs, out_type = build(g)
    s = H.split(g, outs)
    if s.direct:
        raise AssertionError(f"{name}: a local gate reads a shared input directly ({s.direct[:3]})")
    if any(_width(n) != 32 for n in s.frontier):
        raise AssertionError(f"{name}: a frontier value is not a 32-bit word")
    return Hoisted(s, shared, [(p, t, [n.name for n in ns]) for p, t, ns in local], out_type)


class Family:
    """A hoisted per-row function as one registered Definition with statics: its local part at each binding, the DAG built
    once per binding from ``make(*statics) -> build``."""

    def __init__(self, name: str, statics: tuple[str, ...], make: Callable[..., Callable], doc: str) -> None:
        self.name, self.statics, self._make = name, statics, make
        self._cache: dict[tuple, Hoisted] = {}
        self.defn = CompositeDefinition(name, 1, statics, self._sig, self._body, doc=doc, conformance="protocol-defined")

    def _split(self, key: tuple) -> Hoisted:
        hf = self._cache.get(key)
        if hf is None:
            hf = self._cache[key] = _split(self.name, self._make(*key))
        return hf

    def at(self, *key) -> Hoisted:
        hf = self._split(key)
        if hf.definition is None:
            hf.definition = bind(self.defn, **dict(zip(self.statics, key)))
        return hf

    def _key(self, S) -> tuple:
        return tuple(getattr(S, st) for st in self.statics)

    def _sig(self, S):
        hf = self._split(self._key(S))
        return (("frontier", Array(hf.width, U32)), *((p, t) for p, t, _ in hf.local)), hf.out_type

    def _body(self, B, S, frontier, *params):
        hf = self._split(self._key(S))
        s = hf.split
        ins = {n.name: n for n in s.graph.nodes if n.kind == "in" and not n.shared}
        env = {n.i: frontier[j] for j, n in enumerate(s.frontier)}
        for (p, t, names), coll in zip(hf.local, params):
            env.update({ins[nm].i: c for nm, c in zip(names, _leaves(coll, t)) if nm in ins})
        e = H.Emitter(B, env)
        e.emit(s.local)
        return _pack_out(e, s.outputs, hf.out_type)


def _leaves(coll: Coll, t) -> list[Coll]:
    if isinstance(t, Value):
        return [coll]
    return [x for j in range(t.n) for x in _leaves(coll[j], t.elem)]


def _pack_out(e: H.Emitter, outs, t):
    it = iter(outs)

    def build(t):
        if isinstance(t, Value):
            return e.val(next(it), t.w)
        return array_of([build(t.elem) for _ in range(t.n)])

    return build(t)


def _inputs(g: H.Graph, name: str, n: int, shared: bool) -> list[H.Node]:
    return [g.input((name, j), shared) for j in range(n)]


# ---- the tree (root_A) ---------------------------------------------------------------------------------------------------

def _positions(m: int) -> bytes:
    return bytes.fromhex(RangeIndexedDomain(m).identity_digest)


def _dom_message(g, salt, index, m: int, ol: int, il: int):
    binding = H.sha256(g, H.const_bytes(ACTIVATION_TAG + b"\0") + H.frame(b"b", [H.Byte8(x) for x in salt])
                       + H.frame(b"i", H.uint_bytes(index, il)))
    owner = (g.op(W.PouwAdd64Hi, index[0], index[1], 0, 2), g.op(W.PouwAdd64Lo, index[1], 2))
    return H.sha256(g, H.merkle_message(b"domain", H.be_words(binding), H.uint_bytes(owner, ol), H.const_bytes(_positions(m)),
                                        H.const_bytes(m.to_bytes(max(1, (m.bit_length() + 7) // 8), "big"))))


def _ulen(v: int) -> int:
    """The byte length of `verity.randomness.frame`'s and `merkle._uint`'s minimal big-endian encoding of ``v``."""
    return max(1, (v.bit_length() + 7) // 8)


def _leaf(k: int, rl: int):
    def build(g):
        dom = _inputs(g, "dom", 8, True)
        rid = g.input("rid", False)
        row = _inputs(g, "row", k, False)
        rank = H.u32_bytes(rid, rl)
        out = H.sha256(g, H.merkle_message(b"leaf", H.be_words(dom), rank, rank, H.const_bytes(ROW_SCHEMA),
                                           [H.Byte8(x) for x in row]))
        return ({("dom", j): dom[j] for j in range(8)}, [("rid", U32, [rid]), ("row", Array(k, U8), row)], out, DIGEST)

    return build


LEAF = Family("NcpFp8Leaf", ("K", "RL"), _leaf, "One committed row's Merkle leaf (`CommitmentDomain.leaf`, the row's codes as "
                                                 "its value, its rank RL bytes), the domain's gates hoisted.")


def leaf_fn(k: int, rl: int = 1) -> Hoisted:
    return LEAF.at(k, rl)


def _pad(rl: int):
    def build(g):
        dom = _inputs(g, "dom", 8, True)
        rank = g.input("rank", False)
        out = H.sha256(g, H.merkle_message(b"pad", H.be_words(dom), H.u32_bytes(rank, rl)))
        return {("dom", j): dom[j] for j in range(8)}, [("rank", U32, [rank])], out, DIGEST

    return build


PAD = Family("NcpFp8PadLeaf", ("RL",), _pad, "One pad leaf of the power-of-two tree (`tree_levels`, its rank RL bytes), the "
                                               "domain's gates hoisted.")


def pad_fn(rl: int = 1) -> Hoisted:
    return PAD.at(rl)


def _node(depth: int | None, length: int):
    """A tree node: in a strip (``depth`` 0 … 3, its index the first rank beneath it shifted by depth + 1) or above the
    strips (``depth`` None: level and index are inputs); its index ``length`` bytes."""

    def build(g):
        dom = _inputs(g, "dom", 8, True)
        left, right = _inputs(g, "left", 8, False), _inputs(g, "right", 8, False)
        if depth is None:
            lvl, idx = g.input("level", False), g.input("index", False)
            lb, ib, local = H.u32_bytes(lvl, 1), H.u32_bytes(idx, length), [("level", U32, [lvl]), ("index", U32, [idx])]
        else:
            src = g.input("first", False)
            ib = [H.Byte(src, depth + 1 + 8 * (length - 1 - t)) for t in range(length)]
            lb, local = [depth], [("first", U32, [src])]
        out = H.sha256(g, H.merkle_message(b"node", H.be_words(dom), lb, ib, H.be_words(left), H.be_words(right)))
        return ({("dom", j): dom[j] for j in range(8)}, [*local, ("left", DIGEST, left), ("right", DIGEST, right)], out,
                DIGEST)

    return build


STRIP_NODE = Family("NcpFp8StripNode", ("DEPTH", "L"), _node,
                    "One Merkle node inside a 16-leaf strip (`CommitmentDomain.node` at depth DEPTH, its index the first rank "
                    "beneath it >> (DEPTH + 1), L bytes), the domain's gates hoisted.")
UPPER_NODE = Family("NcpFp8UpperNode", ("L",), lambda length: _node(None, length),
                    "One Merkle node above the strips (`CommitmentDomain.node`, its level and its L-byte index inputs), the "
                    "domain's gates hoisted.")


def node_fn(depth: int | None, length: int = 1) -> Hoisted:
    return UPPER_NODE.at(length) if depth is None else STRIP_NODE.at(depth, length)


def strip_lengths(first: int) -> tuple[int, int, int, int, int]:
    """A 16-rank strip's byte lengths from its first rank: its ranks', then its depth-0 … 3 nodes' indices (uniform in the
    strip, since 256 is a multiple of 16)."""
    return (_ulen(first), *(_ulen(first >> (d + 1)) for d in range(4)))


def _subtree(B, leaves: list[Coll], firsts: list[Coll], fronts: list[Coll], lengths: Sequence[int]) -> Coll:
    """The 16 leaves' subtree root: depths 0 … 3 of `tree_levels`, each node's index its first rank >> (depth + 1)."""
    level, first = leaves, firsts
    for d in range(4):
        fn = node_fn(d, lengths[d]).definition
        level = [B.call(fn, fronts[d], first[2 * j], level[2 * j], level[2 * j + 1]) for j in range(len(level) // 2)]
        first = first[::2]
    return level[0]


def domain_layout(m: int) -> list[tuple]:
    """The frontiers `NcpFp8Domain` returns after the domain id, in order: ("leaf", RL), ("strip", DEPTH, L), ("upper", L),
    ("pad", RL), one per byte-length variant the tree of m rows has."""
    width = 1 << merkle_depth(m)
    real = [strip_lengths(D * s) for s in range(m // D)]
    pads = [strip_lengths(D * s) for s in range(m // D, width // D)]
    ups = sorted({_ulen(j) for _lv, j in _upper(width)})
    return ([("leaf", rl) for rl in sorted({x[0] for x in real})]
            + [("strip", d, ln) for d, ln in sorted({(d, x[1 + d]) for x in real + pads for d in range(4)})]
            + [("upper", ln) for ln in ups] + [("pad", rl) for rl in sorted({x[0] for x in pads})])


def _variant(key: tuple, k: int) -> Hoisted:
    kind = key[0]
    if kind == "leaf":
        return leaf_fn(k, key[1])
    if kind == "strip":
        return node_fn(key[1], key[2])
    if kind == "upper":
        return node_fn(None, key[1])
    return pad_fn(key[1])


def _is_public(name) -> bool:
    """The inputs every call of a run (the salt), or of a weight (its id), shares: gates over these alone are the verifier's."""
    return name[0] in ("salt", "weight")


@lru_cache(maxsize=64)
def _dom_graph(m: int, ol: int, il: int) -> tuple[H.Graph, list[H.Node], list[H.Val], H.Public]:
    """The domain id's DAG over (salt, index), and its public part: `activation_binding`'s gates over the salt alone."""
    g = H.Graph()
    sv, iv = _inputs(g, "salt", 32, True), _inputs(g, "index", 2, True)
    dom = _dom_message(g, sv, iv, m, ol, il)
    return g, iv, dom, H.public_part(g, H.cone(g, dom), _is_public, dom)


def _domain_sig(S):
    ret = [DIGEST, *(Array(_variant(key, S.K).width, U32) for key in domain_layout(S.M))]
    return (("prefix", Array(len(_dom_graph(S.M, S.OL, S.IL)[3].frontier), U32)), ("index", PAIR)), Tuple(*ret)


def _domain_body(B, S, pre, index):
    _g, iv, dom, pub = _dom_graph(S.M, S.OL, S.IL)
    env = {n.i: pre[j] for j, n in enumerate(pub.frontier)} | {n.i: index[j] for j, n in enumerate(iv)}
    e = H.Emitter(B, env)
    e.emit(pub.rest)
    dom_c = [e.val(x, 32) for x in dom]
    denv = {("dom", j): dom_c[j] for j in range(8)}
    return tuple_of(array_of(dom_c), *(_variant(key, S.K).frontier(B, denv) for key in domain_layout(S.M)))


NcpFp8DomainDef = CompositeDefinition(
    "NcpFp8Domain", 1, ("M", "K", "IL", "OL"), _domain_sig, _domain_body,
    doc="root_A's commitment domain (`audit.activation_binding`, `CommitmentDomain.domain_id` with owner = index), and the "
        "tree's hoisted frontiers, one per byte-length variant (`domain_layout`); the binding's gates over the salt alone "
        "are the verifier's, read as the anchored prefix.", conformance="protocol-defined")


def _digest_sig(S):
    return ((("leaf", Array(leaf_fn(S.K, S.RL).width, U32)),
             *((f"node{d}", Array(node_fn(d, ln).width, U32)) for d, ln in enumerate((S.L0, S.L1, S.L2, S.L3))),
             ("rows", Array(D, Array(S.K, U8))), ("rowids", Array(D, U32))), DIGEST)


def _digest_body(B, S, leaf, n0, n1, n2, n3, rows, rid):
    lf = leaf_fn(S.K, S.RL).definition
    leaves = [B.call(lf, leaf, rid[r], rows[r]) for r in range(D)]
    return _subtree(B, leaves, [rid[r] for r in range(D)], [n0, n1, n2, n3], (S.L0, S.L1, S.L2, S.L3))


NcpFp8DigestDef = CompositeDefinition(
    "NcpFp8Digest", 1, ("K", "RL", "L0", "L1", "L2", "L3"), _digest_sig, _digest_body,
    doc="A 16-row strip's leaves and their subtree root (depths 0 … 3 of root_A's tree; RL, L0 … L3 the byte lengths of its "
        "ranks and its nodes' indices).", conformance="protocol-defined")


def _pad_sig(S):
    return ((("pad", Array(pad_fn(S.RL).width, U32)),
             *((f"node{d}", Array(node_fn(d, ln).width, U32)) for d, ln in enumerate((S.L0, S.L1, S.L2, S.L3))),
             ("ranks", Array(D, U32))), DIGEST)


def _pad_body(B, S, pad, n0, n1, n2, n3, ranks):
    pf = pad_fn(S.RL).definition
    leaves = [B.call(pf, pad, ranks[r]) for r in range(D)]
    return _subtree(B, leaves, [ranks[r] for r in range(D)], [n0, n1, n2, n3], (S.L0, S.L1, S.L2, S.L3))


NcpFp8PadDef = CompositeDefinition(
    "NcpFp8Pad", 1, ("RL", "L0", "L1", "L2", "L3"), _pad_sig, _pad_body,
    doc="16 pad leaves of root_A's tree and their subtree root.", conformance="protocol-defined")


def _node_sig(S):
    return ((("upper", Array(node_fn(None, S.L).width, U32)), ("id", Array(2, U32)), ("left", DIGEST), ("right", DIGEST)),
            DIGEST)


def _node_body(B, S, upper, nid, left, right):
    return B.call(node_fn(None, S.L).definition, upper, nid[0], nid[1], left, right)


NcpFp8NodeDef = CompositeDefinition(
    "NcpFp8Node", 1, ("L",), _node_sig, _node_body,
    doc="One node of root_A's tree above the strips (its index L bytes); the top is root_A.", conformance="protocol-defined")


# ---- the streams and the forming ------------------------------------------------------------------------------------------

F16_TWO_INV4, F16_MINUS_3_8, F16_2_POW_M8, F16_2_POW_M6 = S8.F16_TWO_INV4, S8.F16_MINUS_3_8, S8.F16_2_POW_M8, S8.F16_2_POW_M6


def _side(g: H.Graph, codes: Sequence[H.Val], stream: Sequence[H.Val], left: bool, sums: str) -> list[H.Val]:
    """`D3Family.side` for law d3s on X_q (or a weight column's codes): one stacked row of X̃ or column of Ỹ, 3k codes."""
    h, amp = _amplitude(g, codes, left)
    return _blocks(g, h, amp, stream, left, sums)


def _amplitude(g: H.Graph, codes: Sequence[H.Val], left: bool) -> tuple[list[H.Val], list[H.Val]]:
    """`D3Family.side`'s part over the codes alone: h (f16) and the amplitudes max(|h_l|, |h_partner(l)|, floor)."""
    k = len(codes)
    pi, inv = perm(k)
    f32add, f32mul, f32div = _prim("F32Add_v1"), _prim("F32Mul_v1"), _prim("F32Div_v1")
    h = [g.op(FW.PouwF16FromE4m3, c) for c in codes]
    z = [g.op(FW.PouwF16Mul, w, F16_2_POW_M8) for w in h]
    lanes: list[H.Val] = []
    fam = S8.D3Family
    for t in range(fam.LANES):
        entries, acc = z[t::fam.LANES], 0
        for c in range(0, len(entries), fam.CHAIN):
            q: H.Val = 0
            for w in entries[c:c + fam.CHAIN]:
                q = g.op(FW.PouwF16Fma, w, w, q)
            acc = g.op(f32add, acc, g.op(FW.PouwF16ToF32, q))
        lanes.append(acc)
    half = fam.LANES // 2
    while half:
        lanes = [g.op(f32add, lanes[t], lanes[t + half]) for t in range(half)]
        half //= 2
    rms = g.op(f32mul, g.op(FW.PouwF32Sqrt, g.op(f32div, lanes[0], S8.f32_word(k))), S8.f32_word(256))
    floor = g.op(FW.PouwF16Max, g.op(FW.PouwF32ToF16Rn, rms), F16_2_POW_M6)
    mag = [g.op(FW.PouwF16Abs, w) for w in h]
    partner = inv if left else pi
    return h, [g.op(FW.PouwF16Max, g.op(FW.PouwF16Max, mag[l], mag[partner[l]]), floor) for l in range(k)]


def _blocks(g: H.Graph, h: Sequence[H.Val], amp: Sequence[H.Val], stream: Sequence[H.Val], left: bool, sums: str,
            wide_h: bool = False) -> list[H.Val]:
    """`D3Family.side`'s part over the stream: the noise, the three blocks and their E4M3 casts.  ``wide_h``: ``h`` is
    already FP32 words (a Ỹ column's under f32 sums, from its weight's `NcpFp8Amp`)."""
    k = len(amp)
    pi, inv = perm(k)
    f32add = _prim("F32Add_v1")
    e = [g.op(FW.PouwF16Mul, amp[l], g.op(FW.PouwF16Fma, F16_TWO_INV4, g.op(FW.PouwF16Field, stream[l // 2], l % 2),
                                          F16_MINUS_3_8)) for l in range(k)]
    if left:
        e2 = [e[pi[l]] for l in range(k)]
        s = [g.op(FW.PouwF16Add, a, b) for a, b in zip(e, e2)]
        blocks = ((h, e), (h, e2), ([g.op(FW.PouwF16Neg, w) for w in h], [g.op(FW.PouwF16Neg, w) for w in s]))
    else:
        f2 = [g.op(FW.PouwF16Neg, e[inv[l]]) for l in range(k)]
        s = [g.op(FW.PouwF16Add, a, b) for a, b in zip(e, f2)]
        blocks = ((h, e), (h, f2), (h, s))
    if sums == "f16":
        return [g.op(FW.PouwF16ToE4m3, g.op(FW.PouwF16Add, x, n)) for base, nz in blocks for x, n in zip(base, nz)]
    wide: dict[int, H.Val] = {}

    def f32(v):
        key = v if isinstance(v, int) else id(v)
        if key not in wide:
            wide[key] = g.op(FW.PouwF16ToF32, v)
        return wide[key]

    cast, base32 = _prim("F32ToE4m3Sat_v1"), (lambda v: v) if wide_h else f32
    return [g.op(cast, g.op(f32add, base32(x), f32(n))) for base, nz in blocks for x, n in zip(base, nz)]


def _stream(g: H.Graph, name: str, sums: str, k: int, salt, index, root, weight, line, il: int, wl: int,
            ll: int = 1) -> list[H.Val]:
    """`D3Family.stream`: SHAKE256("verity/pouw/fp8-is/<tag>/<name>" ‖ 0 ‖ frame(salt, index, root_A, weight, line)), 2k
    bytes, as half-lanes."""
    msg = (H.const_bytes(f"verity/pouw/fp8-is/{tag(sums)}/{name}".encode() + b"\0")
           + H.frame(b"b", [H.Byte8(x) for x in salt]) + H.frame(b"i", H.uint_bytes(index, il))
           + H.frame(b"b", H.be_words(root)) + H.frame(b"i", H.uint_bytes(weight, wl)) + H.frame(b"i", H.u32_bytes(line, ll)))
    return H.shake256(g, msg, 2 * k)


def _row(left: bool):
    def make(k: int, sums: str, il: int, wl: int, ll: int):
        return _row_build(k, sums, il, wl, ll, left)

    return make


def hw_type(sums: str) -> Value:
    """A weight value as a Ỹ column reads it from its `NcpFp8Amp`: h's f16 word, or its FP32 word under f32 sums."""
    return U16 if sums == "f16" else U32


def _row_build(k: int, sums: str, il: int, wl: int, ll: int, left: bool):
    """One X̃ row (``left``: E₁'s stream, from its codes) or Ỹ column (F₁'s, from its weight's h and amplitudes), and its
    ``ll``-byte line id."""

    def build(g):
        salt, index, root, weight = (_inputs(g, "salt", 32, True), _inputs(g, "index", 2, True), _inputs(g, "root", 8, True),
                                     _inputs(g, "weight", 2, True))
        line = g.input("line", False)
        stream = lambda: _stream(g, "E1" if left else "F1", sums, k, salt, index, root, weight, line, il, wl, ll)  # noqa: E731
        shared = {n.name: n for n in salt + index + root + weight}
        if left:
            codes = _inputs(g, "codes", k, False)
            out = _side(g, codes, stream(), True, sums)
            return shared, [("line", U32, [line]), ("codes", Array(k, U8), codes)], out, Array(3 * k, U8)
        h, amp = _inputs(g, "h", k, False), _inputs(g, "amp", k, False)
        out = _blocks(g, h, amp, stream(), False, sums, wide_h=sums == "f32")
        return (shared, [("line", U32, [line]), ("h", Array(k, hw_type(sums)), h), ("amp", Array(k, U16), amp)], out,
                Array(3 * k, U8))

    return build


X_ROW = Family("NcpFp8XRow", ("K", "SUMS", "IL", "WL", "LL"), _row(True),
               "One X̃ row: its E₁ stream (its line LL bytes) and D-3s forming (`D3Family.side`), the stream's call-wide gates "
               "hoisted.")
Y_ROW = Family("NcpFp8YRow", ("K", "SUMS", "IL", "WL", "LL"), _row(False),
               "One Ỹ column from its weight's h and amplitudes (`NcpFp8Amp`): its F₁ stream (its line LL bytes) and the D-3s "
               "blocks, the stream's call-wide gates hoisted.")


def row_fn(k: int, sums: str, il: int, wl: int, left: bool, ll: int = 1) -> Hoisted:
    return (X_ROW if left else Y_ROW).at(k, sums, il, wl, ll)


def key_layout(m: int, n: int) -> list[tuple[str, int]]:
    """The frontiers `NcpFp8Key` returns, in order: ("x", LL) per byte length of A's row lines, then ("y", LL) per byte
    length of the weight's column lines."""
    return ([("x", ll) for ll in sorted({_ulen(D * s) for s in range(m // D)})]
            + [("y", ll) for ll in sorted({_ulen(D * c) for c in range(n // D)})])


@lru_cache(maxsize=64)
def _key_public(k: int, sums: str, il: int, wl: int, left: bool, ll: int) -> H.Public:
    """A stream's call-wide gates, cut at their public part: the gates over the salt and the weight id alone."""
    hf = row_fn(k, sums, il, wl, left, ll)
    return H.public_part(hf.split.graph, hf.split.hoisted, _is_public, hf.split.frontier)


def _key_parts(S) -> list[tuple[Hoisted, H.Public]]:
    return [(row_fn(S.K, S.SUMS, S.IL, S.WL, side == "x", ll), _key_public(S.K, S.SUMS, S.IL, S.WL, side == "x", ll))
            for side, ll in key_layout(S.M, S.N)]


def _key_sig(S):
    parts = _key_parts(S)
    return ((("prefix", Array(sum(len(p.frontier) for _hf, p in parts), U32)), ("index", PAIR), ("root", DIGEST)),
            Tuple(*(Array(hf.width, U32) for hf, _p in parts)))


def _key_body(B, S, pre, index, root):
    named = {("index", j): index[j] for j in range(2)} | {("root", j): root[j] for j in range(8)}
    outs, at = [], 0
    for hf, pub in _key_parts(S):
        env = {n.i: pre[at + j] for j, n in enumerate(pub.frontier)}
        env |= {n.i: named[name] for name, n in hf.shared.items() if name in named}
        at += len(pub.frontier)
        e = H.Emitter(B, env)
        e.emit(pub.rest)
        outs.append(array_of([e.val(n, 32) for n in hf.split.frontier]))
    return tuple_of(*outs)


NcpFp8KeyDef = CompositeDefinition(
    "NcpFp8Key", 1, ("K", "SUMS", "IL", "WL", "M", "N"), _key_sig, _key_body,
    doc="E₁'s and F₁'s call-wide gates, once per call: every stream's shared prefix over (salt, index, root_A, weight id), "
        "one frontier per line length (`key_layout`); its gates over the salt and the weight id alone are the verifier's, "
        "read as the anchored prefix.", conformance="protocol-defined")


def _form_x_body(B, S, frontier, rows, ids):
    fn = row_fn(S.K, S.SUMS, S.IL, S.WL, True, S.LL).definition
    return array_of([B.call(fn, frontier, ids[r], rows[r]) for r in range(D)])


def _form_y_body(B, S, frontier, hs, amps, ids):
    fn = row_fn(S.K, S.SUMS, S.IL, S.WL, False, S.LL).definition
    return array_of([B.call(fn, frontier, ids[r], hs[r], amps[r]) for r in range(D)])


NcpFp8FormXDef = CompositeDefinition(
    "NcpFp8FormX", 1, ("K", "SUMS", "IL", "WL", "LL"),
    lambda S: ((("frontier", Array(row_fn(S.K, S.SUMS, S.IL, S.WL, True, S.LL).width, U32)),
                ("rows", Array(D, Array(S.K, U8))), ("ids", Array(D, U32))), Array(D, Array(3 * S.K, U8))), _form_x_body,
    doc="An activation strip's X̃ rows: [Q(h + e) | Q(h + e₂) | Q(−h − s)] (D-3s, `D3Family.side`).", conformance="protocol-defined")
NcpFp8FormYDef = CompositeDefinition(
    "NcpFp8FormY", 1, ("K", "SUMS", "IL", "WL", "LL"),
    lambda S: ((("frontier", Array(row_fn(S.K, S.SUMS, S.IL, S.WL, False, S.LL).width, U32)),
                ("hs", Array(D, Array(S.K, hw_type(S.SUMS)))), ("amps", Array(D, Array(S.K, U16))), ("ids", Array(D, U32))),
               Array(D, Array(3 * S.K, U8))), _form_y_body,
    doc="A weight strip's Ỹ columns: [Q(h + f) ; Q(h + f₂) ; Q(h + s)] (D-3s, `D3Family.side`), from the strip's h and "
        "amplitudes (`NcpFp8Amp`, the weight's).", conformance="protocol-defined")


# ---- the weight side: h and the amplitudes, once per run and weight strip ---------------------------------------------------

@lru_cache(maxsize=16)
def _amp_graph(k: int, sums: str) -> tuple[H.Graph, list[H.Node], list[H.Val], list[H.Val]]:
    g = H.Graph()
    codes = _inputs(g, "codes", k, False)
    h, amp = _amplitude(g, codes, False)
    return g, codes, (h if sums == "f16" else [g.op(FW.PouwF16ToF32, w) for w in h]), amp


def _amp_row_body(B, S, codes):
    g, ins, hw, amp = _amp_graph(S.K, S.SUMS)
    e = H.Emitter(B, {n.i: codes[j] for j, n in enumerate(ins)})
    e.emit(H.cone(g, [*hw, *amp]))
    return tuple_of(array_of([e.val(v, hw_type(S.SUMS).w) for v in hw]), array_of([e.val(v, 16) for v in amp]))


NcpFp8AmpRowDef = CompositeDefinition(
    "NcpFp8AmpRow", 1, ("K", "SUMS"),
    lambda S: ((("codes", Array(S.K, U8)),), Tuple(Array(S.K, hw_type(S.SUMS)), Array(S.K, U16))), _amp_row_body,
    doc="One weight column's part of `D3Family.side` over its codes alone: h (f16, or FP32 under f32 sums) and the "
        "amplitudes max(|h_l|, |h_π(l)|, floor(rms)).", conformance="protocol-defined")


def _amp_body(B, S, rows):
    fn = bind(NcpFp8AmpRowDef, K=S.K, SUMS=S.SUMS)
    cols = [B.call(fn, rows[r]) for r in range(D)]
    return tuple_of(array_of([c[0] for c in cols]), array_of([c[1] for c in cols]))


NcpFp8AmpDef = CompositeDefinition(
    "NcpFp8Amp", 1, ("K", "SUMS"),
    lambda S: ((("rows", Array(D, Array(S.K, U8))),), Tuple(Array(D, Array(S.K, hw_type(S.SUMS))), Array(D, Array(S.K, U16)))),
    _amp_body, doc="A 16-column weight strip's h and amplitudes, once per run and weight, read by every call's Ỹ strip.",
    conformance="protocol-defined")


def hs_type(k: int, n: int, sums: str) -> Array:
    return Array(n // D, Array(D, Array(k, hw_type(sums))))


def amps_type(k: int, n: int) -> Array:
    return Array(n // D, Array(D, Array(k, U16)))


def _weight_units(B, S, b):
    ns = S.N // D
    amp = B.batch(amp_template(S.K, S.SUMS), b.reshape(Array(ns, Array(D, Array(S.K, U8)))), axes=(0,))
    return array_of([amp[c][0] for c in range(ns)]), array_of([amp[c][1] for c in range(ns)])


NcpFp8WeightDef = CompositeDefinition(
    "NcpFp8Weight", 1, ("K", "N", "SUMS"),
    lambda S: ((("b", Array(S.N, Array(S.K, U8))),), Tuple(hs_type(S.K, S.N, S.SUMS), amps_type(S.K, S.N))),
    lambda B, S, b: tuple_of(*_weight_units(B, S, b)),
    doc="One run's weight side: each strip's h and amplitudes (`NcpFp8Amp`), formed once per run and weight and read by "
        "every call on it.", conformance="protocol-defined")


# ---- the tile and the dequantization ----------------------------------------------------------------------------------------

def _tile_body(B, S, xs, ys):
    steps = 3 * S.K // SLICE
    atom, add, zero = _prim("HopperE4m3QgmmaDot32_v1"), _prim("F32Add_v1"), B.call(verity.ml.prims.const(32, 0))
    words, z = [], []
    for r in range(D):
        zr = []
        for c in range(D):
            atoms = [B.call(atom, zero, xs[r][SLICE * t:SLICE * (t + 1)], ys[c][SLICE * t:SLICE * (t + 1)]) for t in range(steps)]
            run = [atoms[0]]
            for a in atoms[1:]:
                run.append(B.call(add, run[-1], a))
            words += atoms + run[1:-1]
            zr.append(run[-1])
        z.append(array_of(zr))
    return tuple_of(array_of(words), array_of(z))


def tile_words(k: int) -> int:
    """A cell's words in the tile's first return field: I_1 … I_T and R_2 … R_(T−1)."""
    return 2 * (3 * k // SLICE) - 2


def _leaf_tag(sums: str, part: str) -> bytes:
    return f"verity/pouw/{S8.PREFIX}-{tag(sums)}/{part}".encode() + b"\0"


def subtile_digest(sums: str, k: int, words: Sequence[int]) -> bytes:
    """#295's sub-digest of one 16 × 16 subtile (`FP8IS.subdigest`), from the circuit's tile's credited words (its first
    field: cell-major, per cell I_1 … I_T and R_2 … R_(T−1)), with each cell's R_T = RN_f32(R_(T−1) + I_T) by the tile's
    own `F32Add_v1`:

        sub = SHAKE256("verity/pouw/fp8-is-h100-<tag>/subtile" ‖ 0x00 ‖ the layer's words ‖ the core words)[:32]

    The core words are little-endian, step-major, row-major over the cells within a step, I_τ before R_τ, and R_1 = I_1
    unchecked. D-3s's layer has no checked words of its own (`D3Family.form`), so none come before them."""
    steps, per = 3 * k // SLICE, tile_words(k)
    if len(words) != D * D * per:
        raise ValueError(f"a tile at k = {k} credits {D * D * per} words, not {len(words)}")
    add = _prim("F32Add_v1").evaluate
    cells = [list(words[c * per:(c + 1) * per]) for c in range(D * D)]
    out = bytearray()
    for t in range(steps):
        for w in cells:
            out += w[t].to_bytes(4, "little")
            if t == steps - 1:
                out += add(w[-1] if steps > 2 else w[0], w[t]).to_bytes(4, "little")
            elif t:
                out += w[steps + t - 1].to_bytes(4, "little")
    return hashlib.shake_256(_leaf_tag(sums, "subtile") + bytes(out)).digest(32)


def leaf_digest(sums: str, subdigests: Sequence[bytes]) -> bytes:
    """#295's Merkle leaf of a 64 × 64 tile (`FP8IS.digest`): SHAKE256("verity/pouw/fp8-is-h100-<tag>/leaf" ‖ 0x00 ‖ its
    16 subtiles' digests, row-major)[:32]. A proof of one circuit tile opens it with the other 15."""
    return hashlib.shake_256(_leaf_tag(sums, "leaf") + b"".join(subdigests)).digest(32)


NcpFp8TileDef = CompositeDefinition(
    "NcpFp8Tile", 1, ("K",),
    lambda S: ((("xs", Array(D, Array(3 * S.K, U8))), ("ys", Array(D, Array(3 * S.K, U8)))),
               Tuple(Array(D * D * tile_words(S.K), U32), Array(D, Array(D, F32)))), _tile_body,
    doc="A 16 × 16 tile's checked words, cell-major (per cell I_1 … I_T, then R_2 … R_(T−1): `fp8_is.cell_words`), and Z = R_T, "
        "the useful output.", conformance="protocol-defined")


def _dequant_body(B, S, z, s, t, *bias):
    m = B.call(_prim("F32Mul_v2"), s, z)
    if S.BIAS:
        v = B.call(_prim("F32Fma_v2"), t, m, B.call(_prim("Bf16ToF32_v1"), bias[0]))
    else:
        v = B.call(_prim("F32Mul_v2"), t, m)
    return B.call(_prim("F2fpBf16_v1"), v)


NcpFp8DequantDef = CompositeDefinition(
    "NcpFp8Dequant", 1, ("BIAS",),
    lambda S: ((("z", F32), ("s", F32), ("t", F32)) + ((("bias", BF16),) if S.BIAS else ()), BF16), _dequant_body,
    doc="The served output, in the order of vLLM's `cutlass_scaled_mm` epilogue under the batch-invariant dispatch (its "
        "scales swapped, so the row scale is ScaleB): m = RN(s_i·Z), then y = bf16(RN(t_j·m + bias_j)) in one FFMA "
        "(`ScaledEpilogueBias`, BIAS) or bf16(RN(t_j·m)) (`ScaledEpilogue`), with `cvt.rn.bf16.f32`. A width-rule unit "
        "after the Z separator.", conformance="protocol-defined")


def dequant_template(bias: bool = True) -> SpecializedDefinition:
    """`NcpFp8Dequant{BIAS}`: with a bias the epilogue is an FFMA, without it a multiply, and the two differ on a −0."""
    return bind(NcpFp8DequantDef, BIAS=bias)


NcpFp8Dequant = dequant_template(True)


# ---- the linear call -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    m: int
    k: int
    n: int

    def check(self) -> None:
        if self.m % D or self.n % D or self.k % SLICE or min(self.m, self.k, self.n) <= 0:
            raise DomainError(f"the NCP-FP8 circuit needs 16 | m, 16 | n, 32 | k: {self}")
        if self.m > MAX_LINES or self.n > MAX_LINES:
            raise DomainError(f"the NCP-FP8 circuit takes m, n <= {MAX_LINES}, not {self}")
        perm(self.k)

    @property
    def width(self) -> int:
        """root_A's tree width: m padded to a power of two."""
        return 1 << merkle_depth(self.m)


@dataclass(frozen=True)
class Lengths:
    """The byte lengths of the call's minimal integer encodings: the index, index + 2 (the domain's owner), the weight id."""

    il: int
    ol: int
    wl: int

    @staticmethod
    def of(index: int, weight: int) -> "Lengths":
        return Lengths(_ulen(index), _ulen(index + 2), _ulen(weight))


def _upper(width: int) -> list[tuple[int, int]]:
    out, count, level = [], width // D, 4
    while count > 1:
        count //= 2
        out += [(level, j) for j in range(count)]
        level += 1
    return out


def upper_nodes(shape: Shape) -> list[tuple[int, int]]:
    """The tree's nodes above the strips, (level, index), in the order the root body computes them (level ascending)."""
    return _upper(shape.width)


def _runs(keys: Sequence) -> list[tuple[int, int, Any]]:
    """Consecutive equal keys as (lo, hi, key)."""
    out: list[tuple[int, int, Any]] = []
    for j, key in enumerate(keys):
        if out and out[-1][2] == key:
            out[-1] = (out[-1][0], j + 1, key)
        else:
            out.append((j, j + 1, key))
    return out


def prefix_parts(shape: Shape, sums: str, lengths: Lengths) -> list[tuple[H.Graph, list[H.Node]]]:
    """The public values the call's domain and key units read, in the anchored prefix's order: the domain's, then each
    stream variant's (`key_layout`)."""
    m, k, n = shape.m, shape.k, shape.n
    g, _iv, _dom, pub = _dom_graph(m, lengths.ol, lengths.il)
    parts = [(g, pub.frontier)]
    for side, ll in key_layout(m, n):
        p = _key_public(k, sums, lengths.il, lengths.wl, side == "x", ll)
        parts.append((p.graph, p.frontier))
    return parts


def prefix(shape: Shape, sums: str, lengths: Lengths, salt: bytes, weight: int) -> list[int]:
    """The anchored prefix, as the verifier evaluates it natively from the run's salt and the weight id."""
    env = {("salt", j): salt[j] for j in range(32)} | {("weight", 0): weight >> 32, ("weight", 1): weight & W.M32}
    return [v for g, front in prefix_parts(shape, sums, lengths) for v in H.evaluate_cone(g, front, env)]


def _prefix_widths(shape: Shape, sums: str, lengths: Lengths) -> tuple[int, int]:
    """(the domain's, the key's) prefix widths."""
    widths = [len(front) for _g, front in prefix_parts(shape, sums, lengths)]
    return widths[0], sum(widths[1:])


def _call_sig(m: int, k: int, n: int, sums: str, lg: Lengths, bias: bool = True) -> list[tuple[str, Any]]:
    shape = Shape(m, k, n)
    pads, ups = shape.width - m, len(upper_nodes(shape))
    params = [("a", Array(m, Array(k, U8))), ("s", Array(m, F32)), ("hs", hs_type(k, n, sums)), ("amps", amps_type(k, n)),
              ("t", Array(n, F32))]
    params += [("bias", Array(n, BF16))] if bias else []
    params += [("prefix", Array(sum(_prefix_widths(shape, sums, lg)), U32)), ("index", PAIR), ("rowids", Array(m, U32)),
               ("lines", Array(m, U32)), ("colids", Array(n, U32))]
    params += [("padids", Array(pads, U32))] if pads else []
    params += [("nodeids", Array(ups, Array(2, U32)))] if ups else []
    return params


def _call_out(m: int, k: int, n: int) -> Tuple:
    return Tuple(Array(m, Array(n, BF16)), Array((m // D) * (n // D), Array(D * D * tile_words(k), U32)))


def _linear_sig(S):
    return tuple(_call_sig(S.M, S.K, S.N, S.SUMS, Lengths(S.IL, S.OL, S.WL), S.BIAS)), _call_out(S.M, S.K, S.N)


def _call_units(B, S, a, s, hs, amps, t, bias, pre, index, rid, lines, cid, padids, nodeids):
    """One call's units, reading its weight's committed h and amplitudes: (y, the credited words)."""
    m, k, n = S.M, S.K, S.N
    shape = Shape(m, k, n)
    ms, ns, pads = m // D, n // D, shape.width - m
    lg = Lengths(S.IL, S.OL, S.WL)
    pd, _pk = _prefix_widths(shape, S.SUMS, lg)
    dom = B.call(domain_template(m, k, lg), pre[0:pd], index)
    front = {key: dom[1 + j] for j, key in enumerate(domain_layout(m))}

    def strip_fronts(sig, kind):
        return [front[(kind, sig[0])], *(front[("strip", d, sig[1 + d])] for d in range(4))]

    strips_a = a.reshape(Array(ms, Array(D, Array(k, U8))))
    rows_id = rid.reshape(Array(ms, Array(D, U32)))
    level = []
    for lo, hi, sig in _runs([strip_lengths(D * s_) for s_ in range(ms)]):
        dig = B.batch(digest_template(k, sig), *strip_fronts(sig, "leaf"), strips_a[lo:hi], rows_id[lo:hi],
                      axes=(None, None, None, None, None, 0, 0))
        level += [dig[j] for j in range(hi - lo)]
    if pads:
        pad_ids = padids.reshape(Array(pads // D, Array(D, U32)))
        for lo, hi, sig in _runs([strip_lengths(m + D * s_) for s_ in range(pads // D)]):
            pd_ = B.batch(pad_template(sig), *strip_fronts(sig, "pad"), pad_ids[lo:hi], axes=(None, None, None, None, None, 0))
            level += [pd_[j] for j in range(hi - lo)]
    ups, at = upper_nodes(shape), 0
    while len(level) > 1:
        nxt = []
        for j in range(len(level) // 2):
            ln = _ulen(ups[at + j][1])
            nxt.append(B.call(node_template(ln), front[("upper", ln)], nodeids[at + j], level[2 * j], level[2 * j + 1]))
        at += len(nxt)
        level = nxt
    root = level[0]
    key = B.call(key_template(k, S.SUMS, lg, m, n), pre[pd:], index, root)
    kfront = {kk: key[j] for j, kk in enumerate(key_layout(m, n))}
    cols_id = cid.reshape(Array(ns, Array(D, U32)))
    y_strips = []
    for lo, hi, ll in _runs([_ulen(D * c) for c in range(ns)]):
        ys = B.batch(form_template(False, k, S.SUMS, lg, ll), kfront[("y", ll)], hs[lo:hi], amps[lo:hi], cols_id[lo:hi],
                     axes=(None, 0, 0, 0))
        y_strips += [ys[j] for j in range(hi - lo)]
    x_strips = []
    lines_id = lines.reshape(Array(ms, Array(D, U32)))
    for lo, hi, ll in _runs([_ulen(D * s_) for s_ in range(ms)]):
        xs = B.batch(form_template(True, k, S.SUMS, lg, ll), kfront[("x", ll)], strips_a[lo:hi], lines_id[lo:hi], axes=(None, 0, 0))
        x_strips += [xs[j] for j in range(hi - lo)]
    y_all = array_of(y_strips)
    tile_t = tile_template(k)
    tiles = [B.batch(tile_t, x_strips[rs], y_all, axes=(None, 0)) for rs in range(ms)]
    out = {}
    for rs in range(ms):
        for cs in range(ns):
            z = tiles[rs][cs][1]
            zz = Coll(Array(D * D, F32), z.refs)
            s_rep = Coll(Array(D * D, F32), concat([s[D * rs + r].refs for r in range(D) for _ in range(D)]))
            t_rep = Coll(Array(D * D, F32), concat([t[D * cs + c].refs for _ in range(D) for c in range(D)]))
            if S.BIAS:
                b_rep = Coll(Array(D * D, BF16), concat([bias[D * cs + c].refs for _ in range(D) for c in range(D)]))
                out[rs, cs] = B.batch(dequant_template(True), zz, s_rep, t_rep, b_rep, axes=(0, 0, 0, 0))
            else:
                out[rs, cs] = B.batch(dequant_template(False), zz, s_rep, t_rep, axes=(0, 0, 0))
    y = Coll(Array(m, Array(n, BF16)),
             concat([out[i // D, j // D][D * (i % D) + j % D].refs for i in range(m) for j in range(n)]))
    words = array_of([tiles[rs][cs][0] for rs in range(ms) for cs in range(ns)])
    return y, words


def _linear_body(B, S, a, s, hs, amps, t, *rest):
    bias, (pre, index, rid, lines, cid, *rest) = (rest[0], rest[1:]) if S.BIAS else (None, rest)
    shape = Shape(S.M, S.K, S.N)
    padids = rest[0] if shape.width > S.M else None
    nodeids = rest[-1] if upper_nodes(shape) else None
    return tuple_of(*_call_units(B, S, a, s, hs, amps, t, bias, pre, index, rid, lines, cid, padids, nodeids))


NcpFp8LinearDef = CompositeDefinition(
    "NcpFp8Linear", 1, ("M", "K", "N", "SUMS", "IL", "OL", "WL", "BIAS"), _linear_sig, _linear_body,
    doc="One NCP-FP8 (D-3s) linear call under sampled proofs: y = dequant(R_T) with #295's checked words, as instances of "
        "the unit templates only; its weight's h and amplitudes are committed inputs (`NcpFp8Weight`).",
    conformance="protocol-defined")

#: a window's per-call inputs, each with a leading axis of C
WINDOW_PER_CALL = ("a", "s", "prefix", "index", "rowids", "lines", "colids", "padids", "nodeids")


def _window_sig(S):
    call = dict(_call_sig(S.M, S.K, S.N, S.SUMS, Lengths(S.IL, S.OL, S.WL), S.BIAS))
    params = [("b", Array(S.N, Array(S.K, U8))), ("t", call["t"])] + ([("bias", call["bias"])] if S.BIAS else [])
    params += [(p, Array(S.C, call[p])) for p in WINDOW_PER_CALL if p in call]
    out = _call_out(S.M, S.K, S.N)
    return tuple(params), Tuple(Array(S.C, out.fields[0]), Array(S.C, out.fields[1]))


def _window_body(B, S, b, t, *rest):
    bias, (a, s, pre, index, rid, lines, cid, *rest) = (rest[0], rest[1:]) if S.BIAS else (None, rest)
    shape = Shape(S.M, S.K, S.N)
    pads, ups = shape.width > S.M, bool(upper_nodes(shape))
    padids = rest[0] if pads else None
    nodeids = rest[-1] if ups else None
    hs, amps = _weight_units(B, S, b)
    calls = [_call_units(B, S, a[c], s[c], hs, amps, t, bias, pre[c], index[c], rid[c], lines[c], cid[c],
                         padids[c] if pads else None, nodeids[c] if ups else None) for c in range(S.C)]
    return tuple_of(array_of([y for y, _ in calls]), array_of([w for _, w in calls]))


NcpFp8WindowDef = CompositeDefinition(
    "NcpFp8Window", 1, ("M", "K", "N", "SUMS", "IL", "OL", "WL", "C", "BIAS"), _window_sig, _window_body,
    doc="C calls on one weight in one root: the weight's h and amplitudes once (`NcpFp8Amp`), then each call's units "
        "reading them, each call with its own anchored inputs.", conformance="protocol-defined")


def domain_template(m: int, k: int, lg: Lengths) -> SpecializedDefinition:
    return bind(NcpFp8DomainDef, M=m, K=k, IL=lg.il, OL=lg.ol)


def digest_template(k: int, sig: Sequence[int]) -> SpecializedDefinition:
    return bind(NcpFp8DigestDef, K=k, RL=sig[0], L0=sig[1], L1=sig[2], L2=sig[3], L3=sig[4])


def pad_template(sig: Sequence[int]) -> SpecializedDefinition:
    return bind(NcpFp8PadDef, RL=sig[0], L0=sig[1], L1=sig[2], L2=sig[3], L3=sig[4])


def node_template(length: int) -> SpecializedDefinition:
    return bind(NcpFp8NodeDef, L=length)


def key_template(k: int, sums: str, lg: Lengths, m: int, n: int) -> SpecializedDefinition:
    return bind(NcpFp8KeyDef, K=k, SUMS=sums, IL=lg.il, WL=lg.wl, M=m, N=n)


def form_template(left: bool, k: int, sums: str, lg: Lengths, ll: int) -> SpecializedDefinition:
    return bind(NcpFp8FormXDef if left else NcpFp8FormYDef, K=k, SUMS=sums, IL=lg.il, WL=lg.wl, LL=ll)


def tile_template(k: int) -> SpecializedDefinition:
    return bind(NcpFp8TileDef, K=k)


def amp_template(k: int, sums: str) -> SpecializedDefinition:
    return bind(NcpFp8AmpDef, K=k, SUMS=sums)


def weight_templates(k: int, sums: str = "f16") -> list[tuple[str, SpecializedDefinition]]:
    """The weight Program's unit template, with its role."""
    return [("amp", amp_template(k, sums))]


def call_templates(shape: Shape, sums: str = "f16", lengths: Lengths = Lengths(1, 1, 1),
                   bias: bool = True) -> list[tuple[str, SpecializedDefinition]]:
    """Every unit template one call instantiates, with its role, in the order its root body first uses them."""
    m, k, n = shape.m, shape.k, shape.n
    out = [("domain", domain_template(m, k, lengths))]
    out += [("digest", digest_template(k, sig)) for _lo, _hi, sig in _runs([strip_lengths(D * s) for s in range(m // D)])]
    out += [("pad", pad_template(sig)) for _lo, _hi, sig in _runs([strip_lengths(m + D * s) for s in range((shape.width - m) // D)])]
    out += [("node", node_template(ln)) for ln in sorted({_ulen(j) for _lv, j in upper_nodes(shape)})]
    out += [("key", key_template(k, sums, lengths, m, n))]
    out += [("form_y", form_template(False, k, sums, lengths, ll)) for ll in sorted({_ulen(D * c) for c in range(n // D)})]
    out += [("form_x", form_template(True, k, sums, lengths, ll)) for ll in sorted({_ulen(D * s) for s in range(m // D)})]
    out += [("tile", tile_template(k)), ("dequant", dequant_template(bias))]
    seen, uniq = set(), []
    for role, t in out:
        if t.id not in seen:
            seen.add(t.id)
            uniq.append((role, t))
    return uniq


def linear(shape: Shape, sums: str = "f16", lengths: Lengths = Lengths(1, 1, 1),
           bias: bool = True) -> SpecializedDefinition:
    """`NcpFp8Linear{M,K,N,SUMS,IL,OL,WL,BIAS}` for one call (``sums`` "f16" is #295's target cell `d3s-v0-f16`), of a
    linear with a bias or without (vLLM's two epilogues)."""
    shape.check()
    tag(sums)
    return bind(NcpFp8LinearDef, M=shape.m, K=shape.k, N=shape.n, SUMS=sums, IL=lengths.il, OL=lengths.ol, WL=lengths.wl,
                BIAS=bias)


def weight(shape: Shape, sums: str = "f16") -> SpecializedDefinition:
    """`NcpFp8Weight{K,N,SUMS}`: the weight side of every call of this shape's (k, n)."""
    shape.check()
    tag(sums)
    return bind(NcpFp8WeightDef, K=shape.k, N=shape.n, SUMS=sums)


def window(shape: Shape, calls: int, sums: str = "f16", lengths: Lengths = Lengths(1, 1, 1),
           bias: bool = True) -> SpecializedDefinition:
    """`NcpFp8Window{M,K,N,SUMS,IL,OL,WL,C,BIAS}`: ``calls`` calls of this shape on one weight, in one root, their indices
    of one byte length."""
    shape.check()
    tag(sums)
    return bind(NcpFp8WindowDef, M=shape.m, K=shape.k, N=shape.n, SUMS=sums, IL=lengths.il, OL=lengths.ol, WL=lengths.wl,
                C=calls, BIAS=bias)


__all__ = ["BF16", "D", "F32", "Family", "Hoisted", "Lengths", "NcpFp8Dequant", "SUMS", "Shape", "WINDOW_PER_CALL",
           "amp_template", "amps_type", "call_templates", "dequant_template", "digest_template", "domain_layout",
           "domain_template",
           "form_template", "hs_type", "hw_type", "key_layout", "key_template", "leaf_digest", "leaf_fn", "linear", "node_fn",
           "node_template", "pad_fn", "pad_template", "perm", "prefix", "prefix_parts", "row_fn", "strip_lengths",
           "subtile_digest", "tag", "tile_template", "tile_words", "upper_nodes", "weight", "weight_templates", "window"]
