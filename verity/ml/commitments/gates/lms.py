"""The LMS verifier (`verity.ml.crypto.lms`, RFC 8554 over SHA-512) as Boolean Definitions: what a proof runs to accept
a network certificate or a memory challenger's record without revealing it.

    LmsVerify_v1{N, W, H}
        (I: Array<2, Word>, root: Value_N, message: Array<8, Word>, q: Array<H, Value<1>>, C: Value_N,
         y: Array<P, Value_N>, path: Array<H, Value_N>) -> Value<1>
        Word = Array<64, Value<1>>, Value_N = Array<N / 8, Word>; N = 64, W in (1, 2, 4, 8), 1 <= H <= 30
        (or N = 8, `lms.TOY_N`: circuit-check's binding, the same circuit on one-word values, never deployed)

1 iff ``u32(q) || C || y || path`` signs ``message`` under the public key ``I || root`` (`lms.verify`). Every N-byte value,
the identifier and the message are SHA-512 words as `merkle` lays them out (bytes big-endian, each word's bits LSB first);
``q`` is LSB first. An accepted signature means the key's holder signed the message under the claim
``euf-cma/lms-sha512-n64`` (`verity.core.claims`), the scheme's unforgeability, which rests on SHA-512 itself, its whole output.

**A fixed shape.** The circuit is a function of (N, W, H): the digits, the leaf index and the path are values. A chain runs all
2^W - 1 steps, and step j keeps its input unless the chain's digit a is at most j (`LmsChainStep_v1`), so the proof pays
P (2^W - 1) chain hashes whatever the digits. The tree level puts the current node on the side q's bit names
(`LmsNode_v1`, a conditional swap).

**The parts,** each its own traced Definition (`trace.part`, which C-Flock lowers by its builder), so a unit of a proof is at
most one level of the tree's two compressions:

    Sha512Compress_v1                 (cv: Array<8, Word>, block: Array<16, Word>) -> Array<8, Word>, one compression
    LmsMessage_v1{N, W}               (I, q32, C, message) -> Array<P, Array<W, Value<1>>>: Q and its checksum's digits
    LmsChainStep_v1{N, W}             (I, q32, i: Array<16>, j: Array<W>, a: Array<W>, s) -> H(I||q||i||j||s) if a <= j else s
    LmsLeaf_v1{N}                     (I, r: Array<32>, K) -> H(I || u32(r) || D_LEAF || K)
    LmsNode_v1{N}                     (I, r, d, cur, sib) -> H(I || u32(r) || D_INTR || cur || sib), the children swapped if d
    LmsEqual_v1{N}                    (x, y) -> 1 iff x = y

The one-time public key ``K`` is ``Sha512Compress_v1`` over its blocks, wired in `LmsVerify_v1`'s body (the chain ends are
whole bytes at fixed offsets), and the node numbers are q's bits shifted under a constant 1. Each part's message is its
inputs and constants laid out as RFC 8554's bytes, so constant bytes (the identifier's offsets, the separators, the padding)
fold into the traced circuit; a ``Sha512Compress_v1`` call does not fold them, being one circuit for every block.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import ZERO, and_reduce, compress_columns, const_bits, ge_const_free
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array
from verity.ml.crypto import lms as R

from .sha512 import IV, compress_gadget, pad_tail, words_be

__all__ = ["LmsChainStep", "LmsEqual", "LmsLeaf", "LmsMessage", "LmsNode", "LmsVerify", "Sha512Compress", "arguments",
           "chain_step", "leaf_hash", "message_digits", "node_hash", "verifier"]

BIT = BL.BIT
WORD = Array(64, BIT)
DIGEST = Array(8, WORD)
IDENT = Array(2, WORD)
U32 = Array(32, BIT)


def _value(N: int):
    return Array(N // 8, WORD)


def _params(S) -> R.Params:
    return R.Params(S.N, S.W, getattr(S, "H", 1), toy=S.N == R.TOY_N)


# --- bytes and words ---------------------------------------------------------------------------------------------------------
# A byte is 8 bits LSB first; a word is 64 bits LSB first, its bytes big-endian. These rearrange bits of any kind (forms, refs).

def _bytes(words: Sequence[Sequence]) -> list[list]:
    return [list(w[8 * (7 - b):8 * (8 - b)]) for w in words for b in range(8)]


def _words(data: Sequence[Sequence]) -> list[list]:
    assert len(data) % 8 == 0
    return [[x for b in range(8) for x in data[8 * k + 7 - b]] for k in range(len(data) // 8)]


def _const_bytes(data: bytes) -> list[list]:
    return [const_bits(b, 8) for b in data]


def _u32(bits: Sequence) -> list[list]:
    return [list(bits[8 * (3 - b):8 * (4 - b)]) for b in range(4)]


def _sha(C, data: list[list], n_words: int) -> list[list]:
    """SHA-512 over ``data``'s bytes from the constant IV, its first ``n_words`` words."""
    msg = _words(data + _const_bytes(pad_tail(len(data))))
    st = [const_bits(v, 64) for v in IV]
    for i in range(0, len(msg), 16):
        st = compress_gadget(C, st, msg[i:i + 16])
    return st[:n_words]


def _join(*ws: Sequence) -> list:
    return [x for w in ws for x in w]


# --- the parts over forms ----------------------------------------------------------------------------------------------------

def chain_step(C, N: int, W: int, I, q, i, j, a, s):
    """``H(I || u32(q) || u16(i) || u8(j) || s)`` if ``a <= j``, else ``s`` (words of forms; q, i, j, a bits LSB first)."""
    h = _sha(C, _bytes(I) + _u32(q) + [list(i[8:16]), list(i[0:8])] + [list(j) + [ZERO] * (8 - W)] + _bytes(s), N // 8)
    on = ge_const_free(C, list(j), list(a), W)
    return [[y ^ C.AND(on, x ^ y) for x, y in zip(hw, sw)] for hw, sw in zip(h, s)]


def message_digits(C, N: int, W: int, I, q, Cv, message) -> list[list]:
    """The P digits of ``Q = H(I || u32(q) || u16(D_MESG) || C || message)`` and its checksum, each W bits LSB first."""
    p = _params(SimpleNamespace(N=N, W=W))
    Q = _bytes(_sha(C, _bytes(I) + _u32(q) + _const_bytes(R.D_MESG.to_bytes(2, "big")) + _bytes(Cv) + _bytes(message), N // 8))
    per = 8 // W
    digits = [Q[k // per][8 - W * (k % per + 1):8 - W * (k % per)] for k in range(p.U)]
    width = p.V * W
    total = compress_columns(C, [[d[b].inv() for d in digits] for b in range(W)])[:width]
    total += [ZERO] * (width - len(total))
    return digits + [total[W * (p.V - 1 - k):W * (p.V - k)] for k in range(p.V)]


def leaf_hash(C, N: int, I, r, K):
    return _sha(C, _bytes(I) + _u32(r) + _const_bytes(R.D_LEAF.to_bytes(2, "big")) + _bytes(K), N // 8)


def node_hash(C, N: int, I, r, d, cur, sib):
    """The interior node over ``cur`` and its sibling, ``cur`` the right child iff ``d``."""
    t = [[C.AND(d, x ^ y) for x, y in zip(cw, sw)] for cw, sw in zip(cur, sib)]
    left = [[x ^ u for x, u in zip(cw, tw)] for cw, tw in zip(cur, t)]
    right = [[y ^ u for y, u in zip(sw, tw)] for sw, tw in zip(sib, t)]
    return _sha(C, _bytes(I) + _u32(r) + _const_bytes(R.D_INTR.to_bytes(2, "big")) + _bytes(left) + _bytes(right), N // 8)


# --- traced Definitions ------------------------------------------------------------------------------------------------------

def _group(e: Sequence, n: int) -> list[list]:
    return [list(e[k:k + n]) for k in range(0, len(e), n)]


def _check(S) -> R.Params:
    return _params(S)


def _sig_compress(S):
    return (("cv", DIGEST), ("block", Array(16, WORD))), DIGEST


Sha512Compress = TR.part(
    "Sha512Compress", (), _sig_compress, lambda C, S, *e: compress_gadget(C, list(e[:8]), list(e[8:24])),
    "One SHA-512 compression (FIPS 180-4 §6.4.2) with its feed-forward, on 64-bit words LSB first")


def _sig_message(S):
    p = _check(S)
    return (("I", IDENT), ("q", U32), ("C", _value(S.N)), ("message", DIGEST)), Array(p.P, Array(S.W, BIT))


def _build_message(C, S, *e):
    k = S.N // 8
    return message_digits(C, S.N, S.W, e[0:2], e[2], e[3:3 + k], e[3 + k:11 + k])


LmsMessage = TR.part(
    "LmsMessage", ("N", "W"), _sig_message, _build_message,
    "The P Winternitz digits of Q = H(I || u32(q) || u16(D_MESG) || C || message) and of its checksum (RFC 8554 §4.4)")


def _sig_step(S):
    _check(S)
    V = _value(S.N)
    return (("I", IDENT), ("q", U32), ("i", Array(16, BIT)), ("j", Array(S.W, BIT)), ("a", Array(S.W, BIT)), ("s", V)), V


LmsChainStep = TR.part(
    "LmsChainStep", ("N", "W"), _sig_step,
    lambda C, S, *e: chain_step(C, S.N, S.W, e[0:2], e[2], e[3], e[4], e[5], e[6:]),
    "Step j of chain i: H(I || u32(q) || u16(i) || u8(j) || s) if the chain's digit a <= j, else s")


def _sig_leaf(S):
    _check(SimpleNamespace(N=S.N, W=1))
    V = _value(S.N)
    return (("I", IDENT), ("r", U32), ("K", V)), V


LmsLeaf = TR.part(
    "LmsLeaf", ("N",), _sig_leaf, lambda C, S, *e: leaf_hash(C, S.N, e[0:2], e[2], e[3:]),
    "The tree's leaf r over the one-time public key K: H(I || u32(r) || u16(D_LEAF) || K)")


def _sig_node(S):
    _check(SimpleNamespace(N=S.N, W=1))
    V = _value(S.N)
    return (("I", IDENT), ("r", U32), ("d", BIT), ("cur", V), ("sib", V)), V


def _build_node(C, S, *e):
    k = S.N // 8
    return node_hash(C, S.N, e[0:2], e[2], e[3][0], e[4:4 + k], e[4 + k:4 + 2 * k])


LmsNode = TR.part(
    "LmsNode", ("N",), _sig_node, _build_node,
    "The tree's node r over cur and its sibling: H(I || u32(r) || u16(D_INTR) || left || right), cur the right child iff d")


def _sig_equal(S):
    _check(SimpleNamespace(N=S.N, W=1))
    return (("x", _value(S.N)), ("y", _value(S.N))), BIT


def _build_equal(C, S, *e):
    k = S.N // 8
    return [and_reduce(C, [(x ^ y).inv() for a, b in zip(e[:k], e[k:]) for x, y in zip(a, b)])]


LmsEqual = TR.part("LmsEqual", ("N",), _sig_equal, _build_equal, "1 iff x = y")


# --- the verifier ------------------------------------------------------------------------------------------------------------

def _signature(S):
    p = _check(S)
    V = _value(S.N)
    return ((("I", IDENT), ("root", V), ("message", DIGEST), ("q", Array(S.H, BIT)), ("C", V), ("y", Array(p.P, V)),
             ("path", Array(S.H, V))), BIT)


class _Wires:
    """Bit references in a body: the constant bits, and Colls of any references."""

    def __init__(self, B) -> None:
        self.bit = [B.call(BL.constant(b)).refs[0] for b in (0, 1)]

    def const(self, value: int, width: int) -> list:
        return [self.bit[(value >> k) & 1] for k in range(width)]

    @staticmethod
    def refs(c: Coll) -> list:
        return [c.refs[k] for k in range(c.refs.count)]

    @staticmethod
    def coll(t, refs: list) -> Coll:
        return Coll(t, TR._refs(list(refs)))


def _body(B, S, I, root, message, q, Cv, y, path):
    p, k = _check(S), S.N // 8
    X = _Wires(B)
    qb = X.refs(q)
    q32 = X.coll(U32, qb + X.const(0, 32 - S.H))
    digits = B.call(bind(LmsMessage, N=S.N, W=S.W), I, q32, Cv, message)
    chains = X.coll(Array(p.P, Array(16, BIT)), [b for i in range(p.P) for b in X.const(i, 16)])
    step = bind(LmsChainStep, N=S.N, W=S.W)
    z = y
    for j in range(2 ** S.W - 1):
        z = B.batch(step, I, q32, chains, X.coll(Array(S.W, BIT), X.const(j, S.W)), digits, z, axes=(None, None, 0, None, 0, 0))
    data = (_bytes(_group(X.refs(I), 64)) + _u32(qb + X.const(0, 32 - S.H))
            + [X.const(b, 8) for b in R.D_PBLC.to_bytes(2, "big")] + _bytes(_group(X.refs(z), 64)))
    data += [X.const(b, 8) for b in pad_tail(len(data))]
    words = _words(data)
    cv = X.coll(DIGEST, [b for v in IV for b in X.const(v, 64)])
    for blk in range(0, len(words), 16):
        cv = B.call(Sha512Compress, cv, X.coll(Array(16, WORD), _join(*words[blk:blk + 16])))
    K = cv[0:k]
    top = X.const(1, 1)
    cur = B.call(bind(LmsLeaf, N=S.N), I, X.coll(U32, qb + top + X.const(0, 31 - S.H)), K)
    node = bind(LmsNode, N=S.N)
    for lv in range(S.H):
        r = X.coll(U32, qb[lv + 1:] + top + X.const(0, 32 - S.H + lv))
        cur = B.call(node, I, r, q[lv], cur, path[lv])
    return B.call(bind(LmsEqual, N=S.N), cur, root)


LmsVerify = CompositeDefinition(
    "LmsVerify", 1, ("N", "W", "H"), _signature, _body, conformance="exact-model-tested",
    doc="1 iff u32(q) || C || y || path is an LMS signature (RFC 8554 over SHA-512, N bytes of its output, Winternitz width W, "
        "tree height H, no type fields) of message under the public key I || root; every chain runs its 2^W - 1 steps; "
        "sound under the claim euf-cma/lms-sha512-n64 at N = 64")


def verifier(N: int, W: int, H: int):
    return bind(LmsVerify, N=N, W=W, H=H)


def _value_bits(v: bytes) -> list[int]:
    return [(w >> b) & 1 for w in words_be(v) for b in range(64)]


def arguments(params: R.Params, public: bytes, message: bytes, sig: R.Signature) -> tuple[list[int], ...]:
    """``LmsVerify_v1{N, W, H}``'s seven arguments, each a flat list of bits in leaf order."""
    n = params.N
    if len(public) != 16 + n or len(message) != 64 or len(sig.C) != n:
        raise ValueError(f"a public key is {16 + n} bytes, the message 64 and C {n}")
    if len(sig.y) != params.P or len(sig.path) != params.H or not 0 <= sig.q < 2 ** params.H:
        raise ValueError(f"a signature for {params} has P = {params.P} chain values, H path nodes and q below 2^H")
    return (_value_bits(public[:16]), _value_bits(public[16:]), _value_bits(message), [(sig.q >> b) & 1 for b in range(params.H)],
            _value_bits(sig.C), [b for v in sig.y for b in _value_bits(v)], [b for v in sig.path for b in _value_bits(v)])
