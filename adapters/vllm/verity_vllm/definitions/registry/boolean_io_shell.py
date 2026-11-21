"""`io_shell`'s Definitions on bits: `IngressPrompt_v2{MODEL,LP,V}`, its word body line for line over gates (the body is
`io_shell.parse`, written once over its steps), so its word view is its `_v1` at the same statics;
`IngressSampled_v1{MODEL,LP,V,RID}`, ingress schema v1's parse (`io_shell`'s module docstring), which has no word view: its seed
is a 64-bit sum; and `ScanSlots_v5{NS,NP,IDB,K0,K,OCC,LP,MAX,IDS}`, one lane's read of the scan state row (`io_shell`'s
module docstring), which has no word view either: its gates read the row's bits.

`IngressSampled_v1` is `IngressPrompt_v2`'s steps over schema v1's layout (`io_shell.layout`), then each byte of the user, each
ratio's seven digits and its point, top_p's bounds, and the seed field: `io_shell.field_checks`' checks, its value a 64-bit sum
of the same constant selects, and ``null`` as nineteen bytes.

A step is `boolean_gate_ops.SelectGateOps`' gates for the word primitive `io_shell.WORD` names (`forms`' integer add and signed
compare, a select as `forms.mux`, an AND and a NOT of bits), and bit i of a byte as that wire.  A constant word is constant bits,
so a field's selects of constant words are ANDs with one wire and its sums fold above the field's largest value.  No table or
lookup gate.  Every byte of the body reaches ``ok``.

`ScanSlots_v5` takes the row as the u16 words it is registered as and gives its lane's fields as the i32s its readers take:
the token `EmbeddingRead_v3`'s, the slot, occupancy word and head `TokenFeed_v1`'s, `SelectEntry_v1`'s and
`SelectCollect_v1`'s. Its words become bits through `WordBits{16}`, which is wiring, and one call of
`ScanSlotsBits_v1{NS,NP,IDB,K0,K,OCC,LP,MAX,IDS}` on those bits: it copies the lane's fields out of the row as they are, its
ids zero-extended, computes one thing, the slot its first position carries on, whose count is the prompt's length minus one
(mod 2^32), a borrow chain of 30 ANDs, and makes its bits words through `BitsWord{32}`, wiring too. Its gates read its own
parameters, so no batch of them steps through part of a `WordBits` copy (`Flock.Lower`'s `Work.stepOf`).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, and_reduce
from verity.circuits.boolean.trace import emit, trace
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.ops.boolean import elementwise as EW
from verity.circuits.population import BitsWord, WordBits
from verity.circuits.evaluation.batch import register_kernel
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array, Tuple, Value

from verity_vllm.definitions.registry import io_shell as X
from verity_vllm.definitions.registry.boolean_gate_ops import SelectGateOps as GateOps

BITS = SimpleNamespace(U8=Array(8, BL.BIT), BIT=BL.BIT, I32=Array(32, BL.BIT), I64=Array(64, BL.BIT))


def _parse(S):
    def run(C, *body):
        ok, ids, m = X.parse(GateOps(C), S, list(body))
        return [[ok], *ids, m]
    return run


IngressPromptV2 = EW.specialized("IngressPrompt", 2, ["MODEL", "LP", "V"], _parse, X.ingress_sig(BITS),
                                 doc=f"`IngressPrompt_v1`'s body on bits: {X.parse.__doc__}",
                                 word=lambda b: bind(X.IngressPrompt, **b))


def sampled_sig(S):
    n = X.layout(S.MODEL, S.LP, S.V, S.RID)[2]
    return ((("body", Array(n, BITS.U8)),),
            Tuple(BITS.BIT, Array(S.LP, BITS.I32), BITS.I32, BITS.I32, BITS.I32, BITS.BIT, BITS.I64, Array(S.RID, BITS.U8)))


def _byte_is(C, bits: list, c: int):
    return and_reduce(C, [b if c >> i & 1 else NOT(b) for i, b in enumerate(bits)])


def _digit(C, bits: list):
    hi = and_reduce(C, [NOT(bits[7]), NOT(bits[6]), bits[5], bits[4]])
    return C.AND(hi, NOT(C.AND(bits[3], OR(C, bits[2], bits[1]))))


def _name_byte(C, bits: list):
    """A byte in [0x20, 0x7F) that is no quote and no backslash."""
    return and_reduce(C, [NOT(bits[7]), OR(C, bits[6], bits[5]), *[NOT(_byte_is(C, bits, c)) for c in (0x7F, 0x22, 0x5C)]])


def _sum(C, chars: list, width: int) -> list:
    """The characters' low nibbles at their decimal places in ``width`` bits: each constant select is its bit's wire at the
    constant's ones."""
    v = [ZERO] * width
    for k, b in enumerate(chars):
        for j in range(4):
            c = (1 << j) * 10 ** (len(chars) - 1 - k)
            v = add(C, v, [b[j] if c >> i & 1 else ZERO for i in range(width)], width=width)
    return v


def sampled_parse(C, S, body: list) -> list:
    """`IngressSampled_v1`'s body: (ok, ids, max_tokens, temperature, top_p, seeded, seed, user) of `io_shell`'s module
    docstring."""
    o = GateOps(C)
    oks, ids, m, ((t, _), (p, _), (s, sw), (u, _)) = X.parse_head(o, S, body, S.RID)
    user = body[u:u + S.RID]
    oks += [_name_byte(C, c) for c in user]
    ratios = []
    for off in (t, p):
        r = body[off:off + X.RATIO_WIDTH]
        oks += [_byte_is(C, r[1], 0x2E), *[_digit(C, c) for c in r[:1] + r[2:]]]
        ratios.append(_sum(C, r[:1] + r[2:], 32))
    oks += [o.ile(o.word(1), ratios[1]), o.ile(ratios[1], o.word(10 ** 6))]
    chars = body[s:s + sw]
    s_oks, _ = X.field_checks(o, chars)
    seed = _sum(C, chars, 64)
    null = and_reduce(C, [_byte_is(C, c, b) for c, b in zip(chars, X.NULL.rjust(sw))])
    oks.append(OR(C, null, C.AND(and_reduce(C, s_oks), NOT(seed[63]))))
    return [[and_reduce(C, oks)], *ids, m, *ratios, [NOT(null)], seed, *[list(c) for c in user]]


def _sampled(S):
    return lambda C, *body: sampled_parse(C, S, list(body))


IngressSampledV1 = EW.specialized("IngressSampled", 1, ["MODEL", "LP", "V", "RID"], _sampled, sampled_sig,
                                  refuse=X.sampled_refuse, doc=f"Ingress schema v1's parse: {sampled_parse.__doc__}",
                                  conformance="protocol-defined: ingress schema v1's parse, whose function is "
                                              "`io_shell.sampled_reference`")


def decrement(C, bits: list) -> list:
    """``bits`` (least significant first) minus one, mod 2^len: bit i flips while every bit below it is 0."""
    out, borrow = [], ONE
    for a in bits:
        out.append(a ^ borrow)
        borrow = C.AND(borrow, a.inv())
    return out


def _spans(S) -> list[tuple[int, int]]:
    return [(S.OCC, 32), (S.LP, 32), (S.MAX, 32), *((S.IDS + S.IDB * X.position(S.K0, S.K, p), S.IDB) for p in range(S.NP))]


def read_words(S) -> list[int]:
    return sorted({b // 16 for a, w in _spans(S) for b in range(a, a + w)})


def _scan(S):
    words = read_words(S)

    def run(C, *state):
        x = {16 * j + i: b for j, w in zip(words, state) for i, b in enumerate(w)}

        def word(lo: int, w: int) -> list:
            return [x[lo + i] for i in range(w)] + [ZERO] * (32 - w)

        at = [S.IDS + S.IDB * X.position(S.K0, S.K, p) for p in range(S.NP)]
        return [word(at[0], S.IDB), decrement(C, word(S.LP, 32)), *[word(a, S.IDB) for a in at[1:]], [ZERO] * 32,
                word(S.OCC, 32), word(S.LP, 32), word(S.MAX, 32)]
    return run


SCAN_STATICS = ("NS", "NP", "IDB", "K0", "K", "OCC", "LP", "MAX", "IDS")
_SCANS: dict[tuple, Any] = {}


def scan_bits(S) -> tuple[Any, list, Any]:
    params = [Array(len(read_words(S)), Array(16, BL.BIT))]
    ret = Tuple(X.BITS32, Array(S.NP + 1, X.BITS32), X.BITS32, Array(2, X.BITS32))
    key = tuple(getattr(S, k) for k in SCAN_STATICS)
    if key not in _SCANS:
        _SCANS[key] = trace(_scan(S), params, ret)
    return _SCANS[key], params, ret


def _bits_sig(S):
    _T, params, _ret = scan_bits(S)
    return (("bits", params[0]),), X.slots_sig(S)[1]


def _scan_bits_words(B, S, bits):
    T, params, ret = scan_bits(S)
    out = emit(T, params, ret, B, bits)
    words = B.batch(bind(BitsWord, W=32), Coll(Array(S.NP + 5, X.BITS32), out.refs), axes=(0,))
    return Coll(X.slots_sig(S)[1], words.refs)


ScanSlotsBitsV1 = CompositeDefinition("ScanSlotsBits", 1, SCAN_STATICS, _bits_sig, _scan_bits_words,
                                      doc="`ScanSlots_v5`'s lane fields from the read words' bits, each an i32.",
                                      refuse=X.slots_refuse,
                                      conformance="not-yet-established: as `ScanSlots_v5`")


def _scan_words(B, S, state):
    _T, params, _ret = scan_bits(S)
    read = Coll(Array(params[0].n, Value(16)), concat([state.refs.slice(j, j + 1) for j in read_words(S)]))
    bits = B.batch(bind(WordBits, W=16), read, axes=(0,))
    out = B.call(bind(ScanSlotsBitsV1, **{k: getattr(S, k) for k in SCAN_STATICS}), Coll(params[0], bits.refs))
    return Coll(X.slots_sig(S)[1], out.refs)


ScanSlotsV5 = CompositeDefinition("ScanSlots", 5, SCAN_STATICS, X.slots_sig, _scan_words,
                                  doc="One lane's first position's token and slot, occupancy word and head from the scan's "
                                      "state row of u16 words, each an i32 (`io_shell`'s module docstring; "
                                      "`io_shell.scan_reference`).",
                                  refuse=X.slots_refuse,
                                  conformance="not-yet-established: the network attester's scan state row as its Programs "
                                              "read their slots, and each slot's first position's token feed")


def scan_rows(state, NP, IDB, K0, K, OCC, LP, MAX, IDS, **_):
    lanes = [X.scan_reference(r, NP, IDB, K0, K, OCC, LP, MAX, IDS) for r in np.asarray(state, dtype=np.uint64).tolist()]
    return np.asarray([[k, *s, o, *h] for k, s, o, h in lanes], dtype=np.uint64), None


register_kernel(ScanSlotsV5, scan_rows, domain="every state row")

BOOLEAN_IO_SHELL = [IngressPromptV2, IngressSampledV1, ScanSlotsBitsV1, ScanSlotsV5]
