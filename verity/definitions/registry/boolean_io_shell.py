"""`io_shell`'s Definitions on bits: `IngressPrompt_v2{MODEL,LP,V}`, its word body line for line over gates (the body is
`io_shell.parse`, written once over its steps), so its word view is its `_v1` at the same statics;
`IngressSampled_v1{MODEL,LP,V,RID}`, ingress schema v1's parse (`io_shell`'s module docstring), which has no word view: its seed
is a 64-bit sum; and `ScanSlots_v2{NS,NP,IDB,K0,K,OCC,LP,MAX,IDS}`, the scan state row's read (`io_shell`'s module docstring),
which has no word view either: its port is the row's bits.

`IngressSampled_v1` is `IngressPrompt_v2`'s steps over schema v1's layout (`io_shell.layout`), then each byte of the user, each
ratio's seven digits and its point, top_p's bounds, and the seed field: `io_shell.field_checks`' checks, its value a 64-bit sum
of the same constant selects, and ``null`` as nineteen bytes.

A step is `boolean_dsv4_indices.GateOps`' gates for the word primitive `io_shell.WORD` names (`forms`' integer add and signed
compare, a select as `forms.mux`, an AND and a NOT of bits), and bit i of a byte as that wire.  A constant word is constant bits,
so a field's selects of constant words are ANDs with one wire and its sums fold above the field's largest value.  No table or
lookup gate.  Every byte of the body reaches ``ok``.

`ScanSlots_v2` copies each lane's fields out of the row as they are, its ids zero-extended, and computes one thing: the slot
its first position carries on, whose count is the prompt's length minus one (mod 2^32), a borrow chain of 30 ANDs a lane.
Every lane's is computed in every unit, so a unit's lanes but its own Program's are its cost and nobody's input.
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, and_reduce
from verity.circuits.defs import bind
from verity.definitions.ops.boolean import elementwise as EW
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import io_shell as X
from verity.definitions.registry.boolean_dsv4_indices import GateOps

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


def _scan(S):
    def run(C, *state):
        x = [b for w in state for b in w]

        def word(lo: int, w: int) -> list:
            return x[lo:lo + w] + [ZERO] * (32 - w)

        out = []
        for o, lp, m, ids in zip(S.OCC, S.LP, S.MAX, S.IDS):
            at = [ids + S.IDB * X.position(S.K0, S.K, p) for p in range(S.NP)]
            out += [word(at[0], S.IDB), decrement(C, word(lp, 32)), *[word(a, S.IDB) for a in at[1:]], [ZERO] * 32,
                    word(o, 32), word(lp, 32), word(m, 32)]
        return out
    return run


ScanSlotsV2 = EW.specialized("ScanSlots", 2, ["NS", "NP", "IDB", "K0", "K", "OCC", "LP", "MAX", "IDS"], _scan, X.slots_sig,
                             doc="Each lane's first position's token and slot, occupancy word and head from the scan's state "
                                 "row (`io_shell`'s module docstring; `io_shell.scan_reference`).",
                             refuse=X.slots_refuse,
                             conformance="not-yet-established: the network certifier's scan state row as its Programs read "
                                         "their slots, and each slot's first position's token feed")

BOOLEAN_IO_SHELL = [IngressPromptV2, IngressSampledV1, ScanSlotsV2]
