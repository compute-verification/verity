"""The compiled server circuit's ingress: the request bytes the warden attests (the ingress transfer gates), parsed in-circuit to
the prompt's token ids and ``max_tokens``.

Ingress schema v0 is one OpenAI completions request with the prompt as token ids, as compact JSON in a fixed key order with every
integer right-aligned in a fixed-width field of leading spaces, then trailing spaces to a whole number of `BLOCK`-byte SHA-512
blocks (all JSON whitespace, so vLLM's OpenAI server reads it as it is):

    {"model":"<MODEL>","prompt":[<id>,<id>,...],"max_tokens":<m>,"temperature":0,"stream":true}<spaces>

An id's field is `id_width(V)` characters (6 for a 128,256-entry vocabulary), ``max_tokens``' is `MAX_TOKENS_WIDTH`, and the
padding is the fewest spaces that bring the body to a multiple of 128 bytes (1,024 bits), so the warden's attested blocks and the
parse unit's input rows line up and a registered read opens whole blocks (128 B divides the warden's 4 KiB cells).  So the byte
layout is a function of (MODEL, LP, V) alone, never of the prompt's digits: a template's statics carry no prompt content.  Greedy
only (``"temperature":0``).

Ingress schema v1 is v0 sampled: ``temperature``, ``top_p`` and ``seed`` are fields of the body, so a request's sampling is as
private as its prompt, and ``user`` is the request's id, which its seed derives from where it names none (the padding again
the fewest spaces to whole blocks):

    {"model":"<MODEL>","prompt":[<id>,...],"max_tokens":<m>,"temperature":<T>,"top_p":<P>,"seed":<s>,"user":"<RID bytes>",
     "stream":true}<spaces>

``temperature`` and ``top_p`` are `RATIO_WIDTH` characters, a digit, ``.`` and six digits (the network certifier's id request's
``%8.6f``), each read as an integer in millionths; ``seed`` is `SEED_WIDTH` characters, spaces then either decimal digits as an
id's (at least one, no leading zero) or ``null``; ``user`` is RID bytes of printable ASCII without a quote or backslash, its
length public as the prompt's is. `boolean_io_shell.IngressSampled_v1{MODEL, LP, V, RID}` (body) -> (ok, ids, max_tokens,
temperature, top_p, seeded: bit, seed: i64, user: Array<RID, u8>) parses it: ``ok`` is v0's checks, each ratio's form,
``top_p`` in (0, 1], the seed either ``null`` (``seeded`` 0) or below 2^63, and each byte of ``user``; ``seed`` is its field's
low nibbles at their decimal places, whose nineteen characters sum below 2^64 whatever they are, so its bit 63 is the bound
(`sampled_reference` is the function).

`IngressPrompt_v1{MODEL, LP, V}` (body: Array<N, Value<8>>) -> (ok: bit, ids: Array<LP, i32>, max_tokens: i32) checks every byte
against the layout and decodes the fields.  ``ok`` is 1 exactly when the body is in the schema: each literal byte is the layout's
(the padding's included: exactly those spaces, no other whitespace and no other length), each field is spaces then decimal digits
with at least one digit and no leading zero, each id is below V and ``max_tokens`` is at least 1.  A field's value is the sum of its characters' low nibbles at their decimal places (a space's low nibble is 0, and a
digit's is its value), so on a body that isn't in the schema the ids are still that sum: `reference` is the function exactly.
The body is written once over its steps (`parse`); `boolean_io_shell.IngressPrompt_v2` is it on gates.

`TokenFeed_v1{NP, FIRST}` feeds a request Program of a window class, whose prompts are at most NP ids, one token a position: the
prompt's id while the position is in the prompt, else the token the previous position selected, so a Program's units are the
class's whatever the prompt's length. It carries the slot from position to position: at position p the slot is
(lp − p, ids[p], …, ids[NP − 1], 0, …), NP + 1 i32s, lp the prompt's length. The first position's (FIRST) takes the prompt's
first id (a prompt has at least one); every other takes ``slot[1]`` while ``slot[0] − 1`` (mod 2^32) is non-negative, which is
``slot[0]`` positive on every slot but ``slot[0]`` = −2^31, else ``sel``: the sign of the count it carries on, so the count's one
carry chain is the test. Each outputs its token and the next position's slot. A window's slots take their first position's
from the scan state row (`boolean_io_shell.ScanSlots_v2`), so a window places only the later positions'.

A window has a fixed number of slots, each a class Program whether or not it carries a request, so whether a slot
is occupied is a private word of its ingress row, which its entry reads (`SlotEntry_v1`, all zero for an empty slot).

The network certifier's scan carries the slots: the Programs at a boundary read their slots from the scan's
state row before it (a registered row of NS u16 words, its fields bit-packed least significant bit first), and hand their
outputs to the scan's Programs' row at that boundary. `boolean_io_shell.ScanSlots_v2{NS, NP, IDB, K0, K, OCC, LP, MAX,
IDS}` is the read: for each of the row's lanes j (``OCC[j]``, ``LP[j]``, ``MAX[j]`` and ``IDS[j]`` the bit offsets of its
occupancy word, its prompt's length, its ``max_tokens`` and its ids; prompt position p's id the IDB bits at
``IDS[j] + IDB·position(p)`` zero-extended, `position`: K0 ids in a request's first cell, K in each later one), its first
position's token and the slot it carries (`TokenFeed_v1{NP, FIRST}`'s on (lp, ids[0], …, ids[NP − 1])), its occupancy word
and its head (lp, max_tokens). `slots_reference` is each lane's fields and `scan_reference` the Definition's function. `OutCollect_v1{KO, FIRST}` carries a slot's
outputs from position to position, (w, m, ids[0], …, ids[KO − 1]): w counts down from lp − 1, one a position, and while
1 − KO ≤ w ≤ 0 (the KO positions from the prompt's last on) the position's output is shifted in at the end, so after the
last position ids[i] is output i; m is ``max_tokens``. The first position's starts from the head. `SlotEntry_v1{KO, IDB,
NP, STOP}` is the slot's entry of the Programs' row from the last position's state and the slot's occupancy word, in the
scan's layout (`ingress_scan.entry_layout`): ``read``, the occupancy word's bit 0 (1 where the Program read an occupied
slot), then n, the outputs served (up to ``max_tokens``, unsigned, and the first of the model's stop ids ``STOP``, that
one included), as a u32, then output i's low IDB bits for i < n and zeros past it, all zero for an empty slot,
zero-padded to whole words. A Program that reads an empty slot where the scan holds a request gives ``read`` 0, which
the scan refuses. An occupied slot whose request the class can't hold has n = KO + 1 and no ids: its prompt not 1 to
NP ids (after a Program's NP + KO − 1 positions the state's w is lp − NP − KO) or its outputs not ending within KO
(``max_tokens`` past KO and no stop id among them). The scan refuses a Program's n over KO at its anchor
(`ingress_scan`), and the egress renders no output past KO − 1 (`egress.computed`), so such a record fails visibly.

A collector's rows and an entry's are a few i32s at a small KO, under the 112 bytes Flock's composer needs in some row
of a proof block, so a window proves them inside its token select's combine: `SelectCollect_v1{V, C, KO, FIRST}` is
`dead_free.TokenSelectCombine_v2{V, C}`'s select, then `OutCollect_v1{KO, FIRST}` over it, (the select, the state), and
`SelectEntry_v1{V, C, KO, FIRST, IDB, NP, STOP}` the last position's: the select, its collector, then `SlotEntry_v1` over
that state, the entry alone. Each is Calls of those Definitions, no gate of its own. `SelectCollect_v2{V, C, KO, FIRST,
ROUND}` is v1 at a round no gate reads: where a window holds a position's select to a row of a round's input-ids table, each
round's collectors are a template of their own, so a sample stratified by template has a reader of every round's table.

Registered functions:
  IngressPrompt{MODEL,LP,V}  (body: N u8) -> (ok: bit, LP i32, max_tokens: i32)
  IngressSampled{MODEL,LP,V,RID}
                             (body: N u8 bits) -> (ok: bit, LP i32, max_tokens, temperature, top_p: i32, seeded: bit, seed: i64,
                             user: RID u8), schema v1 (`boolean_io_shell`)
  TokenFeed{NP,FIRST}        (slot: NP + 1 i32[, sel: i32]) -> (tok: i32, slot: NP + 1 i32)
  ScanSlots{NS,NP,...}       (state: NS u16 bits) -> (tok: i32 bits, slot: NP + 1 i32 bits, occ: i32 bits, head: 2 i32 bits)
                             a lane (`boolean_io_shell`)
  OutCollect{KO,FIRST}       (head: 2 i32 | state: KO + 2 i32, tok: i32) -> KO + 2 i32
  SlotEntry{KO,IDB,NP,STOP}  (occ: i32, state: KO + 2 i32) -> ⌈(33 + KO·IDB) / 16⌉ u16 bits
  SelectCollect{V,C,KO,FIRST}
                             (x0: bf16, best: n bf16, idx: n i32, head | state) -> (tok: i32, state: KO + 2 i32)
  SelectCollect{V,C,KO,FIRST,ROUND}
                             v2: v1's, ROUND read by no gate
  SelectEntry{V,C,KO,FIRST,IDB,NP,STOP}
                             (x0: bf16, best: n bf16, idx: n i32, head | state, occ: i32) -> SlotEntry's entry
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from verity.circuits.defs import bind, composite
from verity.definitions.ops.prims import const
from verity.definitions.ops.scalar import (
    BitAnd,
    BitNot,
    I32Add,
    I32Eq,
    I32Le,
    SelectI32,
)
from verity.circuits.refs import Coll, broadcast, concat, tuple_of
from verity.circuits.types import Array, Tuple, Value
from verity.definitions.registry import dsa as D
from verity.definitions.registry.dead_free import TokenSelectCombineV2
from verity.definitions.registry.prims import BIT, I32
from verity.definitions.registry.sampling import bit_at

MAX_TOKENS_WIDTH = 6
#: schema v1's ``temperature`` and ``top_p``: a digit, ``.`` and six digits
RATIO_WIDTH = 8
#: schema v1's ``seed``: nineteen characters, whose low nibbles at their decimal places sum below 2^64
SEED_WIDTH = 19
NULL = b"null"
#: one SHA-512 block in bytes (1,024 bits): the body is padded with spaces to a whole number of them
BLOCK = 128

#: the element types of a signature: these words, and their bits in `boolean_io_shell`
WORDS = SimpleNamespace(U8=Value(8), BIT=BIT, I32=I32)

WORD = {"iadd": I32Add, "ile": I32Le, "isel": SelectI32, "band": BitAnd, "bnot": BitNot}


def id_width(V: int) -> int:
    return len(str(V - 1))


def refuse(S: dict) -> str | None:
    """A binding the schema has no layout for: a model name that isn't printable ASCII without a quote or backslash (it is a JSON
    string's bytes as they are), no prompt, or a vocabulary of fewer than 2 ids or over 2^31.  Only the statics bound so far."""
    m, lp, v = S.get("MODEL", "m"), S.get("LP", 1), S.get("V", 2)
    if not (isinstance(m, str) and _name(m)):
        return f"MODEL must be printable ASCII without '\"' or '\\\\' (got {m!r})"
    if not (type(lp) is int and lp >= 1):
        return f"LP must be at least 1 (got {lp!r})"
    if not (type(v) is int and 2 <= v <= 1 << 31):
        return f"V must be in [2, 2^31] (got {v!r})"
    return None


def sampled_refuse(S: dict) -> str | None:
    """`refuse`, and an id of no bytes."""
    rid = S.get("RID", 1)
    return refuse(S) or (None if type(rid) is int and rid >= 1 else f"RID must be at least 1 (got {rid!r})")


def _name(s: str) -> bool:
    return bool(s) and all(0x20 <= ord(c) < 0x7F and c not in '"\\' for c in s)


def length(MODEL: str, LP: int, V: int) -> int:
    """The body's length in bytes, its padding included: `layout`'s, without building it."""
    fixed = len(f'{{"model":"{MODEL}","prompt":[],"max_tokens":,"temperature":0,"stream":true}}') + MAX_TOKENS_WIDTH
    return -(-(fixed + max(LP * (id_width(V) + 1) - 1, 0)) // BLOCK) * BLOCK


def layout(MODEL: str, LP: int, V: int, RID: int | None = None) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    """(literal bytes as (offset, byte), fields as (offset, width): the LP ids then max_tokens, and under schema v1 (an id of
    ``RID`` bytes) temperature, top_p, seed and user; the body's length)."""
    lits: list[tuple[int, int]] = []
    fields: list[tuple[int, int]] = []
    n = 0

    def lit(s: str) -> None:
        nonlocal n
        lits.extend((n + i, b) for i, b in enumerate(s.encode()))
        n += len(s)

    def field(w: int) -> None:
        nonlocal n
        fields.append((n, w))
        n += w

    lit(f'{{"model":"{MODEL}","prompt":[')
    for i in range(LP):
        if i:
            lit(",")
        field(id_width(V))
    lit('],"max_tokens":')
    field(MAX_TOKENS_WIDTH)
    if RID is not None:
        for key, w in (("temperature", RATIO_WIDTH), ("top_p", RATIO_WIDTH), ("seed", SEED_WIDTH)):
            lit(f',"{key}":')
            field(w)
        lit(',"user":"')
        field(RID)
        lit('","stream":true}')
    else:
        lit(',"temperature":0,"stream":true}')
    lit(" " * (-n % BLOCK))
    return lits, fields, n


def _encode(schema: str, MODEL: str, ids, max_tokens: int, V: int, tail: str) -> bytes:
    if why := refuse({"MODEL": MODEL, "LP": len(ids), "V": V}):
        raise ValueError(f"{schema}: {why}")
    if not all(0 <= int(t) < V for t in ids) or not 1 <= max_tokens < 10 ** MAX_TOKENS_WIDTH:
        raise ValueError(f"{schema}: ids must be in [0, {V}) and max_tokens in [1, 10^{MAX_TOKENS_WIDTH})")
    w = id_width(V)
    body = (f'{{"model":"{MODEL}","prompt":[' + ",".join(f"{int(t):>{w}}" for t in ids)
            + f'],"max_tokens":{max_tokens:>{MAX_TOKENS_WIDTH}},{tail},"stream":true}}').encode()
    return body + b" " * (-len(body) % BLOCK)


def encode(MODEL: str, ids, max_tokens: int, V: int) -> bytes:
    """A request's ingress v0 bytes."""
    return _encode("ingress v0", MODEL, ids, max_tokens, V, '"temperature":0')


def _ratio(x: float, what: str) -> str:
    s = "%8.6f" % x if isinstance(x, (int, float)) and not isinstance(x, bool) else ""
    if not (s and 0 <= x < 10 and float(s) == x):
        raise ValueError(f"ingress v1: {what} is a number in [0, 10) of at most six decimals, not {x!r}")
    return s


def sampled_encode(MODEL: str, ids, max_tokens: int, V: int, *, temperature: float, top_p: float, seed: int | None,
                   user: str) -> bytes:
    """A request's ingress v1 bytes; a ``seed`` of None is ``null``."""
    t, p = _ratio(temperature, "temperature"), _ratio(top_p, "top_p")
    if not 0 < top_p <= 1:
        raise ValueError(f"ingress v1: top_p is in (0, 1], not {top_p!r}")
    if seed is not None and not (type(seed) is int and 0 <= seed < 1 << 63):
        raise ValueError(f"ingress v1: a seed is None or an int in [0, 2^63), not {seed!r}")
    if not (isinstance(user, str) and _name(user)):
        raise ValueError(f"ingress v1: a user is printable ASCII without '\"' or '\\\\', not {user!r}")
    s = (NULL.decode() if seed is None else str(seed)).rjust(SEED_WIDTH)
    return _encode("ingress v1", MODEL, ids, max_tokens, V, f'"temperature":{t},"top_p":{p},"seed":{s},"user":"{user}"')


def _int_field(s: bytes) -> tuple[bool, int]:
    """A field's checks (`_field`'s) and its low nibbles' decimal sum."""
    w = len(s)
    sp = [c == 0x20 for c in s]
    dig = [0x30 <= c <= 0x39 for c in s]
    ok = all(a or b for a, b in zip(sp, dig)) and not sp[-1] and all(sp[k - 1] or not sp[k] for k in range(1, w))
    ok &= not any(s[k] == 0x30 and (k == 0 or sp[k - 1]) for k in range(w - 1))
    return ok, sum((c & 15) * 10 ** (w - 1 - k) for k, c in enumerate(s))


def _fields(body: bytes, schema: str, MODEL: str, LP: int, V: int, RID: int | None) -> tuple[bool, list[int], list]:
    """(the literal bytes', the ids' and max_tokens' checks, their values, the fields after them)."""
    lits, fields, n = layout(MODEL, LP, V, RID)
    if len(body) != n:
        raise ValueError(f"{schema} at LP={LP}: the body is {n} bytes, not {len(body)}")
    ok = all(body[i] == c for i, c in lits)
    vals = []
    for off, w in fields[:LP + 1]:
        f_ok, v = _int_field(body[off:off + w])
        ok &= f_ok
        vals.append(v)
    ok &= all(v <= V - 1 for v in vals[:-1]) and vals[-1] >= 1
    return ok, vals, [body[off:off + w] for off, w in fields[LP + 1:]]


def reference(body: bytes, MODEL: str, LP: int, V: int) -> tuple[int, list[int], int]:
    """`IngressPrompt_v1`'s function on a body of the layout's length: (ok, ids, max_tokens), each field's value its low nibbles'
    decimal sum whether or not the body is in the schema."""
    ok, vals, _ = _fields(body, "ingress v0", MODEL, LP, V, None)
    return int(ok), vals[:-1], vals[-1]


def sampled_reference(body: bytes, MODEL: str, LP: int, V: int, RID: int) -> tuple:
    """`IngressSampled_v1`'s function on a body of schema v1's length: (ok, ids, max_tokens, temperature, top_p, seeded, seed,
    user), each ratio its seven digits' low nibbles at their places (millionths), the seed its field's sum and the user its
    bytes, whether or not the body is in the schema."""
    ok, vals, (t, p, s, u) = _fields(body, "ingress v1", MODEL, LP, V, RID)
    ok &= all(0x20 <= c < 0x7F and c not in b'"\\' for c in u)
    ratios = []
    for r in (t, p):
        ok &= r[1] == 0x2E and all(0x30 <= c <= 0x39 for k, c in enumerate(r) if k != 1)
        ratios.append(sum((c & 15) * 10 ** (6 - k) for k, c in enumerate(r[:1] + r[2:])))
    ok &= 1 <= ratios[1] <= 10 ** 6
    s_ok, seed = _int_field(s)
    null = s == NULL.rjust(SEED_WIDTH)
    ok &= null or (s_ok and seed < 1 << 63)
    return int(ok), vals[:-1], vals[-1], ratios[0], ratios[1], int(not null), seed, list(u)


class Ops(D.Ops):
    """`dsa.Ops` over `WORD` with ``bit(x, i)``, bit i of a byte (`BitAtx8_v1`)."""

    def bit(self, x, i: int):
        return self.B.call(bit_at(8), x, self.word(i))


def word_ops(B) -> Ops:
    return Ops(B, WORD, lambda bits: B.call(const(32, bits)), None)


def ingress_sig(T):
    return lambda S: ((("body", Array(length(S.MODEL, S.LP, S.V), T.U8)),), Tuple(T.BIT, Array(S.LP, T.I32), T.I32))


def _all(o, xs: list):
    acc = xs[0]
    for x in xs[1:]:
        acc = o.band(acc, x)
    return acc


def _is(o, bits: list, c: int):
    return _all(o, [b if c >> i & 1 else o.bnot(b) for i, b in enumerate(bits)])


def _field(o, chars: list) -> tuple[list, object]:
    """A field's checks (`field_checks`) and its value, the low nibbles at their decimal places."""
    oks, bits = field_checks(o, chars)
    w = len(chars)
    v = None
    for k, b in enumerate(bits):
        for j in range(4):
            t = o.isel(b[j], o.word((1 << j) * 10 ** (w - 1 - k)), o.word(0))
            v = t if v is None else o.iadd(v, t)
    return oks, v


def field_checks(o, chars: list) -> tuple[list, list]:
    """A field's checks (each character a space or a digit, the spaces leading, the last a digit, no leading zero) and its
    characters' bits."""
    w = len(chars)
    bits = [[o.bit(c, i) for i in range(8)] for c in chars]
    sp, dig, zero = [], [], []
    for b in bits:
        nb = [o.bnot(x) for x in b]
        hi = _all(o, [nb[7], nb[6], b[5], b[4]])
        sp.append(_all(o, [nb[7], nb[6], b[5], nb[4], nb[3], nb[2], nb[1], nb[0]]))
        dig.append(o.band(hi, o.bnot(o.band(b[3], o.bnot(o.band(nb[2], nb[1]))))))
        zero.append(_all(o, [hi, nb[3], nb[2], nb[1], nb[0]]))
    oks = [o.bnot(o.band(o.bnot(sp[k]), o.bnot(dig[k]))) for k in range(w)] + [o.bnot(sp[-1])]
    oks += [o.bnot(o.band(sp[k], o.bnot(sp[k - 1]))) for k in range(1, w)]
    oks += [o.bnot(zero[k] if k == 0 else o.band(zero[k], sp[k - 1])) for k in range(w - 1)]
    return oks, bits


def parse_head(o, S, body: list, RID: int | None = None) -> tuple[list, list, Any, list[tuple[int, int]]]:
    """The literal bytes', the ids' and max_tokens' checks, the ids, max_tokens, and the fields after it (schema v1's)."""
    lits, fields, _ = layout(S.MODEL, S.LP, S.V, RID)
    oks = [_is(o, [o.bit(body[i], j) for j in range(8)], c) for i, c in lits]
    vals = []
    for off, w in fields[:S.LP + 1]:
        f_oks, v = _field(o, body[off:off + w])
        oks += f_oks
        vals.append(v)
    oks += [o.ile(v, o.word(S.V - 1)) for v in vals[:-1]] + [o.ile(o.word(1), vals[-1])]
    return oks, vals[:-1], vals[-1], fields[S.LP + 1:]


def parse(o, S, body: list) -> tuple:
    """`IngressPrompt_v1`'s body: (ok, ids, max_tokens) of the module docstring."""
    oks, ids, m, _ = parse_head(o, S, body)
    return _all(o, oks), ids, m


@composite("IngressPrompt", 1, ["MODEL", "LP", "V"], ingress_sig(WORDS), conformance="exact-model-tested", refuse=refuse)
def IngressPrompt(B, S, body):
    """Ingress v0's bytes (the module docstring's schema at MODEL, LP ids, vocabulary V) to (ok, the prompt's ids, max_tokens)."""
    params, ret = ingress_sig(WORDS)(S)
    ok, ids, m = parse(word_ops(B), S, [body[i] for i in range(params[0][1].n)])
    return Coll(ret, concat([ok.refs, *[v.refs for v in ids], m.refs]))


def feed_refuse(S: dict) -> str | None:
    n, first = S.get("NP", 1), S.get("FIRST", False)
    if not (type(n) is int and n >= 1 and type(first) is bool):
        return f"NP must be at least 1 and FIRST a bool (got NP={n!r}, FIRST={first!r})"
    return None


def feed_sig(S):
    slot = ("slot", Array(S.NP + 1, WORDS.I32))
    return ((slot,) if S.FIRST else (slot, ("sel", WORDS.I32))), Tuple(WORDS.I32, Array(S.NP + 1, WORDS.I32))


def feed_reference(slot: list[int], sel: int | None, NP: int) -> tuple[int, list[int]]:
    """`TokenFeed_v1`'s function on i32 encodings (``sel`` None: FIRST): (its token, the next position's slot)."""
    left, ids = (slot[0] - 1) & 0xFFFFFFFF, slot[1:]
    return (ids[0] if sel is None or not left >> 31 else sel), [left, *ids[1:], 0]


@composite("TokenFeed", 1, ["NP", "FIRST"], feed_sig, refuse=feed_refuse,
           conformance="not-yet-established: vLLM's input id at a position, the prompt's below its length, else the previous "
                       "position's sample")
def TokenFeed(B, S, slot, sel=None):
    """A class Program's token at a position and the slot it carries to the next (the module docstring)."""
    o = word_ops(B)
    left, ids = o.iadd(slot[0], o.word(0xFFFFFFFF)), [slot[1 + i] for i in range(S.NP)]
    tok = ids[0] if S.FIRST else o.isel(B.call(bit_at(32), left, o.word(31)), sel, ids[0])
    _, ret = feed_sig(S)
    return Coll(ret, concat([tok.refs, left.refs, *[x.refs for x in ids[1:]], o.word(0).refs]))


M32 = 0xFFFFFFFF
#: a 32-bit word as its bits, least significant first (the Boolean form of an i32)
BITS32 = Array(32, BIT)


def position(K0: int, K: int, p: int) -> int:
    """Prompt position p's index among a scan slot's ids: a request's first cell's K0 fields first, then each later cell's K."""
    return p if p < K0 else K + p - K0


def _nat(v) -> bool:
    return type(v) is int and v >= 0


def slots_refuse(S: dict) -> str | None:
    """A binding whose fields aren't bits of its row: NS, NP, IDB (at most 32), K0 and K (K0 ≤ K) positive, and OCC, LP, MAX
    and IDS one bit offset a lane each, every field inside the row's 16·NS bits."""
    have = {k: S[k] for k in ("NS", "NP", "IDB", "K0", "K") if k in S}
    if not all(_nat(v) and v >= 1 for v in have.values()) or have.get("IDB", 1) > 32 or have.get("K0", 1) > have.get("K", 1 << 31):
        return f"NS, NP, IDB <= 32 and K0 <= K are positive integers (got {have})"
    lanes = {k: S[k] for k in ("OCC", "LP", "MAX", "IDS") if k in S}
    if not all(isinstance(v, tuple) and v and all(map(_nat, v)) for v in lanes.values()) or len({len(v) for v in lanes.values()}) > 1:
        return f"OCC, LP, MAX and IDS are one bit offset a lane each, as many lanes each (got {lanes})"
    if len(have) == 5 and len(lanes) == 4:
        bits, ids = 16 * S["NS"], S["IDB"] * (position(S["K0"], S["K"], S["NP"] - 1) + 1)
        if max(max(S[k]) + 32 for k in ("OCC", "LP", "MAX")) > bits or max(S["IDS"]) + ids > bits:
            return f"a lane's fields past the row's {bits} bits"
    return None


def slots_sig(S):
    lane = (BITS32, Array(S.NP + 1, BITS32), BITS32, Array(2, BITS32))
    return (("state", Array(S.NS, Array(16, BIT))),), Tuple(*[t for _ in S.OCC for t in lane])


def slots_reference(state: list[int], NP: int, IDB: int, K0: int, K: int, OCC, LP, MAX, IDS) -> list[tuple[list[int], list[int]]]:
    """Each lane's fields of the scan state row's u16 words as i32 encodings: its row (occupancy, prompt length, ids[0], …,
    ids[NP − 1]) and its head (prompt length, ``max_tokens``)."""
    x = sum(w << 16 * i for i, w in enumerate(state))

    def at(lo: int, w: int) -> int:
        return x >> lo & (1 << w) - 1

    return [([at(o, 32), at(lp, 32), *[at(ids + IDB * position(K0, K, p), IDB) for p in range(NP)]], [at(lp, 32), at(m, 32)])
            for o, lp, m, ids in zip(OCC, LP, MAX, IDS)]


def scan_reference(state: list[int], NP: int, IDB: int, K0: int, K: int, OCC, LP, MAX,
                   IDS) -> list[tuple[int, list[int], int, list[int]]]:
    """`boolean_io_shell.ScanSlots_v2`'s function on the row's u16 words: each lane's (token, slot, occupancy word, head),
    its first position's `TokenFeed_v1` on its row (`slots_reference`), as i32 encodings."""
    out = []
    for row, head in slots_reference(state, NP, IDB, K0, K, OCC, LP, MAX, IDS):
        tok, slot = feed_reference(row[1:], None, NP)
        out.append((tok, slot, row[0], head))
    return out


def collect_refuse(S: dict) -> str | None:
    ko, first = S.get("KO", 1), S.get("FIRST", False)
    if not (type(ko) is int and 1 <= ko < 1 << 16 and type(first) is bool):
        return f"KO must be in [1, 2^16) and FIRST a bool (got KO={ko!r}, FIRST={first!r})"
    return None


def collect_sig(S):
    state = Array(S.KO + 2, WORDS.I32)
    return (("head", Array(2, WORDS.I32)) if S.FIRST else ("state", state), ("tok", WORDS.I32)), state


def collect_reference(a: list[int], tok: int, KO: int, FIRST: bool) -> list[int]:
    """`OutCollect_v1`'s function on i32 encodings: ``a`` the head (lp, max_tokens) where FIRST, else the state."""
    w, m, ids = ((a[0] - 1) & M32, a[1], [0] * KO) if FIRST else (a[0], a[1], list(a[2:]))
    sw = w - (1 << 32) if w >> 31 else w
    return [(w - 1) & M32, m, *(ids[1:] + [tok] if 1 - KO <= sw <= 0 else ids)]


@composite("OutCollect", 1, ["KO", "FIRST"], collect_sig, refuse=collect_refuse,
           conformance="not-yet-established: vLLM's output ids of a request, its samples from its prompt's last position on, KO "
                       "at most")
def OutCollect(B, S, a, tok):
    """A slot's outputs carried to the next position, this position's shifted in while it is one of the KO from the prompt's
    last (the module docstring)."""
    o = word_ops(B)
    if S.FIRST:
        w, m, ids = o.iadd(a[0], o.word(M32)), a[1], None
    else:
        w, m, ids = a[0], a[1], [a[2 + i] for i in range(S.KO)]
    take = o.band(o.ile(w, o.word(0)), o.ile(o.word((1 - S.KO) & M32), w))
    if ids is None:
        out = [o.word(0)] * (S.KO - 1) + [o.isel(take, tok, o.word(0))]
    else:
        out = [o.isel(take, x, y) for x, y in zip(ids[1:] + [tok], ids)]
    _, ret = collect_sig(S)
    return Coll(ret, concat([o.iadd(w, o.word(M32)).refs, m.refs, *[x.refs for x in out]]))


def entry_words(KO: int, IDB: int) -> int:
    """An entry's u16 words: ``read``'s bit, n's 32 bits and KO ids of IDB bits, zero-padded."""
    return -(-(33 + KO * IDB) // 16)


def entry_refuse(S: dict) -> str | None:
    ko, idb, np_, stop = S.get("KO", 1), S.get("IDB", 1), S.get("NP", 1), S.get("STOP", ())
    if not (type(ko) is int and 1 <= ko < 1 << 16 and type(idb) is int and 1 <= idb <= 32 and type(np_) is int
            and 1 <= np_ < 1 << 16 and isinstance(stop, tuple) and all(type(s) is int and 0 <= s <= M32 for s in stop)
            and len(set(stop)) == len(stop)):
        return (f"KO and NP in [1, 2^16), IDB in [1, 32] and STOP distinct u32 ids (got KO={ko!r}, IDB={idb!r}, NP={np_!r}, "
                f"STOP={stop!r})")
    return None


def entry_sig(S):
    return (("occ", WORDS.I32), ("state", Array(S.KO + 2, WORDS.I32))), Array(entry_words(S.KO, S.IDB), Array(16, BIT))


def entry_reference(occ: int, state: list[int], KO: int, IDB: int, NP: int, STOP: tuple[int, ...]) -> list[int]:
    """`SlotEntry_v1`'s function: the entry's u16 words from the occupancy word and the last position's state."""
    w, m, ids = state[0] - (1 << 32) if state[0] >> 31 else state[0], state[1], state[2:]
    fits = 1 - NP - KO <= w <= -KO
    alive, n, keep = bool(occ & 1) and fits, 0, []
    for i, x in enumerate(ids):
        k = alive and m >= i + 1
        keep.append(k)
        n = i + 1 if k else n
        alive = k and x not in STOP
    if occ & 1 and not fits or alive and m >= KO + 1:
        n, keep = KO + 1, [False] * KO
    bits = occ & 1 | n << 1 | sum((x & (1 << IDB) - 1) << 33 + IDB * i for i, (x, k) in enumerate(zip(ids, keep)) if k)
    return [bits >> 16 * k & 0xFFFF for k in range(entry_words(KO, IDB))]


@composite("SlotEntry", 1, ["KO", "IDB", "NP", "STOP"], entry_sig, refuse=entry_refuse,
           conformance="not-yet-established: a request's output count and ids as vLLM served them, up to max_tokens and its "
                       "first stop id, KO + 1 where the class can't hold it")
def SlotEntry(B, S, occ, state):
    """A slot's entry of the network certifier's Programs' row (the module docstring): ``read`` the slot's occupancy
    bit, then n and the kept outputs; output i is kept while the slot is occupied, its prompt 1 to NP ids, i <
    ``max_tokens`` (unsigned) and no earlier output is a stop id; n the kept outputs, or KO + 1 and none kept where the
    prompt doesn't fit or the outputs don't end within KO."""
    o, (_, ret) = word_ops(B), entry_sig(S)

    def either(a, b):
        return o.bnot(o.band(o.bnot(a), o.bnot(b)))

    w, m, ids = state[0], state[1], [state[2 + i] for i in range(S.KO)]
    occupied, big = B.call(bit_at(32), occ, o.word(0)), B.call(bit_at(32), m, o.word(31))
    fits = o.band(o.ile(o.word((1 - S.NP - S.KO) & M32), w), o.ile(w, o.word(-S.KO & M32)))
    alive, n, keep = o.band(occupied, fits), o.word(0), []
    for i, x in enumerate(ids):
        k = o.band(alive, either(big, o.ile(o.word(i + 1), m)))
        keep.append(k)
        n = o.isel(k, o.word(i + 1), n)
        stops = [B.call(I32Eq, x, o.word(s)) for s in S.STOP]
        if stops:
            hit = stops[0]
            for s in stops[1:]:
                hit = either(hit, s)
            alive = o.band(k, o.bnot(hit))
        else:
            alive = k
    over = either(o.band(occupied, o.bnot(fits)), o.band(alive, either(big, o.ile(o.word(S.KO + 1), m))))
    n, keep = o.isel(over, o.word(S.KO + 1), n), [o.band(k, o.bnot(over)) for k in keep]
    bits = [occupied, *[B.call(bit_at(32), n, o.word(b)) for b in range(32)]]
    bits += [o.band(k, B.call(bit_at(32), x, o.word(b))) for x, k in zip(ids, keep) for b in range(S.IDB)]
    pad = 16 * entry_words(S.KO, S.IDB) - len(bits)
    return Coll(ret, concat([b.refs for b in bits] + ([broadcast(*B.call(const(1, 0)).refs[0], pad)] if pad else [])))


def _select(S) -> Any:
    return bind(TokenSelectCombineV2, V=S.V, C=S.C)


def select_refuse(S: dict) -> str | None:
    """A binding whose select isn't `TokenSelectCombine_v2{V, C}`'s (V at least 2, C 1 to V), or whose collector or entry
    its own refusal refuses."""
    v, c = S.get("V", 2), S.get("C", 1)
    if not (type(v) is int and type(c) is int and v >= 2 and 1 <= c <= v):
        return f"V must be at least 2 and C in [1, V] (got V={v!r}, C={c!r})"
    return collect_refuse(S) or (entry_refuse(S) if "IDB" in S else None)


def select_collect_sig(S):
    (a, _), state = collect_sig(S)
    return (*_select(S).signature()[0], a), Tuple(WORDS.I32, state)


def select_entry_sig(S):
    (a, _), _ = collect_sig(S)
    (occ, _), entry = entry_sig(S)
    return (*_select(S).signature()[0], a, occ), entry


@composite("SelectCollect", 1, ["V", "C", "KO", "FIRST"], select_collect_sig, refuse=select_refuse,
           conformance="not-yet-established: vLLM's sample at a position and its output ids so far, TokenSelectCombine_v2, then "
                       "OutCollect_v1")
def SelectCollect(B, S, x0, best, idx, a):
    """A position's select over its groups' bests and indices and ``logits[0]`` (`TokenSelectCombine_v2{V, C}`), and the
    slot's outputs carried on with it (`OutCollect_v1{KO, FIRST}`): (the select, the state)."""
    tok = B.call(_select(S), x0, best, idx)
    return tuple_of(tok, B.call(bind(OutCollect, KO=S.KO, FIRST=S.FIRST), a, tok))


def round_refuse(S: dict) -> str | None:
    """`SelectCollect_v2`'s refusal: v1's, or a ROUND that isn't a natural number."""
    r = S.get("ROUND", 0)
    return select_refuse(S) or (None if type(r) is int and r >= 0 else f"ROUND={r!r}: a round is a natural number")


@composite("SelectCollect", 2, ["V", "C", "KO", "FIRST", "ROUND"], select_collect_sig, refuse=round_refuse,
           conformance="not-yet-established: vLLM's sample at a position and its output ids so far: SelectCollect_v1, "
                       "at a round")
def SelectCollectV2(B, S, x0, best, idx, a):
    """`SelectCollect_v1{V, C, KO, FIRST}`'s body; ``ROUND``, the round of the input-ids table its select is held to, is read
    by no gate."""
    tok = B.call(_select(S), x0, best, idx)
    return tuple_of(tok, B.call(bind(OutCollect, KO=S.KO, FIRST=S.FIRST), a, tok))


@composite("SelectEntry", 1, ["V", "C", "KO", "FIRST", "IDB", "NP", "STOP"], select_entry_sig, refuse=select_refuse,
           conformance="not-yet-established: vLLM's last sample, its request's outputs and their count, "
                       "TokenSelectCombine_v2, OutCollect_v1, then SlotEntry_v1")
def SelectEntry(B, S, x0, best, idx, a, occ):
    """The last position's select (`TokenSelectCombine_v2{V, C}`), collected (`OutCollect_v1{KO, FIRST}`), and the slot's
    entry of that state and its occupancy word (`SlotEntry_v1{KO, IDB, NP, STOP}`): the entry."""
    tok = B.call(_select(S), x0, best, idx)
    state = B.call(bind(OutCollect, KO=S.KO, FIRST=S.FIRST), a, tok)
    return B.call(bind(SlotEntry, KO=S.KO, IDB=S.IDB, NP=S.NP, STOP=S.STOP), occ, state)


IO_SHELL_COMPOSITES = [IngressPrompt, TokenFeed, OutCollect, SlotEntry, SelectCollect, SelectEntry, SelectCollectV2]
