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
only (``"temperature":0``); the seeded top-p schema comes next.

`IngressPrompt_v1{MODEL, LP, V}` (body: Array<N, Value<8>>) -> (ok: bit, ids: Array<LP, i32>, max_tokens: i32) checks every byte
against the layout and decodes the fields.  ``ok`` is 1 exactly when the body is in the schema: each literal byte is the layout's
(the padding's included: exactly those spaces, no other whitespace and no other length), each field is spaces then decimal digits
with at least one digit and no leading zero, each id is below V and ``max_tokens`` is at least 1.  A field's value is the sum of its characters' low nibbles at their decimal places (a space's low nibble is 0, and a
digit's is its value), so on a body that isn't in the schema the ids are still that sum: `reference` is the function exactly.
The body is written once over its steps (`parse`); `boolean_io_shell.IngressPrompt_v2` is it on gates.

Registered functions:
  IngressPrompt{MODEL,LP,V}  (body: N u8) -> (ok: bit, LP i32, max_tokens: i32)
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.ml.circuits.defs import composite
from verity.ml.circuits.ops.prims import const
from verity.ml.circuits.ops.scalar import BitAnd, BitNot, I32Add, I32Le, SelectI32
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.definitions.registry import dsa as D
from verity.ml.definitions.registry.prims import BIT, I32
from verity.ml.definitions.registry.sampling import bit_at

MAX_TOKENS_WIDTH = 6
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
    if not (isinstance(m, str) and m and all(0x20 <= ord(c) < 0x7F and c not in '"\\' for c in m)):
        return f"MODEL must be printable ASCII without '\"' or '\\\\' (got {m!r})"
    if not (type(lp) is int and lp >= 1):
        return f"LP must be at least 1 (got {lp!r})"
    if not (type(v) is int and 2 <= v <= 1 << 31):
        return f"V must be in [2, 2^31] (got {v!r})"
    return None


def length(MODEL: str, LP: int, V: int) -> int:
    """The body's length in bytes, its padding included: `layout`'s, without building it."""
    fixed = len(f'{{"model":"{MODEL}","prompt":[],"max_tokens":,"temperature":0,"stream":true}}') + MAX_TOKENS_WIDTH
    return -(-(fixed + max(LP * (id_width(V) + 1) - 1, 0)) // BLOCK) * BLOCK


def layout(MODEL: str, LP: int, V: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    """(literal bytes as (offset, byte), fields as (offset, width): the LP ids then max_tokens, the body's length)."""
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
    lit(',"temperature":0,"stream":true}')
    lit(" " * (-n % BLOCK))
    return lits, fields, n


def encode(MODEL: str, ids, max_tokens: int, V: int) -> bytes:
    """A request's ingress v0 bytes."""
    if why := refuse({"MODEL": MODEL, "LP": len(ids), "V": V}):
        raise ValueError(f"ingress v0: {why}")
    if not all(0 <= int(t) < V for t in ids) or not 1 <= max_tokens < 10 ** MAX_TOKENS_WIDTH:
        raise ValueError(f"ingress v0: ids must be in [0, {V}) and max_tokens in [1, 10^{MAX_TOKENS_WIDTH})")
    w = id_width(V)
    body = (f'{{"model":"{MODEL}","prompt":[' + ",".join(f"{int(t):>{w}}" for t in ids)
            + f'],"max_tokens":{max_tokens:>{MAX_TOKENS_WIDTH}},"temperature":0,"stream":true}}').encode()
    return body + b" " * (-len(body) % BLOCK)


def reference(body: bytes, MODEL: str, LP: int, V: int) -> tuple[int, list[int], int]:
    """`IngressPrompt_v1`'s function on a body of the layout's length: (ok, ids, max_tokens), each field's value its low nibbles'
    decimal sum whether or not the body is in the schema."""
    lits, fields, n = layout(MODEL, LP, V)
    if len(body) != n:
        raise ValueError(f"ingress v0 at LP={LP}: the body is {n} bytes, not {len(body)}")
    ok = all(body[i] == c for i, c in lits)
    vals = []
    for off, w in fields:
        s = body[off:off + w]
        sp = [c == 0x20 for c in s]
        dig = [0x30 <= c <= 0x39 for c in s]
        ok &= all(a or b for a, b in zip(sp, dig)) and not sp[-1] and all(sp[k - 1] or not sp[k] for k in range(1, w))
        ok &= not any(s[k] == 0x30 and (k == 0 or sp[k - 1]) for k in range(w - 1))
        vals.append(sum((c & 15) * 10 ** (w - 1 - k) for k, c in enumerate(s)))
    ok &= all(v <= V - 1 for v in vals[:-1]) and vals[-1] >= 1
    return int(ok), vals[:-1], vals[-1]


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
    """A field's checks (each character a space or a digit, the spaces leading, the last a digit, no leading zero) and its value,
    the low nibbles at their decimal places."""
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
    v = None
    for k, b in enumerate(bits):
        for j in range(4):
            t = o.isel(b[j], o.word((1 << j) * 10 ** (w - 1 - k)), o.word(0))
            v = t if v is None else o.iadd(v, t)
    return oks, v


def parse(o, S, body: list) -> tuple:
    """`IngressPrompt_v1`'s body: (ok, ids, max_tokens) of the module docstring."""
    lits, fields, _ = layout(S.MODEL, S.LP, S.V)
    oks = [_is(o, [o.bit(body[i], j) for j in range(8)], c) for i, c in lits]
    vals = []
    for off, w in fields:
        f_oks, v = _field(o, body[off:off + w])
        oks += f_oks
        vals.append(v)
    oks += [o.ile(v, o.word(S.V - 1)) for v in vals[:-1]] + [o.ile(o.word(1), vals[-1])]
    return _all(o, oks), vals[:-1], vals[-1]


@composite("IngressPrompt", 1, ["MODEL", "LP", "V"], ingress_sig(WORDS), conformance="exact-model-tested", refuse=refuse)
def IngressPrompt(B, S, body):
    """Ingress v0's bytes (the module docstring's schema at MODEL, LP ids, vocabulary V) to (ok, the prompt's ids, max_tokens)."""
    params, ret = ingress_sig(WORDS)(S)
    ok, ids, m = parse(word_ops(B), S, [body[i] for i in range(params[0][1].n)])
    return Coll(ret, concat([ok.refs, *[v.refs for v in ids], m.refs]))


IO_SHELL_COMPOSITES = [IngressPrompt]
