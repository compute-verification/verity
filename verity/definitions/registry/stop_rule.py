"""Where a request's output stops and why, from the ids it emitted: the END position and the finish reason as a block's values,
computed in-circuit, so neither is advice.

`StopRule_v1{T, NSTOP}` (ids: Array<T, i32>, max_tokens: i32, stops: Array<NSTOP, i32>, prompt_len: i32,
max_model_len: i32) -> (n: i32, finish: i32).  ``ids`` are the request's first T emitted ids, ``stops`` the registered stop
ids (the EOS ids and any stop token ids; a repeated id changes nothing, so a smaller set fills NSTOP by repeating one).  The
output stops at the first length L, 1 to T, at which

- ``ids[L - 1]`` is a stop id: finish 1 (stop);
- else L reaches ``max_tokens``: finish 2 (length);
- else ``prompt_len + L`` reaches ``max_model_len``: finish 3 (the model's length, which vLLM also reports as length);

and n = L, the cell the egress record's END goes in (n DATA cells, then END; the finish reason isn't sent).  When no length up
to T stops, n = T and finish is 0: the request runs on.  ``max_tokens`` below 1 never stops.  The order is vLLM d9105ea8's
`check_stop`: the last id against the stop ids first, then the two length caps.

The comparisons are signed i32 and ``max_model_len - prompt_len`` wraps, so the function is total on every word (`reference`
is it exactly).  On the served domain (``max_tokens``, ``prompt_len`` and ``max_model_len`` in [0, 2^31), any ids) it is the
network encodings' derived rule (`benchmarks/network_e2e/encodings/derived.py`, ``stop_rule``) over the prefixes: n is the
shortest prefix that rule ends when finish isn't 0 (1 its "eos", 2 and 3 its "length"), and the rule ends no prefix up to T
when finish is 0.

The body (`stop`) is written once over its steps; `boolean_stop_rule.StopRule_v2` is it on gates.  It finds the first stop id
with one select a position from the last position down (T + 1 when none), and the length caps without a compare a position:
the first length that reaches ``max_tokens`` is ``max_tokens``, and the first that reaches the model's length is
``max_model_len - prompt_len``, at least 1.  A position costs NSTOP equality tests and a select of small constants.

`StopRule16_v1{T, NSTOP}`, T at most 8,191, is the same rule on the same inputs with one output, a 16-bit word: n in bits 0
to 12 and the finish code in bits 13 to 15, so the word is n + finish * 8192 (`reference16`).  n is at most T, so 13 bits
hold it, and three bits hold the finish codes 0 to 3 (`FINISH`).  Its body (`stop16`) is `stop` and then the packing, which
the word layer writes without a shift or a low-half primitive: a word added to itself is the word doubled, so 13 doublings
of finish and the sum with n give the packed i32, 16 more move its low half to the high half, and `U32Hi16` returns that
half.  On gates a doubling is wires, so `boolean_stop_rule.StopRule16_v2` adds to `stop`'s gates only the sum's two carries.

Registered functions:
  StopRule{T,NSTOP}    (ids: T i32, max_tokens: i32, stops: NSTOP i32, prompt_len: i32, max_model_len: i32) -> (n: i32, finish: i32)
  StopRule16{T,NSTOP}  (ids: T i32, max_tokens: i32, stops: NSTOP i32, prompt_len: i32, max_model_len: i32) -> u16
"""
from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from verity.circuits.defs import composite
from verity.definitions.ops.prims import const
from verity.definitions.ops.scalar import U16, BitAnd, BitNot, BitOr, I32Add, I32Eq, I32Le, SelectI32, U32Hi16
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import dsa as D
from verity.definitions.registry.dsa import I32Sub
from verity.definitions.registry.prims import I32

M32 = 0xFFFFFFFF
#: the finish reasons, by `StopRule_v1`'s code
FINISH = {0: None, 1: "stop", 2: "length", 3: "model-length"}

#: `StopRule16_v1`'s word holds n in its low N_BITS bits and the finish code in the three above
N_BITS = 13

#: the element types of a signature: these words, and their bits in `boolean_stop_rule`
WORDS = SimpleNamespace(I32=I32, U16=U16)

WORD = {"isub": I32Sub, "ile": I32Le, "ieq": I32Eq, "isel": SelectI32, "band": BitAnd, "bnot": BitNot, "bor": BitOr,
        "iadd": I32Add, "hi16": U32Hi16}


def word_ops(B) -> D.Ops:
    return D.Ops(B, WORD, lambda bits: B.call(const(32, bits)), None)


def refuse(S: dict) -> str | None:
    t, k = S.get("T", 1), S.get("NSTOP", 1)
    if not (type(t) is int and 1 <= t < (1 << 31) - 1 and type(k) is int and 1 <= k < 1 << 16):
        return f"T must be in [1, 2^31 - 1) and NSTOP in [1, 2^16) (got T={t!r}, NSTOP={k!r})"
    return None


def refuse16(S: dict) -> str | None:
    """`refuse`, and T past what the word's n holds."""
    why, t = refuse(S), S.get("T", 1)
    if why is None and t >> N_BITS:
        why = (f"T must be at most {(1 << N_BITS) - 1}: n is at most T, and the 16-bit word holds n in {N_BITS} bits, under "
               f"the finish code's three (got T={t!r})")
    return why


def stop_sig(T):
    return lambda S: ((("ids", Array(S.T, T.I32)), ("max_tokens", T.I32), ("stops", Array(S.NSTOP, T.I32)),
                       ("prompt_len", T.I32), ("max_model_len", T.I32)), Tuple(T.I32, T.I32))


def stop16_sig(T):
    return lambda S: (stop_sig(T)(S)[0], T.U16)


def _signed(x: int) -> int:
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def reference(ids: Sequence[int], max_tokens: int, stops: Sequence[int], prompt_len: int, max_model_len: int) -> tuple[int, int]:
    """`StopRule_v1{T, NSTOP}`'s function on i32 encodings, T = len(ids): (n, finish)."""
    T, stop_set, mt = len(ids), {x & M32 for x in stops}, _signed(max_tokens)
    room = _signed(max_model_len - prompt_len)
    cap = mt if mt <= room else room
    cap = 1 if cap <= 0 else cap
    s = next((i + 1 for i, x in enumerate(ids) if x & M32 in stop_set), T + 1)
    n = s if s <= cap else cap
    if not (1 <= mt and n <= T):
        return T, 0
    return n, 1 if s <= cap else 2 if mt <= n else 3


def reference16(ids: Sequence[int], max_tokens: int, stops: Sequence[int], prompt_len: int, max_model_len: int) -> int:
    """`StopRule16_v1{T, NSTOP}`'s function on i32 encodings, T = len(ids): the word n + finish * 8192 of `reference`'s
    (n, finish)."""
    if len(ids) >> N_BITS:
        raise ValueError(f"StopRule16 holds n in {N_BITS} bits: at most {(1 << N_BITS) - 1} ids (got {len(ids)})")
    n, finish = reference(ids, max_tokens, stops, prompt_len, max_model_len)
    return n + (finish << N_BITS)


def stop(o, S, ids: list, max_tokens, stops: list, prompt_len, max_model_len) -> tuple:
    """`StopRule_v1`'s body: (n, finish) of the module docstring."""
    room = o.isub(max_model_len, prompt_len)
    cap = o.isel(o.ile(max_tokens, room), max_tokens, room)
    cap = o.isel(o.ile(cap, o.word(0)), o.word(1), cap)
    s = o.word(S.T + 1)
    for i in reversed(range(S.T)):
        hit = o.ieq(ids[i], stops[0])
        for x in stops[1:]:
            hit = o.bor(hit, o.ieq(ids[i], x))
        s = o.isel(hit, o.word(i + 1), s)
    stopped = o.ile(s, cap)
    n = o.isel(stopped, s, cap)
    done = o.band(o.ile(o.word(1), max_tokens), o.ile(n, o.word(S.T)))
    finish = o.isel(stopped, o.word(1), o.isel(o.ile(max_tokens, n), o.word(2), o.word(3)))
    return o.isel(done, n, o.word(S.T)), o.isel(done, finish, o.word(0))


def doubled(o, w, k: int):
    """``w`` * 2^k modulo 2^32: k times, the word added to itself (the word layer's shift left)."""
    for _ in range(k):
        w = o.iadd(w, w)
    return w


def stop16(o, S, ids: list, max_tokens, stops: list, prompt_len, max_model_len):
    """`StopRule16_v1`'s body: `stop`'s (n, finish) as the 16-bit word n + finish * 8192, the low half of that i32."""
    n, finish = stop(o, S, ids, max_tokens, stops, prompt_len, max_model_len)
    return o.hi16(doubled(o, o.iadd(n, doubled(o, finish, N_BITS)), 16))


@composite("StopRule", 1, ["T", "NSTOP"], stop_sig(WORDS), refuse=refuse,
           conformance="model-untested: read from vLLM d9105ea8's check_stop (the last id against the stop ids, then "
                       "max_tokens and max_model_len) and held to the network encodings' derived stop rule; no capture of "
                       "vLLM's finish yet")
def StopRule(B, S, ids, max_tokens, stops, prompt_len, max_model_len):
    """A request's END position and finish reason from its first T emitted ids (the module docstring)."""
    _, ret = stop_sig(WORDS)(S)
    n, finish = stop(word_ops(B), S, [ids[i] for i in range(S.T)], max_tokens, [stops[j] for j in range(S.NSTOP)],
                     prompt_len, max_model_len)
    return Coll(ret, concat([n.refs, finish.refs]))


@composite("StopRule16", 1, ["T", "NSTOP"], stop16_sig(WORDS), refuse=refuse16,
           conformance="model-untested: StopRule_v1's (n, finish), read from vLLM d9105ea8's check_stop, as the one word "
                       "n + finish * 8192; no capture of vLLM's finish yet")
def StopRule16(B, S, ids, max_tokens, stops, prompt_len, max_model_len):
    """A request's END position and finish reason from its first T emitted ids, as one 16-bit word (the module docstring)."""
    return stop16(word_ops(B), S, [ids[i] for i in range(S.T)], max_tokens, [stops[j] for j in range(S.NSTOP)],
                  prompt_len, max_model_len)
