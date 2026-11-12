"""The seed rule a request Program checks, which the verifier requires to be 1: the seed its sampler read is its certified body's
``seed`` where the body names one, else `served_seed.derived` of the window's verifier coin and the request's id, in gates.

    ServedSeed_v1{COIN, RID}  (seeded: bit, seed: Word, coin: Array<COIN, u8>, user: Array<RID, u8>) -> Word
    SeedCheck_v1{COIN, RID}   (served: Word, seeded, seed, coin, user) -> bit
    SeedCheck_v2{COIN, RID}   (served: i64, seeded, seed, coin, user) -> bit: SeedCheck_v1's Calls on the seed word's bits
    SeedWord_v1               (served: i64) -> Word: the word's 64 bits
    SeedBlocks_v1{COIN, RID}  (coin, user) -> Array<8 + 16 n, Word>: SHA-512's IV, then the message's n padded blocks
    SeedSelect_v1             (seeded: bit, seed: Word, cv: Array<8, Word>, block: Array<16, Word>) -> Word: seed if seeded,
                              else bits 1-63 of the first word of cv's compression by block
    SeedEqual_v1              (a: Word, b: Word) -> bit: 1 iff a = b

(Word is ``Array<64, Value<1>>``, LSB first; a byte's bits are LSB first, a block's words big-endian.) `ServedSeed_v1` is the
rule: ``seed`` where ``seeded``, else the first 63 bits of SHA-512 over `served_seed.DERIVED`, u32be(COIN), the coin, u32be(RID)
and the id (`served_seed.derived`). It and `SeedCheck_v1` are calls and wiring only: `SeedBlocks_v1`, then a Call of
`lms.Sha512Compress_v1` on each block but the last (`blocks`: one where the message and its padding fit 128 bytes, else two),
then `SeedSelect_v1` on the last, which computes only the digest's first word, so no gate of the rule is one its output doesn't
read. COIN and RID are the coin's and the id's lengths in bytes, so the padding is
constant. ``seeded``, ``seed`` and ``user`` are ingress schema v1's (`boolean_io_shell.IngressSampled_v1`), certified, and the
coin is the window's. `SeedCheck_v1` is 1 exactly when ``served``, the seed the Program's sampler read, is the rule's
(`SeedEqual_v1`): any other seed fails it.

Every stochastic request Program returns `SeedCheck_v2{COIN, RID}` as a check (`check`, `predict.common.with_checks` and
`in_derive`), COIN the window's seed salt's 64 bytes and RID its id's length (`rid_of`), so serving's choice of seed is a witness
the check holds to one value (`program.inputs`). Its sampler reads the seed as one i64 word, which a Program reads at that width
alone, so `SeedCheck_v2` takes the word, splits it into its 64 bits (`SeedWord_v1`: `BitAtx64_v1` at each position) and checks
them by `SeedCheck_v1`'s Calls, so it too is calls and wiring only.
The rule's other inputs are the Program's certified parameters `PARAMS` (`params`), after its ``seed``: ``seeded``,
``body_seed`` and ``user`` are ingress schema v1's (`boolean_io_shell.IngressSampled_v1`'s outputs, of the same types), and
``seed_salt`` is the window's (``window.seed_salt``). An unseeded request's seed is derived (``seeded`` 0: `served_seed.derived`
of the salt and its id); a client-seeded one's is its body's (``seeded`` 1, ``body_seed``), and nothing else passes either.
`reference`, `check_reference` and `values` (a request's parameter values) are the functions.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import NOT, ZERO, and_reduce, const_bits, mux
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.prims import const
from verity.ml.circuits.refs import Coll, Explicit, concat
from verity.ml.circuits.types import Array, Value
from verity.ml.commitments.gates import sha512 as H
from verity.ml.commitments.gates.lms import Sha512Compress
from verity.ml.definitions.registry.sampling import bit_at
from verity_vllm.engine import served_seed

__all__ = ["COIN", "NAME", "PARAMS", "SeedBlocks", "SeedCheck", "SeedCheckWord", "SeedEqual", "SeedSelect", "SeedWord", "ServedSeed",
           "blocks", "check", "check_reference", "definition", "in_derive", "params", "reference", "rid_entry", "rid_of", "values"]

BIT = BL.BIT
U8 = Array(8, BIT)
WORD = Array(64, BIT)
I64 = Value(64)
STATICS = ("COIN", "RID")
CONFORMANCE = "protocol-defined: the served seed rule, whose function is `served_seed.derived` or the body's seed (`reference`)"
#: the window's seed salt in bytes (128 hex characters, `reads.seed_of`): every Program check's COIN
COIN = 64
#: the certified Program parameters the check reads after the sampler's ``seed``, in the order `SeedCheck_v1` takes them
PARAMS = ("seeded", "body_seed", "seed_salt", "user")
#: the check's Call name and its trace's node hint
NAME = "seed_rule"


def blocks(COIN: int, RID: int) -> int:
    """The SHA-512 blocks `ServedSeed_v1{COIN, RID}` compresses."""
    return -(-(len(served_seed.DERIVED) + 8 + COIN + RID + 17) // 128)


def _refuse(S: dict) -> str | None:
    bad = [k for k in STATICS if k in S and not (type(S[k]) is int and S[k] >= 1)]
    return f"the seed rule's {bad[0]} is a length in bytes, at least 1, not {S[bad[0]]!r}" if bad else None


def _ins(S: Any) -> tuple:
    return ("coin", Array(S.COIN, U8)), ("user", Array(S.RID, U8))


def _blocks(C: Any, S: Any, *e: list) -> list[list]:
    def const(data: bytes) -> list[list]:
        return [const_bits(x, 8) for x in data]

    data = (const(served_seed.DERIVED + S.COIN.to_bytes(4, "big")) + [list(b) for b in e[:S.COIN]]
            + const(S.RID.to_bytes(4, "big")) + [list(b) for b in e[S.COIN:]])
    data += const(H.pad_tail(len(data)))
    return [const_bits(v, 64) for v in H.IV] + [[x for b in range(8) for x in data[8 * k + 7 - b]] for k in range(len(data) // 8)]


SeedBlocks = TR.part("SeedBlocks", STATICS, lambda S: (_ins(S), Array(8 + 16 * blocks(S.COIN, S.RID), WORD)), _blocks,
                     "SHA-512's IV, then the padded blocks of served_seed.DERIVED, u32be(COIN), the coin, u32be(RID) and the id, "
                     "as 64-bit words: constants and wiring", conformance=CONFORMANCE)
SeedSelect = TR.part("SeedSelect", (), lambda S: ((("seeded", BIT), ("seed", WORD), ("cv", Array(8, WORD)), ("block", Array(16, WORD))),
                                                 WORD),
                     lambda C, S, seeded, seed, *e: [mux(C, seeded[0], list(seed),
                                                         list(H.compress_gadget(C, list(e[:8]), list(e[8:]))[0][1:]) + [ZERO])],
                     "seed where seeded, else bits 1-63 of the first word of SHA-512's last compression of cv by block (with its "
                     "feed-forward): the digest's first 8 bytes >> 1, and no gate for the words it doesn't read",
                     conformance=CONFORMANCE)
SeedEqual = TR.part("SeedEqual", (), lambda S: ((("a", WORD), ("b", WORD)), BIT),
                    lambda C, S, a, b: [[and_reduce(C, [NOT(x ^ y) for x, y in zip(a, b)])]], "1 iff a = b",
                    conformance=CONFORMANCE)


def _rule_sig(S: Any) -> tuple:
    return (("seeded", BIT), ("seed", WORD), *_ins(S)), WORD


def _check_sig(S: Any) -> tuple:
    params, _ = _rule_sig(S)
    return (("served", WORD), *params), BIT


def _words(bits: list, k: int, n: int) -> Coll:
    return Coll(Array(n, WORD), Explicit(tuple(bits[64 * k:64 * (k + n)])))


def _rule(B: Any, S: Any, seeded: Coll, seed: Coll, coin: Coll, user: Coll) -> Coll:
    out = B.call(bind(SeedBlocks, COIN=S.COIN, RID=S.RID), coin, user)
    bits = [out.refs[k] for k in range(out.refs.count)]
    cv, n = _words(bits, 0, 8), blocks(S.COIN, S.RID)
    for i in range(n - 1):
        cv = B.call(Sha512Compress, cv, _words(bits, 8 + 16 * i, 16))
    return B.call(SeedSelect, seeded, seed, cv, _words(bits, 8 + 16 * (n - 1), 16))


def _check(B: Any, S: Any, served: Coll, *ins: Coll) -> Coll:
    return B.call(SeedEqual, served, _rule(B, S, *ins))


ServedSeed = CompositeDefinition("ServedSeed", 1, STATICS, _rule_sig, _rule, refuse=_refuse, conformance=CONFORMANCE,
                                 doc="A sampled request's seed: its body's where it names one, else served_seed.derived of the "
                                     "window's verifier coin (COIN bytes) and its id (RID bytes), on Sha512Compress_v1 Calls")
SeedCheck = CompositeDefinition("SeedCheck", 1, STATICS, _check_sig, _check, refuse=_refuse, conformance=CONFORMANCE,
                                doc="1 iff served, the seed a Program's sampler read, is ServedSeed_v1's of the certified inputs")


def _bits(B: Any, S: Any, served: Coll) -> Coll:
    at = [B.call(const(32, i)) for i in range(64)]
    return B.batch(bit_at(64), served, Coll(Array(64, Value(32)), concat([c.refs for c in at])), axes=(None, 0))


SeedWord = CompositeDefinition("SeedWord", 1, (), lambda S: ((("served", I64),), WORD), _bits, conformance=CONFORMANCE,
                               doc="the i64 seed word's 64 bits, LSB first: BitAtx64_v1 at each position")


def _word_sig(S: Any) -> tuple:
    params, ret = _check_sig(S)
    return (("served", I64), *params[1:]), ret


def _word_check(B: Any, S: Any, served: Coll, *ins: Coll) -> Coll:
    return _check(B, S, B.call(SeedWord, served), *ins)


SeedCheckWord = CompositeDefinition("SeedCheck", 2, STATICS, _word_sig, _word_check, refuse=_refuse, conformance=CONFORMANCE,
                                    doc="1 iff served, the i64 seed word a Program's sampler read, is ServedSeed_v1's of the "
                                        "certified inputs: SeedCheck_v1's Calls on its 64 bits (SeedWord_v1)")


def rid_of(request_id: str) -> int:
    """The seed rule's RID for a request: its certified id's length in bytes, as `served_seed.derived` hashes it."""
    if type(request_id) is not str or not request_id:
        raise ValueError(f"a request's id is a non-empty string, not {request_id!r}")
    return len(request_id.encode())


def rid_of_ids(ids: Sequence[str]) -> int | None:
    """The RID a row's requests share (`rid_of` of each id), None where they share none (no ids, or ids of two lengths): one
    Program per shape reads one id length, so the row's stochastic request Programs then refuse at their build (`rid_in`)."""
    rids = {rid_of(i) for i in ids}
    return rids.pop() if len(rids) == 1 else None


def rid_entry(policy: dict, requests: Sequence[dict]) -> dict:
    """``{"rid": n}`` where ``policy`` is stochastic and its row's ``requests`` share the RID n (`rid_of_ids` of their
    ``request_id``), else {}: the entry a row's sampling policy carries for its requests' seed rule (`rid_in`)."""
    if policy.get("kind") != "stochastic":
        return {}
    rid = rid_of_ids([str(r["request_id"]) for r in requests if r.get("request_id") is not None])
    return {} if rid is None else {"rid": rid}


def rid_in(sampling: Any) -> int:
    """The RID a stochastic request's sampling policy states (its ``rid``: `ingress_policy.policy_of`, or `rid_of_ids` over a row)."""
    rid = sampling.get("rid") if isinstance(sampling, dict) else None
    if type(rid) is not int or rid < 1:
        raise ValueError(f"a stochastic request's Program checks its seed against the seed rule over its id, whose length in bytes "
                         f"its sampling policy states as `rid` (`seed_rule.rid_of`, or `rid_of_ids` of a row whose ids share one "
                         f"length), and this one states {rid!r}")
    return rid


def inputs(sampling: Any) -> tuple:
    """Example values of `PARAMS` on torch's meta device for a stochastic request wrapper (`vllm_meta.make_wrapper`): bool
    tensors of the parameters' shapes, at the policy's RID (`rid_in`)."""
    import torch
    return tuple(torch.empty(shape, dtype=torch.bool, device="meta") for shape in ((), (64,), (COIN, 8), (rid_in(sampling), 8)))


def params(RID: int) -> tuple[tuple[str, Any], ...]:
    """The Program parameters `PARAMS` of a request whose id is RID bytes, with their types."""
    if (why := _refuse({"RID": RID})) is not None:
        raise ValueError(why)
    return tuple(zip(PARAMS, (BIT, WORD, Array(COIN, U8), Array(RID, U8))))


def definition(RID: int) -> Any:
    """The rule's Call for a request whose id is RID bytes (`check`'s): `SeedCheck_v2{COIN, RID}` bound."""
    return bind(SeedCheckWord, COIN=COIN, RID=RID)


def check(B: Any, seed: Coll, ins: Sequence[Coll]) -> tuple[Coll, str]:
    """`SeedCheck_v2` of the Program's sampler ``seed`` (its one i64 word) and its `PARAMS` ``ins``, as the next Call of the
    Builder ``B``, and its description: (the bit, what it checks), the form `with_checks` and `derive.Ctx.check` take."""
    if seed.type not in (I64, Array(1, I64)):
        raise ValueError(f"the seed rule reads the sampler's seed, one i64, not {seed.type}")
    last = ins[-1].type if len(ins) == len(PARAMS) else None
    RID = last.n if isinstance(last, Array) and last.n >= 1 else 0
    if not RID or [c.type for c in ins] != [t for _, t in params(RID)]:
        raise ValueError(f"the seed rule reads {', '.join(PARAMS)}: a bit, a 64-bit word, the window's {COIN}-byte seed salt and "
                         f"the id's bytes, not {[str(c.type) for c in ins]}")
    fn = definition(RID)
    return (B.call(fn, Coll(I64, seed.refs), *ins, name=NAME),
            f"the seed rule ({fn.id} of SeedCheck_v1): the sampler's seed is the body's, else derived from the window's seed salt "
            f"and the request's id")


def in_derive(ctx: Any, at: Any) -> None:
    """Where the traced source takes `PARAMS` (a stochastic request wrapper, `vllm_meta.StochasticRequest`), `check` of its
    ``seed`` and them as the Program's next Call and check (`Ctx.check`), which no fx node lowers; one it can't take refuses at fx
    node ``at``."""
    from verity_vllm.program.frontend.outcomes import refuse
    have = [n for n in PARAMS if n in ctx.user_inputs]
    if not have:
        return
    try:
        if have != list(PARAMS) or "seed" not in ctx.user_inputs:
            raise ValueError(f"the source takes {have} of {list(PARAMS)}, and {'a' if 'seed' in ctx.user_inputs else 'no'} seed")
        ctx.check(*check(ctx.B, ctx.user_inputs["seed"], [ctx.user_inputs[n] for n in PARAMS]))
    except (TypeError, ValueError) as e:
        refuse("shape", at, None, f"the seed rule: {e}")
    ctx.node_hints[len(ctx.B.nodes) - 1] = NAME


def values(seed: int | None, coin: bytes, request_id: str) -> list[np.ndarray]:
    """A request's `PARAMS` values as a Program's inputs take them (bits, LSB first; a byte array's bytes in order): ``seeded``
    and ``body_seed`` from its certified body's ``seed`` (0 where it names none), ``seed_salt`` the window's coin, ``user`` its
    id."""
    def bits(x: int, n: int) -> np.ndarray:
        return np.array([(int(x) >> k) & 1 for k in range(n)], dtype=np.uint32)

    def octets(b: bytes) -> np.ndarray:
        return np.concatenate([bits(x, 8) for x in b]) if b else np.zeros(0, dtype=np.uint32)

    if type(coin) is not bytes or len(coin) != COIN:
        raise ValueError(f"the window's seed salt is {COIN} bytes, not {coin!r}")
    return [bits(seed is not None, 1), bits(seed or 0, 64), octets(coin), octets(request_id.encode())]


def reference(seeded: int, seed: int, coin: bytes, user: bytes) -> int:
    """`ServedSeed_v1`'s function."""
    return int(seed) if seeded else served_seed.derived(bytes(coin), bytes(user).decode())


def check_reference(served: int, seeded: int, seed: int, coin: bytes, user: bytes) -> int:
    """`SeedCheck_v1`'s function, and `SeedCheck_v2`'s of the served word's unsigned value."""
    return int(int(served) % (1 << 64) == reference(seeded, seed, coin, user))
