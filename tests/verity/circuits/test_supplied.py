"""`verity.circuits.supplied` (PROTOCOL.md §5.4): a mark of another form is refused by its code; the registry names the
Lean registry's functions; a block's part of a supplied input keeps C's mark; and an evaluation that reads a commitment
registered after its coin is refused by name."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from verity.circuits import supplied as S
from verity.protocols.proof.spot.verifier.one_stage import registered as G


def _entry(name, fn, args, layout, bits, n=1):
    return [name, n, {"supplied": fn, "args": args, "layout": layout, "bits": bits}]


#: the mark of a storage round's coin, three 32-bit leaves
ROUND_3 = _entry("round_3", "pous.session.round_coin/v1", {"round": 3}, "halves", 96)[2]


@pytest.mark.parametrize("entry", [
    ["k", 1, {"supplied": "window.salt/v1"}],
    ["k", 1, {"supplied": "window.salt/v1", "args": {}, "layout": "bytes", "bits": 512}],
    ["round_3/p0", 1, {**S.part("round_3", ROUND_3, 2, 3, 32), "of": ["round_3", 3, 3]}],
])
def test_a_mark_of_another_form_is_refused(entry):
    with pytest.raises(S.SuppliedError, match="^E_SUPPLIED_MARK: ") as e:
        S.mark_of(entry)
    assert e.value.code == "E_SUPPLIED_MARK"


def test_unmarked_entries_have_no_mark():
    assert S.mark_of(["x", 2]) is None and S.mark_of(["round_3", 3, ROUND_3]) == ROUND_3


def test_a_blocks_part_names_cs_input_and_range():
    """The opening's key reads leaf 2 of round 3's coin (i_j's low half): a whole leaf, so no second function gives i_j."""
    p = S.part("round_3", ROUND_3, 2, 3, 32)
    assert p == {**ROUND_3, "bits": 32, "of": ["round_3", 2, 3]} and S.mark_of(["round_3/p0", 1, p]) == p
    with pytest.raises(S.SuppliedError, match="^E_SUPPLIED_WIDTH: "):
        S.part("round_3", ROUND_3, 2, 4, 32)


def test_the_order_rule():
    S.check_order({"store": 0, "session": 1}, coin=2)
    with pytest.raises(S.SuppliedError, match=r"^E_SUPPLIED_ORDER: \['late'\] registered after the coin"):
        S.check_order({"store": 0, "late": 2}, coin=2)


def test_every_function_reads_only_public_things():
    for f in S.FUNCTIONS.values():
        assert f.origin in S.ORIGINS and set(f.reads) <= set(S.READS) and f.reads
        assert f.origin == "supplied" or f.reads == ("coin",), f.name


#: the Lean registry: each function a supplied input may name has its body there, and a function exists when it has one
LEAN_REGISTRY = Path(__file__).parents[3] / "verity/protocols/proof/spot/verifier/lean/Spot/Supplied.lean"
#: a function's body in the Lean registry: an arm of `Supplied.body`'s match on the function's name
LEAN_BODY = re.compile(r'^  \| "([a-z0-9_]+(?:\.[a-z0-9_]+)+/v[1-9][0-9]*)" =>', re.M)
#: the functions the Lean registry has no body of yet; each strict failure below is one to take off this list
NO_LEAN_BODY = frozenset({"poub.read.pass_seed/v1", "poub.read.pass_root/v1"})


@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.xfail(strict=True, reason="no Lean body yet"))
                                  if n in NO_LEAN_BODY else n for n in S.FUNCTIONS])
def test_each_function_has_its_lean_body(name):
    assert name in LEAN_BODY.findall(LEAN_REGISTRY.read_text()), f"{name} has no body in {LEAN_REGISTRY.name}"


#: the storage session's material and each body's value for it, as the memory attester writes them: the session commitment
#: (`SessionCommitment.to_json`), three coin openings, the registered LMS key (at the deployed parameters), the message m and
#: its signature at leaf 0, and the values of the round coins, the header and the context
VECTORS = json.loads((Path(__file__).parent / "pous_session_vectors.json").read_text())
ROUND, HEADER, CONTEXT = "pous.session.round_coin/v1", "pous.session.header/v1", "pous.session.context/v1"
MESSAGE = "pous.session.message/v1"
KEY = VECTORS["commitment"]["coin_key"]
COMMITMENT = {k: v for k, v in VECTORS["commitment"].items() if k != "coin_key"}
#: what the verifier holds of the session from before its first coin: the header's fields (the coin key by its hash), the
#: LMS key, and the place the verifier registered them at (before the place the coin is released at, 1)
HELD = {"commitment": COMMITMENT, "key": VECTORS["key"], "place": 0}
#: what the session releases after it: the round coin's openings and the coin key itself, how it ended with its rounds,
#: which the header's coin is, and the signed message with its signature, which the message's coin is
COIN = {"openings": VECTORS["openings"], "coin_key": KEY}
ENDED = {"ended": "complete", "rounds": 3}
SIGNED = {"message": VECTORS["message"], "signature": VECTORS["signature"]}
#: the configuration's pin of the memory attester's key: its SHA-512
PINNED = (hashlib.sha512(bytes.fromhex(VECTORS["key"])).digest(),)


def text(doc) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"))


def held(commitment=None, coin=None, ended=None, signed=None, **over) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """The verifier's ``(coins, commitments)``: the commitment under each of the four functions' names, ``over`` replacing
    parts of it, and the release under the round coin's, the header's and the message's; ``commitment``, ``coin``, ``ended``
    and ``signed`` give a text as it is."""
    blob = (commitment if commitment is not None else text({**HELD, **over})).encode()
    return ({ROUND: (coin if coin is not None else text(COIN)).encode(),
             HEADER: (ended if ended is not None else text(ENDED)).encode(),
             MESSAGE: (signed if signed is not None else text(SIGNED)).encode()},
            {ROUND: blob, HEADER: blob, CONTEXT: blob, MESSAGE: blob})


def with_commitment(**fields):
    return held(commitment=text({**HELD, "commitment": {**COMMITMENT, **fields}}))


def with_coin(**fields):
    return held(coin=text({**COIN, **fields}))


def with_ended(**fields):
    return held(ended=text({**ENDED, **fields}))


def with_signed(**fields):
    return held(signed=text({**SIGNED, **fields}))


def with_key(key: bytes):
    """A coin key the commitment holds the hash of, so the opening is the only thing that can refuse."""
    return held(commitment=text({**HELD, "commitment": {**COMMITMENT, "coin_key_sha512": hashlib.sha512(key).hexdigest()}}),
                coin=text({**COIN, "coin_key": key.hex()}))


def mark(fn: str, **args) -> dict:
    return {"supplied": fn, "args": args, "layout": "le", "bits": 8}


def refusal(tables, fn: str, pinned=PINNED, **args) -> str:
    with pytest.raises(ValueError) as e:
        G.body(*tables, "x", mark(fn, **args), pinned)
    return str(e.value)


@pytest.mark.parametrize("j", [0, 1, 2])
def test_round_coin_vectors(j):
    got = G.body(*held(), "x", mark(ROUND, round=j))
    assert got == ([("bytes", VECTORS["round_coin"][j])], 12)
    assert got[0][0][1] == j.to_bytes(4, "big").hex() + VECTORS["openings"][j]["index"].to_bytes(8, "big").hex()


def test_header_vector():
    row = bytes.fromhex(VECTORS["header"])
    assert len(row) == 754 and VECTORS["head"] == 753 and row.endswith(b"}\x00")
    assert G.body(*held(), "x", mark(HEADER, head=753)) == ([("bytes", VECTORS["header"])], 754)


def test_header_of_an_even_length_has_no_padding():
    row, n = G.body(*with_commitment(limit_ns=85400), "x", mark(HEADER, head=752))
    assert n == 752 and bytes.fromhex(row[0][1]).endswith(b"}")
    doc = json.loads(bytes.fromhex(row[0][1]))
    assert doc["ended"] == "complete" and doc["rounds"] == doc["k"] == 3 and doc["limit_ns"] == 85400
    assert doc["coin_key_sha512"] == hashlib.sha512(bytes.fromhex(KEY)).hexdigest() == COMMITMENT["coin_key_sha512"]


def test_context_vector():
    assert G.body(*held(), "x", mark(CONTEXT), PINNED) == ([("bytes", VECTORS["context"])], 106)


def test_message_vector():
    """The message is the 64 bytes released in the clear, as released, once its signature verifies under the pinned key."""
    assert G.body(*held(), "x", mark(MESSAGE), PINNED) == ([("bytes", VECTORS["message"])], 64)


def test_no_pin_refuses_the_key():
    """With no key pinned, the context and the message have no statement: the session's own key is never taken."""
    for fn in (CONTEXT, MESSAGE):
        assert refusal(held(), fn, pinned=()) == NO_STATEMENT + fn + "'s " + G.UNPINNED


def test_the_marks_reach_the_leaves_their_layouts_read():
    rows = G.of_mark(*held(), "x", {"supplied": ROUND, "args": {"round": 2}, "layout": "halves", "bits": 96})
    assert G.row_leaves_of(rows[0], lambda _k, n: bytes.fromhex(n), 32, False) == [2, 0, 70000]
    rows = G.of_mark(*held(), "x", {"supplied": CONTEXT, "args": {}, "layout": "le", "bits": 848}, PINNED)
    assert len(G.row_leaves_of(rows[0], lambda _k, n: bytes.fromhex(n), 16, False)) == 53


def _flip(h: str, at: int = 0) -> str:
    return h[:at] + ("1" if h[at] == "0" else "0") + h[at + 1:]


def _openings(**change):
    return [{**o, **change} if o["j"] == 0 else o for o in VECTORS["openings"]]


def coin_of(openings, **more) -> str:
    return text({"openings": openings, "coin_key": KEY, **more})


def raw(old: str, new: str) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """The commitment's text with ``old`` replaced by ``new``, where it is in it."""
    t = text(HELD)
    assert t.count(old) == 1, old
    return held(commitment=t.replace(old, new))


def raw_coin(old: str, new: str):
    t = text(COIN)
    assert t.count(old) == 1, old
    return held(coin=t.replace(old, new))


def raw_ended(old: str, new: str):
    t = text(ENDED)
    assert t.count(old) == 1, old
    return held(ended=t.replace(old, new))


NO_STATEMENT = "supplied input x: no statement: "
CANON = "'s commitment holds a number that is no canonical natural"
COIN_CANON = "'s coin holds a number that is no canonical natural"
R0 = {"round": 0}
BAD_COIN = ROUND + "'s round 0's index 5 is not its committed coin"
NOT_THE_KEY = "'s coin's key is not the one the commitment holds the hash of"


def other_session_coin(**kw):
    """The openings of ANOTHER session of the same attester: its own session id and coin root, the same coin key, indices that
    are blocks of the first session's store too."""
    from verity.protocols.accounting.memory.pous.attester import memory_attester as MA
    key = bytes.fromhex(KEY)
    sid = hashlib.sha512(b"another session").digest()[:32]
    tree = MA.CoinTree(sid, VECTORS["commitment"]["blocks"], [o["index"] for o in VECTORS["openings"]], key,
                       [bytes.fromhex(o["salt"]) for o in VECTORS["openings"]])
    assert tree.root.hex() != VECTORS["commitment"]["coin_root"]
    return text({**COIN, "openings": [tree.opening(j).to_json() for j in range(3)], **kw})


#: each refusal of the three bodies, with its text: the one place the Lean's refusals are read from, as the difference test
#: holds each to it
REFUSALS = [
    ("no commitment held", ({}, {}), ROUND, R0, "the verifier holds no commitment for " + ROUND),
    ("no coin held", ({}, held()[1]), ROUND, R0, "the verifier holds no coin for " + ROUND),
    ("a commitment that is no JSON", held(commitment="{"), ROUND, R0, ROUND + "'s commitment is no JSON"),
    ("a commitment that is not UTF-8", ({}, {ROUND: b"\xff"}), ROUND, R0, ROUND + "'s commitment is no JSON"),
    ("a commitment that is no object", held(commitment="[]"), ROUND, R0, ROUND + "'s commitment is no object"),
    ("no commitment object", held(commitment=text({**HELD, "commitment": 5})), ROUND, R0,
     ROUND + "'s commitment holds no commitment object"),
    ("a session in capitals", with_commitment(session=VECTORS["commitment"]["session"].upper()), ROUND, R0,
     ROUND + "'s commitment's session is no lowercase hex"),
    ("a session of 31 bytes, read by the round coin", with_commitment(session="ab" * 31), ROUND, R0,
     ROUND + "'s commitment's session is 31 bytes, not 32"),
    ("a session of 33 bytes, read by the round coin", with_commitment(session="ab" * 33), ROUND, R0,
     ROUND + "'s commitment's session is 33 bytes, not 32"),
    ("a session of 31 bytes, read by the header", with_commitment(session="ab" * 31), HEADER, {"head": 753},
     HEADER + "'s commitment's session is 31 bytes, not 32"),
    ("a session of 33 bytes, read by the header", with_commitment(session="ab" * 33), HEADER, {"head": 753},
     HEADER + "'s commitment's session is 33 bytes, not 32"),
    ("a session of 31 bytes, read by the context", with_commitment(session="ab" * 31), CONTEXT, {},
     CONTEXT + "'s commitment's session is 31 bytes, not 32"),
    ("a session of 33 bytes, read by the context", with_commitment(session="ab" * 33), CONTEXT, {},
     CONTEXT + "'s commitment's session is 33 bytes, not 32"),
    ("a k that is negative", with_commitment(k=-1), ROUND, R0, ROUND + CANON),
    ("a k that is a boolean", with_commitment(k=True), ROUND, R0, ROUND + "'s commitment's k is no u32"),
    ("a k past u32", with_commitment(k=1 << 32), ROUND, R0, ROUND + "'s commitment's k is no u32"),
    ("blocks past u64", with_commitment(blocks=1 << 64), ROUND, R0, ROUND + "'s commitment's blocks is no u64"),
    ("a round that is no u32", held(), ROUND, {"round": 1 << 32}, ROUND + "'s round 4294967296 is no u32"),
    ("a round past k", held(), ROUND, {"round": 3}, ROUND + "'s round 3 is no round of the session's 3"),
    ("a coin of more than its openings and key", held(coin=coin_of(VECTORS["openings"], coin_root="00" * 64)), ROUND, R0,
     ROUND + "'s coin holds more than its openings and its key: the coin root is the verifier's own"),
    ("a coin that carries a commitment", held(coin=coin_of(VECTORS["openings"], commitment=COMMITMENT)), ROUND, R0,
     ROUND + "'s coin holds more than its openings and its key: the coin root is the verifier's own"),
    ("a coin with no key", held(coin=text({"openings": VECTORS["openings"]})), ROUND, R0,
     ROUND + "'s coin's coin_key is no lowercase hex"),
    ("a coin key that is no hex", with_coin(coin_key="zz"), ROUND, R0, ROUND + "'s coin's coin_key is no lowercase hex"),
    ("a coin key whose hash is not the commitment's", with_coin(coin_key=_flip(KEY)), ROUND, R0, ROUND + NOT_THE_KEY),
    ("a coin key short of a byte", with_coin(coin_key=KEY[2:]), ROUND, R0, ROUND + NOT_THE_KEY),
    ("a coin that is no list of openings", held(coin=text({"openings": {}, "coin_key": KEY})), ROUND, R0,
     ROUND + "'s coin holds no openings list"),
    ("a coin that is no JSON", held(coin="x"), ROUND, R0, ROUND + "'s coin is no JSON"),
    ("no opening of the round", held(coin=coin_of(VECTORS["openings"][1:])), ROUND, R0, ROUND + "'s coin holds no opening of round 0"),
    ("an opening without a path", held(coin=coin_of([{k: v for k, v in o.items() if k != "path"} if o["j"] == 0 else o
                                                       for o in VECTORS["openings"]])), ROUND, R0,
     ROUND + "'s opening of round 0 is malformed"),
    ("an index that is no u64", held(coin=coin_of(_openings(index=1 << 64))), ROUND, R0,
     ROUND + "'s round 0's index 18446744073709551616 is no u64"),
    ("an index past the blocks", with_commitment(blocks=5), ROUND, R0, ROUND + "'s round 0's index 5 is no block of the session's 5"),
    ("another index than the committed coin", held(coin=coin_of(_openings(index=6))), ROUND, R0,
     ROUND + "'s round 0's index 6 is not its committed coin"),
    ("another salt", held(coin=coin_of(_openings(salt=_flip(VECTORS["openings"][0]["salt"])))), ROUND, R0, BAD_COIN),
    ("a salt of the wrong length", held(coin=coin_of(_openings(salt=VECTORS["openings"][0]["salt"][2:]))), ROUND, R0, BAD_COIN),
    ("a short path", held(coin=coin_of(_openings(path=VECTORS["openings"][0]["path"][:1]))), ROUND, R0, BAD_COIN),
    ("another coin root held", with_commitment(coin_root=_flip(VECTORS["commitment"]["coin_root"])), ROUND, R0, BAD_COIN),
    ("a key a byte short, whose hash the commitment holds", with_key(bytes.fromhex(KEY)[:-1]), ROUND, R0, BAD_COIN),
    ("a key with its top bit set, whose hash the commitment holds", with_key(bytes.fromhex(KEY)[:-1] + b"\xff"), ROUND, R0,
     BAD_COIN),
    ("a coin opened under another session's root and id", held(coin=other_session_coin()), ROUND, R0, BAD_COIN),
    ("a coin opened under another session's root, the commitment holding that session's id too",
     held(commitment=text({**HELD, "commitment": {**COMMITMENT, "session": hashlib.sha512(b"another session").hexdigest()[:64]}}),
          coin=other_session_coin()), ROUND, R0, BAD_COIN),
    ("a header of another length", held(), HEADER, {"head": 752}, HEADER + "'s header is 753 bytes, not 752"),
    ("a header of a binding that is no hex", with_commitment(binding="zz"), HEADER, {"head": 753},
     HEADER + "'s commitment's binding is no lowercase hex"),
    ("a header of a limit that is a string", with_commitment(limit_ns="854000"), HEADER, {"head": 753},
     HEADER + "'s commitment's limit_ns is no u64"),
    ("a session that ended late", with_ended(ended="late"), HEADER, {"head": 753}, HEADER + "'s session did not end complete"),
    ("a session with no ended", with_ended(ended=None), HEADER, {"head": 753}, HEADER + "'s coin holds no ended string"),
    ("a session of 2 rounds", with_ended(rounds=2), HEADER, {"head": 753}, HEADER + "'s session has 2 rounds, not its 3"),
    ("a session of 4 rounds", with_ended(rounds=4), HEADER, {"head": 753}, HEADER + "'s session has 4 rounds, not its 3"),
    ("a session of rounds past u32", with_ended(rounds=1 << 32), HEADER, {"head": 753}, HEADER + "'s coin's rounds is no u32"),
    ("a late session of too few rounds", with_ended(ended="late", rounds=1), HEADER, {"head": 753},
     HEADER + "'s session did not end complete"),
    ("no release held for the header", ({ROUND: held()[0][ROUND]}, held()[1]), HEADER, {"head": 753},
     "the verifier holds no coin for " + HEADER),
    ("a key of 15 bytes", held(key="ab" * 15), CONTEXT, {}, CONTEXT + "'s " + G.UNPINNED),
    ("a key the configuration doesn't pin", held(key=_flip(VECTORS["key"], 40)), CONTEXT, {}, CONTEXT + "'s " + G.UNPINNED),
    ("a key that is no hex", held(key="x"), CONTEXT, {}, CONTEXT + "'s commitment's key is no lowercase hex"),
    ("a message under a key not pinned", held(key=_flip(VECTORS["key"], 40)), MESSAGE, {}, MESSAGE + "'s " + G.UNPINNED),
    ("a message the signature doesn't sign", with_signed(message=_flip(VECTORS["message"])), MESSAGE, {},
     MESSAGE + "'s " + G.BAD_SIGNATURE),
    ("a signature of another session", with_commitment(session=_flip(VECTORS["commitment"]["session"])), MESSAGE, {},
     MESSAGE + "'s " + G.BAD_SIGNATURE),
    ("a truncated signature", with_signed(signature=VECTORS["signature"][:-2]), MESSAGE, {}, MESSAGE + "'s " + G.BAD_SIGNATURE),
    ("a signature of another leaf", with_signed(signature="00000001" + VECTORS["signature"][8:]), MESSAGE, {},
     MESSAGE + "'s " + G.BAD_SIGNATURE),
    ("no signature", with_signed(signature=None), MESSAGE, {}, MESSAGE + "'s coin's signature is no lowercase hex"),
    ("a message of 63 bytes", with_signed(message="ab" * 63), MESSAGE, {}, MESSAGE + "'s message is 63 bytes, not 64"),
    ("a message in capitals", with_signed(message=VECTORS["message"].upper()), MESSAGE, {},
     MESSAGE + "'s coin's message is no lowercase hex"),
    ("no message", with_signed(message=None), MESSAGE, {}, MESSAGE + "'s coin's message is no lowercase hex"),
    ("no release held for the message", ({ROUND: held()[0][ROUND], HEADER: held()[0][HEADER]}, held()[1]), MESSAGE, {},
     "the verifier holds no coin for " + MESSAGE),
    # numbers: one strict reader, in every body
    ("a k of 1e0", raw('"k":3,', '"k":3e0,'), ROUND, R0, ROUND + CANON),
    ("a k with a leading zero", raw('"k":3,', '"k":03,'), ROUND, R0, ROUND + "'s commitment is no JSON"),
    ("a k with a sign", raw('"k":3,', '"k":+3,'), ROUND, R0, ROUND + "'s commitment is no JSON"),
    ("a k of -0", raw('"k":3,', '"k":-0,'), ROUND, R0, ROUND + CANON),
    ("a k with a fraction", raw('"k":3,', '"k":3.0,'), ROUND, R0, ROUND + CANON),
    ("blocks of 1e2 (read by the header)", raw('"blocks":1048576', '"blocks":1e2'), HEADER, {"head": 753}, HEADER + CANON),
    ("a limit with a leading zero", raw('"limit_ns":854000', '"limit_ns":0854000'), HEADER, {"head": 753},
     HEADER + "'s commitment is no JSON"),
    ("rounds of 3.0", raw_ended('"rounds":3', '"rounds":3.0'), HEADER, {"head": 753}, HEADER + COIN_CANON),
    ("a number read by the context alone", raw('"block_bytes":2056', '"block_bytes":2056.0'), CONTEXT, {}, CONTEXT + CANON),
    ("an index of 5e0", raw_coin('"index":5', '"index":5e0'), ROUND, R0, ROUND + COIN_CANON),
    ("an index with a leading zero", raw_coin('"index":5', '"index":05'), ROUND, R0, ROUND + "'s coin is no JSON"),
    ("an index of -0", raw_coin('"index":5', '"index":-0'), ROUND, R0, ROUND + COIN_CANON),
    ("a j of 0.0", raw_coin('"j":0', '"j":0.0'), ROUND, R0, ROUND + COIN_CANON),
]


@pytest.mark.parametrize("name, tables, fn, args, why", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_what_has_no_statement_is_refused_by_name(name, tables, fn, args, why):
    assert refusal(tables, fn, **args) == NO_STATEMENT + why


def test_a_number_in_a_string_is_not_one():
    """`1e2` or `-0` inside a string is text: only the numbers of the JSON are held to the canonical form."""
    doc = {**HELD, "note": "limit -0 and 1e2, 03 + 3.0 \\\" 5e1"}
    assert G.body(*held(commitment=text(doc)), "x", mark(CONTEXT), PINNED) == ([("bytes", VECTORS["context"])], 106)


def test_a_last_opening_stands_for_its_round():
    """`session_proofs.clear_reasons` reads the last opening of a round: an earlier, wrong one does not count."""
    bad = {**VECTORS["openings"][0], "index": 6}
    got = G.body(*held(coin=coin_of([bad, *VECTORS["openings"]])), "x", mark(ROUND, round=0))
    assert got == ([("bytes", VECTORS["round_coin"][0])], 12)
    assert refusal(held(coin=coin_of([*VECTORS["openings"], bad])), ROUND, round=0).endswith("index 6 is not its committed coin")


def test_the_arguments_each_function_takes():
    assert refusal(held(), ROUND) == "supplied input x: " + ROUND + " takes the args [round]"
    assert refusal(held(), HEADER, round=0) == "supplied input x: " + HEADER + " takes the args [head]"
    assert refusal(held(), CONTEXT, head=1) == "supplied input x: " + CONTEXT + " takes no args"
    assert refusal(held(), MESSAGE, head=1) == "supplied input x: " + MESSAGE + " takes no args"


def test_the_header_is_compared_with_what_was_released_not_assumed():
    """A header the verifier computes is the complete one: a release that says otherwise makes no statement, and one that
    says the same gives the vector."""
    assert G.body(*with_ended(ended="complete", rounds=3), "x", mark(HEADER, head=753))[1] == 754
    for over in ({"ended": "late"}, {"ended": ""}, {"rounds": 0}, {"rounds": 4}):
        assert "no statement" in refusal(with_ended(**over), HEADER, head=753)


def test_the_coin_root_is_the_verifiers_never_the_coins():
    """A coin that brings its own coin root (another session's, with openings that open under it) is refused by name, and the
    verifier's own root refuses the openings of a root it didn't commit."""
    other = {**COMMITMENT, "coin_root": _flip(COMMITMENT["coin_root"])}
    assert refusal(held(coin=coin_of(VECTORS["openings"], commitment=other)), ROUND, round=0).endswith(
        "holds more than its openings and its key: the coin root is the verifier's own")
    assert refusal(with_commitment(coin_root=other["coin_root"]), ROUND, round=0).endswith("is not its committed coin")


def test_a_coin_valid_under_another_sessions_root_is_refused_by_name():
    """Another session of the same attester: openings that open under its own coin root and session id, to the same key and
    the same blocks, are no coin of the session the verifier holds."""
    assert refusal(held(coin=other_session_coin()), ROUND, round=0) == NO_STATEMENT + BAD_COIN


HEADER_0 = (HEADER, {"head": 753})
ROUND_0 = (ROUND, {"round": 0})
CONTEXT_0 = (CONTEXT, {})
MESSAGE_0 = (MESSAGE, {})
LATE = "E_SUPPLIED_ORDER: ['commitment pous.session.header/v1', 'commitment pous.session.round_coin/v1'] registered after "
NO_COMMITMENT_PLACE = "released: the verifier names no place of " + HEADER + "'s commitment, which a coin it holds follows"
#: each refusal of the order rule over a session read by the round coin and the header, whose coins are the release: (name, the
#: marks, the verifier's tables, the places `released` gives the coins, the text): a commitment after its coin, at its coin's
#: place, with no place, beside a coin with none
ORDER_REFUSALS = [
    ("a commitment placed after its coin", (ROUND_0, HEADER_0), held(place=5), {ROUND: 2, HEADER: 2},
     LATE + HEADER + ", which a supplied value reads"),
    ("a commitment placed at its coin's place", (ROUND_0, HEADER_0), held(place=2), {ROUND: 2, HEADER: 2},
     LATE + HEADER + ", which a supplied value reads"),
    ("a commitment placed at the round coin's place", (ROUND_0, HEADER_0), held(place=2), {ROUND: 2, HEADER: 3},
     LATE + ROUND + ", which a supplied value reads"),
    ("a commitment with no place", (ROUND_0, HEADER_0), held(place=None), {ROUND: 2, HEADER: 2}, NO_COMMITMENT_PLACE),
    ("a commitment whose place is no natural", (ROUND_0, HEADER_0), held(place="0"), {ROUND: 2, HEADER: 2}, NO_COMMITMENT_PLACE),
    ("a coin with no place", (ROUND_0, HEADER_0), held(), {HEADER: 2},
     "released: the verifier names no place of " + ROUND + "'s coin, which follows a commitment it holds"),
    ("the header's coin with no place", (ROUND_0, HEADER_0), held(), {ROUND: 2},
     "released: the verifier names no place of " + HEADER + "'s coin, which follows a commitment it holds"),
    ("the message's coin with no place", (ROUND_0, MESSAGE_0), held(), {ROUND: 2},
     "released: the verifier names no place of " + MESSAGE + "'s coin, which follows a commitment it holds"),
    ("a commitment placed at the message's coin's place", (MESSAGE_0,), held(place=2), {MESSAGE: 2},
     f"E_SUPPLIED_ORDER: ['commitment {MESSAGE}'] registered after {MESSAGE}, which a supplied value reads"),
]
#: the same marks in an order that holds: the commitment's place before the coin's
ORDER_HOLDS = [("a commitment before its coin", (ROUND_0, HEADER_0), held(place=0), {ROUND: 1, HEADER: 1}),
               ("a commitment before the message's coin", (MESSAGE_0,), held(place=0), {MESSAGE: 1}),
               ("a commitment held and no coin read", (CONTEXT_0,), held(place=7), {})]


def order_of(marks, tables, released):
    coins, commitments = tables
    marked = {f"x{i}": mark(fn, **args) for i, (fn, args) in enumerate(marks)}
    G.order(marked, {}, {f: b for f, b in coins.items()}, commitments, released, 0)


@pytest.mark.parametrize("name, marks, tables, released, why", ORDER_REFUSALS, ids=[r[0] for r in ORDER_REFUSALS])
def test_a_commitment_must_precede_the_coin_it_is_read_beside(name, marks, tables, released, why):
    with pytest.raises(ValueError) as e:
        order_of(marks, tables, released)
    assert str(e.value) == why


@pytest.mark.parametrize("name, marks, tables, released", ORDER_HOLDS, ids=[r[0] for r in ORDER_HOLDS])
def test_the_order_that_holds_is_accepted(name, marks, tables, released):
    order_of(marks, tables, released)


def test_a_coin_with_no_commitment_held_has_nothing_to_follow():
    """The order rule alone (the round coin's body refuses what it holds no commitment of, which the program's check does)."""
    order_of((ROUND_0,), ({ROUND: held()[0][ROUND]}, {}), {})


def test_an_unread_commitment_or_coin_has_no_place_to_hold():
    """Only the functions a mark names count: the held ones the program doesn't read need no place."""
    order_of((CONTEXT_0,), held(place=None), {})
    order_of((CONTEXT_0,), held(place=None), {ROUND: 0})


def _anchor(of, table):
    return {"supplied": f"registered.{of}/v1", "args": {"value": table}, "layout": "digest", "bits": 512}


def _table_reads(supplied):
    from verity.circuits.defs import CompositeDefinition, bind
    from verity.circuits.program import Program
    from verity.circuits.types import Array, Value

    d, w = Array(512, Value(1)), Array(1, Value(16))
    sig = (("v", w), ("u", w), ("t1/domain", d), ("t1/root", d), ("t2/root", d), ("k", w))
    fn = bind(CompositeDefinition("TRegisteredValues", 1, (), lambda S: (sig, w), lambda B, S, v, u, *rest: v,
                                  register=False))
    return Program(fn, registered=("v", "u"), supplied=supplied)


@pytest.mark.parametrize("supplied, want", [
    ({}, ("v", "u")),
    ({"t1/domain": _anchor("domain", "t1"), "t1/root": _anchor("root", "t1")}, ("v", "u", "t1")),
    ({"t1/root": _anchor("root", "t1"), "t2/root": _anchor("root", "t2")}, ("v", "u", "t1", "t2")),
    ({"t1/root": _anchor("root", "t1"), "k": {"supplied": "const.u16/v1", "args": {"value": 1}, "layout": "le", "bits": 16}},
     ("v", "u", "t1")),
])
def test_a_records_values_are_the_writers_then_each_table_an_anchors_mark_names_once(supplied, want):
    """`registered_values`: the writer's values in its order, then each table a `registered.root/v1` or
    `registered.domain/v1` mark names, once, in the marks' order; another mark adds nothing; the same from the Program and
    from its description encoded and decoded."""
    from verity.circuits.codec import decode_program, encode_program
    from verity.circuits.defs import DefinitionTable

    prog = _table_reads(supplied)
    assert S.registered_values(prog, ("v", "u")) == want
    back = decode_program(json.loads(json.dumps(encode_program(prog))), DefinitionTable())
    assert S.registered_values(back, ("v", "u")) == want
    with pytest.raises(ValueError, match="each once"):
        S.registered_values(prog, ("v", "v"))
