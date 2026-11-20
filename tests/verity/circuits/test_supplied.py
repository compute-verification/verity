"""`verity.circuits.supplied` (PROTOCOL.md §5.4): an `inputs` entry's supplied mark holds of its nodes for every short
value a verifier makes today (the storage session's round coin, header and context, the storage setup's salt, from which
each unit builds its own base, the bandwidth audit's pass seed and pass root, and the salts, the log's base and the operands'
roots a seed reads), with no case per scheme; the registry names the Lean registry's functions;
the decoder's refusals each name their code; an output held to a supplied constant is checked once; a block's part of a
supplied input keeps C's mark; and an evaluation that reads a commitment registered after its coin is refused by name."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from verity.circuits import supplied as S
from verity.protocols.proofs.spot.verifier.one_stage import registered as G


def _entry(name, fn, args, layout, bits, n=1):
    return [name, n, {"supplied": fn, "args": args, "layout": layout, "bits": bits}]


#: (entry, its nodes as (binding, width, leaves)) at the deployed widths
SERVED = [
    (_entry("round_3", "pous.session.round_coin/v1", {"round": 3}, "halves", 96), [(S.SUPPLIED, 32, 3)]),
    (_entry("header", "pous.session.header/v1", {"head": 756}, "le", 6048), [(S.SUPPLIED, 1, 6048)]),
    (_entry("context", "pous.session.context/v1", {}, "le", 848), [(S.SUPPLIED, 1, 848)]),
    (_entry("setup_salt", "pous.setup.salt/v1", {}, "le", 192), [(S.SUPPLIED, 16, 12)]),
    (_entry("seed_2", "poub.read.pass_seed/v1", {"window": 0, "pass": 2}, "halves", 512), [(S.SUPPLIED, 32, 16)]),
    (_entry("pass_root_2", "poub.read.pass_root/v1", {"window": 0, "pass": 2}, "digest", 512), [(S.SUPPLIED, 1, 512)]),
    (_entry("salt", "window.salt/v1", {}, "halves", 512), [(S.SUPPLIED, 32, 16)]),
    (_entry("pp_salt", "pouw.pearlplus.salt/v1", {}, "le", 256), [(S.SUPPLIED, 32, 8)]),
    (_entry("root_b", "registered.root/v1", {"value": "weights/b"}, "digest", 512), [(S.SUPPLIED, 1, 512)]),
    (_entry("root_a", "record.root/v1", {"port": "a/0@1"}, "halves", 512), [(S.SUPPLIED, 32, 16)]),
    (_entry("log_base", "log.base/v1", {}, "halves", 64), [(S.SUPPLIED, 16, 4)]),
    (_entry("one", "const.uint/v1", {"value": 1, "bits": 16}, "le", 16), [(S.SUPPLIED, 16, 1)]),
]


@pytest.mark.parametrize("entry,nodes", SERVED, ids=[e[0] for e, _ in SERVED])
def test_each_row_a_verifier_makes_holds_its_mark(entry, nodes):
    assert S.check_entry(entry, nodes) == entry[2]


def test_unmarked_entries_are_unchanged():
    assert S.check_entry(["x", 2], [("run", 16, 4), ("run", 32, 1)]) is None
    assert S.check_entry(["w", 1], [("registration", 16, 8)]) is None


@pytest.mark.parametrize("entry,nodes,code", [
    (["k", 1, {"supplied": "window.salt/v1"}], [(S.SUPPLIED, 32, 16)], "E_SUPPLIED_MARK"),
    (["k", 1, {"supplied": "window.salt/v1", "args": {}, "layout": "bytes", "bits": 512}], [(S.SUPPLIED, 32, 16)],
     "E_SUPPLIED_MARK"),
    (_entry("k", "pous.session.round_coin/v1", {"round": 3}, "halves", 96), [("registration", 32, 3)], "E_INPUTS_MIXED"),
    (_entry("k", "pous.session.round_coin/v1", {"round": 3}, "halves", 96), [(S.SUPPLIED, 32, 1), ("run", 32, 2)],
     "E_INPUTS_MIXED"),
    (["k", 1], [(S.SUPPLIED, 32, 3)], "E_INPUTS_MIXED"),
    (_entry("k", "pous.session.round_key/v1", {"round": 3}, "halves", 64), [(S.SUPPLIED, 32, 2)], "E_SUPPLIED_FN"),
    (_entry("k", "pous.session.round_coin/v1", {}, "halves", 96), [(S.SUPPLIED, 32, 3)], "E_SUPPLIED_ARGS"),
    (_entry("k", "const.uint/v1", {"value": 1 << 16, "bits": 16}, "le", 16), [(S.SUPPLIED, 16, 1)], "E_SUPPLIED_ARGS"),
    (_entry("k", "pous.session.header/v1", {"head": "x"}, "le", 6048), [(S.SUPPLIED, 1, 6048)], "E_SUPPLIED_ARGS"),
    (_entry("k", "pous.session.header/v1", {}, "le", 6048), [(S.SUPPLIED, 1, 6048)], "E_SUPPLIED_ARGS"),
    (_entry("k", "pous.session.round_coin/v1", {"round": 3}, "halves", 64), [(S.SUPPLIED, 32, 2)], "E_SUPPLIED_WIDTH"),
    (_entry("k", "pous.session.round_coin/v1", {"round": 3}, "halves", 96), [(S.SUPPLIED, 32, 2), (S.SUPPLIED, 16, 2)],
     "E_SUPPLIED_WIDTH"),
    (_entry("k", "pous.session.context/v1", {}, "halves", 848), [(S.SUPPLIED, 16, 53)], "E_SUPPLIED_WIDTH"),
    (_entry("k", "registered.root/v1", {"value": "a"}, "digest", 512), [(S.SUPPLIED, 32, 16)], "E_SUPPLIED_WIDTH"),
])
def test_refusals(entry, nodes, code):
    with pytest.raises(S.SuppliedError, match=f"^{code}: ") as e:
        S.check_entry(entry, nodes)
    assert e.value.code == code


def test_an_output_held_to_a_supplied_constant():
    """Every unit's ok word (a 16-bit output leaf each, 4 units) held to the one leaf of `const.uint/v1` at 1 (16 bits), as an
    opening's verdict and the storage setup's ok are."""
    one = S.check_entry(*SERVED[-1])
    inputs = {"one": (one, 16, 1), "setup_salt": (SERVED[3][0][2], 16, 12)}
    S.check_held([10, 14, {"equals": ["one", 0, 1]}], 16, inputs)
    for bad, width in ([[10, 14, {"equals": ["salt", 0, 1]}], 16], [[10, 14, {"equals": ["one", 0, 1]}], 1],
                       [[10, 14, {"equals": ["one", 0, 3]}], 16], [[10, 14, {"equals": ["one", 1, 1]}], 16],
                       [[10, 14, {"interval": [0, 1]}], 16]):
        with pytest.raises(S.SuppliedError, match="^E_HELD: "):
            S.check_held(bad, width, inputs)


def test_a_blocks_part_names_cs_input_and_range():
    """The opening's key reads leaf 2 of round 3's coin (i_j's low half): a whole leaf, so no second function gives i_j."""
    mark = S.check_entry(*SERVED[0])
    assert S.part("round_3", mark, 2, 3, 32) == {**mark, "bits": 32, "of": ["round_3", 2, 3]}
    with pytest.raises(S.SuppliedError, match="^E_SUPPLIED_WIDTH: "):
        S.part("round_3", mark, 2, 4, 32)


def test_a_blocks_part_is_a_mark_its_own_nodes_hold():
    p = S.part("round_3", S.check_entry(*SERVED[0]), 2, 3, 32)
    assert S.check_entry(["round_3/p0", 1, p], [(S.SUPPLIED, 32, 1)]) == p
    with pytest.raises(S.SuppliedError, match="^E_SUPPLIED_WIDTH: "):
        S.check_entry(["round_3/p0", 1, p], [(S.SUPPLIED, 32, 2)])
    with pytest.raises(S.SuppliedError, match="^E_SUPPLIED_MARK: "):
        S.check_entry(["round_3/p0", 1, {**p, "of": ["round_3", 3, 3]}], [(S.SUPPLIED, 32, 1)])


def test_the_order_rule():
    S.check_order({"store": 0, "session": 1}, coin=2)
    with pytest.raises(S.SuppliedError, match=r"^E_SUPPLIED_ORDER: \['late'\] registered after the coin"):
        S.check_order({"store": 0, "late": 2}, coin=2)


def test_every_function_reads_only_public_things():
    for f in S.FUNCTIONS.values():
        assert f.origin in S.ORIGINS and set(f.reads) <= set(S.READS) and f.reads
        assert f.origin == "supplied" or f.reads == ("coin",), f.name


#: the Lean registry: each function a supplied input may name has its body there, and a function exists when it has one
LEAN_REGISTRY = Path(__file__).parents[3] / "verity/protocols/proofs/spot/verifier/lean/Spot/Supplied.lean"
#: a function's body in the Lean registry: an arm of `Supplied.body`'s match on the function's name
LEAN_BODY = re.compile(r'^  \| "([a-z0-9_]+(?:\.[a-z0-9_]+)+/v[1-9][0-9]*)" =>', re.M)
#: the functions the Lean registry has no body of yet; each strict failure below is one to take off this list
NO_LEAN_BODY = frozenset({"poub.read.pass_seed/v1", "poub.read.pass_root/v1"})


def _lean_names() -> set[str]:
    return set(LEAN_BODY.findall(LEAN_REGISTRY.read_text()))


@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.xfail(strict=True, reason="no Lean body yet"))
                                  if n in NO_LEAN_BODY else n for n in S.FUNCTIONS])
def test_each_function_has_its_lean_body(name):
    assert name in _lean_names(), f"{name} has no body in {LEAN_REGISTRY.name}"


def test_the_lean_registry_names_no_function_the_format_lacks():
    assert _lean_names() <= set(S.FUNCTIONS)


#: a function's value at some args, in bits (None: no value there), as the format fixes it (§5.4)
LENGTHS = [("const.uint/v1", {"value": 7, "bits": 32}, 32), ("const.uint/v1", {"value": 1 << 32, "bits": 32}, None),
           ("const.uint/v1", {"value": 1, "bits": 8}, 8), ("const.uint/v1", {"value": 3, "bits": 64}, 64),
           ("const.uint/v1", {"value": 5, "bits": 12}, None), ("const.uint/v1", {"value": 0, "bits": 72}, None),
           ("const.uint/v1", {"value": 0, "bits": 0}, None),
           ("pouw.pearlplus.salt/v1", {}, 256),
           ("registered.root/v1", {"value": "w"}, 512)]


@pytest.mark.parametrize(("name", "args", "bits"), LENGTHS)
def test_a_functions_length(name, args, bits):
    n = S.FUNCTIONS[name].nbytes(args)
    assert (None if n is None else 8 * n) == bits


#: the Lean decoder's closed list of the functions a supplied input may name (`SUPPLIED_FUNCTIONS`)
DECODER = Path(__file__).parents[3] / "verity/circuits/lean/Circuits/Descriptor.lean"


def test_the_decoder_lists_the_format_s_functions():
    """The decoder's list names exactly `supplied.FUNCTIONS`, each with its args; `decode_difftest`'s marks hold the two
    lists' lengths equal."""
    text = DECODER.read_text()
    block = text[text.index("def SUPPLIED_FUNCTIONS"):text.index("def SUPPLIED_LAYOUTS")]
    got = {m[1]: tuple(re.findall(r'"([^"]+)"', m[2])) for m in re.finditer(r'⟨"([^"]+)", \[([^\]]*)\]', block)}
    want = {f.name: f.args for f in S.FUNCTIONS.values()}
    assert got == want, f"only the decoder's: {sorted(got.keys() - want)}; only the codec's: {sorted(want.keys() - got)}"


#: the storage session's released material and each body's value for it, as the memory attester writes them: the session
#: commitment (`SessionCommitment.to_json`), three coin openings, the registered LMS key, and the values of the round coins,
#: the header and the context
VECTORS = json.loads((Path(__file__).parent / "pous_session_vectors.json").read_text())
ROUND, HEADER, CONTEXT = "pous.session.round_coin/v1", "pous.session.header/v1", "pous.session.context/v1"
#: what the verifier holds of the session as a released commitment: the commitment, how the session ended and its rounds, the key,
#: and the place the verifier registered it at (before the place its coin is released at, 1)
RELEASE = {"commitment": VECTORS["commitment"], "ended": "complete", "rounds": 3, "key": VECTORS["key"], "place": 0}


def text(doc) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"))


def held(commitment=None, coin=None, **over) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """The verifier's ``(coins, commitments)``: the commitment held under each of the three functions' names, ``over`` replacing
    parts of it, and the round coin (the openings) under the round coin's; ``commitment`` and ``coin`` give a text as it is."""
    blob = (commitment if commitment is not None else text({**RELEASE, **over})).encode()
    return ({ROUND: (coin if coin is not None else text({"openings": VECTORS["openings"]})).encode()},
            {ROUND: blob, HEADER: blob, CONTEXT: blob})


def with_commitment(**fields):
    return held(commitment=text({**RELEASE, "commitment": {**RELEASE["commitment"], **fields}}))


def mark(fn: str, **args) -> dict:
    return {"supplied": fn, "args": args, "layout": "le", "bits": 8}


def refusal(tables, fn: str, **args) -> str:
    with pytest.raises(ValueError) as e:
        G.body(*tables, "x", mark(fn, **args))
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
    c = RELEASE["commitment"]
    row, n = G.body(*with_commitment(limit_ns=85400), "x", mark(HEADER, head=752))
    assert n == 752 and bytes.fromhex(row[0][1]).endswith(b"}")
    doc = json.loads(bytes.fromhex(row[0][1]))
    assert doc["ended"] == "complete" and doc["rounds"] == doc["k"] == 3 and doc["limit_ns"] == 85400
    assert doc["coin_key_sha512"] == hashlib.sha512(bytes.fromhex(c["coin_key"])).hexdigest()


def test_context_vector():
    assert G.body(*held(), "x", mark(CONTEXT)) == ([("bytes", VECTORS["context"])], 106)


def test_the_marks_reach_the_leaves_their_layouts_read():
    rows = G.of_mark(*held(), "x", {"supplied": ROUND, "args": {"round": 2}, "layout": "halves", "bits": 96})
    assert G.row_leaves_of(rows[0], lambda _k, n: bytes.fromhex(n), 32, False) == [2, 0, 70000]
    rows = G.of_mark(*held(), "x", {"supplied": CONTEXT, "args": {}, "layout": "le", "bits": 848})
    assert len(G.row_leaves_of(rows[0], lambda _k, n: bytes.fromhex(n), 16, False)) == 53


def _flip(h: str, at: int = 0) -> str:
    return h[:at] + ("1" if h[at] == "0" else "0") + h[at + 1:]


def _openings(**change):
    return [{**o, **change} if o["j"] == 0 else o for o in VECTORS["openings"]]


def coin_of(openings, **more) -> str:
    return text({"openings": openings, **more})


def raw(old: str, new: str) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """The commitment's text with ``old`` replaced by ``new``, where it is in it."""
    t = text(RELEASE)
    assert t.count(old) == 1, old
    return held(commitment=t.replace(old, new))


def raw_coin(old: str, new: str):
    t = text({"openings": VECTORS["openings"]})
    assert t.count(old) == 1, old
    return held(coin=t.replace(old, new))


NO_STATEMENT = "supplied input x: no statement: "
CANON = "'s commitment holds a number that is no canonical natural"
COIN_CANON = "'s coin holds a number that is no canonical natural"
R0 = {"round": 0}
BAD_COIN = ROUND + "'s round 0's index 5 is not its committed coin"

#: each refusal of the three bodies, with its text: the one place the Lean's refusals are read from, as the difference test
#: holds each to it
REFUSALS = [
    ("no commitment held", ({}, {}), ROUND, R0, "the verifier holds no commitment for " + ROUND),
    ("no coin held", ({}, held()[1]), ROUND, R0, "the verifier holds no coin for " + ROUND),
    ("a commitment that is no JSON", held(commitment="{"), ROUND, R0, ROUND + "'s commitment is no JSON"),
    ("a commitment that is not UTF-8", ({}, {ROUND: b"\xff"}), ROUND, R0, ROUND + "'s commitment is no JSON"),
    ("a commitment that is no object", held(commitment="[]"), ROUND, R0, ROUND + "'s commitment is no object"),
    ("no commitment object", held(commitment=text({**RELEASE, "commitment": 5})), ROUND, R0,
     ROUND + "'s commitment holds no commitment object"),
    ("a session in capitals", with_commitment(session=VECTORS["commitment"]["session"].upper()), ROUND, R0,
     ROUND + "'s commitment's session is no lowercase hex"),
    ("a k that is negative", with_commitment(k=-1), ROUND, R0, ROUND + CANON),
    ("a k that is a boolean", with_commitment(k=True), ROUND, R0, ROUND + "'s commitment's k is no u32"),
    ("a k past u32", with_commitment(k=1 << 32), ROUND, R0, ROUND + "'s commitment's k is no u32"),
    ("blocks past u64", with_commitment(blocks=1 << 64), ROUND, R0, ROUND + "'s commitment's blocks is no u64"),
    ("a round that is no u32", held(), ROUND, {"round": 1 << 32}, ROUND + "'s round 4294967296 is no u32"),
    ("a round past k", held(), ROUND, {"round": 3}, ROUND + "'s round 3 is no round of the session's 3"),
    ("a coin of more than its openings", held(coin=coin_of(VECTORS["openings"], coin_root="00" * 64)), ROUND, R0,
     ROUND + "'s coin holds more than its openings: the coin root is the verifier's own"),
    ("a coin that carries a commitment", held(coin=coin_of(VECTORS["openings"], commitment=RELEASE["commitment"])), ROUND, R0,
     ROUND + "'s coin holds more than its openings: the coin root is the verifier's own"),
    ("a coin that is no list of openings", held(coin=text({"openings": {}})), ROUND, R0, ROUND + "'s coin holds no openings list"),
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
    ("another key held", with_commitment(coin_key=VECTORS["commitment"]["coin_key"][:-2]), ROUND, R0, BAD_COIN),
    ("a key with its top bit set", with_commitment(coin_key=VECTORS["commitment"]["coin_key"][:-2] + "ff"), ROUND, R0, BAD_COIN),
    ("a header of another length", held(), HEADER, {"head": 752}, HEADER + "'s header is 753 bytes, not 752"),
    ("a header of a binding that is no hex", with_commitment(binding="zz"), HEADER, {"head": 753},
     HEADER + "'s commitment's binding is no lowercase hex"),
    ("a header of a limit that is a string", with_commitment(limit_ns="854000"), HEADER, {"head": 753},
     HEADER + "'s commitment's limit_ns is no u64"),
    ("a session that ended late", held(ended="late"), HEADER, {"head": 753}, HEADER + "'s session did not end complete"),
    ("a session with no ended", held(ended=None), HEADER, {"head": 753}, HEADER + "'s commitment holds no ended string"),
    ("a session of 2 rounds", held(rounds=2), HEADER, {"head": 753}, HEADER + "'s session has 2 rounds, not its 3"),
    ("a session of 4 rounds", held(rounds=4), HEADER, {"head": 753}, HEADER + "'s session has 4 rounds, not its 3"),
    ("a session of rounds past u32", held(rounds=1 << 32), HEADER, {"head": 753}, HEADER + "'s commitment's rounds is no u32"),
    ("a late session of too few rounds", held(ended="late", rounds=1), HEADER, {"head": 753}, HEADER + "'s session did not end complete"),
    ("a key of 15 bytes", held(key="ab" * 15), CONTEXT, {}, CONTEXT + "'s key is 15 bytes, not an LMS identifier's 16"),
    ("a key that is no hex", held(key="x"), CONTEXT, {}, CONTEXT + "'s commitment's key is no lowercase hex"),
    ("a session of 31 bytes", with_commitment(session="ab" * 31), CONTEXT, {}, CONTEXT + "'s session is 31 bytes, not 32"),
    # numbers: one strict reader, in every body
    ("a k of 1e0", raw('"k":3,', '"k":3e0,'), ROUND, R0, ROUND + CANON),
    ("a k with a leading zero", raw('"k":3,', '"k":03,'), ROUND, R0, ROUND + "'s commitment is no JSON"),
    ("a k with a sign", raw('"k":3,', '"k":+3,'), ROUND, R0, ROUND + "'s commitment is no JSON"),
    ("a k of -0", raw('"k":3,', '"k":-0,'), ROUND, R0, ROUND + CANON),
    ("a k with a fraction", raw('"k":3,', '"k":3.0,'), ROUND, R0, ROUND + CANON),
    ("blocks of 1e2 (read by the header)", raw('"blocks":1048576', '"blocks":1e2'), HEADER, {"head": 753}, HEADER + CANON),
    ("a limit with a leading zero", raw('"limit_ns":854000', '"limit_ns":0854000'), HEADER, {"head": 753},
     HEADER + "'s commitment is no JSON"),
    ("rounds of 3.0", raw('"rounds":3', '"rounds":3.0'), HEADER, {"head": 753}, HEADER + CANON),
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
    doc = {**RELEASE, "note": "limit -0 and 1e2, 03 + 3.0 \\\" 5e1"}
    assert G.body(*held(commitment=text(doc)), "x", mark(CONTEXT)) == ([("bytes", VECTORS["context"])], 106)


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


def test_the_header_is_compared_with_what_was_released_not_assumed():
    """A header the verifier computes is the complete one: a release that says otherwise makes no statement, and one that
    says the same gives the vector."""
    assert G.body(*held(ended="complete", rounds=3), "x", mark(HEADER, head=753))[1] == 754
    for over in ({"ended": "late"}, {"ended": ""}, {"rounds": 0}, {"rounds": 4}):
        assert "no statement" in refusal(held(**over), HEADER, head=753)


def test_the_coin_root_is_the_verifiers_never_the_cois():
    """A coin that brings its own coin root and key (another session's, with openings that open under it) is refused by name,
    and the verifier's own root refuses the openings of a root it didn't commit."""
    other = {**VECTORS["commitment"], "coin_root": _flip(VECTORS["commitment"]["coin_root"])}
    assert refusal(held(coin=coin_of(VECTORS["openings"], commitment=other)), ROUND, round=0).endswith(
        "holds more than its openings: the coin root is the verifier's own")
    assert refusal(with_commitment(coin_root=other["coin_root"]), ROUND, round=0).endswith("is not its committed coin")


HEADER_0 = (HEADER, {"head": 753})
ROUND_0 = (ROUND, {"round": 0})
CONTEXT_0 = (CONTEXT, {})
LATE = "E_SUPPLIED_ORDER: ['commitment pous.session.header/v1', 'commitment pous.session.round_coin/v1'] registered after "
#: each refusal of the order rule over a session read by the round coin and the header: (name, the marks, the verifier's tables,
#: the places `released` gives the coins, the text): a commitment after its coin, at its coin's place, with no place, beside a
#: coin with none
ORDER_REFUSALS = [
    ("a commitment placed after its coin", (ROUND_0, HEADER_0), held(place=5), {ROUND: 2}, LATE + ROUND + ", which a supplied value reads"),
    ("a commitment placed at its coin's place", (ROUND_0, HEADER_0), held(place=2), {ROUND: 2}, LATE + ROUND + ", which a supplied value reads"),
    ("a commitment with no place", (ROUND_0, HEADER_0), held(place=None), {ROUND: 2},
     "released: the verifier names no place of " + HEADER + "'s commitment, which a coin it holds follows"),
    ("a commitment whose place is no natural", (ROUND_0, HEADER_0), held(place="0"), {ROUND: 2},
     "released: the verifier names no place of " + HEADER + "'s commitment, which a coin it holds follows"),
    ("a coin with no place", (ROUND_0, HEADER_0), held(), {},
     "released: the verifier names no place of " + ROUND + "'s coin, which follows a commitment it holds"),
]
#: the same marks in an order that holds: the commitment's place before the coin's
ORDER_HOLDS = [("a commitment before its coin", (ROUND_0, HEADER_0), held(place=0), {ROUND: 1}),
               ("a commitment held and no coin read", (HEADER_0,), held(place=7), {})]


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
    order_of((HEADER_0,), held(place=None), {})
    order_of((CONTEXT_0,), held(place=None), {ROUND: 0})
