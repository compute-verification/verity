"""`verity.circuits.supplied` (PROTOCOL.md §5.4): an `inputs` entry's supplied mark holds of its nodes for every short
value a verifier makes today (the storage session's round coin, header and context, the storage setup's salt, from which
each unit builds its own base, the bandwidth audit's pass seed and pass root, and the salts, the log's base and the operands'
roots a seed reads), with no case per scheme; the registry names the Lean registry's functions;
the decoder's refusals each name their code; an output held to a supplied constant is checked once; a block's part of a
supplied input keeps C's mark; and an evaluation that reads a commitment registered after its coin is refused by name."""
from __future__ import annotations

import copy
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
    (_entry("one", "const.u16/v1", {"value": 1}, "le", 16), [(S.SUPPLIED, 16, 1)]),
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
    (_entry("k", "const.u16/v1", {"value": 1 << 16}, "le", 16), [(S.SUPPLIED, 16, 1)], "E_SUPPLIED_ARGS"),
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
    """Every unit's ok word (a 16-bit output leaf each, 4 units) held to the one leaf of `const.u16/v1` at 1, as an
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


@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.xfail(strict=True, reason="no Lean body yet"))
                                  if n in NO_LEAN_BODY else n for n in S.FUNCTIONS])
def test_each_function_has_its_lean_body(name):
    assert name in LEAN_BODY.findall(LEAN_REGISTRY.read_text()), f"{name} has no body in {LEAN_REGISTRY.name}"


#: the storage session's released material and each body's value for it, as the memory attester writes them: the session
#: commitment (`SessionCommitment.to_json`), three coin openings, the registered LMS key, and the values of the
#: round coins, the header and the context
VECTORS = json.loads((Path(__file__).parent / "pous_session_vectors.json").read_text())
ROUND, HEADER, CONTEXT = "pous.session.round_coin/v1", "pous.session.header/v1", "pous.session.context/v1"
RELEASED = {k: VECTORS[k] for k in ("commitment", "openings", "key")}


def held(**over) -> dict[str, bytes]:
    """The verifier's copy of the release under each of the three functions' names, with ``over`` replacing parts of it."""
    blob = json.dumps({**RELEASED, **over}, sort_keys=True, separators=(",", ":")).encode()
    return {ROUND: blob, HEADER: blob, CONTEXT: blob}


def with_commitment(**fields) -> dict[str, bytes]:
    return held(commitment={**RELEASED["commitment"], **fields})


def mark(fn: str, **args) -> dict:
    return {"supplied": fn, "args": args, "layout": "le", "bits": 8}


def refusal(coins, fn: str, **args) -> str:
    with pytest.raises(ValueError) as e:
        G.body(coins, "x", mark(fn, **args))
    return str(e.value)


@pytest.mark.parametrize("j", [0, 1, 2])
def test_round_coin_vectors(j):
    got = G.body(held(), "x", mark(ROUND, round=j))
    assert got == ([("bytes", VECTORS["round_coin"][j])], 12)
    assert got[0][0][1] == j.to_bytes(4, "big").hex() + VECTORS["openings"][j]["index"].to_bytes(8, "big").hex()


def test_header_vector():
    row = bytes.fromhex(VECTORS["header"])
    assert len(row) == 754 and VECTORS["head"] == 753 and row.endswith(b"}\x00")
    assert G.body(held(), "x", mark(HEADER, head=753)) == ([("bytes", VECTORS["header"])], 754)


def test_header_of_an_even_length_has_no_padding():
    c = with_commitment(limit_ns=85400)
    row, n = G.body(c, "x", mark(HEADER, head=752))
    assert n == 752 and bytes.fromhex(row[0][1]).endswith(b"}")
    doc = json.loads(bytes.fromhex(row[0][1]))
    assert doc["ended"] == "complete" and doc["rounds"] == doc["k"] == 3 and doc["limit_ns"] == 85400
    assert doc["coin_key_sha512"] == hashlib.sha512(bytes.fromhex(RELEASED["commitment"]["coin_key"])).hexdigest()


def test_context_vector():
    assert G.body(held(), "x", mark(CONTEXT)) == ([("bytes", VECTORS["context"])], 106)


def test_the_marks_reach_the_leaves_their_layouts_read():
    rows = G.of_mark(held(), "x", {"supplied": ROUND, "args": {"round": 2}, "layout": "halves", "bits": 96})
    assert G.row_leaves_of(rows[0], lambda _k, n: bytes.fromhex(n), 32, False) == [2, 0, 70000]
    rows = G.of_mark(held(), "x", {"supplied": CONTEXT, "args": {}, "layout": "le", "bits": 848})
    assert len(G.row_leaves_of(rows[0], lambda _k, n: bytes.fromhex(n), 16, False)) == 53


def _tamper(path, f):
    doc = copy.deepcopy(RELEASED)
    f(doc)
    return held(**doc)


def _flip(h: str) -> str:
    return ("1" if h[0] == "0" else "0") + h[1:]


def _opening(doc, field, value, j=0):
    next(o for o in doc["openings"] if o["j"] == j)[field] = value


NO_STATEMENT = "supplied input x: no statement: "


@pytest.mark.parametrize("name, coins, fn, args, why", [
    ("no release held", {}, ROUND, {"round": 0}, "the verifier holds no coin for " + ROUND),
    ("a release that is no JSON", {ROUND: b"{"}, ROUND, {"round": 0}, ROUND + "'s released session is no JSON"),
    ("a release that is not UTF-8", {ROUND: b"\xff"}, ROUND, {"round": 0}, ROUND + "'s released session is no JSON"),
    ("a release that is no object", {ROUND: b"[]"}, ROUND, {"round": 0}, ROUND + "'s released session is no object"),
    ("no commitment", held(commitment=None), ROUND, {"round": 0}, ROUND + "'s released session holds no commitment object"),
    ("a session in capitals", with_commitment(session=VECTORS["commitment"]["session"].upper()), ROUND, {"round": 0},
     ROUND + "'s commitment's session is no lowercase hex"),
    ("a k that is no natural", with_commitment(k=-1), ROUND, {"round": 0}, ROUND + "'s commitment's k is no natural"),
    ("a k that is a boolean", with_commitment(k=True), ROUND, {"round": 0}, ROUND + "'s commitment's k is no natural"),
    ("a round that is no u32", held(), ROUND, {"round": 1 << 32}, ROUND + "'s round 4294967296 is no u32"),
    ("a round past k", held(), ROUND, {"round": 3}, ROUND + "'s round 3 is no round of the session's 3"),
    ("no list of openings", held(openings={}), ROUND, {"round": 0}, ROUND + "'s released session holds no openings list"),
    ("no opening of the round", held(openings=RELEASED["openings"][1:]), ROUND, {"round": 0},
     ROUND + "'s released session holds no opening of round 0"),
    ("an opening without a path", _tamper(0, lambda d: d["openings"][0].pop("path")), ROUND, {"round": 0},
     ROUND + "'s opening of round 0 is malformed"),
    ("an index that is no u64", _tamper(0, lambda d: _opening(d, "index", 1 << 64)), ROUND, {"round": 0},
     ROUND + "'s round 0's index 18446744073709551616 is no u64"),
    ("an index past the blocks", with_commitment(blocks=5), ROUND, {"round": 0},
     ROUND + "'s round 0's index 5 is no block of the session's 5"),
    ("another index than the committed coin", _tamper(0, lambda d: _opening(d, "index", 6)), ROUND, {"round": 0},
     ROUND + "'s round 0's index 6 is not its committed coin"),
    ("another salt", _tamper(0, lambda d: _opening(d, "salt", _flip(RELEASED["openings"][0]["salt"]))), ROUND,
     {"round": 0}, ROUND + "'s round 0's index 5 is not its committed coin"),
    ("a salt of the wrong length", _tamper(0, lambda d: _opening(d, "salt", RELEASED["openings"][0]["salt"][2:])), ROUND,
     {"round": 0}, ROUND + "'s round 0's index 5 is not its committed coin"),
    ("a short path", _tamper(0, lambda d: _opening(d, "path", RELEASED["openings"][0]["path"][:1])), ROUND, {"round": 0},
     ROUND + "'s round 0's index 5 is not its committed coin"),
    ("another coin root", with_commitment(coin_root=_flip(RELEASED["commitment"]["coin_root"])), ROUND, {"round": 0},
     ROUND + "'s round 0's index 5 is not its committed coin"),
    ("another key", with_commitment(coin_key=RELEASED["commitment"]["coin_key"][:-2]), ROUND, {"round": 0},
     ROUND + "'s round 0's index 5 is not its committed coin"),
    ("a key with its top bit set", with_commitment(coin_key=RELEASED["commitment"]["coin_key"][:-2] + "ff"), ROUND,
     {"round": 0}, ROUND + "'s round 0's index 5 is not its committed coin"),
    ("a header of another length", held(), HEADER, {"head": 752}, HEADER + "'s header is 753 bytes, not 752"),
    ("a header of a binding that is no hex", with_commitment(binding="zz"), HEADER, {"head": 753},
     HEADER + "'s commitment's binding is no lowercase hex"),
    ("a header of a limit that is no natural", with_commitment(limit_ns="854000"), HEADER, {"head": 753},
     HEADER + "'s commitment's limit_ns is no natural"),
    ("a key of 15 bytes", held(key="ab" * 15), CONTEXT, {}, CONTEXT + "'s key is 15 bytes, not an LMS identifier's 16"),
    ("a key that is no hex", held(key="x"), CONTEXT, {}, CONTEXT + "'s released key is no lowercase hex"),
    ("a session of 31 bytes", with_commitment(session="ab" * 31), CONTEXT, {}, CONTEXT + "'s session is 31 bytes, not 32"),
])
def test_what_has_no_statement_is_refused_by_name(name, coins, fn, args, why):
    assert refusal(coins, fn, **args) == NO_STATEMENT + why


def test_a_last_opening_stands_for_its_round():
    """`session_proofs.clear_reasons` reads the last opening of a round: an earlier, wrong one does not count."""
    bad = {**RELEASED["openings"][0], "index": 6}
    got = G.body(held(openings=[bad, *RELEASED["openings"]]), "x", mark(ROUND, round=0))
    assert got == ([("bytes", VECTORS["round_coin"][0])], 12)
    assert refusal(held(openings=[*RELEASED["openings"], bad]), ROUND, round=0).endswith("index 6 is not its committed coin")


def test_the_arguments_each_function_takes():
    assert refusal(held(), ROUND) == "supplied input x: " + ROUND + " takes the args [round]"
    assert refusal(held(), HEADER, round=0) == "supplied input x: " + HEADER + " takes the args [head]"
    assert refusal(held(), CONTEXT, head=1) == "supplied input x: " + CONTEXT + " takes no args"
