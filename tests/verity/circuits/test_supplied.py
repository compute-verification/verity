"""`verity.circuits.supplied` (PROTOCOL.md §5.4): an `inputs` entry's supplied mark holds of its nodes for every short
value a verifier makes today (the storage session's round coin, header and context, the storage setup's salt, from which
each unit builds its own base, the bandwidth audit's pass seed and pass root, and the salts, the log's base and the operands'
roots a seed reads), with no case per scheme; the registry names the Lean registry's functions;
the decoder's refusals each name their code; an output held to a supplied constant is checked once; a block's part of a
supplied input keeps C's mark; and an evaluation that reads a commitment registered after its coin is refused by name."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from verity.circuits import supplied as S


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
#: a function's name as the Lean registry spells it: a string literal of the name, as in `supplied.FUNCTIONS`
LEAN_NAME = re.compile(r'"([a-z0-9_]+(?:\.[a-z0-9_]+)+/v[1-9][0-9]*)"')
#: the functions the Lean registry has no body of yet; each strict failure below is one to take off this list
NO_LEAN_BODY = frozenset(S.FUNCTIONS)


def _lean_names() -> set[str]:
    return set(LEAN_NAME.findall(LEAN_REGISTRY.read_text()))


@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.xfail(strict=True, reason="no Lean body yet"))
                                  if n in NO_LEAN_BODY else n for n in S.FUNCTIONS])
def test_each_function_has_its_lean_body(name):
    assert name in _lean_names(), f"{name} has no body in {LEAN_REGISTRY.name}"


def test_the_lean_registry_names_no_function_the_format_lacks():
    assert _lean_names() <= set(S.FUNCTIONS)


#: a function's value at some args, in bits (None: no value there), as the format fixes it (§5.4)
LENGTHS = [("const.u32/v1", {"value": 7}, 32), ("const.u32/v1", {"value": 1 << 32}, None),
           ("pouw.pearlplus.eb/v1", {"col": 0, "tn": 3}, 768), ("pouw.pearlplus.eb/v1", {"col": 1, "tn": 0}, None),
           ("pouw.pearlplus.fa/v1", {"k": 2}, 512), ("pouw.pearlplus.fb/v1", {"k": 5}, 1280),
           ("pouw.pearlplus.gram/v1", {"k": 9, "dev": 3}, 32768), ("pouw.pearlplus.salt/v1", {}, 256),
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
