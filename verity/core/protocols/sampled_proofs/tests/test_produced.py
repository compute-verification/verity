"""Produced values (`one_stage.produced`): a value committed after the draw under a domain bound to the registration's
digest, the verifier's own entries and ports for it, the post-draw record's check, and the opened roots held to the
record."""
from __future__ import annotations

import pytest

from verity.core.protocols.sampled_proofs.one_stage import audit
from verity.core.protocols.sampled_proofs.one_stage import produced as PR
from verity.core.protocols.sampled_proofs.one_stage import registered as G

PD = "12" * 64
REG = "ab" * 64
OTHER_REG = "cd" * 64
DRAW = "ef" * 64
ROWS = [[1, 2], [3, 4], [5, 6]]


def test_the_produced_tree_opens_under_its_registrations_domain_only():
    tree = PR.commit(PD, "partials", ROWS, REG)
    mine = PR.entry(PD, "partials", 3, 2, REG)
    assert tree.entry() == {"value": "partials", "root": tree.root.hex(), **{k: mine[k] for k in ("leaves", "words", "domain")}}
    dom = PR.domain(PD, "partials", 2, 3, 16, REG)
    for r in range(3):
        assert G.verify_read(dom, tree.root, r, tree.bcs[r], tree.path(r))
        assert not G.verify_read(PR.domain(PD, "partials", 2, 3, 16, OTHER_REG), tree.root, r, tree.bcs[r], tree.path(r))
        assert not G.verify_read(G.domain(PD, "partials", 2, 3), tree.root, r, tree.bcs[r], tree.path(r))
    assert not G.verify_read(dom, tree.root, 0, tree.bcs[1], tree.path(0)), "a swapped row opens at no other position"
    other = PR.commit(PD, "partials", [[1, 2], [3, 4], [5, 7]], REG)
    assert not G.verify_read(dom, tree.root, 2, other.bcs[2], other.path(2)), "another tree's row opens under no root but its own"
    with pytest.raises(ValueError, match="registration digest"):
        PR.domain(PD, "partials", 2, 3, 16, "AB" * 64)
    with pytest.raises(ValueError, match="not both"):
        G.domain(PD, "partials", 2, 3, 16, window="00" * 64, produced=REG)


def test_a_32_bit_entry_and_its_port():
    tree = PR.commit(PD, "partials", ROWS, REG, word_bits=32)
    e = PR.entry(PD, "partials", 3, 2, REG, word_bits=32)
    assert tree.entry() == {"value": "partials", "root": tree.root.hex(), "leaves": 3, "words": 2, "word_bits": 32,
                            "domain": e["domain"]}
    assert e["domain"] != PR.entry(PD, "partials", 3, 2, REG)["domain"]
    port = PR.port("partials", e, tree.root.hex(), [None, 2, None], G.SCHEMA_V2)
    assert port == {"value": "partials", "root": tree.root.hex(), "leaves": 3, "words": 2, "word_bits": 32,
                    "domain": e["domain"], **G.schema_entry(G.SCHEMA_V2), "reads": [None, 2, None]}


def _rec(registration=REG):
    tree = PR.commit(PD, "partials", ROWS, registration)
    return PR.record(REG, DRAW, [tree.entry()]), tree


def test_the_record_is_held_to_the_registration_the_draw_and_the_verifiers_entries():
    mine = {"partials": PR.entry(PD, "partials", 3, 2, REG)}
    rec, _tree = _rec()
    assert PR.check(rec, REG, DRAW, mine) == ()
    e = rec["produced"][0]
    other, _ = _rec(OTHER_REG)
    cases = [
        ({**rec, "x": 1}, "P-keys"), ({**rec, "format": "v1"}, "P-format"), ({**rec, "registration": OTHER_REG}, "P-registration"),
        ({**rec, "draw": REG}, "P-draw"), ({**rec, "produced": {}}, "P-produced"), ({**rec, "produced": [{**e, "x": 1}]}, "P-produced"),
        ({**rec, "produced": []}, "P-values"), ({**rec, "produced": [e, e]}, "P-values"),
        ({**rec, "produced": [{**e, "value": "q"}]}, "P-values"), ({**rec, "produced": [{**e, "root": "AB" * 64}]}, "P-root"),
        ({**rec, "produced": [{**e, "leaves": 4}]}, "P-leaves"), ({**rec, "produced": [{**e, "leaves": True}]}, "P-leaves"),
        ({**rec, "produced": [{**e, "words": 3}]}, "P-words"), ({**rec, "produced": [{**e, "word_bits": 16}]}, "P-word_bits"),
        ({**rec, "produced": [{**e, "word_bits": 32}]}, "P-word_bits"),
        ({**rec, "produced": other["produced"]}, "P-domain"), (None, "P-keys")]
    for bad, code in cases:
        assert code in PR.check(bad, REG, DRAW, mine), (bad, code)


def test_the_opened_roots_are_the_records():
    rec, tree = _rec()
    v = lambda root: {"lean": {"of_record": True, "accepted": True,
                               "registered": {"partials": {"value": "partials", "root": root},
                                              "p0": {"value": "partials", "root": root}}}}
    assert PR.opened(rec, audit.session_registered(v(tree.root.hex()))) == ()
    assert PR.opened(rec, audit.session_registered(v("00" * 64))) == ("P-opened",)
    assert PR.opened(rec, audit.session_registered({})) == ("P-opened",), "a produced value no verifier of record opened"
    two = {**v(tree.root.hex()), "lean2": {**v("00" * 64)["lean"]}}
    assert PR.opened(rec, audit.session_registered(two)) == ("P-opened",), "two roots for one value"
