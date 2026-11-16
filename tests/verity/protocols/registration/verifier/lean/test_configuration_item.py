"""The registration log's `configuration` item in Python (`verity.protocols.registration.verifier.registration`,
`Core.Registration.Item.configuration` in `RegistrationLog.lean`): the reader takes it, the check refuses it by name
(`configuration-unordered`), its writer is not written yet, and a log of the other items hashes as it did before the kind
was added."""

from __future__ import annotations

import json

import pytest

from verity.device import Auditor, Malformed
from verity.protocols.registration.verifier import registration as G


def _hex(c: str) -> str:
    return c * 128


ITEMS = ({"checkpoint_sha512": _hex("c"), "domain": _hex("b"), "kind": "theta", "leaves": 4, "root": _hex("a"),
          "value": "weights", "words": 64},
         {"domain": _hex("e"), "kind": "theta", "leaves": 1, "root": _hex("d"), "value": "state", "word_bits": 32,
          "words": 8},
         {"advice": "verity/advice/v1", "cap": 32, "digest": _hex("f"), "kind": "builder"})
CONFIGURATION = {"digest": _hex("9"), "kind": "configuration"}


def test_the_reader_takes_a_configuration_item_and_nothing_beside_its_digest():
    assert G.item_of(CONFIGURATION) == CONFIGURATION
    with pytest.raises(Malformed):
        G.item_of({**CONFIGURATION, "extra": 1})
    with pytest.raises(Malformed):
        G.item_of({**CONFIGURATION, "digest": "ab"})


@pytest.mark.parametrize("items", [(CONFIGURATION,), (*ITEMS, CONFIGURATION)])
def test_the_check_refuses_a_configuration_item_by_name(items):
    log = G.chain("onsite", [G.Information(items)])
    with pytest.raises(G.Refused) as r:
        G.check(Auditor(), "onsite", log)
    assert (r.value.entry, r.value.name) == (0, "configuration-unordered")
    assert "configuration-unordered" in G.REFUSALS


def test_the_writer_is_not_written():
    with pytest.raises(NotImplementedError, match="proofs/architecture-on-branch"):
        G.configuration(_hex("9"))


def test_a_log_of_the_other_items_hashes_and_reads_as_before():
    log = G.chain("onsite", [G.Information(ITEMS), G.Declaration(0, _hex("1"), ("block0",))])
    assert G.head("onsite", log) == (
        "7be8d741709b6c8382948bdf10265252daf121e825692f1867e6a2a184e9b888"
        "006d69c834475a9138f9909d30ab89f7ef5b448032278678bc953c9d79864868")
    assert G.log_of(json.loads(G.committed("onsite", log))) == ("onsite", log)
