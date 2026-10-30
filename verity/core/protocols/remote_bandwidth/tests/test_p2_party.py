"""The P2 session's party (`p2_party.py`): its registration, plan and openings round-trip their JSON, and a registration
whose digest isn't its fields' or a plan that isn't one of its units is refused. Its sessions, decided by the Lean
verifier with these books beside it, are `lean/tests/test_p2_party.py`'s, which run where Lake is."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from verity.core.protocols.pous import setup
from verity.core.protocols.pous.schemes.p2 import P2Params
from verity.core.protocols.remote_bandwidth import p2, p2_party, receipts

FIXTURES = Path(__file__).parent / "fixtures"
SOURCE = b"the verifier's secret"
NODE = b"p2-node-under-test"


class Entropy:
    def __init__(self, label: bytes) -> None:
        self.label, self.count = label, 0

    def __call__(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(self.label + self.count.to_bytes(8, "big")).digest()
            self.count += 1
        return out[:n]


@pytest.fixture(scope="module")
def store() -> p2.Store:
    doc = json.loads((FIXTURES / "p2-store.json").read_text())
    data = (FIXTURES / "p2-c-star.bin").read_bytes()
    blocks = [data[p2.BLOCK_BYTES * j:p2.BLOCK_BYTES * (j + 1)] for j in range(len(data) // p2.BLOCK_BYTES)]
    m = doc["params"]
    params = P2Params(m["w"], m["c0"], m["blocks_per_segment"], m["segments"], m["expander"])
    commitment, salt, n = bytes.fromhex(doc["commitment"]), bytes.fromhex(doc["salt"]), params.blocks_per_segment
    vk = setup.verifier_key_from_encoding(params, None, commitment, salt,
                                          [blocks[n * s:n * (s + 1)] for s in range(params.segments)])
    return p2.Store(p2.register(SOURCE, NODE, commitment, vk), blocks)


def test_the_registration_and_openings_round_trip_their_json(store) -> None:
    reg = store.registration
    assert p2_party.registration_of(json.loads(json.dumps(p2_party.registration_json(reg)))) == reg
    bad = p2_party.registration_json(reg) | {"rows": 64}
    with pytest.raises(ValueError, match="digest"):
        p2_party.registration_of(bad)
    prover = p2.Prover(store, entropy=Entropy(b"s"))
    ps = prover.pass_under(0, 0, b"\x01" * 64)
    o = prover.open(ps, 1)
    assert p2_party.opening_of(json.loads(json.dumps(p2_party.opening_json(o)))) == o
    assert p2_party.opening_of({"pass": 0}) is None


def test_the_registrations_plan_round_trips_its_json_and_a_bad_one_is_refused(store) -> None:
    reg = store.registration
    plan = receipts.Plan(((1, 1),), reg.blocks)
    doc = json.loads(json.dumps(p2_party.registration_json(reg, plan)))
    assert p2_party.registration_of(doc) == reg and p2_party.plan_of(doc, reg) == plan
    assert p2_party.plan_of(p2_party.registration_json(reg), reg) == receipts.Plan.whole(reg.blocks)
    for bad in ([[0, reg.blocks + 1]], [[0, 1], [1, 1]], [], "0,1", [[0]]):
        with pytest.raises(ValueError):
            p2_party.plan_of(doc | {"plan": bad}, reg)
