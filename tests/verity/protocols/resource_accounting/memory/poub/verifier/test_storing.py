"""Onsite's one store (`p2_case.storing`): the verifier puts the root of `C*` PoUS's run case is committed to (its
setup's) into PoUB-P2's case as the params' `store_root`, whatever the case held, and refuses by name a PoUS case that
names no setup. Whether the session's registration names that root is the Lean's to decide
(`P2Protocol.NOT_POUS_STORE`): `lean/test_p2_case.py` replays a case held to its own store, accepted, and to another,
refused."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from verity.commitments.identity import canonical_json_bytes
from verity.protocols.resource_accounting.memory.poub.prover import p2_prover
from verity.protocols.resource_accounting.memory.poub.verifier import p2, p2_case, p2_parties
from verity.protocols.resource_accounting.memory.pous.attester import memory_attester as mc
from verity.protocols.resource_accounting.memory.pous.verifier import period as pous
from verity.protocols.resource_accounting.memory.pous.prover import period_prover
from verity.protocols.resource_accounting.memory.pous.verifier import run as R
from verity.protocols.resource_accounting.memory.pous.verifier import setup
from verity.protocols.resource_accounting.memory.pous.verifier.schemes.p2 import P2Params

HERE = Path(__file__).resolve().parent
SIGNER = Path(__file__).resolve().parents[7] / "tests/verity/protocols/resource_accounting/memory/pous/verifier/pous_test_signer.py"
FIXTURES = HERE / "fixtures"
SOURCE = b"the verifier's secret"
COINS = hashlib.sha512(b"the period's live coins").digest()
OTHER_POUS_STORE = hashlib.shake_256(b"another tiny MLP's weights").digest(500)
"""Four blocks of P2's 126 payload bytes at the small point's width: a store PoUS sets up, never the committed `C*`."""
REQUEST = {"deadline_ns": 400, "passes": 2, "samples": 4, "window_ns": 800, "windows": 2}


def _signer() -> ModuleType:
    """PoUS's tests' test signer, loaded from its file: S1's LMS at a small height, under a test scheme id."""
    spec = importlib.util.spec_from_file_location("pous_test_signer", SIGNER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _committed() -> p2_prover.Store:
    """The committed 9-block `C*` (`fixtures/p2-store.json`), registered for P2 on its PoUS setup."""
    doc = json.loads((FIXTURES / "p2-store.json").read_text())
    data = (FIXTURES / "p2-c-star.bin").read_bytes()
    blocks = [data[p2.BLOCK_BYTES * j:p2.BLOCK_BYTES * (j + 1)] for j in range(len(data) // p2.BLOCK_BYTES)]
    m = doc["params"]
    params = P2Params(m["w"], m["c0"], m["blocks_per_segment"], m["segments"], m["expander"])
    commitment, salt, n = bytes.fromhex(doc["commitment"]), bytes.fromhex(doc["salt"]), params.blocks_per_segment
    vk = setup.verifier_key_from_encoding(params, None, commitment, salt,
                                          [blocks[n * s:n * (s + 1)] for s in range(params.segments)])
    return p2_prover.Store(p2.register(SOURCE, b"n0", commitment, vk), blocks)


def _root(pous_case: dict) -> bytes:
    """The root of `C*` PoUS's run case is committed to: its setup's, in its commitment."""
    return R.Setup.from_json(json.loads(bytes.fromhex(pous_case["commitment"]))).root


@pytest.fixture(scope="module")
def cases(tmp_path_factory) -> dict[str, dict]:
    """PoUS's audit of the committed `C*` and its honest run on another store (a test signer's), and PoUB-P2's case on
    the committed `C*`."""
    tmp, S = tmp_path_factory.mktemp("storing"), _signer()
    kw = {"signer": S.TestSigner(b"the memory challenger", directory=tmp), "log": S.session_log(tmp),
          "schemes": S.SCHEMES, "protocol": mc.TEST_PREFIX + pous.PROTOCOL, "clock": lambda: 10 ** 9}
    store = _committed()
    reg, q = store.registration, store.registration.params
    n = q.blocks_per_segment
    held = period_prover.EncodedStore(R.Point(q, 8, pous.LIMIT_NS), R.Setup(q.blocks, reg.commitment, reg.root, reg.salt),
                             store.blocks, b"".join(mc.block_digest(b) for b in store.blocks),
                             pous.setup_case(q, reg.commitment, reg.salt,
                                             [list(store.blocks[n * s:n * (s + 1)]) for s in range(q.segments)]))
    space = period_prover.audit("n0", held, COINS, **kw)
    other = period_prover.run("n0", OTHER_POUS_STORE, COINS, point=pous.point_for(len(OTHER_POUS_STORE), pous.W, pous.C0, k=8),
                     **kw)
    sid = hashlib.sha512(b"a P2 session").digest()[:32]
    held = p2_parties.registration_json(reg, None)
    head = p2_case.head(canonical_json_bytes(held), 10 ** 9)
    reads = p2_case.case_of(held, REQUEST | {"node": reg.node.hex()}, 10 ** 9, [], session=sid,
                            trusted=[kw["signer"].public_key], close=p2_case.close_of(kw["signer"], sid, head, []))
    return {"space": space.case, "other-space": other.case, "reads": reads}


def test_the_case_names_its_registrations_root_until_it_is_stored(cases) -> None:
    """The prover's case names its registration's root in its params, 128 lowercase hex characters, and has no field
    outside the served three."""
    case = cases["reads"]
    root = _committed().registration.root
    assert sorted(case) == ["commitment", "evidence", "params"]
    assert case["params"]["store_root"] == root.hex() == json.loads(bytes.fromhex(case["commitment"]))["root"]


def test_storing_puts_pous_root_in_the_params(cases) -> None:
    """The params name PoUS's root, whichever store that is, and nothing else of the case changes."""
    reads = cases["reads"]
    for pous_case in (cases["space"], cases["other-space"]):
        stored = p2_case.storing(reads, pous_case)
        assert stored["params"]["store_root"] == _root(pous_case).hex()
        assert {k: v for k, v in stored.items() if k != "params"} == {k: v for k, v in reads.items() if k != "params"}
        assert {k: v for k, v in stored["params"].items() if k != "store_root"} == \
            {k: v for k, v in reads["params"].items() if k != "store_root"}
    assert _root(cases["space"]) != _root(cases["other-space"])
    assert reads["params"]["store_root"] == _root(cases["space"]).hex()


def test_a_pous_case_that_names_no_setup_is_refused(cases) -> None:
    for case in ({}, cases["space"] | {"commitment": "00"}, cases["space"] | {"commitment": cases["reads"]["commitment"]}):
        with pytest.raises(ValueError, match="^PoUS's case names no setup: "):
            p2_case.storing(cases["reads"], case)
