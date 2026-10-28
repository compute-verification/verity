"""Onsite's one store (`p2_case.store_root`, `p2_case.same_store`): PoUB-P2's case names its store, its registration's
root, which is the root of `C*` PoUS's setup published, and onsite holds it to PoUS's run case. A P2 case on the `C*`
PoUS audited meets PoUS's run; a P2 case and a PoUS run on different stores, a generic PoUB case (whose payload root is
its own salted tree's, never PoUS's `C*`), a case that names no store and one whose `store_root` its commitment doesn't
register are refused, each by name. The case's evidence is not read here: `lean/tests/test_p2_case.py` replays it."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period as pous
from verity.core.protocols.pous import run as R
from verity.core.protocols.pous import setup
from verity.core.protocols.pous.schemes.p2 import P2Params
from verity.core.protocols.remote_bandwidth import p2, p2_case, p2_party
from verity.core.protocols.remote_bandwidth import period as P

HERE = Path(__file__).resolve().parent
SIGNER = HERE.parents[1] / "pous" / "tests" / "pous_test_signer.py"
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


class Clock:
    """The generic verifier's clock: each reading a microsecond after the last, and waiting moves it on."""

    def __init__(self) -> None:
        self.now = 10 ** 9

    def __call__(self) -> int:
        self.now += 1_000
        return self.now

    def sleep_until(self, clock, t: int) -> None:
        self.now = max(self.now, t)


def _entropy(label: bytes):
    """SHA-512 in counter mode under `label`, in place of the operating system's generator."""
    count = 0

    def draw(n: int) -> bytes:
        nonlocal count
        out = b""
        while len(out) < n:
            out += hashlib.sha512(label + count.to_bytes(8, "big")).digest()
            count += 1
        return out[:n]
    return draw


def _committed() -> p2.Store:
    """The committed 9-block `C*` (`fixtures/p2-store.json`), registered for P2 on its PoUS setup."""
    doc = json.loads((FIXTURES / "p2-store.json").read_text())
    data = (FIXTURES / "p2-c-star.bin").read_bytes()
    blocks = [data[p2.BLOCK_BYTES * j:p2.BLOCK_BYTES * (j + 1)] for j in range(len(data) // p2.BLOCK_BYTES)]
    m = doc["params"]
    params = P2Params(m["w"], m["c0"], m["blocks_per_segment"], m["segments"], m["expander"])
    commitment, salt, n = bytes.fromhex(doc["commitment"]), bytes.fromhex(doc["salt"]), params.blocks_per_segment
    vk = setup.verifier_key_from_encoding(params, None, commitment, salt,
                                          [blocks[n * s:n * (s + 1)] for s in range(params.segments)])
    return p2.Store(p2.register(SOURCE, b"n0", commitment, vk), blocks)


def _root(pous_case: dict) -> bytes:
    """The root of `C*` PoUS's run case is committed to: its setup's, in its commitment."""
    return R.Setup.from_json(json.loads(bytes.fromhex(pous_case["commitment"]))).root


def _refused(ours: bytes, theirs: bytes) -> str:
    return f"bandwidth's store {ours.hex()[:16]}… is not PoUS's {theirs.hex()[:16]}…"


@pytest.fixture(scope="module")
def cases(tmp_path_factory) -> dict[str, dict]:
    """PoUS's audit of the committed `C*` and its honest run on another store (a test signer's), PoUB-P2's case on the
    committed `C*`, and the generic PoUB's honest session on its own store."""
    tmp, S = tmp_path_factory.mktemp("same-store"), _signer()
    kw = {"signer": S.TestSigner(b"the memory challenger", directory=tmp), "log": S.session_log(tmp),
          "schemes": S.SCHEMES, "protocol": mc.TEST_PREFIX + pous.PROTOCOL, "clock": lambda: 10 ** 9}
    store = _committed()
    reg, q = store.registration, store.registration.params
    n = q.blocks_per_segment
    held = pous.Store(R.Point(q, 8, pous.LIMIT_NS), R.Setup(q.blocks, reg.commitment, reg.root, reg.salt), store.blocks,
                      b"".join(mc.block_digest(b) for b in store.blocks),
                      pous.setup_case(q, reg.commitment, reg.salt,
                                      [list(store.blocks[n * s:n * (s + 1)]) for s in range(q.segments)]))
    space = pous.audit("n0", held, COINS, **kw)
    other = pous.run("n0", OTHER_POUS_STORE, COINS, point=pous.point_for(len(OTHER_POUS_STORE), pous.W, pous.C0, k=8),
                     **kw)
    reads = p2_case.case_of(p2_party.registration_json(reg, None), REQUEST | {"node": reg.node.hex()}, 10 ** 9, [])
    c = Clock()
    generic = P.session("n0", P.Local("n0", hashlib.shake_256(b"a tiny MLP's weights").digest(2 * 4032 - 17),
                                      entropy=_entropy(b"salts")), COINS, windows=1, clock=c, sleep_until=c.sleep_until)
    assert generic.answer["verdict"] == "accept"
    return {"space": space.case, "other-space": other.case, "reads": reads, "generic": generic.case}


def test_the_case_names_its_registrations_root(cases) -> None:
    """The case's `store_root` is its registration's root, 128 lowercase hex characters, and `store_root` reads it
    back."""
    case = cases["reads"]
    root = _committed().registration.root
    assert case["store_root"] == root.hex() == json.loads(bytes.fromhex(case["commitment"]))["root"]
    assert len(case["store_root"]) == 128 and p2_case.store_root(case) == root


def test_a_p2_case_on_pous_store_meets_pous_run(cases) -> None:
    """PoUB-P2's case registered on the `C*` PoUS audited names PoUS's store, so onsite takes the two as one store."""
    assert p2_case.store_root(cases["reads"]) == _root(cases["space"])
    assert p2_case.same_store(cases["reads"], cases["space"]) is None


def test_two_cases_on_different_stores_are_refused(cases) -> None:
    reads, space, other = cases["reads"], cases["space"], cases["other-space"]
    assert _root(space) != _root(other)
    assert p2_case.same_store(reads, other) == _refused(_root(space), _root(other))


def test_a_generic_poub_case_is_refused_as_not_pous_store(cases) -> None:
    """A generic PoUB case names no store; given its payload root as one, its commitment is still no P2
    registration."""
    generic = cases["generic"]
    assert "store_root" not in generic
    assert p2_case.same_store(generic, cases["space"]).startswith("the case names no store")
    payload_root = json.loads(bytes.fromhex(generic["commitment"]))["payload_root"]
    named = generic | {"store_root": payload_root}
    assert p2_case.same_store(named, cases["space"]).startswith("the case's commitment is no PoUB-P2 registration: ")


@pytest.mark.parametrize(("change", "why"), [
    (lambda c, root: {k: v for k, v in c.items() if k != "store_root"}, "the case names no store"),
    (lambda c, root: c | {"store_root": None}, "the case names no store"),
    (lambda c, root: c | {"store_root": c["store_root"].upper()}, "the case names no store"),
    (lambda c, root: c | {"store_root": c["store_root"][:-2]}, "the case names no store"),
    (lambda c, root: c | {"store_root": bytes.fromhex(c["store_root"])}, "the case names no store"),
    (lambda c, root: c | {"store_root": root.hex()}, "the case's store_root is not the root its commitment registers"),
    (lambda c, root: c | {"commitment": "00"}, "the case's commitment is no PoUB-P2 registration"),
    (lambda c, root: c | {"commitment": json.dumps({"node": "00"}).encode().hex()},
     "the case's commitment is no PoUB-P2 registration"),
    (lambda c, root: c | {"commitment": json.dumps(json.loads(bytes.fromhex(c["commitment"])) | {"digest": "00" * 64})
                                         .encode().hex()}, "the case's commitment is no PoUB-P2 registration"),
])
def test_a_case_that_names_no_store_its_commitment_registers_is_refused(cases, change, why) -> None:
    """The field is required: a case without it, or with anything but 128 lowercase hex characters, names no store; a
    commitment that isn't a P2 registration (its stated digest not its fields') registers none; and since no digest
    binds the field, a case that names another store than its registration's is refused."""
    case = change(cases["reads"], _root(cases["other-space"]))
    with pytest.raises(ValueError, match=f"^{why}"):
        p2_case.store_root(case)
    assert p2_case.same_store(case, cases["space"]).startswith(why)


def test_a_pous_case_that_names_no_setup_is_refused(cases) -> None:
    for case in ({}, cases["space"] | {"commitment": "00"}, cases["space"] | {"commitment": cases["reads"]["commitment"]}):
        assert p2_case.same_store(cases["reads"], case).startswith("PoUS's case names no setup: ")
