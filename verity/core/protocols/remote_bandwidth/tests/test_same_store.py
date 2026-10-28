"""Onsite's one store (`period.store_root`, `period.same_store`): bandwidth's case names its store, its registration's
payload root, and onsite holds it to the root of `C*` that PoUS's run case is committed to. A bandwidth case on PoUS's
store meets PoUS's run; two cases on different stores, a case that names no store, and one whose `store_root` its
commitment doesn't register are refused, each by name."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period as pous
from verity.core.protocols.pous import run as RUN
from verity.core.protocols.remote_bandwidth import period as P

SIGNER = Path(__file__).resolve().parents[2] / "pous/tests/pous_test_signer.py"
STORE = hashlib.shake_256(b"a tiny MLP's weights").digest(2 * 4032 - 17)
"""Two blocks of the bandwidth codec's 4,032 payload bytes."""
POUS_STORE = STORE[:500]
"""Four blocks of P2's 126 payload bytes at the small point's width."""
OTHER_POUS_STORE = hashlib.shake_256(b"another tiny MLP's weights").digest(len(POUS_STORE))
COINS = hashlib.sha512(b"the period's live coins").digest()


def _signer() -> ModuleType:
    """PoUS's tests' test signer, loaded from its file: S1's LMS at a small height, under a test scheme id."""
    spec = importlib.util.spec_from_file_location("pous_test_signer", SIGNER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Clock:
    """The bandwidth verifier's clock: each reading a microsecond after the last, and waiting moves it on."""

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


class OnStore(P.Local):
    """A prover that registers `root` as its store's payload root and holds its own store: its session stops before any
    window is verified, so nothing it opens is checked against `root`, and it ends undecided on that store."""

    def __init__(self, root: bytes) -> None:
        super().__init__("n0", STORE, entropy=_entropy(b"salts"))
        self.root = root

    def register(self) -> tuple[int, bytes]:
        return len(self.blocks), self.root

    def hold(self, registration, commitment) -> None:
        super().hold(dataclasses.replace(registration, payload_root=super().register()[1]), commitment)


def _root(pous_case: dict) -> bytes:
    """The root of `C*` PoUS's run case is committed to: its setup's, in its commitment."""
    return RUN.Setup.from_json(json.loads(bytes.fromhex(pous_case["commitment"]))).root


def _refused(ours: bytes, theirs: bytes) -> str:
    return f"bandwidth's store {ours.hex()[:16]}… is not PoUS's {theirs.hex()[:16]}…"


@pytest.fixture(scope="module")
def cases(tmp_path_factory) -> dict[str, dict]:
    """PoUS's honest runs on `POUS_STORE` and `OTHER_POUS_STORE` (a test signer's, at the small point), bandwidth's honest
    session on its own store, and its session registered on `POUS_STORE`'s `C*` (`OnStore`): each one's case and
    answer."""
    tmp, S = tmp_path_factory.mktemp("same-store"), _signer()

    def space(store: bytes) -> pous.Period:
        return pous.run("n0", store, COINS, signer=S.TestSigner(b"the memory challenger", directory=tmp),
                        log=S.session_log(tmp), point=pous.point_for(len(store), pous.W, pous.C0, k=8),
                        schemes=S.SCHEMES, protocol=mc.TEST_PREFIX + pous.PROTOCOL, clock=lambda: 10 ** 9)

    def reads(prover: P.Prover | None = None, **kw) -> P.Period:
        c = Clock()
        if prover is None:
            prover = P.Local("n0", STORE, entropy=_entropy(b"salts"))
        return P.session("n0", prover, COINS, windows=1, clock=c, sleep_until=c.sleep_until, **kw)

    space_, other = space(POUS_STORE), space(OTHER_POUS_STORE)
    out = {"space": space_, "other-space": other, "reads": reads(), "on-space": reads(OnStore(_root(space_.case)), verify=0)}
    assert {k: p.answer["verdict"] for k, p in out.items()} == {"space": "accept", "other-space": "accept",
                                                                 "reads": "accept", "on-space": "undecided"}
    return {k: p.case for k, p in out.items()}


def test_the_case_names_its_registrations_store(cases) -> None:
    """The case's `store_root` is its registration's payload root, the payload tree's root over the store it holds, as
    128 lowercase hex characters, and `store_root` reads it back."""
    case = cases["reads"]
    root = P.Local("n0", STORE, entropy=_entropy(b"salts")).register()[1]
    assert case["store_root"] == root.hex() == json.loads(bytes.fromhex(case["commitment"]))["payload_root"]
    assert len(case["store_root"]) == 128 and P.store_root(case) == root


def test_a_case_on_pous_store_meets_pous_run(cases) -> None:
    """Bandwidth's case registered on the `C*` PoUS's run is committed to names PoUS's store, so onsite takes the two as
    one store."""
    assert P.store_root(cases["on-space"]) == _root(cases["space"])
    assert P.same_store(cases["on-space"], cases["space"]) is None


def test_two_cases_on_different_stores_are_refused(cases) -> None:
    """Bandwidth's honest case is on its own store, never PoUS's `C*`; and a case on PoUS's store meets no run of PoUS
    on another store."""
    reads, on_space, space, other = cases["reads"], cases["on-space"], cases["space"], cases["other-space"]
    assert _root(space) != _root(other)
    assert P.same_store(reads, space) == _refused(P.store_root(reads), _root(space))
    assert P.same_store(on_space, other) == _refused(_root(space), _root(other))


@pytest.mark.parametrize(("change", "why"), [
    (lambda c, root: {k: v for k, v in c.items() if k != "store_root"}, "the case names no store"),
    (lambda c, root: c | {"store_root": None}, "the case names no store"),
    (lambda c, root: c | {"store_root": c["store_root"].upper()}, "the case names no store"),
    (lambda c, root: c | {"store_root": c["store_root"][:-2]}, "the case names no store"),
    (lambda c, root: c | {"store_root": bytes.fromhex(c["store_root"])}, "the case names no store"),
    (lambda c, root: c | {"store_root": root.hex()},
     "the case's store_root is not the payload root its commitment registers"),
    (lambda c, root: c | {"commitment": "00"}, "the case's commitment registers no payload root"),
    (lambda c, root: c | {"commitment": json.dumps({"node": "00"}).encode().hex()},
     "the case's commitment registers no payload root"),
])
def test_a_case_that_names_no_store_its_commitment_registers_is_refused(cases, change, why) -> None:
    """The field is required: a case without it, or with anything but 128 lowercase hex characters, names no store. And
    since no digest binds it, a case that names PoUS's store while its registration names another is refused."""
    case = change(cases["reads"], _root(cases["space"]))
    with pytest.raises(ValueError, match=f"^{why}"):
        P.store_root(case)
    assert P.same_store(case, cases["space"]).startswith(why)


def test_a_pous_case_that_names_no_setup_is_refused(cases) -> None:
    for case in ({}, cases["space"] | {"commitment": "00"}, cases["space"] | {"commitment": cases["on-space"]["commitment"]}):
        assert P.same_store(cases["on-space"], case).startswith("PoUS's case names no setup: ")
