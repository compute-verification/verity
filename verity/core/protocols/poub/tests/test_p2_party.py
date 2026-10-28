"""The P2 session's verifier as a party of its own (`p2_party.py`) over a channel to a prover serving the 9-block P2
store of `test_p2.py`'s fixture: an honest prover accepted with its credited reads, under every unit or the
registration's plan, one whose receipts skip a unit's reads, one whose openings are out of form and one that reads
every unit when the plan is fewer rejected, and a request for another node refused before any coin."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import socket
import threading
from pathlib import Path

import pytest

from verity.core.protocols.pous import setup
from verity.core.protocols.pous.schemes.p2 import P2Params
from verity.core.protocols.poub import coins, p2, p2_party, receipts

FIXTURES = Path(__file__).parent / "fixtures"
SOURCE = b"the verifier's secret"
NODE = b"p2-node-under-test"
STEP_NS = 400


class Clock:
    """The verifier's clock, which the prover's steps and the verifier's waits advance."""

    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now

    def sleep_until(self, _clock, t: int) -> None:
        self.now = max(self.now, t)


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


class Prover:
    """The honest prover over `store`, each pass the units of the plan the verifier holds, each step `STEP_NS` on the
    verifier's clock; `skip` names a unit whose reads it skips, its receipt from zeros, `garble` sends openings whose
    leaf salts are a byte short, and `whole` reads every unit whatever the plan."""

    def __init__(self, store: p2.Store, clock: Clock, *, skip: int | None = None, garble: bool = False,
                 whole: bool = False) -> None:
        self.store, self.clock, self.skip, self.garble, self.whole = store, clock, skip, garble, whole
        self.passes: dict[tuple[int, int], object] = {}
        self.held = self.plan = None

    def hold(self, registration, commitment) -> None:
        if p2_party.registration_of(registration) != self.store.registration:
            raise ValueError("not this store's registration")
        self.plan = p2_party.plan_of({} if self.whole else registration, self.store.registration)
        self.held = p2.Prover(self.store, commitment, entropy=Entropy(b"p2-party-test/leaf-salts"))

    def run(self, challenge, t, i, previous) -> bytes:
        seed = receipts.seed(coins.coin(challenge), i, previous)
        sk = p2.Sketcher(seed, self.store.registration.rows)
        ys = [sk(g, bytes(p2.UNIT_BYTES) if g == self.skip else self.store.unit(g)) for g in self.plan]
        ps = self.held.commit(t, i, seed, self.plan, ys)
        self.passes[t, i] = ps
        self.clock.now += STEP_NS
        return ps.root

    def open(self, t, draws):
        out = [self.held.open(self.passes[t, i], p) for i, p in draws]
        return [dataclasses.replace(o, leaf_salt=o.leaf_salt[:-1]) for o in out] if self.garble else out


def run(store: p2.Store, prover_of, request: dict, plan: receipts.Plan | None = None) -> tuple[dict, dict]:
    """One session over a socket pair: the verifier's side in a thread, holding `plan`, the prover's here; both
    answers."""
    clock = Clock()
    a, b = socket.socketpair()
    got: dict = {}

    def verifier() -> None:
        with b, b.makefile("rb") as r, b.makefile("wb") as w:
            got["answer"] = p2_party.session(r, w, store.registration, coins=b"\x07" * 64, plan=plan, clock=clock,
                                             sleep_until=clock.sleep_until, host="auditor-under-test")

    th = threading.Thread(target=verifier)
    th.start()
    prover = prover_of(clock)
    with a, a.makefile("rb") as r, a.makefile("wb") as w:
        out = p2_party.prove(r, w, prover, request)
    th.join()
    return out, got["answer"]


def request(store: p2.Store, **kw) -> dict:
    base = {"windows": 2, "passes": 2, "samples": 4, "window_ns": 2 * STEP_NS, "deadline_ns": STEP_NS}
    return p2_party.request_of(store.registration.node, **{**base, **kw})


def test_the_registration_and_openings_round_trip_their_json(store) -> None:
    reg = store.registration
    assert p2_party.registration_of(json.loads(json.dumps(p2_party.registration_json(reg)))) == reg
    bad = p2_party.registration_json(reg) | {"rows": 64}
    with pytest.raises(ValueError, match="digest"):
        p2_party.registration_of(bad)
    prover = p2.Prover(store, coins.local(reg.digest, 0, 1, Entropy(b"c")).commitment, entropy=Entropy(b"s"))
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


def test_a_plan_of_fewer_units_is_what_each_pass_reads_and_is_credited(store) -> None:
    units = store.registration.blocks
    plan = receipts.Plan(((1, units - 1),), units)
    proved, answer = run(store, lambda c: Prover(store, c), request(store), plan)
    assert proved == answer and answer["verdict"] == "accept", answer
    assert answer["units"] == units and answer["plan_units"] == units - 1
    assert answer["receipts"] == 2 * 2 * (units - 1)
    _, answer = run(store, lambda c: Prover(store, c, whole=True), request(store), plan)
    assert answer["verdict"] == "reject" and "q_bytes" not in answer, answer


def test_an_honest_prover_is_accepted_with_its_credited_reads(store) -> None:
    proved, answer = run(store, lambda c: Prover(store, c), request(store))
    assert proved == answer
    assert answer["verdict"] == "accept", answer
    assert answer["host"] == "auditor-under-test"
    units = store.registration.blocks
    assert answer["receipts"] == 2 * 2 * units and answer["plan_units"] == units
    assert [w["passes"] for w in answer["windows"]] == [2, 2]
    assert all(w["verdict"]["accepted"] for w in answer["windows"])
    assert answer["q_bytes"] == (answer["receipts"] - answer["wrong_at_most"]) * p2.UNIT_BYTES
    assert answer["span_ns"] > 0 and answer["credited_bytes_per_s"] is not None


def test_a_prover_that_skips_a_units_reads_is_rejected(store) -> None:
    _, answer = run(store, lambda c: Prover(store, c, skip=1), request(store))
    assert answer["verdict"] == "reject" and answer["failure"] == "receipt", answer
    assert answer["window"] == 0 and answer["at"][1] % store.registration.blocks == 1
    assert "q_bytes" not in answer


def test_openings_out_of_form_are_rejected(store) -> None:
    _, answer = run(store, lambda c: Prover(store, c, garble=True), request(store))
    assert answer["verdict"] == "reject" and answer["failure"] == "malformed", answer


def test_a_request_for_another_node_is_refused_before_any_coin(store) -> None:
    bad = request(store) | {"node": b"another-node".hex()}
    _, answer = run(store, lambda c: Prover(store, c), bad)
    assert answer["verdict"] == "refused" and "node" in answer["reason"]
