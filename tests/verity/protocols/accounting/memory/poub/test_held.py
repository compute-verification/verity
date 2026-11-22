"""``bandwidth`` attached by name beside PoUS on a stand-in exchange (`poub/scheme.py`, `prover/p2_stage/bandwidth.py`): P2's prover
runs on the store PoUS holds, encoded once on the CPU at P2's registered point, 4 blocks a segment."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from verity import hooks as H
from verity.protocols.accounting import scheme as S
from verity.protocols.accounting.memory.poub import scheme as poub
from verity.protocols.accounting.memory.poub.verifier import codec, coins, p2, p2_parties, receipts
from verity.protocols.accounting.memory.pous.prover import weight_store as ws
from verity.protocols.accounting.memory.pous.scheme import POUS
from verity.protocols.accounting.memory.pous.verifier.schemes.p2 import P2Params

torch = pytest.importorskip("torch")

CTX = {"salt_source": "a5" * 32, "experimental": True}
NODE = b"held-test/node"


class Exchange:
    def __init__(self) -> None:
        self.subscribers: dict[str, list] = {h: [] for h in H.HOOKS}

    def subscribe(self, name, hook, fn):
        self.subscribers[hook].append(fn)

    def request(self, name, request):
        pass

    def emit(self, event):
        for fn in self.subscribers[event.hook]:
            fn(event)


@pytest.fixture(scope="module")
def run():
    """PoUS attached, ``bandwidth`` attached before it binds its store (``early``, with its refusal then), the weights
    registered, and ``bandwidth`` attached again after (``late``)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ws, "geometry", lambda scheme: P2Params(*codec.P2_POINT, 4))
        ex = Exchange()
        storage = POUS.attach(ex, CTX)
        early = poub.BANDWIDTH.attach(ex, {})
        with pytest.raises(RuntimeError, match="holds no store yet"):
            early.peer(NODE)
        g = torch.Generator().manual_seed(7)
        w = (("a.weight", torch.randint(-128, 128, (6, 40), dtype=torch.int8, generator=g)),
             ("b.weight", torch.randint(-128, 128, (3, 24), dtype=torch.int8, generator=g)))
        ex.emit(H.Declared("card", {"eager": True}, (POUS.name, poub.BANDWIDTH.name), {"pous": S.settings(POUS, {})}))
        ex.emit(H.RegisteredValues(w, operands={"a.weight": 0, "b.weight": 1}))
        yield SimpleNamespace(storage=storage, early=early, late=poub.BANDWIDTH.attach(ex, {}))


def test_bandwidth_is_refused_by_name_on_an_exchange_without_pous() -> None:
    with pytest.raises(LookupError, match="no pous is attached on it: attach pous before bandwidth"):
        poub.BANDWIDTH.attach(Exchange(), {})


def test_bandwidth_reads_the_store_pous_binds_whether_attached_before_or_after(run) -> None:
    assert run.early.store is run.storage.store and run.late.store is run.storage.store


def test_the_audit_runs_on_the_store_pous_holds(run) -> None:
    store = run.storage.store
    peer = run.early.peer(NODE, spot=4)
    reg = peer.registration
    assert (reg.commitment, reg.salt, reg.root) == (store.commitment, store.salt, store.root)
    peer.hold(p2_parties.registration_json(reg))
    challenge = coins.Released(0, coins.CHALLENGE, 0, bytes(64), bytes(range(coins.COIN_BYTES)))
    previous = bytes(64)
    root = peer.run(challenge, 0, 0, previous)
    seed = receipts.seed(coins.coin(challenge), 0, previous)
    positions = [0, peer.plan.length - 1]
    openings = peer.open(0, [(0, p) for p in positions])
    assert [p2.check(reg, seed, peer.plan, root, 0, p, o) for p, o in zip(positions, openings)] == [None, None]
    assert p2.check(reg, seed, peer.plan, bytes(64), 0, 0, openings[0]) == "receipt-path"
