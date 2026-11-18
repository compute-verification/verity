"""PoUS on a served run through the exchange (`prover/served.py`), on the CPU at the small point (8 blocks a segment), with a
stand-in exchange that keeps what a scheme subscribes and files, as the vLLM adapter's `messages.Exchange` does."""
from __future__ import annotations

from fractions import Fraction

import pytest
from verity import hooks as H
from verity.protocols.resource_accounting import scheme as S
from verity.protocols.resource_accounting.memory.pous.prover import served
from verity.protocols.resource_accounting.memory.pous.prover import weight_store as ws
from verity.protocols.resource_accounting.memory.pous.scheme import POUS
from verity.protocols.resource_accounting.memory.pous.verifier import period
from verity.protocols.resource_accounting.memory.pous.verifier.schemes.p2 import P2Params

torch = pytest.importorskip("torch")

P2 = "p2-16448/v4"
SOURCE = "a5" * 32
OWN = {"salt_source": SOURCE, "experimental": True}
SMALL = P2Params(period.W, period.C0, 8)


class Exchange:
    def __init__(self) -> None:
        self.subscribers: dict[str, list] = {h: [] for h in H.HOOKS}
        self.requests: list[tuple[str, H.Request]] = []

    def subscribe(self, name, hook, fn):
        self.subscribers[hook].append(fn)

    def request(self, name, request):
        self.requests.append((name, request))

    def emit(self, event):
        for fn in self.subscribers[event.hook]:
            fn(event)

    def filed(self, kind):
        return [r for _, r in self.requests if isinstance(r, kind)]


@pytest.fixture
def run(monkeypatch):
    monkeypatch.setattr(ws, "geometry", lambda scheme: SMALL)
    g = torch.Generator().manual_seed(7)
    w = {"a.weight": torch.randint(-128, 128, (6, 40), dtype=torch.int8, generator=g),
         "b.weight": torch.randint(-128, 128, (3, 24), dtype=torch.int8, generator=g)}
    s, ex = served.Served(P2, POUS.settings), Exchange()
    s.install(ex)
    ex.emit(H.Declared("card", {"eager": True}, (POUS.option,), {"pous": OWN}))
    plain = {n: t.clone() for n, t in w.items()}
    values = (("a.weight", w["a.weight"]), ("a.weight_scale", torch.ones(6, 1)), ("b.weight", w["b.weight"]))
    ex.emit(H.RegisteredValues(values, linear=frozenset(w)))
    return s, ex, plain


def test_the_weights_are_committed_before_the_salt_released_and_restored_from_c_star(run):
    s, ex, plain = run
    _, commitment = ws.commit(P2, plain)
    assert s.record()["commitment"] == commitment.hex() and s.summary()["weights"] == 2
    released = ex.filed(H.ReleaseValue)
    assert [r.value for r in released] == ["a.weight", "b.weight"]                  # the scale is no linear weight
    for r in released:
        assert torch.equal(r.restore(), plain[r.value])


def test_the_kernel_hands_inward_the_weight_decoded_from_c_star_and_leaves_other_blocks_alone(run):
    s, ex, plain = run
    [kernel] = [r.kernel for r in ex.filed(H.ComputeBlockWith)]
    due = H.BlockDue("b", 1, {"n": 3, "k": 24}, "int8", {"x": None}, weights={"weight": "b.weight"})
    assert torch.equal(kernel(due, lambda given: given["weight"]), plain["b.weight"])
    other = H.BlockDue("lm_head", 2, {"n": 1, "k": 1}, "int8", {"x": None}, weights={"weight": "lm_head.weight"})
    assert kernel(other, lambda given: given) is None


def test_its_settings_are_the_declared_builds_and_each_is_checked():
    assert served.refusals(P2, OWN, POUS.settings) == ()
    assert any("unknown pous settings ['port']" in r for r in served.refusals(P2, {**OWN, "port": 1}, POUS.settings))
    assert any("challenges is a positive integer, not 0" in r
               for r in served.refusals(P2, {**OWN, "challenges": 0}, POUS.settings))
    assert any("experimental" in r for r in served.refusals(P2, {"salt_source": SOURCE}, POUS.settings))
    assert any("no salt_source" in r for r in served.refusals(P2, {"experimental": True}, POUS.settings))
    s = served.Served(P2, POUS.settings)
    with pytest.raises(ValueError, match="no salt_source"):
        s.declared(H.Declared("card", {}, (POUS.option,), {"pous": {"experimental": True}}))
    with pytest.raises(RuntimeError, match="before the run declared its settings"):
        s.registered(H.RegisteredValues((("a.weight", torch.zeros(1, 1, dtype=torch.int8)),), linear=frozenset({"a.weight"})))


def test_the_scheme_requires_eager_and_its_attach_waits_on_the_linear_site_by_name():
    assert POUS.requests(None) == (served.EAGER,)
    with pytest.raises(NotImplementedError, match=served.GOAL):
        POUS.attach(Exchange(), None)
    with pytest.raises(ValueError, match="help"):
        S.bound(POUS, Fraction(1, 100), 0, {c.name: 0 for c in POUS.charges})
