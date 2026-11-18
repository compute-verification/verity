"""PoUS on a served run through the exchange (`prover/served.py`), on the CPU at the small point (8 blocks a segment), with a
stand-in exchange that keeps what a scheme subscribes and files, as the vLLM adapter's `messages.Exchange` does."""
from __future__ import annotations

import weakref
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
CTX = {"salt_source": SOURCE, "experimental": True}
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
    s, ex = served.Served(P2, CTX), Exchange()
    s.install(ex)
    ex.emit(H.Declared("card", {"eager": True}, (POUS.name,), {"pous": S.settings(POUS, {})}))
    plain = {n: t.clone() for n, t in w.items()}
    values = (("a.weight", w["a.weight"]), ("a.weight_scale", torch.ones(6, 1)), ("b.weight", w["b.weight"]))
    ex.emit(H.RegisteredValues(values, operands={"b.weight": 1, "a.weight": 0}))
    return s, ex, plain


def test_the_weights_are_committed_before_the_salt_released_and_restored_from_c_star(run):
    s, ex, plain = run
    _, commitment = ws.commit(P2, plain)
    assert s.commitment()["commitment"] == commitment.hex() and s.summary()["weights"] == 2
    released = ex.filed(H.ReleaseValue)
    assert [r.value for r in released] == ["a.weight", "b.weight"]                  # the scale is no linear weight
    for r in released:
        assert torch.equal(r.restore(), plain[r.value])


def test_what_the_setup_holds_and_its_prover_are_the_stores(run):
    s, _, plain = run
    d = s.details()
    assert d["tensors"] == 2 and d["tensor_bytes"] == sum(t.numel() for t in plain.values())
    assert d["blocks"] == s.store.params.blocks and d["salt_source"] == "settings" and d["device"] == "cpu"
    assert s.prover(())(0) == s.store.block(0) and s.answered == 1


def test_the_kernel_hands_inward_the_weight_decoded_from_c_star_and_leaves_other_blocks_alone(run):
    s, ex, plain = run
    [kernel] = [r.kernel for r in ex.filed(H.ComputeBlockWith)]
    due = H.BlockDue("b", 1, {"n": 3, "k": 24}, "int8", {"x": None}, weights={"weight": "b.weight"})
    assert torch.equal(kernel(due, lambda given: given["weight"]), plain["b.weight"])
    other = H.BlockDue("lm_head", 2, {"n": 1, "k": 1}, "int8", {"x": None}, weights={"weight": "lm_head.weight"})
    assert kernel(other, lambda given: given) is None
    s.release()
    assert s.store.rows is None and s.commitment()["blocks"] > 0


def test_no_weight_outlives_its_call_and_no_registered_tensor_is_kept_alive(monkeypatch):
    """What `POUS.keeps_weight_copy` (False) says: once the weights are registered nothing of PoUS's keeps a registered
    tensor alive, and the weight decoded for a block is gone when its call returns."""
    monkeypatch.setattr(ws, "geometry", lambda scheme: SMALL)
    g = torch.Generator().manual_seed(7)
    s, ex = served.Served(P2, CTX), Exchange()
    s.install(ex)
    ex.emit(H.Declared("card", {"eager": True}, (POUS.name,), {"pous": S.settings(POUS, {})}))
    w = torch.randint(-128, 128, (6, 40), dtype=torch.int8, generator=g)
    registered = weakref.ref(w)
    ex.emit(H.RegisteredValues((("a.weight", w),), operands={"a.weight": 0}))
    del w
    assert registered() is None and POUS.keeps_weight_copy is False
    [kernel] = [r.kernel for r in ex.filed(H.ComputeBlockWith)]
    handed: list = []
    due = H.BlockDue("a", 0, {"n": 6, "k": 40}, "int8", {"x": None}, weights={"weight": "a.weight"})
    assert kernel(due, lambda given: handed.append(weakref.ref(given["weight"]))) is None
    assert len(handed) == 1 and handed[0]() is None


def test_its_settings_are_checked_where_every_setting_is_and_its_launchers_values_when_attached():
    assert S.settings(POUS, {}) == {"challenges": 320}
    with pytest.raises(ValueError, match="challenges is a positive integer, not 0"):
        S.settings(POUS, {"challenges": 0})
    with pytest.raises(ValueError, match="no setting"):
        S.settings(POUS, {"salt_source": SOURCE})                                   # a coin is the launcher's, never a setting
    assert served.launcher_refusals(P2, CTX) == ()
    assert any("experimental" in r for r in served.launcher_refusals(P2, {"salt_source": SOURCE}))
    assert any("no salt_source" in r for r in served.launcher_refusals(P2, {"experimental": True}))
    with pytest.raises(ValueError, match="no salt_source"):
        served.Served(P2, {"experimental": True}).install(Exchange())
    s = served.Served(P2, CTX)
    with pytest.raises(RuntimeError, match="before the run declared its settings"):
        s.registered(H.RegisteredValues((("a.weight", torch.zeros(1, 1, dtype=torch.int8)),), operands={"a.weight": 0}))


def test_the_scheme_requires_eager_attaches_through_the_exchange_and_is_refused_a_bound():
    assert POUS.requests(None) == (served.EAGER,)
    ex = Exchange()
    got = POUS.attach(ex, CTX)
    assert isinstance(got, served.Served) and got.scheme == POUS.option.partition(":")[2]
    assert [h for h, fns in ex.subscribers.items() if fns] == ["declared", "memory"] and ex.filed(H.ComputeBlockWith)
    with pytest.raises(ValueError, match="help"):
        S.bound(POUS, Fraction(1, 100), 0, {c.name: 0 for c in POUS.charges})


def test_a_released_weight_pous_holds_no_name_for_is_refused_by_name(run):
    s, ex, _ = run
    [kernel] = [r.kernel for r in ex.filed(H.ComputeBlockWith)]
    tied = H.BlockDue("c", 3, {"n": 1, "k": 1}, "int8", {"weight": torch.empty(0)}, weights={"weight": "c.weight"})
    with pytest.raises(RuntimeError, match="c.weight is released, and PoUS holds no weight by that name"):
        kernel(tied, lambda given: given)
