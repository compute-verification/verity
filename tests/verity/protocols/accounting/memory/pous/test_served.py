"""PoUS on a served run through the exchange (`prover/served.py`), on the CPU at the small point (8 blocks a segment), with a
stand-in exchange that keeps what a scheme subscribes and files, as the vLLM adapter's `messages.Exchange` does."""
from __future__ import annotations

import weakref
from fractions import Fraction

import pytest
from verity import hooks as H
from verity.protocols.accounting import scheme as S
from verity.protocols.accounting.memory.pous.prover import served
from verity.protocols.accounting.memory.pous.prover import weight_store as ws
from verity.protocols.accounting.memory.pous.scheme import POUS
from verity.protocols.accounting.memory.pous.verifier import period
from verity.protocols.accounting.memory.pous.verifier.schemes.p2 import P2Params

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


def same(got, want) -> bool:
    """``got`` is ``want`` bit for bit: its bytes, dtype, shape and device. Not `torch.equal`, which compares values: a
    NaN is unequal to itself and ``-0.0`` equal to ``+0.0``."""
    return (got.dtype == want.dtype and tuple(got.shape) == tuple(want.shape) and got.device == want.device
            and got.contiguous().view(torch.uint8).numpy().tobytes() == want.contiguous().view(torch.uint8).numpy().tobytes())


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
        assert same(r.restore(), plain[r.value])


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
    assert same(kernel(due, lambda given: given["weight"]), plain["b.weight"])
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


def every_event(values) -> tuple:
    """One of each `verity.hooks.EVENTS`, in a served run's order: whatever the adapter can hand a subscriber of a hook,
    whose dispatch reads only the hook (`verity_vllm.messages.Exchange.emit`)."""
    pool = torch.zeros(2 * 2 * 16 * 64, dtype=torch.float16)
    got = (H.Declared("card", {"eager": True}, (POUS.name,), {"pous": S.settings(POUS, {})}),
           H.RegisteredValues(values, operands={n: i for i, (n, _) in enumerate(values)}),
           H.RangesDeclared(2, 16, 256, 1, (pool,)), H.InputArrived("a", 3, 0), H.StepBegan(0, ("a",), 1),
           H.RangeAllocated((0,), "a"), H.RangeWritten(0, 0, 3), H.PhaseOpened(0, 0),
           H.ValueProduced("b", "x", 0, None), H.BlockDue("b", 1, {"n": 1}, "int8", {"x": None}), H.StepComputed(0),
           H.StepEnded(0), H.OutputsComplete("a", 0), H.RangeFreed((0,)), H.PhaseClosed(0, 0, b""),
           H.PhaseCommitted(0, bytes(64)))
    assert {type(e) for e in got} == set(H.EVENTS)
    return got


def test_every_event_on_its_hooks_passes_its_handlers_and_its_own_are_still_handled(monkeypatch):
    """vLLM's KV cache declaring its ranges (`RangesDeclared`) rides the hook PoUS's weights do (``memory``): every
    event of the hooks it subscribes to passes, its own are handled, and what is no event is refused by name."""
    monkeypatch.setattr(ws, "geometry", lambda scheme: SMALL)
    w = torch.randint(-128, 128, (6, 40), dtype=torch.int8, generator=torch.Generator().manual_seed(7))
    s, ex = served.Served(P2, CTX), Exchange()
    s.install(ex)
    for e in every_event((("a.weight", w),)):
        ex.emit(e)
    assert s.commitment()["commitment"] == ws.commit(P2, {"a.weight": w})[1].hex() and s.settings == {"challenges": 320}
    for hook in ("declared", "memory"):
        for fn in ex.subscribers[hook]:
            with pytest.raises(TypeError, match=r"pous:.*: dict is no event of a served run"):
                fn({"values": ()})


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


def special_bf16(shape, seed):
    """A bf16 tensor of ``shape``, random bit patterns led by ``-0``, ``+0``, NaNs with payloads (of either sign), ``±inf``
    and subnormals."""
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(0, 1 << 16, shape, generator=g, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).clone()
    lead = torch.tensor([0x8000, 0x0000, 0x7FC1, 0xFFC3, 0x7F81, 0x7F80, 0xFF80, 0x0001, 0x8001, 0x007F],
                        dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    k = min(lead.numel(), t.numel())
    t.view(-1)[:k] = lead[:k]
    return t


def decoded_bits(monkeypatch, point, weights):
    """``weights`` registered through PoUS's served path at ``point``: each released weight's ``restore`` and the
    kernel's decode for its block, each held to the registered weight bit for bit (`same`)."""
    monkeypatch.setattr(ws, "geometry", lambda scheme: point)
    plain = {n: t.clone() for n, t in weights.items()}
    s, ex = served.Served(P2, CTX), Exchange()
    s.install(ex)
    ex.emit(H.Declared("card", {"eager": True}, (POUS.name,), {"pous": S.settings(POUS, {})}))
    ex.emit(H.RegisteredValues(tuple(weights.items()), operands={n: i for i, n in enumerate(weights)}))
    pb = s.store.params.payload_bytes
    assert all(0 < (t.numel() * t.element_size()) % pb for t in plain.values())        # each ends inside its last block
    [kernel] = [r.kernel for r in ex.filed(H.ComputeBlockWith)]
    released = ex.filed(H.ReleaseValue)
    assert [r.value for r in released] == list(plain)
    for r in released:
        due = H.BlockDue(r.value, 0, {}, "bf16", {"x": None}, weights={"weight": r.value})
        for got in (r.restore(), kernel(due, lambda given: given["weight"])):
            assert same(got, plain[r.value]), r.value
            assert got.is_contiguous() and got.data_ptr() != plain[r.value].data_ptr()
    return s


def test_same_is_a_bit_comparison_where_torch_equal_is_not():
    nan, z = special_bf16((4,), 0)[2:3], torch.zeros(1, dtype=torch.bfloat16)
    assert same(nan, nan.clone()) and not torch.equal(nan, nan.clone())
    assert not same(-z, z) and torch.equal(-z, z)


def test_the_decode_hands_back_each_registered_weights_bytes_at_the_toy_point(monkeypatch):
    """bf16 weights of awkward lengths, ending partway into their last blocks, over many segments of the toy point,
    beside a tiny one and an int8 one: every byte back, NaN payloads, ``-0``, infinities and subnormals among them."""
    s = decoded_bits(monkeypatch, SMALL, {"a.weight": special_bf16((7, 1031), 1), "b.weight": special_bf16((3,), 2),
                                          "c.weight": torch.randint(-128, 128, (5, 13), dtype=torch.int8,
                                                                    generator=torch.Generator().manual_seed(3))})
    assert s.store.params.segments > 1


@pytest.mark.slow
def test_the_decode_hands_back_each_registered_weights_bytes_at_the_deployed_field(monkeypatch):
    """The toy point's test at the registered scheme's own field (``p2-16448/v4``: w = 16448, c0 = 21065, 2054-byte
    payloads), one segment of 8 blocks. About 35 s on one CPU core, the encode's roots at 16448 bits; it catches what the
    toy point can't: a payload of P2's width, whose byte order, padding and field reduction the toy's 126-byte payload
    doesn't exercise."""
    from verity.protocols.accounting.memory.pous.verifier import schemes as pous_schemes

    q = pous_schemes.get(P2).params
    point = P2Params(q.w, q.c0, 8, 1, q.expander)
    assert (point.w, point.c0, point.payload_bytes) == (16448, 21065, 2054)
    s = decoded_bits(monkeypatch, point, {"a.weight": special_bf16((3, 1031), 4), "b.weight": special_bf16((5, 211), 5),
                                          "c.weight": special_bf16((3,), 6),
                                          "d.weight": torch.randint(-128, 128, (5, 13), dtype=torch.int8,
                                                                    generator=torch.Generator().manual_seed(7))})
    assert s.store.params.blocks == 8
