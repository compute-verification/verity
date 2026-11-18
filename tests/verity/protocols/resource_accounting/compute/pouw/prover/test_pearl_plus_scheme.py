"""Pearl+ as a resource scheme (`verity.protocols.resource_accounting.compute.pouw.scheme`). Its block cost against the
run verifier's credit is `tests/benchmarks/pouw/test_pearl_plus_scheme_credit.py`, beside the verifier's other tests."""
from __future__ import annotations

from fractions import Fraction

import pytest

from verity import hooks as H
from verity.protocols.resource_accounting import scheme as S
from verity.protocols.resource_accounting.compute.pouw import scheme as PS
from verity.protocols.resource_accounting.compute.pouw.prover import serving as SV
from verity.protocols.resource_accounting.compute.pouw.prover.serving import tensors as T
from verity.protocols.resource_accounting.compute.pouw.prover.circuit.pearl_c_tile import PearlCTileRead
from verity.protocols.resource_accounting.compute.pouw.prover.circuit.pearl_plus_tile import PearlPlusTile
from verity.protocols.resource_accounting.compute.pouw.prover.schemes.pearl_c import PearlC


def test_found_by_name() -> None:
    assert S.SCHEMES["pearl-plus"] is PS.PEARL_PLUS
    assert {b.definition for b in PS.PEARL_PLUS.blocks} == {PearlPlusTile.id, PearlCTileRead.id}
    assert PS.PEARL_PLUS.option == f"pouw:{PearlC().name}"


def test_bound_reads_its_term_at_the_parameters_given() -> None:
    names = {c.name: 0 for c in PS.PEARL_PLUS.charges}
    with pytest.raises(ValueError, match=r"Pouw.PearlC.δPearlC is evaluated at \['q', 'N'\], and \['N'\] were not given"):
        S.bound(PS.PEARL_PLUS, Fraction(1, 100), 0, names, {"q": 2**64})
    got = S.bound(PS.PEARL_PLUS, Fraction(1, 100), 0, names, {"q": 2**64, "N": 2**20})
    assert got.term == "Pouw.PearlC.δPearlC" and got.delta_scheme == Fraction(2**64 + 2**20, 2**128)
    with pytest.raises(ValueError, match="charges are"):
        S.bound(PS.PEARL_PLUS, Fraction(1, 100), 0, {**names, "rebuilt": 0}, {"q": 1, "N": 1})


CTX = {"settings": {"challenger": "127.0.0.1:1", "kernel": "k.so"}}


class _Exchange:
    def __init__(self) -> None:
        self.subscribed: dict[str, list] = {}
        self.filed: list = []

    def subscribe(self, name: str, hook: str, fn) -> None:
        assert name == "pearl-plus" and hook in PS.PEARL_PLUS.hooks
        self.subscribed.setdefault(hook, []).append(fn)

    def request(self, name: str, request) -> None:
        self.filed.append((name, request))

    def emit(self, event) -> None:
        for fn in self.subscribed.get(event.hook, ()):
            fn(event)


class _Run:
    def __init__(self) -> None:
        self.details: dict = {"max_step_tokens": 64}
        self.calls: list = []
        self.passes = 0

    def gemm(self, w: int, x, weight, bias):
        self.calls.append((w, x, weight, bias))
        return "y"

    def end_pass(self, step: bool = False) -> None:
        self.passes += 1


def _due(fmt: str = "bf16") -> H.BlockDue:
    return H.BlockDue("model.layers.0.qkv", 3, {"n": 16, "k": 16}, fmt, {"x": "x", "weight": "w", "bias": None},
                      weights={"weight": "model.layers.0.qkv.weight"})


def _attached(monkeypatch) -> tuple[_Exchange, _Run]:
    ex, run = _Exchange(), _Run()
    monkeypatch.setattr(PS.Attached, "keyed", lambda self, event: {3: "w"})
    monkeypatch.setattr(PS.Attached, "open", lambda self, weights, own: run)
    on = PS.PEARL_PLUS.attach(ex, CTX)
    assert on.record() == {"entry": PS.OPTION, "commitment": None}
    ex.emit(H.RegisteredValues((("model.layers.0.qkv.weight", "w"),), linear=frozenset({"model.layers.0.qkv.weight"})))
    return ex, run


def test_attach_computes_each_linear_block_with_its_run(monkeypatch) -> None:
    ex, run = _attached(monkeypatch)
    [(name, req)] = ex.filed
    assert name == "pearl-plus" and isinstance(req, H.ComputeBlockWith)
    assert req.kernel(_due(), lambda given: "stock") == "y"
    assert run.calls == [(3, "x", "w", None)]
    with pytest.raises(ValueError, match="e4m3"):
        req.kernel(_due("e4m3"), lambda given: "stock")


def test_attach_ends_a_pass_at_each_step_and_refuses_a_step_over_one(monkeypatch) -> None:
    ex, run = _attached(monkeypatch)
    ex.emit(H.StepBegan(0, ("a",), 64))
    ex.emit(H.StepEnded(0))
    assert run.passes == 1
    with pytest.raises(RuntimeError, match="65 tokens is over one pass"):
        ex.emit(H.StepBegan(1, ("a",), 65))


def test_attach_refuses_without_coins_or_a_challenger() -> None:
    for ctx in ({}, {"settings": {"kernel": "k.so"}}):
        with pytest.raises(ValueError, match="neither 'coins' nor 'challenger' is among its settings"):
            PS.PEARL_PLUS.attach(_Exchange(), ctx)
    with pytest.raises(ValueError, match="no 'kernel' among its settings"):
        PS.PEARL_PLUS.attach(_Exchange(), {"settings": {"challenger": "127.0.0.1:1"}})


def test_opening_the_run_names_what_is_missing() -> None:
    ex = _Exchange()
    PS.PEARL_PLUS.attach(ex, CTX)
    with pytest.raises(NotImplementedError, match="vllm builds each linear weight's block key in RegisteredValues"):
        ex.emit(H.RegisteredValues((("w", "w"),), linear=frozenset({"w"})))


class _Kernel:
    @staticmethod
    def load(path, cubin=None):
        return ("kernel", path, cubin)


class _Coins:
    commitment = "commitment"

    def __init__(self) -> None:
        self.opened: list = []

    def open(self, label, root):
        self.opened.append((label, root))
        return "opening"


class _DeviceRun(_Run):
    def __init__(self, scheme, entry, kernel, weights, roots, coins, opening, **kw) -> None:
        super().__init__()
        self.made = (scheme.name, entry, kernel, weights, roots, coins, opening, kw)
        self.commits: dict = {}


class _Device:
    PearlCDeviceRun = _DeviceRun

    @staticmethod
    def weight_roots(kernel, weights, scheme):
        return {w: bytes([w]) * 64 for w in weights}

    @staticmethod
    def step_tokens(run, calls):
        return 128


def test_open_runs_the_handed_in_device_run_on_the_salt_coin(monkeypatch) -> None:
    monkeypatch.setattr(SV, "PearlCKernel", _Kernel, raising=False)
    monkeypatch.setattr(T, "_pearl_c", lambda: _Device)
    coins = _Coins()
    run = PS.Attached(CTX).open({0: "w0", 1: "w1"}, {"kernel": "k.so", "coins": coins})
    roots = {0: bytes([0]) * 64, 1: bytes([1]) * 64}
    assert coins.opened == [("salt", SV.weights_root(roots))]
    assert run.made == ("pearl-c-h100-v1-h3", PS.OPTION, ("kernel", "k.so", None), {0: "w0", 1: "w1"}, roots,
                        "commitment", "opening", {"pass_index": None})
    assert run.details["max_step_tokens"] == 128


def test_open_refuses_without_the_hand_in(monkeypatch) -> None:
    monkeypatch.setattr(T, "_pearl_c", None)
    with pytest.raises(NotImplementedError, match="use_pearl_c"):
        PS.Attached(CTX).open({0: "w0"}, {"kernel": "k.so", "coins": _Coins()})


def test_tie_uniform_is_its_one_setting() -> None:
    assert S.settings(PS.PEARL_PLUS, {}) == {"tie-uniform": False}
    assert S.settings(PS.PEARL_PLUS, {"tie-uniform": True}) == {"tie-uniform": True}
    with pytest.raises(ValueError, match="tie-uniform is a bool"):
        S.settings(PS.PEARL_PLUS, {"tie-uniform": "yes"})


def test_requests_eager_one_rank_and_steps_in_turn() -> None:
    assert H.unmet([("pearl-plus", r) for r in PS.PEARL_PLUS.requests({})],
                   {"eager": True, "world": 1, "async_steps": False}) == []


def test_reports_are_the_runs_commitment_and_details(monkeypatch) -> None:
    ex, run, finished = _Exchange(), _Run(), []
    run.commits, run.finish = {"salt_opening": "o"}, lambda: finished.append(True)
    monkeypatch.setattr(PS.Attached, "keyed", lambda self, event: {3: "w"})
    monkeypatch.setattr(PS.Attached, "open", lambda self, weights, own: run)
    on = PS.PEARL_PLUS.attach(ex, CTX)
    ex.emit(H.RegisteredValues((("w", "w"),), linear=frozenset({"w"})))
    reports: H.Reports = on
    assert reports.record() == {"entry": PS.OPTION, "commitment": {"salt_opening": "o"}}
    assert reports.summary() == {"max_step_tokens": 64}
    on.release()
    assert finished == [True]


def test_a_challenger_opens_one_session_for_the_servers_life(monkeypatch) -> None:
    opened, finished = [], []
    monkeypatch.setattr(SV, "PearlCKernel", _Kernel, raising=False)
    monkeypatch.setattr(T, "_pearl_c", lambda: _Device)
    monkeypatch.setattr(SV, "salt_session", lambda entry, settings, root, opens: (
        opened.append((entry, settings.challenger, root, opens)) or ("session", "c", "o", b"salt", {"gate": 1}, 0)),
        raising=False)
    monkeypatch.setattr(SV, "finish_window", lambda run, session: finished.append(session), raising=False)
    on = PS.Attached(CTX)
    on.run = on.open({0: "w0"}, CTX["settings"])
    assert opened == [(PS.OPTION, "127.0.0.1:1", SV.weights_root({0: bytes([0]) * 64}), True)]
    assert on.run.made[2] == ("kernel", "k.so", None) and on.run.made[5:] == ("c", "o", {"pass_index": 0})
    assert on.run.commits == {"coin_source": SV.H1_CHALLENGER, "session": {"gate": 1}}
    on.release()
    assert finished == ["session"]
