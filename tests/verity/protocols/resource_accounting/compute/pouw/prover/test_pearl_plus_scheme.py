"""Pearl+ as a resource scheme (`verity.protocols.resource_accounting.compute.pouw.scheme`). Its block cost against the
run verifier's credit is `tests/benchmarks/pouw/test_pearl_plus_scheme_credit.py`, beside the verifier's other tests."""
from __future__ import annotations

from fractions import Fraction

import pytest

from verity import hooks as H
from verity.protocols.resource_accounting import scheme as S
from verity.protocols.resource_accounting.compute.pouw import scheme as PS
from verity.protocols.resource_accounting.compute.pouw.prover.circuit.pearl_c_tile import PearlCTileRead
from verity.protocols.resource_accounting.compute.pouw.prover.circuit.pearl_plus_tile import PearlPlusTile
from verity.protocols.resource_accounting.compute.pouw.prover.schemes.pearl_c import PearlC


def test_found_by_name() -> None:
    assert S.SCHEMES["pearl-plus"] is PS.PEARL_PLUS
    assert {b.definition for b in PS.PEARL_PLUS.blocks} == {PearlPlusTile.id, PearlCTileRead.id}
    assert PS.PEARL_PLUS.option == f"pouw:{PearlC().name}"


def test_bound_refuses_a_charge_it_does_not_state() -> None:
    names = {c.name: 0 for c in PS.PEARL_PLUS.charges}
    with pytest.raises(ValueError, match="pearl-plus: its delta_scheme .* is not said"):
        S.bound(PS.PEARL_PLUS, Fraction(1, 100), 0, names)
    with pytest.raises(ValueError, match="charges are"):
        S.bound(PS.PEARL_PLUS, Fraction(1, 100), 0, {**names, "rebuilt": 0})


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
    monkeypatch.setattr(PS.Attached, "open", lambda self, weights: run)
    PS.PEARL_PLUS.attach(ex, {})
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


def test_opening_the_run_names_what_is_missing() -> None:
    ex = _Exchange()
    PS.PEARL_PLUS.attach(ex, {})
    with pytest.raises(NotImplementedError, match="block key in RegisteredValues"):
        ex.emit(H.RegisteredValues((("w", "w"),), linear=frozenset({"w"})))


def test_tie_uniform_is_its_one_setting() -> None:
    assert S.settings(PS.PEARL_PLUS, {}) == {"tie-uniform": False}
    assert S.settings(PS.PEARL_PLUS, {"tie-uniform": True}) == {"tie-uniform": True}
    with pytest.raises(ValueError, match="tie-uniform is a bool"):
        S.settings(PS.PEARL_PLUS, {"tie-uniform": "yes"})


def test_requests_eager_one_rank_and_steps_in_turn() -> None:
    assert H.unmet([("pearl-plus", r) for r in PS.PEARL_PLUS.requests({})],
                   {"eager": True, "world": 1, "async_steps": False}) == []
