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
        S.bound(PS.PEARL_PLUS, Fraction(1, 100), 0, names, {"q": 2**64}, epsilon=Fraction(0))
    got = S.bound(PS.PEARL_PLUS, Fraction(1, 100), 0, names, {"q": 2**64, "N": 2**20}, epsilon=Fraction(1, 2**40))
    assert got.term == "Pouw.PearlC.δPearlC" and got.delta_scheme == Fraction(2**64 + 2**20, 2**128)
    assert got.epsilon == Fraction(1, 2**40)
    with pytest.raises(ValueError, match="charges are"):
        S.bound(PS.PEARL_PLUS, Fraction(1, 100), 0, {**names, "rebuilt": 0}, {"q": 1, "N": 1}, epsilon=Fraction(0))
    with pytest.raises(ValueError, match="the audit's error ε"):
        S.bound(PS.PEARL_PLUS, Fraction(1, 100), 0, names, {"q": 1, "N": 1})


def _gate(sid, root):
    return {}


SESSION = {"session": "s-1"}
CTX = {"settings": {"challenger": "127.0.0.1:1", "session": SESSION, "kernel": "k.so"}, "gate": _gate}


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
    monkeypatch.setattr(PS.Attached, "open", lambda self, weights, own: (weights == {3: "w"} or 1 / 0) and run)
    on = PS.PEARL_PLUS.attach(ex, CTX)
    assert on.commitment() == {}
    ex.emit(H.RegisteredValues((("model.layers.0.qkv.weight", "w"), ("model.norm.weight", "n")),
                               operands={"model.layers.0.qkv.weight": 3}))
    return ex, run


def test_attach_computes_each_linear_block_with_its_run(monkeypatch) -> None:
    ex, run = _attached(monkeypatch)
    [(name, req)] = ex.filed
    assert name == "pearl-plus" and isinstance(req, H.ComputeBlockWith)
    assert req.kernel(_due(), lambda given: "stock") == "y"
    assert run.calls == [(3, "x", "w", None)]
    with pytest.raises(ValueError, match="e4m3"):
        req.kernel(_due("e4m3"), lambda given: "stock")


def test_a_pass_ends_at_each_steps_computed_and_a_step_over_one_is_refused(monkeypatch) -> None:
    ex, run = _attached(monkeypatch)
    ex.emit(H.StepBegan(0, ("a",), 64))
    ex.emit(H.StepBegan(1, ("a",), 64))
    ex.emit(H.StepComputed(0))
    ex.emit(H.StepEnded(0))
    ex.emit(H.StepComputed(1))
    ex.emit(H.StepEnded(1))
    ex.emit(H.StepBegan(2))
    ex.emit(H.StepEnded(2))
    assert run.passes == 2
    with pytest.raises(RuntimeError, match="65 tokens is over one pass"):
        ex.emit(H.StepBegan(3, ("a",), 65))


def test_a_step_that_computed_without_steps_computed_is_refused(monkeypatch) -> None:
    ex, run = _attached(monkeypatch)
    ex.emit(H.StepBegan(0, ("a",), 8))
    with pytest.raises(RuntimeError, match="step 0 computed and ended with no StepComputed.*multiprocess executor"):
        ex.emit(H.StepEnded(0))
    assert run.passes == 0


def test_the_weight_operands_are_keyed_by_their_blocks() -> None:
    on = PS.Attached(CTX)
    event = H.RegisteredValues((("a", "A"), ("b", "B"), ("norm", "N")), operands={"a": 1, "b": 0})
    assert on.keyed(event) == {1: "A", 0: "B"}
    with pytest.raises(ValueError, match="name no weight operand"):
        on.keyed(H.RegisteredValues((("a", "A"),)))
    with pytest.raises(ValueError, match=r"weight operands \['c'\] are not among the registered values"):
        on.keyed(H.RegisteredValues((("a", "A"),), operands={"a": 0, "c": 1}))


def test_attach_refuses_without_coins_or_a_challenger_its_gate_or_a_kernel() -> None:
    for ctx in ({}, {"settings": {"kernel": "k.so"}}):
        with pytest.raises(ValueError, match="neither 'coins' nor 'challenger' is among its settings"):
            PS.PEARL_PLUS.attach(_Exchange(), ctx)
    with pytest.raises(ValueError, match=r"its launcher passes ctx\['gate'\]"):
        PS.PEARL_PLUS.attach(_Exchange(), {"settings": CTX["settings"]})
    with pytest.raises(ValueError, match="no 'kernel' among its settings"):
        PS.PEARL_PLUS.attach(_Exchange(), {"settings": {"coins": "in-process"}})


def test_attach_refuses_an_undeclared_setting() -> None:
    with pytest.raises(ValueError, match=r"pearl-plus has no setting \['runner'\]"):
        PS.PEARL_PLUS.attach(_Exchange(), {**CTX, "settings": {**CTX["settings"], "runner": "r"}})


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


IN_PROCESS = {"settings": {"coins": "in-process", "kernel": "k.so"}}


class _Weight:
    def __init__(self, name: str, shape=(256, 512)) -> None:
        self.name, self.shape = name, shape

    def __repr__(self) -> str:
        return self.name


W0, W1 = _Weight("w0"), _Weight("w1")


@pytest.fixture(autouse=True)
def _in_domain(monkeypatch) -> list:
    seen: list = []
    monkeypatch.setattr(SV, "pearl_c_refusals", lambda name, shapes: seen.append((name, shapes)) or [], raising=False)
    return seen


def test_open_runs_the_handed_in_device_run_on_the_salt_coin(monkeypatch, _in_domain) -> None:
    coins = _Coins()
    monkeypatch.setattr(SV, "PearlCKernel", _Kernel, raising=False)
    monkeypatch.setattr(SV, "in_process_coins", lambda window: window == 0 and coins)
    monkeypatch.setattr(T, "_pearl_c", lambda: _Device)
    on = PS.Attached(IN_PROCESS)
    run = on.open({0: W0, 1: W1}, on.own)
    roots = {0: bytes([0]) * 64, 1: bytes([1]) * 64}
    assert _in_domain == [("pearl-c-h100-v1-h3", {0: (256, 512), 1: (256, 512)})]
    assert coins.opened == [("salt", SV.weights_root(roots))]
    assert run.made == ("pearl-c-h100-v1-h3", PS.OPTION, ("kernel", "k.so", None), {0: W0, 1: W1}, roots,
                        "commitment", "opening", {"pass_index": None, "schedule": "serial"})
    assert run.commits["registration"] == PS.registration(SV.pearl_c_scheme("pearl-c-h100-v1-h3"), SV.DomainError)
    assert run.commits["registration"]["claims"] == PS.UNREGISTERED
    assert run.details["max_step_tokens"] == 128


def test_unregistered_weights_say_their_claim_is_about_nothing() -> None:
    class Refusing:
        def check_registration(self, transforms, curated, needs_gamma_fold):
            raise SV.DomainError("not curated")

    assert PS.registration(Refusing(), SV.DomainError) == {
        "transforms": [], "curated": False, "registered": False, "claims": PS.UNREGISTERED, "refused": "not curated"}


def test_open_refuses_without_the_hand_in(monkeypatch) -> None:
    monkeypatch.setattr(T, "_pearl_c", None)
    on = PS.Attached(IN_PROCESS)
    with pytest.raises(NotImplementedError, match="use_pearl_c"):
        on.open({0: W0}, on.own)


def test_weights_outside_the_kernels_domain_are_refused_before_any_commitment(monkeypatch) -> None:
    monkeypatch.setattr(SV, "pearl_c_refusals", lambda name, shapes: [f"linear {w}: (n, k) = {s}" for w, s in shapes.items()],
                        raising=False)
    monkeypatch.setattr(T, "_pearl_c", lambda: _Device)
    on = PS.Attached(IN_PROCESS)
    with pytest.raises(ValueError, match=r"refused before any commitment: linear 0: \(n, k\) = \(3, 5\)"):
        on.open({0: _Weight("w0", (3, 5))}, on.own)


def test_its_settings_are_declared_and_checked() -> None:
    assert S.settings(PS.PEARL_PLUS, {}) == PS.SETTINGS
    assert S.settings(PS.PEARL_PLUS, {"tie-uniform": True})["tie-uniform"] is True
    with pytest.raises(ValueError, match="tie-uniform is a bool"):
        S.settings(PS.PEARL_PLUS, {"tie-uniform": "yes"})
    for given, refused in (({"schedule": "sideways"}, "schedule is one of serial"),
                           ({"coins": "beacon"}, "coins is 'in-process'"),
                           ({"coins": "in-process", **CTX["settings"]}, "coins and challenger are both set"),
                           ({"challenger": "nowhere", "session": SESSION}, "challenger is the auditor device's"),
                           ({"challenger": "127.0.0.1:1"}, "challenger and session go together"),
                           ({"challenger": "127.0.0.1:1", "session": {"id": 1}}, "session is the gate's reserve answer")):
        with pytest.raises(ValueError, match=f"pearl-plus: {refused}"):
            S.settings(PS.PEARL_PLUS, given)


def test_requests_eager_and_one_rank() -> None:
    assert H.unmet([("pearl-plus", r) for r in PS.PEARL_PLUS.requests({})], {"eager": True, "world": 1}) == []


def test_reports_are_the_runs_commitment_and_details(monkeypatch) -> None:
    ex, run, finished = _Exchange(), _Run(), []
    run.commits, run.finish = {"salt_opening": "o"}, lambda: finished.append(True)
    monkeypatch.setattr(PS.Attached, "open", lambda self, weights, own: run)
    on = PS.PEARL_PLUS.attach(ex, CTX)
    ex.emit(H.RegisteredValues((("w", "w"),), operands={"w": 0}))
    reports: H.Commits = on
    assert reports.commitment() == {"salt_opening": "o"}
    assert reports.record() == {"entry": PS.OPTION, "schedule": "serial", "tie-uniform": False}
    assert reports.summary() == {"max_step_tokens": 64}
    on.release()
    assert finished == [True]


def test_a_challenger_opens_one_session_for_the_servers_life(monkeypatch) -> None:
    opened, finished = [], []
    monkeypatch.setattr(SV, "PearlCKernel", _Kernel, raising=False)
    monkeypatch.setattr(T, "_pearl_c", lambda: _Device)
    monkeypatch.setattr(SV, "salt_session", lambda entry, settings, root, opens: (
        opened.append((entry, settings.challenger, settings.session, settings.gate, root, opens))
        or ("session", "c", "o", b"salt", {"gate": 1}, 0)), raising=False)
    monkeypatch.setattr(SV, "finish_window", lambda run, session: finished.append(session), raising=False)
    on = PS.Attached(CTX)
    on.run = on.open({0: W0}, on.own)
    assert opened == [(PS.OPTION, "127.0.0.1:1", SESSION, _gate, SV.weights_root({0: bytes([0]) * 64}), True)]
    assert on.run.made[2] == ("kernel", "k.so", None)
    assert on.run.made[5:] == ("c", "o", {"pass_index": 0, "schedule": "serial"})
    assert on.run.commits["coin_source"] == SV.H1_CHALLENGER and on.run.commits["session"] == {"gate": 1}
    on.release()
    assert finished == ["session"]
