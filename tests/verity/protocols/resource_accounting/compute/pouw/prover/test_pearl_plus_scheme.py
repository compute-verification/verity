"""Pearl+ as a resource scheme (`verity.protocols.resource_accounting.compute.pouw.scheme`). Its block cost against the
run verifier's credit is `tests/benchmarks/pouw/test_pearl_plus_scheme_credit.py`, beside the verifier's other tests."""
from __future__ import annotations

import json
from fractions import Fraction
from types import SimpleNamespace

import pytest

from verity import hooks as H
from verity.protocols.resource_accounting import scheme as S
from verity.protocols.resource_accounting.compute.pouw import scheme as PS
from verity.protocols.resource_accounting.compute.pouw.prover import serving as SV
from verity.protocols.resource_accounting.compute.pouw.prover.serving import tensors as T
from verity.protocols.resource_accounting.compute.pouw.prover.serving.h1 import window as Wn
from verity.protocols.resource_accounting.compute.pouw.verifier import salt as PSALT
from verity.protocols.resource_accounting.compute.pouw.prover.circuit.pearl_c_tile import PearlCTileRead
from verity.protocols.resource_accounting.compute.pouw.prover.circuit.pearl_plus_tile import PearlPlusTile
from verity.protocols.resource_accounting.compute.pouw.prover.schemes.pearl_c import PearlC


def test_found_by_name() -> None:
    assert S.SCHEMES["pearl-plus"] is PS.PEARL_PLUS
    assert {b.definition for b in PS.PEARL_PLUS.blocks} == {PearlPlusTile.id, PearlCTileRead.id}
    assert PS.PEARL_PLUS.option == f"pouw:{PearlC().name}"


def test_it_declares_the_cut_and_is_the_one_scheme_that_does() -> None:
    from verity.protocols.compose import schemes

    assert [s.name for s in schemes.load().values() if "cut" in s.provides] == ["pearl-plus"]


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


def _gate(*words):
    return {"session": words[2], "root": words[4], "commitment": "c"} if words[0] == "open" else {}


SESSION = {"session": "s-1"}
CTX = {"settings": {"challenger": "127.0.0.1:1", "session": SESSION, "kernel": "k.so"}, "gate": _gate,
       "private": "/nonexistent/pearl-plus"}


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
        self.commits: dict = {}
        self.calls: list = []
        self.passes = 0
        self.finished = False

    def finish(self) -> None:
        self.finished = True

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
    assert req.kernel(_due(), lambda given: given or "stock") == "stock"
    ex.emit(H.StepBegan(0))
    assert req.kernel(_due(), lambda given: "stock") == "stock"
    ex.emit(H.StepBegan(1, ("a",), 8))
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


def _phased(monkeypatch) -> tuple[_Exchange, list[_Run], PS.Attached]:
    ex, runs = _Exchange(), []
    monkeypatch.setattr(PS.Attached, "open", lambda self, weights, own: runs.append(_Run()) or runs[-1])
    on = PS.PEARL_PLUS.attach(ex, CTX)
    ex.emit(H.RegisteredValues((("w", "w"),), operands={"w": 3}))
    return ex, runs, on


def test_each_phase_is_one_epoch_and_its_root_may_come_after_the_next_opens(monkeypatch) -> None:
    ex, runs, on = _phased(monkeypatch)
    [(_, req)] = ex.filed
    p0, p1, root = b"\x01" * 64, b"\x02" * 64, b"\x0a" * 64
    ex.emit(H.PhaseOpened(0, 0))
    ex.emit(H.StepBegan(0, ("a",), 8))
    assert req.kernel(_due(), lambda given: "stock") == "y"
    ex.emit(H.StepComputed(0))
    ex.emit(H.StepEnded(0))
    ex.emit(H.PhaseClosed(0, 0, p0))
    assert runs[0].finished and runs[0].passes == 1 and on.run is None
    ex.emit(H.StepBegan(1, ("a",), 8))
    assert req.kernel(_due(), lambda given: "stock") == "stock" and len(runs) == 1
    ex.emit(H.StepEnded(1))
    ex.emit(H.PhaseOpened(1, 2))
    ex.emit(H.PhaseCommitted(0, root))
    assert on.run is runs[1] and not runs[1].finished
    ex.emit(H.StepBegan(2, ("a",), 8))
    assert req.kernel(_due(), lambda given: "stock") == "y" and runs[1].calls == [(3, "x", "w", None)]
    ex.emit(H.StepComputed(2))
    ex.emit(H.StepEnded(2))
    ex.emit(H.PhaseClosed(1, 2, p1))
    with pytest.raises(RuntimeError, match="phase 0's root comes twice"):
        ex.emit(H.PhaseCommitted(0, root))
    with pytest.raises(RuntimeError, match="phase 2's root comes before the phase closed"):
        ex.emit(H.PhaseCommitted(2, root))
    on.release()
    assert on.commitment() == {"phases": {
        "0": {"commitment": {}, "first_step": 0, "last_step": 0, "program": p0.hex(), "root": root.hex()},
        "1": {"commitment": {}, "first_step": 2, "last_step": 2, "program": p1.hex(), "root": None}}}
    ex.emit(H.PhaseCommitted(1, b""))
    assert on.commitment()["phases"]["1"]["root"] == ""
    assert on.summary() == {"phases": {"0": {"max_step_tokens": 64}, "1": {"max_step_tokens": 64}}}


def test_phases_out_of_order_are_refused_and_an_unclosed_one_is_finished_at_release(monkeypatch) -> None:
    ex, runs, on = _phased(monkeypatch)
    with pytest.raises(RuntimeError, match="phase 0 closes, and the open phase is None"):
        ex.emit(H.PhaseClosed(0, 0, b""))
    ex.emit(H.PhaseOpened(0, 0))
    with pytest.raises(RuntimeError, match="phase 1 opens while phase 0 is open"):
        ex.emit(H.PhaseOpened(1, 0))
    on.release()
    assert runs[0].finished and on.commitment()["phases"]["0"]["closed"] is False
    ex, runs, on = _phased(monkeypatch)
    ex.emit(H.StepBegan(0, ("a",), 8))
    with pytest.raises(RuntimeError, match="after steps outside any phase opened its epoch for the server's life"):
        ex.emit(H.PhaseOpened(0, 1))
    on = PS.PEARL_PLUS.attach(_Exchange(), CTX)
    with pytest.raises(RuntimeError, match="phase 0 opens before its linear weights registered"):
        on.step(H.PhaseOpened(0, 0))


def test_each_phase_opens_a_salt_session_of_its_own(monkeypatch, tmp_path) -> None:
    gate, reserved = _gated(monkeypatch), []
    monkeypatch.setattr(SV, "finish_window", lambda run, session: run.finish(), raising=False)
    monkeypatch.setattr(SV, "reserve_salt", lambda call, ledger, lane, peer, epoch: reserved.append(
        (call, ledger(), lane, peer, epoch)) or ({"session": f"s-r{epoch}"}, {"period": 7}), raising=False)
    ledger = lambda: ["line"]  # noqa: E731
    on = PS.Attached({**CTX, "settings": {**CTX["settings"], "lane": "compute", "peer": "node-1"}, "ledger": ledger,
                      "private": str(tmp_path)})
    on.weights = {0: W0}
    for p in (0, 1, 2):
        on.step(H.PhaseOpened(p, p))
        on.step(H.PhaseClosed(p, p, b""))
    w = SV.weights_root({0: bytes([0]) * 64})
    assert gate.roots == [PSALT.epoch_open_root(w, k) for k in (0, 1, 2)]
    assert reserved == [(_gate, ["line"], "compute", "node-1", 1), (_gate, ["line"], "compute", "node-1", 2)]
    phases = on.commitment()["phases"]
    assert [phases[p]["reserved"] for p in "012"] == [{"session": SESSION, "period": None},
                                                     {"session": {"session": "s-r1"}, "period": {"period": 7}},
                                                     {"session": {"session": "s-r2"}, "period": {"period": 7}}]
    assert [phases[p]["commitment"]["session"]["session"] for p in "012"] == ["s-1", "s-r1", "s-r2"]
    gate.roots.clear()
    one = PS.Attached({**CTX, "private": str(tmp_path / "one")})
    (tmp_path / "one").mkdir()
    one.weights = {0: W0}
    one.step(H.PhaseOpened(0, 0))
    one.step(H.PhaseClosed(0, 0, b""))
    with pytest.raises(ValueError, match="salt epoch 1 needs a session of its own.*give 'lane' and 'peer'"):
        one.step(H.PhaseOpened(1, 1))


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
    with pytest.raises(ValueError, match=r"its launcher passes ctx\['private'\].*must survive a restart"):
        PS.PEARL_PLUS.attach(_Exchange(), {"settings": CTX["settings"], "gate": _gate})
    with pytest.raises(ValueError, match=r"with a 'lane', its launcher passes ctx\['ledger'\]"):
        PS.PEARL_PLUS.attach(_Exchange(), {**CTX, "settings": {"challenger": "127.0.0.1:1", "lane": "l", "peer": "p",
                                                               "kernel": "k.so"}})
    PS.PEARL_PLUS.attach(_Exchange(), {"settings": {"coins": "in-process", "kernel": "k.so"}})
    with pytest.raises(ValueError, match="no 'kernel' among its settings"):
        PS.PEARL_PLUS.attach(_Exchange(), {"settings": {"coins": "in-process"}})


def test_a_capture_is_refused_by_name_until_what_it_records_is_settled() -> None:
    with pytest.raises(NotImplementedError, match=r"Match capture \(ctx\['stage'\] 'capture'\) records rather than commits"):
        PS.PEARL_PLUS.attach(_Exchange(), {**IN_PROCESS, "stage": "capture"})
    PS.PEARL_PLUS.attach(_Exchange(), {**IN_PROCESS, "stage": "commit"})


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
    salted: list = []

    @staticmethod
    def weight_roots(kernel, weights, scheme, *, weight_salts):
        assert {w: len(s) for w, s in weight_salts.items()} == {w: len(SV.fresh_salts(1)[0]) * t.shape[0] for w, t in weights.items()}
        _Device.salted.append(weight_salts)
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
    monkeypatch.setattr(SV, "in_process_coins", lambda window: window in (0, 1) and coins)
    monkeypatch.setattr(T, "_pearl_c", lambda: _Device)
    on = PS.Attached(IN_PROCESS)
    run = on.open({0: W0, 1: W1}, on.own)
    roots = {0: bytes([0]) * 64, 1: bytes([1]) * 64}
    assert _in_domain == [("pearl-c-h100-v1-h3", {0: (256, 512), 1: (256, 512)})]
    assert coins.opened == [("salt", SV.weights_root(roots))]
    assert run.made == ("pearl-c-h100-v1-h3", PS.OPTION, ("kernel", "k.so", None), {0: W0, 1: W1}, roots,
                        "commitment", "opening", {"pass_index": None, "schedule": "serial",
                                                  "weight_salts": _Device.salted[-1]})
    again = on.open({0: W0, 1: W1}, on.own)
    assert again.made[7]["weight_salts"][0] != run.made[7]["weight_salts"][0]
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
                           ({"challenger": "127.0.0.1:1"}, "a challenger needs a session"),
                           ({**CTX["settings"], "lane": "compute"}, "lane and peer go together"),
                           ({"session": SESSION}, "session, lane and peer are a challenger's"),
                           ({"challenger": "127.0.0.1:1", "session": {"id": 1}}, "session is the gate's reserve answer")):
        with pytest.raises(ValueError, match=f"pearl-plus: {refused}"):
            S.settings(PS.PEARL_PLUS, given)


def test_requests_eager_and_one_rank() -> None:
    assert H.unmet([("pearl-plus", r) for r in PS.PEARL_PLUS.requests({})], {"eager": True, "world": 1}) == []


def test_reports_are_the_runs_commitment_and_details(monkeypatch) -> None:
    ex, run = _Exchange(), _Run()
    run.commits = {"salt_opening": "o"}
    monkeypatch.setattr(PS.Attached, "open", lambda self, weights, own: run)
    on = PS.PEARL_PLUS.attach(ex, CTX)
    ex.emit(H.RegisteredValues((("w", "w"),), operands={"w": 0}))
    assert on.commitment() == {} and on.run is None
    ex.emit(H.StepBegan(0, ("a",), 8))
    assert on.commitment() == {"salt_opening": "o"}
    assert on.record() == {"entry": PS.OPTION, "schedule": "serial", "tie-uniform": False}
    assert on.summary() == {"max_step_tokens": 64}
    on.release()
    assert run.finished and on.run is None
    assert on.commitment() == {"salt_opening": "o"} and on.summary() == {"max_step_tokens": 64}


def test_a_challenger_opens_one_session_for_the_servers_life(monkeypatch, tmp_path) -> None:
    opened, finished = [], []
    monkeypatch.setattr(SV, "PearlCKernel", _Kernel, raising=False)
    monkeypatch.setattr(T, "_pearl_c", lambda: _Device)
    monkeypatch.setattr(SV, "salt_session", lambda entry, settings, root, opens, epoch: (
        opened.append((entry, settings.challenger, settings.session, settings.gate, root, opens, epoch))
        or ("session", "c", "o", b"salt", {"gate": 1}, epoch)), raising=False)
    monkeypatch.setattr(SV, "finish_window", lambda run, session: finished.append(session), raising=False)
    on = PS.Attached({**CTX, "private": str(tmp_path)})
    on.run = on.open({0: W0}, on.own)
    [(entry, at, session, gate, root, opens, epoch)] = opened
    w = SV.weights_root({0: bytes([0]) * 64})
    assert (entry, at, session, root, opens, epoch) == (PS.OPTION, "127.0.0.1:1", SESSION, w, True, 0)
    assert gate("s-1", "ab") == {"session": "s-1", "root": "ab", "commitment": "c"}
    assert on.run.made[2] == ("kernel", "k.so", None)
    assert on.run.made[5:] == ("c", "o", {"pass_index": 0, "schedule": "serial", "weight_salts": _Device.salted[-1]})
    assert on.run.commits["coin_source"] == SV.H1_CHALLENGER and on.run.commits["session"] == {"gate": 1}
    on.release()
    assert finished == ["session"]


class _Gate:
    """The auditor's gate as `window.opened` meets it: one session a root, ever, whichever server asks."""

    def __init__(self) -> None:
        self.roots: list[bytes] = []

    def opened(self, entry, at, steps, gate, sid, root, named, layout=None, **kw):
        if root in self.roots:
            raise ValueError(f"root {root.hex()[:16]}… already has its session in the ledger: one session a root, ever")
        self.roots.append(root)
        return "client", "c", SimpleNamespace(coin=b"coin"), {"session": sid, "root": root.hex()}


def _gated(monkeypatch) -> _Gate:
    gate = _Gate()
    monkeypatch.setattr(Wn, "opened", gate.opened)
    monkeypatch.setattr(Wn, "salt_of", lambda c, coin, root: b"salt")
    monkeypatch.setattr(SV, "PearlCKernel", _Kernel, raising=False)
    monkeypatch.setattr(T, "_pearl_c", lambda: _Device)
    return gate


def test_a_server_after_a_killed_one_opens_its_session_on_a_root_not_used(monkeypatch, tmp_path) -> None:
    gate, weights = _gated(monkeypatch), {0: W0}
    w = SV.weights_root({0: bytes([0]) * 64})
    killed = PS.Attached({**CTX, "private": str(tmp_path)})
    killed.run = killed.open(weights, killed.own)
    Wn.EPOCHS.clear()
    later = PS.Attached({**CTX, "private": str(tmp_path)})
    later.run = later.open(weights, later.own)
    assert gate.roots == [PSALT.epoch_open_root(w, 0), PSALT.epoch_open_root(w, 1)]
    assert later.run.made[7]["pass_index"] == 1 and (tmp_path / PS.EPOCH_FILE).read_text() == "1\n"
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    with pytest.raises(ValueError, match="one session a root, ever"):
        PS.Attached({**CTX, "private": str(fresh)}).open(weights, PS.Attached(CTX).own)


def test_the_epoch_number_is_spent_on_disk_before_its_session_opens(monkeypatch, tmp_path) -> None:
    _gated(monkeypatch)

    def refused(entry, at, steps, gate, sid, root, named, layout=None, **kw):
        assert (tmp_path / PS.EPOCH_FILE).read_text() == "0\n"
        raise ConnectionError("the challenger is unreachable")

    monkeypatch.setattr(Wn, "opened", refused)
    on = PS.Attached({**CTX, "private": str(tmp_path)})
    with pytest.raises(ConnectionError):
        on.open({0: W0}, on.own)
    assert PS.spend_epoch(tmp_path) == 1 and (tmp_path / PS.EPOCH_FILE).read_text() == "1\n"
    for torn in ("", "1", "x\n", f"{2**64 - 1}\n"):
        (tmp_path / PS.EPOCH_FILE).write_text(torn)
        with pytest.raises(ValueError, match="refused rather than risk a number already used"):
            PS.spend_epoch(tmp_path)


def test_the_plugins_report_reads_as_the_served_verdicts_read_served_json(tmp_path) -> None:
    session = {"session": "s-1", "seq": 4, "root": "ab" * 64, "commitment": "cd" * 64, "host": "auditor.s-1"}
    w = SV.weights_root({0: bytes([0]) * 64})
    reports = {"configuration": {"scheme": "pearl-plus", "settings": {**PS.SETTINGS, **CTX["settings"]},
                                 "record": {"entry": PS.OPTION, "schedule": "serial", "tie-uniform": False}},
               "summary": {"max_step_tokens": 64, "passes": [{"units": 3, "tiles": 9, "transcript_root": "ef"}],
                           "challenger": {"closed": True}},
               "commitment": {"coin_source": SV.H1_CHALLENGER, "session": session, "pass_index": 2,
                              "weights_root": w.hex(), "salt": "00" * 32}}
    for name, doc in reports.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(doc))
    got = PS.served_record(tmp_path)
    assert got["arm"] == PS.OPTION and got["retained"] is False and got["max_step_tokens"] == 64
    assert got["passes"] == [{"pass": 0, "timed": True, "units": 3, "tiles": 9, "transcript_root": "ef", "macs": None}]
    coins = got["coins"]
    assert (coins["source"], coins["reserved"], coins["pass_index"], coins["closed"]) == (SV.H1_CHALLENGER, SESSION, 2,
                                                                                          {"closed": True})
    assert coins["salt_after"] == PSALT.epoch_open_root(w, 2).hex()
    assert coins["manifest_head"] == {"session": {k: session[k] for k in ("session", "seq", "root", "commitment")},
                                      "pass_index": 2}
