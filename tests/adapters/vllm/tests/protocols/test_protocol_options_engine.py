"""`verity_vllm.protocols` at a built engine: whose settings the engine fills and the refusal where it cannot, a
composition's hooks taken off before its close, an option closed before its composition, a Match capture's install, a
Commit's install, and which entry binds a row's Build.

CPU only.  ``exec`` and ``hold`` are `test_protocol_options`'s stand-ins over its toy linear layers; SPOT's
adapter is the real one.
"""
from __future__ import annotations

from typing import Any

import pytest

from tests.protocols.test_protocol_options import (
    ALL,
    Model,
    _fake_adapter,
    _hold,
    _hold_between,  # noqa: F401  (autouse: `hold` in its place in config.PROTOCOLS)
    _exec,
    _serve,
)
from verity_vllm import protocols as PO
from verity_vllm.engine import hooks
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.protocols import spot
from verity_vllm.protocols.interface import BUILD_BOUND

EXEC, HOLD, SPOT = ALL


def _captured() -> TargetProfile:
    """Built in the test, where `_hold_between` has put ``exec`` in `config.PROTOCOLS`."""
    return TargetProfile.default().with_overrides(protocols=(EXEC, SPOT), compilation="piecewise")


def _both(monkeypatch, log):
    """The two stand-in adapters, each under the scheme `ALL` names."""
    ex = _fake_adapter(monkeypatch, "exec", _exec(log), schemes=(EXEC.partition(":")[2],))
    return ex, _fake_adapter(monkeypatch, "hold", _hold(log), schemes=(HOLD.partition(":")[2],))


def test_the_engine_fills_only_an_adapter_that_reads_it_and_another_entry_keeps_its_settings(monkeypatch):
    asked = []

    def from_engine(scheme, settings, llm, epoch):
        asked.append((scheme, dict(settings), llm, epoch))
        return {"runner": "r", "rounds": 9}

    ex, _ = _both(monkeypatch, [])
    ex.from_engine = from_engine
    llm = object()
    own = {"rounds": 7, "steps": 2}
    ctx = _serve(EXEC, HOLD, settings={"exec": own, "hold": {"key": "k"}}, filled={"exec": ("runner",)})
    got = PO.from_engine(ctx, llm, epoch=3)
    assert asked == [("ncp-v1", own, llm, 3)]
    # what the adapter returns is taken, over the caller's where both name a key
    assert got.settings == {"exec": {"rounds": 9, "steps": 2, "runner": "r"}, "hold": {"key": "k"}} and got.filled == {}
    assert (got.stage, got.enabled) == ("serve", (EXEC, HOLD)) and ctx.settings["exec"] == {"rounds": 7, "steps": 2}


@pytest.mark.parametrize("error", [RuntimeError, AttributeError, ValueError])
def test_an_engine_that_cannot_supply_an_adapters_settings_is_refused_naming_every_such_entry(monkeypatch, error):
    def from_engine(scheme, settings, llm, epoch):
        raise error(f"no runner for {scheme}")

    ex, hold = _both(monkeypatch, [])
    ex.from_engine = hold.from_engine = from_engine
    with pytest.raises(PO.Refused) as e:
        PO.from_engine(_serve(EXEC, HOLD, filled={"exec": ("runner", "coins")}), object())
    assert e.value.reasons == (
        f"{EXEC}: its settings ['runner', 'coins'] come from the built engine, which doesn't supply them: no runner for ncp-v1",
        f"{HOLD}: its settings [] come from the built engine, which doesn't supply them: no runner for band-chain/d12/v1")


def test_an_error_of_another_kind_from_an_adapters_read_of_the_engine_is_raised_as_it_is(monkeypatch):
    def from_engine(scheme, settings, llm, epoch):
        raise KeyError("runner")

    ex, _ = _both(monkeypatch, [])
    ex.from_engine = from_engine
    with pytest.raises(KeyError, match="^'runner'$"):
        PO.from_engine(_serve(EXEC, HOLD), object())


def test_unhook_takes_every_options_hooks_off_and_leaves_each_release_to_the_close(monkeypatch):
    log: list[str] = []
    _both(monkeypatch, log)
    model, before = Model(), hooks.live()
    with PO.install(model, _serve(EXEC, HOLD)) as comp:
        layer = model.layers[0]
        assert layer.quant_method.apply(layer, "x") == ("exec", 2, "x") and log == ["exec", "responder up", "hold"]
        comp.unhook()
        assert log[3:] == ["responder down handle"]                          # its service stopped, nothing released
        assert hooks.live() == before and all("apply" not in vars(m.quant_method) for m in model.layers)
        assert layer.quant_method.apply(layer, "x") == ("gemm", 2, "x")
        assert not comp.closed and comp.entries == (EXEC, HOLD) and not any(opt.closed for opt in comp.installed)
        comp.unhook()                                                        # nothing is left to take off
    assert log[3:] == ["responder down handle", "hold released"] and hooks.live() == before


def test_unhook_tries_every_options_hooks_and_raises_the_first_error(monkeypatch):
    def stop(handle):
        raise RuntimeError("the responder did not stop")

    def install(scheme, target):
        h = target.hooks(f"hold:{scheme}")
        h.service("responder", lambda: "handle", stop)
        return PO.Option(name=f"hold:{scheme}", hooks=h)

    _fake_adapter(monkeypatch, "exec", _exec([]), schemes=(EXEC.partition(":")[2],))
    _fake_adapter(monkeypatch, "hold", install, schemes=(HOLD.partition(":")[2],))
    model, before = Model(), hooks.live()
    with PO.install(model, _serve(EXEC, HOLD)) as comp:                      # its close: a stop that raised is not tried again
        with pytest.raises(RuntimeError, match="^the responder did not stop$"):
            comp.unhook()
        assert hooks.live() == before and all("apply" not in vars(m.quant_method) for m in model.layers)


def test_an_option_closed_before_its_composition_is_released_once(monkeypatch):
    log: list[str] = []
    _both(monkeypatch, log)
    model, before = Model(), hooks.live()
    with PO.install(model, _serve(HOLD)) as comp:
        opt = comp.get("hold")
        opt.close()
        assert opt.closed and log == ["responder up", "hold", "responder down handle", "hold released"]
        assert hooks.live() == before
        opt.close()
    assert log.count("hold released") == 1 and log.count("responder down handle") == 1


def test_a_capture_installs_its_set_without_spot_on_the_engines_model_and_says_so(monkeypatch, capsys):
    _both(monkeypatch, [])
    model, before = Model(), hooks.live()
    comp = PO.at_capture(_captured(), model, {"exec": {"rounds": 7}, SPOT.partition(":")[0]: {}})
    try:
        assert capsys.readouterr().out == f"[m1] protocol options installed for the capture: ['{EXEC}']\n"
        assert comp.entries == (EXEC,) and [e.tag for e in hooks.live()[len(before):]] == [EXEC] * 2
        ctx = comp.context
        assert (ctx.stage, ctx.enabled, ctx.enforce_eager, ctx.settings) == ("capture", (EXEC,), False, {"exec": {"rounds": 7}})
        layer = model.layers[1]
        assert layer.quant_method.apply(layer, "x") == ("exec", 3, "x")
    finally:
        comp.close()
    assert hooks.live() == before


def test_a_capture_installs_nothing_for_a_target_with_no_such_protocol_or_a_set_it_refuses(monkeypatch, capsys):
    _both(monkeypatch, [])
    model, before = Model(), hooks.live()
    assert PO.at_capture(None, model) is None and PO.at_capture(TargetProfile.default(), model) is None
    with pytest.raises(PO.Refused) as e:
        PO.at_capture(_captured(), model, {"hold": {}})
    assert e.value.reasons == ("the protocols config names ['hold'], which this run does not enable",)
    assert capsys.readouterr().out == "" and hooks.live() == before
    assert all("apply" not in vars(m.quant_method) for m in model.layers)


def test_a_commit_installs_its_set_on_its_model_records_it_in_the_versions_and_says_so(monkeypatch, capsys):
    _both(monkeypatch, [])
    model, before = Model(), hooks.live()
    versions: dict[str, Any] = {"engine_kwargs": {}}
    comp = PO.at_commit(TargetProfile.default().with_overrides(protocols=(HOLD, SPOT)), model, versions)
    try:
        assert capsys.readouterr().out == f"[delta] protocol options installed beside the committers: ['{HOLD}', '{SPOT}']\n"
        assert (comp.context.stage, comp.entries) == ("commit", (HOLD, SPOT))
        assert [e.tag for e in hooks.live()[len(before):]] == [HOLD] * 3
        assert versions == {"engine_kwargs": {}, "protocols": {"entries": [HOLD, SPOT], "info": comp.info()}}
        assert versions["protocols"]["info"][HOLD] == {"entries": 3}
    finally:
        comp.close()
    assert hooks.live() == before


def test_a_commit_that_installs_nothing_records_nothing_and_says_nothing(monkeypatch, capsys):
    """A target with no protocols, a set the Commit refuses, and a set whose install raises."""
    _both(monkeypatch, [])
    model, before = Model(), hooks.live()
    versions: dict[str, Any] = {"engine_kwargs": {}}
    declared = TargetProfile.default().with_overrides(protocols=(HOLD, SPOT))
    assert PO.at_commit(None, model, versions) is None and PO.at_commit(TargetProfile.default(), model, versions) is None
    with pytest.raises(PO.Refused) as e:
        PO.at_commit(declared, model, versions, {"exec": {}})
    assert e.value.reasons == ("the protocols config names ['exec'], which this run does not enable",)

    def install(scheme, target):
        raise RuntimeError(f"no device for {scheme}")

    _fake_adapter(monkeypatch, "hold", install, schemes=(HOLD.partition(":")[2],))
    with pytest.raises(RuntimeError, match="no device for band-chain/d12/v1"):
        PO.at_commit(declared, model, versions)
    assert capsys.readouterr().out == "" and versions == {"engine_kwargs": {}} and hooks.live() == before
    assert all("apply" not in vars(m.quant_method) for m in model.layers)


def test_spot_binds_a_rows_build_and_an_adapter_that_does_not_say_binds_none(monkeypatch):
    _both(monkeypatch, [])
    ctx = PO.Context(stage="commit", enabled=(HOLD, SPOT))
    assert spot.requests(SPOT.partition(":")[2], ctx) == (BUILD_BOUND,)
    assert [e for e, r in PO.requests(ctx) if r.fixes] == [SPOT]
