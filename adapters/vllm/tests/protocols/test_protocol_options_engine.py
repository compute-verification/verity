"""`verity_vllm.protocols` at a built engine: whose settings the engine fills and the refusal where it cannot, a
composition's hooks taken off before its close, an option closed before its composition, a Match capture's install, a
Commit's install, and which entry binds a row's Build.

CPU only.  The PoUW and PoUS adapters are `test_protocol_options`'s stand-ins over its toy linear layers; SPOT's
adapter is the real one.
"""
from __future__ import annotations

from typing import Any

import pytest

from tests.protocols.test_protocol_options import (
    ALL,
    Model,
    _fake_adapter,
    _pous,
    _pouw,
    _serve,
)
from verity_vllm import protocols as PO
from verity_vllm.engine import hooks
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.protocols import spot

POUW, POUS, SPOT = ALL
CAPTURED = TargetProfile(protocols=(POUW, SPOT), compilation="piecewise")


def _both(monkeypatch, log):
    """The two stand-in adapters, each under the scheme `ALL` names."""
    pouw = _fake_adapter(monkeypatch, "pouw", _pouw(log), schemes=(POUW.partition(":")[2],))
    return pouw, _fake_adapter(monkeypatch, "pous", _pous(log), schemes=(POUS.partition(":")[2],))


def test_the_engine_fills_only_an_adapter_that_reads_it_and_another_entry_keeps_its_settings(monkeypatch):
    asked = []

    def from_engine(scheme, settings, llm, epoch):
        asked.append((scheme, dict(settings), llm, epoch))
        return {"runner": "r", "rounds": 9}

    pouw, _ = _both(monkeypatch, [])
    pouw.from_engine = from_engine
    llm = object()
    own = {"rounds": 7, "steps": 2}
    ctx = _serve(POUW, POUS, settings={"pouw": own, "pous": {"key": "k"}}, filled={"pouw": ("runner",)})
    got = PO.from_engine(ctx, llm, epoch=3)
    assert asked == [("ncp-v1", own, llm, 3)]
    # what the adapter returns is taken, over the caller's where both name a key
    assert got.settings == {"pouw": {"rounds": 9, "steps": 2, "runner": "r"}, "pous": {"key": "k"}} and got.filled == {}
    assert (got.stage, got.enabled) == ("serve", (POUW, POUS)) and ctx.settings["pouw"] == {"rounds": 7, "steps": 2}


@pytest.mark.parametrize("error", [RuntimeError, AttributeError, ValueError])
def test_an_engine_that_cannot_supply_an_adapters_settings_is_refused_naming_every_such_entry(monkeypatch, error):
    def from_engine(scheme, settings, llm, epoch):
        raise error(f"no runner for {scheme}")

    pouw, pous = _both(monkeypatch, [])
    pouw.from_engine = pous.from_engine = from_engine
    with pytest.raises(PO.Refused) as e:
        PO.from_engine(_serve(POUW, POUS, filled={"pouw": ("runner", "coins")}), object())
    assert e.value.reasons == (
        f"{POUW}: its settings ['runner', 'coins'] come from the built engine, which doesn't supply them: no runner for ncp-v1",
        f"{POUS}: its settings [] come from the built engine, which doesn't supply them: no runner for band-chain/d12/v1")


def test_an_error_of_another_kind_from_an_adapters_read_of_the_engine_is_raised_as_it_is(monkeypatch):
    def from_engine(scheme, settings, llm, epoch):
        raise KeyError("runner")

    pouw, _ = _both(monkeypatch, [])
    pouw.from_engine = from_engine
    with pytest.raises(KeyError, match="^'runner'$"):
        PO.from_engine(_serve(POUW, POUS), object())


def test_unhook_takes_every_options_hooks_off_and_leaves_each_release_to_the_close(monkeypatch):
    log: list[str] = []
    _both(monkeypatch, log)
    model, before = Model(), hooks.live()
    comp = PO.install(model, _serve(POUW, POUS))
    layer = model.layers[0]
    assert layer.quant_method.apply(layer, "x") == ("pouw", 2, "x") and log == ["pouw", "responder up", "pous"]
    comp.unhook()
    assert log[3:] == ["responder down handle"]                              # its service stopped, nothing released
    assert hooks.live() == before and all("apply" not in vars(m.quant_method) for m in model.layers)
    assert layer.quant_method.apply(layer, "x") == ("gemm", 2, "x")
    assert not comp.closed and comp.entries == (POUW, POUS) and not any(opt.closed for opt in comp.installed)
    comp.unhook()                                                            # nothing is left to take off
    comp.close()
    assert log[3:] == ["responder down handle", "pous released"] and hooks.live() == before


def test_unhook_tries_every_options_hooks_and_raises_the_first_error(monkeypatch):
    def stop(handle):
        raise RuntimeError("the responder did not stop")

    def install(scheme, target):
        h = target.hooks(f"pous:{scheme}")
        h.service("responder", lambda: "handle", stop)
        return PO.Option(name=f"pous:{scheme}", hooks=h)

    _fake_adapter(monkeypatch, "pouw", _pouw([]), schemes=(POUW.partition(":")[2],))
    _fake_adapter(monkeypatch, "pous", install, schemes=(POUS.partition(":")[2],))
    model, before = Model(), hooks.live()
    comp = PO.install(model, _serve(POUW, POUS))
    with pytest.raises(RuntimeError, match="^the responder did not stop$"):
        comp.unhook()
    assert hooks.live() == before and all("apply" not in vars(m.quant_method) for m in model.layers)
    comp.close()                                                             # a stop that raised is not tried again


def test_an_option_closed_before_its_composition_is_released_once(monkeypatch):
    log: list[str] = []
    _both(monkeypatch, log)
    model, before = Model(), hooks.live()
    comp = PO.install(model, _serve(POUS))
    opt = comp.get("pous")
    opt.close()
    assert opt.closed and log == ["responder up", "pous", "responder down handle", "pous released"]
    assert hooks.live() == before
    opt.close()
    comp.close()
    assert log.count("pous released") == 1 and log.count("responder down handle") == 1


def test_a_capture_installs_its_set_without_spot_on_the_engines_model_and_says_so(monkeypatch, capsys):
    _both(monkeypatch, [])
    model, before = Model(), hooks.live()
    comp = PO.at_capture(CAPTURED, model, {"pouw": {"rounds": 7}, SPOT.partition(":")[0]: {}})
    try:
        assert capsys.readouterr().out == f"[m1] protocol options installed for the capture: ['{POUW}']\n"
        assert comp.entries == (POUW,) and [e.tag for e in hooks.live()[len(before):]] == [POUW] * 2
        ctx = comp.context
        assert (ctx.stage, ctx.enabled, ctx.enforce_eager, ctx.settings) == ("capture", (POUW,), False, {"pouw": {"rounds": 7}})
        layer = model.layers[1]
        assert layer.quant_method.apply(layer, "x") == ("pouw", 3, "x")
    finally:
        comp.close()
    assert hooks.live() == before


def test_a_capture_installs_nothing_for_a_target_with_no_such_protocol_or_a_set_it_refuses(monkeypatch, capsys):
    _both(monkeypatch, [])
    model, before = Model(), hooks.live()
    assert PO.at_capture(None, model) is None and PO.at_capture(TargetProfile(), model) is None
    with pytest.raises(PO.Refused) as e:
        PO.at_capture(CAPTURED, model, {"pous": {}})
    assert e.value.reasons == ("the protocols config names ['pous'], which this run does not enable",)
    assert capsys.readouterr().out == "" and hooks.live() == before
    assert all("apply" not in vars(m.quant_method) for m in model.layers)


def test_a_commit_installs_its_set_on_its_model_records_it_in_the_versions_and_says_so(monkeypatch, capsys):
    _both(monkeypatch, [])
    model, before = Model(), hooks.live()
    versions: dict[str, Any] = {"engine_kwargs": {}}
    comp = PO.at_commit(TargetProfile(protocols=(POUS, SPOT)), model, versions)
    try:
        assert capsys.readouterr().out == f"[delta] protocol options installed beside the committers: ['{POUS}', '{SPOT}']\n"
        assert (comp.context.stage, comp.entries) == ("commit", (POUS, SPOT))
        assert [e.tag for e in hooks.live()[len(before):]] == [POUS] * 3
        assert versions == {"engine_kwargs": {}, "protocols": {"entries": [POUS, SPOT], "info": comp.info()}}
        assert versions["protocols"]["info"][POUS] == {"entries": 3}
    finally:
        comp.close()
    assert hooks.live() == before


def test_a_commit_that_installs_nothing_records_nothing_and_says_nothing(monkeypatch, capsys):
    """A target with no protocols, a set the Commit refuses, and a set whose install raises."""
    _both(monkeypatch, [])
    model, before = Model(), hooks.live()
    versions: dict[str, Any] = {"engine_kwargs": {}}
    declared = TargetProfile(protocols=(POUS, SPOT))
    assert PO.at_commit(None, model, versions) is None and PO.at_commit(TargetProfile(), model, versions) is None
    with pytest.raises(PO.Refused) as e:
        PO.at_commit(declared, model, versions, {"pouw": {}})
    assert e.value.reasons == ("the protocols config names ['pouw'], which this run does not enable",)

    def install(scheme, target):
        raise RuntimeError(f"no device for {scheme}")

    _fake_adapter(monkeypatch, "pous", install, schemes=(POUS.partition(":")[2],))
    with pytest.raises(RuntimeError, match="no device for band-chain/d12/v1"):
        PO.at_commit(declared, model, versions)
    assert capsys.readouterr().out == "" and versions == {"engine_kwargs": {}} and hooks.live() == before
    assert all("apply" not in vars(m.quant_method) for m in model.layers)


def test_spot_binds_a_rows_build_and_an_adapter_that_does_not_say_binds_none(monkeypatch):
    _both(monkeypatch, [])
    ctx = PO.Context(stage="commit", enabled=(POUS, SPOT))
    assert spot.requires_build(SPOT.partition(":")[2], ctx) is True
    assert PO.requires_build(SPOT, ctx) is True and PO.requires_build(POUS, ctx) is False
