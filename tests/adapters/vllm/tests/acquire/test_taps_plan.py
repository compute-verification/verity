"""`acquire.sources.taps.attach` and `attached`, and the call-boundary source's front (`sites`, `claimed`, `unattached`), over
capture plan classes (`check.capture_plan.of_build`): the int8 source is attached where the plan names its classes, an
Int8RowQuant call boundary is its claim, and a class no source acquires refuses the Commit by name.  CPU only."""
from __future__ import annotations

import pytest
from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.acquire.sources import int8_source as I8
from verity_vllm.acquire.sources import taps as TAPS

from tests.acquire.test_int8_source import QUANT_OP, _class, _Committer, _Model, _plan

ADD = _class("model.norm/add", "0", "call_boundary", 16, "bf16", "Add16_v1")


def test_the_plans_int8_classes_attach_the_int8_source_and_add_its_capture_class():
    com, model = _Committer(), _Model()
    assert TAPS.attached(_plan()) == [I8.SHORT_NAME]
    assert TAPS.attach(com, _plan(), ("module_output",), model=model) == ("module_output", "call_boundary")
    assert [s.short_name for s in com.sources] == [I8.SHORT_NAME]
    com = _Committer()
    assert TAPS.attach(com, _plan(), ("module_output",), explicit_classes=True, model=model) == ("module_output",)


def test_a_plan_without_a_class_of_any_source_attaches_none():
    plan = [c for c in _plan(quant=False) if "Int8KvQuant_v1" not in c["definitions"]]
    com = _Committer()
    assert TAPS.attached(plan) == [] and TAPS.attach(com, plan, ("module_output",), model=_Model()) == ("module_output",)
    assert com.sources == []


def test_an_int8_call_boundary_is_the_int8_sources_and_any_other_is_the_call_boundary_sources():
    plan = [*_plan(), ADD]
    assert [CB.claimed(c) for c in plan[:2]] == [I8.SHORT_NAME] * 2 and CB.claimed(ADD) is None
    assert CB.sites(plan) == [ADD] and CB.sites(_plan()) == []
    assert CB.unattached(plan, (I8.SHORT_NAME,)) == []
    assert CB.unattached(plan, ()) == [f"{QUANT_OP}/{m}: claimed by int8, a source the Commit does not attach" for m in "01"]


def test_a_call_boundary_no_source_claims_refuses_the_commit_by_name(monkeypatch):
    com, said = _Committer(), []
    monkeypatch.setattr(TAPS.logger, "error", lambda fmt, *a: said.append(fmt % a))
    with pytest.raises(SystemExit) as e:
        TAPS.attach(com, [*_plan(), ADD], ("module_output",), model=_Model())
    assert e.value.code == 3 and com.sources == [] and "model.norm/add/0: a call boundary no attached source claims" in said[0]
