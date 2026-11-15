"""Cases for the suite's Inductor guard (`conftest.py`), which `test_env.py` runs in a pytest of their own.  Never collected with the
suite: the guard is meant to fail the third of them, at its teardown, and the fourth to pass after it."""
from __future__ import annotations

import sys
from types import SimpleNamespace

from verity_vllm.engine import env


def test_a_switch_set_with_monkeypatch_is_not_left(monkeypatch):
    monkeypatch.setattr(sys.modules["torch._inductor.config"], "dynamic_scale_rblock", True)


def test_a_stand_in_module_set_with_monkeypatch_is_not_left(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch._inductor.config", SimpleNamespace(dynamic_scale_rblock=True))


def test_a_switch_left_changed_is_failed_by_name():
    sys.modules["torch._inductor.config"].dynamic_scale_rblock = True


def test_the_test_after_it_reads_the_pin():
    assert sys.modules["torch._inductor.config"].dynamic_scale_rblock is False
    env.require_inductor({})
