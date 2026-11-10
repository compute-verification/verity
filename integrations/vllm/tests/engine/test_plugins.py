"""`engine.plugins`: the distribution's vLLM general plugins, and the environment that loads them."""
from __future__ import annotations

import ast
import importlib.util
import tomllib
from pathlib import Path

import pytest

from verity_vllm.engine import plugins

ROOT = Path(__file__).resolve().parents[2]


def test_the_distribution_declares_exactly_its_plugins_each_a_top_level_function():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["entry-points"][plugins.GROUP] == {n: p.target for n, p in plugins.PLUGINS.items()}
    for p in plugins.PLUGINS.values():
        module, _, attr = p.target.partition(":")
        top = ast.parse(Path(importlib.util.find_spec(module).origin).read_text()).body
        assert any(isinstance(n, ast.FunctionDef) and n.name == attr for n in top), p.target


def test_it_loads_the_always_on_plugins_and_refuses_a_name_it_lacks():
    assert plugins.env() == plugins.env("verity_moe_align") == {"VLLM_PLUGINS": "verity_moe_align"}
    with pytest.raises(ValueError, match="no vLLM plugin named"):
        plugins.env("verity_moe_aling")
    with pytest.raises(ValueError, match="always on"):
        plugins.env("verity_moe_align", verity_moe_align="1")


def test_a_named_plugin_comes_with_its_setting_and_an_unnamed_one_stays_off(monkeypatch):
    monkeypatch.setitem(plugins.PLUGINS, "verity_x", plugins.Plugin("m:register", "VERITY_X"))
    assert plugins.env("verity_x", verity_x={"a": 1, "b": [2]}) == {"VLLM_PLUGINS": "verity_moe_align,verity_x",
                                                                   "VERITY_X": '{"a":1,"b":[2]}'}
    assert plugins.env("verity_x", verity_x="raw")["VERITY_X"] == "raw"
    assert plugins.env() == {"VLLM_PLUGINS": "verity_moe_align"}
    with pytest.raises(ValueError, match="VERITY_X set"):
        plugins.env("verity_x")
    with pytest.raises(ValueError, match="aren't named"):
        plugins.env(verity_x="1")
