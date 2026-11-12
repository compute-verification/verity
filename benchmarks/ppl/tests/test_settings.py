"""The bench's environment variables (settings.py): each is declared with its default, and run.py, the one module that
reads any, reads them only by a declared name."""
import argparse
import ast
from pathlib import Path

import pytest

import run
import settings


def test_an_unset_variable_takes_its_default_and_a_set_one_its_value(monkeypatch):
    for name, (default, what) in settings.DECLARED.items():
        monkeypatch.delenv(name, raising=False)
        assert settings.get(name) == default and what
        monkeypatch.setenv(name, "set")
        assert settings.get(name) == "set"


def test_an_undeclared_variable_is_refused_by_name():
    with pytest.raises(KeyError, match="PPL_NOPE"):
        settings.get("PPL_NOPE")


def test_run_reads_each_declared_variable_and_no_other():
    read = set()
    for node in ast.walk(ast.parse(Path(run.__file__).read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "settings"):
            read.add(node.args[0].value)
    assert read == set(settings.DECLARED)


def test_an_eval_uses_the_gpus_its_lease_names(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    assert run.visible_gpus() == ["3", "5"]


def test_a_submitted_run_names_who_leases_its_gpus(monkeypatch):
    a = argparse.Namespace(model="llama-3.1-8b", set="c4", ctx=2048, gpus=0, seeds="0", plugin=None, spec=["adopted"],
                           max_min=120, campaign="cmp-13")
    monkeypatch.delenv("GPU_LEASE_WHO", raising=False)
    assert "GPU_LEASE_WHO=ppl-bench" in run.submit_command(a, run.LOCAL_DATA)
    monkeypatch.setenv("GPU_LEASE_WHO", "a-lane")
    assert "GPU_LEASE_WHO=a-lane" in run.submit_command(a, run.LOCAL_DATA)
