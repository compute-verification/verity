"""The bench's environment variables (settings.py): each is declared with its default, and run.py, steps.py and
int8_kscale_gpu.py, the modules that read any, read them only by a declared name."""
import ast
from pathlib import Path

import pytest

import run
import settings
import steps


def test_an_undeclared_variable_is_refused_by_name():
    with pytest.raises(KeyError, match="PPL_NOPE"):
        settings.get("PPL_NOPE")
    with pytest.raises(KeyError, match="PPL_NOPE"):
        settings.given("PPL_NOPE")


def test_run_and_steps_read_each_declared_variable_and_no_other():
    read = set()
    for node in (n for m in (run, steps) for n in ast.walk(ast.parse(Path(m.__file__).read_text()))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "settings"):
            read.add(node.args[0].value)
    assert read == set(settings.DECLARED)

