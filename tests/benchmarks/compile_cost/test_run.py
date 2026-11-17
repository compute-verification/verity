from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("compile_cost_run", Path(__file__).parents[3] / "benchmarks/compile_cost/run.py")
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


def test_the_bare_command_compiles_long_w0() -> None:
    assert run.parse([]).window == "long-w0"


def test_requests_stay_in_place_of_a_window() -> None:
    a = run.parse(["--requests", "0:300:3"])
    assert (a.window, a.requests) == (None, "0:300:3")
    with pytest.raises(SystemExit):
        run.parse(["--window", "one-300", "--requests", "0:300:3"])
