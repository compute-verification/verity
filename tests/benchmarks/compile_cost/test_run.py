from __future__ import annotations

import importlib.util
import re
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


def test_the_bare_command_stays_glm_5_3() -> None:
    assert run.parse([]).model == "glm-5.3"


def test_llama_3_2_1b_compiles_its_class_circuit_in_place_of_a_window() -> None:
    a = run.parse(["--model", "llama-3.2-1b"])
    assert (a.model, a.window, a.requests) == ("llama-3.2-1b", None, None)
    for extra in (["--window", "long-w0"], ["--requests", "0:300:3"], ["--layers", "2"], ["--guard"]):
        with pytest.raises(SystemExit):
            run.parse(["--model", "llama-3.2-1b", *extra])


H200, H100, UNKNOWN = "NVIDIA H200", "NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB"


class Writer:
    def __init__(self, card: str) -> None:
        self.stamp, self.members = {"card": card}, []

    def measurement(self, name, value, unit, *a, member=None, extra=None, **k) -> dict:
        self.members.append(member)
        return {"name": name, "value": value, "unit": unit, "member": member, **(extra or {})}


def writer(monkeypatch: pytest.MonkeyPatch, card: str) -> Writer:
    from verity_bench import kit
    w = Writer(card)
    monkeypatch.setattr(kit, "writer", lambda out, *a, **k: w)
    return w


@pytest.mark.parametrize("card, member", [(H200, "glm-5.3-long-w0-eager-h200"), (H100, "glm-5.3-long-w0-eager-h100"),
                                          ("none", "glm-5.3-long-w0-eager-cpu")])
def test_a_member_names_the_card_records_short(monkeypatch, tmp_path, card, member) -> None:
    w = writer(monkeypatch, card)
    run.record(tmp_path, "glm-5.3", "long-w0", "eager", {"wall_min": 1.0, "peak_gib": 2.0})
    assert w.members == [member, member]


def test_a_card_no_record_names_writes_no_point_and_is_named(monkeypatch, tmp_path) -> None:
    w = writer(monkeypatch, UNKNOWN)
    with pytest.raises(SystemExit, match=re.escape(repr(UNKNOWN))):
        run.record(tmp_path, "glm-5.3", "long-w0", "eager", {"wall_min": 1.0, "peak_gib": 2.0})
    assert w.members == []
