from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("compile_cost_run", Path(__file__).parents[3] / "benchmarks/compile_cost/run.py")
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)
H200_ARGS = ["--card", "h200-sxm-141gb"]


def test_the_card_alone_compiles_long_w0() -> None:
    assert run.parse(H200_ARGS).window == "long-w0"


def test_requests_stay_in_place_of_a_window() -> None:
    a = run.parse([*H200_ARGS, "--requests", "0:300:3"])
    assert (a.window, a.requests) == (None, "0:300:3")
    with pytest.raises(SystemExit):
        run.parse([*H200_ARGS, "--window", "one-300", "--requests", "0:300:3"])


def test_the_card_alone_stays_glm_5_3() -> None:
    assert run.parse(H200_ARGS).model == "glm-5.3"


def test_llama_3_2_1b_compiles_its_class_circuit_in_place_of_a_window() -> None:
    a = run.parse([*H200_ARGS, "--model", "llama-3.2-1b"])
    assert (a.model, a.window, a.requests) == ("llama-3.2-1b", None, None)
    for extra in (["--window", "long-w0"], ["--requests", "0:300:3"], ["--layers", "2"], ["--guard"]):
        with pytest.raises(SystemExit):
            run.parse([*H200_ARGS, "--model", "llama-3.2-1b", *extra])


H200, H100, UNKNOWN = "NVIDIA H200", "NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB"

H200_TARGET = {"compute_capability": [9, 0], "num_sms": 132, "batch_invariant": False, "attention_backend": None,
               "vllm_revision": "d9105ea80", "numerics_profile": "b1-eager-v3", "flash_attn_version": None, "device_name": "NVIDIA H200",
               "moe_backend": "triton", "card": "h200-sxm-141gb"}


def test_the_card_has_no_default_and_the_h200_target_is_the_one_written_in_naming_its_card(capsys) -> None:
    with pytest.raises(SystemExit):
        run.parse([])
    assert "--card" in capsys.readouterr().err
    assert run.parse(H200_ARGS).card == run.PINS_CARD == "h200-sxm-141gb"
    assert json.dumps(run.target(run.PINS_CARD)) == json.dumps(H200_TARGET)


def test_another_card_is_read_from_its_card_file() -> None:
    assert run.target(run.parse(["--card", "h100-sxm-80gb"]).card) == {**H200_TARGET, "device_name": H100, "card": "h100-sxm-80gb"}
    assert {k: run.target("rtx-pro-6000-server")[k] for k in ("compute_capability", "num_sms", "device_name")} == {
        "compute_capability": [12, 0], "num_sms": 188, "device_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}


@pytest.mark.parametrize("card, why", [("a100-sxm4-80gb", "no card 'a100-sxm4-80gb'"), ("xeon-6776p-2s", "is no compile target")])
def test_a_card_with_no_gpu_file_is_refused_by_name(capsys, card, why) -> None:
    with pytest.raises(SystemExit):
        run.parse(["--card", card])
    assert why in capsys.readouterr().err


class Writer:
    def __init__(self, card: str) -> None:
        self.stamp, self.members, self.cards = {"card": card}, [], []

    def measurement(self, name, value, unit, *a, member=None, extra=None, **k) -> dict:
        self.members.append(member)
        self.cards.append((extra or {}).get("card"))
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
    run.record(tmp_path, "glm-5.3", "long-w0", "eager", {"wall_min": 1.0, "peak_gib": 2.0}, card=run.PINS_CARD)
    assert w.members == [member, member]


@pytest.mark.parametrize("model, card, member, short", [
    ("glm-5.3", "h200-sxm-141gb", "glm-5.3-long-w0-eager-cpu", "h200"),
    ("glm-5.3", "h100-sxm-80gb", "glm-5.3-long-w0-eager-cpu-for-h100", "h100"),
    ("llama-3.2-1b", "h200-sxm-141gb", "llama-3.2-1b-h200", "h200"),
    ("llama-3.2-1b", "h100-sxm-80gb", "llama-3.2-1b-h100", "h100")])
def test_a_point_names_the_card_it_was_compiled_for(monkeypatch, tmp_path, model, card, member, short) -> None:
    w = writer(monkeypatch, "none")
    run.record(tmp_path, model, "long-w0", "eager", {"wall_min": 1.0, "peak_gib": 2.0}, card=card)
    assert (w.members, w.cards) == ([member, member], [short, short])


def test_a_card_no_record_names_writes_no_point_and_is_named(monkeypatch, tmp_path) -> None:
    w = writer(monkeypatch, UNKNOWN)
    with pytest.raises(SystemExit, match=re.escape(repr(UNKNOWN))):
        run.record(tmp_path, "glm-5.3", "long-w0", "eager", {"wall_min": 1.0, "peak_gib": 2.0}, card=run.PINS_CARD)
    assert w.members == []
