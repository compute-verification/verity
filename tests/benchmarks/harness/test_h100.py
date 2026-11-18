from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path

import pytest

from verity_bench import jobs, kit
from verity_bench.vllm.presets import SPEC_ROOT, resolve_spec

POUW = "llama32-1b-1gpu-pouw-protocol-served"
#: the specs that were an H200 file and an H100 file differing in their card alone
ON_BOTH = (POUW, "llama32-1b-bf16-served-arm")
#: nvidia-smi's names for the H100s and the H200, and the card each is, by the node side's test (`kit.is_card`)
NAMES = {"NVIDIA H100 80GB HBM3": "h100", "NVIDIA H200": "h200", "NVIDIA H100 NVL": None, "NVIDIA H100 PCIe": None}


@pytest.mark.parametrize("spec", ON_BOTH)
def test_a_spec_runs_on_the_h200_and_the_h100_with_only_the_card_differing(spec):
    assert not tomllib.loads((SPEC_ROOT / f"{spec}.toml").read_text()).get("card_settings")
    on = {card: resolve_spec(f"{spec}@{card}") for card in ("h200", "h100")}
    assert {card: spec.preset.gpu_id for card, spec in on.items()} == {"h200": "h200", "h100": "h100"}
    assert dataclasses.replace(on["h100"].preset, gpu_id="h200") == on["h200"].preset
    assert dataclasses.replace(on["h100"], preset=on["h200"].preset) == on["h200"]
    assert on["h200"].config_sha256 != on["h100"].config_sha256


@pytest.mark.parametrize("name, card", NAMES.items())
def test_an_h100_spec_takes_the_h100_sxm_card_alone_and_an_h200_spec_no_h100(name, card):
    assert [gpu for gpu in ("h100", "h200") if kit.is_card(name, gpu)] == ([card] if card else [])


def test_the_h100_pouw_spec_runs_its_one_point_in_a_one_card_lease(monkeypatch):
    root, name = Path(__file__).resolve().parents[3], f"{POUW}@h100"
    monkeypatch.setattr(jobs, "lease_host", lambda: True)
    for variable in ("GPU_LEASE_UUID", "GPU_LEASE_ON", "GPU_LEASE_CORES"):
        monkeypatch.delenv(variable, raising=False)
    plan = jobs.make_plan(name, root=root, commit="a" * 40, lookup=lambda identity, commit: None)
    commands: list[list[str]] = []
    assert jobs.dispatch(plan, name, root=root, execute=lambda command, cwd: commands.append(list(command)) or 0) == 0
    assert len(commands) == len(plan.points) == 1
    assert commands[0][:3] == ["gpu-lease", "1", "--wait"]
