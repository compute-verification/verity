from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from verity_bench import jobs
from verity_bench.vllm.presets import GPUS, SPEC_ROOT, resolve_spec

COPIES = {"llama32-1b-1gpu-pouw-protocol-served-h100": "llama32-1b-1gpu-pouw-protocol-served-h200",
          "llama32-1b-bf16-h100-served-arm": "llama32-1b-bf16-h200-served-arm"}
#: nvidia-smi's names for the H100s and the H200, and the card each is, by the node side's test that a card's
#: `nvidia_smi_name` is part of the leased card's name
NAMES = {"NVIDIA H100 80GB HBM3": "h100", "NVIDIA H200": "h200", "NVIDIA H100 NVL": None, "NVIDIA H100 PCIe": None}


@pytest.mark.parametrize("h100, h200", COPIES.items())
def test_each_h100_spec_is_its_h200_spec_with_only_the_card_changed(h100, h200):
    copy, original = (tomllib.loads((SPEC_ROOT / f"{name}.toml").read_text()) for name in (h100, h200))
    assert resolve_spec(h100).preset.gpu_id == "h100"
    copy["id"] = h200
    copy["preset"] |= {"id": h200, "gpu_id": "h200"}
    assert copy == original


@pytest.mark.parametrize("name, card", NAMES.items())
def test_an_h100_spec_takes_the_h100_sxm_card_alone_and_an_h200_spec_no_h100(name, card):
    assert [gpu for gpu in ("h100", "h200") if GPUS[gpu].nvidia_smi_name in name] == ([card] if card else [])


def test_the_h100_pouw_spec_runs_direct_on_the_h100_vm_through_its_slot_and_one_card_lease():
    from research.pods.nebius.slot_only import through_slot

    root, name, node = Path(__file__).resolve().parents[3], "llama32-1b-1gpu-pouw-protocol-served-h100", "vy-h100-1"
    assert node in jobs.DIRECT_NODES
    plan = jobs.make_plan(name, root=root, commit="a" * 40, lookup=lambda identity, commit: None)
    commands: list[list[str]] = []
    assert jobs.dispatch(plan, name, root=root, direct=node,
                         execute=lambda command, cwd: commands.append(list(command)) or 0) == 0
    assert len(commands) == len(plan.points) == 1
    flags, workload = commands[0][:commands[0].index("--")], commands[0][commands[0].index("--") + 1:]
    assert flags[:4] == ["research", "run", "--on", node] and through_slot(workload)
    lease = workload[workload.index("--", 4) + 1:]
    assert lease[:3] == ["gpu-lease", "1", "--wait"]
