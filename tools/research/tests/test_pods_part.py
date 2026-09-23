"""The reference-part gate's comparison logic (research.pods.part), with no pod: verdicts, the nvidia-smi probe parse, the families table."""
from __future__ import annotations

from pathlib import Path

from research.pods import part

FAMILIES = {"rtx4090": {"vram_mb": 24564}, "a100": {"vram_mb": 81920}, "h100": {"vram_mb": 81559}, "rtx5090": {"vram_mb": 32607}}


def test_reference_board_is_reference():
    v = part.verdict("NVIDIA GeForce RTX 4090", 24564, FAMILIES)
    assert v["family"] == "rtx4090" and v["reference"] is True and v["verdict"] == "reference"


def test_re_memoried_4090_is_a_variant_with_its_percentage():
    v = part.verdict("NVIDIA GeForce RTX 4090", 49140, FAMILIES)   # the RunPod board lane ada-fp8-dumps landed on (48 GiB)
    assert v["reference"] is False and v["verdict"] == "variant (+100 %)"


def test_driver_rounding_stays_within_tolerance_but_not_beyond():
    assert part.verdict("NVIDIA A100-SXM4-80GB", 81920 - 2000, FAMILIES)["reference"] is True          # -2.4 %
    assert part.verdict("NVIDIA H100 80GB HBM3", 95830, FAMILIES)["verdict"] == "variant (+17 %)"      # an NVL 94 GB board named H100
    assert part.verdict("NVIDIA GeForce RTX 5090", 32607, FAMILIES)["reference"] is True


def test_unknown_family_or_missing_fact_is_unknown_not_reference():
    assert part.verdict("NVIDIA L40S", 46068, FAMILIES)["verdict"].startswith("unknown (no [families.l40s]")
    assert part.verdict(None, None, FAMILIES) == {"sku": None, "vram_mb": None, "family": None, "reference_vram_mb": None, "reference": None,
                                                  "verdict": "unknown"}
    assert part.verdict("NVIDIA GeForce RTX 4090", None, FAMILIES)["reference"] is None


def test_probe_parse_reads_nvidia_smi_cpu_load_and_hostname():
    text = ("NVIDIA GeForce RTX 4090, 24564\n@@\n128\n@@\nmodel name\t: AMD EPYC 7B13 64-Core Processor\n@@\n"
            "6.29 5.80 5.10 3/2000 12345\n@@\n842bb84436d4\n")
    f = part.parse_probe(text)
    assert f == {"sku": "NVIDIA GeForce RTX 4090", "count": 1, "vram_mb": 24564, "cpu": "AMD EPYC 7B13 64-Core Processor x128",
                 "hostname": "842bb84436d4", "load": 6.29}
    assert part.parse_probe("")["sku"] is None       # nvidia-smi absent: nothing guessed


def test_families_table_is_read_from_machines_toml(tmp_path: Path, monkeypatch):
    p = tmp_path / "machines.toml"
    p.write_text('[machines.m1]\nprovider = "runpod"\npod_id = "x"\nproject = "verity"\n\n[families.rtx4090]\nvram_mb = 24564\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(p))
    assert part.load_families() == {"rtx4090": {"vram_mb": 24564}}
    assert part.load_families(tmp_path / "absent.toml") == {}
    assert "reference 24564" in part.render({"machine": "m1", "pod_id": "x", "sku": "NVIDIA GeForce RTX 4090", "count": 1, "vram_mb": 24564,
                                             "reference_vram_mb": 24564, "cpu": "c x8", "load": 1.0, "hostname": "h", "verdict": "reference"})
