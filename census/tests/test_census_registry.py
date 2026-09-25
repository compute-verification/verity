"""The census: stable ids with their values and sources, and the dependency direction (it imports nothing from Verity)."""

from __future__ import annotations

import dataclasses
import pathlib
import re

import pytest
import verity_census
from verity_census import (
    HARDWARE,
    INSTANCE_SETS,
    NETWORKS,
    REFERENCE_NETWORK,
    SENSITIVITY_NETWORKS,
    VU_K1536,
    WORKLOADS,
    Hardware,
    line,
)

#: The ids renders cite.  Removing or renaming one breaks every consumer's history: add a new entry instead.
PINNED = {
    "hardware": {"a100-sxm4-80gb/bf16": 312e12, "h100-sxm5-80gb/bf16": 989.4e12, "h100-sxm5-80gb/e4m3": 1978.9e12,
                 "rtx-4090/e4m3": 330.3e12, "rtx-5090/e2m1": 1676e12},
    "networks": {"dc-1ms-100gbps": (0.001, 100e9), "rtt-20ms-100gbps": (0.020, 100e9), "rtt-100ms-100gbps": (0.100, 100e9),
                 "dc-1ms-10gbps": (0.001, 10e9)},
    "workloads": {"vu-k1536": (1536, 4096)},
    "instance_sets": {"vu-k1536", "vu-k1536-fp8-ada", "vu-k1536-bf16-hopper", "vu-k1536-fp8-hopper", "vu-k1536-nvfp4-sm120"},
}


def test_the_census_imports_nothing_from_verity() -> None:
    """The census is data read by the renderer, benchmarks and tools; it depends on none of them (core, likewise, never imports
    the census: ``packages/verity/tests/test_boundaries.py``)."""
    src = pathlib.Path(verity_census.__file__).parent
    offenders = [f"{p.relative_to(src)}: {ln.strip()}" for p in src.rglob("*.py") for ln in p.read_text(encoding="utf-8").splitlines()
                 if re.match(r"^\s*(from|import)\s+(verity\b|verity\.|verity_(?!census)\w*|research|packages|veritor|backends|frontends|integrations|numpy)\b", ln)]
    assert not offenders, offenders


def test_ids_are_pinned_with_their_values() -> None:
    assert {k: h.flop_per_second for k, h in HARDWARE.items()} == PINNED["hardware"]
    assert {k: (n.rtt_s, n.bandwidth_bps) for k, n in NETWORKS.items()} == PINNED["networks"]
    assert {k: (w.K, w.batch) for k, w in WORKLOADS.items()} == PINNED["workloads"]
    assert set(INSTANCE_SETS) == PINNED["instance_sets"]


def test_every_entry_has_sources() -> None:
    for reg in (HARDWARE, NETWORKS, WORKLOADS, INSTANCE_SETS):
        for k, v in reg.items():
            assert k == v.id and v.sources and all(s.label for s in v.sources), k


def test_hardware_lines_are_dense_fp32_accumulate_peaks_with_their_datasheet() -> None:
    for h in HARDWARE.values():
        assert h.accumulate == "fp32" and "dense" in h.rate and "fp32 accumulate" in h.rate and h.methodology
        assert h.id.endswith("/" + h.dtype) and line(h.device, h.dtype) is h
    assert "FP32 accumulate" in HARDWARE["rtx-4090/e4m3"].source
    assert "1978.9" in HARDWARE["h100-sxm5-80gb/bf16"].source and "sparsity" in HARDWARE["h100-sxm5-80gb/bf16"].source
    fp4 = HARDWARE["rtx-5090/e2m1"]
    assert "3352" in fp4.source and "Appendix A" in fp4.source and "fp4" in fp4.rate
    assert HARDWARE["a100-sxm4-80gb/bf16"].memory_bytes == 80 * 2**30 and HARDWARE["rtx-4090/e4m3"].memory_bytes == 24 * 2**30
    with pytest.raises(KeyError):
        line("NVIDIA A100 SXM4 80GB", "e4m3")


def test_the_reference_network_is_the_same_datacenter_and_the_sensitivity_profiles_vary_one_axis() -> None:
    assert REFERENCE_NETWORK.id == "dc-1ms-100gbps" and (REFERENCE_NETWORK.rtt_s, REFERENCE_NETWORK.bandwidth_bps) == (0.001, 100e9)
    assert [n.id for n in SENSITIVITY_NETWORKS] == ["rtt-20ms-100gbps", "rtt-100ms-100gbps", "dc-1ms-10gbps"]
    for n in SENSITIVITY_NETWORKS:
        assert (n.rtt_s == REFERENCE_NETWORK.rtt_s) != (n.bandwidth_bps == REFERENCE_NETWORK.bandwidth_bps), n.id


def test_instance_sets_are_at_the_workload_batch_one_per_target() -> None:
    targets = [s.target for s in INSTANCE_SETS.values()]
    assert len(targets) == len(set(targets))
    assert all(s.size == VU_K1536.batch and s.id == s.tier and re.fullmatch(r"[0-9a-f]{64}", s.manifest_sha256) for s in INSTANCE_SETS.values())


def test_seconds_per_vu_and_frozen_entries() -> None:
    h = HARDWARE["a100-sxm4-80gb/bf16"]
    assert h.seconds_per_vu(1536) == 2 * 1536 / 312e12
    with pytest.raises(ValueError):
        h.seconds_per_vu(0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.flop_per_second = 1.0  # type: ignore[misc]
    assert isinstance(h, Hardware)
