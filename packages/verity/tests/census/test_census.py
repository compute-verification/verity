"""The census registry: stable ids, the values they name, and that the targets cite it rather than restate it."""

from __future__ import annotations

import dataclasses

import pytest

from verity.census import HARDWARE, INSTANCE_SETS, NETWORKS, REFERENCE_NETWORK, SENSITIVITY_NETWORKS, VU_K1536, WORKLOADS, Hardware
from verity.ml.tc.instructions import INSTRUCTIONS
from verity.proofs.target import TARGETS

#: The ids renders cite.  Removing or renaming one breaks every consumer's history: add a new entry instead.
PINNED = {
    "hardware": {"a100-sxm4-80gb/bf16": 312e12, "h100-sxm5-80gb/bf16": 989.4e12, "h100-sxm5-80gb/e4m3": 1978.9e12,
                 "rtx-4090/e4m3": 330.3e12, "rtx-5090/e2m1": 1676e12},
    "networks": {"dc-1ms-100gbps": (0.001, 100e9), "rtt-20ms-100gbps": (0.020, 100e9), "rtt-100ms-100gbps": (0.100, 100e9),
                 "dc-1ms-10gbps": (0.001, 10e9)},
    "workloads": {"vu-k1536": (1536, 4096)},
    "instance_sets": {"vu-k1536", "vu-k1536-fp8-ada", "vu-k1536-bf16-hopper", "vu-k1536-fp8-hopper", "vu-k1536-nvfp4-sm120"},
}


def test_ids_are_pinned_with_their_values() -> None:
    assert {k: h.flop_per_second for k, h in HARDWARE.items()} == PINNED["hardware"]
    assert {k: (n.rtt_s, n.bandwidth_bps) for k, n in NETWORKS.items()} == PINNED["networks"]
    assert {k: (w.K, w.batch) for k, w in WORKLOADS.items()} == PINNED["workloads"]
    assert set(INSTANCE_SETS) == PINNED["instance_sets"]


def test_every_registry_is_keyed_by_its_entries_id() -> None:
    for reg in (HARDWARE, NETWORKS, WORKLOADS, INSTANCE_SETS):
        assert all(k == v.id for k, v in reg.items())


def test_the_reference_network_is_the_same_datacenter_and_the_sensitivity_profiles_vary_one_axis() -> None:
    assert (REFERENCE_NETWORK.rtt_s, REFERENCE_NETWORK.bandwidth_bps) == (0.001, 100e9)
    for n in SENSITIVITY_NETWORKS:
        assert (n.rtt_s == REFERENCE_NETWORK.rtt_s) != (n.bandwidth_bps == REFERENCE_NETWORK.bandwidth_bps), n.id


def test_every_target_peak_is_a_census_line_and_every_line_is_used() -> None:
    peaks = [t.native_peak for t in TARGETS.values()]
    assert all(any(p is h for h in HARDWARE.values()) for p in peaks)
    assert {p.id for p in peaks} == set(HARDWARE)


def test_a_hardware_line_names_its_targets_operand_dtype() -> None:
    for t in TARGETS.values():
        assert t.native_peak.dtype == INSTRUCTIONS[t.instruction].a_dtype, t.name
        assert t.native_peak.id.endswith("/" + t.native_peak.dtype)
        assert t.native_peak.accumulate == "fp32"


def test_instance_sets_name_targets_one_each_at_the_workload_batch() -> None:
    targets = [s.target for s in INSTANCE_SETS.values()]
    assert set(targets) <= set(TARGETS) and len(targets) == len(set(targets))
    assert all(s.size == VU_K1536.batch and s.id == s.tier for s in INSTANCE_SETS.values())


def test_seconds_per_vu() -> None:
    h = HARDWARE["a100-sxm4-80gb/bf16"]
    assert h.seconds_per_vu(1536) == 2 * 1536 / 312e12
    with pytest.raises(ValueError):
        h.seconds_per_vu(0)


def test_entries_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        HARDWARE["rtx-4090/e4m3"].flop_per_second = 1.0  # type: ignore[misc]
    assert isinstance(next(iter(HARDWARE.values())), Hardware)
