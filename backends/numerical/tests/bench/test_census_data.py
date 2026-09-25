"""The census (``census/``, data only): every file validates against its JSON Schema, the ids renders cite keep their values,
and the directory holds nothing but data (nothing imports it; it can move to its own repository)."""

from __future__ import annotations

import json
import re

import pytest

from verity.ml.tc.instructions import INSTRUCTIONS
from verity.proofs.target import TARGETS
from verity_numerical.bench import census, contract

jsonschema = pytest.importorskip("jsonschema")

FILES = ("hardware", "networks", "workloads", "instance_sets")

#: The ids renders cite.  Removing or renaming one breaks every consumer's history: add a new entry instead.
PINNED = {
    "hardware": {"a100-sxm4-80gb/bf16": 312e12, "h100-sxm5-80gb/bf16": 989.4e12, "h100-sxm5-80gb/e4m3": 1978.9e12,
                 "rtx-4090/e4m3": 330.3e12, "rtx-5090/e2m1": 1676e12},
    "networks": {"dc-1ms-100gbps": (0.001, 100e9), "rtt-20ms-100gbps": (0.020, 100e9), "rtt-100ms-100gbps": (0.100, 100e9),
                 "dc-1ms-10gbps": (0.001, 10e9)},
    "workloads": {"vu-k1536": (1536, 4096)},
    "instance_sets": {"vu-k1536", "vu-k1536-fp8-ada", "vu-k1536-bf16-hopper", "vu-k1536-fp8-hopper", "vu-k1536-nvfp4-sm120"},
}


@pytest.mark.parametrize("name", FILES)
def test_every_census_file_validates_against_its_schema_with_unique_ids(name: str) -> None:
    schema = json.loads((census.ROOT / "schemas" / f"{name}.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    rows = census.raw(name)
    jsonschema.Draft202012Validator(schema).validate(rows)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_the_census_is_data_only() -> None:
    files = sorted(p.relative_to(census.ROOT).as_posix() for p in census.ROOT.rglob("*") if p.is_file())
    assert files == sorted(["README.md", *(f"{n}.json" for n in FILES), *(f"schemas/{n}.schema.json" for n in FILES)])


def test_ids_are_pinned_with_their_values() -> None:
    assert {k: h.flop_per_second for k, h in census.HARDWARE.items()} == PINNED["hardware"]
    assert {k: (n.rtt_s, n.bandwidth_bps) for k, n in census.NETWORKS.items()} == PINNED["networks"]
    assert {k: (w.K, w.batch) for k, w in census.WORKLOADS.items()} == PINNED["workloads"]
    assert set(census.INSTANCE_SETS) == PINNED["instance_sets"]


def test_hardware_lines_are_dense_fp32_accumulate_peaks_with_their_datasheet() -> None:
    for h in census.HARDWARE.values():
        assert "dense" in h.rate and "fp32 accumulate" in h.rate and h.id.endswith("/" + h.dtype) and census.line(h.device, h.dtype) is h
    assert "FP32 accumulate" in census.source(census.HARDWARE["rtx-4090/e4m3"])
    h100 = census.source(census.HARDWARE["h100-sxm5-80gb/bf16"])
    assert "1978.9" in h100 and "sparsity" in h100
    fp4 = census.HARDWARE["rtx-5090/e2m1"]
    assert "3352" in census.source(fp4) and "Appendix A" in census.source(fp4) and "fp4" in fp4.rate
    assert census.HARDWARE["a100-sxm4-80gb/bf16"].memory_bytes == 80 * 2**30 and census.HARDWARE["rtx-4090/e4m3"].memory_bytes == 24 * 2**30
    with pytest.raises(KeyError):
        census.line("NVIDIA A100 SXM4 80GB", "e4m3")


def test_the_reference_network_is_the_same_datacenter_and_the_sensitivity_profiles_vary_one_axis() -> None:
    ref = census.REFERENCE_NETWORK
    assert ref.id == "dc-1ms-100gbps" and (ref.rtt_s, ref.bandwidth_bps) == (0.001, 100e9)
    assert [n.id for n in census.SENSITIVITY_NETWORKS] == ["rtt-20ms-100gbps", "rtt-100ms-100gbps", "dc-1ms-10gbps"]
    for n in census.SENSITIVITY_NETWORKS:
        assert (n.rtt_s == ref.rtt_s) != (n.bandwidth_bps == ref.bandwidth_bps), n.id


def test_every_target_resolves_its_census_line_and_every_line_is_a_targets() -> None:
    """The overhead denominator: the census line of the target's anchor device and operand dtype (core names only the device)."""
    lines = {t.name: contract.native_peak_of(t) for t in TARGETS.values()}
    for name, h in lines.items():
        t = TARGETS[name]
        assert h.device == t.anchor_device and h.dtype == INSTRUCTIONS[t.instruction].a_dtype and h is contract.native_reference(name)
    assert {h.id for h in lines.values()} == set(census.HARDWARE)
    sets = [s.target for s in census.INSTANCE_SETS.values()]
    assert set(sets) <= set(TARGETS) and len(sets) == len(set(sets))
    assert all(s.size == census.VU_K1536.batch and s.id == s.tier and re.fullmatch(r"[0-9a-f]{64}", s.manifest_sha256)
               for s in census.INSTANCE_SETS.values())


def test_seconds_per_vu() -> None:
    h = census.HARDWARE["a100-sxm4-80gb/bf16"]
    assert census.seconds_per_vu(h, 1536) == 2 * 1536 / 312e12
    with pytest.raises(ValueError):
        census.seconds_per_vu(h, 0)
