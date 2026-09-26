"""The census (``census/``, data only): every file validates against its JSON Schema, the ids renders cite keep their values,
and the directory holds nothing but data (nothing imports it; it can move to its own repository)."""

from __future__ import annotations

import json
import re

import pytest

from verity.ml.tc.instructions import INSTRUCTIONS
from verity.proofs.target import FP8_HOPPER_WGMMA, TARGETS
from verity_numerical.bench import census, contract

jsonschema = pytest.importorskip("jsonschema")

FILES = ("hardware", "datatypes", "networks", "subcircuits", "input_sets")

#: The ids renders cite.  Removing or renaming one breaks every consumer's history: add a new entry instead.
PINNED = {
    "hardware": {"a100-sxm4-80gb/bf16": 312e12, "h100-sxm5-80gb/bf16": 989.4e12, "h100-sxm5-80gb/e4m3": 1978.9e12,
                 "rtx-4090/e4m3": 330.3e12, "rtx-5090/e2m1": 1676e12},
    "networks": {"dc-1ms-100gbps": (0.001, 100e9), "rtt-20ms-100gbps": (0.020, 100e9), "rtt-100ms-100gbps": (0.100, 100e9),
                 "dc-1ms-10gbps": (0.001, 10e9)},
    "subcircuit_templates": {"gemm-coordinate": ("K", "datatype", "semantics")},
    "datatypes": {"bf16": "BF16", "e4m3": "FP8 (E4M3)", "e2m1": "FP4 (NVFP4, E2M1)"},
    "names": {"a100-sxm4-80gb/bf16": "A100 SXM4", "h100-sxm5-80gb/bf16": "H100 SXM5", "h100-sxm5-80gb/e4m3": "H100 SXM5",
              "rtx-4090/e4m3": "RTX 4090", "rtx-5090/e2m1": "RTX 5090"},
    "input_sets": {"bench-instances-v1", "bench-instances-fp8-ada-v1", "bench-instances-bf16-hopper-v1", "bench-instances-fp8-hopper-v1",
                      "bench-instances-nvfp4-sm120-v1"},
}


@pytest.mark.parametrize("name", FILES)
def test_every_census_file_validates_against_its_schema_with_unique_ids(name: str) -> None:
    schema = json.loads((census.ROOT / "schemas" / f"{name}.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    rows = census.raw(name)
    jsonschema.Draft202012Validator(schema).validate(rows)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_the_census_encodes_no_units() -> None:
    """The census holds workloads and subcircuits, never replay or verification units (docs: units design)."""
    text = " ".join((census.ROOT / f"{n}.json").read_text(encoding="utf-8") for n in FILES)
    ids = [e["id"] for n in FILES for e in census.raw(n)]
    assert not [i for i in ids if re.search(r"(^|[-/])(vu|ru)([-/]|$)", i)] and "verification unit" not in text.lower()


def test_the_census_is_data_only() -> None:
    files = sorted(p.relative_to(census.ROOT).as_posix() for p in census.ROOT.rglob("*") if p.is_file())
    assert files == sorted(["README.md", *(f"{n}.json" for n in FILES), *(f"schemas/{n}.schema.json" for n in FILES)])


def test_ids_are_pinned_with_their_values() -> None:
    assert {k: h.flop_per_second for k, h in census.HARDWARE.items()} == PINNED["hardware"]
    assert {k: (n.rtt_s, n.bandwidth_bps) for k, n in census.NETWORKS.items()} == PINNED["networks"]
    assert {k: tuple(vars(t.parameters)) for k, t in census.SUBCIRCUIT_TEMPLATES.items()} == PINNED["subcircuit_templates"]
    assert census.GEMM_COORDINATE.reference_batch == 4096 and census.GEMM_COORDINATE.accumulate == "fp32"
    assert set(census.INPUT_SETS) == PINNED["input_sets"]
    assert {k: d.display for k, d in census.DATATYPES.items()} == PINNED["datatypes"]
    assert {k: h.name for k, h in census.HARDWARE.items()} == PINNED["names"]


def test_a_datatype_displays_its_common_name_then_its_format_and_every_hardware_dtype_is_one() -> None:
    for d in census.DATATYPES.values():
        assert d.display == (d.common if d.format == d.common else f"{d.common} ({d.format})")
    assert {h.dtype for h in census.HARDWARE.values()} <= set(census.DATATYPES)
    assert all(h.name in h.device.replace("GeForce ", "") for h in census.HARDWARE.values())


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
    sets = [s.target for s in census.INPUT_SETS.values()]
    assert set(sets) <= set(TARGETS) and len(sets) == len(set(sets))
    assert all(s.size == census.SUBCIRCUIT_TEMPLATES[census.SUBCIRCUITS[s.subcircuit].template].reference_batch
               and census.SUBCIRCUITS[s.subcircuit].legacy_target == s.target and re.fullmatch(r"[0-9a-f]{64}", s.manifest_sha256)
               for s in census.INPUT_SETS.values())


def test_every_target_is_a_subcircuit_bound_to_its_exact_instruction() -> None:
    """A subcircuit is a fully bound template: gemm-coordinate with K, datatype and semantics (the exact
    instruction) for each target, with a stable id template/params/semantics and a display name."""
    for t in TARGETS.values():
        s = census.subcircuit_of(t.name)
        assert s.template == "gemm-coordinate" and vars(s.parameters) == {"K": 1536, "datatype": INSTRUCTIONS[t.instruction].a_dtype, "semantics": t.instruction}
        assert s.parameters.datatype in census.DATATYPES
        sem = "-".join(x for x in t.instruction.split(".") if not re.fullmatch(r"m\d+n\d+k\d+", x))
        assert s.id == f"gemm-coordinate/k1536/{sem}" and set(vars(s.parameters)) == set(vars(census.GEMM_COORDINATE.parameters))
        assert s.display.startswith("GemmCoordinate<1536> · ") and s.display.endswith(census.DATATYPES[s.parameters.datatype].display)
    assert census.subcircuit_of(FP8_HOPPER_WGMMA.name).id == "gemm-coordinate/k1536/sm90-wgmma-e4m3"
    notes = {s.id: [n.note for n in s.fidelity_notes] for s in census.SUBCIRCUITS.values()}
    assert notes["gemm-coordinate/k1536/sm90-mma-bf16"] == [
        "H100 BF16 models mma (sm90.mma.m16n8k16.bf16); vLLM's H100 BF16 GEMMs run wgmma (sm90.wgmma.m64n8k16.bf16). "
        "On the H100 the two write the same words on every captured input, NaN and infinities included, so the gap is in the "
        "name only; gemm-coordinate/k1536/sm90-wgmma-bf16 names the instruction production runs"]
    assert {i.id: i.provenance for i in census.INPUT_SETS.values()} == {
        "bench-instances-v1": "captured", "bench-instances-fp8-ada-v1": "synthetic", "bench-instances-bf16-hopper-v1": "synthetic",
        "bench-instances-fp8-hopper-v1": "synthetic", "bench-instances-nvfp4-sm120-v1": "synthetic"}


def test_every_subcircuit_is_bound_to_a_registry_instruction() -> None:
    """Every subcircuit's semantics is an exact instruction of the registry, with its operand datatype and a stable id; a
    subcircuit need not have a (legacy) target: the H100 BF16 wgmma one (lane wgmma-bf16) came after the targets froze."""
    for s in census.SUBCIRCUITS.values():
        ins = INSTRUCTIONS[s.parameters.semantics]
        sem = "-".join(x for x in ins.id.split(".") if not re.fullmatch(r"m\d+n\d+k\d+", x))
        assert s.id == f"{s.template}/k{s.parameters.K}/{sem}" and s.parameters.datatype == ins.a_dtype and ins.model is not None
        assert s.display == f"GemmCoordinate<{s.parameters.K}> · {sem.split('-')[0]} {sem.split('-')[1]} · {census.DATATYPES[ins.a_dtype].display}"
    untargeted = {s.id for s in census.SUBCIRCUITS.values() if getattr(s, "legacy_target", None) is None}
    assert untargeted == {"gemm-coordinate/k1536/sm90-wgmma-bf16"}
    assert len(census.SUBCIRCUITS) == len(TARGETS) + len(untargeted)
    wgmma = census.SUBCIRCUITS["gemm-coordinate/k1536/sm90-wgmma-bf16"]
    assert INSTRUCTIONS[wgmma.parameters.semantics].status == "pinned" and wgmma.fidelity_notes == []


def test_native_seconds_per_instance() -> None:
    h = census.HARDWARE["a100-sxm4-80gb/bf16"]
    assert census.native_seconds_per_instance(h, 1536) == 2 * 1536 / 312e12
    with pytest.raises(ValueError):
        census.native_seconds_per_instance(h, 0)
