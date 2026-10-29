"""The census (``benchmarks/census/``, data only): every file validates against its JSON Schema, the ids renders cite keep their values,
and the directory holds nothing but data (nothing imports it; it can move to its own repository)."""

from __future__ import annotations

import json
import re

import pytest

from verity.ml.circuits.silicon.instructions import INSTRUCTIONS
from verity.ml.circuits.targets.target import FP8_HOPPER_WGMMA, TARGETS
from verity_numerical.bench import census, contract

jsonschema = pytest.importorskip("jsonschema")

FILES = ("hardware", "datatypes", "networks", "subcircuits", "input_sets", "workloads", "deployments", "adversary")

#: The ids renders cite.  Removing or renaming one breaks every consumer's history: add a new entry instead.
PINNED = {
    "hardware": {"a100-sxm4-80gb/bf16": 312e12, "h100-sxm5-80gb/bf16": 989.4e12, "h100-sxm5-80gb/e4m3": 1978.9e12,
                 "rtx-4090/e4m3": 330.3e12, "rtx-5090/e2m1": 1676e12, "rtx-pro-6000-server/bf16": 500e12,
                 "rtx-pro-6000-server/e4m3": 1000e12, "rtx-pro-6000-server/e2m1": 2000e12, "l40s-48gb/bf16": 362.05e12,
                 "l40s-48gb/e4m3": 733e12},
    "deployments": {"dc-12500-machines-600s": (12500, 600)},
    "networks": {"dc-1ms-100gbps": (0.001, 100e9), "rtt-20ms-100gbps": (0.020, 100e9), "rtt-100ms-100gbps": (0.100, 100e9),
                 "dc-1ms-10gbps": (0.001, 10e9)},
    "subcircuit_templates": {"gemm-coordinate": ("K", "datatype", "semantics"), "attention-head": ("D", "BN", "semantics"), "rope-head": ("D",),
                             "rmsnorm-fused-cuda": ("N", "EPS"), "rmsnorm-triton": ("N", "EPS"), "silu-mul": ("I",),
                             "gumbel-top-p-token-select": ("V",), "token-select-greedy": ("V",), "moe-router-topk": ("E", "TOPK", "VPT"),
                             "moe-expert-coordinate": ("E", "K"), "moe-expert-coordinate-weighted": ("E", "K"), "moe-sum": ("TOPK", "H"),
                             "scaled-mm-fp8-block-coordinate": ("K", "G", "semantics"), "bias-add": ("N",), "bf16-mul-scalar": ("N",),
                             "embedding-row": ("V", "H")},
    "datatypes": {"bf16": "BF16", "e4m3": "FP8 (E4M3)", "e2m1": "FP4 (NVFP4, E2M1)"},
    "names": {"a100-sxm4-80gb/bf16": "A100 SXM4", "h100-sxm5-80gb/bf16": "H100 SXM5", "h100-sxm5-80gb/e4m3": "H100 SXM5",
              "rtx-4090/e4m3": "RTX 4090", "rtx-5090/e2m1": "RTX 5090", "rtx-pro-6000-server/bf16": "RTX PRO 6000",
              "rtx-pro-6000-server/e4m3": "RTX PRO 6000", "rtx-pro-6000-server/e2m1": "RTX PRO 6000", "l40s-48gb/bf16": "L40S",
              "l40s-48gb/e4m3": "L40S"},
    "input_sets": {"bench-instances-v1", "bench-instances-fp8-ada-v1", "bench-instances-bf16-hopper-v1", "bench-instances-fp8-hopper-v1",
                      "bench-instances-nvfp4-sm120-v1"},
    "adversary": {"world-inference-2027": (9.36e30, 21224489795918368)},
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
    """The census holds workloads and subcircuits, never replay or proof blocks, which applications choose per run
    (``benchmarks/census/README.md``)."""
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
    assert {d["id"]: (d["machines"], d["window_s"]) for d in census.raw("deployments")} == PINNED["deployments"]
    assert {k: (a.total_flop, a.requests) for k, a in census.load("adversary").items()} == PINNED["adversary"]


def test_the_adversary_resource_model_is_the_world_and_its_totals_follow_from_its_parts() -> None:
    """The figure script's arithmetic: FLOP_REQ = 4500 * 2 * 49e9 and R = int(1e16*104/49) over a year."""
    models = census.load("adversary")
    assert [a.id for a in models.values() if a.role == "reference"] == ["world-inference-2027"]
    for a in models.values():
        assert a.flop_per_request == a.tokens_per_request * a.flop_per_active_param_per_token * a.active_params, a.id
        assert a.total_flop == a.requests * a.flop_per_request and a.datatype in census.DATATYPES, a.id
    world = models["world-inference-2027"]
    assert world.requests == int(1e16 * 104 / 49) and world.monitoring_period_s == 365 * 86400
    assert world.h100_hour_flop == 1979e12 * 3600 and world.model == "DeepSeek-V4-Pro"


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


def test_every_target_resolves_its_census_line_and_every_line_serves_a_target_or_a_workload() -> None:
    """The overhead denominator: the census line of the target's anchor device and operand dtype (core names only the device)."""
    lines = {t.name: contract.native_peak_of(t) for t in TARGETS.values()}
    for name, h in lines.items():
        t = TARGETS[name]
        assert h.device == t.anchor_device and h.dtype == INSTRUCTIONS[t.instruction].a_dtype and h is contract.native_reference(name)
    used = {h.id for h in lines.values()} | {w.serving.hardware for w in census.WORKLOADS.values()}
    # the L40S FP8 line serves no row yet; the RTX PRO 6000 lines serve the POUS bench's --hardware and the GemmCoordinate
    # hillclimbs (verity/ml/flock/pod/gemm_hill.py)
    rtx_pro = {f"rtx-pro-6000-server/{d}" for d in ("bf16", "e4m3", "e2m1")}
    assert used <= set(census.HARDWARE) and set(census.HARDWARE) - used <= {"l40s-48gb/e4m3"} | rtx_pro
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
    gemm = [s for s in census.SUBCIRCUITS.values() if s.template == "gemm-coordinate"]
    for s in gemm:
        ins = INSTRUCTIONS[s.parameters.semantics]
        sem = "-".join(x for x in ins.id.split(".") if not re.fullmatch(r"m\d+n\d+k\d+", x))
        assert s.id == f"{s.template}/k{s.parameters.K}/{sem}" and s.parameters.datatype == ins.a_dtype and ins.model is not None
        assert s.display == f"GemmCoordinate<{s.parameters.K}> · {sem.split('-')[0]} {sem.split('-')[1]} · {census.DATATYPES[ins.a_dtype].display}"
    untargeted = {s.id for s in gemm if getattr(s, "legacy_target", None) is None}
    real_k = {f"gemm-coordinate/k{k}/{sem}" for k in (2048, 8192) for sem in ("sm80-mma-bf16", "sm90-wgmma-bf16", "sm89-mma-e4m3", "sm90-wgmma-e4m3")}
    served = {f"gemm-coordinate/k{m['subcircuit_parameters']['K']}/{'-'.join(x for x in m['subcircuit_parameters']['semantics'].split('.') if not re.fullmatch(r'm[0-9]+n[0-9]+k[0-9]+', x))}"
              for w in census.raw("workloads") for m in w["template_mix"] if m["subcircuit_template"] == "gemm-coordinate"}
    assert untargeted == ({"gemm-coordinate/k1536/sm90-wgmma-bf16"} | real_k | served) - {s.id for s in gemm if getattr(s, "legacy_target", None)}
    assert served <= {s.id for s in gemm}                                                  # every served workload's GEMM is bound
    assert len(gemm) == len(TARGETS) + len(untargeted)
    wgmma = census.SUBCIRCUITS["gemm-coordinate/k1536/sm90-wgmma-bf16"]
    assert INSTRUCTIONS[wgmma.parameters.semantics].status == "pinned" and wgmma.fidelity_notes == []


def test_native_seconds_per_instance() -> None:
    h = census.HARDWARE["a100-sxm4-80gb/bf16"]
    assert census.native_seconds_per_instance(h, 1536) == 2 * 1536 / 312e12
    with pytest.raises(ValueError):
        census.native_seconds_per_instance(h, 0)


def test_the_h100_bf16_cells_and_input_set_stand_for_the_wgmma_bf16_subcircuit() -> None:
    wg, mma = "gemm-coordinate/k1536/sm90-wgmma-bf16", "gemm-coordinate/k1536/sm90-mma-bf16"
    assert census.stood_for_by(wg) == mma and census.stood_for_by(mma) is None
    e = census.SUBCIRCUITS[wg].stood_for_by
    assert "r20260926-023321-ecb5" in census.source(e) and "bench-instances-bf16-hopper-v1" in e.covers
    assert census.INPUT_SETS["bench-instances-bf16-hopper-v1"].subcircuit == mma
    assert all(census.stood_for_by(s) in (None, *census.SUBCIRCUITS) for s in census.SUBCIRCUITS)


def test_workloads_are_the_regression_rows_template_mix() -> None:
    """Each workload is a served vLLM configuration with its template mix: the proof blocks per family and bound
    parameters, read from its regression row's replay partition (the counts the coverage audit cites).  The file is kept equal
    to those expectations, and only families the census has a template for name one."""
    import collections
    import pathlib

    repo = pathlib.Path(census.ROOT).parents[1]
    assert census.WORKLOADS, "no workloads"
    for w in census.WORKLOADS.values():
        x = json.loads((repo / w.source.expected).read_text(encoding="utf-8"))
        rp = x["replay_partition"]
        mix = collections.Counter()
        for k, v in rp["strata"].items():
            mix[k.split("|")[0]] += v["population"]
        got = {e.vu_family + ("{" + ",".join(f"{k}={v}" for k, v in vars(e.parameters).items()) + "}" if vars(e.parameters) else ""): e.vus
               for e in w.template_mix}
        assert got == dict(mix) and w.vus == rp["population"]["vus"] == sum(e.vus for e in w.template_mix), w.id
        assert x["row"] == w.source.row and (x.get("program_digest") or {}).get("program_digest_of_record") == w.source.program_digest
        assert w.serving.datatype in census.DATATYPES and (w.serving.hardware is None or w.serving.hardware in census.HARDWARE)
        for e in w.template_mix:
            t = census.SUBCIRCUIT_TEMPLATES[e.subcircuit_template]              # every family maps to a template
            assert set(vars(e.subcircuit_parameters)) == set(vars(t.parameters)) and e.subcircuits_per_vu >= 1
            if e.vu_family.startswith(("Gemm_", "MoeExpertGemm", "ScaledMm")):
                assert e.subcircuits_per_vu == e.parameters.N and e.subcircuit_parameters.K == e.parameters.K
            if e.vu_family.startswith("Attention_"):
                assert e.subcircuits_per_vu == e.parameters.NH and e.subcircuit_parameters.D == e.parameters.D
        if w.serving.hardware_class == "l40s":
            assert w.serving.hardware == f"l40s-48gb/{w.serving.datatype}"
    r101 = next(w for w in census.WORKLOADS.values() if w.source.row == 101)
    gemm = sum(e.vus for e in r101.template_mix if e.vu_family == "Gemm_v1")
    assert (r101.vus, gemm, r101.model.name, r101.serving.hardware_class) == (46558, 18400, "Llama-3.2-1B", "l40s")


#: bench-spine's input-set generator keys on these template ids (cursor/bench-spine-5777, bench/templates.py); the census must
#: name them alike.
BENCH_SPINE_TEMPLATES = {"gemm-coordinate", "attention-head", "rope-head", "rmsnorm-fused-cuda", "rmsnorm-triton", "silu-mul",
                         "gumbel-top-p-token-select"}


def test_subcircuit_templates_cover_every_mixed_family_with_ir_definitions() -> None:
    templates = census.SUBCIRCUIT_TEMPLATES
    assert BENCH_SPINE_TEMPLATES <= set(templates)
    for t in templates.values():
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", t.id) and t.ir_definitions and t.description and vars(t.parameters)
        assert all(p.type in ("integer", "string", "number") for p in vars(t.parameters).values())
    mixed = {e.subcircuit_template for w in census.WORKLOADS.values() for e in w.template_mix}
    assert mixed <= set(templates) and len(mixed) == len(templates)                     # each template serves a family in a mix


def test_the_l40s_line_is_the_datasheets_dense_peak() -> None:
    bf16, fp8 = census.HARDWARE["l40s-48gb/bf16"], census.HARDWARE["l40s-48gb/e4m3"]
    assert (bf16.flop_per_second, fp8.flop_per_second) == (362.05e12, 733e12) and bf16.memory_bytes == 48 * 2**30
    assert "733*" in census.source(bf16) and "1,466*" in census.source(fp8) and "sparsity" in census.source(bf16)
    assert "accumulate" in bf16.methodology and bf16.name == "L40S"


#: Bound subcircuit ids as bench-spine's generator writes them (bench/templates.py: GEMM, FA2 head, RoPE, the row templates with
#: a float EPS slug) and as the other census templates bind: every one must validate.
BOUND_ID_EXAMPLES = (
    "gemm-coordinate/k1536/sm90-wgmma-e4m3", "gemm-coordinate/k2048/sm80-mma-bf16", "gemm-coordinate/k2560/sm90-wgmma-bf16",
    "attention-head/d64-bn128/sm80-fa2-bf16", "attention-head/d128-bn128/sm90-fa3-bf16", "rope-head/d64/neox-bf16",
    "rmsnorm-fused-cuda/n2048-eps1e-05/bf16", "rmsnorm-triton/n2048-eps9.999999747378752e-06/bf16", "silu-mul/i8192/bf16",
    "gumbel-top-p-token-select/v128256/fp32", "token-select-greedy/v256000/fp32", "moe-router-topk/e64-topk8-vpt8/fp32",
    "moe-expert-coordinate/e64-k2048/sm80-mma-bf16", "moe-sum/topk8-h2048/bf16", "scaled-mm-fp8-block-coordinate/k2560-g128/sm90-wgmma-e4m3",
)


def test_every_templates_bound_ids_validate() -> None:
    subs = json.loads((census.ROOT / "schemas" / "subcircuits.schema.json").read_text(encoding="utf-8"))
    sets = json.loads((census.ROOT / "schemas" / "input_sets.schema.json").read_text(encoding="utf-8"))
    pat = subs["items"]["properties"]["subcircuits"]["items"]["properties"]["id"]["pattern"]
    assert sets["items"]["properties"]["subcircuit"]["pattern"] == pat
    for i in BOUND_ID_EXAMPLES:
        assert re.fullmatch(pat, i), i
        assert i.split("/")[0] in census.SUBCIRCUIT_TEMPLATES, i
    for bad in ("gemm-coordinate/k1536", "GEMM/k1536/x", "gemm-coordinate//x", "attention-head/d64/sm80 fa2"):
        assert not re.fullmatch(pat, bad), bad
    for t in census.SUBCIRCUIT_TEMPLATES.values():                       # a bound id names its own template
        assert all(s.id.startswith(t.id + "/") for s in t.subcircuits)


def _ops(expr: str, env: dict) -> int:
    """An integer work-model expression (+ - * // and parentheses over names) evaluated without eval."""
    import ast
    import operator

    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv}

    def ev(n: ast.AST) -> int:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.Name):
            return int(env[n.id])
        raise ValueError(f"not a work-model expression: {ast.dump(n)}")

    return ev(ast.parse(expr, mode="eval"))


def test_every_template_carries_a_work_model() -> None:
    version = "work-model/v0"
    for t in census.SUBCIRCUIT_TEMPLATES.values():
        wm = t.work_model
        assert wm.version == version and wm.basis and wm.source
        assert set(vars(wm.variables)) <= {"T"} and ("T" in wm.ops_per_input) == ("T" in vars(wm.variables))
    gemm = census.SUBCIRCUIT_TEMPLATES["gemm-coordinate"].work_model
    assert _ops(gemm.ops_per_input, {"K": 1536}) == (1536 // 16) * 32 == 3072   # K/16 tensor-core steps of 32 FLOP
    for w in census.WORKLOADS.values():                                  # every bound parameter set in a mix evaluates
        for e in w.template_mix:
            wm = census.SUBCIRCUIT_TEMPLATES[e.subcircuit_template].work_model
            env = {**{k: v for k, v in vars(e.subcircuit_parameters).items() if isinstance(v, int)}, "T": 1}
            assert _ops(wm.ops_per_input, env) >= 0, (w.id, e.vu_family)
            assert all(_ops(x, env) >= 0 for x in vars(wm.special_per_input).values())
    rms = census.SUBCIRCUIT_TEMPLATES["rmsnorm-triton"].work_model
    assert _ops(rms.ops_per_input, {"N": 2048}) == 3 * 2048 and vars(rms.special_per_input) == {"rsqrt": "1"}
    fp8 = census.SUBCIRCUIT_TEMPLATES["scaled-mm-fp8-block-coordinate"].work_model
    assert _ops(fp8.ops_per_input, {"K": 2560, "G": 128}) == 2 * 2560 + 3 * 20



def test_memory_bound_templates_carry_a_bandwidth_work_model_and_the_h100_its_bandwidth() -> None:
    """The four non-GEMM templates C-Flock has cells on: bytes per input over their ports (bench-spine's templates), evaluated at
    their #101 bindings; the H100's HBM3 bandwidth is the denominator."""
    want = {"rope-head/d64/neox-bf16": 6 * 64, "silu-mul/i8192/bf16": 6 * 8192, "rmsnorm-fused-cuda/n2048-eps1e-05/bf16": 10 * 2048,
            "rmsnorm-triton/n2048-eps1e-05/bf16": 6 * 2048}
    for sid, nbytes in want.items():
        s = census.SUBCIRCUITS[sid]
        wm = census.SUBCIRCUIT_TEMPLATES[s.template].work_model
        env = {k: v for k, v in vars(s.parameters).items() if isinstance(v, int)}
        assert census.evaluate(wm.bytes_per_input, env) == nbytes and wm.bytes_basis and wm.bytes_source
        assert s.id.startswith(s.template + "/") and s.display
    for hid in ("h100-sxm5-80gb/bf16", "h100-sxm5-80gb/e4m3"):
        h = census.HARDWARE[hid]
        assert h.memory_bytes_per_second == 3.35e12 and "3.35 TB/s" in h.memory_sources[0].label and "3.35 TB/s" not in census.source(h)
    with pytest.raises(ValueError):
        census.evaluate("__import__('os')", {})
