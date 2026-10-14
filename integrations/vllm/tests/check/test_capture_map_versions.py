"""Each Program the GLM-5.3 capture map marks `exists` cites the Definitions the Build binds at the map's target.

The map names, per site, the Program that checks its taps (`programs[*].definitions`).  The Build binds the same site through its
vocabulary (`vocabulary_for("profile", target.numerics_profile)`, each role at the map's target's `fp8_scale_construction` and
`sampler_construction`), so a cited Definition the Build does not bind there checks the taps with another function: `SiluMul_v1` for
the served `SiluMul_v2`, or the CUDA `_f16Vec` add `Bf16AddF2fp_v1` (NaN 0x7FFF) for an ATen add (`Bf16Add_v1`, NaN 0x7FC0).  For
every `exists` Program:

1. it cites what each of its roles binds at the map's target;
2. every word Definition it cites is one its roles bind under some construction;
3. every Boolean Definition it cites is the Boolean version (`program.boolean.boolean_version`) of one its roles bind at the target.

`SITES` maps each `exists` Program to its roles, with representative statics.
"""
from __future__ import annotations

import copy
import inspect
import itertools
from pathlib import Path

from verity.ml.circuits.defs import SpecializedDefinition
from verity_vllm.commit import capture_map as CM
from verity_vllm.program.boolean import boolean_version
from verity_vllm.program.frontend.rules.vocab import vocabulary_for
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.registry import catalog

MAP = Path(__file__).parents[2] / "manifests" / "capture-maps" / "glm-moe-dsa-eager-hopper-tp8-v0.json"
CMAP = CM.load(MAP)

#: `exists` Program -> {vocabulary role: representative statics}
SITES = {
    "embedding-shard": {"embedding_shard": dict(VS=8, H=8, START=0)},
    "fp8-block-linear-deepgemm": {"flashinfer_fp8_linear": dict(K=256, N=256), "deepgemm_fp8_linear": dict(K=256, N=256)},
    "rmsnorm-cuda": {"rmsnorm_cuda": dict(N=8, EPS=1e-6, BLOCK=1), "rmsnorm_fused_cuda": dict(N=8, EPS=1e-6, BLOCK=8)},
    "silu-mul": {"silu_mul": dict(I=8)},
    "bf16-add": {"residual_add": dict(N=8)},
    "moe-sum": {"moe_sum": dict(TOPK=2, H=8)},
    "token-select": {"token_select": dict(V=16)},
    "all-gather": {"all_gather": dict(WORLD=8, N=8)},
}


def _base(d) -> str:
    return d.base_id if isinstance(d, SpecializedDefinition) else d.id


def _selecting(tp: TargetProfile) -> dict[str, tuple[tuple, str]]:
    """The statics that select among Definitions: every registered value, and the target's."""
    return {"SCALE": (TargetProfile.FP8_SCALE_CONSTRUCTIONS, tp.effective_fp8_scale_construction()),
            "CONSTRUCTION": (TargetProfile.SAMPLER_CONSTRUCTIONS, tp.effective_sampler_construction())}


def build_bindings(cmap: dict, role: str, statics: dict) -> tuple[object, set[str]]:
    """(what `role` binds at the map's target, the base ids it binds under every construction)."""
    tp = TargetProfile.from_json(cmap["target"]["profile"])
    vocab = vocabulary_for("profile", tp.numerics_profile)
    selecting = {k: v for k, v in _selecting(tp).items() if k in inspect.signature(vocab.kinds[role]).parameters}
    at_target = vocab.kind(role, **statics, **{k: target for k, (_, target) in selecting.items()})
    every = {_base(vocab.kind(role, **statics, **dict(zip(selecting, combo))))
             for combo in itertools.product(*(values for values, _ in selecting.values()))}
    return at_target, every


def violations(cmap: dict) -> list[str]:
    defs = catalog.load()[0].defs
    out = []
    for pid, p in sorted(cmap["programs"].items()):
        if p["status"] != "exists":
            continue
        cited = set(p["definitions"])
        target, every = {}, set()
        for role, statics in SITES[pid].items():
            fn, any_construction = build_bindings(cmap, role, statics)
            target[_base(fn)] = fn
            every |= any_construction
        out += [f"{pid}: the Build binds {d} at the map's target; the map does not cite it" for d in sorted(target.keys() - cited)]
        for d in sorted(cited):
            if getattr(defs[d], "word", None) is None:
                if d not in every:
                    out.append(f"{pid}: cites {d}; its roles {sorted(SITES[pid])} bind {sorted(every)}")
                continue
            bits = sorted(_base(boolean_version(fn)) for fn in target.values())
            if d not in bits:
                out.append(f"{pid}: cites the Boolean {d}; the Boolean versions of what its roles bind are {bits}")
    return out


def test_every_exists_program_names_its_vocabulary_roles():
    assert sorted(SITES) == sorted(pid for pid, p in CMAP["programs"].items() if p["status"] == "exists")


def test_the_map_cites_the_definitions_the_build_binds_at_its_target():
    assert violations(CMAP) == []


def test_a_stale_or_foreign_version_is_reported():
    """The map as it was cited: `SiluMul_v1` beside the Build's `SiluMul_v2`, the fused-norm residual add for MoERunner's ATen add,
    the small-M FlashInfer linear alone where the target also binds DeepGEMM's, and token selection's Boolean version alone."""
    cmap = copy.deepcopy(CMAP)
    cmap["programs"]["silu-mul"]["definitions"] = ["SiluMul_v1", "SiluMul_v2"]
    cmap["programs"]["bf16-add"]["definitions"] = ["Bf16AddF2fp_v1"]
    cmap["programs"]["fp8-block-linear-deepgemm"]["definitions"] = ["FlashInferFp8Linear_v1"]
    cmap["programs"]["token-select"]["definitions"] = ["TokenSelect_v2"]
    assert violations(cmap) == [
        "bf16-add: the Build binds ResidualAdd_v1 at the map's target; the map does not cite it",
        "bf16-add: cites Bf16AddF2fp_v1; its roles ['residual_add'] bind ['ResidualAdd_v1']",
        "fp8-block-linear-deepgemm: the Build binds DeepGemmFp8Linear_v1 at the map's target; the map does not cite it",
        "silu-mul: cites SiluMul_v1; its roles ['silu_mul'] bind ['SiluMul_v2']",
        "token-select: the Build binds TokenSelect_v1 at the map's target; the map does not cite it",
    ]


def test_a_boolean_version_of_another_definition_is_reported():
    cmap = copy.deepcopy(CMAP)
    cmap["programs"]["moe-sum"]["definitions"] = ["MoeSum_v1", "TokenSelect_v2"]
    assert violations(cmap) == ["moe-sum: cites the Boolean TokenSelect_v2; the Boolean versions of what its roles bind are ['MoeSum_v2']"]
