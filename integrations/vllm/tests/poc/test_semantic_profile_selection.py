"""`verify.py --semantic-profile auto` picks the one shipped profile covering the checkpoint's architecture whose
requirements the bundle's manifest meets.

HP's Level-3 finding (hopper-profile.md §7): the verifier's default was the Llama profile, which refuses
`Qwen2ForCausalLM`, so no B1 bundle could reach a Table-1 PASS without the operator naming the Qwen2 profile that
RC had already shipped; and that profile pins FlashAttention-2 (`fa_version` 2), which the H100 (FA3) fails.
Selection is by architecture, then by `requires` against the manifest (compute capability is now a requirement of
every shipped profile: `["8.0", "8.9"]` for the measured Ampere/Ada FA2 profiles, `"9.0"` for the Hopper FA3
profile). Zero or several matches fail closed."""

import json
from pathlib import Path

import pytest

from verity_vllm_adapter import binding as vbinding

QWEN2 = {"architectures": ["Qwen2ForCausalLM"]}
LLAMA = {"architectures": ["LlamaForCausalLM"]}


def _manifest(cc: str, **over):
    m = {"compilation": {"enforce_eager": True, "cudagraph_mode": "NONE"}, "engine_flags": {"VLLM_BATCH_INVARIANT": "1"},
         "dtype": "bfloat16", "parallelism": {"tensor_parallel_size": 1}, "quantization": None, "gpu": {"compute_capability": cc}}
    m.update(over)
    return m


def test_architecture_alone_is_ambiguous_for_qwen2_and_unique_for_llama():
    assert vbinding.select_semantic_profile(LLAMA).name == "llama-eager-batch-invariant-v2.json"
    with pytest.raises(vbinding.BindingError, match="covered by 2 shipped semantic profiles"):
        vbinding.select_semantic_profile(QWEN2)  # FA2 (Ampere/Ada) and FA3 (Hopper) both cover Qwen2: the manifest decides


def test_manifest_compute_capability_selects_the_hardware_class():
    assert vbinding.select_semantic_profile(QWEN2, _manifest("8.0")).name == "qwen2-eager-batch-invariant-v2.json"
    assert vbinding.select_semantic_profile(QWEN2, _manifest("8.9")).name == "qwen2-eager-batch-invariant-v2.json"
    assert vbinding.select_semantic_profile(QWEN2, _manifest("9.0")).name == "qwen2-eager-batch-invariant-hopper-v1.json"
    assert vbinding.select_semantic_profile(LLAMA, _manifest("8.9")).name == "llama-eager-batch-invariant-v2.json"


def test_unmet_requirements_are_named_and_fail_closed():
    with pytest.raises(vbinding.BindingError, match="rejected: .*llama-eager-batch-invariant-v2.json.*gpu.compute_capability = '9.0'"):
        vbinding.select_semantic_profile(LLAMA, _manifest("9.0"))  # no Llama Hopper profile is shipped
    with pytest.raises(vbinding.BindingError, match="covered by 0 shipped semantic profiles"):
        vbinding.select_semantic_profile(QWEN2, _manifest("9.0", compilation={"enforce_eager": False, "cudagraph_mode": "FULL_AND_PIECEWISE"}))
    with pytest.raises(vbinding.BindingError):
        vbinding.select_semantic_profile({"architectures": ["MixtralForCausalLM"]}, _manifest("8.0"))
    with pytest.raises(vbinding.BindingError):
        vbinding.select_semantic_profile({})


def test_list_valued_requirement_is_one_of():
    prof = {"requires": {"gpu.compute_capability": ["8.0", "8.9"], "dtype": "bfloat16"}}
    assert vbinding._profile_requirements(prof, _manifest("8.9")) == []
    assert vbinding._profile_requirements(prof, _manifest("9.0")) == ["manifest.gpu.compute_capability = '9.0', semantic profile requires ['8.0', '8.9']"]


def test_hopper_profile_differs_from_the_ampere_profile_only_in_fa_version_and_requirements():
    sp = vbinding.SEMANTIC_PROFILES
    a = json.loads((sp / "qwen2-eager-batch-invariant-v2.json").read_text())
    h = json.loads((sp / "qwen2-eager-batch-invariant-hopper-v1.json").read_text())
    assert a["requires"]["gpu.compute_capability"] == ["8.0", "8.9"] and h["requires"]["gpu.compute_capability"] == "9.0"
    assert {k: v for k, v in a["requires"].items() if k != "gpu.compute_capability"} == {k: v for k, v in h["requires"].items() if k != "gpu.compute_capability"}
    for key in a:
        if key in ("id", "purpose", "requires", "layer_nodes", "supersedes", "derived_from"):  # runtime_impl is identical: FA3 is the same FlashAttentionImpl class, fa_version is the const
            continue
        assert a[key] == h[key], key
    fa = [(n.get("consts") or {}).get("fa_version") for n in a["layer_nodes"]]
    fh = [(n.get("consts") or {}).get("fa_version") for n in h["layer_nodes"]]
    assert fa.count(2) == 1 and fh.count(3) == 1 and fa.index(2) == fh.index(3)
    a_rest = [{k: v for k, v in n.items() if k != "consts"} for n in a["layer_nodes"]]
    h_rest = [{k: v for k, v in n.items() if k != "consts"} for n in h["layer_nodes"]]
    assert a_rest == h_rest


def test_every_shipped_profile_states_exactly_the_recorders_patch_list():
    """`description.patches` must equal the profile's list on every binding (binding.architecture_binding). W9M added the two
    MoE patches to `capture.PATCHES` (module-level patches, recorded for every model) after WB wrote the profiles from a
    3-entry list; from then until 2026-09-08 every fresh capture from the merged tree failed architecture binding (found by
    the HP re-verification). The list is read from the source because capture.py imports torch at module level."""
    import ast
    src = (vbinding.REPO_ROOT / "verity_vllm_adapter" / "capture.py").read_text()
    patches = next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "PATCHES")
    assert len(patches) >= 5
    for p in sorted(vbinding.SEMANTIC_PROFILES.glob("*.json")):
        doc = json.loads(p.read_text())
        if doc.get("status") == "frozen":  # GUIDANCE §24: historical recorder regime, kept byte-for-byte
            assert len(doc["patches"]) == 3, p.name
            continue
        assert doc["patches"] == patches, p.name


def test_two_profiles_covering_one_architecture_is_ambiguous(tmp_path):
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps({"schema": "verity-vllm/semantic-profile/v1", "id": name, "architectures": ["XForCausalLM"]}))
    (tmp_path / "not-a-profile.json").write_text(json.dumps({"schema": "other", "architectures": ["XForCausalLM"]}))
    with pytest.raises(vbinding.BindingError, match="covered by 2 shipped semantic profiles"):
        vbinding.select_semantic_profile({"architectures": ["XForCausalLM"]}, profiles_dir=tmp_path)


def test_frozen_v1_profiles_are_never_auto_selected_but_remain_loadable():
    """GUIDANCE §24: the v1 profiles are the three-patch recorder's contract, byte-for-byte (plus the frozen marker); `auto`
    resolves to their v2 twins, and an archive captured under v1 re-verifies only by naming v1 explicitly."""
    sp = Path(vbinding.__file__).resolve().parent.parent / "manifests" / "semantic-profiles"
    for arch in ("llama", "qwen2"):
        v1 = json.loads((sp / f"{arch}-eager-batch-invariant-v1.json").read_text())
        v2 = json.loads((sp / f"{arch}-eager-batch-invariant-v2.json").read_text())
        assert v1["status"] == "frozen" and "status" not in v2 and v2["supersedes"] == v1["id"]
        assert len(v1["patches"]) == 3 and len(v2["patches"]) == 5 and v2["patches"][:3] == v1["patches"]
        assert "gpu.compute_capability" not in v1["requires"] and v2["requires"]["gpu.compute_capability"] == ["8.0", "8.9"]
        assert {k: v for k, v in v2["requires"].items() if k != "gpu.compute_capability"} == v1["requires"]
        for k in ("symbols", "nodes", "consts", "runtime_impl", "relations", "weight_map"):
            if k in v1:
                assert v1[k] == v2[k], (arch, k)  # numerical semantics identical; only recorder compatibility + hardware class differ
    with pytest.raises(vbinding.BindingError, match="covered by 0 shipped"):
        vbinding.select_semantic_profile(QWEN2, _manifest("7.5"))  # neither v2 (8.0/8.9) nor hopper (9.0); v1 is not a fallback
