"""cb-fix (R10 §5): the applicability contract of an AOT-constructed Program, pure Python (no torch, no vLLM).

The Program digest is a COMPONENT of the artifact identity, never a function of the target: two targets that produce the same
circuit share the digest and differ in identity; two hosts building the same profile share both; reuse under another profile or
under other applicability conditions is refused with every differing field named."""

import json

import pytest

from verity_vllm.program.frontend.target_profile import (ApplicabilityError, ArtifactIdentity, TargetProfile, canonical_bytes, check_reuse, load_artifact,
                                               parse_target_arg, profile_mismatches, reuse_mismatches)

B0_DIGEST = "59b35d80faf29438d21a0ea1c96c877ec69e2f21174f11b835ac1ef04c1ed496"  # B0 request 8/4 (cp-cba, cp-cbc, cp-cb8: identical)
CV = {"schema": "cb-a/aot-artifact/v1", "rules": [{"id": "TritonGemm[vllm-bi]", "version": 1}], "sources_sha256": "ab" * 32}
APPLIC = {"numerics_profile": "b1-eager-v3", "attention_impl": ["vllm.v1.attention.backends.flash_attn.FlashAttentionImpl"], "vllm_flash_attn_version": ["2"],
          "fused_add_rms_norm_dispatch": "batch_invariant", "vllm_version": "0.28.1rc1.dev472+gd9105ea80", "torch_version": "2.13.0+cu129",
          "triton_version": "3.7.1", "tuned_gemm_arch_family": "ada"}


def _applied(p: TargetProfile, **over) -> dict:
    d = {"compute_capability": list(p.compute_capability), "num_sms": p.num_sms, "batch_invariant": p.batch_invariant,
         "vllm_version": "0.28.1rc1.dev472+gd9105ea80", "attention_backend": p.attention_backend,
         "tuned_gemm_arch_family": p.tuned_gemm_arch_family()}   # [R13 ampere] what vLLM's own family function answers for the capability
    d.update(over)
    return d


def test_default_profile_is_the_admitted_sm89_record_and_canonical():
    p = TargetProfile.default()
    assert p.compute_capability == (8, 9) and p.num_sms == 128 and p.batch_invariant and p.attention_backend is None
    assert p.capability_str == "8.9" and p.sm_arch == "sm_89"
    assert p.canonical() == canonical_bytes(p.to_json()) and TargetProfile.from_json(json.loads(p.canonical())) == p
    assert TargetProfile(compute_capability=[8, 9]) == p  # list vs tuple canonicalise to the same profile


def test_profile_rejects_unknown_and_invalid_fields():
    with pytest.raises(ValueError):
        TargetProfile.from_json({"compute_capability": [8, 9], "gpu_name": "L40S"})
    with pytest.raises(ValueError):
        TargetProfile(num_sms=0)
    with pytest.raises(ValueError):
        parse_target_arg('{"VLLM_ATTENTION_BACKEND": "TRITON_ATTN"}')  # a dead env var is not a profile field


def test_declared_vs_applied_names_every_field_and_treats_absent_as_mismatch():
    p = TargetProfile.default()
    assert profile_mismatches(p, _applied(p)) == []
    bad = profile_mismatches(p, _applied(p, compute_capability=[9, 0]))
    assert bad == ["compute_capability declared [8, 9], applied [9, 0]"]
    bad = profile_mismatches(p, {"vllm_version": "0.28.1rc1.dev472+gd9105ea80"})
    assert any(b.startswith("compute_capability declared [8, 9], applied None") for b in bad)
    assert any(b.startswith("num_sms declared 128, applied None") for b in bad)
    assert any(b.startswith("batch_invariant declared True, applied None") for b in bad)
    assert any("attention_backend declared None, applied '<absent>'" in b for b in bad)
    assert any("tuned_gemm_arch_family for cc 8.9 is 'ada'" in b and "'<absent>'" in b for b in bad)
    bad = profile_mismatches(p, _applied(p, vllm_version="0.29.0+gdeadbeef"))
    assert bad == ["vllm_revision declared 'd9105ea80', running vllm '0.29.0+gdeadbeef'"]


def test_case_i_inconsistent_capability_same_digest_different_identity():
    """The reported case: another capability whose circuit is genuinely unchanged (sm_80 / 108 SMs: same tiling table row = default,
    FA2 on both) -> the SAME Program digest but a DIFFERENT artifact identity; the profile is what says which target it is."""
    p89, p80 = TargetProfile.default(), TargetProfile(compute_capability=(8, 0), num_sms=108)
    a89 = ArtifactIdentity.build(CV, p89, APPLIC, B0_DIGEST)
    a80 = ArtifactIdentity.build(CV, p80, APPLIC, B0_DIGEST)
    assert a89.program_digest == a80.program_digest == B0_DIGEST
    assert a89.identity != a80.identity
    assert a89.target_profile["compute_capability"] == [8, 9] and a80.target_profile["compute_capability"] == [8, 0]
    # the identity is a function of exactly its four components (deterministic, length-prefixed)
    assert a89.identity == ArtifactIdentity.compute(CV, p89, APPLIC, B0_DIGEST) == a89.recomputed()
    assert ArtifactIdentity.build(CV, p89, dict(APPLIC, vllm_flash_attn_version=["3"]), B0_DIGEST).identity != a89.identity
    assert ArtifactIdentity.build(dict(CV, sources_sha256="cd" * 32), p89, APPLIC, B0_DIGEST).identity != a89.identity
    assert ArtifactIdentity.build(CV, p89, APPLIC, "0" * 64).identity != a89.identity


def test_case_iii_two_hosts_same_profile_same_identity():
    """Host facts (hostname, CUDA visibility, host GPU, pod) are recorded in the manifest's `target.host` and enter NO component."""
    p = TargetProfile.default()
    hosts = [{"hostname": "cp-cba", "cuda_available": True, "host_gpu": "NVIDIA GeForce RTX 4090", "host_sm_count": 128},
             {"hostname": "cp-cbc", "cuda_available": False, "host_gpu": None},
             {"hostname": "6ded48a1b348", "cuda_available": False, "CUDA_VISIBLE_DEVICES": "", "host_gpu": None, "pod": "cp-cb8 (L40S, 142 SMs hidden)"}]
    ids = {ArtifactIdentity.build(CV, p, APPLIC, B0_DIGEST).identity for _ in hosts}
    assert len(ids) == 1
    # the wrapper round-trips through JSON unchanged (what two hosts would write)
    a = ArtifactIdentity.build(CV, p, APPLIC, B0_DIGEST)
    b = ArtifactIdentity.from_json(json.loads(json.dumps(a.to_json())))
    assert b.identity == a.identity and b.recomputed() == a.identity


def test_case_iv_reuse_under_mismatched_profile_is_refused_verbatim(tmp_path):
    p89 = TargetProfile.default()
    art = ArtifactIdentity.build(CV, p89, APPLIC, B0_DIGEST)
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(art.to_json()))
    # the profile it was built for, under the same conditions: admitted
    assert load_artifact(str(path), p89, APPLIC, CV).identity == art.identity
    # another target (H100: 9.0 / 132 SMs): refused, both differing fields named with both values
    with pytest.raises(ApplicabilityError) as ei:
        load_artifact(str(path), TargetProfile(compute_capability=(9, 0), num_sms=132), APPLIC)
    assert ei.value.reasons == ["target_profile.compute_capability: requested [9, 0], artifact built for [8, 9]",
                                "target_profile.num_sms: requested 132, artifact built for 128"]
    # same profile, but the requester's environment differs in an applicability condition (FA3 host): refused
    bad = reuse_mismatches(art, p89, dict(APPLIC, vllm_flash_attn_version=["3"]))
    assert bad == ["applicability.vllm_flash_attn_version: artifact requires ['2'], requester has ['3']"]
    # a requester that reports no conditions cannot reuse anything
    assert reuse_mismatches(art, p89, None) == ["applicability conditions of the requester were not provided (every condition is unverifiable)"]
    # a condition the requester did not report is a mismatch, never assumed
    partial = {k: v for k, v in APPLIC.items() if k != "fused_add_rms_norm_dispatch"}
    assert reuse_mismatches(art, p89, partial) == ["applicability.fused_add_rms_norm_dispatch: artifact requires 'batch_invariant', requester did not report it"]
    # another construction version (edited rule/shim): refused
    assert reuse_mismatches(art, p89, APPLIC, dict(CV, sources_sha256="00" * 32)) == ["construction_version differs from the artifact's"]


def test_tampered_wrapper_is_refused():
    p = TargetProfile.default()
    art = ArtifactIdentity.build(CV, p, APPLIC, B0_DIGEST)
    d = art.to_json()
    d["target_profile"]["compute_capability"] = [9, 0]  # relabel the target without rebuilding
    forged = ArtifactIdentity.from_json(d)
    bad = reuse_mismatches(forged, TargetProfile(compute_capability=(9, 0)), APPLIC)
    assert bad and bad[0].startswith(f"identity {art.identity} does not match its components")
    with pytest.raises(ApplicabilityError):
        check_reuse(forged, TargetProfile(compute_capability=(9, 0)), APPLIC)


def test_parse_target_arg_overrides_the_default():
    assert parse_target_arg(None) == TargetProfile.default()
    p = parse_target_arg('{"compute_capability": [9, 0]}')
    assert p.compute_capability == (9, 0) and p.num_sms == 128
    assert parse_target_arg('{"attention_backend": "TRITON_ATTN"}').attention_backend == "TRITON_ATTN"
    assert TargetProfile.default().env() == {"VLLM_BATCH_INVARIANT": "1"} and TargetProfile(batch_invariant=False).env() == {"VLLM_BATCH_INVARIANT": "0"}


def test_compiled_profile_declares_the_execution_and_keeps_eager_digests():
    """[R15 compiled] `compilation="piecewise"` -> execution label `bi-compiled`, run-time engine args enforce_eager False + PIECEWISE CUDA
    graphs, derivation-side args unchanged (eager trace by construction), a declared engine block for GP-01/GM-01; eager profiles serialise
    and digest exactly as before the field existed; an unadmitted compilation is refused."""
    e = TargetProfile.default()
    assert "compilation" not in e.to_json() and e.modelled_engine() == {"enforce_eager": True} and e.engine_args() == {}
    c = TargetProfile.from_json({**e.to_json(), "compilation": "piecewise"})
    assert c.execution_label == "bi-compiled" and TargetProfile(batch_invariant=False, compilation="piecewise").execution_label == "compiled"
    assert c.with_overrides(compute_capability=(9, 0), flash_attn_version=2).execution_label == "bi-compiled-fa2"
    # [ruling M-0380] `piecewise` = vLLM's production DEFAULT compiled configuration (no precision-cast emulation): its Program is P_compiled;
    # `piecewise-emu` = the separate declared specialization with Inductor emulate_precision_casts (a different numerical computation)
    assert c.engine_args() == {"enforce_eager": False, "compilation_config": {"cudagraph_mode": "PIECEWISE",
                                                                              "inductor_compile_config": {"emulate_precision_casts": False}}}
    assert c.engine_args(execution=False) == {}
    me = c.modelled_engine()
    assert me["enforce_eager"] is False and "P_compiled" in me["claim"] and "NOT a model" in me["claim"]
    assert me["compilation"]["emulate_precision_casts"] is False
    emu = TargetProfile.from_json({**e.to_json(), "compilation": "piecewise-emu"})
    assert emu.engine_args()["compilation_config"]["inductor_compile_config"] == {"emulate_precision_casts": True}
    assert emu.modelled_engine()["compilation"]["emulate_precision_casts"] is True and emu.execution_label == "bi-compiled"
    assert emu.digest() != c.digest()
    # GP-01's COMPILATION_FACT_KEYS in the run header's read-back form (vllm_adapter.compilation_facts); capture sizes / cache switch are
    # not declared (absent, not None: GM-01 compares every key both sides carry)
    assert {k: me["compilation"][k] for k in ("mode", "cudagraph_mode", "use_inductor_graph_partition")} == \
        {"mode": "VLLM_COMPILE", "cudagraph_mode": "PIECEWISE", "use_inductor_graph_partition": False}
    assert "capture_sizes" not in me["compilation"] and "compile_cache" not in me["compilation"]
    assert c.to_json()["compilation"] == "piecewise" and c.digest() != e.digest()
    assert TargetProfile.from_json(c.to_json()) == c and TargetProfile.from_json(e.to_json()).digest() == e.digest()
    with pytest.raises(ValueError):
        TargetProfile(compilation="full")
    assert parse_target_arg('{"compilation": "piecewise"}').execution_label == "bi-compiled"


def test_fp8_block_gemm_declares_the_cutlass_kernel_and_keeps_undeclared_digests():
    """[R15 fp8] `fp8_block_gemm='cutlass'` applies VLLM_USE_DEEP_GEMM=0 and requires the read-back kernel class; an undeclared profile
    serialises exactly as before (no key), so every pre-R15 profile digest and artifact identity is unchanged."""
    e = TargetProfile(compute_capability=(9, 0), num_sms=132)
    assert "fp8_block_gemm" not in e.to_json() and TargetProfile.from_json(e.to_json()).digest() == e.digest()
    assert e.env() == {"VLLM_BATCH_INVARIANT": "1"} and e.fp8_block_gemm_kernel() is None
    f = e.with_overrides(fp8_block_gemm="cutlass")
    assert f.to_json()["fp8_block_gemm"] == "cutlass" and f.digest() != e.digest()
    assert TargetProfile.from_json(f.to_json()) == f
    assert f.env() == {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_DEEP_GEMM": "0"}
    assert f.fp8_block_gemm_kernel() == "CutlassFp8BlockScaledMMKernel"
    assert f.execution_label == "bi-eager"          # the kernel choice is not another execution: the row id is unchanged
    with pytest.raises(ValueError):
        TargetProfile(fp8_block_gemm="deepgemm")
    # declared vs applied: the selected kernel must be CUTLASS on every FP8 linear; absent / another class / no FP8 linears = mismatch
    applied = _applied(f, compute_capability=[9, 0], num_sms=132, tuned_gemm_arch_family="hopper", fp8_block_gemm_kernel="CutlassFp8BlockScaledMMKernel")
    assert profile_mismatches(f, applied) == []
    for got in ("FlashInferFp8DeepGEMMDynamicBlockScaledKernel", None, "<mixed>"):
        bad = profile_mismatches(f, {**applied, "fp8_block_gemm_kernel": got})
        assert bad == [f"fp8_block_gemm declared 'cutlass' (CutlassFp8BlockScaledMMKernel), applied {got!r}"]
    bad = profile_mismatches(f, {k: v for k, v in applied.items() if k != "fp8_block_gemm_kernel"})
    assert bad == ["fp8_block_gemm declared 'cutlass' (CutlassFp8BlockScaledMMKernel), applied '<absent>'"]
    # undeclared: the kernel read-back is not judged by the profile (the FP8 rules refuse an unregistered kernel's op by name)
    assert profile_mismatches(e, {**applied, "fp8_block_gemm_kernel": "FlashInferFp8DeepGEMMDynamicBlockScaledKernel"}) == []


def test_moe_backend_pins_the_triton_experts_and_keeps_undeclared_digests():
    """`moe_backend='triton'` is the `EngineArgs.moe_backend` the derivation and the engine both apply, and requires every fused-MoE
    layer's read-back experts class to be `TritonExperts`; an undeclared profile serialises exactly as before."""
    e = TargetProfile(compute_capability=(12, 0), num_sms=188)
    assert "moe_backend" not in e.to_json() and e.moe_experts_class() is None and "moe_backend" not in e.engine_args()
    t = e.with_overrides(moe_backend="triton")
    assert t.to_json()["moe_backend"] == "triton" and t.digest() != e.digest() and TargetProfile.from_json(t.to_json()) == t
    assert t.engine_args()["moe_backend"] == "triton" and t.engine_args(execution=False)["moe_backend"] == "triton"
    assert t.moe_experts_class() == "TritonExperts" and t.execution_label == e.execution_label and t.env() == e.env()
    assert parse_target_arg('{"moe_backend": "triton"}', base=e) == t
    for v in ("auto", "deep_gemm", "TRITON"):
        with pytest.raises(ValueError):
            TargetProfile(moe_backend=v)
    applied = _applied(t, moe_experts_class="TritonExperts")
    assert profile_mismatches(t, applied) == []
    for got in ("DeepGemmExperts", None, "<mixed>"):
        assert profile_mismatches(t, {**applied, "moe_experts_class": got}) == [
            f"moe_backend declared 'triton' (TritonExperts), applied {got!r}"]
    assert profile_mismatches(t, {k: v for k, v in applied.items() if k != "moe_experts_class"}) == [
        "moe_backend declared 'triton' (TritonExperts), applied '<absent>'"]
    assert profile_mismatches(e, {**applied, "moe_experts_class": "DeepGemmExperts"}) == []


class _Layer:
    def __init__(self, experts_cls=None, **attrs):
        self.quant_method = type("QM", (), {"experts_cls": experts_cls, **attrs})() if experts_cls or attrs else None


class _Model:
    def __init__(self, *layers):
        self.layers = layers

    def modules(self):
        return iter((self, *self.layers))


def test_moe_experts_class_is_read_back_from_every_fused_moe_layer():
    from verity_vllm.program.frontend.target_profile import moe_experts_class_of
    TritonExperts, DeepGemmExperts = type("TritonExperts", (), {}), type("DeepGemmExperts", (), {})
    linear = _Layer(fp8_linear=object())
    assert moe_experts_class_of(_Model(linear, _Layer())) is None
    assert moe_experts_class_of(_Model(linear, _Layer(TritonExperts), _Layer(TritonExperts))) == "TritonExperts"
    assert moe_experts_class_of(_Model(_Layer(TritonExperts), _Layer(DeepGemmExperts))) == "<mixed>"


def test_the_engine_refuses_another_experts_class_than_the_declared_one(monkeypatch):
    pytest.importorskip("torch")
    import types
    from verity_vllm.engine import build as B
    t = TargetProfile(moe_backend="triton")
    monkeypatch.setattr(B.torch.cuda, "get_device_properties",
                        lambda i: types.SimpleNamespace(major=t.compute_capability[0], minor=t.compute_capability[1], multi_processor_count=t.num_sms))
    monkeypatch.setattr(B, "observed_flash_attn_version", lambda model: t.effective_flash_attn_version())
    monkeypatch.setattr(B, "compilation_facts", lambda llm: {})

    def engine(*layers):
        return types.SimpleNamespace(target=t, llm=None, model=_Model(*layers))
    assert B.target_mismatches(engine(_Layer(type("TritonExperts", (), {})))) == []
    assert B.target_mismatches(engine(_Layer(type("DeepGemmExperts", (), {})))) == [
        "moe_backend declared 'triton' (TritonExperts), layers report 'DeepGemmExperts'"]
    assert B.target_mismatches(engine()) == ["moe_backend declared 'triton' (TritonExperts), layers report None"]


def test_parse_target_arg_accepts_the_fp8_field_that_the_default_omits():
    p = parse_target_arg('{"compute_capability": [9, 0], "num_sms": 132, "fp8_block_gemm": "cutlass"}')
    assert p.fp8_block_gemm == "cutlass" and p.env()["VLLM_USE_DEEP_GEMM"] == "0"
    with pytest.raises(ValueError):
        parse_target_arg('{"fp8_block_gem": "cutlass"}')


def test_parse_target_arg_lays_a_partial_declaration_over_the_host_profile():
    """[R15 fp8] a row that declares only a knob is built for the pod it runs on: the JSON overrides lay over `base` (the device profile
    on a CUDA host), never over the L40S default -- rc3's Build of the FP8 panel row declared capability 8.9 on the H100 and vLLM selected
    the sm89 kernel, refused against the declared cutlass one."""
    h100 = TargetProfile(compute_capability=(9, 0), num_sms=132)
    p = parse_target_arg('{"fp8_block_gemm": "cutlass"}', base=h100)
    assert p.compute_capability == (9, 0) and p.num_sms == 132 and p.fp8_block_gemm == "cutlass"
    assert p == h100.with_overrides(fp8_block_gemm="cutlass")
    # an explicit capability in the declaration still wins over the host
    q = parse_target_arg('{"compute_capability": [8, 9], "num_sms": 128}', base=h100)
    assert q.compute_capability == (8, 9) and q.num_sms == 128


def test_moe_construction_is_a_declared_and_bound_static_never_applied_to_the_engine_and_keeps_undeclared_digests():
    """[R17-42 route A (3)] `moe_construction` (None = indexed-read-ordered, `padded` = MoeFfnFixedL_v1 per token) is a Program static:
    an undeclared profile serialises exactly as before (no key, digest unchanged); `padded` changes the digest; it is NOT an engine knob
    (`engine_args` unchanged), NOT judged by `profile_mismatches` (nothing is read back: never attested), and one spelling only."""
    e = TargetProfile()
    assert "moe_construction" not in e.to_json() and TargetProfile.from_json(e.to_json()).digest() == e.digest()
    assert e.effective_moe_construction() == "indexed-read-ordered" and e.moe_construction_declaration()["declared_explicitly"] is False
    p = e.with_overrides(moe_construction="padded")
    assert p.to_json()["moe_construction"] == "padded" and p.digest() != e.digest() and TargetProfile.from_json(p.to_json()) == p
    assert p.effective_moe_construction() == "padded" and p.execution_label == e.execution_label == "bi-eager"   # not another execution
    assert p.engine_args() == e.engine_args() and p.modelled_engine() == e.modelled_engine()                     # never applied to vLLM
    assert p.lag() == e.lag() == 1                                                                                  # orthogonal to LAG
    d = p.moe_construction_declaration()
    assert d["value"] == "padded" and d["declared_explicitly"] is True and d["attested"] is False
    assert "MoeFfnFixedL_v1{E,TOPK,VPT,H,I}" in d["definitions"] and "LMoeFfnL_v1" in d["lift_of_record"] and d["profile_digest"] == p.digest()
    assert "MoeExpertGemm_v1{E,K,N}" in e.moe_construction_declaration()["definitions"]
    # explicit indexed-read-ordered: the same construction as undeclared, a different (explicit) digest; explicit indexed-read keeps the old router
    o = e.with_overrides(moe_construction="indexed-read-ordered")
    assert o.effective_moe_construction() == "indexed-read-ordered" and o.to_json()["moe_construction"] == "indexed-read-ordered" and o.digest() != e.digest()
    i = e.with_overrides(moe_construction="indexed-read")
    assert i.effective_moe_construction() == "indexed-read" and i.moe_construction_declaration()["definitions"][0] == "MoeRouterTopK_v1{E,TOPK,VPT}"
    for bad in ("MOE-01", "fixed-slot", "Padded", True, 1):
        with pytest.raises(ValueError):
            TargetProfile(moe_construction=bad)
    # never attested: the read-back has no such field and the declared profile is not mismatched by its absence
    applied = _applied(p)
    assert profile_mismatches(p, applied) == [] and profile_mismatches(e, applied) == []
    assert parse_target_arg('{"moe_construction": "padded"}').moe_construction == "padded"
    with pytest.raises(ValueError):
        parse_target_arg('{"moe_constructon": "padded"}')
