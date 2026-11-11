"""Compositional reuse: the Ampere and Hopper B1 programs are two binding
records of one `Serve_v2`, and the factored functions evaluated standalone on *captured* operands give the profile's
words under the profile's statics.

§2  `Serve_v2{C, LP, STEPS, GEMM, ATTENTION, NORM}` under the Ampere records corresponds node for node to TA1's `Serve_v1`,
    under the Hopper records to HP's `ServeHopper_v1` (`correspond_rebound`: same bodies, callee identity through the
    pairing; digests differ by construction because the identities carry the statics). The two records differ in
    `GEMM` and `ATTENTION` only; the cross binding is a different program.
§3  Standalone on captured operands is `test_composition_of_record`: the captures it reads were made on GPU pods.

Everything runs without torch in well under a minute; the authored (`b1_authored`) v2 check needs torch and skips.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "vllm-poc", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from verity.circuits import Program, bind  # noqa: E402
from verity_vllm.program.frontend.correspond import correspond, correspond_rebound  # noqa: E402
from verity.definitions.registry import b1, hopper as H, prims as P  # noqa: E402
from verity_vllm.program.frontend import serve_families

TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
        "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 8, "BIAS": True, "tie_word_embeddings": True, "profile": {}}
SHAPES = [("TINY", TINY, 3, 2), ("B0", b1.B0_CONFIG, 4, 2), ("B1", b1.B1_CONFIG, 2, 1)]  # never B1 at LP=512: small specialisations only


def _hp_config() -> dict:
    """HP's port configuration (`vllm-poc/profiles/b1-hopper/static_config.py`: B1_CONFIG with the H100 profile record)."""
    path = REPO / "vllm-poc" / "profiles" / "b1-hopper" / "static_config.py"
    if not path.exists():
        return {**b1.B1_CONFIG, "profile": {**b1.B1_CONFIG["profile"], "hardware": "H100 (sm_90)"}}
    spec = importlib.util.spec_from_file_location("hp_static_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONFIG


# =====================================================================================================================
# §2 one Serve, two binding records
# =====================================================================================================================


def test_factored_definitions_are_new_versions_beside_the_old():
    """Registry version discipline: every factored definition is `Name_v2`, every v1 (and HP's `*Hopper_v1`) is untouched."""
    from verity.circuits.defs import REGISTRY
    for v1, v2 in b1.FACTORED_FROM.items():
        assert v2.name == v1.name and v2.version == v1.version + 1
        assert REGISTRY[v1.id] is v1 and REGISTRY[v2.id] is v2
        assert set(v1.statics) < set(v2.statics), (v1.id, v2.id)
    assert set(b1.ServeV2.statics) == {"C", "LP", "STEPS", "GEMM", "ATTENTION", "NORM"}
    assert set(b1.AttentionHeadV2.statics) == {"T", "D", "BN", "DOT", "INV"}
    assert set(b1.GemmV2.statics) == {"K", "N", "DOT"}
    for d in H.HOPPER_COMPOSITES:
        assert REGISTRY[d.id] is d and d.version == 1


def _renamed_statics(report) -> dict[str, set[str]]:
    """base name of the left definition -> the statics its right-hand pair introduced (`right_only` keys)."""
    out: dict[str, set[str]] = {}
    for p in report.renamed:
        if p["left"].startswith("Root["):
            continue
        base = p["left"].split("{")[0]
        out.setdefault(base, set()).update(p.get("statics", {}).get("right_only", {}).keys())
        assert not p.get("statics", {}).get("left_only") and not p.get("statics", {}).get("differ"), p
    return out


@pytest.mark.parametrize("label,C,LP,STEPS", SHAPES, ids=[s[0] for s in SHAPES])
def test_ampere_binding_corresponds_to_ta1_serve(label, C, LP, STEPS):
    """`Serve_v2` under GEMM_AMPERE / ATTENTION_FA2 / norm_bundle(C) is TA1's `Serve_v1` node for node: every paired body
    `identical` (byte-equal up to callee ids), same gates, same S.out; digests differ (the identities carry the statics)."""
    left = Program(bind(b1.Serve, C=C, LP=LP, STEPS=STEPS))
    right = Program(b1.SERVE_AMPERE(C, LP, STEPS))
    r = correspond_rebound(left, right)
    assert r.corresponds, [m.to_json() for m in r.mismatches][:3]
    assert r.root_out == "identical" and r.same_references == 0 and r.identical == r.definitions
    assert not r.digest_equal and r.gates_left == r.gates_right
    assert not r.unpaired_left and not r.unpaired_right
    intro = _renamed_statics(r)
    assert intro == {"DotBf16_v1": {"DOT"}, "GemmCoordinate_v1": {"DOT"}, "Gemm_v1": {"DOT"},
                     "AttnBlock_v1": {"BN", "DOT"}, "AttentionHead_v1": {"BN", "DOT", "INV"}, "Attention_v1": {"BN", "DOT", "INV"},
                     "LayerPre_v1": {"GEMM", "NORM"}, "LayerPost_v1": {"GEMM", "NORM"}, "Final_v1": {"GEMM", "NORM"}}, intro
    for p in r.renamed:  # the Ampere values
        s = json.dumps(p.get("statics", {}))
        assert "Hopper" not in s and "Fa3" not in s and '"BN": 128' not in s, p
    # and the plain same-id correspondence still refuses: the definition sets differ by identity
    assert not correspond(left, right).corresponds


@pytest.mark.parametrize("label,C,LP,STEPS", SHAPES, ids=[s[0] for s in SHAPES])
def test_hopper_binding_corresponds_to_hp_serve_hopper(label, C, LP, STEPS):
    """`Serve_v2` under GEMM_HOPPER / ATTENTION_FA3 / norm_bundle(C) is HP's `ServeHopper_v1` node for node."""
    if label == "B1":
        C = _hp_config()  # HP's own C (the profile record differs from TA1's; both sides get the same C)
    left = Program(bind(H.ServeHopper, C=C, LP=LP, STEPS=STEPS))
    right = Program(H.SERVE_HOPPER(C, LP, STEPS))
    r = correspond_rebound(left, right)
    assert r.corresponds, [m.to_json() for m in r.mismatches][:3]
    assert r.root_out == "identical" and r.same_references == 0 and r.identical == r.definitions
    assert not r.digest_equal and r.gates_left == r.gates_right
    intro = _renamed_statics(r)
    assert intro == {"DotBf16Hopper_v1": {"DOT"}, "GemmCoordinateHopper_v1": {"DOT"}, "GemmHopper_v1": {"DOT"},
                     "AttnBlockFA3_v1": {"BN", "DOT"}, "AttentionHeadFA3_v1": {"BN", "DOT", "INV"}, "AttentionFA3_v1": {"BN", "DOT", "INV"},
                     "LayerPreHopper_v1": {"GEMM", "NORM"}, "LayerPostHopper_v1": {"GEMM", "NORM"}, "FinalHopper_v1": {"GEMM", "NORM"}}, intro
    for p in r.renamed:
        s = p.get("statics", {}).get("right_only", {})
        if "DOT" in s:
            assert s["DOT"] == {"fn": "HopperBF16WgmmaDot16_v1"}
        if "INV" in s:
            assert s["INV"] == {"fn": "Fa3InvSum_v1"}
        if "BN" in s:
            assert s["BN"] == 128


def test_two_records_differ_in_gemm_and_attention_only():
    """The Ampere and Hopper programs: one definition, binding records equal except `GEMM` and `ATTENTION`; the
    statics that differ are exactly the environment substitution HP made (`_BoundPrims._over`, `ENV["BN"]`)."""
    a, h = b1.SERVE_AMPERE(TINY, 3, 2), H.SERVE_HOPPER(TINY, 3, 2)
    assert a.defn is h.defn is b1.ServeV2
    differ = {k for k in b1.ServeV2.statics if a.bindings[k] != h.bindings[k]}
    assert differ == {"GEMM", "ATTENTION"}
    assert a.bindings["NORM"] == h.bindings["NORM"]  # TA1's norms on both (the H100 runs the same kernels)
    assert b1.GEMM_AMPERE == {"DOT": P.AmpereBF16TcDot16} and H.GEMM_HOPPER == {"DOT": H.HopperBF16WgmmaDot16}
    assert b1.ATTENTION_FA2 == {"BN": 64, "DOT": P.AmpereBF16TcDot16, "INV": P.Fa2InvSum}
    assert H.ATTENTION_FA3 == {"BN": 128, "DOT": H.HopperBF16WgmmaDot16, "INV": H.Fa3InvSum}
    assert H._BoundPrims._over == {"AmpereBF16TcDot16": H.HopperBF16WgmmaDot16, "Fa2InvSum": H.Fa3InvSum} and H.KBLOCK_N_FA3 == 128


def test_cross_binding_is_a_different_program():
    """Ampere `Serve_v2` vs Hopper `Serve_v2`: no correspondence — every mismatch is inside attention (the head's call of
    the block: `Array<BN>` K/V rows, 64 vs 128), where the two profiles compute differently."""
    r = correspond_rebound(Program(b1.SERVE_AMPERE(TINY, 3, 2)), Program(H.SERVE_HOPPER(TINY, 3, 2)))
    assert not r.corresponds and r.mismatches
    assert all(m.id.startswith(("AttentionHead_v2", "AttnBlock_v2")) for m in r.mismatches), [m.id[:40] for m in r.mismatches][:3]
    assert all("Array<64" in m.detail and "Array<128" in m.detail for m in r.mismatches), r.mismatches[0].detail


def test_partial_specialisations_are_not_static_values():
    """The limit the factoring works within (composition.md §2, finding F1): a composite with unbound statics cannot be
    a static value (the codec has no identity for it), so the attention *algorithm* is selected through leaf statics
    on one body, not by passing `AttentionFA3{...}` as a value."""
    from verity.circuits.codec import encode_program
    with pytest.raises(TypeError):
        encode_program(Program(bind(b1.ServeV2, C=TINY, LP=1, STEPS=0, GEMM=b1.GEMM_AMPERE, ATTENTION=b1.ATTENTION_FA2,
                                    NORM={"L0": b1.RMSNormTriton, "FUSED": b1.RMSNormFusedCuda})))


# =====================================================================================================================
# the authored (torch) v2 functions against the registry's — pod only
# =====================================================================================================================

AUTHORED_CASES = [
    ("AttnBlock_v2{D=16,NVIS=37,FA2}", "AttnBlockV2", {"D": 16, "NVIS": 37, "FIRST": False, "BN": 64, "DOT": P.AmpereBF16TcDot16}),
    ("AttnBlock_v2{D=16,NVIS=100,FA3}", "AttnBlockV2", {"D": 16, "NVIS": 100, "FIRST": True, "BN": 128, "DOT": H.HopperBF16WgmmaDot16}),
    ("AttentionHead_v2{T=70,FA2}", "AttentionHeadV2", {"T": 70, "D": 16, **b1.ATTENTION_FA2}),
    ("AttentionHead_v2{T=130,FA3}", "AttentionHeadV2", {"T": 130, "D": 16, **H.ATTENTION_FA3}),
    ("Attention_v2{T=5,FA3}", "AttentionV2", {"T": 5, "NH": 2, "KVH": 1, "D": 16, **H.ATTENTION_FA3}),
    ("LayerPre_v2{TINY,L0,Ampere}", "LayerPreV2", {"C": TINY, "L0": True, "GEMM": b1.GEMM_AMPERE, "NORM": b1.norm_bundle(TINY)}),
    ("LayerPost_v2{TINY,Hopper}", "LayerPostV2", {"C": TINY, "GEMM": H.GEMM_HOPPER, "NORM": b1.norm_bundle(TINY)}),
    ("Final_v2{TINY,Ampere}", "FinalV2", {"C": TINY, "GEMM": b1.GEMM_AMPERE, "NORM": b1.norm_bundle(TINY)}),
    ("Serve_v2{TINY,Ampere}", "ServeV2", {"C": TINY, "LP": 3, "STEPS": 2, "GEMM": b1.GEMM_AMPERE, "ATTENTION": b1.ATTENTION_FA2, "NORM": b1.norm_bundle(TINY)}),
    ("Serve_v2{TINY,Hopper}", "ServeV2", {"C": TINY, "LP": 3, "STEPS": 2, "GEMM": H.GEMM_HOPPER, "ATTENTION": H.ATTENTION_FA3, "NORM": b1.norm_bundle(TINY)}),
]


@pytest.mark.parametrize("label,name,statics", AUTHORED_CASES, ids=[c[0] for c in AUTHORED_CASES])
def test_authored_v2_corresponds_to_registry_v2(label, name, statics):
    """`b1_authored` v2 functions take the same statics and elaborate to the registry's v2 bodies (digest-equal or
    same-references across the broadcast encodings, as FE2's v1 test). Needs torch (pod)."""
    pytest.importorskip("torch")
    from tests.program import b1_authored as A
    from verity_vllm.program.frontend import export_report
    rep = export_report(getattr(A, name), static=statics, dead_gates="ignore")
    P_t = Program(bind(getattr(b1, name), **statics))
    r = correspond(rep.program, P_t)
    assert r.corresponds, (label, [m.to_json() for m in r.mismatches][:3], [u.to_json() for u in r.unverified][:3])
    assert r.root_out == "identical" and rep.gates == P_t.gates


# =====================================================================================================================
# Serve_v4: the kernel-geometry attention chain beside the untouched Serve_v1
# =====================================================================================================================


def test_fa2_kblock_n_is_the_pinned_kernels_align_rule():
    """vllm-flash-attn 506341a `flash_fwd_launch_template.h` `run_mha_fwd_splitkv_align`: kBlockN = Headdim <= 64 ? 128 : 64."""
    assert b1.fa2_kblock_n(32) == 128 and b1.fa2_kblock_n(64) == 128 and b1.fa2_kblock_n(96) == 64 and b1.fa2_kblock_n(128) == 64
    assert b1.BN == 64 == b1.fa2_kblock_n(128)  # the v1 constant IS the head-dim-128 value (W11's measured trait)
    assert b1.ATTENTION_FA2 == b1.attention_fa2(128) and b1.attention_fa2(64)["BN"] == 128


def test_serve_v1_is_untouched_and_serve_v4_is_registered_beside_it():
    """Registry version discipline: `Serve_v1`'s chain still reads the constant (its pinned digests must keep reproducing);
    the corrected chain is a new set of versions, resolvable through `SERVE_FAMILIES`."""
    from verity.circuits.defs import REGISTRY
    for d, ver in ((b1.AttnBlock, 1), (b1.AttentionHead, 1), (b1.Attention, 1), (b1.Serve, 1),
                   (b1.AttnBlockV3, 3), (b1.AttentionHeadV3, 3), (b1.AttentionV3, 3), (b1.ServeV4, 4)):
        assert REGISTRY[d.id] is d and d.version == ver
    assert serve_families.serve_family(None) is b1.Serve and serve_families.serve_family("Serve_v1") is b1.Serve and serve_families.serve_family("Serve_v4") is b1.ServeV4
    with pytest.raises(KeyError):
        serve_families.serve_family("Serve_v9")
    # the v1 block takes 64 K rows at any head dim; the v3 block takes the DECLARED BN
    assert dict(bind(b1.AttnBlock, D=64, NVIS=1, FIRST=True).signature()[0])["k"].n == 64
    assert dict(bind(b1.AttnBlockV3, D=64, NVIS=1, FIRST=True, BN=128).signature()[0])["k"].n == 128
    assert dict(bind(b1.AttnBlockV3, D=64, NVIS=1, FIRST=True, BN=64).signature()[0])["k"].n == 64
    # the identity binder: no geometry -> Serve_v1; a geometry -> Serve_v4{..., BN}; mismatched pairs refused
    G = {"kernel": "fa2-splitkv-align", "kBlockM": 64, "kBlockN": 128, "num_splits": 1}
    assert serve_families.serve_program(TINY, 3, 2).id == bind(b1.Serve, C=TINY, LP=3, STEPS=2).id
    assert serve_families.serve_program(TINY, 3, 2, family="Serve_v4", attention=G).id == bind(b1.ServeV4, C=TINY, LP=3, STEPS=2, BN=128).id
    assert serve_families.serve_program(TINY, 3, 2, family="Serve_v4", attention=G).id.endswith(",BN=128}")
    for bad in ({**G, "kBlockN": 100}, {**G, "num_splits": 2}, {**G, "kernel": "fa3"}, {**G, "kBlockM": 128}):
        with pytest.raises(ValueError):
            serve_families.serve_program(TINY, 3, 2, family="Serve_v4", attention=bad)
    with pytest.raises(ValueError):
        serve_families.serve_program(TINY, 3, 2, family="Serve_v4")  # no geometry
    with pytest.raises(ValueError):
        serve_families.serve_program(TINY, 3, 2, attention=G)  # Serve_v1 with a foreign geometry


def test_serve_v4_differs_from_serve_v1_only_in_the_attention_chain():
    """Everything but the attention chain is TA1's definition object itself (LayerPre/LayerPost/Final/Embedding/Gemm...), so the
    two serving programs can only differ inside `Attention_v3` -- and there only in the block geometry (the gate-for-gate row
    identity for T <= 64 keys is `test_derived_rows.test_v1_and_v3_attention_heads_are_gate_for_gate_equal_within_one_block`)."""
    left = Program(bind(b1.Serve, C=TINY, LP=3, STEPS=2))
    right = Program(bind(b1.ServeV4, C=TINY, LP=3, STEPS=2, BN=128))
    assert left.gates == right.gates
    names_l = {n.fn.defn.name for n in left.root.body.nodes if hasattr(n.fn, "defn")}
    names_r = {n.fn.defn.name for n in right.root.body.nodes if hasattr(n.fn, "defn")}
    assert names_l == names_r
    vers = {(n.fn.defn.name, n.fn.defn.version) for n in right.root.body.nodes if hasattr(n.fn, "defn")}
    assert ("Attention", 3) in vers and ("Attention", 1) not in vers and ("LayerPre", 1) in vers and ("Final", 1) in vers


def test_serve_v4_binding_record_is_serve_v2_with_attention_fa2_of_D():
    """`Serve_v2` under `attention_fa2(C["D"])` (BN = kBlockN(D)) corresponds to `Serve_v4` node for node, as `ATTENTION_FA2`
    corresponds to `Serve_v1` -- the same binding-record relation, one geometry rule apart.  At a two-block T (D=16: 128 keys)."""
    C = dict(TINY)
    LP, STEPS = 3, 2  # TINY's MAX_POS is 8; the correspondence is about bodies, not shapes
    left = Program(bind(b1.ServeV4, C=C, LP=LP, STEPS=STEPS, BN=128))
    right = Program(b1.serve_v2(C, LP, STEPS, attention={**b1.ATTENTION_FA2, "BN": 128}))
    r = correspond_rebound(left, right)
    assert r.corresponds and r.root_out == "identical", [m.to_json() for m in r.mismatches][:3]
    assert r.gates_left == r.gates_right and not r.digest_equal
    # and `AttentionHead_v3{T=70, BN=128}` has ONE block where v1 has two; at BN=64 the same count
    assert bind(b1.AttentionHeadV3, T=70, D=16, BN=128).gates != bind(b1.AttentionHead, T=70, D=16).gates
    assert bind(b1.AttentionHeadV3, T=70, D=16, BN=64).gates == bind(b1.AttentionHead, T=70, D=16).gates


def test_fa2_geometry_is_derived_from_the_pinned_profiles_declared_kernel():
    """The semantic profiles the instances pin (`vocabulary.profile_sha256`) declare the attention kernel (`FlashAttentionBackend`,
    `fa_version 2`, `num_splits 1`, `head_size = HD`); `fa2_kernel_geometry` is the pinned source's dispatch rule over that
    declaration: kBlockN 128 at head dim 64 (SmolLM2-135M/360M, Qwen2.5-0.5B), 64 at head dim 128 -- what `Serve_v4` builds.
    A declaration outside the rule (FA3, num_splits != 1) is refused, never defaulted."""
    import json as _json
    root = Path(__file__).resolve().parents[2] / "manifests" / "semantic-profiles"
    if not root.exists():
        pytest.skip("semantic profiles not in this checkout")
    for name, D in (("llama-eager-batch-invariant-v3.json", 64), ("qwen2-eager-batch-invariant-v2.json", 64), ("llama-eager-batch-invariant-v3.json", 128)):
        consts = serve_families.profile_attention_consts(_json.loads((root / name).read_text()))
        assert consts["fa_version"] == 2 and consts["num_splits"] == 1
        g = serve_families.fa2_kernel_geometry(consts, D)
        assert g["kBlockM"] == 64 and g["kBlockN"] == b1.fa2_kblock_n(D) == (128 if D <= 64 else 64)
        assert serve_families.attention_geometry_static(g) == g["kBlockN"]  # the pin E6 writes binds as BN
    hopper = root / "qwen2-eager-batch-invariant-hopper-v1.json"
    if hopper.exists():
        with pytest.raises(ValueError):
            serve_families.fa2_kernel_geometry(serve_families.profile_attention_consts(_json.loads(hopper.read_text())), 128)  # FA3: not this rule
    with pytest.raises(ValueError):
        serve_families.fa2_kernel_geometry({"backend": "FlashAttentionBackend", "fa_version": 2, "num_splits": 0}, 64)  # the 256-key path: refused
