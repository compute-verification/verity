"""Compositional reuse (lane CF1): the Ampere and Hopper B1 programs are two binding
records of one `Serve_v2`, and the factored functions evaluated standalone on *captured* operands give the profile's
words under the profile's statics.

§2  `Serve_v2{C, LP, STEPS, GEMM, ATTENTION, NORM}` under the Ampere records corresponds node for node to TA1's `Serve_v1`,
    under the Hopper records to HP's `ServeHopper_v1` (`correspond_rebound`: same bodies, callee identity through the
    pairing; digests differ by construction because the identities carry the statics). The two records differ in
    `GEMM` and `ATTENTION` only; the cross binding is a different program.
§3  Standalone on captured operands (CPU, numpy, the IR oracle): GEMM on the 30 authenticated RTX 4090 coordinates and
    on the H100 wgmma tiles; RMSNorm on the RTX 4090 Triton rows and the SmolLM2 fused rows; attention on the RTX 4090
    FA2 calls. Same function + Ampere statics -> Ampere words; Hopper statics -> the Hopper words (tiles) / HP's FA3
    model (no Hopper attention operands are archived locally: CAPTURE_MISSING, composition.md §3); the cross recorded.
    Held-out shapes/models predicted then compared: RMSNorm at widths 64/1024/2048/3072 (Triton) and 2048 (fused, a
    Qwen1.5-MoE row), attention at T = 100/193/300 (partial last blocks), GEMM K=2048 x N=60 / N=1 (the MoE router).
    The compiled profile's RMSNorm is shown NOT to be a rebinding (the eager function on the compiled operands misses
    the compiled words; W4c's Inductor model hits them).

Everything runs without torch in well under a minute; the authored (`b1_authored`) v2 check needs torch and skips.
Fixtures: fixtures/{B1-b3eba618b1b5-*, W11-*, W11R-*, M1-*, B0-divergence-*}; ~/veritor-artifacts/hp/tc-hopper.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "vllm-poc", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
FIX = REPO / "fixtures"
W11 = FIX / "W11-40f0cebeb670-20260907T1800Z"
W11R = FIX / "W11R-40f0cebeb670-20260907T1802Z"
GEMM_FIX = FIX / "B1-b3eba618b1b5-20260907T1731Z"
M1 = FIX / "M1-0c9af072e5a9-20260907T2128Z"
B0_DIV = FIX / "B0-divergence-20260907T1604Z"
TC_HOPPER = Path(os.environ.get("VERITY_HP_ARCHIVE", Path.home() / "veritor-artifacts" / "hp")) / "tc-hopper" / "r1"

from verity.ml.circuits import Program, bind  # noqa: E402
from verity.ml.circuits.evaluate import evaluate_call  # noqa: E402
from verity_vllm.program.frontend.correspond import correspond, correspond_rebound  # noqa: E402
from verity_vllm.program.registry import b1, hopper as H, prims as P  # noqa: E402
from verity_vllm.program.frontend import serve_families
from tests.program.crosscheck import _standalone  # noqa: E402

TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
        "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 8, "BIAS": True, "tie_word_embeddings": True, "profile": {}}
SHAPES = [("TINY", TINY, 3, 2), ("B0", b1.B0_CONFIG, 4, 2), ("B1", b1.B1_CONFIG, 2, 1)]  # never B1 at LP=512: small specialisations only


def _need(path: Path, what: str):
    if not path.exists():
        pytest.skip(f"{what} not present: {path}")


def _ev(spec, *arrays) -> np.ndarray:
    """Evaluate one call of `spec` on word arrays (flattened in parameter order) with the IR oracle."""
    prog, call = _standalone(spec)
    args = [int(v) for a in arrays for v in np.asarray(a).ravel()]
    return np.array(evaluate_call(prog.circuit, call, args), dtype=np.uint32)


def _u16(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.uint32).astype(np.uint16)


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
    from verity.ml.circuits.defs import REGISTRY
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
    from verity.ml.circuits.codec import encode_program
    with pytest.raises(TypeError):
        encode_program(Program(bind(b1.ServeV2, C=TINY, LP=1, STEPS=0, GEMM=b1.GEMM_AMPERE, ATTENTION=b1.ATTENTION_FA2,
                                    NORM={"L0": b1.RMSNormTriton, "FUSED": b1.RMSNormFusedCuda})))


# =====================================================================================================================
# §3 standalone on captured operands
# =====================================================================================================================


def _gemm_fixtures():
    _need(GEMM_FIX, "B1 GEMM fixtures (RTX 4090 eager)")
    for d in sorted(p for p in GEMM_FIX.iterdir() if p.is_dir()):
        f = json.loads((d / "fixture.json").read_text())
        x = np.fromfile(d / "x.u16", dtype="<u2")
        if f["x_arg_cols"] is not None:
            lo, hi = f["x_arg_cols"]
            x = x[lo:hi]
        yield f, x, np.fromfile(d / "w_row.u16", dtype="<u2"), np.fromfile(d / "out_row.u16", dtype="<u2")


def test_gemm_on_ampere_captured_coordinates():
    """30 authenticated B1 GEMM coordinates (5 families x prefill/decode, K = 1536 / 8960): `GemmCoordinate_v2{K, DOT}` with
    the Ampere step gives the committed word every time. The cross — the Hopper step on the same operands — gives the
    same bf16 word on all 30 (recorded) although the f32 accumulators differ on 26/30: the step difference is below the
    bf16 store rounding on these rows."""
    n = exact = cross_bf16 = cross_f32 = 0
    for f, x, w, out in _gemm_fixtures():
        K, j = f["K"], f["j"]
        assert x.size == K == w.size
        ya = _ev(bind(b1.GemmCoordinateV2, K=K, DOT=P.AmpereBF16TcDot16), x, w)[0]
        yh = _ev(bind(b1.GemmCoordinateV2, K=K, DOT=H.HopperBF16WgmmaDot16), x, w)[0]
        aa = _ev(bind(b1.DotBf16V2, K=K, DOT=P.AmpereBF16TcDot16), x, w)[0]
        ah = _ev(bind(b1.DotBf16V2, K=K, DOT=H.HopperBF16WgmmaDot16), x, w)[0]
        n += 1
        exact += int(ya == out[j] == f["committed_word"])
        cross_bf16 += int(yh != out[j])
        cross_f32 += int(aa != ah)
    assert n == 30 and exact == 30
    assert (cross_bf16, cross_f32) == (0, 26), (cross_bf16, cross_f32)  # recorded: composition.md §3


def test_gemm_on_hopper_captured_tiles():
    """H100 `wgmma.mma_async m64n128k16` tiles (HP's tc-hopper campaign): from a zero accumulator (`precision` family)
    `DotBf16_v2{K=16, DOT=Hopper}` gives the captured f32 word on every sampled coordinate; the Ampere step gives the
    same words from zero (recorded: 0 differ) but, on the `acc` family (a non-zero accumulator, as every k-step after the
    first in a GEMM chain), the Hopper primitive is exact and the Ampere one differs on 756/2048 (37%) of the words."""
    _need(TC_HOPPER / "precision.npz", "HP's tc-hopper archive")
    z = np.load(TC_HOPPER / "precision.npz")
    A, Bt, C, D = z["A"], z["Bt"], z["C"], z["D"]
    assert not np.any(C[0]) and not np.any(C[1])
    n = hopper_exact = ampere_diff = 0
    for t in (0, 1):
        for m in range(0, 64, 8):
            for c in range(0, 128, 4):
                yh = _ev(bind(b1.DotBf16V2, K=16, DOT=H.HopperBF16WgmmaDot16), A[t, m], Bt[t, c])[0]
                ya = _ev(bind(b1.DotBf16V2, K=16, DOT=P.AmpereBF16TcDot16), A[t, m], Bt[t, c])[0]
                n += 1
                hopper_exact += int(yh == D[t, m, c])
                ampere_diff += int(ya != D[t, m, c])
    assert n == 512 and hopper_exact == 512 and ampere_diff == 0
    z = np.load(TC_HOPPER / "acc.npz")
    A, Bt, C, D = z["A"], z["Bt"], z["C"], z["D"]
    n = hopper_exact = ampere_diff = 0
    for m in range(0, 64, 2):
        for c in range(0, 128, 2):
            a, b, acc, d = [int(v) for v in A[0, m]], [int(v) for v in Bt[0, c]], int(C[0, m, c]), int(D[0, m, c])
            n += 1
            hopper_exact += int(H.HopperBF16WgmmaDot16.evaluate(acc, *a, *b) == d)
            ampere_diff += int(P.AmpereBF16TcDot16.evaluate(acc, *a, *b) != d)
    assert n == 2048 and hopper_exact == 2048
    assert ampere_diff == 756, ampere_diff  # recorded: 756/2048 on tile 0 (36.9 %), composition.md §3.1


def _w11r_weight(index: dict, sub: str, name: str) -> np.ndarray:
    w = index["weights"][name]
    return np.load(W11R / sub / (w["file"] if isinstance(w, dict) else w))


def test_rmsnorm_on_ampere_captured_rows():
    """The two norm functions of the NORM record on RTX 4090 eager rows: `RMSNormTriton_v1{1536}` on the batch-invariant
    Triton kernel's natural rows (W11R), `RMSNormFusedCuda_v2{576}` on SmolLM2's fused rows (W4c's divergence fixture,
    eager side) and `RMSNormTriton_v1{576}` on its layer-0 rows — exact. The Hopper record binds the same two functions
    (HP: the H100 runs the same kernels), so there is no Hopper cross for this family in eager mode."""
    _need(W11R / "natural" / "index.json", "W11R RMSNorm fixtures")
    idx = json.loads((W11R / "natural" / "index.json").read_text())
    c = next(c for c in idx["calls"] if c["rows"] <= 4)
    x, out = np.load(W11R / "natural" / c["x_file"]), np.load(W11R / "natural" / c["out_file"])
    w = _w11r_weight(idx, "natural", c["weight"])
    for r in range(x.shape[0]):
        assert np.array_equal(_u16(_ev(bind(b1.RMSNormTriton, N=1536, EPS=c["eps"]), x[r], w)), out[r]), r
    assert b1.norm_bundle(b1.B1_CONFIG) == H.SERVE_HOPPER(b1.B1_CONFIG, 1, 0).bindings["NORM"]
    _need(B0_DIV / "index.json", "B0 divergence fixture")
    fused = B0_DIV / "layers.0.post_attention_layernorm_fused_add_rms_norm"
    w = np.load(fused / "weight.npy")
    x, res = np.load(fused / "eager" / "in_x.npy"), np.load(fused / "eager" / "in_residual.npy")
    normed, rout = np.load(fused / "eager" / "out_normed.npy"), np.load(fused / "eager" / "out_residual_out.npy")
    for r in range(6):
        y = _u16(_ev(bind(b1.RMSNormFusedCuda, N=576, EPS=1e-5), x[r], res[r], w))
        assert np.array_equal(y[:576], normed[r]) and np.array_equal(y[576:], rout[r]), r
    l0 = B0_DIV / "layers.0.input_layernorm_embedding"
    w, x, normed = np.load(l0 / "weight.npy"), np.load(l0 / "eager" / "in_x.npy"), np.load(l0 / "eager" / "out_normed.npy")
    for r in range(3):
        assert np.array_equal(_u16(_ev(bind(b1.RMSNormTriton, N=576, EPS=1e-5), x[r], w)), normed[r]), r


def test_rmsnorm_compiled_profile_is_not_a_rebinding():
    """The third profile (Ampere-compiled, W1c/W1d): the same fused-norm function applied to the compiled kernel's own
    operands misses the compiled words on a quarter of them (recorded 74,044 / 294,912) while W4c's Inductor model —
    the sum kept unrounded in f32, `libdevice.rsqrt` — hits all but 3; on the residual-free layer-0 norm (same operands
    on both sides) eager and compiled words differ on 1 word. Compiled RMSNorm is a different function, not a binding
    of `NORM` (compose vs extend: composition.md §1 row 2, §4)."""
    _need(B0_DIV / "index.json", "B0 divergence fixture")
    from tests.program import inductor_models as IM
    fused = B0_DIV / "layers.0.post_attention_layernorm_fused_add_rms_norm"
    w = np.load(fused / "weight.npy")
    cx, cr, cn = (np.load(fused / "compiled" / f) for f in ("in_x.npy", "in_residual.npy", "out_normed.npy"))
    eager_normed, _ = IM.fused_add_rms_norm_eager(cx, cr, w, 1e-5)  # W4b's twin of RMSNormFusedCuda on the compiled operands
    compiled_normed = IM.fused_add_rms_norm_compiled(cx, cr, w, 1e-5)
    miss_eager = int(np.count_nonzero(eager_normed != cn))
    miss_compiled = int(np.count_nonzero(compiled_normed != cn))
    assert cn.size == 294912 and miss_eager == 74044 and miss_compiled == 3, (miss_eager, miss_compiled)
    # the IR function agrees with its twin where the twin was checked above (a few rows of the compiled operands, too)
    for r in range(3):
        y = _u16(_ev(bind(b1.RMSNormFusedCuda, N=576, EPS=1e-5), cx[r], cr[r], w))
        assert np.count_nonzero(y[:576] != cn[r]) == np.count_nonzero(eager_normed[r] != cn[r])
    l0 = B0_DIV / "layers.0.input_layernorm_embedding"
    en, cn0 = np.load(l0 / "eager" / "out_normed.npy"), np.load(l0 / "compiled" / "out_normed.npy")
    assert np.array_equal(np.load(l0 / "eager" / "in_x.npy"), np.load(l0 / "compiled" / "out_embedding.npy"))
    assert int(np.count_nonzero(en != cn0)) == 1
    assert int(np.count_nonzero(IM.rms_norm_compiled(np.load(l0 / "compiled" / "out_embedding.npy"), np.load(l0 / "weight.npy"), 1e-5) != cn0)) == 0


def _attention_call(T: int, layer: int):
    idx = json.loads((W11 / "natural" / "index.json").read_text())
    c = next(c for c in idx["calls"] if c["phase"] == "decode" and c["Lk"] == T and c["layer"] == layer)
    q = np.load(W11 / "natural" / c["q_file"])[0]  # (12, 128)
    out = np.load(W11 / "natural" / c["out_file"])[0]
    k = np.load(W11 / "natural" / c["k_file"])[:T]  # (T, 2, 128): head h reads kv head h // 6
    v = np.load(W11 / "natural" / c["v_file"])[:T]
    return q, k, v, out


# (T, layer, heads evaluated, recorded FA2-vs-FA3 differing words over those heads)
ATTENTION_CASES = [(64, 13, (0, 7), 0), (65, 27, (0, 7), 3), (100, 0, (0, 3, 7, 11), 1), (193, 13, (0, 3, 7, 11), 30), (300, 27, (0, 3, 7, 11), 10)]


@pytest.mark.parametrize("T,layer,heads,recorded_cross", ATTENTION_CASES, ids=[f"T={c[0]}" for c in ATTENTION_CASES])
def test_attention_on_ampere_captured_calls(T, layer, heads, recorded_cross):
    """FA2 decode calls of Qwen2.5-1.5B on the RTX 4090 (W11 natural): `AttentionHead_v2{T, 128, **ATTENTION_FA2}` gives the
    kernel's words on every head; under `ATTENTION_FA3` (BN 128, wgmma step, Fa3InvSum) the same function gives HP's FA3
    CPU model's words (the model HP checked word for word against 39 H100 leaves; no H100 attention operands are archived
    locally); the two differ on the recorded number of words. T = 100/193/300 are held-out positions (composition.md §3:
    2/4/5 FA2 blocks, the last-visited one partial; 1/2/3 FA3 blocks)."""
    _need(W11 / "natural" / "index.json", "W11 attention fixtures")
    from verity_vllm.program.kernels import fa2_model as FA2, fa3_model as FA3
    from verity_vllm.program.kernels.fa2_relation import tables_dir
    try:
        FA3.lib()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"FA3 CPU model not buildable: {e}")
    tables = FA2.MufuTables.load(tables_dir())
    q, k, v, out = _attention_call(T, layer)
    cross = 0
    for h in heads:
        g = h // 6
        y2 = _u16(_ev(bind(b1.AttentionHeadV2, T=T, D=128, **b1.ATTENTION_FA2), q[h], k[:, g], v[:, g]))
        assert np.array_equal(y2, out[h]), (T, h)
        y3 = _u16(_ev(bind(b1.AttentionHeadV2, T=T, D=128, **H.ATTENTION_FA3), q[h], k[:, g], v[:, g]))
        rr = FA3.attention_call(q[h][None, None, :], k[:, g][:, None, :], v[:, g][:, None, :], np.array([0]), 1, T, 128 ** -0.5, tables)
        assert np.array_equal(y3, np.asarray(rr.out).reshape(-1)[:128]), (T, h)
        cross += int(np.count_nonzero(y2 != y3))
    assert cross == recorded_cross, (T, cross)


# =====================================================================================================================
# §3 held-out shapes / models: predicted exact, then compared
# =====================================================================================================================


@pytest.mark.parametrize("case,N", [("n_cols_64", 64), ("n_cols_1024", 1024), ("n_cols_2048", 2048), ("n_cols_3072", 3072)])
def test_heldout_rmsnorm_width_triton(case, N):
    """`RMSNormTriton_v1{N}` at widths never used in development (576 and 1536 were): 8 production rows each (W11R stress
    `n_cols`, RTX 4090). Predicted exact — the body's chunk/lane structure is a function of N."""
    _need(W11R / "stress" / "index.json", "W11R stress fixtures")
    idx = json.loads((W11R / "stress" / "index.json").read_text())
    c = next(c for c in idx["calls"] if c["case"] == case)
    x, out = np.load(W11R / "stress" / c["x_file"]), np.load(W11R / "stress" / c["out_file"])
    w = _w11r_weight(idx, "stress", c["weight"])
    assert c["n_cols"] == N and x.shape == (8, N)
    for r in range(8):
        assert np.array_equal(_u16(_ev(bind(b1.RMSNormTriton, N=N, EPS=c["eps"]), x[r], w)), out[r]), (N, r)


def _m1_openings(fixture: str):
    f = json.loads((M1 / fixture / "fixture.json").read_text())
    op = f["openings"]

    def val(suffix: str) -> np.ndarray:
        key = next(k for k in op if k.endswith(suffix))
        return np.frombuffer(base64.b64decode(op[key]["value_b64"]), dtype="<u2")

    return f["selection"]["layer"], val


@pytest.mark.parametrize("fixture", ["M1-l3-decode", "M1-l23-prefill"])
def test_heldout_model_qwen_moe_width_2048(fixture):
    """A held-out model and width: Qwen1.5-MoE-A2.7B (H = 2048, RTX 4090 eager, W9's authenticated MoE fixtures). The fused
    add+norm `RMSNormFusedCuda_v2{2048}` (every thread owns exactly two elements: a pattern neither 576 nor 1536 has) and the
    router GEMM `Gemm_v2{K=2048, N=60}` / shared-expert gate `Gemm_v2{K=2048, N=1}` (shapes never used in development).
    Predicted exact under the Ampere records; the Hopper step on the router gives the same 60 bf16 words (recorded)."""
    _need(M1 / fixture / "fixture.json", "M1 MoE fixtures")
    L, val = _m1_openings(fixture)
    x, res = val(f"layers.{L}.self_attn.o_proj.gemm/port=0"), val(f"layers.{L}.input_layernorm.rmsnorm/port=1")
    w, normed = val(f"w=layers.{L}.post_attention_layernorm.weight"), val(f"layers.{L}.post_attention_layernorm.rmsnorm/port=0")
    y = _u16(_ev(bind(b1.RMSNormFusedCuda, N=2048, EPS=1e-6), x, res, w))
    assert np.array_equal(y[:2048], normed)
    W = val(f"w=layers.{L}.mlp.gate.weight").reshape(60, 2048)
    logits = val(f"layers.{L}.mlp.gate.gemm/port=0")
    assert np.array_equal(_u16(_ev(bind(b1.GemmV2, K=2048, N=60, DOT=P.AmpereBF16TcDot16), normed, W)), logits)
    assert np.array_equal(_u16(_ev(bind(b1.GemmV2, K=2048, N=60, DOT=H.HopperBF16WgmmaDot16), normed, W)), logits)  # recorded: no cross at bf16
    Wg = val(f"w=layers.{L}.mlp.shared_expert_gate.weight").reshape(1, 2048)
    assert _u16(_ev(bind(b1.GemmV2, K=2048, N=1, DOT=P.AmpereBF16TcDot16), normed, Wg))[0] == val(f"layers.{L}.mlp.shared_expert_gate.gemm/port=0")[0]


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
# [E7 D54 2026-09-13] Serve_v4: the kernel-geometry attention chain beside the untouched Serve_v1
# =====================================================================================================================


def test_fa2_kblock_n_is_the_pinned_kernels_align_rule():
    """vllm-flash-attn 506341a `flash_fwd_launch_template.h` `run_mha_fwd_splitkv_align`: kBlockN = Headdim <= 64 ? 128 : 64."""
    assert b1.fa2_kblock_n(32) == 128 and b1.fa2_kblock_n(64) == 128 and b1.fa2_kblock_n(96) == 64 and b1.fa2_kblock_n(128) == 64
    assert b1.BN == 64 == b1.fa2_kblock_n(128)  # the v1 constant IS the head-dim-128 value (W11's measured trait)
    assert b1.ATTENTION_FA2 == b1.attention_fa2(128) and b1.attention_fa2(64)["BN"] == 128


def test_serve_v1_is_untouched_and_serve_v4_is_registered_beside_it():
    """Registry version discipline: `Serve_v1`'s chain still reads the constant (its pinned digests must keep reproducing);
    the corrected chain is a new set of versions, resolvable through `SERVE_FAMILIES`."""
    from verity.ml.circuits.defs import REGISTRY
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
    # the identity binder (D57): no geometry -> Serve_v1; a geometry -> Serve_v4{..., BN}; mismatched pairs refused
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
