"""R15 dense (F-dense-05) — the unfused norm chain (`ir.ops.rms_norm` / `ir.ops.fused_add_rms_norm` as ATen on an f32 copy, vLLM
`GemmaRMSNorm.forward_native`) derived by the op-shaped rules of `vllm_bindings` into the promoted `verity_vllm.program.registry.dense` kinds of
the pinned `b1-eager-v3` vocabulary — the SAME specialization ids, in the SAME order, as the eager Match fold resolves the chain to
(`verity_vllm/engine/profiles/gen_dense_gemma2_patterns.py`), so `cb_a.compare` can be canonical-equal.

What is under test:
  * the residual-free form (`x -> (1+w) * x/rms(x)`): `AddScalarBf16{N,1.0} x1, SquareBf16{N} xM, MeanTriton{N} xM, AddScalarF32{1,eps} xM,
    RsqrtF32{1} xM, ScaleRowBf16{N} xM, MulVecF32{N} xM, NarrowF32ToBf16{N} xM` — the widening `x.to(f32)` emits NO gate;
  * the residual form: `AddWidenedBf16{N} xM, NarrowF32ToBf16{N} xM (member 1), SquareF32{N} xM, ..., ScaleRowF32{N} xM, MulVecF32, NarrowF32ToBf16 (member 0)`;
  * fail-closed: the `aten.mean.dim` binding refuses without the derived batch-invariant dispatch / with foreign kernel sources; a
    non-2 exponent, a genuine-f32 + genuine-f32 add, rsqrt of a widened view are refused (`numerics_unregistered`), never bound elsewhere;
  * bit-exactness vs the pinned vLLM `GemmaRMSNorm` under `VLLM_BATCH_INVARIANT=1` on CUDA (skipped without a GPU + vLLM): the derived
    Program evaluates to the kernel's words on every element (the MeanTriton reduction order is the GPU kernel's, not torch CPU's).

Run (pod venv): `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/program/test_norm_chain.py`
GPU part: `PYTHONPATH=. VLLM_BATCH_INVARIANT=1 python -m pytest -q -p no:cacheprovider tests/program/test_norm_chain.py -k cuda`
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.rules import vllm_bindings as VB
from verity_vllm.program.frontend.rules.vllm_bindings import MEAN_PINS, clear_observations, observe_export_facts, vllm_ruleset
from verity_vllm.program.kernels.serve3_reference import materialize

PROF = dict(mode="profile", profile="b1-eager-v3")
BF = torch.bfloat16
EPS = 1e-6


class GemmaNorm(nn.Module):
    """vLLM `GemmaRMSNorm.forward_native` verbatim (layernorm.py + ir/ops/layernorm.py): `(1 + w.float()) * rms_norm(x)`, the
    residual form returning `(normed, residual)`."""

    def __init__(self, H: int, eps: float = EPS) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(H, dtype=BF), requires_grad=False)
        self.variance_epsilon = eps

    def forward(self, x, residual=None):
        weight = self.weight.float() + 1.0
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            x_residual = x.to(orig_dtype)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(weight.dtype) * weight
        if residual is None:
            return x.to(orig_dtype)
        return x.to(orig_dtype), x_residual


class CubeNorm(GemmaNorm):
    """`pow(x, 3)` in place of the square: no registered kind."""

    def forward(self, x):
        weight = self.weight.float() + 1.0
        x32 = x.to(torch.float32)
        variance = x32.pow(3).mean(dim=-1, keepdim=True)
        return ((x32 * torch.rsqrt(variance + self.variance_epsilon)) * weight).to(x.dtype)


class F32PlusF32(nn.Module):
    """A genuine f32 + f32 add (neither operand a widening): no Definition."""

    def forward(self, x, y):
        a = x.to(torch.float32) * 2.0
        b = y.to(torch.float32) * 2.0
        return (a + b).to(x.dtype)


class RsqrtOfWidened(nn.Module):
    def forward(self, x):
        return torch.rsqrt(x.to(torch.float32)).to(x.dtype)


def _bi_facts(mean_override: dict | None = None) -> dict:
    """Export facts as `derive_step` observes them for a BI construction: BI GEMM launches in the graph + the installed mean override."""
    mo = mean_override if mean_override is not None else {k: MEAN_PINS[k] for k in ("kernel", "src_sha256", "mean_dim_src_sha256", "mean_batch_invariant_src_sha256")}
    mo = {**mo, "enable_batch_invariant_mode_registers_mean_dim": mo.get("enable_batch_invariant_mode_registers_mean_dim", True)}
    return {"fused_add_rms_norm_dispatch": "absent",
            "evidence": {"fused_add_rms_norm_ops": [], "bi_triton_kernels_launched": [VB.GEMM_PINS["kernel"]], "override_library_installed": False,
                         "forward_cuda_switch_in_process": True, "forward_native_ops_present": ["aten.pow.Tensor_Scalar", "aten.rsqrt.default"],
                         "mean_dim_override": mo}}


@pytest.fixture(autouse=True)
def _observed():
    clear_observations()
    observe_export_facts(_bi_facts())
    yield
    clear_observations()


def _derive(m: nn.Module, *inputs: torch.Tensor):
    return derive_report(m, tuple(torch.empty_like(t) for t in inputs), **PROF, rules=vllm_ruleset(), name=type(m).__name__)


def _emitted(rep) -> list[str]:
    return [a["emitted"] for a in rep.provenance["applications"] if a.get("emitted")]


def _bits16(t: torch.Tensor) -> list[int]:
    return [int(v) for v in t.detach().contiguous().cpu().view(torch.int16).numpy().astype(np.uint16).reshape(-1)]


def _evaluate(rep, module: nn.Module, inputs: list[torch.Tensor]) -> np.ndarray:
    """The Program's root outputs (the returned tensors flattened in return order) as words."""
    tensors = {**dict(module.named_buffers()), **dict(module.named_parameters())}
    words = [_bits16(x) for x in inputs]
    wbits: list[int] = []
    for w in rep.provenance["weights"]:
        wbits += _bits16(tensors[w["name"]])
    if wbits:
        words.append(wbits)
    t = materialize(rep.program, words)
    return np.array([np.uint32(t[g]) for g in rep.program.circuit.root_outputs()], dtype=np.uint32)


def _det(a: dict) -> dict:
    return a.get("preconditions_checked") or {}


# ---------------------------------------------------------------------------------------------------------
# structure: the chain binds to the promoted kinds, in the fold's order, with member hints on the narrows
# ---------------------------------------------------------------------------------------------------------


def test_residual_free_form_binds_the_chain():
    torch.manual_seed(0)
    m = GemmaNorm(32)
    with torch.no_grad():
        m.weight.copy_((torch.randn(32) * 0.1).to(BF))
    rep = _derive(m, torch.randn(3, 32).to(BF))
    assert rep.unsupported is None, f"refused: {rep.unsupported}"
    assert _emitted(rep) == [
        "transparent widening (no gates; readers bind *Bf16 kinds)",      # weight.float()
        "AddScalarBf16_v1{N=32,C=1.0} x1",                                  # + 1.0
        "transparent widening (no gates; readers bind *Bf16 kinds)",      # x.to(f32)
        "SquareBf16_v1{N=32} x3", "MeanTriton_v1{N=32} x3", "AddScalarF32_v1{N=1,C=1e-06} x3", "RsqrtF32_v1{N=1} x3",
        "ScaleRowBf16_v1{N=32} x3", "MulVecF32_v1{N=32} x3", "NarrowF32ToBf16_v1{N=32} x3",
    ], _emitted(rep)
    rules = [a["rule"] for a in rep.provenance["applications"]]
    for r in ("WidenBf16ToF32[aten]_v1", "AddScalar[aten]_v1", "Square[aten]_v1", "MeanDimBI[aten]_v1", "Rsqrt[aten]_v1", "MulF32[aten]_v1",
              "NarrowF32ToBf16[aten]_v1"):
        assert r in rules, (r, rules)
    narrow = [a for a in rep.provenance["applications"] if a["rule"] == "NarrowF32ToBf16[aten]_v1"]
    assert [_det(a)["member"] for a in narrow] == ["out"]
    mean = next(a for a in rep.provenance["applications"] if a["rule"] == "MeanDimBI[aten]_v1")
    assert _det(mean)["pins"]["src_sha256"] == MEAN_PINS["src_sha256"] and _det(mean)["mean_dim_override_observed"]["kernel"] == MEAN_PINS["kernel"]
    # the (1 + w) weight role is recorded against the parameter
    assert rep.provenance["weights"][0]["name"] == "weight"
    assert rep.provenance["weight_roles"]["weight"]["role"] == "norm.weight_plus_one", rep.provenance["weight_roles"]


class Layer(nn.Module):
    """A parent module (as `model.layers.k`), so the norm's rows carry a module path like the model's."""

    def __init__(self, H: int) -> None:
        super().__init__()
        self.post_attention_layernorm = GemmaNorm(H)

    def forward(self, x, residual):
        return self.post_attention_layernorm(x, residual)


def test_residual_form_binds_the_fused_chain_with_members():
    torch.manual_seed(1)
    m = Layer(48)
    rep = _derive(m, torch.randn(2, 48).to(BF), torch.randn(2, 48).to(BF))
    assert rep.unsupported is None, f"refused: {rep.unsupported}"
    assert _emitted(rep) == [
        "transparent widening (no gates; readers bind *Bf16 kinds)", "AddScalarBf16_v1{N=48,C=1.0} x1",
        "transparent widening (no gates; readers bind *Bf16 kinds)", "transparent widening (no gates; readers bind *Bf16 kinds)",
        "AddWidenedBf16_v1{N=48} x2", "NarrowF32ToBf16_v1{N=48} x2",                     # the new residual (member 1)
        "SquareF32_v1{N=48} x2", "MeanTriton_v1{N=48} x2", "AddScalarF32_v1{N=1,C=1e-06} x2", "RsqrtF32_v1{N=1} x2",
        "ScaleRowF32_v1{N=48} x2", "MulVecF32_v1{N=48} x2", "NarrowF32ToBf16_v1{N=48} x2",  # the normed output (member 0)
    ], _emitted(rep)
    narrow = [a for a in rep.provenance["applications"] if a["rule"] == "NarrowF32ToBf16[aten]_v1"]
    assert [_det(a)["member"] for a in narrow] == ["1", "0"], narrow
    # the Program's row names carry the member hint the required-value manifest reads (`<module>/<node>/<member>[row]`)
    from verity_vllm.query.manifest.format import split_name
    hints = [h for h in rep.program.annotations["node_hints"].values() if "/1[" in h or "/0[" in h]
    assert len(hints) == 4, rep.program.annotations["node_hints"]
    parsed = [split_name(h) for h in hints]
    assert sorted(p[2] for p in parsed) == ["0", "0", "1", "1"], hints
    assert all(p[0] == "post_attention_layernorm" for p in parsed), hints


def test_three_d_rows_are_row_major():
    torch.manual_seed(2)
    rep = _derive(GemmaNorm(16), torch.randn(2, 3, 16).to(BF))
    assert rep.unsupported is None, f"refused: {rep.unsupported}"
    assert "SquareBf16_v1{N=16} x6" in _emitted(rep) and "MeanTriton_v1{N=16} x6" in _emitted(rep)


# ---------------------------------------------------------------------------------------------------------
# fail-closed
# ---------------------------------------------------------------------------------------------------------


def _refused(m: nn.Module, *inputs: torch.Tensor) -> str:
    rep = _derive(m, *inputs)
    assert rep.unsupported is not None, "must refuse"
    return f"{rep.unsupported.reason}: {rep.unsupported.hint}"


def test_mean_refused_without_derived_bi_dispatch():
    facts = _bi_facts()
    facts["evidence"]["bi_triton_kernels_launched"] = []
    observe_export_facts(facts)
    hint = _refused(GemmaNorm(16), torch.randn(2, 16).to(BF))
    assert hint.startswith("numerics_unregistered:") and "MeanDimBI[aten]_v1" in hint and "no batch-invariant Triton launch" in hint, hint


def test_mean_refused_when_kernel_source_differs():
    observe_export_facts(_bi_facts({**{k: MEAN_PINS[k] for k in ("kernel", "mean_dim_src_sha256", "mean_batch_invariant_src_sha256")},
                                    "src_sha256": "0" * 64}))
    hint = _refused(GemmaNorm(16), torch.randn(2, 16).to(BF))
    assert "numerics_unregistered" in hint and "installed src_sha256" in hint, hint


def test_mean_refused_when_facts_missing():
    clear_observations()
    hint = _refused(GemmaNorm(16), torch.randn(2, 16).to(BF))
    assert "MeanDimBI[aten]_v1: batch-invariant dispatch was not observed" in hint, hint


def test_cube_refused():
    hint = _refused(CubeNorm(16), torch.randn(2, 16).to(BF))
    assert "numerics_unregistered" in hint and "exponent 3 is not 2" in hint, hint


def test_genuine_f32_add_refused():
    hint = _refused(F32PlusF32(), torch.randn(2, 16).to(BF), torch.randn(2, 16).to(BF))
    assert "numerics_unregistered" in hint, hint


def test_rsqrt_of_widened_refused():
    hint = _refused(RsqrtOfWidened(), torch.rand(2, 16).to(BF))
    assert "numerics_unregistered" in hint and "rsqrt of a widened bf16 view" in hint, hint


def test_kinds_registered_in_the_pinned_vocabulary_only():
    from verity_vllm.program.frontend.rules.vocab import PROFILES
    from verity_vllm.program.registry import dense as DN
    from verity_vllm.program.registry.quarantine.dense import mean_triton as q
    v3 = PROFILES["b1-eager-v3"]
    assert v3.kinds["mean_triton"](N=2304).id == "MeanTriton_v1{N=2304}"
    assert v3.kinds["attention_softcap"](T=5, NH=8, KVH=4, D=256, CAP=50.0).id == "AttentionSoftcap_v1{T=5,NH=8,KVH=4,D=256,BN=64,CAP=50.0}"
    assert q.MeanTriton is DN.MeanTriton, "quarantine must re-export the promoted Definition (no registry clash)"
    for k in ("mean_triton", "square_bf16", "narrow_f32_bf16", "gelu_tanh_mul", "attention_softcap"):
        assert k not in PROFILES["b1-eager"].kinds and k not in PROFILES["b1-eager-v2"].kinds, "the accepted b1-eager vocabularies are unchanged"
    assert PROFILES["dense2-eager-v1"].kinds.keys() == v3.kinds.keys()


def test_mean_pins_match_installed_vllm():
    """Live pin check when the pinned vLLM is importable: the installed `mean_kernel` / `mean_dim` / `mean_batch_invariant` hash to MEAN_PINS."""
    pytest.importorskip("vllm")
    from verity_vllm.program.frontend.vllm_meta import mean_dim_override_facts
    mo = mean_dim_override_facts()
    assert mo and not mo.get("error"), mo
    for k in ("kernel", "src_sha256", "mean_dim_src_sha256", "mean_batch_invariant_src_sha256"):
        assert mo[k] == MEAN_PINS[k], (k, mo[k], MEAN_PINS[k])
    assert mo["enable_batch_invariant_mode_registers_mean_dim"] is True


# ---------------------------------------------------------------------------------------------------------
# numerics: bit-exact vs the pinned vLLM GemmaRMSNorm under batch invariance (GPU)
# ---------------------------------------------------------------------------------------------------------


def _vllm_gemma_norm(H: int):
    pytest.importorskip("vllm")
    import vllm.envs as envs
    from vllm.model_executor.determinism import batch_invariant as bi
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    if not envs.VLLM_BATCH_INVARIANT:
        pytest.skip("VLLM_BATCH_INVARIANT=1 is needed for the BI mean_kernel dispatch")
    from vllm.config import VllmConfig, set_current_vllm_config
    bi.init_batch_invariance()                      # installs the aten override library (aten::mean.dim -> mean_batch_invariant) as the engine worker does
    assert bi._batch_invariant_MODE, "the override library must be installed for the reference to dispatch to mean_kernel"
    with set_current_vllm_config(VllmConfig()):     # CustomOp dispatch needs a config (eager: forward_native for GemmaRMSNorm on CUDA)
        norm = GemmaRMSNorm(H, eps=EPS)
    assert norm._forward_method.__name__ in ("forward_cuda", "forward_native"), norm._forward_method
    return norm


@pytest.mark.parametrize("H,M,residual", [(2304, 3, False), (2304, 3, True), (256, 5, False), (1280, 2, True)])
def test_cuda_bit_exact_vs_vllm_gemma_rms_norm(H, M, residual):
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed: MeanTriton models the GPU mean_kernel's reduction order")
    torch.manual_seed(H + M)
    w = (torch.randn(H) * 0.2).to(BF)
    x = (torch.randn(M, H) * 2).to(BF)
    r = (torch.randn(M, H) * 2).to(BF)
    m = GemmaNorm(H)
    with torch.no_grad():
        m.weight.copy_(w)
    inputs = [x, r] if residual else [x]
    rep = _derive(m, *inputs)                       # export first: init_batch_invariance() sets TF32 flags torch.export then rejects
    assert rep.unsupported is None, f"refused: {rep.unsupported}"
    ref = _vllm_gemma_norm(H).cuda()
    with torch.no_grad():
        ref.weight.copy_(w.cuda())
    x, r = x.cuda(), r.cuda()
    try:
        with torch.no_grad():
            got_ref = ref(x, r) if residual else ref(x)
    finally:
        # init_batch_invariance() sets the cuDNN conv / RNN fp32 precision through the new API and torch.export then refuses the
        # "mixed API" state; restore the legacy defaults so the next parametrization can export (the norm chain has no conv / RNN).
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
    want = np.array(_bits16(got_ref[0]) + _bits16(got_ref[1]), dtype=np.uint32) if residual else np.array(_bits16(got_ref), dtype=np.uint32)
    got = _evaluate(rep, m, inputs)
    assert got.shape == want.shape, (got.shape, want.shape)
    diff = int((got != want).sum())
    assert diff == 0, f"{diff} of {want.size} words differ from vLLM GemmaRMSNorm (BI) on CUDA"


class Variance(nn.Module):
    """The chain's f32 interior up to the mean (`SquareBf16 -> MeanTriton`), returned as the root: the words Match compares for the
    `MeanTriton` rows of every norm (the eager capture holds the f32 mean, not only the bf16 output)."""

    def forward(self, x):
        return x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)


class LlamaStyleNorm(GemmaNorm):
    """`w * x/rms(x)` (RMSNorm) in place of Gemma's `(1 + w) * x/rms(x)`: the weight-semantics gap of F-dense-05."""

    def forward(self, x):
        x32 = x.to(torch.float32)
        variance = x32.pow(2).mean(dim=-1, keepdim=True)
        return ((x32 * torch.rsqrt(variance + self.variance_epsilon)) * self.weight.float()).to(x.dtype)


def _bits32(t: torch.Tensor) -> np.ndarray:
    return t.detach().contiguous().cpu().view(torch.int32).numpy().astype(np.uint32).reshape(-1)


def test_cuda_match_negative_perturbed_numerics_are_caught():
    """Match negative (numerics, not structure).  The Program derived from the DECLARED implementation (`GemmaRMSNorm.forward_native`
    under BI; `MeanTriton` = the GPU `mean_kernel`'s reduction order) is compared word-for-word on CUDA against two perturbed
    implementations of the same function: (a) the variance reduced in torch's CUDA `sum` order (`x.pow(2).sum(-1) / H`; the same
    f32 arithmetic, a different summation order) -- the f32 `MeanTriton` words differ on a third of the rows (16..23 of 64 measured on
    L40S for H in 2304/1280/256), which is what a Match with a tolerance would wave through and this one does not; (b) Llama-style
    `w` in place of `(1 + w)` on the whole chain -- every bf16 output word differs.  The positive counterpart (0 words differ vs the
    installed vLLM GemmaRMSNorm) is `test_cuda_bit_exact_vs_vllm_gemma_rms_norm`."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    H, M = 2304, 64
    torch.manual_seed(7)
    x = (torch.randn(M, H) * 2).to(BF)
    # (a) the f32 interior: the derived MeanTriton words vs torch's CUDA sum order
    var = Variance()
    rep = _derive(var, x)
    assert rep.unsupported is None, f"refused: {rep.unsupported}"
    assert "MeanTriton_v1{N=2304} x64" in _emitted(rep), _emitted(rep)
    got32 = _evaluate(rep, var, [x])
    x32 = x.cuda().to(torch.float32)
    other = x32.pow(2).sum(dim=-1, keepdim=True) / H
    rows_differ = int((got32 != _bits32(other)).sum())
    assert got32.size == M and rows_differ >= M // 8, f"only {rows_differ} of {M} f32 mean rows differ from the torch sum order -- not caught"
    # (b) the whole chain vs Llama-style weight semantics
    w = (torch.randn(H) * 0.2).to(BF)
    m = GemmaNorm(H)
    with torch.no_grad():
        m.weight.copy_(w)
    rep = _derive(m, x)
    assert rep.unsupported is None, f"refused: {rep.unsupported}"
    got = _evaluate(rep, m, [x])
    alt = LlamaStyleNorm(H).cuda()
    with torch.no_grad():
        alt.weight.copy_(w.cuda())
        other = alt(x.cuda())
    diff = int((got != np.array(_bits16(other), dtype=np.uint32)).sum())
    assert diff >= got.size // 4, f"LlamaStyleNorm: only {diff} of {got.size} words differ -- the perturbation was not caught"
