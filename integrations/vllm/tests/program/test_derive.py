"""Derivation mode (`verity_vllm.program.frontend.derive`): unannotated modules -> Program | Unsupported.

S1 reference mode (MLP; evaluated vs torch CPU f32 and vs the primitives' own sequential semantics), S2 profile mode
(a vLLM-style decoder layer at B0 dims vs the hand-built TA1 layer), S3 (a greedy serving loop vs `Serve_v1`), and the
negatives every stage must refuse. Everything runs on meta tensors except the S1 evaluation; the full-B0 S3 case takes
~10 s / 0.5 GB.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from verity_vllm.program.frontend.derive import Unsupported, derive, derive_report
from verity_vllm.program.frontend import derive_examples as ex
from verity_vllm.program.frontend.rules import DEFAULT_RULES
from verity.ml.circuits.program import Program
from verity.ml.circuits.codec import program_digest
from verity.ml.circuits.refs import Coll
from verity.ml.definitions.registry import b1, prims as P
from verity.ml.definitions.numerics.serve3_reference import materialize

from verity_vllm.program.frontend import examples
from verity_vllm.program.frontend.correspond import correspond_rebound
from verity_vllm.program.frontend.torch_frontend import export as fe_export, function

REF = dict(mode="reference")
PROF = dict(mode="profile", profile="b1-eager")


def _tiny_config(**over):
    C = dict(b1.B0_CONFIG)
    C.update(NL=2, V=64, MAX_POS=64)
    C.update(over)
    return C


def _meta(*shape, dtype=torch.bfloat16):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _unsupported(source, inputs, reason, **kw) -> Unsupported:
    r = derive(source, inputs, **kw)
    assert isinstance(r, Unsupported), f"expected Unsupported({reason}), got a program {r}"
    assert r.reason == reason, f"expected {reason}, got {r.reason}: {r.hint}"
    return r


# ---------------------------------------------------------------------------------------------------------
# S1 — reference mode
# ---------------------------------------------------------------------------------------------------------


@function(static=("K", "N"))
def LinearRow(x, weights, *, K, N):
    """What an author writes for one bf16 Linear row with `examples.py`'s TA1 composites and a weights record (the S1
    structural oracle); shapes come from the example inputs."""
    return examples.BiasAdd(examples.Gemm(x[0], weights["weight"]), weights["bias"]).view(1, N)


def _bits(x: torch.Tensor) -> np.ndarray:
    return x.detach().contiguous().view(torch.int32).numpy().astype(np.uint32).reshape(-1)


def _evaluate(prog: Program, weights: list[dict], example: torch.Tensor, module: torch.nn.Module, x_bits: list[int] | None = None) -> np.ndarray:
    """Evaluate the derived program on the example and the module's tensors (program input order: the example's leaves
    row-major, then the weights record in declared order); returns the output bits in return order."""
    tensors = dict(module.named_parameters())
    tensors.update(dict(module.named_buffers()))
    wbits = np.concatenate([_bits(tensors[w["name"]]) for w in weights]) if weights else np.zeros(0, np.uint32)
    assert example.numel() + wbits.size == prog.input_gates
    t = materialize(prog, [x_bits if x_bits is not None else [int(v) for v in _bits(example)], [int(v) for v in wbits]])
    return np.array([np.uint32(t[g]) for g in prog.circuit.root_outputs()], dtype=np.uint32)


def _sequential_linear_f32(x: np.ndarray, w: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    """The reference kind's stated semantics: per output, a sequential f32 FMA chain from +0 over k, then bias add."""
    fma = P.F32Fma.evaluate
    add = P.F32Add.evaluate
    N, K = w.shape
    out = np.empty((x.shape[0], N), dtype=np.uint32)
    for r in range(x.shape[0]):
        for n in range(N):
            acc = 0
            for k in range(K):
                acc = fma(int(x[r, k]), int(w[n, k]), acc)
            out[r, n] = add(acc, int(b[n])) if b is not None else acc
    return out


def _f32_bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().contiguous().view(torch.int32).numpy().astype(np.uint32)


def test_s1_linear_reference_mode_structure_and_provenance():
    torch.manual_seed(0)
    m = torch.nn.Linear(16, 8)
    rep = derive_report(m, (torch.empty(4, 16),), **REF, name="Linear")
    assert rep.ok, rep.unsupported
    prov = rep.provenance
    assert prov["mode"] == "reference" and prov["vocabulary"]["name"] == "reference-ref-prims" and prov["vocabulary"]["pin"].startswith("verity-ref-vocab/1 digest ")
    assert prov["rules_applied"] == ["Linear[reference]_v2"]
    # weights are prescribed inputs: one record with weight and bias, in module order
    assert [w["name"] for w in prov["weights"]] == ["weight", "bias"]
    assert rep.gates is not None and rep.program.input_gates == 4 * 16 + 8 * 16 + 8
    apps = [a["rule"] for a in prov["applications"]]
    assert apps.count("Linear[reference]_v2") == 1
    # every rule application records its version and the checked preconditions
    for a in prov["applications"]:
        assert a["version"] >= 1 and "preconditions_checked" in a


def test_s1_mlp_reference_mode_evaluates_and_matches_stated_semantics():
    torch.manual_seed(1)
    m = ex.MLP2(16, 32, 8)
    x = torch.randn(4, 16)
    rep = derive_report(m, (torch.empty(4, 16),), **REF, name="MLP2")
    assert rep.ok, rep.unsupported
    prog = rep.program
    got = _evaluate(prog, rep.provenance["weights"], x, m).reshape(4, 8)

    # (a) the derived program computes the reference kinds' stated semantics exactly
    W1, b1_, W2, b2 = (_f32_bits(t) for t in (m.fc1.weight, m.fc1.bias, m.fc2.weight, m.fc2.bias))
    h = _sequential_linear_f32(_f32_bits(x), W1, b1_)
    relu = np.vectorize(lambda v: P.F32Max.evaluate(int(v), 0))
    h = relu(h).astype(np.uint32)
    want = _sequential_linear_f32(h, W2, b2)
    assert np.array_equal(got, want), "derived program != sequential-FMA reference semantics"

    # (b) MEASURED vs torch CPU f32 (BLAS reduction order differs): within a few ULP, never structurally off
    with torch.no_grad():
        ref = m(x)
    ref_f = ref.numpy().reshape(-1)
    got_f = got.view(np.float32).reshape(-1)
    ulp = np.abs(got.reshape(-1).astype(np.int64) - _f32_bits(ref).reshape(-1).astype(np.int64))
    assert np.allclose(got_f, ref_f, rtol=1e-5, atol=1e-6)
    assert ulp.max() <= 64


def test_s1_gemm_structure_corresponds_to_examples_gemm():
    """Profile-mode Linear (bf16) derives to exactly the registered `Gemm_v1{K,N}` + `BiasAdd_v1{N}` kinds — the same
    definitions `examples.py`'s authored Gemm uses, so the derived body equals the authored body node for node."""
    from verity.ml.circuits.defs import bind as ir_bind
    from verity_vllm.program.frontend.correspond import compare_bodies

    m = torch.nn.Linear(16, 8, device="meta", dtype=torch.bfloat16)
    rep = derive_report(m, (_meta(1, 16),), **PROF, name="Linear")
    assert rep.ok, rep.unsupported
    fsd = rep.flat_definition
    ids = [n.fn.id for n in fsd.body.nodes]
    assert ids == ["Gemm_v1{K=16,N=8}", "BiasAdd_v1{N=8}"], ids
    # the derived Gemm callee is *the* registered definition (guard/codec/digest apply); its body is the authored one
    gemm = fsd.body.nodes[0].fn
    v = compare_bodies(gemm.body, ir_bind(b1.Gemm, K=16, N=8).body, fid=gemm.id)
    assert v.verdict == "identical", v
    # ... and the whole derived program corresponds (frontend/correspond.py, identities rebound) to the program an author
    # writes with `examples.py`'s Gemm/BiasAdd: the structural oracle of S1
    authored = fe_export(LinearRow, (_meta(1, 16), {"weight": _meta(8, 16), "bias": _meta(8)}), static={"K": 16, "N": 8})
    rc = correspond_rebound(rep.program, authored)
    assert rc.corresponds and rc.to_json()["root_out"] == "identical", rc.to_json()
    assert not rc.digest_equal  # identities differ (Derived[Linear] vs LinearRow): the claim is correspondence, not the digest


def test_s1_tied_parameters_are_one_input():
    m = ex.TiedMLP(8)
    rep = derive_report(m, (torch.empty(2, 8),), **REF, name="Tied")
    assert rep.ok, rep.unsupported
    ws = rep.provenance["weights"]
    assert len(ws) == 1 and ws[0]["name"] == "fc1.weight" and list(ws[0]["aliases"]) == ["fc2.weight"], ws
    assert rep.program.input_gates == 2 * 8 + 8 * 8
    # both Linear calls read the same Input leaves
    reads = [n for n in rep.flat_definition.body.nodes if n.fn.base_id == "LinearRef_v1"]
    assert len(reads) == 2 * 2 and len({str(n.args[1].refs) for n in reads}) == 1
    assert rep.provenance["mutation_effects"] == []


def test_s1_untied_copy_is_two_inputs():
    """Sharing is preserved exactly as the source has it: two equal-valued but distinct tensors are two inputs."""
    m = ex.MLP2(8, 8, 8, bias=False)
    with torch.no_grad():
        m.fc2.weight.copy_(m.fc1.weight)
    rep = derive_report(m, (torch.empty(2, 8),), **REF, name="Untied")
    assert rep.ok
    assert [w["name"] for w in rep.provenance["weights"]] == ["fc1.weight", "fc2.weight"]
    assert rep.program.input_gates == 2 * 8 + 2 * 64


def test_gpt2_mlp_block_reference_mode_layernorm_conv1d_gelu():
    """Broadening: GPT-2's LayerNorm -> Conv1D (addmm, transposed weight) -> gelu_new -> Conv1D -> residual, in reference
    mode against NUMERIC's LayerNormRef / LinearRef / GeluRef(tanh) / AddRef. Structure asserted; numerics MEASURED vs torch."""
    torch.manual_seed(3)
    m = ex.GPT2MLPBlock(16, 32)
    x = torch.randn(3, 16)
    rep = derive_report(m, (torch.empty(3, 16),), **REF, name="GPT2MLP")
    assert rep.ok, rep.unsupported
    ids = [n.fn.id for n in rep.flat_definition.body.nodes]
    bases = [n.fn.base_id for n in rep.flat_definition.body.nodes]
    assert bases[:3] == ["LayerNormRef_v1"] * 3 and bases.count("LinearRef_v1") == 6, bases
    assert bases.count("GeluRef_v1") == 1 and bases.count("AddRef_v1") == 1
    st = {n.fn.base_id: dict(n.fn.statics) for n in rep.flat_definition.body.nodes}
    assert st["LayerNormRef_v1"] == {"N": 16, "EPS": 1e-5, "DT": "f32"} and st["GeluRef_v1"] == {"N": 96, "DT": "f32", "APPROX": "tanh"}
    assert st["AddRef_v1"] == {"N": 48, "DT": "f32"} and st["LinearRef_v1"]["BIAS"] is True
    assert set(rep.provenance["rules_applied"]) == {"LayerNorm[reference]_v1", "Linear[reference]_v2", "Elementwise[reference]_v3"}
    apps = {a["node"]: a for a in rep.provenance["applications"] if a["rule"] == "Linear[reference]_v2"}
    assert all(a["preconditions_checked"]["W_transposed_view"] for a in apps.values())
    got = _evaluate(rep.program, rep.provenance["weights"], x, m).view(np.float32).reshape(3, 16)
    with torch.no_grad():
        ref = m(x).numpy()
    assert np.allclose(got, ref, rtol=1e-4, atol=1e-5), np.abs(got - ref).max()


def test_gpt2_tiny_lm_reference_mode_end_to_end():
    """A genuinely different architecture (XFER's second family): GPT-2 tiny LM with int64 ids, wte+wpe[arange], causal
    eager attention (matmul / div sqrt(d) / masked_fill(-inf) / softmax f32 / matmul) recognised as a pattern and bound to
    `EagerAttentionRef` per query row, tied lm_head. Evaluated vs torch CPU f32 (MEASURED)."""
    torch.manual_seed(0)
    m = ex.GPT2TinyLM()
    rep = derive_report(m, (torch.empty(4, dtype=torch.int64),), **REF, name="GPT2TinyLM")
    assert rep.ok, rep.unsupported
    assert set(rep.provenance["rules_applied"]) == {"EagerAttention[reference]_v1", "Elementwise[reference]_v3", "Embedding[reference]_v2",
                                                    "LayerNorm[reference]_v1", "Linear[reference]_v2", "StaticGather_v1", "StaticIndex_v2"}
    kinds = Counter(getattr(n.fn, "base_id", getattr(n.fn, "id", None)) for n in rep.flat_definition.body.nodes)
    assert kinds["EagerAttentionRef_v1"] == 2 * 4 and kinds["I64ToI32_v1"] == 4 and kinds["EmbeddingRef_v1"] == 4 and kinds["LayerNormRef_v1"] == 5 * 4
    att = [a for a in rep.provenance["applications"] if a["rule"] == "EagerAttention[reference]_v1"]
    assert len(att) == 2 and all(a["preconditions_checked"]["causal_mask_verified"] and a["preconditions_checked"]["scale_op"] == "div" for a in att)
    ws = rep.provenance["weights"]
    assert ws[0]["name"] == "wte.weight" and ws[0]["aliases"] == ["lm_head.weight"]
    ids = torch.tensor([3, 17, 5, 30])
    got = _evaluate(rep.program, ws, ids, m, x_bits=[int(v) for v in ids.tolist()]).view(np.float32).reshape(4, -1)
    with torch.no_grad():
        ref = m(ids).numpy()
    assert np.allclose(got, ref, rtol=1e-4, atol=1e-5), np.abs(got - ref).max()


def test_gpt2_greedy_request_reference_mode_tokens_match_torch():
    """Bounded GPT-2 request (LP=2 prompt, 2 greedy steps, prefix recomputed): int64 prompt narrowed once, ArgmaxRef tokens fed
    back through a narrowing cat; the selected tokens equal torch's."""
    torch.manual_seed(0)
    g = ex.GPT2Greedy(STEPS=2)
    rep = derive_report(g, (torch.empty(2, dtype=torch.int64),), **REF, name="GPT2Greedy")
    assert rep.ok, rep.unsupported
    kinds = Counter(getattr(n.fn, "base_id", getattr(n.fn, "id", None)) for n in rep.flat_definition.body.nodes)
    assert kinds["ArgmaxRef_v1"] == 3 and kinds["I64ToI32_v1"] == 4 and kinds["EagerAttentionRef_v1"] == 2 * (2 + 3 + 4)
    assert "IntCatNarrow[reference]_v1" in rep.provenance["rules_applied"]
    ids = torch.tensor([3, 17])
    from verity.ml.definitions.numerics.serve3_reference import materialize
    wbits = np.concatenate([_bits(dict(g.named_parameters())[w["name"]]) for w in rep.provenance["weights"]])
    t = materialize(rep.program, [[int(v) for v in ids.tolist()], [int(v) for v in wbits]])
    got = [int(t[o]) for o in rep.program.circuit.root_outputs()]
    with torch.no_grad():
        assert got == g(ids).tolist()


def test_xfer_conv1d_matmul_layout_and_bias_broadcast():
    """XFER-G1: `x @ W[in,out] + b` (matmul with the parameter on the right, bias broadcast over rows)."""

    class XConv1D(torch.nn.Module):
        def __init__(self, nin, nout):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(nin, nout) * 0.05)
            self.bias = torch.nn.Parameter(torch.randn(nout))

        def forward(self, x):
            return x @ self.weight + self.bias

    torch.manual_seed(2)
    m = XConv1D(16, 8)
    x = torch.randn(4, 16)
    rep = derive_report(m, (torch.empty(4, 16),), **REF, name="XConv1D")
    assert rep.ok, rep.unsupported
    assert set(rep.provenance["rules_applied"]) == {"Matmul[reference]_v1", "BroadcastBiasAdd[reference]_v1"}
    got = _evaluate(rep.program, rep.provenance["weights"], x, m).view(np.float32).reshape(4, 8)
    with torch.no_grad():
        assert np.allclose(got, m(x).numpy(), rtol=1e-5, atol=1e-6)


def test_refusal_records_what_fired_before_it():
    """XFER-G8: on `Unsupported`, provenance lists the applications that fired before the refusing node (evidence only)."""
    rep = derive_report(ex.NegUnknownOp(), (torch.empty(2, 8),), **REF)
    assert not rep.ok and rep.unsupported.reason == "numerics_unregistered"
    assert rep.provenance["rules_applied_before_refusal"] == ["Linear[reference]_v2"]
    assert rep.program is None and "no program exists" in rep.provenance["note"]


def test_literal_written_into_a_tensor_is_a_mutation():
    """XFER-G6: `x[0] = 0.0` lifts a scalar constant AND mutates; the mutation is the reason, not the constant."""

    class M(torch.nn.Module):
        def forward(self, x):
            y = x * x
            y[0] = 0.0
            return y

    r = derive(M(), (torch.empty(2, 8),), **REF)
    assert isinstance(r, Unsupported) and r.reason == "mutation", r


# ---------------------------------------------------------------------------------------------------------
# S2 — profile mode: one decoder layer at B0 dims
# ---------------------------------------------------------------------------------------------------------


def _layer_oracle_digest(C, L0: bool, T: int, T0: int = 0) -> str:
    from verity.ml.circuits.defs import bind as ir_bind
    from verity_vllm.program.frontend.rules.family import DecoderLayerT
    return program_digest(Program(ir_bind(DecoderLayerT, C=C, L0=L0, T=T, T0=T0)))


@pytest.mark.parametrize("first,T,pos0", [(True, 4, 0), (False, 3, 5)])
def test_s2_decoder_layer_b0_matches_hand_built_layer(first, T, pos0):
    C = b1.B0_CONFIG
    with torch.device("meta"):
        m = ex.SingleLayer(C, layer_idx=0 if first else 1, pos0=pos0, first=first)
    inputs = (_meta(T, C["H"]),) if first else (_meta(T, C["H"]), _meta(T, C["H"]))
    rep = derive_report(m, inputs, **PROF, config=C, name="Layer")
    assert rep.ok, rep.unsupported
    assert rep.family and rep.family["matched"], rep.provenance.get("families_not_matched")
    assert rep.family["family"].startswith("DeriveOracleDecoderLayer_v1{")
    assert rep.digest == _layer_oracle_digest(C, L0=first, T=T, T0=pos0)
    roles = rep.family["weights_binding"]
    assert roles["qkv_w"].startswith("layer.qkv_proj.weight")
    assert roles["cos_sin"].startswith("layer.rotary_emb.cos_sin_cache")
    # the flat program (same circuit, no hierarchy) has a different identity, and it is recorded
    assert rep.flat_digest != rep.digest
    assert rep.provenance["rules_applied"]  # versions recorded per application
    kinds = {a["rule"] for a in rep.provenance["applications"]}
    assert {"RmsNorm[vllm]_v1" if first else "FusedAddRmsNorm[vllm]_v1", "RotaryEmbedding[vllm]_v1", "UnifiedAttention[vllm]_v1",
            "SiluAndMul[vllm]_v1", "Linear[profile]_v1"} <= kinds


def test_s2_hf_eager_layer_is_a_different_circuit():
    """HF's eager RMSNorm decomposes into pow/mean/rsqrt; the profile binds fused norm kernels only. Reported honestly as
    `decomposition` at the first decomposed node — never relabelled to the fused program."""
    C = b1.B0_CONFIG
    with torch.device("meta"):
        m = ex.HFDecoderLayer(C)
    r = derive(m, (_meta(4, C["H"]),), **PROF, config=C, name="HFLayer")
    assert isinstance(r, Unsupported), r
    # the first HF node the profile cannot bind is the norm's f32 upcast (no cast kind in the profile) — either reason is honest
    assert r.reason in ("decomposition", "numerics_unregistered"), r
    assert r.module_path and "norm" in r.module_path, r


def test_s2_allowed_dims_refuses_other_shapes():
    C = b1.B0_CONFIG
    with torch.device("meta"):
        m = ex.SingleLayer(C, layer_idx=0)
    r = _unsupported(m, (_meta(5, C["H"]),), "shape", **PROF, config=C, name="Layer", allowed_dims={"T": {4}})
    assert "T" in r.hint


# ---------------------------------------------------------------------------------------------------------
# S3 — a bounded request: prefill + decode with explicit KV dependencies and TokenSelect
# ---------------------------------------------------------------------------------------------------------


def _serve_oracle_digest(C, LP, STEPS) -> str:
    from verity.ml.circuits.defs import bind as ir_bind
    return program_digest(Program(ir_bind(b1.Serve, C=C, LP=LP, STEPS=STEPS)))


def test_s3_tiny_greedy_serve_matches_serve_v1():
    C = _tiny_config()
    LP, STEPS = 3, 2
    with torch.device("meta"):
        m = ex.GreedyServe(C, STEPS)
    rep = derive_report(m, (_meta(LP, dtype=torch.int32),), **PROF, config=C, name="Serve")
    assert rep.ok, rep.unsupported
    assert rep.family and rep.family["matched"], rep.provenance.get("families_not_matched")
    assert rep.family["family"].startswith("Serve_v1{") and f"LP={LP},STEPS={STEPS}" in rep.family["family"]
    assert rep.digest == _serve_oracle_digest(C, LP, STEPS)
    wb = rep.family["weights_binding"]
    assert wb["embed"] == "lm.model.embed_tokens.weight"
    assert rep.family["module_weights_by_role"]["embed"] == "lm.model.embed_tokens.weight (tied: lm.lm_head.weight)"
    assert wb["cos_sin"].startswith("lm.model.layers.0.rotary_emb.cos_sin_cache")
    # KV dependencies are explicit: every attention call's K/V rows are produced by this request's earlier nodes
    fsd = rep.flat_definition
    attn = [n for n in fsd.body.nodes if n.fn.base_id == "Attention_v1"]
    assert len(attn) == C["NL"] * (LP + STEPS)
    assert all(a["preconditions_checked"]["kv_rows_from_this_request"] for a in rep.provenance["applications"]
               if a["rule"] == "UnifiedAttention[vllm]_v1")
    # TokenSelect: one per generated token (STEPS + the prefill's), and the sampled token feeds the next embedding
    assert sum(1 for n in fsd.body.nodes if n.fn.base_id == "TokenSelect_v1") == STEPS + 1
    assert rep.provenance["mutation_effects"] == []


def test_s3_final_norm_on_all_rows_is_a_different_program():
    C = _tiny_config()
    with torch.device("meta"):
        m = ex.GreedyServe(C, 2, final_norm_all_rows=True)
    rep = derive_report(m, (_meta(3, dtype=torch.int32),), **PROF, config=C, name="ServeAllRows")
    assert rep.ok, rep.unsupported  # a complete program of its own ...
    assert not rep.family  # ... that is not Serve_v1 (extra dead norm + LM-head calls) and is not claimed to be
    nm = rep.provenance["families_not_matched"]
    assert nm and "leaf counts differ" in nm[0]["reason"]
    assert rep.root.startswith("Derived[ServeAllRows]_v1{")


def test_s3_untied_lm_head_loses_sharing_and_is_refused_by_family():
    C = _tiny_config(tie_word_embeddings=False)
    with torch.device("meta"):
        m = ex.GreedyServe(C, 2)
    rep = derive_report(m, (_meta(3, dtype=torch.int32),), **PROF, config=C, name="ServeUntied")
    assert rep.ok, rep.unsupported
    assert not rep.family  # Serve_v1 reads `embed` for both the embedding and the LM head: two tensors cannot bind one role
    tied = _serve_inputs(dict(C, tie_word_embeddings=True), 3)
    assert rep.program.input_gates == _serve_inputs(C, 3) == tied + C["V"] * C["H"]   # the lm_head is a [V][H] tensor of its own


def _serve_inputs(C, LP):
    return LP + b1.weights_type(C).leaves


@pytest.mark.slow
def test_s3_full_b0_matches_certified_digest():
    """The certificate of record (INT-PAD-2 B0) pins `Serve_v1{C=B0,LP=32,STEPS=15}`; the derived program is that program."""
    C = b1.B0_CONFIG
    LP, STEPS = 32, 15
    with torch.device("meta"):
        m = ex.GreedyServe(C, STEPS)
    rep = derive_report(m, (_meta(LP, dtype=torch.int32),), **PROF, config=C, name="Serve")
    assert rep.ok, rep.unsupported
    assert rep.family and rep.family["matched"]
    assert rep.digest == "7d09b7a652a8364525bf82a34bb7e261efc033d7b53ceda20d5e5bf2a774bbbdd402d04cd3bd7868946582956090359a851d01cf6415568b830647b8b2f56df7"   # the re-baseline re-key (AmpereBF16TcDot16_v2); d4f4b1f1 before
    assert rep.digest == _serve_oracle_digest(C, LP, STEPS)


# ---------------------------------------------------------------------------------------------------------
# negatives — every stage must show these REFUSED
# ---------------------------------------------------------------------------------------------------------


def test_neg_unknown_op_has_no_kind():
    r = _unsupported(ex.NegUnknownOp(), (torch.empty(2, 8),), "numerics_unregistered", **REF)
    assert "tanh" in r.hint


def test_neg_custom_op_without_binding():
    r = _unsupported(ex.NegCustomOp(), (torch.empty(2, 8),), "custom_op_without_binding", **REF)
    assert "mystery" in r.hint


def test_neg_hidden_constant():
    r = _unsupported(ex.NegHiddenConstant(), (torch.empty(2, 8),), "hidden_state", **REF)
    assert "lifted" in r.hint


def test_neg_in_place_without_rule():
    r = _unsupported(ex.NegInPlace(), (torch.empty(2, 8),), "mutation", **REF)
    assert "add_" in r.hint


def test_neg_data_dependent_gather():
    with torch.device("meta"):
        m = ex.NegDataDependent()
    r = _unsupported(m, (_meta(2, 8),), "dynamic_control_flow", **PROF)
    assert "data-dependent" in r.hint


def test_neg_data_dependent_cond():
    r = _unsupported(ex.NegCond(), (torch.empty(2, 8),), "dynamic_control_flow", **REF)
    assert "cond" in r.hint


def test_neg_forward_writes_a_buffer():
    with torch.device("meta"):
        m = ex.NegMutatesWeight()
    _unsupported(m, (_meta(2, 8),), "mutation", **PROF)


def test_neg_stale_alias_read_after_in_place_rotation():
    C = _tiny_config()
    with torch.device("meta"):
        m = ex.NegStaleAlias(C)
    r = _unsupported(m, (_meta(3, C["H"]),), "mutation", **PROF, config=C)
    assert "stale" in r.hint or "overlapping alias" in r.hint


def test_neg_wrong_shape_is_an_export_error_not_a_program():
    m = torch.nn.Linear(16, 8)
    r = derive(m, (torch.empty(4, 15),), **REF)
    assert isinstance(r, Unsupported) and r.reason == "export"


def test_neg_profile_mode_refuses_f32_linear():
    """The profile's Gemm is bf16; an f32 Linear is not that kind (dtype is checked structurally, not the op name)."""
    m = torch.nn.Linear(16, 8)
    r = derive(m, (torch.empty(4, 16),), **PROF)
    assert isinstance(r, Unsupported) and r.reason in ("numerics_unregistered", "decomposition"), r


def test_neg_profile_mode_refuses_a_batch_invariant_linear_on_a_target_without_batch_invariance(monkeypatch):
    """Without batch invariance, unquantized `aten.linear` is cuBLAS and cannot bind the profile's Triton GEMM."""
    from verity_vllm.program.frontend.rules.vllm_bindings import observations as VB
    from verity_vllm.program.frontend.target_profile import TargetProfile
    m = torch.nn.Linear(16, 8, bias=False).to(torch.bfloat16)
    x = (torch.empty(4, 16, dtype=torch.bfloat16),)
    assert derive_report(m, x, **PROF).ok
    monkeypatch.setitem(VB._OBSERVED, "target_profile", TargetProfile(compute_capability=(9, 0), num_sms=132, batch_invariant=False))
    r = derive(m, x, **PROF)
    assert isinstance(r, Unsupported) and "aten.linear on a target without batch invariance" in r.hint, r


@pytest.mark.parametrize(
    "fused,expected",
    [
        (False, "RMSNormCudaVector8_v1{N=6144,EPS=1e-05,BLOCK=768}"),
        (True, "RMSNormFusedCudaVector8_v1{N=6144,EPS=1e-05,BLOCK=1024}"),
    ],
)
def test_profile_mode_binds_the_non_batch_invariant_cuda_rmsnorm(fused, expected, monkeypatch):
    """The rule asks its selector for the served M=4 block and binds one CUDA vector-8 call per token."""
    from verity_vllm.program.frontend.rules.vllm_bindings import vllm_ruleset
    from verity_vllm.program.frontend.rules.vllm_bindings import observations as VB
    from verity_vllm.program.frontend.target_profile import TargetProfile
    monkeypatch.setitem(VB._OBSERVED, "target_profile", TargetProfile(compute_capability=(9, 0), num_sms=132, batch_invariant=False))
    m = ex.RMSNorm(6144, 1e-5)
    x = torch.empty(4, 6144, dtype=torch.bfloat16)
    rep = derive_report(m, (x, x.clone()) if fused else (x,), rules=vllm_ruleset(), **PROF)
    assert rep.ok, rep.unsupported
    calls = [node.fn.id for node in rep.program.fn.body.nodes]
    assert calls == [expected] * 4
    rule = "FusedAddRmsNorm[vllm-cuda]_v1" if fused else "RmsNorm[vllm-cuda]_v1"
    assert [a["emitted"] for a in rep.provenance["applications"] if a["rule"] == rule] == [f"{expected} x4"]


def test_returned_parameter_is_recorded_explicitly():
    rep = derive_report(ex.NegReturnsWeight(), (torch.empty(2, 8),), **REF)
    assert rep.ok, rep.unsupported
    ri = rep.provenance["returned_inputs"]
    assert len(ri) == 1 and ri[0]["output"] == 1 and ri[0]["weights"]


def test_unsupported_is_never_a_partial_program():
    """Unsupported carries no program; nothing is registered by a failed derivation."""
    from verity.ml.circuits.defs import REGISTRY
    before = set(REGISTRY.defs)
    r = derive(ex.NegUnknownOp(), (torch.empty(2, 8),), **REF)
    assert isinstance(r, Unsupported)
    assert set(REGISTRY.defs) == before


def test_rule_set_is_versioned_and_recorded():
    infos = DEFAULT_RULES.to_json()
    assert infos["rules"] and all(i["version"] >= 1 and i["rule"].endswith(f"_v{i['version']}") for i in infos["rules"])
    assert all(i["preconditions"] for i in infos["rules"])  # every rule states what it checks
    rep = derive_report(ex.MLP2(4, 4, 4), (torch.empty(1, 4),), **REF)
    assert rep.provenance["ruleset"] == infos
    assert rep.program.annotations["derive"]["schema"] == "verity_ir/derive-provenance/v1"


# ---------------------------------------------------------------------------------------------------------
# PAD — the padded circuit (PAD-INT stage-1 requirements; serve3-spec) — after freeze
# ---------------------------------------------------------------------------------------------------------

PROF_V2 = dict(mode="profile", profile="b1-eager-v2")


def _serve_v3_oracle_digest(C3, LP, D_MAX) -> str:
    from verity.ml.definitions.registry import serve3
    return program_digest(Program(serve3.serve_v3(C3, LP, D_MAX)))


def test_pad_tiny_padded_serve_matches_serve_v3():
    """An ordinary padded serving module (guard from EOS detection, muxed slots, muxed cache rows against one zero per
    step, fixed D_MAX steps) derives to `Serve_v3{C3,LP,D_MAX}`: digest equality with the hand-built `serve3.serve_v3`
    after verified isomorphism at leaf granularity (primitive gates and `batch` nodes included)."""
    C3 = _tiny_config(EOS_ID=0)
    LP, D_MAX = 3, 2
    with torch.device("meta"):
        m = ex.PaddedGreedyServe(C3, D_MAX, eos_id=0)
    rep = derive_report(m, (_meta(LP, dtype=torch.int32),), **PROF_V2, config=C3, name="PaddedServe")
    assert rep.ok, rep.unsupported
    assert rep.family and rep.family["matched"], rep.provenance.get("families_not_matched")
    assert rep.family["family"].startswith("Serve_v3{") and f"LP={LP},D_MAX={D_MAX}" in rep.family["family"]
    assert rep.digest == _serve_v3_oracle_digest(C3, LP, D_MAX)
    rules = set(rep.provenance["rules_applied"])
    assert {"EqScalar[pad]_v1", "BitNot[pad]_v1", "BitAnd[pad]_v1", "Select[pad]_v1", "ConstScalar[pad]_v1", "CacheRowWrite[vllm]_v1"} <= rules
    fsd = rep.flat_definition
    ids = [n.fn.id for n in fsd.body.nodes]
    # D_MAX+1 guards: one stop test and one token mux each; live_0 is ONE constant gate; one zero gate per decode step
    assert ids.count("I32Eq_v1") == D_MAX + 1 and ids.count("SelectI32_v1") == D_MAX + 1 and ids.count("BitAnd_v1") == D_MAX + 1
    assert ids.count("Const1[0x1]_v1") == 1
    assert ids.count("Const16[0x0000]_v1") == D_MAX
    # every decode step muxes every layer's appended K and V row (batch nodes of KVH*D selects)
    muxes = [n for n in fsd.body.nodes if n.form == "batch" and n.fn.id == "SelectBf16_v1"]
    assert len(muxes) == 2 * C3["NL"] * D_MAX and all(n.n == C3["KVH"] * C3["D"] for n in muxes)
    # the rewritten rows are what later attention calls read: the state rule recorded one rewrite per layer per step
    rewrites = [a for a in rep.provenance["applications"] if a["rule"] == "CacheRowWrite[vllm]_v1"]
    assert len(rewrites) == C3["NL"] * D_MAX and all(a["preconditions_checked"]["slot"] == a["preconditions_checked"]["history_rows"] - 1 for a in rewrites)
    # output: (D_MAX+1 token slots i32, D_MAX+1 live bits); the live vector is on the return type, designation is PAD-INT's manifest
    from verity.ml.circuits.types import Array as Arr_, Tuple as Tup_
    _, ret = fsd.signature()
    assert isinstance(ret, Tup_) and ret.fields[0] == Arr_(D_MAX + 1, P.I32) and ret.fields[1] == Arr_(D_MAX + 1, P.BIT)


def test_pad_guard_inside_the_step_is_not_serve_v3():
    """A mux on the step's logits (inside the body) derives honestly as its own program and is NOT Serve_v3 (spec §6)."""
    C3 = _tiny_config(EOS_ID=0)
    with torch.device("meta"):
        m = ex.PaddedGreedyServe(C3, 2, eos_id=0, guard_inside=True)
    rep = derive_report(m, (_meta(3, dtype=torch.int32),), **PROF_V2, config=C3, name="PaddedServeGuardInside")
    assert rep.ok and rep.family is None
    nm = rep.provenance["families_not_matched"]
    assert any(f["family"].startswith("Serve_v3{") and "leaf counts differ" in f["reason"] for f in nm)


def test_pad_eos_of_the_guards_must_be_the_configured_eos():
    """The binder refuses to claim `Serve_v3{EOS_ID=5}` for guards that test 0: identity is the source's, not the config's."""
    C3 = _tiny_config(EOS_ID=5)
    with torch.device("meta"):
        m = ex.PaddedGreedyServe(C3, 2, eos_id=0)
    rep = derive_report(m, (_meta(3, dtype=torch.int32),), **PROF_V2, config=C3, name="PaddedServeEosMismatch")
    assert rep.ok and rep.family is None
    assert any(f["rule"] == "Family[Serve_v3]_v1" for f in rep.provenance["families_not_applied"])


def test_pad_kinds_are_not_in_the_v1_vocabulary():
    """The pad-serve kinds belong to the v2-skeleton profile; the v1 vocabulary reports numerics_unregistered, never a guess."""
    C3 = _tiny_config(EOS_ID=0)
    with torch.device("meta"):
        m = ex.PaddedGreedyServe(C3, 1, eos_id=0)
    _unsupported(m, (_meta(3, dtype=torch.int32),), "numerics_unregistered", **PROF, config=C3)


def test_pad_cache_write_of_an_older_row_is_a_mutation():
    """Rewriting any row but the one this step appended is refused: the history has one writer per row per step."""
    class BadWrite(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = ex.Attention("l0")

        def forward(self, q, k, v):
            o = self.attn(q, k, v)
            live = torch.ones(1, dtype=torch.bool) & (torch.argmax(q[0].view(-1), dim=-1).view(1) == 1)
            zero = torch.zeros((), dtype=torch.bfloat16)
            torch.ops.vllm_iface.unified_cache_write(torch.where(live, k[:1], zero), torch.where(live, v[:1], zero), "l0", 0)
            return o

    D = 64
    r = derive(BadWrite(), (_meta(2, 9 * D), _meta(2, 3 * D), _meta(2, 3 * D)), **PROF_V2, config={"D": D})
    assert isinstance(r, Unsupported) and r.reason == "mutation", r


# ---------------------------------------------------------------------------------------------------------
# HF Llama-style layer in reference mode (XFER-G7 / support-matrix row "HF Llama layer, reference"): the decomposed
# RMSNorm is bound as a whole to RMSNormRef (which IS HF's decomposition), SDPA to EagerAttentionRef, repeat_kv is a view
# ---------------------------------------------------------------------------------------------------------

_HF_C = dict(H=16, NH=2, KVH=1, D=8, I=32, EPS=1e-5, BIAS=True)


def _bf16_bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().contiguous().view(torch.int16).numpy().astype(np.uint16).astype(np.uint32).reshape(-1)


def test_hf_llama_layer_reference_mode_f32_tracks_torch_cpu():
    """HF-style layer (three Linears, HF RMSNorm, repeat_kv + SDPA) derives in reference mode as f32; the reference kinds
    (sequential f32 sums) agree with torch CPU to a few f32 ulps (MEASURED, not exact: torch's sums are blocked)."""
    torch.manual_seed(0)
    m = ex.HFDecoderLayer(_HF_C, torch.float32)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
        m.input_layernorm.weight.fill_(1.0)
        m.post_attention_layernorm.weight.fill_(1.0)
    x = torch.randn(3, 16) * 0.5
    rep = derive_report(m, (torch.empty(3, 16),), mode="reference", name="HFLayerF32")
    assert rep.ok, rep.unsupported
    assert {"HFRMSNorm[reference]_v1", "Sdpa[reference]_v2", "RepeatInterleave_v1", "Linear[reference]_v2"} <= set(rep.provenance["rules_applied"])
    norm = [a for a in rep.provenance["applications"] if a["rule"] == "HFRMSNorm[reference]_v1"]
    assert len(norm) == 2 and all(a["preconditions_checked"]["EPS"] == 1e-5 and a["preconditions_checked"]["N"] == 16 for a in norm)
    out = _evaluate(rep.program, rep.provenance["weights"], x, m)
    got = torch.from_numpy(out.astype(np.int32)).view(torch.float32).reshape(3, 16)
    with torch.no_grad():
        want = m(x)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5), (got - want).abs().max()


def test_hf_llama_layer_reference_mode_bf16_within_a_bf16_ulp_of_torch():
    torch.manual_seed(0)
    m = ex.HFDecoderLayer(_HF_C, torch.bfloat16)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_((torch.randn_like(p, dtype=torch.float32) * 0.3).to(torch.bfloat16))
        m.input_layernorm.weight.fill_(1.0)
        m.post_attention_layernorm.weight.fill_(1.0)
    x = (torch.randn(3, 16) * 0.5).to(torch.bfloat16)
    rep = derive_report(m, (torch.empty(3, 16, dtype=torch.bfloat16),), mode="reference", name="HFLayerBf16")
    assert rep.ok, rep.unsupported
    tensors = dict(m.named_parameters())
    wbits = np.concatenate([_bf16_bits(tensors[w["name"]]) for w in rep.provenance["weights"]])
    tab = materialize(rep.program, [[int(v) for v in _bf16_bits(x)], [int(v) for v in wbits]])
    out = np.array([np.uint32(tab[g]) for g in rep.program.circuit.root_outputs()], dtype=np.uint32)
    got = torch.from_numpy(out.astype(np.uint16).view(np.int16)).view(torch.bfloat16).reshape(3, 16).float()
    with torch.no_grad():
        want = m(x).float()
    assert torch.allclose(got, want, rtol=1 / 64, atol=1 / 32), (got - want).abs().max()
    sd = [a for a in rep.provenance["applications"] if a["rule"] == "Sdpa[reference]_v2"]
    assert len(sd) == 1 and sd[0]["preconditions_checked"]["NH"] == 2 and sd[0]["preconditions_checked"]["T"] == 3


def test_hf_rmsnorm_pattern_refuses_a_deviating_chain():
    """The pattern is the whole HF chain; a normalised value read from outside it or a different reduction axis is not it."""
    class Leaky(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.n = ex.HFRMSNorm(8, 1e-5)

        def forward(self, x):
            h = x.to(torch.float32)
            y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)
            return self.n.weight * y.to(x.dtype), y

    class Dim0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.n = ex.HFRMSNorm(8, 1e-5)

        def forward(self, x):
            h = x.to(torch.float32)
            return self.n.weight * (h * torch.rsqrt(h.pow(2).mean(0, keepdim=True) + 1e-5)).to(x.dtype)

    for cls in (Leaky, Dim0):
        r = derive(cls(), (torch.empty(2, 8, dtype=torch.bfloat16),), mode="reference")
        assert isinstance(r, Unsupported) and r.reason == "decomposition" and "HFRMSNorm" in r.hint, r


def test_sdpa_with_an_explicit_mask_is_not_bound():
    """An explicit mask binds only when it is a statically known bool tensor equal to the causal pattern (G-RH-8, `Sdpa[reference]_v2`;
    torch: `is_causal=True` IS `attn_mask=tril(ones)`); a non-causal static mask and a mask that is a value are refused."""
    class Masked(torch.nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=torch.ones(3, 3, dtype=torch.bool).tril())

    class NonCausal(torch.nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=torch.ones(3, 3, dtype=torch.bool).triu())

    class ValueMask(torch.nn.Module):
        def forward(self, q, k, v, m):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=m)

    e = (torch.empty(2, 3, 8, dtype=torch.bfloat16),) * 3
    rep = derive_report(Masked(), e, mode="reference")
    assert rep.ok, rep.unsupported
    sd = [a for a in rep.provenance["applications"] if a["rule"] == "Sdpa[reference]_v2"]
    assert len(sd) == 1 and sd[0]["preconditions_checked"]["causal"].startswith("static bool attn_mask verified")
    r = derive(NonCausal(), e, mode="reference")
    assert isinstance(r, Unsupported) and r.reason == "decomposition" and "not the causal pattern" in r.hint, r
    r = derive(ValueMask(), e + (torch.empty(3, 3, dtype=torch.bool),), mode="reference")
    assert isinstance(r, Unsupported) and r.reason == "decomposition" and "attn_mask that is a value" in r.hint, r


# ---------------------------------------------------------------------------------------------------------
# HF rotary embedding in reference mode: the cos/sin table is a DERIVED INPUT (declared by the rule, derivation recorded),
# `apply_rotary_pos_emb` is RoPEHeadsRef per (batch, position)
# ---------------------------------------------------------------------------------------------------------


class _RopeOnly(torch.nn.Module):
    def __init__(self, D: int) -> None:
        super().__init__()
        self.rotary_emb = ex.HFRotaryEmbedding(D)

    def forward(self, q):  # [1, NH, T, D]
        cos, sin = self.rotary_emb(q, torch.arange(q.shape[2]).unsqueeze(0))
        qe, _ = ex.hf_apply_rotary_pos_emb(q, q, cos, sin)
        return qe


def _hf_tables(m: torch.nn.Module, x: torch.Tensor, T: int) -> dict[str, torch.Tensor]:
    """The derived tables as the source computes them (torch f32 cos/sin, cast to x's dtype) — the values a verifier supplies."""
    with torch.no_grad():
        cos, sin = m.rotary_emb(x, torch.arange(T).unsqueeze(0))
    return {"rotary_emb.cos_table": cos[0], "rotary_emb.sin_table": sin[0]}


def _bits_any(t: torch.Tensor) -> np.ndarray:
    return _bf16_bits(t) if t.dtype == torch.bfloat16 else _bits(t)


@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float32])
def test_hf_rope_is_bit_exact_vs_torch_cpu_and_the_table_is_a_derived_input(dt):
    torch.manual_seed(1)
    m = _RopeOnly(8)
    q = torch.randn(1, 2, 5, 8).to(dt)
    rep = derive_report(m, (torch.empty(1, 2, 5, 8, dtype=dt),), mode="reference", name="RopeOnly")
    assert rep.ok, rep.unsupported
    assert set(rep.provenance["rules_applied"]) == {"HFRoPE[reference]_v1", "HFRotaryTable[reference]_v4", "StaticIndex_v2"}
    ws = {w["name"]: w for w in rep.provenance["weights"]}
    assert ws["rotary_emb.inv_freq"]["kind"] == "buffer"
    for nm in ("rotary_emb.cos_table", "rotary_emb.sin_table"):
        assert ws[nm]["kind"] == "derived_table" and ws[nm]["shape"] == [5, 8]
        assert ws[nm]["derivation"]["source_buffer"] == "rotary_emb.inv_freq" and ws[nm]["derivation"]["rule"] == "HFRotaryTable[reference]_v4"
    tbl = [a for a in rep.provenance["applications"] if a["rule"] == "HFRotaryTable[reference]_v4"]
    assert len(tbl) == 1 and tbl[0]["preconditions_checked"]["positions"] == [0, 1, 2, 3, 4]
    tensors = dict(m.named_buffers())
    tensors.update(_hf_tables(m, q, 5))
    wbits = np.concatenate([_bits_any(tensors[w["name"]]) for w in rep.provenance["weights"]])
    tab = materialize(rep.program, [[int(v) for v in _bits_any(q)], [int(v) for v in wbits]])
    out = np.array([np.uint32(tab[g]) for g in rep.program.circuit.root_outputs()], dtype=np.uint32)
    with torch.no_grad():
        want = m(q)
    assert (out == _bits_any(want)).all()  # HF's three-rounding decomposition == torch CPU, bit for bit


def test_hf_llama_layer_with_rope_reference_mode_tracks_torch_cpu():
    torch.manual_seed(0)
    m = ex.HFDecoderLayer(_HF_C, torch.float32, rope=True)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
        m.input_layernorm.weight.fill_(1.0)
        m.post_attention_layernorm.weight.fill_(1.0)
    x = torch.randn(3, 16) * 0.5
    rep = derive_report(m, (torch.empty(3, 16),), mode="reference", name="HFLayerRoPE")
    assert rep.ok, rep.unsupported
    assert {"HFRMSNorm[reference]_v1", "HFRoPE[reference]_v1", "HFRotaryTable[reference]_v4", "Sdpa[reference]_v2", "RepeatInterleave_v1",
            "Linear[reference]_v2"} <= set(rep.provenance["rules_applied"])
    tensors = dict(m.named_parameters())
    tensors.update(dict(m.named_buffers()))
    tensors.update(_hf_tables(m, x, 3))
    wbits = np.concatenate([_bits(tensors[w["name"]]) for w in rep.provenance["weights"]])
    assert x.numel() + wbits.size == rep.program.input_gates
    tab = materialize(rep.program, [[int(v) for v in _bits(x)], [int(v) for v in wbits]])
    out = np.array([np.uint32(tab[g]) for g in rep.program.circuit.root_outputs()], dtype=np.uint32)
    got = torch.from_numpy(out.astype(np.int32)).view(torch.float32).reshape(3, 16)
    with torch.no_grad():
        want = m(x)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5), (got - want).abs().max()


def test_hf_rope_deviations_are_refused():
    """Positions that are a value (not static structure) and a rotate_half with the halves swapped are not the pattern."""
    class PosInput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_emb = ex.HFRotaryEmbedding(8)

        def forward(self, q, pos):
            cos, sin = self.rotary_emb(q, pos)
            return ex.hf_apply_rotary_pos_emb(q, q, cos, sin)[0]

    r = derive(PosInput(), (torch.empty(1, 2, 5, 8, dtype=torch.bfloat16), torch.empty(1, 5, dtype=torch.int64)), mode="reference")
    assert isinstance(r, Unsupported) and r.reason in ("numerics_unregistered", "decomposition"), r
    assert "HFRotaryTable" in r.hint or "to.dtype" in str(r), r

    class Swapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_emb = ex.HFRotaryEmbedding(8)

        def forward(self, q):
            cos, sin = self.rotary_emb(q, torch.arange(q.shape[2]).unsqueeze(0))
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
            x1, x2 = q[..., :4], q[..., 4:]
            return q * cos + torch.cat((x1, -x2), dim=-1) * sin  # not rotate_half

    r = derive(Swapped(), (torch.empty(1, 2, 5, 8, dtype=torch.bfloat16),), mode="reference")
    assert isinstance(r, Unsupported) and "HFRoPE" in r.hint, r


def test_hf_tiny_lm_reference_mode_greedy_token_matches_torch_and_shares_one_rope_table():
    """HF `LlamaForCausalLM` shape: int64 ids -> tied embedding -> ONE model-level rotary table feeding every layer -> NL layers ->
    final norm -> tied lm_head -> argmax. The greedy token equals torch CPU's on several prompts (MEASURED; argmax over reference
    logits within ulps of torch's). Sharing preserved: one weights field for embed/lm_head, one cos/sin table pair for both layers."""
    torch.manual_seed(0)
    C = dict(H=16, NH=2, KVH=1, D=8, I=32, EPS=1e-5, BIAS=False, V=32, NL=2)
    m = ex.HFTinyLM(C, torch.bfloat16)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_((torch.randn_like(p, dtype=torch.float32) * 0.3).to(torch.bfloat16))
    rep = derive_report(m, (torch.empty(4, dtype=torch.int64),), mode="reference", name="HFTinyLM")
    assert rep.ok, rep.unsupported
    ws = {w["name"]: w for w in rep.provenance["weights"]}
    assert ws["embed_tokens.weight"]["aliases"] == ["lm_head.weight"] and "lm_head.weight" not in ws
    assert [w for w in ws if ws[w]["kind"] == "derived_table"] == ["rotary_emb.cos_table", "rotary_emb.sin_table"]
    apps = Counter(a["rule"] for a in rep.provenance["applications"])
    assert apps["HFRotaryTable[reference]_v4"] == 1 and apps["HFRoPE[reference]_v1"] == 2 * C["NL"] and apps["HFRMSNorm[reference]_v1"] == 2 * C["NL"] + 1
    tensors = dict(m.named_parameters())
    tensors.update(dict(m.named_buffers()))
    for ids in (torch.tensor([3, 7, 11, 2]), torch.tensor([0, 31, 5, 5]), torch.tensor([9, 9, 9, 9])):
        with torch.no_grad():
            cos, sin = m.rotary_emb(m.embed_tokens(ids), torch.arange(4).unsqueeze(0))
            want = m(ids)
        tensors["rotary_emb.cos_table"], tensors["rotary_emb.sin_table"] = cos[0], sin[0]
        wbits = np.concatenate([_bits_any(tensors[w["name"]]) for w in rep.provenance["weights"]])
        tab = materialize(rep.program, [[int(v) for v in ids.tolist()], [int(v) for v in wbits]])
        got = [tab[g] for g in rep.program.circuit.root_outputs()]
        assert got == want.tolist(), (ids, got, want)


def test_moe_routing_is_refused_not_approximated():
    """A sparse MoE gathers expert weights by a top-k routing index: data-dependent structure. Refused at `topk` (no kind; a
    `TopKRef` would only move the refusal to the gather on a value index -> `dynamic_control_flow`). Never a partial program."""
    r = derive(ex.NegMoE(), (torch.empty(3, 16),), mode="reference")
    assert isinstance(r, Unsupported) and r.reason in ("numerics_unregistered", "dynamic_control_flow") and "topk" in str(r), r


@pytest.mark.slow
def test_hf_greedy_request_reference_mode_tokens_match_torch():
    """Bounded HF request (prefix recomputed each step): the STEPS+1 greedy tokens equal torch CPU's. Each step's rotary table is
    its own derived input (the source computes it per call; positions recorded — the prefix relation between them is not asserted)."""
    torch.manual_seed(0)
    C = dict(H=16, NH=2, KVH=1, D=8, I=32, EPS=1e-5, BIAS=False, V=32, NL=2)
    m = ex.HFGreedy(C, STEPS=2)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_((torch.randn_like(p, dtype=torch.float32) * 0.3).to(torch.bfloat16))
    rep = derive_report(m, (torch.empty(3, dtype=torch.int64),), mode="reference", name="HFGreedy")
    assert rep.ok, rep.unsupported
    tables = [w for w in rep.provenance["weights"] if w["kind"] == "derived_table"]
    assert sorted(w["shape"][0] for w in tables) == [3, 3, 4, 4, 5, 5]
    pos = sorted(a["preconditions_checked"]["positions"] for a in rep.provenance["applications"] if a["rule"] == "HFRotaryTable[reference]_v4")
    assert pos == [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]]
    ws = dict(m.named_parameters())
    ws.update(dict(m.named_buffers()))
    for T in (3, 4, 5):
        with torch.no_grad():
            cos, sin = m.lm.rotary_emb(torch.zeros(1, dtype=torch.bfloat16), torch.arange(T).unsqueeze(0))
        for w in tables:
            if w["shape"][0] == T:
                ws[w["name"]] = (cos if "cos" in w["name"] else sin)[0]
    for prompt in (torch.tensor([3, 7, 11]), torch.tensor([31, 0, 4])):
        with torch.no_grad():
            want = m(prompt)
        tab = materialize(rep.program, [[int(v) for v in prompt.tolist()], [int(v) for v in np.concatenate([_bits_any(ws[w["name"]]) for w in rep.provenance["weights"]])]])
        assert [tab[g] for g in rep.program.circuit.root_outputs()] == want.tolist()
