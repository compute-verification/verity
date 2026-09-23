"""R15 dense (F-dense-05) -- `members_for` / `build_manifest` for the Program families of the unfused norm chain (vLLM
`GemmaRMSNorm.forward_native`), GeGLU (`GeluTanhMul`) and the soft caps (FA2 `AttentionSoftcap`, the LogitsProcessor's
`Bf16DivScalar -> Bf16Tanh -> Bf16MulScalar` chain), on a synthetic Gemma-2-shaped request Program (no torch, no pod).

The manifest must (a) require exactly the tensors the modules RETURN -- the norm's `.to(bf16)` rows (`out`; `0` + `1` in the fused
form), the GeGLU output, the attention output, the SOFT-CAPPED logits -- once each per module per step, (b) treat the f32 interior
(square, mean, +eps, rsqrt, row scale, (1+w), row-vector mul, the soft cap's div / tanh, `embed * normalizer`) as no identity at all
(never zero-width), (c) let an explicit member hint beat the default claim on the same (module, member): the soft-capped
LogitsProcessor's Gemm rows are interior, (d) stay complete (no `unmodelled`), and (e) fail closed on a narrowing without a member hint
and on two hinted producers of one member."""
from __future__ import annotations

import pytest

from verity_capture.commit.required_manifest import UnmodelledFamily, build_manifest, members_for, spec_statics

EPS = "C=1e-06"
H, I, V, NH, KVH, D = 64, 128, 96, 2, 1, 32          # small Gemma-2 geometry; hd 32 keeps the FA2 geometry arithmetic honest (BN=128)


def _gemma_program(LP: int = 3, T: int = 1, layers: int = 2, softcap_logits: bool = True, hint_narrow: bool = True, two_hinted: bool = False,
                   hint_cap: bool = True, cap_width: int | None = None, swap_narrows: bool = False, cross_operands: bool = False,
                   narrow_operand: str = "instance") -> dict:
    """`narrow_operand` [rev-pm M-0468]: what the fused-norm narrowings' FIRST operand ref is -- "instance" (derive_step's ["n", src, ..]),
    "param" (a ["p", ..] parameter ref: the Program feeds the narrowing from a parameter), "none" (args stripped: a synthetic Program)."""
    rows: list[dict] = []
    i = [0]

    def add(spec, name, src: int | None = None, reads: int | None = None) -> int:
        """append a row; `src` = the index of the instance whose output is its first operand (`args` ref ["n", src, 0, N], as derive_step
        writes them) -- given for the narrowings so `build_manifest` can confirm `producer_operand` from the Program's dataflow.
        `reads` [R19 int-20] = the instance whose output the row consumes as its module's INPUT (the first norm chain reads the root
        module's scaled embedding): a plain operand run, not subject to the `narrow_operand` variants (those are the narrowings' own)."""
        args = [[["n", src, 0, H]]] if src is not None else []
        if src is not None and narrow_operand == "param":
            args = [[["p", 1, 0, H]]]
        elif src is not None and narrow_operand == "none":
            args = []
        if reads is not None:
            args = args + [[["n", reads, 0, H]]]
        rows.append({"i": i[0], "spec": spec, "form": "call", "name": name, "args": args})
        i[0] += 1
        return i[0] - 1

    def norm_chain(mod: str, sfx: str, n_rows: int, fused: bool, tag: str, x: list[int] | None = None):
        """`x` [R19 int-20]: the rows this norm's INPUT comes from (per row) -- the ops that read the module input (the square / row scale, and
        the residual add of the fused form) carry the operand run, exactly as the real Program wires layer 0's norms to `model/mul`."""
        add(f"AddScalarBf16_v1{{N={H},C=1.0}}", f"{mod}/add{tag}{sfx}[0]")                              # (1 + w): one row, weight-derived
        adds: list[int] = []
        if fused and swap_narrows:                                                                       # a Program whose normed narrowing
            for r in range(n_rows):                                                                      # precedes the residual's (never
                add(f"NarrowF32ToBf16_v1{{N={H}}}", f"{mod}/to_5{tag}{sfx}/0[{r}]")                      # vLLM's order): ordinal contradiction
        if fused:
            for r in range(n_rows):
                adds.append(add(f"AddWidenedBf16_v1{{N={H}}}", f"{mod}/add_1{tag}{sfx}[{r}]", reads=(x[r] if x else None)))
            for r in range(n_rows):
                add(f"NarrowF32ToBf16_v1{{N={H}}}", f"{mod}/to_3{tag}{sfx}/1[{r}]" if hint_narrow else f"{mod}/to_3{tag}{sfx}[{r}]",
                    src=adds[r])                                                                         # the residual narrows the widened sum
        for r in range(n_rows):
            add(f"{'SquareF32' if fused else 'SquareBf16'}_v1{{N={H}}}", f"{mod}/pow{tag}{sfx}[{r}]", reads=(x[r] if x else None))
        for r in range(n_rows):
            add(f"MeanTriton_v1{{N={H}}}", f"{mod}/mean{tag}{sfx}[{r}]")
        for r in range(n_rows):
            add(f"AddScalarF32_v1{{N=1,{EPS}}}", f"{mod}/add_2{tag}{sfx}[{r}]")
        for r in range(n_rows):
            add("RsqrtF32_v1{N=1}", f"{mod}/rsqrt{tag}{sfx}[{r}]")
        for r in range(n_rows):
            add(f"{'ScaleRowF32' if fused else 'ScaleRowBf16'}_v1{{N={H}}}", f"{mod}/mul{tag}{sfx}[{r}]", reads=(x[r] if x else None))
        muls: list[int] = []
        for r in range(n_rows):
            muls.append(add(f"MulVecF32_v1{{N={H}}}", f"{mod}/mul_1{tag}{sfx}[{r}]"))
        if not (fused and swap_narrows):
            for r in range(n_rows):
                m = "0" if fused else "out"
                # cross_operands: the normed narrowing's operand is the AddWidened sum (a Program whose "0" narrowing is fed by the residual
                # producer -- never vLLM's dataflow): producer_operand contradiction, must be refused by name
                add(f"NarrowF32ToBf16_v1{{N={H}}}", f"{mod}/to_5{tag}{sfx}/{m}[{r}]" if hint_narrow else f"{mod}/to_5{tag}{sfx}[{r}]",
                    src=(adds[r] if (cross_operands and fused) else muls[r]))

    pos = 0
    for s in range(T + 1):
        n_rows = LP if s == 0 else 1
        sfx = "" if s == 0 else f"_{s}"
        embeds: list[int] = []
        for r in range(n_rows):
            embeds.append(add(f"Embedding_v1{{V={V},H={H}}}", f"model.model.embed_tokens/embedding{sfx}[{r}]"))
        # [R19 int-20, F-r19-int-20] `embed * normalizer`: the recorded Program (gemma2-2b, row 1024 `model/mul[0]`) puts the scale at the
        # UNROOTED wrapper-level path `model` -- the op is emitted in a module's forward BODY, not by a leaf, so no leaf module owns it -- and
        # its output is read by layer 0's `input_layernorm` and `pre_feedforward_layernorm` (the residual).  This fixture used to assume
        # `model.model/mul` (a rooted Gemma2Model path) with no operand runs: a Program shape that never existed.  The rows now carry the real
        # wiring so the manifest's dataflow audit sees the crossing.
        scaled: list[int] = []
        for r in range(n_rows):
            scaled.append(add(f"Bf16MulScalarTensor_v1{{N={H}}}", f"model/mul{sfx}[{r}]", reads=embeds[r]))
        for L in range(layers):
            norm_chain(f"model.model.layers.{L}.input_layernorm", sfx, n_rows, fused=L > 0, tag="", x=(scaled if L == 0 else None))
            for r in range(n_rows):
                add(f"Gemm_v2{{K={H},N={(NH + 2 * KVH) * D},DOT=k16}}", f"model.model.layers.{L}.self_attn.qkv_proj/triton_launch_1{sfx}[{r}]")
            for r in range(n_rows):
                add(f"RoPE_v1{{NHEADS={NH},D={D}}}", f"model.model.layers.{L}.self_attn.rotary_emb/rotary_embedding{sfx}/query[{r}]")
                add(f"RoPE_v1{{NHEADS={KVH},D={D}}}", f"model.model.layers.{L}.self_attn.rotary_emb/rotary_embedding{sfx}/key[{r}]")
            for r in range(n_rows):
                add(f"AttentionSoftcap_v1{{T={pos + r + 1},NH={NH},KVH={KVH},D={D},BN=128,CAP=50.0}}",
                    f"model.model.layers.{L}.self_attn.attn/attention{sfx}_{r}[0]")
            for r in range(n_rows):
                add(f"Gemm_v2{{K={NH * D},N={H},DOT=k16}}", f"model.model.layers.{L}.self_attn.o_proj/triton_launch_2{sfx}[{r}]")
            norm_chain(f"model.model.layers.{L}.post_attention_layernorm", sfx, n_rows, fused=False, tag="_p")
            norm_chain(f"model.model.layers.{L}.pre_feedforward_layernorm", sfx, n_rows, fused=True, tag="_f", x=(scaled if L == 0 else None))
            for r in range(n_rows):
                add(f"Gemm_v2{{K={H},N={2 * I},DOT=k16}}", f"model.model.layers.{L}.mlp.gate_up_proj/triton_launch_3{sfx}[{r}]")
                add(f"GeluTanhMul_v1{{I={I}}}", f"model.model.layers.{L}.mlp.act_fn/gelu_tanh_and_mul{sfx}[{r}]")
                add(f"Gemm_v2{{K={I},N={H},DOT=k16}}", f"model.model.layers.{L}.mlp.down_proj/triton_launch_4{sfx}[{r}]")
            norm_chain(f"model.model.layers.{L}.post_feedforward_layernorm", sfx, n_rows, fused=False, tag="_q")
        norm_chain("model.model.norm", sfx, n_rows, fused=True, tag="_n")
        add(f"Gemm_v2{{K={H},N={V},DOT=k16}}", f"model.logits_processor/triton_launch_9{sfx}[0]")
        if softcap_logits:
            add(f"Bf16DivScalar_v1{{N={V},C=30.0}}", f"model.logits_processor/div{sfx}[0]")
            add(f"Bf16Tanh_v1{{N={V}}}", f"model.logits_processor/tanh{sfx}[0]")
            add(f"Bf16MulScalar_v1{{N={cap_width or V},C=30.0}}", f"model.logits_processor/mul_2{sfx}/out[0]" if hint_cap else f"model.logits_processor/mul_2{sfx}[0]")
            if two_hinted:
                add(f"NarrowF32ToBf16_v1{{N={V}}}", f"model.logits_processor/to_9{sfx}/out[0]")
        add(f"TokenSelect_v1{{V={V}}}", f"argmax{sfx}[0]")
        pos += n_rows
    weights = "{model.model.embed_tokens.weight: Array<96,Array<64,Value<16>>>, model.model.layers.0.input_layernorm.weight: Array<64,Value<16>>}"
    inst = {"schema": "cb-a/instance-sequence/v1", "digest": "d" * 64, "params": [["prompt", f"Array<{LP},Value<32>>"], ["weights", weights]], "rows": rows}
    res = {"program": {"digest": "g" * 64, "function": "Derived[vLLMRequest[Gemma2]]"}, "wrapper": "request", "LP": LP, "T": T,
           "model_pin": {"model": "unsloth/gemma-2-2b"}, "target": {"declared": {"compute_capability": [8, 9]}},
           "attention_impls_observed": {f"model.layers.{L}.self_attn.attn": {"vllm_flash_attn_version": 2, "num_heads": NH, "num_kv_heads": KVH,
                                                                             "head_size": D, "logits_soft_cap": 50.0} for L in range(layers)}}
    return {"instances": inst, "result": res, "artifact": {"program_digest": "g" * 64, "identity": "i" * 64}, "dir": "<synthetic>"}


def _outputs(man: dict, step: int) -> dict[tuple[str, str], dict]:
    return {(r["op_path"], r["output_member"]): r for r in man["identities"] if r["family"] == "instance_outputs" and r["step"] == step}


# ---------------------------------------------------------------------------------------------------------------------------------
# members_for
# ---------------------------------------------------------------------------------------------------------------------------------

def test_members_for_dense_families():
    st = spec_statics(f"NarrowF32ToBf16_v1{{N={H}}}")
    assert members_for("NarrowF32ToBf16_v1", st, "model.model.layers.0.input_layernorm", "out") == [("out", H, "bf16")]
    # [M-0323] the fused form's two narrowings carry the producer's index among the module's NarrowF32ToBf16 ops in Program order:
    # #0 = the residual `s32.to(bf16)` (member "1"), #1 = the normed `x.to(orig_dtype)` at the return (member "0")
    # [rev-pm M-0402] + `producer_operand`: the family the producing narrowing consumes (order-free selector for the C2 oracle)
    assert members_for("NarrowF32ToBf16_v1", st, "model.model.layers.1.input_layernorm", "0") == [("0", H, "bf16", {"producer_ordinal": 1, "producer_operand": "MulVecF32_v1"})]
    assert members_for("NarrowF32ToBf16_v1", st, "model.model.layers.1.input_layernorm", "1") == [("1", H, "bf16", {"producer_ordinal": 0, "producer_operand": "AddWidenedBf16_v1"})]
    with pytest.raises(UnmodelledFamily):
        members_for("NarrowF32ToBf16_v1", st, "model.model.layers.1.input_layernorm", "")          # a narrowing with no declared member
    assert members_for("Bf16MulScalar_v1", spec_statics(f"Bf16MulScalar_v1{{N={V},C=30.0}}"), "model.logits_processor", "out") == [("out", V, "bf16")]
    assert members_for("Bf16MulScalar_v1", spec_statics(f"Bf16MulScalar_v1{{N={V},C=30.0}}"), "model.logits_processor", "") == []
    assert members_for("GeluTanhMul_v1", spec_statics(f"GeluTanhMul_v1{{I={I}}}"), "model.model.layers.0.mlp.act_fn", "") == [("out", I, "bf16")]
    assert members_for("AttentionSoftcap_v1", spec_statics(f"AttentionSoftcap_v1{{T=3,NH={NH},KVH={KVH},D={D},BN=128,CAP=50.0}}"),
                       "model.model.layers.0.self_attn.attn", "") == [("out", NH * D, "bf16")]
    for fam in ("SquareBf16_v1", "SquareF32_v1", "MeanTriton_v1", "AddScalarF32_v1", "AddScalarBf16_v1", "AddWidenedBf16_v1", "RsqrtF32_v1",
                "ScaleRowBf16_v1", "ScaleRowF32_v1", "MulVecF32_v1", "Bf16DivScalar_v1", "Bf16Tanh_v1", "Bf16MulScalarTensor_v1"):
        assert members_for(fam, {"N": H, "C": 1.0}, "model.model.layers.0.input_layernorm", "") == [], fam
    with pytest.raises(UnmodelledFamily):
        members_for("Phantom_v1", {"N": 4}, "model.model.layers.0.mlp.phantom", "")                  # MAN-02 unchanged


# ---------------------------------------------------------------------------------------------------------------------------------
# build_manifest on the Gemma-2-shaped Program
# ---------------------------------------------------------------------------------------------------------------------------------

def test_gemma_manifest_requires_exactly_the_returned_tensors():
    LP, T = 3, 1
    man = build_manifest(_gemma_program(LP, T), {"request_ids": [0], "sweep": {"arrive_steps": [0]}})
    assert man["complete"] is True and not man.get("unmodelled"), man.get("unmodelled")
    for step, n_rows in ((0, LP), (1, 1)):
        outs = _outputs(man, step)
        for L in range(2):
            p = f"model.layers.{L}"
            # norms: the residual-free form returns ONE tensor (`out`); the fused form (normed, residual) -> `0` and `1`
            if L == 0:
                assert outs[(f"{p}.input_layernorm", "out")]["element_range"] == [0, n_rows * H]
                assert (f"{p}.input_layernorm", "0") not in outs and (f"{p}.input_layernorm", "1") not in outs
            else:
                assert outs[(f"{p}.input_layernorm", "0")]["element_range"] == [0, n_rows * H]
                assert outs[(f"{p}.input_layernorm", "1")]["element_range"] == [0, n_rows * H]
                assert (f"{p}.input_layernorm", "out") not in outs
            assert outs[(f"{p}.post_attention_layernorm", "out")]["element_range"] == [0, n_rows * H]
            assert outs[(f"{p}.pre_feedforward_layernorm", "0")]["element_range"] == [0, n_rows * H]
            assert outs[(f"{p}.pre_feedforward_layernorm", "1")]["element_range"] == [0, n_rows * H]
            assert outs[(f"{p}.post_feedforward_layernorm", "out")]["element_range"] == [0, n_rows * H]
            # GeGLU and the soft-capped attention: one output tensor each
            assert outs[(f"{p}.mlp.act_fn", "out")]["element_range"] == [0, n_rows * I] and outs[(f"{p}.mlp.act_fn", "out")]["spec"] == "GeluTanhMul_v1"
            assert outs[(f"{p}.self_attn.attn", "out")]["element_range"] == [0, n_rows * NH * D] and outs[(f"{p}.self_attn.attn", "out")]["spec"] == "AttentionSoftcap_v1"
            # the linears are unchanged (member 0)
            for leaf in ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"):
                assert (f"{p}.{leaf}", "0") in outs
        assert outs[("model.norm", "0")]["element_range"] == [0, n_rows * H] and outs[("model.norm", "1")]["element_range"] == [0, n_rows * H]
        # the soft-capped logits: ONE identity, produced by the cap's last mul (the Gemm rows are interior), width V, one row
        lg = outs[("logits_processor", "out")]
        assert lg["spec"] == "Bf16MulScalar_v1" and lg["element_range"] == [0, V] and lg["rows"] == 1, lg
        # the interior never appears: no identity mentions the chain's f32 rows, the (1+w) vector or the soft cap's div / tanh -- every
        # one of them is read only inside its own module
        for (op, member), r in outs.items():
            assert not r["spec"].startswith(("SquareBf16", "SquareF32", "MeanTriton", "AddScalar", "AddWidened", "Rsqrt", "ScaleRow", "MulVec",
                                            "Bf16DivScalar", "Bf16Tanh")), (op, member, r["spec"])
            assert r["element_range"][1] > 0
        # [R19 int-20, F-r19-int-20] the embed scale is DIFFERENT: `members_for` still names no member for Bf16MulScalarTensor (it is interior
        # to whatever module computes it), but this Program's dataflow reads its output from layer 0's norms -- another module -- so the
        # manifest PROMOTES it: (`model`, `out`) per step, the crossing rows, width H bf16 from the registry Definition's output signature,
        # the reading modules named for the collector.  This test used to assert the member never appears; that assertion encoded the
        # finding (a value crossing a module boundary that no required member covered), not the rule.
        sc = outs[("model", "out")]
        assert sc["spec"] == "Bf16MulScalarTensor_v1" and sc["element_range"] == [0, n_rows * H] and sc["rows"] == n_rows and sc["dtype"] == "bf16", sc
        assert sc["promoted"] is True and sc["boundary_class"] == "interior_crossing"
        assert sc["consumers"] == ["model.layers.0.input_layernorm", "model.layers.0.pre_feedforward_layernorm"], sc["consumers"]
        assert ("", "out") not in outs
    promo = man["query"]["promotion"]
    assert promo["families"]["Bf16MulScalarTensor_v1"]["rows"] == LP + T and promo["identities"] == T + 1 and promo["faults"] == {}, promo
    # coverage families of the query are all present; the hidden stream is per layer per step
    fams = man["populations"]["by_family"]
    assert fams["instance_outputs"] > 0 and fams["fa2_hidden_m1_stream"] == 2 * (T + 1)


def test_uncapped_logits_processor_is_the_gemm_output():
    """Without a soft cap the LogitsProcessor's Gemm rows keep their default claim (`out`, width V) -- the R14 rule, unchanged."""
    man = build_manifest(_gemma_program(softcap_logits=False), {"request_ids": [0], "sweep": {"arrive_steps": [0]}})
    lg = _outputs(man, 0)[("logits_processor", "out")]
    assert lg["spec"] == "Gemm_v2" and lg["element_range"] == [0, V] and man["complete"] is True


def test_fused_norm_rows_carry_the_program_order_producer_ordinal():
    """[manbind M-0323] the fused norm's two NarrowF32ToBf16 identities name WHICH of the module's two narrowings produces them
    (`producer_ordinal`: 0 = the residual, member "1"; 1 = the normed row, member "0"; [rev-pm M-0402] `producer_operand`: the family
    the narrowing consumes, AddWidenedBf16 / MulVecF32 -- an order-free selector), on every step; a single-narrowing norm's `out` carries
    neither.  Both are checked against the Program itself: a Program whose normed narrowing precedes the residual's (ordinal) or whose
    "0" narrowing is fed by the residual sum (operand) is refused by name (fail closed) instead of sending the C2 oracle to the wrong
    fold instance."""
    LP, T = 3, 1
    man = build_manifest(_gemma_program(LP, T), {"request_ids": [0], "sweep": {"arrive_steps": [0]}})
    assert man["complete"] is True and not man.get("unmodelled"), man.get("unmodelled")
    for step in (0, 1):
        outs = _outputs(man, step)
        for mod in ("model.layers.1.input_layernorm", "model.layers.0.pre_feedforward_layernorm", "model.layers.1.pre_feedforward_layernorm", "model.norm"):
            assert outs[(mod, "1")]["producer_ordinal"] == 0 and outs[(mod, "1")]["spec"] == "NarrowF32ToBf16_v1", outs[(mod, "1")]
            assert outs[(mod, "0")]["producer_ordinal"] == 1 and outs[(mod, "0")]["spec"] == "NarrowF32ToBf16_v1", outs[(mod, "0")]
            assert outs[(mod, "1")]["producer_operand"] == "AddWidenedBf16_v1" and outs[(mod, "0")]["producer_operand"] == "MulVecF32_v1"
        for mod in ("model.layers.0.input_layernorm", "model.layers.0.post_attention_layernorm", "model.layers.1.post_feedforward_layernorm"):
            assert "producer_ordinal" not in outs[(mod, "out")] and "producer_operand" not in outs[(mod, "out")], outs[(mod, "out")]
        # no other identity carries an ordinal / operand (the linears, attention, GeGLU, logits, embedding are single-producer members)
        assert all("producer_ordinal" not in r and "producer_operand" not in r for (op, m), r in outs.items() if r["spec"] != "NarrowF32ToBf16_v1")
    # the same Program with the normed narrowing emitted BEFORE the residual's: both declared ordinals contradict the row order
    bad = build_manifest(_gemma_program(LP, T, swap_narrows=True), {"request_ids": [0], "sweep": {"arrive_steps": [0]}})
    assert bad["complete"] is False
    keys = [k for k in bad["unmodelled"] if "producer_ordinal" in k]
    assert keys, bad["unmodelled"]
    assert any("(0: declared producer_ordinal 1 but the Program's row order puts this op at 0)" in k for k in keys), keys
    assert any("(1: declared producer_ordinal 0 but the Program's row order puts this op at 1)" in k for k in keys), keys
    outs = _outputs(bad, 0)
    assert ("model.layers.1.input_layernorm", "0") not in outs and ("model.layers.1.input_layernorm", "1") not in outs
    # [rev-pm M-0402] the Program's DATAFLOW contradicts the declared operand: the "0" narrowing fed by the AddWidened sum (row order intact,
    # so the ordinal alone would have accepted it) -> refused by name, complete:false, the member absent
    bad2 = build_manifest(_gemma_program(LP, T, cross_operands=True), {"request_ids": [0], "sweep": {"arrive_steps": [0]}})
    assert bad2["complete"] is False
    keys = [k for k in bad2["unmodelled"] if "producer_operand" in k]
    assert keys and all("producer_ordinal" not in k for k in bad2["unmodelled"]), bad2["unmodelled"]
    assert any("(0: declared producer_operand MulVecF32_v1 but the Program's dataflow feeds this op from AddWidenedBf16_v1)" in k for k in keys), keys
    outs = _outputs(bad2, 0)
    assert ("model.layers.1.input_layernorm", "0") not in outs and outs[("model.layers.1.input_layernorm", "1")]["producer_operand"] == "AddWidenedBf16_v1"
    # [rev-pm M-0468] an operand the Program CANNOT confirm (the narrowing's first operand is a parameter ref / the row carries no args) is
    # refused by name too -- never emitted on the strength of its presence, exactly on the Programs where the dataflow cannot be checked
    for kind, why in (("param", "the Program feeds this op from a parameter, not an instance output"), ("none", "the Program row carries no args")):
        bad3 = build_manifest(_gemma_program(LP, T, narrow_operand=kind), {"request_ids": [0], "sweep": {"arrive_steps": [0]}})
        assert bad3["complete"] is False, kind
        keys = [k for k in bad3["unmodelled"] if "producer_operand" in k]
        assert keys and all("producer_ordinal" not in k for k in bad3["unmodelled"]), (kind, bad3["unmodelled"])
        assert any(f"(1: declared producer_operand AddWidenedBf16_v1 but {why}: the declared operand cannot be confirmed)" in k for k in keys), (kind, keys)
        assert any(f"(0: declared producer_operand MulVecF32_v1 but {why}: the declared operand cannot be confirmed)" in k for k in keys), (kind, keys)
        outs = _outputs(bad3, 0)
        assert ("model.layers.1.input_layernorm", "0") not in outs and ("model.layers.1.input_layernorm", "1") not in outs, kind
        assert outs[("model.layers.0.input_layernorm", "out")]["spec"] == "NarrowF32ToBf16_v1", kind          # the un-fused norms are untouched


def test_explicit_hint_beats_default_in_either_row_order():
    """The precedence rule does not depend on which producer the Program lists first."""
    prog = _gemma_program()
    rows = prog["instances"]["rows"]
    # move the soft cap's `out` row of step 0 BEFORE the Gemm row of the same module
    idx_mul = next(k for k, r in enumerate(rows) if r["name"].startswith("model.logits_processor/mul_2/out"))
    idx_gemm = next(k for k, r in enumerate(rows) if r["name"].startswith("model.logits_processor/triton_launch_9["))
    assert idx_gemm < idx_mul
    rows.insert(idx_gemm, rows.pop(idx_mul))
    for k, r in enumerate(rows):
        r["i"] = k
    man = build_manifest(prog, {"request_ids": [0], "sweep": {"arrive_steps": [0]}})
    lg = _outputs(man, 0)[("logits_processor", "out")]
    assert lg["spec"] == "Bf16MulScalar_v1" and lg["element_range"] == [0, V] and lg["rows"] == 1


def test_fail_closed_narrowing_without_member_and_two_hinted_producers():
    man = build_manifest(_gemma_program(hint_narrow=False), {"request_ids": [0], "sweep": {"arrive_steps": [0]}})
    assert man["complete"] is False
    assert any(k.startswith("NarrowF32ToBf16_v1 (no member hint") for k in man["unmodelled"]), man["unmodelled"]
    man2 = build_manifest(_gemma_program(two_hinted=True), {"request_ids": [0], "sweep": {"arrive_steps": [0]}})
    assert man2["complete"] is False
    assert any("two hinted producers" in k for k in man2["unmodelled"]), man2["unmodelled"]


def test_sliding_layer_decode_geometry_is_unswapped():
    """Gemma-2's even layers carry `sliding_window [4095, 0]` (vLLM hands FA2 `window_size=(4095, 0)` whatever the request length): the pinned
    FA2 does not fold the GQA decode into the swapped geometry there (flash_api.cpp:646; `hidden_source.fa2_swapped`), so the hidden
    identity's geometry name is `g{b*H}x1x..` on those layers and `g{b*Hk}x{H/Hk}x..` on the global ones -- same word count."""
    prog = _gemma_program(LP=3, T=1)
    obs = prog["result"]["attention_impls_observed"]
    obs["model.layers.0.self_attn.attn"]["sliding_window"] = [4095, 0]
    obs["model.layers.1.self_attn.attn"]["sliding_window"] = [-1, -1]
    man = build_manifest(prog, {"request_ids": [0], "sweep": {"arrive_steps": [0]}})
    hid = {(r["op_path"], r["step"]): r for r in man["identities"] if r["family"] == "fa2_hidden_m1_stream"}
    # prefill (3 rows): no swap on either layer
    assert hid[("fa2.m1/L0", 0)]["output_member"] == "g2x3x1x128x32" and hid[("fa2.m1/L1", 0)]["output_member"] == "g2x3x1x128x32"
    # decode: windowed layer un-swapped (b*H = 2 slabs of 1 row), global layer swapped (b*Hk = 1 slab of H/Hk = 2 rows)
    assert hid[("fa2.m1/L0", 1)]["output_member"] == "g2x1x1x128x32", hid[("fa2.m1/L0", 1)]
    assert hid[("fa2.m1/L1", 1)]["output_member"] == "g1x2x1x128x32", hid[("fa2.m1/L1", 1)]
    assert hid[("fa2.m1/L0", 1)]["element_range"] == hid[("fa2.m1/L1", 1)]["element_range"]        # same words either way
    assert hid[("fa2.m1/L0", 1)]["geometry"]["HB"] == 2 and hid[("fa2.m1/L0", 1)]["geometry"]["M"] == 1


def test_fail_closed_unhinted_softcap_chain_and_width_mismatch_on_replace():
    """[rev-cb M-0210 D1/D2] (D1) a module carrying a soft-cap chain whose last mul is NOT hinted must not silently fall back to the Gemm's
    default `out` claim (the collector binds the RETURNED, post-cap tensor; the identity key has no family, so nothing would name the miss)
    -> unmodelled, complete:false; (D2) a hinted producer that replaces a default claim with a different width is a Program defect -> unmodelled."""
    man = build_manifest(_gemma_program(hint_cap=False), {"request_ids": [0], "sweep": {"arrive_steps": [0]}})
    assert man["complete"] is False
    assert any("soft-cap chain without a hinted producer" in k and "logits_processor" in k for k in man["unmodelled"]), man["unmodelled"]
    man2 = build_manifest(_gemma_program(cap_width=V // 2), {"request_ids": [0], "sweep": {"arrive_steps": [0]}})
    assert man2["complete"] is False
    assert any("replaces a default claim" in k and "logits_processor" in k for k in man2["unmodelled"]), man2["unmodelled"]
    # the replace check does not depend on row order: hinted first, then a default claim of another width is the two-widths case
    prog = _gemma_program(cap_width=V // 2)
    rows = prog["instances"]["rows"]
    idx_mul = next(k for k, r in enumerate(rows) if r["name"].startswith("model.logits_processor/mul_2/out"))
    idx_gemm = next(k for k, r in enumerate(rows) if r["name"].startswith("model.logits_processor/triton_launch_9["))
    rows.insert(idx_gemm, rows.pop(idx_mul))
    for k, r in enumerate(rows):
        r["i"] = k
    man3 = build_manifest(prog, {"request_ids": [0], "sweep": {"arrive_steps": [0]}})
    assert man3["complete"] is False
    assert any("vs a default claim" in k and "logits_processor" in k for k in man3["unmodelled"]), man3["unmodelled"]


def test_pre_r15_program_manifest_unchanged():
    """The R14 synthetic Program (RMSNormTriton / RMSNormFusedCuda / SiluMul / Attention_v3 / Gemm_v1) builds the same manifest as before
    the precedence rule: no pre-R15 family names its rows `hint == member`."""
    from verity_capture.commit.tests.test_required_manifest import _program
    man = build_manifest(_program(), {"request_ids": [0], "sweep": {"arrive_steps": [0]}})
    assert man["complete"] is True
    outs = _outputs(man, 0)
    assert outs[("logits_processor", "out")]["spec"] == "Gemm_v1"
    assert outs[("model.layers.1.input_layernorm", "0")]["spec"] == "RMSNormFusedCuda_v2" and outs[("model.layers.1.input_layernorm", "1")]["spec"] == "RMSNormFusedCuda_v2"
    assert outs[("model.layers.0.self_attn.rotary_emb", "0")]["spec"] == "RoPE_v1" and outs[("model.layers.0.self_attn.rotary_emb", "1")]["spec"] == "RoPE_v1"
