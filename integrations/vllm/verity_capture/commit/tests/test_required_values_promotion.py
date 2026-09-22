"""R19 int-20 (F-r19-int-20): boundary PROMOTION -- an INTERIOR-classified row (the vocabulary `members_for` names no member for its family)
whose output the Program's own dataflow reads from ANOTHER module is promoted to a required member `(module, out)` of its step, in all three
engines (traversal / reference / inventory: the one-release cross-check must agree row for row), with the member's width / dtype derived
from the registry Definition's output signature and the reading modules recorded as `consumers` (the collector observes the value at a
consumer's input).  The rule is ADDITIVE and GENERIC: it names no model, no vLLM class, no module path.  Fail closed by name: a promoted
family whose signature the registry cannot derive, an unrooted producer whose consumers are not under it (MAN-08).  No torch, no pod."""
from __future__ import annotations

import copy

from verity_capture.commit import required_manifest as RM
from verity_capture.commit import required_values as RV
from verity_capture.commit.tests.test_required_manifest import EPS, _layouts_from
from verity_capture.commit.tests.test_required_values import DF, assert_three_way, traversal_rows

WL = {"sweep": {"arrive_steps": [0]}}


def scaled_embed_df(*, LP: int = 3, T: int = 1, producer: str = "model", scale_family: str = "Bf16MulScalarTensor_v1", scale_statics: str | None = None,
                    cross: bool = True, second_consumer: bool = True) -> dict:
    """A dense request Program whose root module scales the embedding in its OWN forward body (`<producer>/mul[r]`: a value produced by a
    module whose only rows are these, read by its CHILD modules) -- the shape F-r19-int-20 found on a real Program: the scale row is
    interior by the vocabulary (`_DENSE_INTERIOR`), its output feeds layer 0's first norm (and, as the residual, the post-attention norm).
    `producer` = the Program module path of the scaling module (the real Program names it UNROOTED: `model`, MAN-08 form -- the parent
    of every consumer).  `cross=False`: the scale row is read by a row of ITS OWN module only (an interior op stays interior).
    `scale_family` / `scale_statics`: the promoted family (a family without a derivable output signature must fail closed by name)."""
    b = DF(LP=LP, T=T)
    H, V, NH, KVH, D, I = b.H, b.V, b.NH, b.KVH, b.D, b.I
    layers = 1
    b.weight("model.model.layers.0.self_attn.qkv_proj.weight", f"Array<{(NH + 2 * KVH) * D},Array<{H},Value<16>>>")
    b.weight("model.model.layers.0.self_attn.o_proj.weight", f"Array<{H},Array<{NH * D},Value<16>>>")
    b.weight("model.model.layers.0.mlp.gate_up_proj.weight", f"Array<{2 * I},Array<{H},Value<16>>>")
    b.weight("model.model.layers.0.mlp.down_proj.weight", f"Array<{H},Array<{I},Value<16>>>")
    b.weight("model.lm_head.weight", f"Array<{V},Array<{H},Value<16>>>")
    wp = 1                                                                       # the weights param; the scale's scalar is a slice of it (a buffer)
    keys: list[int] = []
    pos = 0
    last_select = None
    spec = f"{scale_family}{{{scale_statics if scale_statics is not None else f'N={H}'}}}"
    for s in range(T + 1):
        n = LP if s == 0 else 1
        sfx = "" if s == 0 else f"_{s}"
        emb = [b.emit(f"Embedding_v1{{V={V},H={H}}}", f"model.model.embed_tokens/embedding{sfx}[{r}]", inputs=([] if s == 0 else [last_select]),
                      params=([(0, pos + r, pos + r + 1)] if s == 0 else []) + [(wp, 0, V * H)]) for r in range(n)]
        scaled = [b.emit(spec, f"{producer}/mul{sfx}[{r}]", inputs=[emb[r]], params=[(wp, V * H, V * H + 1)]) for r in range(n)]
        if cross:
            src = scaled
        else:
            # the scale stays INSIDE its module: only a same-module interior op reads it (a norm-chain style f32 mean), the children read
            # the embedding itself -- two interior families under the producer, neither crosses
            for r in range(n):
                b.emit(f"MeanTriton_v1{{N={H}}}", f"{producer}/mean{sfx}[{r}]", inputs=[scaled[r]])
            src = emb
        resid = src
        normed = [b.emit(f"RMSNormTriton_v1{{N={H},{EPS}}}", f"model.model.layers.0.input_layernorm/triton_launch{sfx}[{r}]", inputs=[src[r]]) for r in range(n)]
        qkv = [b.emit(f"Gemm_v1{{K={H},N={(NH + 2 * KVH) * D}}}", f"model.model.layers.0.self_attn.qkv_proj/triton_launch_1{sfx}[{r}]", inputs=[normed[r]]) for r in range(n)]
        q = [b.emit(f"RoPE_v1{{NHEADS={NH},D={D}}}", f"model.model.layers.0.self_attn.rotary_emb/rotary_embedding{sfx}/query[{r}]", inputs=[qkv[r]]) for r in range(n)]
        k = [b.emit(f"RoPE_v1{{NHEADS={KVH},D={D}}}", f"model.model.layers.0.self_attn.rotary_emb/rotary_embedding{sfx}/key[{r}]", inputs=[qkv[r]]) for r in range(n)]
        keys += k
        att = [b.emit(f"Attention_v3{{T={pos + r + 1},NH={NH},KVH={KVH},D={D},BN=128}}", f"model.model.layers.0.self_attn.attn/attention{sfx}_{r}[0]",
                      inputs=[q[r]] + keys[: pos + r + 1] + [qkv[r]]) for r in range(n)]
        o = [b.emit(f"Gemm_v1{{K={NH * D},N={H}}}", f"model.model.layers.0.self_attn.o_proj/triton_launch_2{sfx}[{r}]", inputs=[att[r]]) for r in range(n)]
        post = [b.emit(f"RMSNormFusedCuda_v2{{N={H},{EPS}}}", f"model.model.layers.0.post_attention_layernorm/fused_add_rms_norm_1{sfx}[{r}]",
                       inputs=[o[r]] + ([resid[r]] if second_consumer else [normed[r]])) for r in range(n)]
        gu = [b.emit(f"Gemm_v1{{K={H},N={2 * I}}}", f"model.model.layers.0.mlp.gate_up_proj/triton_launch_3{sfx}[{r}]", inputs=[post[r]]) for r in range(n)]
        act = [b.emit(f"SiluMul_v1{{I={I}}}", f"model.model.layers.0.mlp.act_fn/silu_and_mul{sfx}[{r}]", inputs=[gu[r]]) for r in range(n)]
        mlp_out = [b.emit(f"Gemm_v1{{K={I},N={H}}}", f"model.model.layers.0.mlp.down_proj/triton_launch_4{sfx}[{r}]", inputs=[act[r]]) for r in range(n)]
        fin = [b.emit(f"RMSNormFusedCuda_v2{{N={H},{EPS}}}", f"model.model.norm/fused_add_rms_norm_9{sfx}[{r}]", inputs=[mlp_out[r], post[r]]) for r in range(n)]
        logits = b.emit(f"Gemm_v1{{K={H},N={V}}}", f"model.logits_processor/triton_launch_9{sfx}[0]", inputs=[fin[-1]])
        last_select = b.emit(f"TokenSelect_v1{{V={V}}}", f"argmax{sfx}[0]", inputs=[logits])
        pos += n
    return b.program(layers=layers)


def _promoted(rows: list[dict]) -> dict[tuple[int, str], dict]:
    return {(int(r["step"]), r["op_path"]): r for r in rows if r.get("promoted")}


# ---------------------------------------------------------------------------------------------------------------------------------
# (a) the crossing interior row at an UNROOTED producer path is promoted; the audit is clean; the three engines agree
# ---------------------------------------------------------------------------------------------------------------------------------

def test_a_crossing_interior_row_at_an_unrooted_producer_is_promoted_and_the_engines_agree():
    LP, T = 3, 1
    prog = scaled_embed_df(LP=LP, T=T)
    tr, t = assert_three_way(prog, workload=WL)                                  # traversal == reference == inventory, digest included
    H = 32
    pr = _promoted(tr)
    assert set(pr) == {(0, "model"), (1, "model")}, sorted(pr)
    for step, n in ((0, LP), (1, 1)):
        r = pr[(step, "model")]
        assert r["family"] == "instance_outputs" and r["output_member"] == RM.PROMOTED_MEMBER == "out"
        assert r["element_range"] == [0, n * H] and r["rows"] == n and r["width"] == H and r["dtype"] == "bf16"      # from the registry Definition
        assert r["spec"] == "Bf16MulScalarTensor_v1" and r["boundary_class"] == RM.INTERIOR_CROSSING == "interior_crossing"
        # the reading modules, Program order, as collector paths (the wrapper prefix `model.` stripped; the unrooted producer keeps its name)
        assert r["consumers"] == ["model.layers.0.input_layernorm", "model.layers.0.post_attention_layernorm"], r["consumers"]
        assert "fx_nodes" not in r and RM.canonical_identity(r)[4:6] == ["model", "out"]
    # the audit: nothing unnamed, the rows are class `interior_crossing` (every one of them crosses), counted under `promoted`
    aud = t.boundary_audit
    assert t.complete and aud["available"] and aud["fully_wired"] and aud["unnamed_crossing"] == {}, (t.unmodelled, aud["unnamed_crossing"])
    assert aud["promoted"] == {"Bf16MulScalarTensor_v1": LP + T}
    assert aud["families"]["Bf16MulScalarTensor_v1"] == {RM.INTERIOR_CROSSING: {"rows": LP + T, "crossing": LP + T}}
    assert aud["named_interior"] == {}
    # the manifest header says what was promoted, by rule; the guard counts it; the additive rule left every vocabulary member in place
    man = RV.manifest_from_traversal(copy.deepcopy(prog), WL)
    assert man["complete"] is True and man["unmodelled"] == {}
    ph = man["query"]["promotion"]
    assert ph["member"] == "out" and ph["boundary_class"] == "interior_crossing" and ph["identities"] == 2 and ph["rows"] == LP + T and ph["faults"] == {}
    assert ph["families"]["Bf16MulScalarTensor_v1"]["consumers"] == ["model.layers.0.input_layernorm", "model.layers.0.post_attention_layernorm"]
    assert "Bf16MulScalarTensor" in ph["rule"] or "INTERIOR" in ph["rule"]
    assert man["guard"]["promoted"] == {"Bf16MulScalarTensor_v1": LP + T} and man["guard"]["flags"]["unnamed_crossing"] == 0
    outs = {(r["op_path"], r["output_member"]) for r in man["identities"] if r["step"] == 0 and r["family"] == "instance_outputs"}
    assert ("model.layers.0.input_layernorm", "out") in outs and ("model.embed_tokens", "out") in outs and ("logits_processor", "out") in outs
    # the promoted identity is a bindable required value: a collector covering the manifest exactly passes, one omitting it fails BY NAME.
    # [Commit 2, PROMOTED_OBSERVATION_RULE] its committed name is its first consumer's INPUT (`model.layers.0.input_layernorm/in`), never the
    # producer module's own output (`model`: the model root's forward output is not the scaled embedding) -- a bare `model` tensor is
    # counted as misbound and covers nothing
    cov = RM.coverage_check(man, _layouts_from(man))
    assert cov["ok"] and cov["promoted_observed_as"] == {"model.layers.0.input_layernorm/in": ["model", "out"]} and cov["promoted_misbound_n"] == 0
    lay = _layouts_from(man)
    for step in lay:
        lay[step] = [m for m in lay[step] if m["name"] != "model.layers.0.input_layernorm/in"]
    cov = RM.coverage_check(man, lay)
    assert not cov["ok"] and cov["missing_n"] == 2 and all(m["op_path"] == "model" and m["output_member"] == "out" for m in cov["missing"]), cov["missing"]
    for step in lay:
        lay[step].append({"name": "model", "numel": 4 * 3 * 2304 * 16, "inv": 0})    # the producer module's own output, committed: not a binder
    cov = RM.coverage_check(man, lay)
    assert not cov["ok"] and cov["missing_n"] == 2 and cov["promoted_misbound"] == ["model"] and cov["promoted_misbound_n"] == 2, cov
    # the inventory (the other side of the one-release cross-check) promotes the same rows: agree
    inv = RM.build_manifest(copy.deepcopy(prog), WL, engine="inventory")
    cc = RM.cross_check_manifests(man, inv, primary_engine="traversal", other_engine="inventory")
    assert cc["agree"], cc.get("why")
    assert inv["query"]["promotion"] == man["query"]["promotion"]


def test_promotion_is_deferred_until_every_consumer_is_walked_and_the_rows_stay_ordered_by_key():
    """A consumer may sit in a LATER step than its producer (attention reads earlier steps' keys): the traversal decides the promotion
    after the whole walk, and the manifest rows still sort by the canonical key (the promoted rows are not appended at the end)."""
    prog = scaled_embed_df(LP=2, T=2)
    tr, t = traversal_rows(prog, RV.Query(arrive_step=0))
    assert tr == sorted(tr, key=RM._row_key)
    steps = [r["step"] for r in tr if r.get("promoted")]
    assert steps == [0, 1, 2]
    # the streamed view (rows never all in memory) promotes the same
    v = RV.ProgramView.of(copy.deepcopy(prog))
    assert [x.row() for x in RV.Traversal(v)].__len__() == len(tr)


def test_an_unrooted_producer_whose_consumers_are_not_under_it_is_refused_by_name_man_08():
    """[MAN-08] an unrooted module path is legitimate as a promoted producer ONLY when the dataflow shows it is the ancestor of every
    reader (a root-level module whose forward body produced the value).  A bare leaf that lost its parent (`gate`) reading into
    `model.model.layers...` is a Program naming defect: incomplete, named, never re-addressed."""
    prog = scaled_embed_df(LP=2, T=0, producer="gate")
    tr, t = traversal_rows(prog)
    assert not t.complete
    keys = [k for k in t.unmodelled if "MAN-08" in k]
    assert len(keys) == 1 and "unrooted module path 'gate'" in keys[0] and "Bf16MulScalarTensor_v1 under gate" in keys[0], t.unmodelled
    # the rows stay interior with their crossings (not promoted): the audit still counts them as an unnamed crossing (the fact), the
    # fault names the module and the reason
    assert t.boundary_audit["unnamed_crossing"] == {"Bf16MulScalarTensor_v1": 2}
    assert t.boundary_audit["families"]["Bf16MulScalarTensor_v1"] == {"interior": {"rows": 2, "crossing": 2}}
    assert not any(r.get("promoted") for r in tr)
    # the generic "unnamed crossing" line is NOT emitted beside the named fault (one fault, named per module)
    assert not any(k.startswith("Bf16MulScalarTensor_v1: 2 row(s)") for k in t.unmodelled), t.unmodelled
    # the inventory and the reference agree (same fault key)
    inv = RM.build_manifest(copy.deepcopy(prog), WL, engine="inventory")
    assert inv["complete"] is False and set(inv["unmodelled"]) == set(t.unmodelled)
    _ref, summ = RV.reference_required_values(copy.deepcopy(prog), RV.Query())
    assert summ["complete"] is False and set(summ["unmodelled"]) == set(t.unmodelled)


# ---------------------------------------------------------------------------------------------------------------------------------
# (b) an interior row that does NOT cross stays interior: no identity, nothing promoted, the manifest is the vocabulary's
# ---------------------------------------------------------------------------------------------------------------------------------

def test_an_interior_row_that_does_not_cross_stays_interior():
    prog = scaled_embed_df(LP=3, T=1, cross=False)
    tr, t = assert_three_way(prog, workload=WL)
    assert t.complete and t.unmodelled == {} and t.boundary_audit["unnamed_crossing"] == {} and "promoted" not in t.boundary_audit
    assert t.boundary_audit["families"]["Bf16MulScalarTensor_v1"] == {"interior": {"rows": 4, "crossing": 0}}
    assert t.boundary_audit["families"]["MeanTriton_v1"] == {"interior": {"rows": 4, "crossing": 0}}
    assert not any(r.get("promoted") for r in tr) and not any(r["op_path"] == "model" for r in tr)
    assert t.promotion is not None and t.promotion["identities"] == [] and t.promotion["faults"] == {}   # the pass ran (armed) and named nothing
    man = RV.manifest_from_traversal(copy.deepcopy(prog), WL)
    assert "promotion" not in man["query"] and "promoted" not in man["guard"] and "promoted" not in man["boundary_audit"]
    # the same Program with the scale CROSSING differs from this one by exactly the promoted identities (additive: nothing else moves)
    crossing = RV.manifest_from_traversal(scaled_embed_df(LP=3, T=1, cross=True), WL)
    eq = RV.manifest_equal(man, crossing)
    assert not eq["equal"] and eq["only_old"] == [] and len(eq["only_new"]) == 2 and eq["changed"] == [], eq["why"]
    assert all(k[0] == "instance_outputs" and k[5:7] == ["model", "out"] for k in eq["only_new"]), eq["only_new"]


def test_a_norm_chain_interior_row_inside_its_module_is_not_promoted_and_a_program_without_operand_runs_promotes_nothing():
    """The unfused norm chain's f32 interior (square / mean / rsqrt / row scale / the (1+w) mul) is read by the chain's own narrowing:
    interior, no crossing, no promotion -- the R15 fixture's `_gemma_program` (wired narrowings only).  Stripping every `args` disarms
    the audit: nothing promotes (a Program without operand runs is the vocabulary's word alone)."""
    from verity_capture.commit.tests.test_required_manifest_dense import _gemma_program
    prog = _gemma_program(LP=3, T=1)
    tr, t = traversal_rows(prog)
    aud = t.boundary_audit["families"]
    for fam in ("SquareF32_v1", "SquareBf16_v1", "MeanTriton_v1", "RsqrtF32_v1", "ScaleRowF32_v1", "ScaleRowBf16_v1", "MulVecF32_v1", "AddScalarF32_v1"):
        assert set(aud[fam]) == {"interior"} and aud[fam]["interior"]["crossing"] == 0, (fam, aud[fam])
    # the crossing fixture with every `args` stripped: the audit is not armed, the promotion pass does not run, and the manifest is exactly
    # the vocabulary's word (the crossing manifest minus its promoted identities)
    wired = scaled_embed_df(LP=3, T=1)
    trw, tw = traversal_rows(wired)
    unwired = copy.deepcopy(wired)
    for r in unwired["instances"]["rows"]:
        r["args"] = []
    tr2, t2 = traversal_rows(unwired)
    assert t2.boundary_audit["available"] is False and not any(r.get("promoted") for r in tr2) and t2.promotion is None and t2.complete
    assert RM.manifest_digest(tr2) == RM.manifest_digest([r for r in trw if not r.get("promoted")]) != RM.manifest_digest(trw)


# ---------------------------------------------------------------------------------------------------------------------------------
# (c) a promoted family without a derivable output signature is a NAMED fault (never a guessed width / dtype)
# ---------------------------------------------------------------------------------------------------------------------------------

def test_a_promoted_family_without_a_derivable_signature_fails_closed_by_name():
    # Fp8GroupQuant_v1 is interior by the vocabulary (the activation quant of an FP8 linear) and its registry Definition returns a TUPLE
    # (e4m3 words, f32 group scales): there is no single `out` member to size -- a Program that reads it from another module is refused
    prog = scaled_embed_df(LP=2, T=0, scale_family="Fp8GroupQuant_v1", scale_statics="K=32,G=32")
    assert RM.promoted_signature("Fp8GroupQuant_v1{K=32,G=32}")[2] is not None
    tr, t = traversal_rows(prog)
    assert not t.complete and not any(r.get("promoted") for r in tr)
    keys = [k for k in t.unmodelled if RM.UNDERIVABLE in k]
    assert len(keys) == 1 and "Fp8GroupQuant_v1 under model: 2 row(s)" in keys[0] and "not one array of one word width" in keys[0], t.unmodelled
    assert t.boundary_audit["families"]["Fp8GroupQuant_v1"] == {"interior": {"rows": 2, "crossing": 2}}
    # a family that is not a registered Definition at all: refused by name too, and the coverage of the manifest refuses (complete:false)
    prog2 = scaled_embed_df(LP=2, T=0, scale_family="Bf16MulScalarTensor_v7")
    tr2, t2 = traversal_rows(prog2)
    keys2 = [k for k in t2.unmodelled if RM.UNDERIVABLE in k]
    assert len(keys2) == 1 and "Bf16MulScalarTensor_v7 is not a registered Definition" in keys2[0], t2.unmodelled
    man = RV.manifest_from_traversal(copy.deepcopy(prog2), WL)
    assert man["complete"] is False and man["query"]["promotion"]["identities"] == 0 and man["query"]["promotion"]["faults"] == {keys2[0]: 2}
    assert not RM.coverage_check(man, _layouts_from(man))["ok"]
    # both engines and the reference name the same fault
    inv = RM.build_manifest(copy.deepcopy(prog2), WL, engine="inventory")
    assert inv["complete"] is False and set(inv["unmodelled"]) == set(t2.unmodelled)
    _ref, summ = RV.reference_required_values(copy.deepcopy(prog2), RV.Query())
    assert set(summ["unmodelled"]) == set(t2.unmodelled)


def test_promoted_signature_is_the_registry_definitions_output_bound_to_the_rows_statics():
    """width x dtype come from the Definition's return type at the row's statics -- never from the vocabulary, never a default."""
    assert RM.promoted_signature("Bf16MulScalarTensor_v1{N=2304}") == (2304, "bf16", None)
    assert RM.promoted_signature("MeanTriton_v1{N=64}") == (1, "f32", None)                       # a per-row scalar in f32
    assert RM.promoted_signature("NarrowF32ToBf16_v1{N=8}") == (8, "bf16", None)
    w, d, why = RM.promoted_signature("Bf16MulScalarTensor_v1{}")                                 # statics that do not bind
    assert (w, d) == (None, None) and "statics ['N']" in why
    w, d, why = RM.promoted_signature("LMoeFfn_v1{E=4,TOPK=2,VPT=2,H=8,I=8}")                     # a lifted word width (17) has no manifest dtype
    assert (w, d) == (None, None) and "no manifest dtype" in why


def test_two_promoted_families_on_one_member_and_a_collision_with_a_vocabulary_member_are_refused():
    """The promoted member is ONE name per module: two crossing interior families under one module in one step, or a module whose
    vocabulary already names `out`, cannot be merged silently -- named, incomplete."""
    # two families: the scale AND a same-module MeanTriton both read by the child norm
    b = DF(LP=2, T=0)
    H, V = b.H, b.V
    e = [b.emit(f"Embedding_v1{{V={V},H={H}}}", f"model.model.embed_tokens/embedding[{r}]", params=[(0, r, r + 1)]) for r in range(2)]
    sc = [b.emit(f"Bf16MulScalarTensor_v1{{N={H}}}", f"model.model/mul[{r}]", inputs=[e[r]]) for r in range(2)]
    mn = [b.emit(f"MeanTriton_v1{{N={H}}}", f"model.model/mean[{r}]", inputs=[e[r]]) for r in range(2)]
    nm = [b.emit(f"RMSNormTriton_v1{{N={H},{EPS}}}", f"model.model.layers.0.input_layernorm/triton_launch[{r}]", inputs=[sc[r], mn[r]]) for r in range(2)]
    lg = b.emit(f"Gemm_v1{{K={H},N={V}}}", "model.logits_processor/triton_launch_9[0]", inputs=[nm[-1]])
    b.emit(f"TokenSelect_v1{{V={V}}}", "argmax[0]", inputs=[lg])
    tr, t = traversal_rows(b.program(layers=0, attn_obs={}))
    assert not t.complete and not any(r.get("promoted") for r in tr)
    assert any("2 promoted families ['Bf16MulScalarTensor_v1', 'MeanTriton_v1'] claim one member `out`" in k for k in t.unmodelled), t.unmodelled
    # a collision: the producing module's vocabulary already names `out` (an RMSNormTriton row under the same module in the same step)
    b2 = DF(LP=2, T=0)
    e = [b2.emit(f"Embedding_v1{{V={V},H={H}}}", f"model.model.embed_tokens/embedding[{r}]", params=[(0, r, r + 1)]) for r in range(2)]
    sc = [b2.emit(f"Bf16MulScalarTensor_v1{{N={H}}}", f"model.model.pre/mul[{r}]", inputs=[e[r]]) for r in range(2)]
    own = [b2.emit(f"RMSNormTriton_v1{{N={H},{EPS}}}", f"model.model.pre/triton_launch[{r}]", inputs=[e[r]]) for r in range(2)]
    nm = [b2.emit(f"RMSNormTriton_v1{{N={H},{EPS}}}", f"model.model.layers.0.input_layernorm/triton_launch[{r}]", inputs=[sc[r], own[r]]) for r in range(2)]
    lg = b2.emit(f"Gemm_v1{{K={H},N={V}}}", "model.logits_processor/triton_launch_9[0]", inputs=[nm[-1]])
    b2.emit(f"TokenSelect_v1{{V={V}}}", "argmax[0]", inputs=[lg])
    tr2, t2 = traversal_rows(b2.program(layers=0, attn_obs={}))
    assert not t2.complete and any("collides with the module's vocabulary member `out`" in k for k in t2.unmodelled), t2.unmodelled
    assert sum(1 for r in tr2 if r["op_path"] == "model.pre") == 1              # the vocabulary's `out` stands; nothing merged into it
