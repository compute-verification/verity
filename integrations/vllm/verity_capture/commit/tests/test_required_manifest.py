"""R14 MAN-01: the Program/query-derived required-value manifest and its coverage check -- the brief §4 negatives that the manifest
must make detectable BY CONSTRUCTION (omit a whole producer family; omit one step / request / rank; drop `/1` and bind `/0` twice),
on a synthetic 2-layer Program instance sequence (no torch, no pod)."""
from __future__ import annotations

import copy

import pytest

from verity_capture.commit.required_manifest import (build_global_manifest, build_manifest, canonical_identity, coverage_check, manifest_digest, merge_manifests,
                                                     request_label, segment_steps, spec_statics, split_name)

EPS = 'EPS={"f64":"0x1.4f8b588e368f1p-17"}'


def _program(LP: int = 4, T: int = 2, layers: int = 2, NH: int = 2, KVH: int = 1, D: int = 64, H: int = 128, V: int = 64) -> dict:
    """A request-wrapper instance sequence in the derive_step vocabulary: prefill of LP rows + T decode steps, each step ends in TokenSelect."""
    rows: list[dict] = []
    i = [0]

    def add(spec, name):
        rows.append({"i": i[0], "spec": spec, "form": "call", "name": name, "args": []})
        i[0] += 1

    pos = 0
    for s in range(T + 1):
        n_rows = LP if s == 0 else 1
        sfx = "" if s == 0 else f"_{s}"
        for r in range(n_rows):
            add(f"Embedding_v1{{V={V},H={H}}}", f"model.model.embed_tokens/embedding{sfx}[{r}]")
        for L in range(layers):
            for r in range(n_rows):
                if L == 0:
                    add(f"RMSNormTriton_v1{{N={H},{EPS}}}", f"model.model.layers.{L}.input_layernorm/triton_launch{sfx}[{r}]")
                else:
                    add(f"RMSNormFusedCuda_v2{{N={H},{EPS}}}", f"model.model.layers.{L}.input_layernorm/fused_add_rms_norm{sfx}[{r}]")
            for r in range(n_rows):
                add(f"Gemm_v1{{K={H},N={(NH + 2 * KVH) * D}}}", f"model.model.layers.{L}.self_attn.qkv_proj/triton_launch_1{sfx}[{r}]")
            for r in range(n_rows):
                add(f"RoPE_v1{{NHEADS={NH},D={D}}}", f"model.model.layers.{L}.self_attn.rotary_emb/rotary_embedding{sfx}/query[{r}]")
                add(f"RoPE_v1{{NHEADS={KVH},D={D}}}", f"model.model.layers.{L}.self_attn.rotary_emb/rotary_embedding{sfx}/key[{r}]")
            for r in range(n_rows):
                add(f"Attention_v3{{T={pos + r + 1},NH={NH},KVH={KVH},D={D},BN=128}}", f"model.model.layers.{L}.self_attn.attn/attention{sfx}_{r}[0]")
            for r in range(n_rows):
                add(f"Gemm_v1{{K={NH * D},N={H}}}", f"model.model.layers.{L}.self_attn.o_proj/triton_launch_2{sfx}[{r}]")
            for r in range(n_rows):
                add(f"RMSNormFusedCuda_v2{{N={H},{EPS}}}", f"model.model.layers.{L}.post_attention_layernorm/fused_add_rms_norm_1{sfx}[{r}]")
            for r in range(n_rows):
                add(f"Gemm_v1{{K={H},N={2 * H}}}", f"model.model.layers.{L}.mlp.gate_up_proj/triton_launch_3{sfx}[{r}]")
                add(f"SiluMul_v1{{I={H}}}", f"model.model.layers.{L}.mlp.act_fn/silu_and_mul{sfx}[{r}]")
                add(f"Gemm_v1{{K={H},N={H}}}", f"model.model.layers.{L}.mlp.down_proj/triton_launch_4{sfx}[{r}]")
        for r in range(n_rows):
            add(f"RMSNormFusedCuda_v2{{N={H},{EPS}}}", f"model.model.norm/fused_add_rms_norm_9{sfx}[{r}]")
        add(f"Gemm_v1{{K={H},N={V}}}", f"model.logits_processor/triton_launch_9{sfx}[0]")
        add(f"TokenSelect_v1{{V={V}}}", f"argmax{sfx}[0]")
        pos += n_rows
    weights = "{model.model.embed_tokens.weight: Array<64,Array<128,Value<16>>>, model.model.layers.0.self_attn.qkv_proj.weight: Array<256,Array<128,Value<16>>>}"
    inst = {"schema": "cb-a/instance-sequence/v1", "digest": "d" * 64, "params": [["prompt", f"Array<{LP},Value<32>>"], ["weights", weights]], "rows": rows}
    res = {"program": {"digest": "p" * 64, "function": "Derived[vLLMRequest[Test]]"}, "wrapper": "request", "LP": LP, "T": T,
           "model_pin": {"model": "test"}, "target": {"declared": {"compute_capability": [8, 9]}},
           "attention_impls_observed": {f"model.layers.{L}.self_attn.attn": {"vllm_flash_attn_version": 2, "num_heads": NH, "num_kv_heads": KVH, "head_size": D} for L in range(layers)}}
    return {"instances": inst, "result": res, "artifact": {"program_digest": "p" * 64, "identity": "i" * 64}, "dir": "<synthetic>"}


def _layouts_from(man: dict, *, rope_per_layer_names: bool = False) -> dict[int, list[dict]]:
    """A collector's per-step layouts that cover the manifest exactly (the positive case), in the committed-name vocabulary."""
    lay: dict[int, list[dict]] = {}
    for r in man["identities"]:
        if r["family"] == "weights":
            continue
        step = int(r["step"])
        op, member = r["op_path"], r["output_member"]
        if r["family"].endswith("hidden_m1_stream"):
            name = f"{op}/{member}"
        elif r.get("promoted"):
            name = f"{r['consumers'][0]}/in"              # [R19 int-20] a promoted member is observed at its first consumer's input (PROMOTED_OBSERVATION_RULE)
        elif member == "out":
            name = op
        else:
            name = f"{op}/{member}"
        if rope_per_layer_names and r.get("program_op_path"):
            name = f"{r['program_op_path']}/{member}"
        lay.setdefault(step, []).append({"name": name, "numel": r["element_range"][1], "inv": r["invocation"]})
    for step in lay:                     # committed order: the shared rotary module's k-th call is the k-th occurrence in the step
        lay[step].sort(key=lambda m: (m["name"], m["inv"]))
    return lay


def test_parsing():
    assert spec_statics("Gemm_v1{K=576,N=960}") == {"K": 576, "N": 960}
    assert spec_statics(f"RMSNormFusedCuda_v2{{N=576,{EPS}}}")["N"] == 576
    assert split_name("model.model.layers.3.self_attn.rotary_emb/rotary_embedding_3/key[7]") == ("model.model.layers.3.self_attn.rotary_emb", "rotary_embedding_3", "key", 7)
    assert split_name("argmax_2[0]") == ("", "argmax_2", "", 0)


def test_population_and_digest_determinism():
    prog = _program()
    man = build_manifest(prog, {"sweep": {"row_id": "t", "batch": 1, "arrive_steps": [0]}})
    assert man["steps"] == [0, 1, 2] and man["engine_steps"] == [0, 1, 2]
    assert len(segment_steps(prog["instances"]["rows"])) == 3
    fam = man["populations"]["by_family"]
    assert fam["sampled_token_ids"] == 3 and fam["input_token_ids"] == 3
    assert fam["fa2_hidden_m1_stream"] == 2 * 3                       # one launch per attention layer per step
    assert fam["weights"] == 2
    # per step: layer0 (ln out 1, qkv 1, rope 2, attn 1, o 1, post 2, gate_up 1, act 1, down 1 = 11) + layer1 (12) + norm 2 + logits 1 + embed 1 = 27
    assert fam["instance_outputs"] == 27 * 3
    step0 = [r for r in man["identities"] if r["step"] == 0 and r["op_path"] == "model.layers.1.self_attn.qkv_proj"]
    assert step0 and step0[0]["element_range"] == [0, 4 * (2 + 2) * 64] and step0[0]["output_member"] == "0"
    # prefill hidden geometry: HB = heads, M = LP rows, NB = ceil(LP / 128); decode (swapped kernel): HB = KVH, M = NH/KVH
    h0 = [r for r in man["identities"] if r["step"] == 0 and r["op_path"] == "fa2.m1/L1"][0]
    assert h0["output_member"] == "g2x4x1" and h0["element_range"][1] > 0
    h1 = [r for r in man["identities"] if r["step"] == 1 and r["op_path"] == "fa2.m1/L1"][0]
    assert h1["output_member"] == "g1x2x1" and h1["geometry"]["seqlen_k"] == 5
    # the shared RotaryEmbedding: canonical path = layer 0's, invocation = layer index
    rope = sorted((r["invocation"], r["output_member"]) for r in man["identities"] if r["step"] == 1 and r["op_path"] == "model.layers.0.self_attn.rotary_emb")
    assert rope == [(0, "0"), (0, "1"), (1, "0"), (1, "1")]
    assert man["op_path_aliases"]["model.layers.1.self_attn.rotary_emb"] == ["model.layers.0.self_attn.rotary_emb", 1]
    # deterministic digest: rebuild -> same; annotations do not enter; a different Program digest does
    man2 = build_manifest(_program(), {"sweep": {"row_id": "t", "batch": 1, "arrive_steps": [0]}})
    assert man2["manifest_digest"] == man["manifest_digest"]
    ids = copy.deepcopy(man["identities"])
    for r in ids:
        r.pop("geometry", None); r.pop("rows", None)
    assert manifest_digest(ids) == man["manifest_digest"]
    prog3 = _program(); prog3["artifact"]["program_digest"] = "q" * 64
    assert build_manifest(prog3)["manifest_digest"] != man["manifest_digest"]
    assert len(canonical_identity(man["identities"][0])) == 8


def test_coverage_positive_and_negatives():
    man = build_manifest(_program(), {"sweep": {"row_id": "t", "batch": 1, "arrive_steps": [0]}})
    lay = _layouts_from(man)
    ok = coverage_check(man, lay)
    assert ok["ok"], ok["missing"][:3]
    assert ok["committed_unmatched_n"] == 0 and ok["checked"] == man["populations"]["identities"] - man["populations"]["by_family"]["weights"]
    # a collector naming the shared module per layer is accepted through the manifest's aliases
    assert coverage_check(man, _layouts_from(man, rope_per_layer_names=True))["ok"]

    # (1) omit a WHOLE producer family: the FA2 hidden tap never fired -> coverage failure naming the family
    lay1 = {s: [m for m in ms if not m["name"].startswith("fa2.m1/")] for s, ms in lay.items()}
    r1 = coverage_check(man, lay1)
    assert not r1["ok"] and r1["families_missing_entirely"] == ["fa2_hidden_m1_stream"] and r1["missing_by_family"] == {"fa2_hidden_m1_stream": 6}

    # (2) omit one step (the last decode step never committed) -> coverage failure on exactly that step
    lay2 = {s: ms for s, ms in lay.items() if s != 2}
    r2 = coverage_check(man, lay2)
    assert not r2["ok"] and r2["steps_with_missing"] == [2] and r2["missing_n"] == 27 + 2 + 1 + 1

    # (3) drop output member `/1` of a norm and bind `/0` twice: same count, same module set -> coverage failure on `/1`
    lay3 = copy.deepcopy(lay)
    ms = lay3[1]
    k = next(i for i, m in enumerate(ms) if m["name"] == "model.layers.1.post_attention_layernorm/1")
    ms[k] = dict(ms[k], name="model.layers.1.post_attention_layernorm/0")
    r3 = coverage_check(man, lay3)
    assert not r3["ok"] and r3["missing_n"] == 1 and r3["missing"][0]["output_member"] == "1" and r3["missing"][0]["step"] == 1
    assert r3["committed_unmatched_n"] == 1                                    # the duplicate binds nothing required (invocation 1 of `/0` is not expected)

    # (4) a member committed SHORT (wrong element range: 3 of 4 prefill rows) -> coverage failure (containment)
    lay4 = copy.deepcopy(lay)
    m = next(m for m in lay4[0] if m["name"] == "model.layers.0.mlp.down_proj/0")
    m["numel"] -= 128
    r4 = coverage_check(man, lay4)
    assert not r4["ok"] and r4["missing"][0]["op_path"] == "model.layers.0.mlp.down_proj"

    # (5) omit one request of a B=2 manifest -> every identity of that request is missing; the other request is intact
    a = build_manifest(_program(), request_id="r0")
    b = build_manifest(_program(LP=3), request_id=1)
    g = merge_manifests([a, b])
    assert g["requests"] == ["r0", "r1"] and g["populations"]["by_family"]["weights"] == 2      # shared weights once
    assert g["complete"] is True and g["unmodelled"] == {}
    only_a = _layouts_from(a)
    ra = coverage_check(g, only_a, request_id="r0")
    assert ra["ok"]
    rb = coverage_check(g, only_a, request_id="r1")
    assert not rb["ok"] and rb["missing_n"] == rb["checked"] - 0 or rb["missing_n"] > 0

    # (6) omit one TP rank: [R15 tp TP-04] a TP1 Program can no longer be LABELLED rank 1 of 2 (build_manifest refuses result.tp != (W, r));
    # the rank negatives run on genuine rank Programs in test_required_manifest_tp.py (omit rank 1's partials, relabel, rank rows vs the
    # other rank's layouts, merge with a rank missing)
    import pytest
    with pytest.raises(ValueError, match="rank Program mismatch"):
        build_manifest(_program(), rank=1, ranks=2)
    from verity_capture.commit.tests.test_required_manifest_tp import _man
    r_rank = _man(1)
    assert all(r["rank"] == 1 for r in r_rank["identities"])
    assert coverage_check(r_rank, {}, rank=1)["missing_n"] == r_rank["populations"]["identities"] - r_rank["populations"]["by_family"]["weights"]


def test_request_id_is_r_k_never_the_workload_uuid():
    """[R15 COORD ruling 2 / F-rel-01] the key's request_id is `r<k>` (position in the workload manifest); the workload's UUID is recorded
    beside it, never used as the key (MAN-01 c41cae46a did, and the rc2 canary covered 303/12943: only the request-less weights matched)."""
    uuid = "df9c1ef5-8525-5b08-99ec-116584d5f55a"
    wl = {"requests": [{"request_id": uuid, "index": 0, "prompt_len": 4, "max_tokens": 3, "arrive_step": 0}],
          "sweep": {"row_id": "t", "batch": 1, "arrive_steps": [0]}}
    man = build_manifest(_program(), wl)
    assert man["requests"] == ["r0"] and all(r["request_id"] == "r0" for r in man["identities"] if r["family"] != "weights")
    assert man["workload"]["request_id"] == "r0" and man["workload"]["workload_request_id"] == uuid
    assert build_manifest(_program())["requests"] == ["r0"]                                 # bare Program: r0
    assert build_manifest(_program(), request_id=3)["requests"] == ["r3"]                   # int index -> r<k>
    assert build_manifest(_program(), request_id="2")["requests"] == ["r2"]
    assert build_manifest(_program(), wl, request_id=uuid)["requests"] == ["r0"]            # a workload id resolves THROUGH the manifest
    assert request_label(None, {"requests": [{"request_id": uuid, "index": 5}]}) == "r5"
    # negative: a label no binding map can carry is refused (fail closed), not emitted
    import pytest
    with pytest.raises(ValueError):
        build_manifest(_program(), request_id=uuid)                                         # UUID without a workload to resolve it
    with pytest.raises(ValueError):
        build_manifest(_program(), request_id="req-a")
    # negative (the F-rel-01 shape, kept as a regression): a manifest whose rows carry the UUID matches NOTHING under the r<k> filter
    bad = copy.deepcopy(man)
    for r in bad["identities"]:
        if r["family"] != "weights":
            r["request_id"] = uuid
    res = coverage_check(bad, _layouts_from(man), request_id="r0")
    assert res["checked"] == 0
    # the manifest digest carries the convention: r0 is not 0
    assert build_manifest(_program(), request_id="r0")["manifest_digest"] == man["manifest_digest"]
    # the global manifest: declaration UUIDs -> r<index>, UUID beside, Programs keyed either way
    u1, u2 = "1111aaaa-0000-5000-8000-000000000001", "2222bbbb-0000-5000-8000-000000000002"
    fixture = {"fixture": "t_b2", "requests": [{"request_id": u1, "index": 0, "prompt_len": 4, "max_tokens": 2, "arrive_step": 0},
                                                {"request_id": u2, "index": 1, "prompt_len": 3, "max_tokens": 3, "arrive_step": 1}]}
    g = build_global_manifest(fixture, {u1: _program(LP=4, T=1), "r1": _program(LP=3, T=2)}, workload_digest="w" * 64)
    assert g["requests"] == ["r0", "r1"] and g["complete"] is True
    assert [(d["request_id"], d["workload_request_id"]) for d in g["requests_declared"]] == [("r0", u1), ("r1", u2)]
    assert g["components"]["r1"]["workload_request_id"] == u2
    assert {r["request_id"] for r in g["identities"]} == {"r0", "r1", "*"}


@pytest.mark.parametrize("engine", ["inventory", "traversal"])
def test_a_global_manifests_rows_annotation_is_each_identitys_own_block_height_not_the_engine_steps_token_count(engine):
    """[R17 reqval C2 audit finding 1] the B>=2 fold used to stamp every identity of an engine step with the request's token count in the
    step: per-token members (LP at prefill, 1 at decode) were unchanged, but PER-SEQUENCE members -- logits_processor/out is (n_seq, V),
    one row per request -- carried `rows` = LP (245 on the Llama B8 record).  Digest-neutral (no consumer read it), false on the manifest.
    `rows` == element_range / width on every identity that carries a width, in both engines; the composition check's per-token reference
    (input_token_ids) still equals the engine step's member count."""
    fixture = {"fixture": "t_b2", "requests": [{"request_id": "a", "index": 0, "prompt_len": 4, "max_tokens": 2, "arrive_step": 0},
                                                {"request_id": "b", "index": 1, "prompt_len": 3, "max_tokens": 3, "arrive_step": 1}]}
    g = build_global_manifest(fixture, {"r0": _program(LP=4, T=1), "r1": _program(LP=3, T=2)}, workload_digest="w" * 64, engine=engine)
    logits = [r for r in g["identities"] if r["op_path"] == "logits_processor" and r["step"] == 0]
    assert len(logits) == 2 and all(r["rows"] == 1 and r["element_range"] == [0, r["width"]] for r in logits), logits
    per_token = [r for r in g["identities"] if r["op_path"].endswith("down_proj") and r["step"] == 0]
    assert sorted(r["rows"] for r in per_token) == [3, 3, 4, 4]                  # two layers x two requests: LP each at prefill
    for r in g["identities"]:
        if isinstance(r.get("width"), int) and r["element_range"][1] is not None:
            assert r["rows"] == (r["element_range"][1] - r["element_range"][0]) // r["width"], r
    for es in g["engine_steps"]:                                                 # the composition's per-request row counts stand (ids member)
        for m in es["members"]:
            ids = next(r for r in g["identities"] if r["family"] == "input_token_ids" and r["request_id"] == m["request_id"] and r["step"] == m["step"])
            assert ids["rows"] == m["rows"]


def test_man02_zero_width_unmodelled_biasadd():
    """[MAN-02] (a) never a zero-width identity; (b) a family without a rule -> complete:false, coverage refuses with the family named;
    (c) BiasAdd under a biased linear (Qwen2 qkv_proj) is the module's single bound output -> no separate identity."""
    # (c) positive: a Qwen2-shaped qkv_proj (Gemm -> BiasAdd under one module) yields exactly one identity for that module per step
    prog = _program()
    rows = prog["instances"]["rows"]
    extra = []
    for r in rows:
        if r["name"].startswith("model.model.layers.0.self_attn.qkv_proj/") and "[" in r["name"]:
            extra.append((r["i"], {"spec": "BiasAdd_v1{N=256}", "form": "call", "name": r["name"].replace("triton_launch_1", "bias_add"), "args": []}))
    out = []
    for r in rows:
        out.append(r)
        for i, e in extra:
            if i == r["i"]:
                out.append(dict(e))
    for k, r in enumerate(out):
        r["i"] = k
    prog["instances"]["rows"] = out
    man = build_manifest(prog)
    assert man["complete"] is True and man["unmodelled"] == {}
    qkv = [r for r in man["identities"] if r["op_path"] == "model.layers.0.self_attn.qkv_proj" and r["step"] == 0]
    assert [(r["output_member"], r["element_range"]) for r in qkv] == [("0", [0, 4 * 256])]
    # [MAN-07, exec M-0528] the member's PRODUCER is the BiasAdd row (the tensor the module returns); the Gemm row is interior
    assert all(r["spec"].startswith("BiasAdd_v1") for r in qkv), [r["spec"] for r in qkv]
    assert all(r["spec"].startswith("Gemm_v1") for r in man["identities"] if r["op_path"].endswith("o_proj") and r["step"] == 0)
    # rev-cb D2: an epilogue whose width differs from its Gemm's is a Program defect -> incomplete, named; never a silent claim
    progw = _program()
    rows = progw["instances"]["rows"]
    out, k = [], 0
    for r in rows:
        out.append(r)
        if r["name"].startswith("model.model.layers.0.self_attn.qkv_proj/") and "[" in r["name"]:
            out.append({"spec": "BiasAdd_v1{N=128}", "form": "call", "name": r["name"].replace("triton_launch_1", "bias_add"), "args": []})
    for k, r in enumerate(out):
        r["i"] = k
    progw["instances"]["rows"] = out
    mw = build_manifest(progw)
    assert mw["complete"] is False and any("!= its BiasAdd epilogue width 128" in key for key in mw["unmodelled"]), mw["unmodelled"]
    assert all(r["element_range"][1] is None or r["element_range"][1] > r["element_range"][0] for r in man["identities"])
    # (a) a rule resolving to zero elements (Gemm with N=0) is dropped AND makes the manifest incomplete -- never `[0, 0]`
    prog0 = _program()
    for r in prog0["instances"]["rows"]:
        if r["name"].startswith("model.model.layers.1.mlp.down_proj/"):
            r["spec"] = "Gemm_v1{K=128,N=0}"
    m0 = build_manifest(prog0)
    assert m0["complete"] is False and any("width 0" in k for k in m0["unmodelled"])
    assert not any(r["element_range"] == [0, 0] for r in m0["identities"])
    assert not any(r["op_path"] == "model.layers.1.mlp.down_proj" for r in m0["identities"])
    # (b) an unknown family: incomplete, named; coverage_check refuses even a collector that committed everything else
    progu = _program()
    for r in progu["instances"]["rows"]:
        if r["name"].startswith("model.model.layers.1.mlp.act_fn/"):
            r["spec"] = "GeGLU_v9{I=128}"
    mu = build_manifest(progu)
    assert mu["complete"] is False and list(mu["unmodelled"]) == ["GeGLU_v9 under model.layers.1.mlp.act_fn"] and mu["unmodelled"]["GeGLU_v9 under model.layers.1.mlp.act_fn"] == 6
    res = coverage_check(mu, _layouts_from(mu))
    assert res["ok"] is False and "GeGLU_v9 under model.layers.1.mlp.act_fn" in res["error"] and res["families_missing_entirely"] == ["GeGLU_v9 under model.layers.1.mlp.act_fn"]
    # the incompleteness propagates through the global / merged manifests
    g = build_global_manifest({"fixture": "t", "requests": [{"request_id": "r0", "index": 0, "prompt_len": 4, "max_tokens": 3, "arrive_step": 0}]}, {"r0": progu})
    assert g["complete"] is False and not coverage_check(g, {})["ok"] and "GeGLU_v9" in coverage_check(g, {})["error"]
    assert merge_manifests([build_manifest(_program(), request_id=0), build_manifest(progu, request_id=1)])["complete"] is False


def test_fp8_block_linears_are_required_once_as_their_bf16_output():
    """[R15 fp8] a blockwise-FP8 linear is Fp8GroupQuant_v1 -> ScaledMmFp8Block_v1 under ONE LinearBase module: the manifest requires the
    module's bf16 output ("0", width N) exactly once -- the same identities as the bf16 twin (Gemm_v1) -- and stays complete (no unmodelled
    family); the e4m3 words / f32 group scales are interior (Match's), never a required value.  Dropping one FP8 linear output is detected."""
    prog = _program()
    rows = []
    for r in prog["instances"]["rows"]:
        spec, name = r["spec"], r["name"]
        if spec.startswith("Gemm_v1") and "logits_processor" not in name:
            st = spec_statics(spec)
            mod, tail = name.split("/", 1)
            sfx = tail.split("[", 1)[1]
            rows.append(dict(r, spec=f"Fp8GroupQuant_v1{{K={st['K']},G=128}}", name=f"{mod}/per_token_group_fp8_quant[{sfx}"))
            rows.append(dict(r, spec=f"ScaledMmFp8Block_v1{{K={st['K']},N={st['N']},G=128}}", name=f"{mod}/cutlass_scaled_mm[{sfx}"))
        else:
            rows.append(dict(r))
    for i, r in enumerate(rows):
        r["i"] = i
    fp8 = copy.deepcopy(prog)
    fp8["instances"]["rows"] = rows
    man_bf16 = build_manifest(prog)
    man = build_manifest(fp8)
    assert man["complete"] and not man["unmodelled"]
    assert man["populations"]["identities"] == man_bf16["populations"]["identities"]
    ids = lambda m: sorted((r["step"], r["op_path"], r["output_member"], tuple(r["element_range"])) for r in m["identities"] if r["family"] != "weights")
    assert ids(man) == ids(man_bf16)
    lay = _layouts_from(man)
    assert coverage_check(man, lay)["ok"]
    lay1 = {s: [m for m in ms if m["name"] != "model.layers.1.mlp.down_proj/0"] for s, ms in lay.items()}
    r1 = coverage_check(man, lay1)
    assert not r1["ok"] and r1["missing_n"] == 3 and all("down_proj" in x["op_path"] for x in r1["missing"])


def test_family_rules_gelu_tanh_mul_and_sampler_selectors():
    """[R15 manbind] generic family rules added after the panel sweep: GeGLU (`GeluTanhMul_v1{I}`, Gemma-2 `mlp.act_fn` = vLLM GeluAndMul)
    is one [rows, I] output like SiluMul; the stochastic / explicit selectors (`GumbelTokenSelect_v1{V}`, `ArgmaxTokenSelect_v1{V}`) end an
    engine step and require one sampled id like `TokenSelect_v1` (a stochastic Program must not collapse into ONE step)."""
    prog = _program()
    for r in prog["instances"]["rows"]:
        if r["name"].startswith("model.model.layers.1.mlp.act_fn/"):
            r["spec"] = "GeluTanhMul_v1{I=128}"
        if r["spec"].startswith("TokenSelect_v1"):
            r["spec"] = "GumbelTokenSelect_v1{V=64}"
    man = build_manifest(prog)
    assert man["complete"] is True and man["steps"] == [0, 1, 2]                                      # 3 steps, not 1
    act = [r for r in man["identities"] if r["op_path"] == "model.layers.1.mlp.act_fn" and r["step"] == 0]
    assert [(r["output_member"], r["element_range"], r["spec"]) for r in act] == [("out", [0, 4 * 128], "GeluTanhMul_v1")]
    sid = [r for r in man["identities"] if r["family"] == "sampled_token_ids"]
    assert len(sid) == 3 and all(r["spec"] == "GumbelTokenSelect_v1" and r["element_range"] == [0, 1] for r in sid)
    assert coverage_check(man, _layouts_from(man))["ok"]
    # negative: zero-width GeGLU (I=0) is dropped and the manifest is incomplete; a sampler that never emitted ids is a missing family
    prog0 = _program()
    for r in prog0["instances"]["rows"]:
        if r["name"].startswith("model.model.layers.1.mlp.act_fn/"):
            r["spec"] = "GeluTanhMul_v1{I=0}"
    m0 = build_manifest(prog0)
    assert m0["complete"] is False and not any(r["element_range"] == [0, 0] for r in m0["identities"])
    lay = {s: [m for m in ms if m["name"] != "runner.sampler/sampled_token_ids"] for s, ms in _layouts_from(man).items()}
    r = coverage_check(man, lay)
    assert not r["ok"] and r["families_missing_entirely"] == ["sampled_token_ids"]


def test_stochastic_selector_requires_the_seed_and_segments_steps():
    """[R15 stoch M-0107] `GumbelTopPTokenSelect_v1{V}` (temperature -> top-p -> Gumbel(seed,pos) -> argmax) is a sampling event: it closes the
    engine step like TokenSelect_v1 AND requires the per-request seed (`runner.sampler/seed`, family sampler_seed, one i64 per sequence per
    step) beside the sampled id.  Greedy Programs emit no seed identity.  Negative: a collector that never binds the seed fails coverage
    naming the family (never a silent pass)."""
    prog = _program()
    for r in prog["instances"]["rows"]:
        if r["spec"].startswith("TokenSelect_v1"):
            r["spec"] = "GumbelTopPTokenSelect_v1{V=64}"
    man = build_manifest(prog)
    assert man["complete"] is True and man["steps"] == [0, 1, 2]
    seeds = [r for r in man["identities"] if r["family"] == "sampler_seed"]
    assert [(r["step"], r["op_path"], r["output_member"], r["element_range"], r["spec"]) for r in seeds] == \
        [(s, "runner.sampler", "seed", [0, 1], "GumbelTopPTokenSelect_v1") for s in (0, 1, 2)]
    assert "sampler_seed" in man["query"]["required_families"] and man["query"]["stochastic_selectors"] == ["GumbelTopPTokenSelect_v1"]
    assert len([r for r in man["identities"] if r["family"] == "sampled_token_ids"]) == 3
    lay = _layouts_from(man)
    assert coverage_check(man, lay)["ok"]
    no_seed = {s: [m for m in ms if m["name"] != "runner.sampler/seed"] for s, ms in lay.items()}
    r = coverage_check(man, no_seed)
    assert not r["ok"] and r["families_missing_entirely"] == ["sampler_seed"] and r["missing_by_family"] == {"sampler_seed": 3}
    greedy = build_manifest(_program())
    assert not any(r["family"] == "sampler_seed" for r in greedy["identities"]) and "sampler_seed" not in greedy["query"]["required_families"]


def test_extra_committed_data_is_allowed():
    man = build_manifest(_program())
    lay = _layouts_from(man)
    lay[0].append({"name": "model.layers.1.post_attention_layernorm/row4_f32", "numel": 4 * 128})    # regenerate baseline: extra, not required
    lay[0].append({"name": "model.layers.1.self_attn.qkv_proj/0", "numel": 4 * 256})                # a second physical copy of a bound value
    r = coverage_check(man, lay)
    assert r["ok"] and r["committed_extras_n"] == 1 and r["committed_unmatched_n"] == 1


def _global_layouts(g: dict, drop_request: str | None = None) -> dict[int, list[dict]]:
    """Packed per-engine-step layouts a collector would commit for the global fixture: one member per module with all co-scheduled rows;
    `drop_request` = a request whose rows the server never ran (its row block missing -> members shorter, one sampled id fewer)."""
    lay: dict[int, list[dict]] = {}
    for es in g["engine_steps"]:
        e = es["engine_step"]
        members = [m for m in es["members"] if m["request_id"] != drop_request]
        if not members:
            continue
        n_rows = sum(m["rows"] for m in members)
        rows_by_rid = {m["request_id"]: m for m in members}
        ms: dict[tuple, dict] = {}
        for r in g["identities"]:
            if r.get("engine_step") != e or r["family"] == "weights":
                continue
            op, mem = r["op_path"], r["output_member"]
            name = op if mem == "out" else f"{op}/{mem}"
            if r["family"].endswith("hidden_m1_stream"):
                ms[(name, 0)] = {"name": name, "numel": r["element_range"][1], "inv": 0}
            elif r["family"] == "sampled_token_ids":
                ms[(name, 0)] = {"name": name, "numel": len(members), "inv": 0}
            elif r["request_id"] in rows_by_rid:
                ms[(name, r["invocation"])] = {"name": name, "numel": n_rows * int(r["width"]), "inv": r["invocation"]}
        lay[e] = sorted(ms.values(), key=lambda m: (m["name"], m["inv"]))
    return lay


def test_global_manifest_staggered_b3():
    fixture = {"fixture": "t_b3", "requests": [{"request_id": "r0", "index": 0, "prompt_len": 4, "max_tokens": 2, "arrive_step": 0},
                                                {"request_id": "r1", "index": 1, "prompt_len": 3, "max_tokens": 3, "arrive_step": 1},
                                                {"request_id": "r2", "index": 2, "prompt_len": 5, "max_tokens": 2, "arrive_step": 3}]}
    progs = {"r0": _program(LP=4, T=1), "r1": _program(LP=3, T=2), "r2": _program(LP=5, T=1)}
    g = build_global_manifest(fixture, progs, workload_digest="w" * 64)
    assert g["steps"] == [0, 1, 2, 3, 4] and g["populations"]["requests"] == 3
    e1 = next(es for es in g["engine_steps"] if es["engine_step"] == 1)
    assert [(m["request_id"], m["step"], m["rows"]) for m in e1["members"]] == [("r0", 1, 1), ("r1", 0, 3)]     # decode r0 + prefill r1 co-scheduled
    # [R15 B8] request-LOCAL addressing: each request's identity is its own row block [0, rows x width) of the packed member (its position
    # is the engine's row order, bound per map entry as `element_offset`, never declared); the sampled id is [0, 1) of the request's slot.
    # The declared composition (who, how many rows) is `engine_steps[e]` -- the binding map's composition_check holds the map to it.
    assert g["addressing"] == "request-local" and e1["rows"] == 4 and e1["n_seq"] == 2
    q = [r for r in g["identities"] if r.get("engine_step") == 1 and r["op_path"] == "model.layers.1.self_attn.qkv_proj"]
    assert sorted((r["request_id"], tuple(r["element_range"]), r["rows"]) for r in q) == [("r0", (0, 256), 1), ("r1", (0, 768), 3)]
    sid = [r for r in g["identities"] if r.get("engine_step") == 1 and r["family"] == "sampled_token_ids"]
    assert sorted((r["request_id"], tuple(r["element_range"])) for r in sid) == [("r0", (0, 1)), ("r1", (0, 1))]
    assert g["composition_reference"]["input_token_ids"]["op_path"] == "model.embed_tokens" and g["composition_reference"]["sampled_token_ids"]
    assert g["program_digest_kind"] == "workload" and g["component_digests"] == sorted({p["program_digest"] for p in g["components"].values()})
    assert all(hi > lo for lo, hi in (r["element_range"] for r in g["identities"] if r["element_range"][1] is not None))    # MAN-02: no zero width
    # one hidden launch per layer per engine step over both sequences (HB = 2 seq x 2 heads, M = max rows 3)
    h = [r for r in g["identities"] if r.get("engine_step") == 1 and r["family"] == "fa2_hidden_m1_stream" and r["op_path"] == "fa2.m1/L0"]
    assert len(h) == 1 and h[0]["requests"] == ["r0", "r1"] and h[0]["output_member"] == "g4x3x1" and h[0]["request_id"] == "*"
    assert g["populations"]["by_family"]["weights"] == 2 and all(r["program_digest"] == "w" * 64 for r in g["identities"])
    assert build_global_manifest(fixture, progs, workload_digest="w" * 64)["manifest_digest"] == g["manifest_digest"]
    # positive: packed layouts of every engine step cover every request's identities
    ok = coverage_check(g, _global_layouts(g))
    assert ok["ok"], ok["missing"][:3]
    # negative: the server never ran r1 -> r1's row blocks / sampled ids / hidden geometry are uncovered; r0 and r2 stay covered; the
    # layouts check also names the composition mismatch (packed ids member / sampled slots shorter than declared) per engine step
    bad = coverage_check(g, _global_layouts(g, drop_request="r1"))
    assert not bad["ok"] and bad["missing_n"] > 0
    assert bad["composition_mismatch_n"] > 0 and bad["composition_mismatch"][0]["engine_step"] == 1 and "r1" in bad["composition_mismatch"][0]["requests"]
    # negative: a request with the SAME row count as another dropped in a decode-only step (rows equal -> only the composition sees it)
    g2 = build_global_manifest({"fixture": "t_b2", "requests": [{"request_id": "a", "index": 0, "prompt_len": 4, "max_tokens": 3, "arrive_step": 0},
                                                                {"request_id": "b", "index": 1, "prompt_len": 4, "max_tokens": 3, "arrive_step": 0}]},
                               {"r0": _program(LP=4, T=2), "r1": _program(LP=4, T=2)}, workload_digest="w" * 64)
    lay3 = _global_layouts(g2)
    for m in lay3[2]:                                            # engine step 2: both decode (1 row each); one request never ran -> every per-token member
        if not m["name"].startswith("fa2.m1/"):                  # and the sampled slots are half as long (the hidden launch geometry is left alone)
            m["numel"] //= 2
    r3 = coverage_check(g2, lay3)
    assert not r3["ok"] and r3["composition_mismatch_n"] >= 1 and r3["composition_mismatch"][0]["engine_step"] == 2, r3["composition_mismatch"]
    assert coverage_check(g, _global_layouts(g, drop_request="r1"), request_id="r0")["missing_by_family"].get("instance_outputs") is None
    assert not coverage_check(g, _global_layouts(g, drop_request="r1"), request_id="r1")["ok"]
    # negative: a whole engine step missing (r2's last decode never committed)
    lay = _global_layouts(g); lay.pop(4)
    r = coverage_check(g, lay)
    assert not r["ok"] and r["steps_with_missing"] == [1] or r["missing_n"] > 0


def test_fresh_b1_manifest_loads_in_binding_and_composition_skips(tmp_path):
    """rel M-0213/M-0245 (r15-int-2 canary #1 and #2 RED): a FRESH B=1 `build` manifest writes engine_steps as the request's engine-step
    NUMBERS (ints); binding.load_required_manifest must keep them as numbers and declare no composition (composition_check skips, named)
    -- the rc3 canary manifests had no engine_steps key at all, which is why the crash was only seen on the row procedure."""
    import json

    from verity_capture.commit import binding as B

    man = build_manifest(_program(), {"sweep": {"row_id": "t", "batch": 1, "arrive_steps": [0]}})
    assert man["engine_steps"] and all(isinstance(e, int) for e in man["engine_steps"]), man["engine_steps"]
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(man))
    pop, hdr = B.load_required_manifest(str(p), man["program_digest"], requests=["r0"], steps=len(man["steps"]), ranks=[0], arrivals={"r0": 0})
    assert len(pop) > 0 and hdr["engine_steps"] == [] and hdr["engine_step_numbers"] == man["engine_steps"] and hdr["global"] is False
    comp = B.composition_check(hdr, {"entries": []})
    assert comp["ok"] and comp["checked_steps"] == 0 and "no engine-step composition" in comp["skipped"]
    # a fresh B3 GLOBAL manifest declares the dict composition and the loader keeps it (rows / n_seq beside)
    fx = {"requests": [{"request_id": "A", "prompt_len": 4, "max_tokens": 3, "arrive_step": 0}, {"request_id": "B", "prompt_len": 4, "max_tokens": 3, "arrive_step": 1}]}
    g = build_global_manifest(fx, {"A": _program(), "B": _program()})
    pg = tmp_path / "global.json"
    pg.write_text(json.dumps(g))
    popg, hdrg = B.load_required_manifest(str(pg), g["program_digest"], requests=["r0", "r1"], steps=4, ranks=[0], arrivals={"r0": 0, "r1": 1})
    assert hdrg["global"] and len(hdrg["engine_steps"]) == len(g["engine_steps"]) and hdrg["engine_step_numbers"] == [e["engine_step"] for e in g["engine_steps"]]
    assert all(isinstance(e["members"], list) and e["rows"] is not None for e in hdrg["engine_steps"])
    # a manifest mixing the two shapes is refused by name
    bad = dict(man, engine_steps=[0, {"engine_step": 1, "members": []}])
    pb = tmp_path / "bad.json"
    pb.write_text(json.dumps(bad))
    try:
        B.load_required_manifest(str(pb), man["program_digest"], requests=["r0"], steps=2, ranks=[0], arrivals={"r0": 0})
    except ValueError as e:
        assert "engine_steps mixes" in str(e)
    else:
        raise AssertionError("mixed engine_steps accepted")


def test_members_for_extras_carry_producer_ordinal_onto_the_row(monkeypatch):
    """[R15 manbind / dense M-0254 M-0270] a family rule may return (member, width, dtype, {extras}); `producer_ordinal` lands on the manifest
    row (additive; the identity key is unchanged) and oracle_compare.producers_of reads it for the C2 oracle.  Rows of one member that
    disagree on their extras are refused."""
    from verity_capture.commit import required_manifest as RM
    from verity_capture.commit import oracle_compare as OC
    orig = RM.members_for

    def rule(family, statics, module, hint, **kw):                                 # [rel int-4] TP-04 passes tp=
        if family.startswith("RMSNormFusedCuda"):
            n = int(statics.get("N", 0))
            return [("0", n, "bf16", {"producer_ordinal": 1}), ("1", n, "bf16", {"producer_ordinal": 0})]
        return orig(family, statics, module, hint, **kw)

    from verity_capture.commit import required_values as RV
    monkeypatch.setattr(RM, "members_for", rule)
    monkeypatch.setattr(RV, "members_for", rule)                                    # [R17 task 3] the traversal binds the vocabulary by name at import
    fx = {"requests": [{"request_id": "r0", "prompt_len": 4, "max_tokens": 2, "arrive_step": 0}]}
    for engine in RM.ENGINES:                                                      # the SAME extras land on the row from either engine
        man = RM.build_manifest(_program(), fx, engine=engine)
        assert man["complete"] is True
        fused = [r for r in man["identities"] if str(r.get("spec") or "").startswith("RMSNormFusedCuda")]
        assert fused and {(r["output_member"], r["producer_ordinal"]) for r in fused} == {("0", 1), ("1", 0)}, engine
        assert all("producer_ordinal" not in r for r in man["identities"] if r.get("spec") in ("Gemm_v1", "RMSNormTriton_v1"))
        assert list(man["identity_key"]) == list(RM.IDENTITY_KEY)                # additive only
        prod = OC.producers_of(man)
        assert prod[("model.layers.1.input_layernorm", "0")] == {"spec": "RMSNormFusedCuda_v2", "ordinal": 1, "operand": None}
        assert prod[("model.layers.1.input_layernorm", "1")] == {"spec": "RMSNormFusedCuda_v2", "ordinal": 0, "operand": None}
    assert prod[("model.layers.0.self_attn.qkv_proj", "0")] == {"spec": "Gemm_v1", "ordinal": None, "operand": None}

    # rows of one member disagreeing on the ordinal: refused, named
    calls = {"n": 0}

    def flaky(family, statics, module, hint, **kw):
        if family.startswith("RMSNormFusedCuda"):
            calls["n"] += 1
            return [("0", int(statics["N"]), "bf16", {"producer_ordinal": calls["n"] % 2})]
        return orig(family, statics, module, hint, **kw)

    monkeypatch.setattr(RM, "members_for", flaky)
    monkeypatch.setattr(RV, "members_for", flaky)
    import pytest
    with pytest.raises(ValueError, match="disagree on their extras"):
        RM.build_manifest(_program(), fx, engine="inventory")
    # the traversal binds the vocabulary ONCE per claim (module x family x step; the rows of a claim share their statics), so a rule that
    # answers differently per CALL cannot make two rows of one claim disagree -- its refusal (required_values.py, the same text) is for two
    # claims landing on one member; a per-row flip is unreachable there.  Same identities, same ordinals as the inventory under `rule` above.
    calls["n"] = 0
    RM.build_manifest(_program(), fx, engine="traversal")


def test_verify_cli_pins_the_manifest_of_record_to_the_program(tmp_path):
    """[rev-cb M-0334 F-cb-11] `required_manifest verify`: the manifest the Commit used must equal the one REBUILT from the Build's Program x
    workload -- a weakened manifest with an honest digest (one row removed, re-digested) is refused as 'not the manifest of record'."""
    import gzip
    import json
    import subprocess
    import sys
    from verity_capture.commit import required_manifest as RM
    prog = _program()
    d = tmp_path / "build_request"
    d.mkdir()
    with gzip.open(d / "instances.json.gz", "wt") as f:
        json.dump(prog["instances"], f)
    (d / "result.json").write_text(json.dumps(prog["result"]))
    (d / "artifact.json").write_text(json.dumps(prog["artifact"]))
    wl = {"requests": [{"request_id": "r0", "prompt_len": 4, "max_tokens": 3, "arrive_step": 0}]}
    (tmp_path / "wl.json").write_text(json.dumps(wl))
    man = RM.build_manifest(RM.load_program(str(d)), wl)
    (tmp_path / "manifest.json").write_text(json.dumps(man))

    def run(path):
        r = subprocess.run([sys.executable, "-m", "verity_capture.commit.required_manifest", "verify", "--manifest", str(path), "--program", str(d),
                            "--workload", str(tmp_path / "wl.json"), "--out", str(tmp_path / "verify.json")], capture_output=True, text=True)
        return r.returncode, json.loads((tmp_path / "verify.json").read_text()), r.stdout
    rc, rec, out = run(tmp_path / "manifest.json")
    assert rc == 0 and rec["ok"] is True and rec["used_digest"] == rec["rebuilt_digest"] and "VERIFY OK" in out
    # weakened, honest digest: one identity dropped and the digest recomputed over the remaining rows
    weak = copy.deepcopy(man)
    weak["identities"] = [r for r in weak["identities"] if not (r["op_path"].endswith("layers.1.mlp.down_proj") and r["step"] == 0)]
    weak["manifest_digest"] = RM.manifest_digest(weak["identities"])
    (tmp_path / "weak.json").write_text(json.dumps(weak))
    rc, rec, out = run(tmp_path / "weak.json")
    assert rc == 1 and rec["ok"] is False and "not the manifest of record" in rec["why"] and rec["used_identities"] == rec["rebuilt_identities"] - 1
    # edited rows, stale digest
    stale = copy.deepcopy(man)
    stale["identities"][0]["element_range"] = [0, 1]
    (tmp_path / "stale.json").write_text(json.dumps(stale))
    rc, rec, out = run(tmp_path / "stale.json")
    assert rc == 1 and "edited after digesting" in rec["why"]


def test_const_rows_are_structure_not_identities():
    """[stoch M-0379 / manbind M-0395] a stochastic Program's Const32[top_p]/Const64[pos]/Const32[temperature] rows carry the DECLARED policy
    inside their spec (hence inside the Program digest): no identity, never a zero-width one, never an unmodelled family."""
    from verity_capture.commit import required_manifest as RM
    for spec in ("Const32[top_p]_v1{V=0x3f733333}", "Const64[pos]_v1{V=17}", "Const32[temperature]_v1{V=0x3f4ccccd}"):
        assert RM.members_for(RM.spec_family(spec), RM.spec_statics(spec), "runner.sampler", "") == []


def test_man08_unrooted_module_path_is_incomplete_named_not_one_folded_identity():
    """[MAN-08, moe M-0583] a Program that names an identity-bearing row under a BARE module path (`gate/triton_launch_k`: vLLM 0.28's
    MoE runner calls the router so nn_module_stack loses `model.layers.<L>.mlp`) must not fold 48 routers into one `gate/0` identity of
    element_range [0, 48*V*T): incomplete, named with the path.  Sampler rows keep their bare `argmax_k` names (not an identity op_path)."""
    prog = _program()
    for r in prog["instances"]["rows"]:
        if r["name"].startswith("model.model.layers.1.mlp.down_proj/"):
            r["name"] = "gate/" + r["name"].split("/", 1)[1]                    # the router Gemm exported under a bare leaf
    man = build_manifest(prog)
    assert man["complete"] is False
    assert any("unrooted module path 'gate'" in k for k in man["unmodelled"]), man["unmodelled"]
    assert not any(r["op_path"] == "gate" for r in man["identities"])
    assert any(r["op_path"] == "runner.sampler" for r in man["identities"])       # the bare-named sampler rows are still the sampler identity
    ok = build_manifest(_program())
    assert ok["complete"] is True                                                 # every module of the dense Program is rooted


# ---------------------------------------------------------------------------------------------------------------------------------
# the registry not importable is an ENVIRONMENT fault (RegistryUnavailable), never a manifest with `unmodelled ... registry is not importable`
# ---------------------------------------------------------------------------------------------------------------------------------

def test_registry_unavailable_is_a_named_fault_not_a_manifest_result(monkeypatch, capsys):
    """Simulate the core `verity` package missing (sys.modules["verity.ir.defs"] = None makes its import raise): the builder must raise
    RegistryUnavailable (infra/precondition, PRECHECK_FAIL_REGISTRY_UNAVAILABLE) instead of deriving `(None, None, 'registry is not
    importable')` and emitting complete:false; the CLI maps it to exit 3 with the fail class on stderr; nothing is cached as None so a
    repaired environment in the same process recovers."""
    import sys

    from verity_capture.commit import manifest_format as MF
    from verity_capture.commit import required_manifest as RM

    monkeypatch.setattr(RM, "_registry_cache", {})
    monkeypatch.setitem(sys.modules, "verity.ir.defs", None)
    with pytest.raises(RM.RegistryUnavailable) as ei:
        RM.promoted_signature("Bf16MulScalarTensor_v1{N=8}")
    e = ei.value
    assert isinstance(e, ImportError) and e.fail_class == MF.REGISTRY_UNAVAILABLE_FAIL_CLASS == "PRECHECK_FAIL_REGISTRY_UNAVAILABLE"
    assert e.kind == "infra/precondition" and "verity.ir.defs" in str(e) and str(e).startswith("PRECHECK_FAIL_REGISTRY_UNAVAILABLE")
    assert "registry" not in RM._registry_cache                                  # not cached as None
    # the v2 query engine names the same fault
    from verity_vllm.query import program_view as PV
    monkeypatch.setattr(PV, "_REGISTRY", {})
    with pytest.raises(RM.RegistryUnavailable):
        PV._registry()
    # the CLI: exit REGISTRY_UNAVAILABLE_EXIT (row_pod.sh's PRECHECK_FAIL_* status), the class named on stderr, no manifest
    monkeypatch.setattr(RM, "_main", lambda: RM._registry() and 0)
    assert RM.main() == RM.REGISTRY_UNAVAILABLE_EXIT == 3
    assert "PRECHECK_FAIL_REGISTRY_UNAVAILABLE" in capsys.readouterr().err
    # repaired environment (the real module back): the registry loads and the signature derives
    monkeypatch.delitem(sys.modules, "verity.ir.defs", raising=False)
    monkeypatch.setattr(RM, "_registry_cache", {})
    reg, _errors = RM._registry()
    assert reg is not None
