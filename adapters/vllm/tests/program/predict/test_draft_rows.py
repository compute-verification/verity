"""verity_vllm.program.predict.draft_rows: a launch that decodes speculatively compiles to request Programs whose decode
tokens come in passes of a token and its drafts, each pass's kept count read off the advice's rows and checked against the
accept; the accept and the draft layer are reached by declared roles; and what the passes don't hold is refused by name, a
quantized KV cache first.
"""
from __future__ import annotations

import re
from functools import cache

import numpy as np
import pytest
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.evaluation.batch import type_dtype, type_shape
from verity.circuits.evaluation.reference import signature_of
from verity.definitions.registry import spec
from verity_vllm.pipeline import advice as ADV
from verity_vllm.pipeline import window_advice as WA
from verity_vllm.program import inputs
from verity_vllm.program.predict import draft_rows as DR
from verity_vllm.program.predict import phi as PHI
from verity_vllm.program.predict.build import Unexpressible, build_program
from verity_vllm.program.predict.domain import cache_refusal
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import Registered, compile_steps

pytestmark = pytest.mark.usefixtures("code_versions_stood_in")

RTXPRO6000 = {"compute_capability": [12, 0], "num_sms": 188}
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
        "max_position_embeddings": 64, "hidden_act": "silu", "eos_token_id": 255}
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": None}
STOCH = {"temperature": 0.8, "top_p": 1.0, "top_k": 0, "seed": 11, "min_p": 0.0, "repetition_penalty": 1.0}
#: the two served requests: (id, prompt, outputs kept)
SERVED = (("a", (1, 8, 15, 22), 12), ("b", (40, 3, 77), 9))
MODELOPT = {"quant_method": "modelopt", "quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"}}
R, St, Row = WA.RequestAdvice, WA.StepAdvice, WA.Row


def _phi(K: int | None = None, cfg: dict = TINY, target: dict = RTXPRO6000, **kw) -> Phi:
    engine = {**({} if K is None else {DR.ENGINE_ARG: K}), **kw.pop("engine", {})}
    return Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, target=dict(target), **({"engine": engine} if engine else {}), **kw)


def _registered(K: int | None = None, layer: DR.StandIn | None = None, **kw) -> Registered:
    return Registered.of(_phi(K, **kw), 7, draft_layer=layer)


def _walk(fn, args: list) -> list[int]:
    """A Program's output leaves on ``args`` (one flat integer array a parameter), node by node with each callee's kernel."""
    vals = {("p", i): np.asarray(a, dtype=np.uint64) for i, a in enumerate(args)}
    for k, node in enumerate(fn.body.nodes):
        params, _ = signature_of(node.fn)
        operands = [np.array([vals[s, i][j] for s, i, j in c.refs], dtype=np.uint64).astype(type_dtype(t)).reshape((1, *type_shape(t)))
                    for c, t in zip(node.args, params)]
        vals["n", k] = np.asarray(evaluate_batch(node.fn, *operands)).reshape(-1).astype(np.uint64)
    return [int(vals[s, i][j]) for s, i, j in fn.body.ret]


def _weights(fn) -> np.ndarray:
    """Finite bf16 weights for a TINY request Program, the same for every Program of the model."""
    w = np.random.default_rng(3).integers(0, 1 << 16, fn.signature()[0][1][1].leaves, dtype=np.uint64)
    return (w & 0x8000) | ((w & 0x3FFF) | 0x3800) & 0x7FFF


@cache
def _truth() -> dict[str, tuple[int, ...]]:
    """What the model outputs for each served request: its plain request Program's tokens, one token a decode row."""
    out = {}
    for rid, prompt, n in SERVED:
        fn = build_program(Phi.from_json({**_phi().to_json(), "sampling": {"kind": "greedy"}}), "request", LP=len(prompt), T=n - 1, C=0,
                           dead_free=True, with_desc=False).program.fn
        out[rid] = tuple(_walk(fn, [np.array(prompt), _weights(fn)]))
    return out


def _layer(K: int) -> DR.StandIn:
    """A draft layer's table for the served requests: after a request's first output its next K, right; after its fifth one
    right draft and then wrong ones; after its seventh wrong ones; a token already in the table keeps its row."""
    table: dict[int, tuple[int, ...]] = {}
    for rid, _, n in SERVED:
        ids = _truth()[rid]
        for at, right in ((0, K), (4, 1), (6, 0)):
            after = [ids[at + 1 + j] if at + 1 + j < n else 0 for j in range(K)]
            table.setdefault(ids[at], tuple(x if j < right else (x + 1) % 256 for j, x in enumerate(after)))
    return DR.StandIn(tuple(sorted(table.items())))


def _trace(K: int, layer: DR.StandIn, *, header: dict | None = None) -> tuple[list[dict], dict, dict[str, list[int]]]:
    """A step trace of the served requests under K drafts a decode row, as the sync scheduler runs them (request b arrives a
    step late), the Commit record's requests, and each request's kept counts: each decode row is the request's last token and
    the layer's K drafts, it emits the drafts the model's own tokens agree with and one more, cut at the request's length,
    and the next row starts past them."""
    truth = _truth()
    recs: list[dict] = [{"kind": "header", "config": {"speculative": {"method": "mtp", "num_speculative_tokens": K}, **(header or {})}}]
    known = {rid: 0 for rid, _, _ in SERVED}
    kept: dict[str, list[int]] = {rid: [] for rid, _, _ in SERVED}
    step = 0
    while any(known[rid] < n for rid, _, n in SERVED):
        rows, emitted, finished = [], {}, {}
        for k, (rid, prompt, n) in enumerate(SERVED):
            if step < k or known[rid] == n:
                continue
            if step == k:
                recs.append({"kind": "arrive", "request": rid, "prompt_len": len(prompt)})
                rows.append([rid, 0, len(prompt)])
                emitted[rid] = 1
            else:
                drafts, c = layer.propose(truth[rid][known[rid] - 1]), 0
                while c < K and known[rid] + c < n - 1 and drafts[c] == truth[rid][known[rid] + c]:
                    c += 1
                rows.append([rid, len(prompt) + known[rid] - 1, 1 + K])
                emitted[rid] = c + 1
                kept[rid].append(c)
            known[rid] += emitted[rid]
            if known[rid] == n:
                finished[rid] = {"reason": "length"}
        recs.append({"kind": "step", "step": step, "rows": rows, "emitted": emitted, "finished": finished, "cache_hit": {}, "resumed": [],
                     "preempted": []})
        step += 1
    io = {rid: {"user": rid, "prompt_ids": list(prompt), "output_ids": list(truth[rid])} for rid, prompt, _ in SERVED}
    return recs, io, kept


def _ingress(**params) -> dict:
    return {rid: {"prompt": list(prompt), "model": "m", "user": rid, **GREEDY, "max_tokens": n, **params} for rid, prompt, n in SERVED}


@pytest.mark.parametrize("K", [1, 3, 5])
def test_a_step_trace_with_draft_tokens_compiles_to_programs_that_give_its_kept_ids(K):
    """A recorded step trace whose decode rows carry K draft tokens becomes advice with no field for a kept count, the
    advice compiles to each request's Program in passes of 1 + K rows that take kept + 1 tokens, and evaluating a Program
    with the roles' stand-ins gives the ids the trace kept with every check 1: each pass's kept count against `Accept_v1`,
    and the length rule."""
    layer = _layer(K)
    recs, io, kept = _trace(K, layer)
    advice = ADV.from_trace(recs, io).advice
    assert advice.version == WA.DRAFTS and WA.decode(WA.encode(advice)) == advice
    assert all(row.kind == "decode" and row.n == 1 + K for st in advice.steps for row in st.rows if row.start >= advice.requests[row.request].prompt_len)
    counts = WA.drafts(advice)
    assert [[c for (s, j), (_, c) in sorted(counts.items()) if advice.steps[s].rows[j].request == i] for i in range(2)] == [kept["a"], kept["b"]]
    assert {c == K for c in kept["a"]} == {True, False} and 0 in kept["a"]

    compiled = compile_steps(_registered(K, layer), advice, _ingress())
    for i, (rid, prompt, n) in enumerate(SERVED):
        unit = compiled.requests[i]
        assert (unit.LP, unit.T, unit.drafts) == (len(prompt), n - 1, tuple((K, c) for c in kept[rid]))
        fn = compiled.programs[unit.digest].fn
        callees = [node.fn.id for node in fn.body.nodes]
        assert callees.count(layer.definition(K).id) == len(kept[rid]) == callees.count(bind(spec.Accept, K=K).id)
        assert sum(x.startswith("KeptCheck_v1") for x in callees) == len(kept[rid])
        assert sum(x.startswith("TokenSelect_v") for x in callees) == 1 + len(kept[rid]) * (1 + K)
        out = _walk(fn, [np.array(prompt), _weights(fn)])
        assert tuple(out[:n]) == _truth()[rid] and out[n:] == [1] * (len(kept[rid]) + 1)
    # a draft row is held whole, its samples its kept drafts and one more
    rows = {(r.request, r.start): (r.n, r.samples) for step in compiled.steps for r in step}
    for i, (rid, prompt, _) in enumerate(SERVED):
        at = len(prompt)
        for c in kept[rid]:
            assert rows[i, at] == (1 + K, c + 1)
            at += c + 1


def test_a_schedule_the_model_did_not_keep_makes_a_check_zero():
    """The kept counts shape the Program before the tokens they count exist, so each is checked: the Program of a schedule
    that claims one draft fewer than the accept kept, or one more, outputs a 0 check."""
    K, (rid, prompt, n) = 3, SERVED[0]
    layer, truth = _layer(K), _truth()[rid]
    _, _, kept = _trace(K, layer)
    assert kept[rid][0] == K and kept[rid][1] == 1
    for claim in ((K - 1, 1), (K, 2)):
        rest, passes = n - 1 - sum(c + 1 for c in claim), [(K, c) for c in claim]
        passes += [(K, 0)] * rest
        fn = build_program(Phi.from_json({**_phi(K).to_json(), "sampling": {"kind": "greedy"}}), "request", LP=len(prompt), T=n - 1, C=0,
                           dead_free=True, with_desc=False, drafts=tuple(passes), draft_layer=layer).program.fn
        out = _walk(fn, [np.array(prompt), _weights(fn)])
        assert 0 in out[n:] and (claim[0] < K or tuple(out[:1 + K + 1]) == truth[:1 + K + 1])


def test_a_launch_without_drafts_compiles_as_it_did():
    """A launch that registers no draft holds no schedule: its units' keys, its summary and its Programs' SRC configs say
    nothing of drafts."""
    steps = (St((Row(0, "prefill", 0, 4),)), St((Row(0, "decode", 4, 1),)), St((Row(0, "decode", 5, 1),)))
    advice = WA.Advice((R("a", 4, 3, "length"),), steps, (0,) * 3)
    assert advice.version == 0 and WA.drafts(advice) == {}
    ingress = {"a": {"prompt": [1, 8, 15, 22], "model": "m", "user": "a", **GREEDY, "max_tokens": 3}}
    compiled = compile_steps(_registered(), advice, ingress)
    unit = compiled.requests[0]
    assert unit.drafts == () and "drafts" not in unit.key() and "drafts" not in compiled.summary()["requests"][0]
    assert DR.src_config(_phi(), (), None) == {}
    assert not any(node.fn.id.startswith(("KeptCheck", "Accept", "DraftLayer")) for node in compiled.programs[unit.digest].fn.body.nodes)


def test_the_advice_reads_a_draft_rows_kept_count_off_the_next_rows_start():
    """A draft row is a decode row of more than one position, and the advice has no field for what it kept: the count is the
    request's next row's start less its own, less one, and the last row keeps what the request's outputs still lack. A
    window with one is version 3 in version 2's layout, and I1 v0 can't say it."""
    steps = (St((Row(0, "prefill", 0, 4),)), St((Row(0, "decode", 4, 4),)), St((Row(0, "decode", 7, 4),)))
    advice = WA.Advice((R("a", 4, 8, "length"),), steps, (0,) * 8)
    assert WA.drafts(advice) == {(1, 0): (3, 2), (2, 0): (3, 3)} and advice.version == WA.DRAFTS == 3
    assert list(WA.Row.__dataclass_fields__) == ["request", "kind", "start", "n", "cache_hit", "preempted", "sources"]
    assert WA.decode(WA.encode(advice)) == advice and WA.from_json(WA.to_json(advice)) == advice and WA.encode(advice)[0] == 3
    assert len(WA.outputs(advice)) == 8
    with pytest.raises(WA.MalformedAdvice, match="holds a draft row isn't I1 v0 advice"):
        advice.i1()
    plain = WA.Advice((R("a", 4, 2, "length"),), (steps[0], St((Row(0, "decode", 4, 1),))), (0,) * 2)
    assert plain.version == 0 and WA.drafts(plain) == {}
    for bad, why in (((steps[0], steps[1]), "is its last here and leaves it outputs to keep"),
                     ((steps[0], steps[1], St((Row(0, "decode", 9, 3),))), "next row starts at 9, 4 past its draft row's token, which carries 3 drafts"),
                     ((St((Row(0, "decode", 0, 4),)),), "a draft row follows its request's first output, and position 0 is in request 0's 4-token prompt")):
        with pytest.raises(WA.MalformedAdvice, match=why):
            WA.Advice((R("a", 4, 8, "length"),), bad, (0,) * 8)


@pytest.mark.parametrize("change, why", [
    (lambda recs: recs[0]["config"].update(async_scheduling=True), "under async scheduling, where a row starts where the scheduler expects"),
    (lambda recs: recs[0]["config"].pop("speculative"), "past its prompt, and the trace's header states no draft"),
    (lambda recs: recs[0]["config"]["speculative"].update(num_speculative_tokens=2), "past its prompt, and the trace's header drafts 2 tokens a row"),
    (lambda recs: next(r for r in recs if r["kind"] == "step" and r["step"] == 1)["emitted"].update(a=0), "draft row of 4 positions emitted 0 tokens"),
])
def test_a_trace_whose_draft_rows_the_advice_cannot_read_is_refused(change, why):
    recs, io, _ = _trace(3, _layer(3))
    change(recs)
    with pytest.raises(ValueError, match=re.escape(why)):
        ADV.from_trace(recs, io)


def test_the_accept_and_the_draft_layer_are_reached_by_declared_roles():
    """The greedy accept's role names the registered `Accept_v1{K}` and its ports; the sampled accept's role names its
    Definition family and ports and is bound to nothing; the draft layer's role is a stand-in."""
    assert set(DR.ROLES) == {"draft_accept", "sampled_accept", "draft_layer"}
    greedy, sampled, layer = (DR.ROLES[k] for k in ("draft_accept", "sampled_accept", "draft_layer"))
    assert (greedy.definition, greedy.bound, greedy.stand_in) == ("Accept_v1{K}", True, False)
    for K in (1, 3, 5):
        got = DR.accept("greedy", K)
        (cand, argmax), ret = got.signature()
        assert got == bind(spec.Accept, K=K) and (cand[1].leaves, argmax[1].leaves, ret.leaves) == (K, K + 1, K + 1 + K + 1)
    assert (sampled.definition, sampled.bound) == ("SampledAccept_v1{K,V,TILE}", False)
    assert sampled.ports == ("(logits: Array<K+1, Array<V, f32>>, draft: Array<K, i32>, seed: i64, pos: Array<K+1, i64>, temp: f32) -> "
                             "(verdict: Array<K, i32>, accepted: i32, appended: Array<K+1, i32>)")
    with pytest.raises(Unexpressible, match=r"request 1: a stochastic request with a draft row: its accept is the role sampled_accept's, "
                                            r"SampledAccept_v1\{K,V,TILE\} .* no Definition is bound to it"):
        DR.accept("stochastic", 3, "request 1: ")
    assert (layer.definition, layer.bound, layer.stand_in) == ("DraftLayerStandIn_v1{K,TABLE}", True, True)
    stand_in = DR.StandIn(((3, (7, 8, 9)), (200, (0, 0, 1))))
    defn = stand_in.definition(3)
    for token in (3, 200, 5):
        got = np.asarray(evaluate_batch(defn, np.array([token], dtype=np.int32))).reshape(-1).tolist()
        assert tuple(got) == stand_in.propose(token) == {3: (7, 8, 9), 200: (0, 0, 1)}.get(token, (token,) * 3)
    with pytest.raises(Unexpressible, match="the draft layer's stand-in has no Definition at 5 drafts"):
        stand_in.definition(5)


def test_the_kept_check_accepts_the_accepts_own_count_alone():
    """`KeptCheck_v1{KEPT, LAST}` over `Accept_v1`'s count: 1 exactly at KEPT before a request's last pass, and from KEPT up
    in its last, where the length rule cuts the pass; and it is a relation a check bit may come from."""
    assert inputs.RELATIONS["KeptCheck_v1"].endswith("test_draft_rows.py::test_the_kept_check_accepts_the_accepts_own_count_alone")
    counts = np.arange(0, 7, dtype=np.int32)
    for kept in range(6):
        exact = np.asarray(evaluate_batch(bind(DR.KeptCheck, KEPT=kept, LAST=0), counts)).reshape(-1)
        least = np.asarray(evaluate_batch(bind(DR.KeptCheck, KEPT=kept, LAST=1), counts)).reshape(-1)
        assert exact.tolist() == [int(c == kept) for c in counts] and least.tolist() == [int(c >= kept) for c in counts]
    K = 3
    argmax = np.array([[5, 6, 7, 8]] * 4, dtype=np.int32)
    cand = np.array([[5, 6, 7], [5, 6, 0], [5, 0, 7], [0, 6, 7]], dtype=np.int32)
    out = np.asarray(evaluate_batch(bind(spec.Accept, K=K), cand, argmax)).reshape(4, -1)
    assert out[:, K].tolist() == [3, 2, 1, 0]
    for kept in range(K + 1):
        bits = np.asarray(evaluate_batch(bind(DR.KeptCheck, KEPT=kept, LAST=0), out[:, K].astype(np.int32))).reshape(-1)
        assert bits.tolist() == [int(kept == c) for c in (3, 2, 1, 0)]
    with pytest.raises(ValueError, match="KEPT must be in"):
        bind(DR.KeptCheck, KEPT=-1, LAST=0)


def _launched(**engine_kwargs) -> dict:
    return {"model": "m", "revision": "r", "target": dict(RTXPRO6000), "execution": "eager",
            "engine_record": {"engine_kwargs": {"max_model_len": 2048, **engine_kwargs}}}


@pytest.mark.parametrize("cache", ["fp8", "fp8_e4m3", "fp8_ds_mla"])
def test_an_fp8_cache_launch_is_refused_by_name(cache):
    """A request Program's history is the bf16 KV cache. A served engine's φ refuses a launch that names another format by
    its setting, and the step compiler refuses a registration that states one, before any Program."""
    with pytest.raises(ValueError, match=re.escape(f"engine kv_cache_dtype={cache!r}: a request Program models the KV cache of a launch "
                                                    "at kv_cache_dtype auto or bfloat16 and no other format (an FP8 KV cache is not modelled)")):
        PHI.served_phi(_launched(kv_cache_dtype=cache), TINY)
    for stated in ({}, {"kv_cache_dtype": None}, {"kv_cache_dtype": "auto"}, {"kv_cache_dtype": "bfloat16"}):
        assert PHI.served_phi(_launched(**stated), TINY).engine == {"max_model_len": 2048}
    why = f"an FP8-cache launch (engine arg kv_cache_dtype {cache!r}): a request Program's history is the KV cache of a launch at"
    assert cache_refusal(_phi(engine={"kv_cache_dtype": cache})).startswith(why)
    recs, io, _ = _trace(3, _layer(3))
    with pytest.raises(Unexpressible, match=re.escape(why)):
        compile_steps(_registered(3, _layer(3), engine={"kv_cache_dtype": cache}), ADV.from_trace(recs, io).advice, _ingress())
    assert cache_refusal(_phi(engine={"kv_cache_dtype": "bfloat16"})) is None and cache_refusal(_phi()) is None


@pytest.mark.parametrize("quantization, cache", [
    (MODELOPT, "fp8_e4m3"),
    ({"quant_method": "modelopt", "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"}}, "fp8_e4m3"),
    ({"quant_method": "modelopt_fp4", "quantization": {"kv_cache_scheme": {"num_bits": 4, "type": "float"}}}, "nvfp4"),
])
def test_a_modelopt_checkpoint_that_turns_a_quantized_cache_on_at_auto_is_refused_by_name(quantization, cache):
    """vLLM serves a `modelopt` checkpoint that names a KV cache scheme on a quantized cache where the launch leaves
    ``kv_cache_dtype`` at ``auto``, with no engine arg saying so. The served φ and the step compiler both refuse it; the same
    checkpoint under a launch that names the bf16 cache compiles, its φ keeping the arg; and a checkpoint that names no
    scheme, or one vLLM doesn't know, leaves the default."""
    cfg = {**TINY, "quantization_config": quantization}
    assert PHI.checkpoint_kv_cache(cfg) == cache
    for stated in ({}, {"kv_cache_dtype": "auto"}):
        with pytest.raises(ValueError, match=re.escape(f"the checkpoint's quantization_config turns a {cache} KV cache on at "
                                                        "kv_cache_dtype auto (a modelopt checkpoint that names a KV cache scheme)")):
            PHI.served_phi(_launched(**stated), cfg)
    why = (f"a modelopt checkpoint whose quantization_config names a KV cache scheme: vLLM serves it on a {cache} KV cache at "
           "kv_cache_dtype auto, and a quantized cache isn't modelled")
    assert cache_refusal(_phi(cfg=cfg)) == why
    recs, io, _ = _trace(3, _layer(3))
    with pytest.raises(Unexpressible, match=re.escape(why)):
        compile_steps(_registered(3, _layer(3), cfg=cfg), ADV.from_trace(recs, io).advice, _ingress())
    named = PHI.served_phi(_launched(kv_cache_dtype="bfloat16"), cfg)
    assert named.engine == {"max_model_len": 2048, "kv_cache_dtype": "bfloat16"} and cache_refusal(named) is None
    for other in ({"quant_method": "modelopt"}, {"quant_method": "fp8", "kv_cache_scheme": "fp8"},
                  {"quant_method": "modelopt", "kv_cache_scheme": {"dynamic": True, "num_bits": 8, "type": "float"}}, None):
        assert PHI.checkpoint_kv_cache({**TINY, **({"quantization_config": other} if other else {})}) is None


def test_a_served_engine_that_decodes_speculatively_is_still_refused():
    """A draft Program calls a stand-in for the model's draft layer, which no engine ran, so a served engine's φ refuses a
    launch that decodes speculatively by its setting."""
    for setting in ({"speculative_config": {"method": "mtp", "num_speculative_tokens": 5}}, {"spec_tokens": 5}):
        (name, value), = setting.items()
        with pytest.raises(ValueError, match=re.escape(f"engine {name}={value!r}: a served engine's request Programs model no draft tokens "
                                                        "(speculative decoding): no Definition of the model's draft layer is registered")):
            PHI.served_phi(_launched(**setting), TINY)


@pytest.mark.parametrize("registered, why", [
    (lambda: _registered(3), "a launch that drafts 3 tokens a decode row, and the registration names no draft layer"),
    (lambda: _registered(None, _layer(3)), "the registration names a draft layer, and its launch drafts no token"),
    (lambda: _registered(2, _layer(2)), "request 0: step 1 runs a draft row of 3 drafts, over the registered num_speculative_tokens 2"),
    (lambda: _registered(True, _layer(3)), "engine arg num_speculative_tokens True: the drafts a decode row carries at most, a positive count"),
    (lambda: _registered(3, _layer(3), target={**RTXPRO6000, "batch_invariant": False}),
     "a launch that drafts 3 tokens a decode row on a target without batch invariance"),
    (lambda: _registered(3, _layer(3), tp={"world": 2, "rank": 0}), "tensor-parallel world 2: draft rows compile on one rank"),
    (lambda: _registered(5, _layer(3)), "the draft layer's stand-in has no Definition at 5 drafts"),
])
def test_a_launch_whose_draft_rows_the_compiler_does_not_hold_is_refused_by_name(registered, why):
    recs, io, _ = _trace(3, _layer(3))
    with pytest.raises(Unexpressible, match=re.escape(why)):
        compile_steps(registered(), ADV.from_trace(recs, io).advice, _ingress())


def test_a_draft_row_in_a_launch_that_registers_no_draft_is_refused_by_name():
    recs, io, _ = _trace(3, _layer(3))
    with pytest.raises(Unexpressible, match=re.escape("request 0: step 1 runs a draft row of 3 drafts, and the registered launch drafts no token")):
        compile_steps(_registered(), ADV.from_trace(recs, io).advice, _ingress())


def test_a_sampled_request_with_a_draft_row_is_refused_by_the_role_it_lacks():
    recs, io, _ = _trace(3, _layer(3))
    with pytest.raises(Unexpressible, match=r"request 0: a stochastic request with a draft row: its accept is the role sampled_accept's"):
        compile_steps(_registered(3, _layer(3)), ADV.from_trace(recs, io).advice, _ingress(**STOCH), salt=bytes(range(64)))


def test_a_draft_schedule_the_passes_are_not_written_for_is_refused_by_name():
    """The builder writes the passes for a greedy request wrapper with no cached prefix, chunk or launch batch, and for a
    schedule that makes the request's decode tokens."""
    phi, layer = Phi.from_json({**_phi(3).to_json(), "sampling": {"kind": "greedy"}}), _layer(3)
    with pytest.raises(Unexpressible, match="a draft schedule but on a greedy request wrapper"):
        build_program(phi, "request", LP=4, T=4, C=2, dead_free=True, with_desc=False, drafts=((3, 3),), draft_layer=layer)
    for drafts in (((3, 2),), ((3, 3), (3, 0)), ((4, 3),)):
        with pytest.raises(Unexpressible, match="a draft schedule .* for 4 decode tokens at 3 drafts a row"):
            build_program(phi, "request", LP=4, T=4, C=0, dead_free=True, with_desc=False, drafts=drafts, draft_layer=layer)
    with pytest.raises(Unexpressible, match="a step Program of a launch that drafts 3 tokens a decode row"):
        build_program(phi, "step", M=1, pos0=4, with_desc=False)
