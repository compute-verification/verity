"""verity_vllm.program.predict.steps (I2): a session's advice (I1 v0) and its certified request bodies compile to the Build's request Programs
at the shapes they ran, its rows to references into them, and every other input is refused by name: ill-formed advice, advice the
bodies contradict, a forged cache hit, and what v0 can't compile.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from verity_vllm.pipeline import window_advice as WA
from verity_vllm.program.predict import batch as B
from verity_vllm.program.predict.build import Unexpressible, _Specs, build_program, model_shape
from verity_vllm.program.predict.common import _arr
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import (
    Registered,
    RowRef,
    compare,
    compile_steps,
    dead_gates_of,
    plan_steps,
    policy_of,
    request_program,
    trace_commands,
    trace_dir,
)
from verity_vllm.program.prune import dead_free_versions
from verity.ml.definitions.registry import io_shell as IO
from verity.ml.definitions.registry.topp_geometry import registered_splits

from verity.core.protocols.certifier import advice as I1
from verity.ml.circuits.defs import bind
from verity.ml.circuits.partition_object import calls

FIX = Path(__file__).parent / "fixtures"
RTXPRO6000 = {"compute_capability": [12, 0], "num_sms": 188}
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
        "max_position_embeddings": 64, "hidden_act": "silu"}
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": None, "max_tokens": 4}
STOCH = {"temperature": 0.8, "top_p": 1.0, "top_k": 0, "seed": None, "max_tokens": 3, "min_p": 0.0, "repetition_penalty": 1.0}
STOCH_POLICY = {"kind": "stochastic", "temperature": 0.8, "top_p": 1.0, "top_k": 0, "seeded": True, "splits": registered_splits(188)}
SERVED = "served-model"


def _registered(cfg: dict = TINY, **kw) -> Registered:
    cap = kw.pop("cap_bits", None)
    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, target=dict(RTXPRO6000), **kw)
    return Registered.of(phi, default_seed=7) if cap is None else Registered.of(phi, default_seed=7, cap_bits=cap)


def _prompt(i: int, lp: int) -> list[int]:
    return [(31 * i + 7 * k + 1) % 256 for k in range(lp)]


def _session(reqs: list[tuple], steps: list[list[tuple]]) -> tuple[I1.Advice, dict]:
    """The advice and the certified bodies of a session: each request is (prompt_len, output_len, body[, prompt]), its id `q<i>` and
    its finish `length` when it kept max_tokens outputs; each row is (request, start, n[, Row fields]), its kind I1's."""
    requests = tuple(I1.RequestAdvice(f"q{i}", r[0], r[1], "length" if r[1] == r[2]["max_tokens"] else "eos") for i, r in enumerate(reqs))

    def row(x: tuple) -> I1.Row:
        r, start, n, *kw = x
        return I1.Row(r, "decode" if n == 1 and start >= reqs[r][0] else "prefill", start, n, **(kw[0] if kw else {}))

    advice = I1.Advice(requests, tuple(I1.StepAdvice(tuple(row(x) for x in st)) for st in steps), (0,) * sum(r[1] for r in reqs))
    ingress = {f"q{i}": {"prompt": list(r[3]) if len(r) > 3 else _prompt(i, r[0]), "model": "m", "user": f"q{i}", **r[2]}
               for i, r in enumerate(reqs)}
    return advice, ingress


def _mixed() -> tuple[I1.Advice, dict]:
    """Three requests: r0 (LP 8, greedy) from step 0, r1 (LP 5, seeded top-p) prefilled beside r0's first decode, r2 (r0's shape)
    prefilled beside two decodes, each step's decode rows first (Model Runner V2's order); each kept its max_tokens."""
    return _session([(8, 4, GREEDY), (5, 3, STOCH), (8, 4, GREEDY)], [
        [(0, 0, 8)],
        [(0, 8, 1), (1, 0, 5)],
        [(0, 9, 1), (1, 5, 1), (2, 0, 8)],
        [(0, 10, 1), (1, 6, 1), (2, 8, 1)],
        [(2, 9, 1)],
        [(2, 10, 1)],
    ])


def _wired(ingress: dict, *ids: str) -> dict:
    return {k: IO.encode(SERVED, b["prompt"], b["max_tokens"], TINY["vocab_size"]) if k in ids else b for k, b in ingress.items()}


def test_byte_ingress_compiles_to_the_same_programs_with_a_parse_unit_each():
    """r0 and r2 bring ingress v0's bytes: their Programs and rows are the ones their parsed bodies compile to, and each gets
    `IngressPrompt_v1` at its LP, the unit that reads the ingress transfer gates; r1's parsed body gets none."""
    reg, (advice, ingress) = Registered.of(_registered().phi, 7, served_model=SERVED), _mixed()
    plain, wired = compile_steps(reg, advice, ingress), compile_steps(reg, advice, _wired(ingress, "q0", "q2"))
    assert [u.digest for u in wired.requests] == [u.digest for u in plain.requests] and wired.steps == plain.steps
    parse = bind(IO.IngressPrompt, MODEL=SERVED, LP=8, V=256)
    assert [u.parse for u in wired.requests] == [parse, None, parse]
    assert [u["parse"] for u in json.loads(json.dumps(wired.summary()))["requests"]] == [parse.id, None, parse.id]
    selects = (f"GumbelTopPTokenSelect_v3{{V=256,S={registered_splits(reg.phi.target_profile().num_sms)}}}", "TokenSelect_v4{V=256}")
    assert plain.transfer_gates == selects and wired.transfer_gates == tuple(sorted((parse.id, *selects)))
    assert wired.summary()["transfer_gates"] == list(wired.transfer_gates)


@pytest.mark.parametrize("edit, served, why", [
    (lambda b: b, None, "names no served model"),
    (lambda b: b.replace(b'"temperature":0', b'"temperature":1'), SERVED, "aren't in ingress schema v0"),
    (lambda b: b[:-1] + b" }", SERVED, "not ingress v0's"),
    (lambda b: b.rstrip(b" "), SERVED, "not ingress v0's"),
    (lambda b: b[:-1] + b"\n", SERVED, "aren't in ingress schema v0"),
])
def test_byte_ingress_without_a_served_model_or_out_of_the_schema_is_refused(edit, served, why):
    advice, ingress = _mixed()
    ingress = _wired(ingress, "q0")
    ingress["q0"] = edit(ingress["q0"])
    reg = Registered.of(_registered().phi, 7, served_model=served)
    with pytest.raises(Unexpressible, match=why):
        compile_steps(reg, advice, ingress)


def test_a_requests_program_reads_its_own_fields_alone():
    """r1 run alone compiles to the Program it had among r0 and r2: its template and statics read its advice record and body, never
    its batch-mates."""
    reg, (advice, ingress) = _registered(), _mixed()
    alone, body = _session([(5, 3, STOCH, ingress["q1"]["prompt"])], [[(0, 0, 5)], [(0, 5, 1)], [(0, 6, 1)]])
    a, b = compile_steps(reg, alone, body).requests[0], compile_steps(reg, advice, ingress).requests[1]
    assert (a.digest, a.LP, a.T, a.policy, a.seed) == (b.digest, b.LP, b.T, b.policy, b.seed)


def test_a_plan_is_the_compilation_before_its_programs_are_built():
    """`plan_steps` fixes the shapes, rows and refusals; `request_program` builds one distinct Program, which a measurement can do
    request by request in other processes."""
    reg, (advice, ingress) = _registered(), _mixed()
    plan, c = plan_steps(reg, advice, ingress), compile_steps(reg, advice, ingress)
    assert [(u.request, u.LP, u.T) for u in plan.distinct()] == [(0, 8, 3), (1, 5, 2)]
    assert [request_program(reg, u).digest for u in plan.distinct()] == [c.requests[0].digest, c.requests[1].digest]
    assert plan.steps == [[(r.request, r.start, r.n, r.samples) for r in rows] for rows in c.steps]
    assert all(u is None or u.digest == "" for u in plan.requests)


def test_the_bodys_seed_is_the_seed_input_and_the_policy_reads_temperature_and_top_p():
    """Top-p takes the registered split count registered_splits(num_SMs) at any top_p, the S serving pins (`engine.topp_pin`), never
    the batch's; a greedy body's top_p is never read."""
    assert policy_of(0, {**STOCH, "seed": 123}, 7, 188) == (STOCH_POLICY, 123)
    assert policy_of(0, {**STOCH, "top_p": 0.95}, 7, 188) == ({**STOCH_POLICY, "top_p": 0.95}, 7)
    assert registered_splits(188) == 32
    assert policy_of(0, {**STOCH, "top_k": -1, "top_p": None}, 7, 188) == (STOCH_POLICY, 7)
    assert policy_of(0, {**GREEDY, "top_p": 0.5, "stop": ["\n"], "ignore_eos": False}, 7, None) == ({"kind": "greedy"}, None)
    with pytest.raises(Unexpressible, match="top-p on a target that declares no num_sms"):
        policy_of(0, STOCH, 7, None)


@pytest.mark.parametrize("sampling,fact", [
    ({**STOCH, "top_k": 50}, "top_k=50"),
    ({**STOCH, "min_p": 0.1}, "min_p=0.1"),
    ({**STOCH, "presence_penalty": 0.5}, "presence_penalty=0.5"),
    ({**STOCH, "repetition_penalty": 1.1}, "repetition_penalty=1.1"),
    ({**STOCH, "logit_bias": {"5": 1.0}}, "logit_bias="),
    ({**STOCH, "min_tokens": 4}, "min_tokens=4"),
    ({**STOCH, "n": 2}, "n=2"),
    ({**STOCH, "bad_words": ["x"]}, "bad_words (a parameter this compiler doesn't know)"),
    ({**STOCH, "temperature": 1e-6}, "temperature 1e-06"),
    ({"top_p": 1.0}, "no temperature"),
    ({**STOCH, "temperature": "0.8"}, "temperature '0.8', top_p 1.0 and seed None aren't"),
    ({**STOCH, "top_p": 0.0}, "top_p 0.0"),
    ({**STOCH, "seed": -1}, "seed -1"),
    (["temperature"], "aren't an object of named values"),
])
def test_sampling_v0_cannot_express_is_refused_by_name(sampling, fact):
    with pytest.raises(Unexpressible, match=f"request 0: .*{fact}".replace("(", r"\(").replace(")", r"\)").replace("[", r"\[")):
        policy_of(0, sampling, 7, 188)


def test_sampling_is_the_bodys_never_the_advices():
    """A request's policy is its certified body's. Advice is I1's bytes or an Advice, neither of which has a sampling field: an
    advice-shaped object carrying sampling is refused, and so are I1 bytes with sampling appended."""
    reg, (advice, ingress) = _registered(), _mixed()
    hot = {**ingress, "q0": {**ingress["q0"], "temperature": 0.8, "top_p": 1.0, "seed": 3}}
    assert compile_steps(reg, advice, hot).requests[0].policy == STOCH_POLICY and compile_steps(reg, advice, hot).requests[0].seed == 3
    smuggled = NS(requests=[NS(id=q.id, prompt_len=q.prompt_len, output_len=q.output_len, finish=q.finish, sampling=STOCH)
                            for q in advice.requests], steps=advice.steps, release=advice.release)
    with pytest.raises(Unexpressible, match="the advice is I1's bytes or an I1 Advice, not a SimpleNamespace"):
        compile_steps(reg, smuggled, ingress)
    with pytest.raises(Unexpressible, match="ill-formed advice: bytes past the advice"):
        compile_steps(reg, I1.encode(advice) + json.dumps(STOCH).encode(), ingress)
    with pytest.raises(TypeError):
        I1.RequestAdvice("q0", 8, 4, "length", sampling=STOCH)  # type: ignore[call-arg]


def test_ill_formed_advice_is_refused_by_name():
    reg, (advice, ingress) = _registered(), _mixed()
    data = I1.encode(advice)
    with pytest.raises(Unexpressible, match="ill-formed advice: truncated"):
        compile_steps(reg, data[: len(data) // 2], ingress)
    with pytest.raises(Unexpressible, match=rf"the advice is {8 * len(data)} bits, over the registered cap ℓ = {8 * len(data) - 8} bits"):
        compile_steps(_registered(cap_bits=8 * len(data) - 8), data, ingress)
    with pytest.raises(Unexpressible, match="ill-formed advice: window advice version 3, not 0, 1 or 2"):
        compile_steps(reg, bytes([3]) + data[1:], ingress)
    with pytest.raises(Unexpressible, match="not a bytearray"):
        compile_steps(reg, bytearray(data), ingress)
    with pytest.raises(ValueError, match="the cap ℓ is a whole number of bits"):
        _registered(cap_bits=True)


def test_the_compiler_is_total_on_every_truncation_and_bit_flip_of_an_advice():
    """Every prefix and every one-bit change of a valid advice's bytes, and random byte strings: each is planned or refused by name
    (`Unexpressible`), the same way twice, and nothing else escapes."""
    reg, (advice, ingress) = _registered(), _mixed()
    data = I1.encode(advice)
    rng = random.Random(0)
    inputs = [data[:k] for k in range(len(data) + 1)]
    inputs += [bytes(b ^ (1 << (7 - j % 8)) if i == j // 8 else b for i, b in enumerate(data)) for j in range(8 * len(data))]
    inputs += [rng.randbytes(rng.randrange(64)) for _ in range(200)]
    planned = 0
    for x in inputs:
        outcomes = []
        for _ in range(2):
            try:
                plan = plan_steps(reg, x, ingress, partial=False)
                outcomes.append(("plan", [u and (u.LP, u.T) for u in plan.requests], plan.steps))
            except Unexpressible as e:
                outcomes.append(("refused", str(e)))
        assert outcomes[0] == outcomes[1]
        planned += outcomes[0][0] == "plan"
    assert planned >= 1


@pytest.mark.parametrize("change,fact", [
    (lambda ing: ing.pop("q1"), r"request 1 \('q1'\): no certified request body carries its id"),
    (lambda ing: ing["q0"].update(prompt=ing["q0"]["prompt"][:-1]), "request 0: the advice says 8 prompt tokens, its certified prompt has 7"),
    (lambda ing: ing["q0"].update(prompt=[1] * 7 + [256]), "request 0: its certified prompt isn't a list of token ids below 256"),
    (lambda ing: ing["q0"].update(prompt="abcdefgh"), "request 0: its certified prompt isn't a list"),
    (lambda ing: ing["q0"].update(max_tokens=5), "request 0: 4 kept outputs, finished on length, against its max_tokens 5"),
    (lambda ing: ing["q0"].update(max_tokens=3), "request 0: 4 kept outputs, finished on length, against its max_tokens 3"),
    (lambda ing: ing["q0"].pop("max_tokens"), "request 0: max_tokens None"),
    (lambda ing: ing.update(q9={"prompt": [1]}), "1 certified requests have no row in the advice, 'q9' first"),
])
def test_advice_the_certified_bodies_contradict_is_refused(change, fact):
    reg, (advice, ingress) = _registered(), _mixed()
    change(ingress)
    with pytest.raises(Unexpressible, match=fact):
        compile_steps(reg, advice, ingress)


def _hit(shared: int, hit: int = 16, body: dict = GREEDY) -> tuple[I1.Advice, dict]:
    """r1 (LP 24) admitted at step 1 on a `hit`-token prefix-cache hit of r0's KV (a block of vLLM's 16 tokens); its prompt shares r0's
    first `shared` tokens."""
    p0 = _prompt(0, 24)
    p1 = p0[:shared] + _prompt(1, 24)[shared:]
    return _session([(24, 4, GREEDY, p0), (24, 3, {**body, "max_tokens": 3}, p1)], [
        [(0, 0, 24)],
        [(0, 24, 1), (1, hit, 24 - hit, {"cache_hit": hit, "sources": (I1.Source(0, hit),)})],
        [(0, 25, 1), (1, 24, 1)],
        [(0, 26, 1), (1, 25, 1)],
    ])


def test_a_forged_cache_hit_is_refused():
    reg = _registered()
    with pytest.raises(Unexpressible, match="request 1: a forged cache hit at step 1: request 0's certified prompt differs from its own "
                                            "before position 16"):
        compile_steps(reg, *_hit(15))
    assert "forged cache hit" in compile_steps(reg, *_hit(15), partial=True).refused[1]
    advice, ingress = _hit(16)
    ingress["q0"]["prompt"] = ingress["q0"]["prompt"][:15] + [255] * 9
    with pytest.raises(Unexpressible, match="forged cache hit"):
        compile_steps(reg, advice, ingress)


def test_an_honest_cache_hit_compiles_to_the_cached_request_program():
    """r1's 16 cached rows are its Program's `prefix` input (the Build's `--cached 16`), r0 the producer of its one block; its
    prefill computes rows 16..23 at positions 16.., and its decode rows are as without a hit. `predict.batch`'s R2 keeps the hit,
    whose source's tokens compile_steps checks equal to r1's own."""
    reg = _registered()
    c = compile_steps(reg, *_hit(16))
    assert c.batch_rule is None and c.summary()["batch_rule"] is None and not c.refused
    assert compile_steps(reg, *_hit(16), partial=True).summary() == c.summary()
    u = c.requests[1]
    want = build_program(Phi.from_json({**reg.phi.to_json(), "sampling": {"kind": "greedy"}}), "request", LP=24, T=2, C=16)
    assert (u.LP, u.T, u.C, u.producers, u.digest) == (24, 2, 16, (0,), want.digest)
    assert u.digest != build_program(Phi.from_json({**reg.phi.to_json(), "sampling": {"kind": "greedy"}}), "request", LP=24, T=2).digest
    params = [(n, t) for n, t in c.programs[u.digest].fn.signature()[0]]
    assert [n for n, _ in params] == ["prompt", "prefix", "weights"] and repr(params[1][1]) == repr(_arr((2, 2, 16, 32), 16))
    assert c.programs[u.digest].fn.bindings["SRC"]["class"] == "verity-vllm/cached-greedy-request/v1"
    assert c.steps[1] == [RowRef(0, c.requests[0].digest, 24, 1, 1), RowRef(1, u.digest, 16, 8, 1)]
    assert c.workload.requests[1].prefix == {"rows": 16, "producers": [0]} and c.workload.requests[0].prefix is None
    s = c.summary()
    assert s["requests"][1]["C"] == 16 and trace_dir(s["requests"][1]) == "build_request_LP24_T2_C16_greedy"
    [_, cmd] = trace_commands(s, reg, "/t")
    assert cmd[cmd.index("--cached") + 1] == "16" and "--cached" not in trace_commands(s, reg, "/t")[0]


@pytest.mark.parametrize("hit,body,fact", [
    (8, GREEDY, "request 1: step 1's cache hit reads 8 tokens of request 0's KV, not whole blocks of 16"),
    (16, STOCH, "request 1: a prefix-cache hit of 16 tokens on a stochastic request"),
])
def test_a_cache_hit_the_cached_wrapper_cannot_express_is_refused(hit, body, fact):
    with pytest.raises(Unexpressible, match=fact):
        compile_steps(_registered(), *_hit(24, hit, body))


def test_a_decodes_attention_guard_is_chosen_from_the_request_alone():
    """FA2's `MASKED_FROM` on cc 12.0 reads the launch's causal flag: r0's decode at P = BN + 1 beside r1's 8-row prefill runs causal
    and binds another static than alone. Its Program takes the static of its own rows (launch_sq 1), so it equals r0's alone."""
    cfg = {**TINY, "max_position_embeddings": 512}
    reg, C = _registered(cfg), model_shape(cfg)
    S = _Specs(C, reg.phi.target_profile(), M=1)
    BN = S.TG.attention_geometry(reg.phi.target_profile(), C.D, 1, C.NH, C.KVH)["BN"]
    alone_fn, beside_fn = S.attention(BN + 1, 0, 1, 1), S.attention(BN + 1, 0, 1, 8)
    assert alone_fn.id != beside_fn.id
    body = {**GREEDY, "max_tokens": 3}
    c = compile_steps(reg, *_session([(BN, 3, body), (8, 2, {**body, "max_tokens": 2})], [
        [(0, 0, BN)], [(0, BN, 1), (1, 0, 8)], [(0, BN + 1, 1), (1, 8, 1)]]))
    alone = compile_steps(reg, *_session([(BN, 3, body, _prompt(0, BN))], [[(0, 0, BN)], [(0, BN, 1)], [(0, BN + 1, 1)]]))
    assert c.requests[0].digest == alone.requests[0].digest
    ids = {fn.id for _, fn in calls(c.programs[c.requests[0].digest])}
    alone_id, beside_id = (bind(dead_free_versions()[fn.base_id], **fn.bindings).id for fn in (alone_fn, beside_fn))
    assert alone_id in ids and beside_id not in ids


def test_compile_steps_raises_the_batch_rules_refusal_and_partial_records_it(monkeypatch):
    def refuse(*_a, **_k):
        raise B.BatchDependent("request 0: refused by the batch rule")
    monkeypatch.setattr(B, "check_compiled", refuse)
    reg, (advice, ingress) = _registered(), _mixed()
    with pytest.raises(B.BatchDependent, match="refused by the batch rule"):
        compile_steps(reg, advice, ingress)
    assert compile_steps(reg, advice, ingress, partial=True).summary()["batch_rule"] == "request 0: refused by the batch rule"


def test_partial_refuses_a_request_whose_cache_hit_reads_a_refused_requests_kv():
    """r0 is refused (top_k), so r1, whose hit reads r0's KV, leaves the compilation with it; r2 compiles."""
    p0 = _prompt(0, 24)
    advice, ingress = _session([(24, 2, {**STOCH, "top_k": 50, "max_tokens": 2}, p0), (24, 1, {**GREEDY, "max_tokens": 1}, p0),
                                (8, 1, {**GREEDY, "max_tokens": 1})], [
        [(0, 0, 24)],
        [(0, 24, 1), (1, 16, 8, {"cache_hit": 16, "sources": (I1.Source(0, 16),)}), (2, 0, 8)],
    ])
    c = compile_steps(_registered(), advice, ingress, partial=True)
    assert sorted(c.refused) == [0, 1] and [u is not None for u in c.requests] == [False, False, True]
    assert c.refused[1] == "request 1: its cache hit reads request 0's KV, and request 0 is refused"
    with pytest.raises(Unexpressible, match="top_k=50"):
        compile_steps(_registered(), advice, ingress)


def test_positions_past_the_models_are_refused_by_name():
    with pytest.raises(Unexpressible, match="request 0: positions to 64, past the model's 64 positions"):
        compile_steps(_registered(), *_session([(62, 4, GREEDY)], [[(0, 0, 62)], [(0, 62, 1)], [(0, 63, 1)], [(0, 64, 1)]]))
    compile_steps(_registered(), *_session([(60, 4, GREEDY)], [[(0, 0, 60)], [(0, 60, 1)], [(0, 61, 1)], [(0, 62, 1)]]))


def _whole(reg: Registered, LP: int, T: int, C: int = 0) -> str:
    return build_program(Phi.from_json({**reg.phi.to_json(), "sampling": {"kind": "greedy"}}), "request", LP=LP, T=T, C=C).digest


def test_a_chunked_prefill_compiles_to_the_whole_prefill_program():
    """Request 0's prompt of 8 runs as chunks of 3 and 5, the first's sample discarded: its Program is the one it has unchunked, and
    each chunk is its prefill rows in turn. Request 1's chunks follow its prefix-cache hit of request 0's first block."""
    reg = _registered()
    c = compile_steps(reg, *_session([(8, 2, GREEDY)], [[(0, 0, 3)], [(0, 3, 5)], [(0, 8, 1)]]))
    d = _whole(reg, 8, 1)
    assert c.requests[0].digest == d and c.requests[0].first_step == 0 and c.discarded == []
    assert c.steps == [[RowRef(0, d, 0, 3, 0)], [RowRef(0, d, 3, 5, 1)], [RowRef(0, d, 8, 1, 1)]]
    assert c.requests[0].digest == compile_steps(reg, *_session([(8, 2, GREEDY)], [[(0, 0, 8)], [(0, 8, 1)]])).requests[0].digest
    p = _prompt(0, 40)
    c = compile_steps(reg, *_session([(40, 1, {**GREEDY, "max_tokens": 1}, p), (40, 2, {**GREEDY, "max_tokens": 2}, p)], [
        [(0, 0, 40)],
        [(1, 16, 8, {"cache_hit": 16, "sources": (I1.Source(0, 16),)})],
        [(1, 24, 16)],
        [(1, 40, 1)],
    ]))
    d = _whole(reg, 40, 1, C=16)
    assert (c.requests[1].digest, c.requests[1].C, c.requests[1].producers) == (d, 16, (0,))
    assert [s[-1] for s in c.steps[1:]] == [RowRef(1, d, 16, 8, 0), RowRef(1, d, 24, 16, 1), RowRef(1, d, 40, 1, 1)]


def test_a_preempted_request_compiles_to_its_program_and_its_recompute_is_discarded():
    """Request 0 is preempted after two kept outputs and resumes at step 3 by recomputing positions 0..9: its Program is the one it
    has unpreempted, position 9 is its resume's one new row, and positions 0..8 are a recompute no Program holds. A gap of steps
    between rows is no matter either."""
    reg = _registered()
    plain = compile_steps(reg, *_session([(8, 3, GREEDY)], [[(0, 0, 8)], [(0, 8, 1)], [(0, 9, 1)]]))
    c = compile_steps(reg, *_session([(8, 3, GREEDY), (4, 1, {**GREEDY, "max_tokens": 1})],
                                     [[(0, 0, 8)], [(0, 8, 1)], [(1, 0, 4)], [(0, 0, 10, {"preempted": True})]]))
    d = plain.requests[0].digest
    assert c.requests[0].digest == d == _whole(reg, 8, 2)
    assert [[r for r in rows if r.request == 0] for rows in c.steps] == [[RowRef(0, d, 0, 8, 1)], [RowRef(0, d, 8, 1, 1)], [],
                                                                         [RowRef(0, d, 9, 1, 1)]]
    assert c.discarded == [(3, 0, 0, 9)]


def test_a_resume_reads_its_own_kv_past_its_prompt():
    """Request 0 (LP 20) is preempted at position 33 and resumes on a hit of 32 tokens of its own KV, past its prompt: I1's replay
    holds that run (the request computed those positions), its resume recomputes position 32 and computes 33 anew."""
    reg = _registered()
    rows = [[(0, 0, 20)]] + [[(0, 20 + k, 1)] for k in range(13)] + [[(0, 32, 2, {"cache_hit": 32, "preempted": True,
                                                                               "sources": (I1.Source(0, 32),)})], [(0, 34, 1)]]
    c = compile_steps(reg, *_session([(20, 16, {**GREEDY, "max_tokens": 16})], rows))
    d = _whole(reg, 20, 15)
    assert (c.requests[0].digest, c.requests[0].C) == (d, 0)
    assert c.steps[14] == [RowRef(0, d, 33, 1, 1)] and c.discarded == [(14, 0, 32, 1)] and c.steps[15] == [RowRef(0, d, 34, 1, 1)]
    p = _prompt(0, 20)
    other = _session([(20, 5, {**GREEDY, "max_tokens": 5}, p), (20, 16, {**GREEDY, "max_tokens": 16}, p)],
                     [[(0, 0, 20), (1, 0, 20)]] + [[(0, 20 + k, 1)] * (k < 4) + [(1, 20 + k, 1)] for k in range(13)]
                     + [[(1, 32, 2, {"cache_hit": 32, "preempted": True, "sources": (I1.Source(0, 24), I1.Source(1, 8))})], [(1, 34, 1)]])
    with pytest.raises(Unexpressible, match="request 1: step 14's cache hit reads request 0's KV at positions 0 … 23, past a certified"):
        compile_steps(reg, *other)


def test_the_row_past_a_requests_last_kept_output_is_discarded():
    """Request 0 kept 3 outputs (its prefill's and two decodes') and stopped on a stop token; async scheduling ran one row past
    them, at step 3: the request's Program holds its live rows only."""
    reg = _registered()
    steps = [[(0, 0, 8)], [(0, 8, 1)], [(0, 9, 1)], [(0, 10, 1)]]
    c = compile_steps(reg, *_session([(8, 3, GREEDY)], steps))
    d = build_program(Phi.from_json({**reg.phi.to_json(), "sampling": {"kind": "greedy"}}), "request", LP=8, T=2).digest
    assert (c.requests[0].digest, c.requests[0].T, c.workload.requests[0].cap) == (d, 2, 3)
    assert c.steps[3] == [] and c.discarded == [(3, 0, 10, 1)] and c.summary()["discarded"] == [[3, 0, 10, 1]]
    c = compile_steps(reg, *_session([(8, 4, GREEDY)], steps))
    assert (c.requests[0].T, c.discarded, len(c.steps[3])) == (3, [], 1)


def test_an_aborted_request_compiles_as_one_stopped_at_its_last_kept_output():
    """Request 0's client aborted it after 3 outputs, and the endpoint freed it after the step in flight, whose row (step 3) ran past
    them: it compiles to the Program of a stop at 3 outputs, that row discarded, with ignore_eos set or not. At its max_tokens vLLM
    has finished it on length, so an abort there is refused."""
    reg = _registered()
    stopped, ingress = _session([(8, 3, GREEDY)], [[(0, 0, 8)], [(0, 8, 1)], [(0, 9, 1)], [(0, 10, 1)]])
    aborted = WA.Advice((WA.RequestAdvice("q0", 8, 3, "abort"),), stopped.steps, stopped.release)
    c, d = compile_steps(reg, stopped, ingress), compile_steps(reg, aborted, {"q0": {**ingress["q0"], "ignore_eos": True}})
    assert [(u.digest, u.T) for u in d.requests] == [(u.digest, u.T) for u in c.requests]
    assert d.steps == c.steps and d.discarded == c.discarded == [(3, 0, 10, 1)]
    with pytest.raises(Unexpressible, match="request 0: 3 kept outputs, finished on abort, against its max_tokens 3"):
        compile_steps(reg, aborted, {"q0": {**ingress["q0"], "max_tokens": 3}})


def test_partial_refuses_request_by_request():
    """A measurement over served traffic: request 1's hit of a part of a block and request 3's top_k refuse those requests by name,
    their rows leave the steps, and requests 0 and 2 (preempted, resumed) compile as they would alone."""
    p0 = _prompt(0, 8)
    advice, ingress = _session([(8, 4, GREEDY, p0), (8, 2, GREEDY, p0[:4] + [9] * 4), (8, 3, GREEDY), (5, 2, {**STOCH, "top_k": 50})], [
        [(0, 0, 8), (1, 4, 4, {"cache_hit": 4, "sources": (I1.Source(0, 4),)}), (2, 0, 8)],
        [(0, 8, 1), (1, 8, 1), (2, 8, 1), (3, 0, 5)],
        [(0, 9, 1), (3, 5, 1)],
        [(0, 10, 1), (2, 0, 10, {"preempted": True})],
    ])
    reg = _registered()
    with pytest.raises(Unexpressible, match="not whole blocks of 16"):
        compile_steps(reg, advice, ingress)
    c = compile_steps(reg, advice, ingress, partial=True)
    assert sorted(c.refused) == [1, 3]
    assert "not whole blocks of 16" in c.refused[1] and "top_k=50" in c.refused[3]
    assert [u is not None for u in c.requests] == [True, False, True, False]
    alone = compile_steps(reg, *_session([(8, 4, GREEDY, p0)], [[(0, 0, 8)], [(0, 8, 1)], [(0, 9, 1)], [(0, 10, 1)]]))
    assert [[r for r in rows if r.request == 0] for rows in c.steps] == alone.steps
    d2 = c.requests[2].digest
    assert [[r for r in rows if r.request == 2] for rows in c.steps] == [[RowRef(2, d2, 0, 8, 1)], [RowRef(2, d2, 8, 1, 1)], [],
                                                                         [RowRef(2, d2, 9, 1, 1)]]
    assert c.discarded == [(3, 2, 0, 9)] and sorted(c.programs) == sorted({alone.requests[0].digest, d2})
    assert sorted(c.summary()["refused"]) == ["1", "3"]
    with pytest.raises(Unexpressible, match=r"every request is refused, request 0 first: request 0 \('q0'\): no certified request body"):
        compile_steps(reg, *_hit(16)[:1], {"q1": _hit(16)[1]["q1"]}, partial=True)


def test_dead_gates_are_counted_as_circuit_check_counts_them():
    """The request wrapper is in the dead-free form (`program.prune.dead_free`: no dead unit, RMSNormTriton, TokenSelect, the top-p
    select at a constant split count and FA2's attention at their versions without dead gates, and the last layer's fused qkv on the
    prompt rows before the last narrowed to its keys and values), so a request Program's only dead gates are inputs no gate reads:
    `dead_gates_of` counts them as circuit-check does and says under which definitions. The second session's prefill has a row of
    16 keys, where `Attention_v5` has 2 dead gates a head."""
    from verity.ml.circuits.liveness import dead_gates
    from verity_vllm.program.prune import dead_units
    sixteen = _session([(16, 2, {**GREEDY, "max_tokens": 2})], [[(0, 0, 16)], [(0, 16, 1)]])
    for advice, ingress in (_mixed(), sixteen):
        c = compile_steps(_registered(), advice, ingress)
        got = dead_gates_of(c)
        assert sorted(got["programs"]) == sorted(c.programs)
        for d, p in c.programs.items():
            rep = dead_gates(p, limit=0, transitive=True)
            rec = got["programs"][d]
            assert (rec["dead_computed"], rec["unread_inputs"]) == (rep.computed, rep.input) and not dead_units(p.fn.body)
            assert rep.computed == 0 and all(k.startswith("Input") for k in rec["by_definition"])
            assert sum(rec["by_definition"].values()) == rep.input
        assert got["dead_computed"] == 0


def test_a_program_with_a_dead_unit_is_refused(monkeypatch):
    """A request Program holding a node no output reads is refused, with the fix (in the lowering, never the advice) in the message."""
    import verity_vllm.program.predict.build as PB
    monkeypatch.setattr(PB, "dead_free_form", lambda b: (b, list(range(len(b.nodes)))))
    with pytest.raises(Unexpressible, match=r"request 0's Program [0-9a-f]{16} holds 56 dead units, node 104 \(RoPE_v1\{NHEADS=4,D=16\}\) first, which no "
                                            r"output reads: drop them in the lowering"):
        compile_steps(_registered(), *_mixed())


def test_a_tensor_parallel_rank_compiles_to_its_dead_free_rank_programs(tmp_path):
    """At TP2 each rank compiles the node's advice to its own rank's request Programs (its shards, a `peers_*` Input per collective)
    in the registered compiler's form, which the TP2 Build of record isn't: there the last layer's rows before the prefill's last end
    in units no output reads, its reduces among them. The rows are the one-rank compilation's, and the traces are the rank's
    `--dead-free` Builds."""
    from verity_vllm.program.prune import dead_units
    advice, ingress = _mixed()
    one = compile_steps(_registered(), advice, ingress)
    digests = []
    for rank in (0, 1):
        reg = _registered(tp={"world": 2, "rank": rank})
        c = compile_steps(reg, advice, ingress)
        assert [[(r.request, r.start, r.n, r.samples) for r in rows] for rows in c.steps] == \
               [[(r.request, r.start, r.n, r.samples) for r in rows] for rows in one.steps]
        assert c.discarded == one.discarded and dead_gates_of(c)["dead_computed"] == 0
        for u in {u.digest: u for u in c.requests if u is not None}.values():
            p = c.programs[u.digest]
            assert not dead_units(p.fn.body) and any(n.startswith("peers_") for n, _ in p.fn.body.params)
            kept = build_program(Phi.from_json({**reg.phi.to_json(), "sampling": u.policy}), "request", LP=u.LP, T=u.T).program.fn.body
            assert {kept.nodes[i].fn.id.split("{")[0] for i in dead_units(kept)} >= {"AllReduce2_v1", "Attention_v5"}
        digests.append(sorted(c.programs))
        cmd = trace_commands(c.summary(), reg, str(tmp_path))[0]
        assert cmd[-5:] == ["--tp", "2", "--rank", str(rank), "--dead-free"]
    assert len({*digests[0], *digests[1], *one.programs}) == 3 * len(one.programs)
    assert "--tp" not in trace_commands(one.summary(), _registered(), str(tmp_path))[0]


def test_a_target_without_batch_invariance_is_refused():
    with pytest.raises(Unexpressible, match="a target without batch invariance"):
        compile_steps(_registered(execution="eager"), *_mixed())
    with pytest.raises(ValueError, match="carries no workload"):
        Registered(Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY}, workload={"requests": [{"prompt_len": 1}]}), 7)


def test_the_traced_side_is_one_build_a_distinct_program(tmp_path):
    reg = _registered()
    s = compile_steps(reg, *_mixed()).summary()
    cmds = trace_commands(s, reg, str(tmp_path))
    assert [c[c.index("--out") + 1].rsplit("/", 1)[1] for c in cmds] == ["build_request_LP8_T3_greedy", "build_request_LP5_T2_t0.8-p1"]
    assert json.loads(cmds[1][cmds[1].index("--sampling") + 1]) == {"temperature": 0.8, "top_p": 1.0, "top_k": 0}
    assert json.loads(cmds[0][cmds[0].index("--target") + 1]) == RTXPRO6000
    (tmp_path / trace_dir(s["requests"][0])).mkdir()
    (tmp_path / trace_dir(s["requests"][0]) / "result.json").write_text(json.dumps({"program": {"digest": s["requests"][0]["digest"]}}))
    (tmp_path / trace_dir(s["requests"][1])).mkdir()
    (tmp_path / trace_dir(s["requests"][1]) / "result.json").write_text(json.dumps({"ok": False, "message": "boom"}))
    got = compare(s, str(tmp_path))
    assert (got["equal"], got["different"], got["no_trace"], got["mismatches"]) == (1, 0, 1, 1)
    assert got["units"][1]["trace_error"] == "boom"


@pytest.mark.slow
def test_a_compiled_request_equals_the_traced_build():
    """SmolLM2-135M, one greedy request of 256 prompt tokens that ran 31 decodes: the compiled request Program is the one the Build
    traced (cov-k01-10, `test_predict`'s pin).  About 20 s: the only test holding a compilation to a trace."""
    cfg = json.loads((FIX / "smollm2-135m.config.json").read_text())
    reg = _registered(cfg)
    body = {**GREEDY, "max_tokens": 32, "prompt": [k % cfg["vocab_size"] for k in range(256)]}
    advice, ingress = _session([(256, 32, body, body["prompt"])], [[(0, 0, 256)]] + [[(0, 256 + k, 1)] for k in range(31)])
    c = compile_steps(reg, advice, ingress)
    assert c.requests[0].digest == ("91d5eaf172589c9a5586e4eb7b8a83e047448018a2a14bcd6e0048f0618bc0a31a390b53b33459979cb706927233124ba258c1043c76a6461ffc91f7d84"
                                    "11888")
