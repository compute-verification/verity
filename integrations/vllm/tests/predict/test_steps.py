"""verity_vllm.predict.steps (I2): a session's advice (I1 v0) and its certified request bodies compile to the Build's request Programs
at the shapes they ran, its rows to references into them, and every other input is refused by name: ill-formed advice, advice the
bodies contradict, a forged cache hit, and what v0 can't compile.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from verity_vllm.predict.build import Unexpressible, build_program
from verity_vllm.predict.common import _arr
from verity_vllm.predict.phi import Phi
from verity_vllm.predict.steps import (
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

from verity.core.protocols.certifier import advice as I1

FIX = Path(__file__).parent / "fixtures"
RTXPRO6000 = {"compute_capability": [12, 0], "num_sms": 188}
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
        "max_position_embeddings": 64, "hidden_act": "silu"}
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": None, "max_tokens": 4}
STOCH = {"temperature": 0.8, "top_p": 0.95, "top_k": 0, "seed": None, "max_tokens": 3, "min_p": 0.0, "repetition_penalty": 1.0}
STOCH_POLICY = {"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "splits": "derived"}


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
    assert policy_of(0, {**STOCH, "seed": 123}, 7) == (STOCH_POLICY, 123)
    assert policy_of(0, {**STOCH, "top_k": -1}, 7) == (STOCH_POLICY, 7)
    assert policy_of(0, {**GREEDY, "top_p": 0.5, "stop": ["\n"], "ignore_eos": False}, 7) == ({"kind": "greedy"}, None)


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
    ({**STOCH, "temperature": "0.8"}, "temperature '0.8', top_p 0.95 and seed None aren't"),
    ({**STOCH, "top_p": 0.0}, "top_p 0.0"),
    ({**STOCH, "seed": -1}, "seed -1"),
    (["temperature"], "aren't an object of named values"),
])
def test_sampling_v0_cannot_express_is_refused_by_name(sampling, fact):
    with pytest.raises(Unexpressible, match=f"request 0: .*{fact}".replace("(", r"\(").replace(")", r"\)").replace("[", r"\[")):
        policy_of(0, sampling, 7)


def test_sampling_is_the_bodys_never_the_advices():
    """A request's policy is its certified body's. Advice is I1's bytes or an Advice, neither of which has a sampling field: an
    advice-shaped object carrying sampling is refused, and so are I1 bytes with sampling appended."""
    reg, (advice, ingress) = _registered(), _mixed()
    hot = {**ingress, "q0": {**ingress["q0"], "temperature": 0.8, "top_p": 0.95, "seed": 3}}
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
    with pytest.raises(Unexpressible, match="ill-formed advice: advice version 1"):
        compile_steps(reg, bytes([1]) + data[1:], ingress)
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
    prefill computes rows 16..23 at positions 16.., and its decode rows are as without a hit."""
    reg = _registered()
    c = compile_steps(reg, *_hit(16))
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


@pytest.mark.parametrize("reqs,steps,fact", [
    ([(8, 2, GREEDY)], [[(0, 0, 3)], [(0, 3, 5)], [(0, 8, 1)]], "a chunked prefill, 3 of its 8 uncached prompt tokens at step 0"),
    ([(62, 4, GREEDY)], [[(0, 0, 62)], [(0, 62, 1)], [(0, 63, 1)], [(0, 64, 1)]],
     "request 0: positions to 64, past the model's 64 positions"),
])
def test_what_v0_cannot_run_is_refused_by_name(reqs, steps, fact):
    with pytest.raises(Unexpressible, match=fact):
        compile_steps(_registered(), *_session(reqs, steps))
    compile_steps(_registered(), *_session([(60, 4, GREEDY)], [[(0, 0, 60)], [(0, 60, 1)], [(0, 61, 1)], [(0, 62, 1)]]))


def test_a_gap_between_decodes_and_a_preemption_are_refused():
    reg = _registered()
    gap = _session([(8, 2, GREEDY), (4, 1, {**GREEDY, "max_tokens": 1})], [[(0, 0, 8)], [(1, 0, 4)], [(0, 8, 1)]])
    with pytest.raises(Unexpressible, match=r"request 0: its row 1 is \(step, start, n\) = \(2, 8, 1\), not the decode \(1, 8, 1\)"):
        compile_steps(reg, *gap)
    resumed = _session([(8, 3, GREEDY)], [[(0, 0, 8)], [(0, 8, 1)], [(0, 0, 10, {"preempted": True})]])
    with pytest.raises(Unexpressible, match="step 2: request 0 preempted"):
        compile_steps(reg, *resumed)


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


def test_partial_refuses_request_by_request():
    """A measurement over served traffic: request 1's hit of a part of a block, request 2's preemption and request 3's top_k refuse
    those requests by name, their rows leave the steps, and request 0 compiles as it would alone."""
    p0 = _prompt(0, 8)
    advice, ingress = _session([(8, 4, GREEDY, p0), (8, 2, GREEDY, p0[:4] + [9] * 4), (8, 3, GREEDY), (5, 2, {**STOCH, "top_k": 50})], [
        [(0, 0, 8), (1, 4, 4, {"cache_hit": 4, "sources": (I1.Source(0, 4),)}), (2, 0, 8)],
        [(0, 8, 1), (1, 8, 1), (2, 8, 1), (3, 0, 5)],
        [(0, 9, 1), (3, 5, 1)],
        [(0, 10, 1), (2, 0, 10, {"preempted": True})],
    ])
    reg = _registered()
    with pytest.raises(Unexpressible, match="step 3: request 2 preempted"):
        compile_steps(reg, advice, ingress)
    c = compile_steps(reg, advice, ingress, partial=True)
    assert sorted(c.refused) == [1, 2, 3]
    assert "not whole blocks of 16" in c.refused[1] and "step 3: request 2 preempted" in c.refused[2] and "top_k=50" in c.refused[3]
    assert [u is not None for u in c.requests] == [True, False, False, False]
    alone = compile_steps(reg, *_session([(8, 4, GREEDY, p0)], [[(0, 0, 8)], [(0, 8, 1)], [(0, 9, 1)], [(0, 10, 1)]]))
    assert c.steps == alone.steps and c.programs.keys() == alone.programs.keys()
    assert sorted(c.summary()["refused"]) == ["1", "2", "3"]
    with pytest.raises(Unexpressible, match=r"every request is refused, request 0 first: request 0 \('q0'\): no certified request body"):
        compile_steps(reg, *_hit(16)[:1], {"q1": _hit(16)[1]["q1"]}, partial=True)


def test_dead_gates_are_counted_as_circuit_check_counts_them():
    """The request wrapper's prefill runs the last layer on every prompt row and only the last row reaches the LM head, so
    today's request Programs have dead gates: `dead_gates_of` counts them as circuit-check does and says under which definitions."""
    from verity.ml.circuits.liveness import dead_gates
    c = compile_steps(_registered(), *_mixed())
    got = dead_gates_of(c)
    assert sorted(got["programs"]) == sorted(c.programs)
    for d, p in c.programs.items():
        rep = dead_gates(p, limit=0, transitive=True)
        rec = got["programs"][d]
        assert (rec["dead_computed"], rec["unread_inputs"]) == (rep.computed, rep.input) and rep.computed > 0
        assert sum(rec["by_definition"].values()) == rep.computed + rep.input
        by = rec["by_definition"]
        assert next(iter(by)) == "Gemm_v2" and sum(n for k, n in by.items() if k.startswith("Input")) == rep.input
    assert got["dead_computed"] == sum(r["dead_computed"] for r in got["programs"].values())


def test_a_target_without_batch_invariance_is_refused():
    with pytest.raises(Unexpressible, match="a target without batch invariance"):
        compile_steps(_registered(execution="eager"), *_mixed())
    with pytest.raises(ValueError, match="carries no workload"):
        Registered(Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY}, workload={"requests": [{"prompt_len": 1}]}), 7)


def test_the_traced_side_is_one_build_a_distinct_program(tmp_path):
    reg = _registered()
    s = compile_steps(reg, *_mixed()).summary()
    cmds = trace_commands(s, reg, str(tmp_path))
    assert [c[c.index("--out") + 1].rsplit("/", 1)[1] for c in cmds] == ["build_request_LP8_T3_greedy", "build_request_LP5_T2_t0.8-p0.95"]
    assert json.loads(cmds[1][cmds[1].index("--sampling") + 1]) == {"temperature": 0.8, "top_p": 0.95, "top_k": 0}
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
    assert c.requests[0].digest == ("52e6bb0b37e31ec3bdcc976027f412955be60cdd7bcd1ed4982aee7b4b716681a00323889897a394808aa040994f53a3e210dc6"
                                    "03dba9a65751817e1df348f5c")
