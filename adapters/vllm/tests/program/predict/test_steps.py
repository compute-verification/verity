"""verity_vllm.program.predict.steps (I2): a session's advice (I1 v0) and its certified request bodies compile to the Build's request Programs
at the shapes they ran, its rows to references into them, and every other input is refused by name: ill-formed advice, advice the
bodies contradict, a forged cache hit, and what v0 can't compile.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from verity_vllm.config import MANIFESTS, WORKLOADS, RowEngine
from verity_vllm.engine.profiles.dense_generic import HF_CONFIG_DIR
from verity_vllm.engine.served_seed import derived
from verity_vllm.pipeline import launch_context as LC
from verity_vllm.pipeline import window_advice as WA
from verity_vllm.pipeline.build_instances import check_rows
from verity_vllm.program.predict import batch as B
from verity_vllm.program.predict import score as S
from verity_vllm.program.predict.build import Unexpressible, _Specs, build_program, model_shape
from verity_vllm.program.predict.common import _arr
from verity_vllm.program.predict.domain import check_phi
from verity_vllm.program.predict.dsa import Net
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.frontend import seed_rule as SEED, stop_rule as SR
from verity_vllm.program.frontend.targets import parse_launch_context
from verity_vllm.program.predict.steps import (
    Registered,
    RowRef,
    compare,
    compile_steps,
    launch_class,
    plan_steps,
    policy_of,
    request_program,
    stop_of,
    trace_commands,
    trace_dir,
)
from verity_vllm.program.prune import dead_free_versions
from verity.definitions.registry import io_shell as IO
from verity.definitions.registry import stop_check as SC
from verity.definitions.registry.topp_geometry import registered_splits

import verity.protocols.certifier.verifier.advice as I1
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.partition_object import calls
from verity.circuits.refs import runs

pytestmark = pytest.mark.usefixtures("code_versions_stood_in")

FIX = Path(__file__).parent / "fixtures"
RTXPRO6000 = {"compute_capability": [12, 0], "num_sms": 188}
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
        "max_position_embeddings": 64, "hidden_act": "silu", "eos_token_id": 255}
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": None, "max_tokens": 4}
STOCH = {"temperature": 0.8, "top_p": 1.0, "top_k": 0, "seed": 11, "max_tokens": 3, "min_p": 0.0, "repetition_penalty": 1.0}
SALT = bytes(range(64))
STOCH_POLICY = {"kind": "stochastic", "temperature": 0.8, "top_p": 1.0, "top_k": 0, "seeded": True, "splits": registered_splits(188),
                "rid": 2}
SERVED = "served-model"
GLM53_B8 = "glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__greedy__eager"
CLASS_G_ENGINE = {"max_model_len": 131072, "max_num_batched_tokens": 2048, "long_prefill_token_threshold": 256, "max_num_seqs": 256,
                  "enable_prefix_caching": False}


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
    assert policy_of(0, {**STOCH, "seed": 123}, 188, request_id="q1") == (STOCH_POLICY, 123)
    assert policy_of(0, {**STOCH, "top_p": 0.95}, 188, request_id="q1") == ({**STOCH_POLICY, "top_p": 0.95}, 11)
    assert registered_splits(188) == 32
    assert policy_of(0, {**STOCH, "top_k": -1, "top_p": None}, 188, request_id="q1") == (STOCH_POLICY, 11)
    assert policy_of(0, {**GREEDY, "top_p": 0.5, "stop": ["\n"], "ignore_eos": False}, None, request_id="q1") == ({"kind": "greedy"}, None)
    with pytest.raises(Unexpressible, match="top-p on a target that declares no num_sms"):
        policy_of(0, STOCH, None, request_id="q1")


def test_an_unseeded_stochastic_requests_seed_is_derived_from_its_id_and_the_window_seed_salt_or_refused_by_name():
    """A seed is its certified body's or derived from the request's id and the window's seed salt, never one serving fills in
    or draws: with no seed salt, or for a request the window carries in (whose admitting window's seed salt this compile doesn't
    read), an unseeded stochastic request is refused by name. The derived seed is the request's alone and in its batch."""
    unseeded = {**STOCH, "seed": None}
    assert policy_of(0, unseeded, 188, request_id="q1", salt=SALT) == (STOCH_POLICY, derived(SALT, "q1"))
    assert derived(SALT, "q1") != derived(SALT, "q2") and derived(SALT, "q1") != derived(SALT[::-1], "q1")
    assert policy_of(0, {**unseeded, "seed": 5}, 188, request_id="q1", salt=SALT) == (STOCH_POLICY, 5)
    with pytest.raises(Unexpressible, match=r"request 0 \('q1'\): stochastic with no seed in its certified body, and the compile has "
                                            r"no window seed salt to derive one from: a seed serving fills in \(`served_seed.DEFAULT_SEED`\) "
                                            r"or draws"):
        policy_of(0, unseeded, 188, request_id="q1")
    with pytest.raises(Unexpressible, match="request 0 \\('a'\\): stochastic with no seed in its certified body, and the window carries "
                                            "it in, so its seed derives from its admitting window's seed salt"):
        plan_steps(_registered(), A_IN, {"a": {**unseeded, "max_tokens": 4, "prompt": [1, 2, 3]}}, salt=SALT)
    reg, (advice, ingress) = _registered(), _mixed()
    ingress = {**ingress, "q1": {**ingress["q1"], "seed": None}}
    with pytest.raises(Unexpressible, match=r"request 1 \('q1'\): stochastic with no seed in its certified body"):
        compile_steps(reg, advice, ingress)
    assert compile_steps(reg, advice, ingress, salt=SALT).requests[1].seed == derived(SALT, "q1")
    with pytest.raises(Unexpressible, match=r"request 1 \('q1'\): .* the window's seed salt isn't non-empty bytes \(bytes b''\)"):
        compile_steps(reg, advice, ingress, salt=b"")


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
    ({**STOCH, "temperature": "0.8"}, "temperature '0.8', top_p 1.0 and seed 11 aren't"),
    ({**STOCH, "top_p": 0.0}, "top_p 0.0"),
    ({**STOCH, "seed": -1}, "seed -1"),
    (["temperature"], "aren't an object of named values"),
])
def test_sampling_v0_cannot_express_is_refused_by_name(sampling, fact):
    with pytest.raises(Unexpressible, match=f"request 0: .*{fact}".replace("(", r"\(").replace(")", r"\)").replace("[", r"\[")):
        policy_of(0, sampling, 188, request_id="q1")


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


def _stops(body: dict = GREEDY) -> tuple[I1.Advice, dict]:
    """r0 (LP 8) finished on a stop id at its 3rd output, under max_tokens 4; r1, its shape under `body`, kept its max_tokens 3."""
    return _session([(8, 3, GREEDY), (8, 3, {**body, "max_tokens": 3})],
                    [[(0, 0, 8), (1, 0, 8)], [(0, 8, 1), (1, 8, 1)], [(0, 9, 1), (1, 9, 1)]])


def _spec(LP: int, T: int, m: int, finish: int | None = None, stops: tuple[int, ...] = (255,), M: int = 64) -> dict:
    """The length rule's config of a TINY request Program with outputs 1..T + 1 under max_tokens `m`, finishing as `_session` has
    it (on a stop id below `m` outputs, on length at `m`) unless `finish` (a `stop_rule.FINISH` code) says."""
    f = finish if finish is not None else 2 if T + 1 == m else 1
    return SR.spec(stops, max_tokens=m, prompt_len=LP, max_model_len=M, outputs=T + 1, finish=f)


def _rule(program):
    return program.fn.body.nodes[-1]


def test_each_request_program_returns_its_length_rule_as_its_last_check():
    """Each request Program's last Call is `StopCheck_v1` (of `StopRule_v1`) over exactly the outputs it returns, at the registered
    model's stop ids, its max_tokens and prompt length and the model's length, FINISH as the request ended: r0 on a stop id (1), r1
    at its max_tokens (2).  It is the Program's last check, at the
    last returned leaf, and requests of one shape under different finishes are different Programs with different traces."""
    reg, (advice, ingress) = _registered(), _stops()
    c = compile_steps(reg, advice, ingress)
    u0, u1 = c.requests
    assert (u0.stop, u1.stop) == (_spec(8, 2, 4), _spec(8, 2, 3)) and (u0.stop["finish"], u1.stop["finish"]) == (1, 2)
    assert (u0.LP, u0.T, u0.policy) == (u1.LP, u1.T, u1.policy) and u0.digest != u1.digest and u0.key() != u1.key()
    for u in (u0, u1):
        body = c.programs[u.digest].fn.body
        rule = _rule(c.programs[u.digest])
        assert rule.fn.id == bind(SC.StopCheck, T=3, STOP=(255,), MAX_TOKENS=u.stop["max_tokens"], PROMPT_LEN=8, MAX_MODEL_LEN=64,
                                  FINISH=u.stop["finish"]).id and rule.name == SR.NAME
        assert runs(rule.args[0].refs) == runs(body.ret.slice(0, 3))
        assert check_rows(body, 1) == [[len(body.nodes) - 1, 0, 1]]
        assert request_program(reg, u).checks == {"output": 1, "accept": 1, "of": [SR.what(u.stop)]}
    s = c.summary()["requests"]
    assert [r["stop"] for r in s] == [u0.stop, u1.stop] and trace_dir(s[0]) != trace_dir(s[1])
    cmds = trace_commands(c.summary(), reg, "/out")
    assert sorted(cmd[cmd.index("--stop") + 1] for cmd in cmds) == sorted(json.dumps(x, sort_keys=True, separators=(",", ":"))
                                                                          for x in (u0.stop, u1.stop))


@pytest.mark.parametrize("finish, ids, ok", [
    (1, [11, 12, 255], 1),        # an honest eos finish
    (2, [11, 12, 13], 1),         # an honest length finish
    (1, [11, 12, 13], 0),         # cut short before its first stop
    (1, [11, 255, 255], 0),       # run past a stop to finish on another
    (2, [11, 255, 13], 0),        # run past a stop to its cap
    (2, [11, 12, 255], 0),        # a stop at its cap read as a length finish
])
def test_a_programs_own_length_rule_passes_an_honest_finish_and_refuses_a_stream_cut_short_or_run_past_a_stop(finish, ids, ok):
    """The rule each Program carries, evaluated on the streams its 3 outputs could hold: 1 only for the stream that ends the way
    the request did."""
    reg, (advice, ingress) = _registered(), _stops()
    c = compile_steps(reg, advice, ingress)
    u = next(u for u in c.requests if u.stop["finish"] == finish)
    assert int(evaluate_batch(_rule(c.programs[u.digest]).fn, np.array([ids], dtype=np.uint32), kernel="reference")[0]) == ok


@pytest.mark.parametrize("body, why", [
    ({"stop": ["\n"]}, r"stop strings \['\\n'\]"),
    ({"stop": "\n"}, r"stop strings '\\n'"),
    ({"stop_token_ids": [7]}, r"stop_token_ids \[7\]"),
])
def test_a_body_with_stops_of_its_own_is_refused_naming_the_fix(body, why):
    reg, (advice, ingress) = _registered(), _stops()
    ingress["q1"].update(body)
    with pytest.raises(Unexpressible, match=f"request 1: {why}.*The fix: "):
        compile_steps(reg, advice, ingress)


def test_ignore_eos_leaves_a_request_no_stop_ids_and_a_rule_over_its_length_caps():
    """Under ignore_eos vLLM stops a request only on its own stop_token_ids (`SamplingParams.update_from_generation_config`), which
    are refused, so it has none: its rule reads the length caps alone, and a finish on a stop id is refused.  Empty stops are none,
    and compile as without them."""
    reg, (advice, ingress) = _registered(), _stops()
    plain = compile_steps(reg, advice, ingress)
    empty = {**ingress, "q1": {**ingress["q1"], "stop": [], "stop_token_ids": [], "ignore_eos": False}}
    assert [u.digest for u in compile_steps(reg, advice, empty).requests] == [u.digest for u in plain.requests]
    ignored = compile_steps(reg, advice, {**ingress, "q1": {**ingress["q1"], "ignore_eos": True}})
    u1 = ignored.requests[1]
    assert u1.stop == _spec(8, 2, 3, stops=()) and _rule(ignored.programs[u1.digest]).fn.id == bind(
        SC.StopCheck, T=3, STOP=(), MAX_TOKENS=3, PROMPT_LEN=8, MAX_MODEL_LEN=64, FINISH=2).id
    with pytest.raises(Unexpressible, match="request 0: finished on a stop token with ignore_eos set"):
        compile_steps(reg, advice, {**ingress, "q0": {**ingress["q0"], "ignore_eos": True}})
    with pytest.raises(Unexpressible, match="request 1: ignore_eos 'yes', not a bool"):
        compile_steps(reg, advice, {**ingress, "q1": {**ingress["q1"], "ignore_eos": "yes"}})


def test_a_model_that_states_no_stop_ids_has_a_rule_over_its_length_caps_and_refuses_a_finish_on_one():
    bare = _registered({k: v for k, v in TINY.items() if k != "eos_token_id"})
    advice, ingress = _stops()
    with pytest.raises(Unexpressible, match="request 0: finished on a stop id, and the registered model states none"):
        compile_steps(bare, advice, ingress)
    c = compile_steps(bare, *_session([(8, 3, {**GREEDY, "max_tokens": 3})], [[(0, 0, 8)], [(0, 8, 1)], [(0, 9, 1)]]))
    (u,) = c.requests
    assert u.stop == _spec(8, 2, 3, stops=()) and _rule(c.programs[u.digest]).name == SR.NAME


@pytest.mark.parametrize("T, ends", [(4, True), (2, False), (0, False)])
def test_a_request_ends_only_in_the_program_that_holds_its_last_output(T, ends):
    """A window that carries a request out before its last output has its Program run on there (finish 0); the one holding its
    last ends it as it finished: 1 on a stop id, 2 at its max_tokens, 3 at the model's length first, and an abort runs on."""
    for finished, code, m, M in (("eos", 1, 9, 64), ("length", 2, 5, 64), ("length", 3, 9, 13), ("abort", 0, 9, 64)):
        want = SR.spec((255,), max_tokens=m, prompt_len=8, max_model_len=M, outputs=T + 1, finish=code if ends else 0)
        assert stop_of(0, WA.RequestAdvice("q0", 8, 5, finished), T, (255,), m, M) == want


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
    assert (u.LP, u.T, u.C, u.producers, u.digest) == (24, 2, 16, (0,), _whole(reg, 24, 2, 3, C=16))
    assert u.digest != _whole(reg, 24, 2, 3)
    params = [(n, t) for n, t in c.programs[u.digest].fn.signature()[0]]
    assert [n for n, _ in params] == ["prompt", "prefix", "weights"] and repr(params[1][1]) == repr(_arr((2, 2, 16, 32), 16))
    assert c.programs[u.digest].fn.bindings["SRC"]["class"] == "verity-vllm/cached-greedy-request/v1"
    assert c.steps[1] == [RowRef(0, c.requests[0].digest, 24, 1, 1), RowRef(1, u.digest, 16, 8, 1)]
    assert c.workload.requests[1].prefix == {"rows": 16, "producers": [0]} and c.workload.requests[0].prefix is None
    s = c.summary()
    assert s["requests"][1]["C"] == 16 and trace_dir(s["requests"][1]).rpartition("_s")[0] == "build_request_LP24_T2_C16_greedy"
    [_, cmd] = trace_commands(s, reg, "/t")
    assert cmd[cmd.index("--cached") + 1] == "16" and "--cached" not in trace_commands(s, reg, "/t")[0]


@pytest.mark.parametrize("body", [GREEDY, STOCH])
def test_a_recomputed_cache_hit_compiles_to_the_uncached_request_program(body):
    """With `recompute_hits`, r1's hit of r0's first block is its own Program's to compute: the uncached request Program (C 0), with
    no `prefix` input and no producer in GP-01, stochastic too, its rows still from the hit (positions 0..15 in no row). R2 passes,
    the stand-in computing the block from r1's own tokens."""
    reg = _registered()
    c = compile_steps(reg, *_hit(16, body=body), recompute_hits=True)
    assert c.batch_rule is None and not c.refused
    u = c.requests[1]
    alone = compile_steps(reg, *_session([(24, 3, {**body, "max_tokens": 3}, _prompt(0, 16) + _prompt(1, 24)[16:])],
                                         [[(0, 0, 24)], [(0, 24, 1)], [(0, 25, 1)]]))
    assert (u.LP, u.T, u.C, u.recomputed, u.producers, u.digest) == (24, 2, 0, 16, (), alone.requests[0].digest)
    assert "prefix" not in [n for n, _ in c.programs[u.digest].fn.signature()[0]]
    assert c.steps[1] == [RowRef(0, c.requests[0].digest, 24, 1, 1), RowRef(1, u.digest, 16, 8, 1)]
    assert [r.prefix for r in c.workload.requests] == [None, None]
    s = c.summary()
    assert s["requests"][1]["recomputed"] == 16 and all("--cached" not in cmd for cmd in trace_commands(s, reg, "/t"))
    if body is GREEDY:
        assert u.digest == _whole(reg, 24, 2, 3)


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


def _whole(reg: Registered, LP: int, T: int, m: int, C: int = 0, **kw) -> str:
    """The greedy request Program under max_tokens `m`, its length rule `_spec`'s."""
    return build_program(Phi.from_json({**reg.phi.to_json(), "sampling": {"kind": "greedy"}}), "request", LP=LP, T=T, C=C,
                         stop=_spec(LP, T, m, **kw)).digest


def test_a_chunked_prefill_compiles_to_the_whole_prefill_program():
    """Request 0's prompt of 8 runs as chunks of 3 and 5, the first's sample discarded: its Program is the one it has unchunked, and
    each chunk is its prefill rows in turn. Request 1's chunks follow its prefix-cache hit of request 0's first block."""
    reg = _registered()
    c = compile_steps(reg, *_session([(8, 2, GREEDY)], [[(0, 0, 3)], [(0, 3, 5)], [(0, 8, 1)]]))
    d = _whole(reg, 8, 1, 4)
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
    d = _whole(reg, 40, 1, 2, C=16)
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
    assert c.requests[0].digest == d == _whole(reg, 8, 2, 4)
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
    d = _whole(reg, 20, 15, 16)
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
    d = _whole(reg, 8, 2, 4)
    assert (c.requests[0].digest, c.requests[0].T, c.workload.requests[0].cap) == (d, 2, 3)
    assert c.steps[3] == [] and c.discarded == [(3, 0, 10, 1)] and c.summary()["discarded"] == [[3, 0, 10, 1]]
    c = compile_steps(reg, *_session([(8, 4, GREEDY)], steps))
    assert (c.requests[0].T, c.discarded, len(c.steps[3])) == (3, [], 1)


def test_an_aborted_request_compiles_as_one_stopped_at_its_last_kept_output():
    """Request 0's client aborted it after 3 outputs, and the endpoint freed it after the step in flight, whose row (step 3) ran past
    them: it compiles to the rows of a stop at 3 outputs, that row discarded, with ignore_eos set or not. Its length rule runs on
    past its last output (finish 0) where the stop's ends there on a stop id (1), under ignore_eos over its length caps alone. At its
    max_tokens vLLM has finished it on length, so an abort there is refused."""
    reg = _registered()
    stopped, ingress = _session([(8, 3, GREEDY)], [[(0, 0, 8)], [(0, 8, 1)], [(0, 9, 1)], [(0, 10, 1)]])
    aborted = WA.Advice((WA.RequestAdvice("q0", 8, 3, "abort"),), stopped.steps, stopped.release)
    c = compile_steps(reg, stopped, ingress)
    a, d = compile_steps(reg, aborted, ingress), compile_steps(reg, aborted, {"q0": {**ingress["q0"], "ignore_eos": True}})
    assert [[(u.digest, u.T, u.stop) for u in x.requests] for x in (c, a, d)] == [
        [(_whole(reg, 8, 2, 4, **k), 2, _spec(8, 2, 4, **k))] for k in ({}, {"finish": 0}, {"finish": 0, "stops": ()})]
    assert len({tuple((k, r.request, r.start, r.n, r.samples) for k, s in enumerate(x.steps) for r in s) for x in (c, a, d)}) == 1
    assert a.discarded == d.discarded == c.discarded == [(3, 0, 10, 1)]
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


def _dead_gates(compiled) -> dict:
    """Each compiled Program's dead gates as circuit-check counts them (`liveness.dead_gates`, transitive: gates no output depends
    on), with the unread inputs apart, and the dead gates under each of the root's nodes by definition (its id up to `{`)."""
    from verity.circuits.defs import PrimitiveDefinition
    from verity.circuits.intervals import full, total
    from verity.circuits.liveness import Liveness, dead_gates
    out = {}
    for digest, program in sorted(compiled.programs.items()):
        rep = dead_gates(program, limit=0, transitive=True)
        root, lv = program.root, Liveness(transitive=True)
        res = lv.analyse(root, full(root.body.ret_type.leaves))
        by: dict[str, int] = {}
        for node, live in zip(root.body.nodes, res.node_live):
            if node.form == "call":
                n = lv.analyse(node.fn, live).dead
            elif node.form == "primitive" or isinstance(node.fn, PrimitiveDefinition):
                n = (node.n if node.form == "batch" else 1) - total(live)
            else:
                n = 0
            if n:
                name = node.fn.id.split("{", 1)[0]
                by[name] = by.get(name, 0) + n
        if rest := res.dead - sum(by.values()):
            by["(batch or scan at the root)"] = rest
        out[digest] = {"dead_computed": rep.computed, "unread_inputs": rep.input, "approximate": rep.approximate,
                       "by_definition": dict(sorted(by.items(), key=lambda kv: -kv[1]))}
    return {"programs": out, "dead_computed": sum(p["dead_computed"] for p in out.values())}


def test_dead_gates_are_counted_as_circuit_check_counts_them():
    """The request wrapper is in the dead-free form (`program.prune.dead_free`: no dead unit, RMSNormTriton, TokenSelect, the top-p
    select at a constant split count and FA2's attention at their versions without dead gates, and the last layer's fused qkv on the
    prompt rows before the last narrowed to its keys and values), so a request Program's only dead gates are inputs no gate reads:
    `_dead_gates` counts them as circuit-check does and says under which definitions. The second session's prefill has a row of
    16 keys, where `Attention_v5` has 2 dead gates a head."""
    from verity.circuits.liveness import dead_gates
    from verity_vllm.program.prune import dead_units
    sixteen = _session([(16, 2, {**GREEDY, "max_tokens": 2})], [[(0, 0, 16)], [(0, 16, 1)]])
    for advice, ingress in (_mixed(), sixteen):
        c = compile_steps(_registered(), advice, ingress)
        got = _dead_gates(c)
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
    monkeypatch.setattr(PB, "dead_free_form", lambda b, **_: (b, list(range(len(b.nodes)))))
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
        assert c.discarded == one.discarded and _dead_gates(c)["dead_computed"] == 0
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


def test_a_target_without_batch_invariance_outside_class_g_is_refused():
    with pytest.raises(Unexpressible, match=re.escape("a target without batch invariance outside class G ('eager'), whose launches the "
                                                      "advice doesn't fix: model_type 'llama', not 'glm_moe_dsa'; world 1, not 8")):
        compile_steps(_registered(execution="eager"), *_mixed())
    with pytest.raises(ValueError, match="carries no workload"):
        Registered(Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY}, workload={"requests": [{"prompt_len": 1}]}), 7)


def _class_g(hf_home: Path, rank: int = 0, engine: dict | None = None, layers: int | None = None, **edit: dict) -> Registered:
    """GLM-5.3's B8 row of record at `rank` as `score.phi_of` makes its φ (the checkpoint's config.json the package's byte copy, or
    with `layers` its first `layers` decoder layers), registered with class G's engine limits (`engine` over them, None dropping
    one; `edit` over φ's other parts)."""
    w = json.loads((WORKLOADS / f"{GLM53_B8}.json").read_text())
    cp = Path(S.hf_config_path(str(hf_home), w["model"], RowEngine.of(w).revision))
    cp.parent.mkdir(parents=True, exist_ok=True)
    cfg = (HF_CONFIG_DIR / "GLM53.config.json").read_bytes()
    cp.write_bytes(cfg if layers is None else json.dumps({**json.loads(cfg), "num_hidden_layers": layers}).encode())
    phi = S.phi_of({"row": GLM53_B8, "workload": w}, hf_home=str(hf_home)).to_json()
    eng = {k: v for k, v in {**phi["engine"], **CLASS_G_ENGINE, **(engine or {})}.items() if v is not None}
    phi = {**phi, "tp": {**phi["tp"], "rank": rank}, "engine": eng}
    return Registered.of(Phi.from_json({**phi, **{k: {**phi[k], **v} for k, v in edit.items()}}), default_seed=7)


def _b8_window() -> tuple[I1.Advice, dict, dict]:
    """The B8 row's recorded schedule as one window's advice and certified bodies, and the launch context its Build traced each
    request shape at (`launch_context.shape_contexts`): its requests at their realized arrivals, each engine step's rows in
    admission order (the scheduler's, which the launch keeps for its decodes), each request to its max_tokens (ignore_eos), greedy."""
    w = json.loads((WORKLOADS / f"{GLM53_B8}.json").read_text())
    reqs = LC._requests(w)
    order = sorted(range(len(reqs)), key=lambda i: reqs[i]["arrive_step"])
    steps = [[(i, LC.start_at(reqs[i], e), LC.rows_at(reqs[i], e)) for i in order if LC.rows_at(reqs[i], e)]
             for e in range(max(r["arrive_step"] + r["max_tokens"] for r in reqs))]
    advice, ingress = _session([(r["prompt_len"], r["max_tokens"], {**GREEDY, "max_tokens": r["max_tokens"], "ignore_eos": True})
                                for r in reqs], [s for s in steps if s])
    return advice, ingress, LC.shape_contexts(w)


def _scheduled(reqs: list[tuple[int, int, int]], budget: int, threshold: int, seqs: int = 256) -> list[list[tuple[int, int, int]]]:
    """vLLM V1's scheduler (no preemption, no prefix cache) over requests (arrive_step, prompt_len, max_tokens), each engine step's
    rows as (request, start, n): its running requests in admission order, each its prompt rows left (or one decode row) up to
    `threshold` and the step's tokens left, then the arrived waiting requests in arrival order while a sequence and a token are
    left; a request leaves once it has its max_tokens outputs. Steps with nothing to run are dropped, as the idle engine brings the
    next arrival forward."""
    done, outs = [0] * len(reqs), [0] * len(reqs)
    waiting, running, steps, e = sorted(range(len(reqs)), key=lambda i: reqs[i][0]), [], [], 0
    while waiting or running:
        left, rows = budget, []
        for i in running:
            if (n := min(max(reqs[i][1] - done[i], 1), threshold, left)):
                rows.append((i, done[i], n))
                left -= n
        while waiting and reqs[waiting[0]][0] <= e and left and len(running) < seqs:
            i = waiting.pop(0)
            running.append(i)
            rows.append((i, 0, min(reqs[i][1], threshold, left)))
            left -= rows[-1][2]
        for i, start, n in rows:
            done[i] = start + n
            outs[i] += done[i] >= reqs[i][1]
        running = [i for i in running if outs[i] < reqs[i][2]]
        steps.append(rows)
        e += 1
    return [s for s in steps if s]


def _chunked_b8(budget: int, threshold: int) -> tuple[I1.Advice, dict, list]:
    """The B8 row's requests (their realized arrivals, prompts and max_tokens, greedy, ignore_eos) as one window's advice under a
    scheduler of `budget` tokens a step and a long-prefill threshold of `threshold` (`_scheduled`), its bodies and its steps."""
    reqs = LC._requests(json.loads((WORKLOADS / f"{GLM53_B8}.json").read_text()))
    steps = _scheduled([(r["arrive_step"], r["prompt_len"], r["max_tokens"]) for r in reqs], budget, threshold)
    advice, ingress = _session([(r["prompt_len"], r["max_tokens"], {**GREEDY, "max_tokens": r["max_tokens"], "ignore_eos": True})
                                for r in reqs], steps)
    return advice, ingress, steps


def test_class_g_plans_a_long_prompt_as_its_prefill_chunks_each_at_the_launch_it_ran_in(tmp_path):
    """The B8 row's requests scheduled under a long-prefill threshold of 64 and a step budget of 160 (no recorded class-G window
    chunks a prefill, so its advice is this scheduler's): each request's prefill chunks are its prompt rows in the steps that ran
    them, each at most 64, the budget cutting some; its launch context is, per chunk and then per decode, every row of that step;
    alone (R2's solo advice, each row a launch of its own) it plans to the same chunks; and each unit's trace is the Build at its
    launch context, a chunked unit's with its chunks (the chunked request wrapper), which the Build parses back."""
    advice, ingress, steps = _chunked_b8(160, 64)
    reg = _class_g(tmp_path, engine={"long_prefill_token_threshold": 64, "max_num_batched_tokens": 160})
    plan = plan_steps(reg, advice, ingress)
    assert not plan.refused and not plan.discarded
    cut = False
    for u in plan.requests:
        mine = [(s, start, n) for s, rows in enumerate(steps) for i, start, n in rows if i == u.request]
        pre = [n for _, start, n in mine if start < u.LP]
        assert sum(pre) == u.LP and max(pre) <= 64 and u.chunks == (tuple(pre) if len(pre) > 1 else ())
        assert u.batch() == [[[n, start] for _, start, n in steps[s]] for s, _, _ in mine[:len(pre) + u.T]]
        cut |= any(n < min(64, u.LP - sum(pre[:j])) for j, n in enumerate(pre))
        solo, bodies, k = B._solo(advice, ingress, u.request)
        a = plan_steps(reg, solo, bodies).requests[k]
        assert B._unit(a) == B._unit(u) and len(a.launch) == len(u.launch)
    assert cut and [u.chunks for u in plan.requests] == [(), (), (64, 64, 64, 39), (64, 1), (64, 64, 59), (), (22, 64, 64, 64, 42),
                                                         (33, 64, 4)]
    summary = {"compiler": "c", "requests": [{"LP": u.LP, "T": u.T, "C": u.C, "policy": u.policy, "launch": u.batch(),
                                              "chunks": list(u.chunks), "digest": f"d{u.request}", "stop": u.stop} for u in plan.requests]}
    (long,) = trace_commands({"requests": [{**summary["requests"][0], "LP": 2300, "T": 16}]}, reg, "/t")
    assert long[long.index("--max-model-len") + 1] == "2316"
    cmds = trace_commands(summary, reg, str(tmp_path))
    assert [c[c.index("--out") + 1] for c in cmds] == [str(tmp_path / trace_dir(u)) for u in plan.requests]
    assert trace_dir(plan.requests[2]) == trace_dir(summary["requests"][2]) != trace_dir({**summary["requests"][2], "chunks": []})
    for u, c in zip(plan.requests, cmds):
        assert c[c.index("--device-name") + 1] == reg.phi.target["device_name"] == "NVIDIA H200"
        assert c[c.index("--max-model-len") + 1] == str(max(2048, u.LP + u.T))
        ctx = parse_launch_context(c[c.index("--launch-context") + 1])
        assert (ctx.get("chunks", []), len(ctx["batch_by_step"])) == (list(u.chunks), len(u.launch))
        assert LC.bad_chunks(ctx, "request", u.LP, u.T, u.C) is None
    assert LC.bad_chunks({**ctx, "chunks": [1, *ctx.get("chunks", [u.LP])]}, "request", u.LP, u.T, u.C)["reason"] == "launch_context"
    got = compare(summary, str(tmp_path))
    assert (got["no_trace"], [x.get("trace_error") for x in got["units"]]) == (8, [None] * 8)


def test_a_chunked_request_forwards_each_chunk_at_its_launch_step_and_samples_after_the_last():
    """`common.drive` over a request in chunks: one forward a chunk at its own launch step and positions, the sample after the
    last chunk only, then each decode at the step after."""
    from verity_vllm.program.predict.common import drive
    seen = []

    class M:
        launch = NS(step=None)

        def forward(self, rows, pos0):
            seen.append(("forward", self.launch.step, pos0, len(rows)))
            return [None]

        def sample(self, row, event, pos, policy):
            seen.append(("sample", self.launch.step, event, pos))
            return NS(refs=NS(count=1))

    toks = drive(M(), "request", list(range(20)), M=1, pos0=0, LP=20, T=2, policy=None, batch=[[]] * 5, chunks=(8, 8, 4))
    assert len(toks) == 3 and seen == [("forward", 0, 0, 8), ("forward", 1, 8, 8), ("forward", 2, 16, 4), ("sample", 2, 0, 19),
                                       ("forward", 3, 20, 1), ("sample", 3, 1, 20), ("forward", 4, 21, 1), ("sample", 4, 2, 21)]


def test_the_builds_chunked_request_wrapper_runs_drives_forwards_at_its_launch_steps():
    """The Build's `vllm_meta.ChunkedGreedyRequest` under the launch context `trace_commands` gives a chunked unit: one forward a
    chunk at its launch step and positions, the sample after the last chunk only, then each decode at the step after, as `drive`
    builds it; a chunked request with a cached prefix or another policy is refused by name."""
    import torch
    from verity_vllm.program.frontend import vllm_meta as vm
    from verity_vllm.program.frontend.rules.vllm_bindings import observations as OB
    seen = []

    class Toy(torch.nn.Module):
        def forward(self, ids, positions):
            seen.append(("forward", OB._OBSERVED["launch_region"], int(positions[0]), len(ids)))
            return torch.zeros((len(ids), 4))

        def compute_logits(self, hidden):
            seen.append(("sample", OB._OBSERVED["launch_region"], len(hidden)))
            return torch.zeros((len(hidden), 8))

    OB.observe_launch_context(parse_launch_context({"batch": [[[8, 0]], [[8, 8]], [[4, 16]], [[1, 20]], [[1, 21]]], "chunks": [8, 8, 4]}))
    try:
        mod, (prompt,) = vm.make_wrapper("request", Toy(), LP=20, T=2, chunks=[8, 8, 4])
        toks = mod(torch.zeros(prompt.shape, dtype=torch.int32))
    finally:
        OB.clear_observations()
    assert type(mod) is vm.ChunkedGreedyRequest and mod.DERIVE_SOURCE == vm.GreedyRequest.DERIVE_SOURCE and len(toks) == 3
    assert seen == [("forward", (0, "tokens"), 0, 8), ("forward", (1, "tokens"), 8, 8), ("forward", (2, "tokens"), 16, 4),
                    ("sample", (2, "sampled"), 1), ("forward", (3, "tokens"), 20, 1), ("sample", (3, "sampled"), 1),
                    ("forward", (4, "tokens"), 21, 1), ("sample", (4, "sampled"), 1)]
    for kw in ({"C": 4}, {"sampling": {"kind": "stochastic"}}):
        with pytest.raises(ValueError, match="a prefill in chunks for the greedy request without a cached prefix only"):
            vm.make_wrapper("request", Toy(), LP=20, T=2, chunks=[8, 8, 4], **kw)
    with pytest.raises(ValueError, match=re.escape("prefill chunks [8, 8]: a chunked request's chunks are its 20 prompt rows")):
        vm.make_wrapper("request", Toy(), LP=20, T=2, chunks=[8, 8])


def test_prefill_chunks_are_a_request_s_prompt_rows_at_a_launch_batch():
    """A request's chunks cover its prefill rows, each at least one; one chunk is the request wrapper; and a prefill in chunks is a
    Program only where the kernels read the launch (a batch-invariant target's chunked prefill is the request wrapper's)."""
    phi = _registered().phi
    with pytest.raises(Unexpressible, match=re.escape("prefill chunks [3, 4] of a request (a request's chunks are its 8 prefill rows "
                                                      "in turn, each at least one)")):
        build_program(phi, "request", LP=8, T=1, chunks=(3, 4))
    assert build_program(phi, "request", LP=8, T=1, chunks=(8,)).digest == build_program(phi, "request", LP=8, T=1).digest
    with pytest.raises(Unexpressible, match=re.escape("a prefill in 2 chunks without a launch batch (on a batch-invariant target")):
        build_program(phi, "request", LP=8, T=1, chunks=(3, 5))


def test_a_class_g_request_whose_launch_no_capture_covers_is_refused_alone(tmp_path):
    """A class-G registration at the class's limits of 8,192 tokens a step and 131,072 positions: a request co-scheduled in a step
    of 3,000 tokens builds no Program, since no captured all-reduce covers that launch, and `partial` refuses it by name, request by
    request, before writing any launch. (A prompt past INDEX_TOPK keys compiles: `test_dsa_long_rows`.)"""
    why = "an all-reduce of 3000 x 6144 on route 'symm_mem_comm'"
    advice, ingress = _session([(10, 1, {**GREEDY, "max_tokens": 1})] + [(230, 1, {**GREEDY, "max_tokens": 1})] * 13,
                               [[(0, 0, 10)] + [(i, 0, 230) for i in range(1, 14)]])
    reg = _class_g(tmp_path, engine={"max_num_batched_tokens": 8192})
    with pytest.raises(Unexpressible, match=re.escape(why)):
        compile_steps(reg, advice, ingress)
    with pytest.raises(Unexpressible, match=re.escape(f"every request is refused, request 0 first: request 0: its Program is refused: "
                                                      f"vocabulary cannot express: {why}")):
        compile_steps(reg, advice, ingress, partial=True)


def test_class_g_plans_each_request_at_the_launch_context_its_build_traced(tmp_path):
    """GLM-5.3's B8 mixed-arrivals row, its schedule as a window's advice under a class-G registration: each request's launch context,
    read from the advice's steps, is the one its Build was traced at (the row's `--launch-context`), its eight shapes are the traced
    Build's eight request Programs, and each trace command builds at its unit's launch context, into a dir of its own."""
    reg, (advice, ingress, traced) = _class_g(tmp_path), _b8_window()
    check_phi(reg.phi, Net.DOMAIN)
    plan = plan_steps(reg, advice, ingress)
    assert not plan.refused and not plan.discarded
    assert [u.batch() for u in plan.requests] == [traced[(u.LP, u.T)] for u in plan.requests]
    record = json.loads((MANIFESTS / "records.json").read_text())["rows"][GLM53_B8]["programs"]["0"]["requests"]
    assert sorted(f"build_request_LP{u.LP}_T{u.T}" for u in plan.distinct()) == sorted(record)
    summary = {"requests": [{"LP": u.LP, "T": u.T, "C": u.C, "policy": u.policy, "launch": u.batch(), "stop": u.stop}
                            for u in plan.requests]}
    cmds = trace_commands(summary, reg, str(tmp_path))
    assert [json.loads(c[c.index("--launch-context") + 1]) for c in cmds] == [{"batch": u.batch()} for u in plan.requests]
    assert len({c[c.index("--out") + 1] for c in cmds}) == 8 and trace_dir(plan.requests[0]).startswith("build_request_LP11_T12_greedy_l")
    assert trace_dir(plan.requests[0]) == trace_dir(summary["requests"][0])


def test_class_g_samples_by_the_selects_circuit_check_binds_at_its_vocabulary(tmp_path):
    """GLM-5.3 in class G (an H200) samples a seeded top-p request at its registered split count and a greedy one by `TokenSelect`;
    `prune`'s dead-free versions of the two at its vocabulary are the bindings `dead_free.circuit-check.py` holds for it."""
    import runpy

    from verity_vllm.program.predict.common import kind
    from verity.definitions.registry import dead_free as DF

    reg = _class_g(tmp_path)
    V = reg.phi.hf_config["vocab_size"]
    policy, _ = policy_of(0, {"temperature": 1.0, "top_p": 0.95, "seed": 1}, reg.phi.target_profile().num_sms, request_id="q1")
    table = dead_free_versions()
    compiled = {bind(table[f.base_id], **f.bindings).id
                for f in (kind("token_select_stochastic_split", V=V, S=policy["splits"]), kind("token_select", V=V))}
    bound = {make().id for _, make in runpy.run_path(str(Path(DF.__file__).with_name("dead_free.circuit-check.py")))["_model_roots"]()}
    assert compiled == {"GumbelTopPTokenSelect_v3{V=154880,S=32}", "TokenSelect_v4{V=154880}"} and compiled <= bound


def test_class_g_compiles_a_request_alone_at_its_launch_context(tmp_path):
    """R2 in class G: each B8 request planned alone (`batch._solo`, each of its rows a launch of its own) is its unit in the batch
    apart from its launch context, with the same rows, while in the batch each request's launches hold another request's rows."""
    reg, (advice, ingress, _) = _class_g(tmp_path), _b8_window()
    plan = plan_steps(reg, advice, ingress)
    for r, u in enumerate(plan.requests):
        solo, bodies, k = B._solo(advice, ingress, r)
        alone = plan_steps(reg, solo, bodies)
        a = alone.requests[k]
        assert B._unit(a) == B._unit(u) and [x[1:] for x in sum(alone.steps, [])] == [x[1:] for x in sum(plan.steps, []) if x[0] == r]
        assert all(len(step) == 1 for step in a.launch) and any(len(step) > 1 for step in u.launch)


def _async_window() -> tuple[I1.Advice, dict]:
    """Two requests under async scheduling: r0 (LP 8) stops on EOS after 3 kept outputs, and the scheduler, a step ahead of its
    sample, ran one row of it past them (step 3), beside r1's decode; r1 (LP 5) keeps its max_tokens of 4."""
    return _session([(8, 3, {**GREEDY, "max_tokens": 16}), (5, 4, GREEDY)],
                    [[(0, 0, 8)], [(0, 8, 1), (1, 0, 5)], [(0, 9, 1), (1, 5, 1)], [(0, 10, 1), (1, 6, 1)], [(1, 7, 1)]])


def test_class_g_admits_async_scheduling_and_plans_the_row_it_discards_into_the_launch_it_ran_in(tmp_path):
    """A class-G registration under async scheduling (LAG 1) passes `launch_class`; in a window where the scheduler ran a row past
    r0's stop, r0 plans to its kept outputs with that row discarded, and r1's launch at that step holds it, since it ran."""
    reg = _class_g(tmp_path, target={"async_scheduling": True})
    assert reg.phi.target_profile().lag() == 1
    launch_class(reg.phi)
    plan = plan_steps(reg, *_async_window())
    assert not plan.refused and plan.discarded == [(3, 0, 10, 1)] and [u.T for u in plan.requests] == [2, 3]
    assert plan.requests[0].batch() == [[[8, 0]], [[1, 8], [5, 0]], [[1, 9], [1, 5]]]
    assert plan.requests[1].batch() == [[[1, 8], [5, 0]], [[1, 9], [1, 5]], [[1, 10], [1, 6]], [[1, 7]]]
    assert plan.steps[3] == [(1, 6, 1, 1)]


@pytest.mark.slow
def test_class_g_compiles_a_window_under_async_scheduling_with_the_row_it_discarded(tmp_path):
    """`_async_window` compiled whole under a class-G registration with async scheduling, on GLM-5.3's first four layers: both
    request Programs at their launch contexts (r1's holding r0's discarded row), GP-01 at LAG 1, the discarded row in no Program,
    and the batch rules pass. What no cheaper test can: that the Build builds a class-G Program at a launch holding a row async
    scheduling discarded, and that `compile_steps` composes them."""
    reg = _class_g(tmp_path, layers=4, target={"async_scheduling": True})
    advice, ingress = _async_window()
    c = compile_steps(reg, advice, ingress)
    assert c.batch_rule is None and not c.refused and len(c.programs) == 2 and c.discarded == [(3, 0, 10, 1)]
    assert [u.launch for u in c.requests] == [u.launch for u in plan_steps(reg, advice, ingress).requests] and c.workload.lag == 1
    assert c.steps[3] == [RowRef(1, c.requests[1].digest, 6, 1, 1)]


@pytest.mark.parametrize("engine,edit,fact", [
    ({"max_num_batched_tokens": 16384}, {}, "engine arg max_num_batched_tokens 16384, not a stated value of at most 8192"),
    ({"max_model_len": None}, {}, "engine arg max_model_len None, not a stated value of at most 131072"),
    ({"max_model_len": -1}, {}, "engine arg max_model_len -1, not a stated value of at most 131072"),
    ({"max_model_len": 262144}, {}, "engine arg max_model_len 262144, not a stated value of at most 131072"),
    ({"long_prefill_token_threshold": 512}, {}, "engine arg long_prefill_token_threshold 512, not a stated value of at most 256"),
    ({"max_num_seqs": 0}, {}, "engine arg max_num_seqs 0, not a stated value of at most 256"),
    ({"enable_prefix_caching": None}, {}, "engine arg enable_prefix_caching None, not stated True or False"),
    ({}, {"tp": {"world": 4}}, "world 4, not 8"),
    ({}, {"target": {"device_name": "NVIDIA H100 80GB HBM3"}}, "device_name 'NVIDIA H100 80GB HBM3', not 'NVIDIA H200'"),
])
def test_class_g_is_the_one_registration_without_batch_invariance_that_compiles(engine, edit, fact, tmp_path):
    with pytest.raises(Unexpressible, match=re.escape(f"a target without batch invariance outside class G ('eager'), whose launches the "
                                                      f"advice doesn't fix: {fact}")):
        plan_steps(_class_g(tmp_path, engine=engine, **edit), *_b8_window()[:2])


A = WA.RequestAdvice("a", 3, 4, "length")


def _wrow(start: int, n: int) -> WA.StepAdvice:
    return WA.StepAdvice((WA.Row(0, "decode" if n == 1 and start >= 3 else "prefill", start, n, 0, False, ()),))


A_BODY = {"a": {**GREEDY, "prompt": [1, 2, 3]}}
#: A's whole run in one window, and split across two: the first leaves it after its prefill and first decode (2 outputs kept,
#: 2 still to keep), the next carries it in at position 4, where its next row continues
A_WHOLE = WA.Advice((A,), (_wrow(0, 3), _wrow(3, 1), _wrow(4, 1), _wrow(5, 1)), (0, 0, 1, 1))
A_OUT = WA.Advice((A,), (_wrow(0, 3), _wrow(3, 1)), (0, 1))
A_IN = WA.Advice((A,), (_wrow(4, 1), _wrow(5, 1)), (0, 1), (WA.Carried(0, 4, 2, ((0, 4),)),))
#: A prefilled in chunks of 2 and 1, whole, and split inside its prompt: the first window leaves it at position 2, no output kept
A_CHUNKED = WA.Advice((A,), (_wrow(0, 2), _wrow(2, 1), _wrow(3, 1), _wrow(4, 1), _wrow(5, 1)), (0, 0, 1, 1))
A_MID_OUT = WA.Advice((A,), (_wrow(0, 2),), ())
A_MID_IN = WA.Advice((A,), (_wrow(2, 1), _wrow(3, 1), _wrow(4, 1), _wrow(5, 1)), (0, 0, 1, 1), (WA.Carried(0, 2, 0, ((0, 2),)),))
CACHE_ON = {"enable_prefix_caching": True}


def _past_its_outputs() -> tuple[WA.Advice, dict]:
    """r0 carried in finished (its 2 outputs kept), its one row here past its last kept output, and r1 admitted beside it on a hit of
    r0's first block, which r0 made in the last window."""
    p = _prompt(0, 16)
    steps = (WA.StepAdvice((WA.Row(0, "decode", 17, 1, 0, False, ()), WA.Row(1, "prefill", 16, 8, 16, False, (I1.Source(0, 16),)))),
             WA.StepAdvice((WA.Row(1, "decode", 24, 1, 0, False, ()),)))
    return (WA.Advice((WA.RequestAdvice("a", 16, 2, "length"), WA.RequestAdvice("b", 24, 2, "length")), steps, (0, 1),
                      (WA.Carried(0, 17, 2, ((0, 17),)),)),
            {"a": {**GREEDY, "max_tokens": 2, "prompt": p}, "b": {**GREEDY, "max_tokens": 2, "prompt": p + _prompt(1, 24)[16:]}})


@pytest.mark.parametrize("session,engine,fact", [
    (_session([(8, 2, GREEDY)], [[(0, 0, 3)], [(0, 0, 8, {"preempted": True})], [(0, 8, 1)]]), {},
     "request 0: step 1 runs its positions 0 … 7 resumed after its prefill reached 3, a recompute without batch invariance"),
    (_session([(8, 3, GREEDY), (4, 1, {**GREEDY, "max_tokens": 1})], [[(0, 0, 8)], [(0, 8, 1)], [(1, 0, 4)], [(0, 0, 10, {"preempted": True})]]),
     {}, "request 0: step 3 runs 10 rows of it resumed, a recompute without batch invariance"),
    (_hit(16), {}, "request 1: a prefix-cache hit of 16 tokens, and the registration states the prefix cache off"),
    (_session([(1100, 1, {**GREEDY, "max_tokens": 1}), (1000, 1, {**GREEDY, "max_tokens": 1})], [[(0, 0, 1100), (1, 0, 1000)]]), {},
     "step 0 runs 2100 rows, over the registered max_num_batched_tokens 2048"),
    (_session([(4, 1, {**GREEDY, "max_tokens": 1})] * 3, [[(0, 0, 4), (1, 0, 4), (2, 0, 4)]]), {"max_num_seqs": 2},
     "step 0 runs 3 sequences, over the registered max_num_seqs 2"),
    (_session([(300, 1, {**GREEDY, "max_tokens": 1})], [[(0, 0, 300)]]), {},
     "request 0: step 0 runs 300 of its rows, over the registered long_prefill_token_threshold 256"),
    (_past_its_outputs(), CACHE_ON, "request 1: its cache hit reads request 0's KV, which has no Program in the window (its rows here "
                                    "are past its last kept output)"),
])
def test_class_g_refuses_a_request_whose_program_misses_a_launch_it_ran_in(session, engine, fact, tmp_path):
    with pytest.raises(Unexpressible, match=re.escape(fact)):
        plan_steps(_class_g(tmp_path, engine=engine), *session)


def _hit_chunked() -> tuple[I1.Advice, dict]:
    """r1 (LP 24) admitted at step 1 on a hit of r0's first block, its 8 uncached rows prefilled in two chunks of 4."""
    return _session([(24, 4, GREEDY, _prompt(0, 24)), (24, 2, {**GREEDY, "max_tokens": 2}, _prompt(0, 16) + _prompt(1, 24)[16:])], [
        [(0, 0, 24)],
        [(0, 24, 1), (1, 16, 4, {"cache_hit": 16, "sources": (I1.Source(0, 16),)})],
        [(0, 25, 1), (1, 20, 4)],
        [(0, 26, 1), (1, 24, 1)],
    ])


@pytest.mark.parametrize("session,units", [
    (_hit(16), [(24, 3, 0, 4, (), False, False, None, ()), (24, 2, 16, 3, (0,), False, False, {"rows": 16, "producers": [0]}, ())]),
    (_hit_chunked(), [(24, 3, 0, 4, (), False, False, None, ()), (24, 1, 16, 2, (0,), False, False, {"rows": 16, "producers": [0]}, (4, 4))]),
    ((A_OUT, A_BODY), [(3, 1, 0, 2, (), False, True, None, ())]),
    ((A_IN, A_BODY), [(3, 3, 4, 2, (), True, False, {"rows": 4, "carried": True}, ())]),
    ((A_MID_OUT, A_BODY), [(3, -1, 0, 0, (), False, True, None, ())]),
    ((A_MID_IN, A_BODY), [(3, 3, 2, 4, (), True, False, {"rows": 2, "carried": True}, ())]),
])
def test_class_g_plans_a_request_from_its_history_in_the_window(session, units, tmp_path):
    """Class G with the prefix cache on: a request's unit is its rows in the window alone after its history C (`segment_of`), and so
    is the unit R2 plans it alone at (`batch._solo`). r1's hit of r0's first block has C 16, its prefill the 8 uncached rows and
    r0 the block's producer, and with those rows prefilled in two launches it is the chunked wrapper after that history; the
    window that leaves A after its prefill and first decode has A to position 3 (T 1, 2 samples); the next carries it in at C 4,
    its rows and last token there its history, and holds its last 2 decodes. The window that leaves A inside its prompt, at
    position 2, has A to there (T -1, no sample) and the next carries it in at C 2, its rows there its history, the rest of its
    prefill and all 4 samples its own."""
    reg = _class_g(tmp_path, engine=CACHE_ON)
    advice, ingress = session
    plan = plan_steps(reg, advice, ingress)
    got = [(u.LP, u.T, u.C, u.events(), u.producers, u.carried_in, u.carried_out, u.prefix(), u.chunks) for u in plan.requests]
    assert got == units
    for r, u in enumerate(plan.requests):
        solo, bodies, k = B._solo(advice, ingress, r)
        a = plan_steps(reg, solo, bodies).requests[k]
        assert B._unit(a) == B._unit(u) and (a.carried_in, a.carried_out, a.prefix() is None) == (u.carried_in, u.carried_out, not u.C)
    assert WA.follows(A_OUT, A_IN) == [(0, 0)] and WA.follows(A_MID_OUT, A_MID_IN) == [(0, 0)]


@pytest.mark.parametrize("session,chunks,entries", [(_hit(16), (16, 8), (24, 8)), (_hit_chunked(), (16, 4, 4), (24, 4, 4))])
def test_class_g_plans_a_recomputed_hit_as_chunks_at_the_launches_that_computed_it(session, chunks, entries, tmp_path):
    """With `recompute_hits`, r1's hit of r0's first block is a prefill chunk of its own Program (C 0, no producer) at the launch of
    step 0, where r0's row of 24 computed it: the chunk computes the first 16 rows of that sequence (`entries`), then r1's own
    chunks and decodes run at their launches. Alone (R2's solo form, the stand-in prefilled in the hit's chunks) it plans the same."""
    reg = _class_g(tmp_path, engine=CACHE_ON)
    advice, ingress = session
    u = plan_steps(reg, advice, ingress, recompute_hits=True).requests[1]
    assert (u.C, u.recomputed, u.hit_steps, u.chunks, u.entries, u.producers, u.prefix()) == (0, 16, (0,), chunks, entries, (), None)
    assert u.launch[:2] == (((24, 0),), ((1, 24), (chunks[1], 16)))
    solo, bodies, k = B._solo(advice, ingress, 1, u.chunks[:1])
    a = plan_steps(reg, solo, bodies, recompute_hits=True).requests[k]
    assert B._unit(a) == B._unit(u) and a.entries == () and a.launch[0] == ((16, 0),)


def _two_producers(split: bool) -> tuple[I1.Advice, dict]:
    """r2 (LP 40) admitted on a hit of 32 tokens, its first block r0's and its second r1's, which r1 computed over a first block of
    its own: in the same row as it (`split` False), or in a row of step 0 beside r0's, the second block a row of step 1."""
    p, one = _prompt(0, 40), {**GREEDY, "max_tokens": 1}
    first = [(0, 0, 16), (1, 0, 16)] if split else [(0, 0, 16), (1, 0, 32)]
    return _session([(16, 1, one, p[:16]), (32, 1, one, p[:32]), (40, 1, one, p)], [
        first, *([[(1, 16, 16)]] if split else []),
        [(2, 32, 8, {"cache_hit": 32, "sources": (I1.Source(0, 16), I1.Source(1, 16))})]])


@pytest.mark.parametrize("session,fact", [
    (_two_producers(False), "request 2: its hit's positions 16 … 31 are request 1's rows from position 0 at step 0, a row that starts "
                            "before them"),
    (_two_producers(True), "request 2: its hit's positions 16 … 31 are request 1's rows at step 1, computed over a history of other "
                           "rows than the hit holds"),
    (_past_its_outputs(), "request 0's KV at positions below 16 has no row of the window that computed it (the window carries it in)"),
])
def test_class_g_refuses_a_hit_no_one_chain_of_its_rows_recomputes(session, fact, tmp_path):
    """A recomputed hit is the chain of rows that computed it, each at its launch over the history before it, so it is refused where
    serving computed it otherwise: a block whose row started before it (two requests computed one block in the same step), a row
    whose own history was another than the hit's, or rows a window before computed."""
    with pytest.raises(Unexpressible, match=re.escape(fact)):
        plan_steps(_class_g(tmp_path, engine=CACHE_ON), *session, recompute_hits=True)


def _traced_form(reg: Registered, u) -> str:
    """`u`'s request Program digest in the form the B8 row's Build was traced in, as `aot.compare` predicts a traced request:
    `request_program`'s Program without `dead_free` (the Build without `--dead-free`), at the same launch context."""
    phi = Phi.from_json({**reg.phi.to_json(), "sampling": u.policy})
    return build_program(phi, "request", LP=u.LP, T=u.T, C=u.C, batch=u.batch()).digest


def _b8_record() -> dict:
    return json.loads((MANIFESTS / "records.json").read_text())["rows"][GLM53_B8]["programs"]["0"]["requests"]


@pytest.mark.slow
def test_a_class_g_request_compiles_to_the_traced_build(tmp_path):
    """GLM-5.3's B8 row as a window's advice under a class-G registration at rank 0, its first request (LP 11, T 12, whose last
    four decodes run beside the second request's prefill and first decodes): at the launch context the advice gives it, its
    Program in the traced Build's form is the row's record, and compiled (`partial`: the window's other requests brought no body,
    so are refused, their rows still in its launches) in the registered compiler's form, the batch rules pass. What no cheaper
    test can: that a registration without a workload, at the advice's launch, builds the Program the Build traced, and that
    `compile_steps` compiles a class-G request. Its two GLM-5.3 builds take about a minute, so the window's other seven are
    `VERITY_VLLM_CLASS_G_WINDOW`'s."""
    reg, (advice, ingress, _) = _class_g(tmp_path), _b8_window()
    u = plan_steps(reg, advice, ingress).requests[0]
    assert (u.LP, u.T) == (11, 12) and _traced_form(reg, u) == _b8_record()["build_request_LP11_T12"]
    c = compile_steps(reg, advice, {"q0": ingress["q0"]}, partial=True)
    assert c.batch_rule is None and len(c.programs) == 1 and sorted(c.refused) == list(range(1, 8))
    assert c.requests[0].launch == u.launch and [len(step) for step in u.launch] == [1] * 9 + [2] * 4


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("VERITY_VLLM_CLASS_G_WINDOW") != "1",
                    reason="VERITY_VLLM_CLASS_G_WINDOW=1: GLM-5.3's B8 window whole, its eight request Programs built in each form at "
                           "rank 0 (about half an hour and 25 GB on one core)")
def test_a_class_g_window_compiles_to_the_traced_build(tmp_path):
    """GLM-5.3's B8 row as a window's advice under a class-G registration at rank 0, whole: each of its eight request Programs, at
    the launch context the advice gives it, is in the traced Build's form the row's record (`aot.compare` over the window, one
    build at a time), and the window compiles in the registered compiler's form with the batch rules passing (R1 over GP-01 of the
    eight, R2 each alone at the same launch context)."""
    reg, (advice, ingress, _) = _class_g(tmp_path), _b8_window()
    plan = plan_steps(reg, advice, ingress)
    assert {f"build_request_LP{u.LP}_T{u.T}": _traced_form(reg, u) for u in plan.distinct()} == _b8_record()
    c = compile_steps(reg, advice, ingress)
    assert c.batch_rule is None and len(c.programs) == 8 and [u.launch for u in c.requests] == [u.launch for u in plan.requests]


def _history_params(p) -> list[tuple[str, int]]:
    return [(n, t.leaves) for n, t in p.fn.body.params if not n.startswith(("peers_", "weights"))]


@pytest.mark.slow
@pytest.mark.parametrize("body,split", [(GREEDY, (A_WHOLE, A_OUT, A_IN)), ({**STOCH, "max_tokens": 4, "seed": 5}, (A_WHOLE, A_OUT, A_IN)),
                                        (GREEDY, (A_CHUNKED, A_MID_OUT, A_MID_IN))], ids=["greedy", "seeded", "mid-prefill"])
def test_a_request_in_flight_compiles_to_its_whole_program_split_at_the_window(body, split, tmp_path):
    """Class G on GLM-5.3's first four layers (three indexer layers and one sharing the last one's indices, as in the model): A whole
    in one window is, call for call in the registered compiler's form, its Program in the window that leaves it after two outputs,
    then its Program in the next, which carries it in at position 4 with its 4 rows of every layer (latent rows and indexer keys and
    scales) and its last token as Inputs; so is A prefilled in chunks of 2 and 1, split inside its prompt at position 2 (its 2 rows
    its history, no last token). Each Program but for its seed rule, which each window's checks, and its length rule over its own outputs: A's 4 there to its max_tokens, 2 in
    the first window running on and 2 in the next from its third on, to its max_tokens; split inside its prompt, none in the first
    window and all 4 in the next. Each window's batch rules pass, R1 holding that history A's own, and its GP-01 carries the edge:
    the window that leaves A returns its latent rows so far (and past its prompt its last sample) as out-ports after its samples,
    and the next takes ports of those types as registered parameters, A's history reading them; the indexer's keys and scales,
    which no row reads below INDEX_TOPK keys, stay A's own unread Inputs. What no cheaper test can: that carrying a request across
    a Commit window neither drops nor repeats nor adds a call, nor moves a sample's position. The seeded case takes about
    two minutes, GLM-5.3's top-p sampler over its whole vocabulary, and is the only one that carries sampler state (the seed
    and its seed rule's Inputs, each sample's position) across the window."""
    from verity.circuits.types import field_types
    reg = _class_g(tmp_path, engine=CACHE_ON, layers=4)
    ingress = {"a": {**body, "prompt": [1, 2, 3]}}
    calls_of, params, wls = [], [], []
    for advice in split:
        c = compile_steps(reg, advice, ingress)
        (p,) = c.programs.values()
        assert c.batch_rule is None
        calls_of.append([n.fn.id for n in p.fn.body.nodes])
        params.append(_history_params(p))
        wls.append(c.workload)
    C = split[2].carried[0].computed
    own = [("prompt", 3)] + [("seed", 1), ("seeded", 1), ("body_seed", 64), ("seed_salt", 512), ("user", 8)] * (body is not GREEDY)
    assert params == [own, own, own + [("prefix", 4 * C * 576), ("prefix_ik", 3 * C * 128), ("prefix_iks", 3 * C)] + [("last_token", 1)] * (C >= 3)]
    stop, M = SR.stop_ids(reg.phi.hf_config), min(int(reg.phi.hf_config["max_position_embeddings"]), reg.phi.engine["max_model_len"])

    def rule(T: int, m: int, p: int, f: int) -> str:
        return bind(SC.StopCheck, T=T, STOP=stop, MAX_TOKENS=m, PROMPT_LEN=p, MAX_MODEL_LEN=M, FINISH=f).id
    rules = [rule(4, 4, 3, 2), rule(2, 4, 3, 0), rule(2, 2, 5, 2)] if C >= 3 else [rule(4, 4, 3, 2), None, rule(4, 4, 3, 2)]
    assert [ids.pop() if ids[-1].startswith("StopCheck_v1") else None for ids in calls_of] == rules
    seed_rule = None if body is GREEDY else bind(SEED.SeedCheckWord, COIN=SEED.COIN, RID=SEED.rid_of("a")).id
    assert [ids.pop() if ids[-1].startswith("SeedCheck_") else None for ids in calls_of] == [seed_rule] * 3
    assert calls_of[0] == calls_of[1] + calls_of[2]
    sig = [w.fn.signature() for w in wls]
    assert wls[0].edge is None and wls[1].edge.outs == wls[2].edge.ins and wls[2].ports == wls[2].registered[-len(wls[2].ports):]
    assert [t for n, t in sig[2][0] if n in wls[2].ports] == list(field_types(sig[1][1])[1:])
    assert wls[2].ports == ("edge.kv", "edge.last")[:1 + (C >= 3)]
    assert [n for n, _ in sig[2][0] if n.startswith("request[") and ".peers_" not in n] == [
        f"request[{wls[2].requests[0].request_id}].{g}" for g in [*(g for g, _ in own), "prefix_ik", "prefix_iks"]]


@pytest.mark.slow
def test_a_class_g_cache_hit_reads_its_cached_rows_as_inputs(tmp_path):
    """Class G on GLM-5.3's first four layers, r1 admitted on a hit of r0's first block: r1's Program reads the block's 16 rows of
    every layer as Inputs, prefills only its 8 uncached rows, and GP-01 records r0 the block's producer; the batch rules pass, R2
    with a stand-in computing the block from r1's own tokens and R1 holding the history r1's own. What no cheaper test can: that a
    DSA request compiles after a cache hit."""
    reg = _class_g(tmp_path, engine=CACHE_ON, layers=4)
    c = compile_steps(reg, *_hit(16))
    assert c.batch_rule is None and [(u.LP, u.T, u.C) for u in c.requests] == [(24, 3, 0), (24, 2, 16)]
    assert [r.prefix for r in c.workload.requests] == [None, {"rows": 16, "producers": [0]}]
    assert _history_params(c.programs[c.requests[1].digest]) == [("prompt", 24), ("prefix", 4 * 16 * 576), ("prefix_ik", 3 * 16 * 128),
                                                                 ("prefix_iks", 3 * 16)]
    assert c.requests[1].launch[0] == ((1, 24), (8, 16))


@pytest.mark.slow
def test_a_class_g_recomputed_hit_compiles_with_no_history_input(tmp_path):
    """Class G on GLM-5.3's first four layers, r1 admitted on a hit of r0's first block, under `recompute_hits`: r1's Program reads
    no history, computing the block's 16 rows in r0's launch of step 0 (24 rows at position 0) before its own 8 at step 1, and GP-01
    records no producer; the batch rules pass, R2 with the stand-in prefilled in the hit's chunk. What no cheaper test can: that a
    window's recomputed hit compiles through the batch rules on the DSA decoder."""
    reg = _class_g(tmp_path, engine=CACHE_ON, layers=4)
    c = compile_steps(reg, *_hit(16), recompute_hits=True)
    u = c.requests[1]
    assert c.batch_rule is None and [(w.LP, w.T, w.C) for w in c.requests] == [(24, 3, 0), (24, 2, 0)]
    assert [r.prefix for r in c.workload.requests] == [None, None] and (u.chunks, u.entries) == ((16, 8), (24, 8))
    assert _history_params(c.programs[u.digest]) == [("prompt", 24)]
    assert u.digest != compile_steps(reg, *_hit(16)).requests[1].digest


@pytest.mark.slow
def test_a_class_g_long_prompt_compiles_as_its_chunks_each_at_its_launch(tmp_path, monkeypatch):
    """A prompt longer than the registered long-prefill threshold, under a class-G registration at rank 0 whose threshold is 8 and
    step budget 16, as vLLM V1 schedules it (`_scheduled`): q0's 20 prompt tokens prefill in chunks of 8, 8 and 4, the second beside
    q1's 6-token prefill and the third beside q1's decode, then q0 decodes twice. Compiled (`partial`, q1 bringing no body), q0's
    Program is the chunked request wrapper, each forward's every call reading its own step of the launch (its rows, that step's
    tokens or sequences, and its rows' key counts, which count the chunks before), and the batch rules pass. What no cheaper test
    can: that GLM-5.3's decoder builds a prefill in chunks, each at its launch. Its one GLM-5.3 build takes about half a minute."""
    from verity_vllm.program.predict import common as CM
    steps = _scheduled([(0, 20, 3), (1, 6, 2)], 16, 8)
    assert steps == [[(0, 0, 8)], [(0, 8, 8), (1, 0, 6)], [(0, 16, 4), (1, 6, 1)], [(0, 20, 1)], [(0, 21, 1)]]
    advice, ingress = _session([(20, 3, {**GREEDY, "max_tokens": 3, "ignore_eos": True}),
                                (6, 2, {**GREEDY, "max_tokens": 2, "ignore_eos": True})], steps)
    reg = _class_g(tmp_path, engine={"long_prefill_token_threshold": 8, "max_num_batched_tokens": 16})
    seen, tokens, keys = [], CM._Launch.tokens, CM._Launch.keys
    monkeypatch.setattr(CM._Launch, "tokens", lambda self, own, sampled=False: (
        seen.append((self.step, own, sampled)), tokens(self, own, sampled))[1])
    monkeypatch.setattr(CM._Launch, "keys", lambda self, own, without=(): (
        seen.append((self.step, tuple(own), "keys")), keys(self, own, without))[1])
    c = compile_steps(reg, advice, {"q0": ingress["q0"]}, partial=True)
    u = c.requests[0]
    assert (u.chunks, u.T, sorted(c.refused), c.batch_rule) == ((8, 8, 4), 2, [1], None)
    assert u.batch() == [[[8, 0]], [[8, 8], [6, 0]], [[4, 16], [1, 6]], [[1, 20]], [[1, 21]]]
    own = {0: (0, 8), 1: (8, 8), 2: (16, 4), 3: (20, 1), 4: (21, 1)}
    assert {s for s, *_ in seen} == set(own) and {how for *_, how in seen} == {False, True, "keys"}
    for s, x, how in seen:
        a, n = own[s]
        assert x == (tuple(range(a + 1, a + n + 1)) if how == "keys" else 1 if how else n), (s, x, how)
    assert sum("TokenSelect" in fn.id for _, fn in calls(c.programs[u.digest])) == 3


@pytest.mark.slow
def test_a_class_g_chunked_prompts_traced_build_is_its_compiled_program(tmp_path):
    """The chunked prompt above (q0's 20 prompt tokens in chunks of 8, 8 and 4 beside q1, then two decodes), compiled at rank 0,
    and its trace: the command `trace_commands` gives it (the Build's chunked request wrapper at its launch context, as a served
    window's trace runs it), run on the pinned vLLM with GLM-5.3's config of record and no CUDA device visible, as the row driver
    runs a Build (`gpu_lease.hide_devices`; vLLM keeps Triton then). `compare` finds the two digests equal. What no cheaper test
    can: that the Build's chunked request wrapper traces the Program the compiler predicts. It needs the pinned vLLM (a pod's
    venv), so it skips without it; its Build takes minutes on one core."""
    pytest.importorskip("vllm")
    advice, ingress = _session([(20, 3, {**GREEDY, "max_tokens": 3, "ignore_eos": True}),
                                (6, 2, {**GREEDY, "max_tokens": 2, "ignore_eos": True})], _scheduled([(0, 20, 3), (1, 6, 2)], 16, 8))
    reg = _class_g(tmp_path / "hub", engine={"long_prefill_token_threshold": 8, "max_num_batched_tokens": 16})
    s = compile_steps(reg, advice, {"q0": ingress["q0"]}, partial=True).summary()
    (cmd,) = trace_commands(s, reg, str(tmp_path / "traces"))
    assert json.loads(cmd[cmd.index("--launch-context") + 1])["chunks"] == [8, 8, 4]
    subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", *cmd[1:]], check=True,
                   env={**os.environ, "HF_HUB_CACHE": str(tmp_path / "hub"), "CUDA_VISIBLE_DEVICES": ""})
    got = compare(s, str(tmp_path / "traces"))
    assert (got["equal"], got["mismatches"]) == (1, 0), got["units"]


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("VERITY_VLLM_CLASS_G_WINDOW") != "1",
                    reason="VERITY_VLLM_CLASS_G_WINDOW=1: GLM-5.3's B8 requests as one window in prefill chunks, its eight request "
                           "Programs built at rank 0 (about twenty minutes and 35 GB on one core)")
def test_a_class_g_window_of_long_prompts_compiles_in_chunks(tmp_path):
    """The B8 row's requests scheduled under a long-prefill threshold of 64 and a step budget of 160 (`_chunked_b8`: five of the eight
    prefill in two to five chunks) compile as one window, each request at its chunks and launches, with the batch rules passing
    (R1 over GP-01 of the eight, R2 each alone at the same chunks and launches)."""
    advice, ingress, _ = _chunked_b8(160, 64)
    reg = _class_g(tmp_path, engine={"long_prefill_token_threshold": 64, "max_num_batched_tokens": 160})
    c = compile_steps(reg, advice, ingress)
    assert c.batch_rule is None and len(c.programs) == 8 and sum(len(u.chunks) > 1 for u in c.requests) == 5


def test_the_traced_side_is_one_build_a_distinct_program(tmp_path):
    reg = _registered()
    s = compile_steps(reg, *_mixed()).summary()
    cmds = trace_commands(s, reg, str(tmp_path))
    names = [c[c.index("--out") + 1].rsplit("/", 1)[1] for c in cmds]
    assert [n.rpartition("_s")[0] for n in names] == ["build_request_LP8_T3_greedy", "build_request_LP5_T2_t0.8-p1-r2"]
    assert [json.loads(c[c.index("--stop") + 1]) for c in cmds] == [_spec(8, 3, 4), _spec(5, 2, 3)]
    assert json.loads(cmds[1][cmds[1].index("--sampling") + 1]) == {"temperature": 0.8, "top_p": 1.0, "top_k": 0, "rid": 2}
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
    """SmolLM2-135M, one greedy request of 256 prompt tokens that ran 31 decodes to its max_tokens: the compiled request Program is
    the Build at its shape with its length rule (at SmolLM2's stop id 0, finish 2; `test_derive` holds a trace with a rule to that
    Build), and that Build without the rule is the one the Build traced (cov-k01-10, `test_predict`'s pin).  About 40 s: the only
    test holding a compilation to a trace."""
    cfg = json.loads((FIX / "smollm2-135m.config.json").read_text())
    reg = _registered(cfg)
    body = {**GREEDY, "max_tokens": 32, "prompt": [k % cfg["vocab_size"] for k in range(256)]}
    advice, ingress = _session([(256, 32, body, body["prompt"])], [[(0, 0, 256)]] + [[(0, 256 + k, 1)] for k in range(31)])
    u = compile_steps(reg, advice, ingress).requests[0]
    assert u.stop == SR.spec((0,), max_tokens=32, prompt_len=256, max_model_len=model_shape(cfg).max_pos, outputs=32, finish=2)
    phi = Phi.from_json({**reg.phi.to_json(), "sampling": {"kind": "greedy"}})
    assert u.digest == build_program(phi, "request", LP=256, T=31, stop=u.stop).digest
    assert build_program(phi, "request", LP=256, T=31).digest == ("91d5eaf172589c9a5586e4eb7b8a83e047448018a2a14bcd6e0048f0618bc0a31a390b53b3"
                                                                  "3459979cb706927233124ba258c1043c76a6461ffc91f7d8411888")
