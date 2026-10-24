"""verity_vllm.predict.steps (I2): the advice's requests compile to the Build's request Programs at the shapes they ran, its rows
to references into them, and what v0 cannot compile is refused by name.

The advice here is I1's draft field names on plain namespaces (`Advice(requests, steps)`, `Row(request, kind, start, n,
cache_hit, preempted)`, each request's `prompt_len`, `output_len` and `sampling`), until I1's dataclasses land.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from verity_vllm.predict.build import Unexpressible, build_program
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

FIX = Path(__file__).parent / "fixtures"
RTXPRO6000 = {"compute_capability": [12, 0], "num_sms": 188}
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
        "max_position_embeddings": 64, "hidden_act": "silu"}
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": None, "max_tokens": 4}
STOCH = {"temperature": 0.8, "top_p": 0.95, "top_k": 0, "seed": None, "max_tokens": 3, "min_p": 0.0, "repetition_penalty": 1.0}
STOCH_POLICY = {"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "splits": "derived"}


def _registered(cfg: dict = TINY, **kw) -> Registered:
    return Registered.of(Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, target=dict(RTXPRO6000), **kw), default_seed=7)


def _row(request: int, start: int, n: int, kind: str | None = None, **kw) -> NS:
    return NS(request=request, kind=kind or ("prefill" if n > 1 or start == 0 else "decode"), start=start, n=n, cache_hit=0,
              preempted=False, **kw)


def _advice(requests: list[tuple[int, dict]], steps: list[list[NS]]) -> NS:
    return NS(requests=[NS(prompt_len=lp, sampling=s) for lp, s in requests], steps=[NS(rows=rows) for rows in steps])


def _mixed() -> NS:
    """Three requests that ran and one that never did: r0 (LP 8, greedy) from step 0, r1 (LP 5, seeded top-p) prefilled beside
    r0's first decode, r2 (r0's shape) prefilled beside two decodes, each step's decode rows first (Model Runner V2's order)."""
    return _advice([(8, GREEDY), (5, STOCH), (8, GREEDY), (4, GREEDY)], [
        [_row(0, 0, 8)],
        [_row(0, 8, 1), _row(1, 0, 5)],
        [_row(0, 9, 1), _row(1, 5, 1), _row(2, 0, 8)],
        [_row(0, 10, 1), _row(1, 6, 1), _row(2, 8, 1)],
        [_row(2, 9, 1)],
        [_row(2, 10, 1)],
    ])


def test_requests_compile_to_the_builds_request_programs_at_the_shapes_they_ran():
    reg = _registered()
    c = compile_steps(reg, _mixed())
    d0 = build_program(Phi.from_json({**reg.phi.to_json(), "sampling": {"kind": "greedy"}}), "request", LP=8, T=3).digest
    d1 = build_program(Phi.from_json({**reg.phi.to_json(), "sampling": STOCH_POLICY}), "request", LP=5, T=2).digest
    assert [u and (u.digest, u.LP, u.T, u.first_step) for u in c.requests] == [(d0, 8, 3, 0), (d1, 5, 2, 1), (d0, 8, 3, 2), None]
    assert sorted(c.programs) == sorted({d0, d1})
    assert c.requests[1].policy == STOCH_POLICY and c.requests[1].seed == 7 and c.requests[0].seed is None
    assert c.steps[1] == [RowRef(0, d0, 8, 1, 1), RowRef(1, d1, 0, 5, 1)]
    assert c.steps[2] == [RowRef(0, d0, 9, 1, 1), RowRef(1, d1, 5, 1, 1), RowRef(2, d0, 0, 8, 1)]
    assert [len(s) for s in c.steps] == [1, 2, 3, 3, 1, 1]
    wl = c.workload
    assert [(r.request_id, r.index, r.prompt_len, r.cap, r.arrive_step) for r in wl.requests] == [
        ("r0", 0, 8, 4, 0), ("r1", 1, 5, 3, 1), ("r2", 2, 8, 4, 2)]
    assert wl.lag == reg.phi.target_profile().lag()
    s = json.loads(json.dumps(c.summary()))
    assert s["workload_digest"] == wl.digest and s["steps"][1] == [[0, 8, 1, 1], [1, 0, 5, 1]]
    assert compile_steps(reg, _mixed()).summary() == c.summary()


def test_a_plan_is_the_compilation_before_its_programs_are_built():
    """`plan_steps` fixes the shapes, rows and refusals; `request_program` builds one distinct Program, which a measurement can do
    request by request in other processes."""
    reg = _registered()
    plan, c = plan_steps(reg, _mixed()), compile_steps(reg, _mixed())
    assert [(u.request, u.LP, u.T) for u in plan.distinct()] == [(0, 8, 3), (1, 5, 2)]
    assert [request_program(reg, u).digest for u in plan.distinct()] == [c.requests[0].digest, c.requests[1].digest]
    assert plan.steps == [[(r.request, r.start, r.n, r.samples) for r in rows] for rows in c.steps]
    assert all(u is None or u.digest == "" for u in plan.requests)


def test_the_clients_seed_is_the_seed_input_and_the_policy_reads_temperature_and_top_p():
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
    ({**STOCH, "bad_words": ["x"]}, "bad_words (a parameter this compiler does not know)"),
    ({**STOCH, "temperature": 1e-6}, "temperature 1e-06"),
    ({"top_p": 1.0}, "no temperature"),
])
def test_sampling_v0_cannot_express_is_refused_by_name(sampling, fact):
    with pytest.raises(Unexpressible, match=f"request 0: .*{fact}".replace("(", r"\(").replace(")", r"\)").replace("[", r"\[")):
        policy_of(0, sampling, 7)


@pytest.mark.parametrize("steps,fact", [
    ([[_row(0, 0, 3)], [_row(0, 3, 5)]], "a chunked prefill, 3 of its 8 prompt tokens at step 0"),
    ([[NS(**{**vars(_row(0, 4, 4)), "cache_hit": 4})]], "a prefix-cache hit of 4 tokens at step 0"),
    ([[_row(0, 0, 8)], [], [_row(0, 8, 1)]], r"its row 1 is \(step, start, n\) = \(2, 8, 1\), not the decode \(1, 8, 1\)"),
    ([[_row(0, 0, 8)], [_row(0, 8, 1)], [_row(0, 0, 9)]], r"its row 2 is \(step, start, n\) = \(2, 0, 9\)"),
    ([[_row(0, 0, 8)], [NS(**{**vars(_row(0, 8, 1)), "preempted": True})]], "step 1: request 0 preempted"),
    ([[_row(0, 0, 8), _row(0, 0, 8)]], "step 0: a row of request 0 twice"),
    ([[_row(5, 0, 8)]], "step 0: a row of request 5 outside"),
    ([[_row(0, 0, 8, kind="decode")]], "its row at step 0 is marked 'decode'"),
])
def test_what_v0_cannot_run_is_refused_by_name(steps, fact):
    with pytest.raises(Unexpressible, match=fact):
        compile_steps(_registered(), _advice([(8, GREEDY)], steps))


def test_a_steps_preempted_set_is_refused():
    advice = _advice([(8, GREEDY)], [[_row(0, 0, 8)], []])
    advice.steps[1].preempted = [0]
    with pytest.raises(Unexpressible, match=r"step 1: preempted requests \[0\]"):
        compile_steps(_registered(), advice)


def test_the_row_past_a_requests_last_kept_output_is_discarded():
    """Request 0 kept 3 outputs (its prefill's and two decodes'), and async scheduling ran one row past them, at step 3: the
    request's Program holds its live rows only."""
    reg = _registered()
    advice = _advice([(8, GREEDY)], [[_row(0, 0, 8)], [_row(0, 8, 1)], [_row(0, 9, 1)], [_row(0, 10, 1)]])
    advice.requests[0].output_len = 3
    c = compile_steps(reg, advice)
    d = build_program(Phi.from_json({**reg.phi.to_json(), "sampling": {"kind": "greedy"}}), "request", LP=8, T=2).digest
    assert (c.requests[0].digest, c.requests[0].T, c.workload.requests[0].cap) == (d, 2, 3)
    assert c.steps[3] == [] and c.discarded == [(3, 0, 10, 1)] and c.summary()["discarded"] == [[3, 0, 10, 1]]
    advice.requests[0].output_len = 4
    c = compile_steps(reg, advice)
    assert (c.requests[0].T, c.discarded, len(c.steps[3])) == (3, [], 1)
    advice.requests[0].output_len = 2
    with pytest.raises(Unexpressible, match="request 0: 4 rows for 2 kept outputs"):
        compile_steps(reg, advice)


def test_partial_refuses_request_by_request():
    """A measurement over served traffic: request 1's prefix-cache hit, request 2's preemption and request 3's top_k refuse those
    requests by name, their rows leave the steps, and request 0 compiles as it would alone."""
    hit = NS(**{**vars(_row(1, 4, 4)), "cache_hit": 4})
    resumed = NS(**{**vars(_row(2, 0, 9)), "preempted": True})
    advice = _advice([(8, GREEDY), (8, GREEDY), (8, GREEDY), (5, {**STOCH, "top_k": 50})], [
        [_row(0, 0, 8), hit, _row(2, 0, 8)],
        [_row(0, 8, 1), _row(1, 8, 1), _row(2, 8, 1), _row(3, 0, 5)],
        [_row(0, 9, 1), _row(1, 9, 1), _row(3, 5, 1)],
        [_row(0, 10, 1), resumed],
    ])
    reg = _registered()
    with pytest.raises(Unexpressible, match="step 3: request 2 preempted"):
        compile_steps(reg, advice)
    c = compile_steps(reg, advice, partial=True)
    assert sorted(c.refused) == [1, 2, 3]
    assert "prefix-cache hit of 4" in c.refused[1] and "step 3: request 2 preempted" in c.refused[2] and "top_k=50" in c.refused[3]
    assert [u is not None for u in c.requests] == [True, False, False, False]
    alone = compile_steps(reg, _advice([(8, GREEDY)], [[_row(0, 0, 8)], [_row(0, 8, 1)], [_row(0, 9, 1)], [_row(0, 10, 1)]]))
    assert c.steps == alone.steps and c.programs.keys() == alone.programs.keys()
    assert sorted(c.summary()["refused"]) == ["1", "2", "3"]


def test_dead_gates_are_counted_as_circuit_check_counts_them():
    """The request wrapper's prefill runs the last layer on every prompt row and only the last row reaches the LM head, so
    today's request Programs have dead gates: `dead_gates_of` counts them as circuit-check does and says under which definitions."""
    from verity.ml.circuits.liveness import dead_gates
    c = compile_steps(_registered(), _mixed())
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
        compile_steps(_registered(execution="eager"), _mixed())
    with pytest.raises(ValueError, match="carries no workload"):
        Registered(Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY}, workload={"requests": [{"prompt_len": 1}]}), 7)


def test_the_traced_side_is_one_build_a_distinct_program(tmp_path):
    reg = _registered()
    s = compile_steps(reg, _mixed()).summary()
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
    reg = _registered(json.loads((FIX / "smollm2-135m.config.json").read_text()))
    steps = [[_row(0, 0, 256)]] + [[_row(0, 256 + k, 1)] for k in range(31)]
    c = compile_steps(reg, _advice([(256, GREEDY)], steps))
    assert c.requests[0].digest == ("52e6bb0b37e31ec3bdcc976027f412955be60cdd7bcd1ed4982aee7b4b716681a00323889897a394808aa040994f53a3e210dc6"
                                    "03dba9a65751817e1df348f5c")
