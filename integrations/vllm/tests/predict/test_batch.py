"""verity_vllm.predict.batch: the request-major compiler's templates hold their declared batch-invariance verdicts, and
`check_compiled` refuses by name a compilation in which a request's outputs could depend on its batch-mates.

I1 v0 (`verity.core.protocols.certifier.advice`) and `steps.plan_steps` aren't on main yet, so the rule's tests run it against
stand-ins: advice classes with I1's fields and its kept-output rule, and a planner with v0's shapes.
`test_compile_steps_passes_its_own_rule` runs it on the real compiler once both are importable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import NamedTuple

import pytest

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Tuple, Value
from verity_vllm.check.batch import BATCH, SHARED, check, own, report
from verity_vllm.predict.batch import BatchDependent, check_compiled, registered_phi, templates, workload_labels
from verity_vllm.predict.build import Unexpressible
from verity_vllm.predict.phi import Phi

TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
        "max_position_embeddings": 512, "hidden_act": "silu"}
#: the RTX PRO 6000 (cc 12.0, 188 SMs)
RTX = {"compute_capability": [12, 0], "num_sms": 188}


def _phi(cfg: dict = TINY) -> Phi:
    return Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, workload={"requests": []}, target=dict(RTX),
               execution="bi-eager")


def _verdicts(phi: Phi, **kw) -> list:
    vs = check(templates(phi, **kw))
    assert all(v.ok for v in vs), report(vs)
    return vs


def _got(vs: list, prefix: str) -> tuple:
    return next(v.got for v in vs if v.template.name.startswith(prefix))


@pytest.mark.slow
def test_templates_hold_their_declared_verdicts():
    """About 6 s, most of it building GumbelTopPTokenSelect_v2's body at two split counts, which (b)'s structure comparison reads."""
    vs = _verdicts(_phi(), max_tokens=512, max_model_len=512)
    assert _got(vs, "rmsnorm ") == _got(vs, "embedding ") == _got(vs, "gemm down ") == (True, True)
    assert _got(vs, "attention ") == (True, False)
    assert [s.request for v in vs if v.template.name.startswith("attention ") for s in v.splits] == [(129, 0, 1)]
    assert _got(vs, "token select (top-p, derived S)") == (False, None)
    assert _got(vs, "token select (top-p, S = SplitsFor_v1") == (True, False)
    assert _got(vs, "token select (top-p, registered S)") == (True, True)
    assert _got(vs, "GP-01 over 2 requests (greedy)") == (True, None)
    assert _got(vs, "GP-01 over 2 requests (top-p, derived S)") == (False, None)


@pytest.mark.slow
def test_the_frozen_v0_models_templates_hold_their_declared_verdicts():
    """The frozen v0 registration's own kinds (Llama-3.2-1B, V 128,256, on the RTX PRO 6000): a vocabulary change that moves one of
    them (a launch static on a GEMM at its N, a sampler construction at its V) shows only at the model's shapes (about 60 s)."""
    vs = _verdicts(registered_phi("LLAMA32_1B", "rtxpro6000"))
    assert sum(1 for v in vs if v.got == (True, True)) == 13


# --- the compile-unit rule against stand-ins -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestAdvice:
    id: str
    prompt_len: int
    output_len: int
    finish: str = "length"


@dataclass(frozen=True)
class Source:
    request: int
    n: int


@dataclass(frozen=True)
class Row:
    request: int
    kind: str
    start: int
    n: int
    cache_hit: int = 0
    preempted: bool = False
    sources: tuple = ()


@dataclass(frozen=True)
class StepAdvice:
    rows: tuple


@dataclass(frozen=True)
class Advice:
    requests: tuple
    steps: tuple
    release: tuple


class Output(NamedTuple):
    step: int
    row: int
    request: int
    index: int


def outputs(advice: Advice) -> tuple[Output, ...]:
    """I1's kept outputs: a row's sample is kept when it reaches its request's known tokens."""
    known, kept, out = [q.prompt_len for q in advice.requests], [0] * len(advice.requests), []
    for s, st in enumerate(advice.steps):
        for j, row in enumerate(st.rows):
            r = row.request
            if row.start + row.n == known[r] and kept[r] < advice.requests[r].output_len:
                out.append(Output(s, j, r, kept[r]))
                kept[r] += 1
                known[r] += 1
    return tuple(out)


@dataclass(frozen=True)
class Unit:
    request: int
    digest: str
    LP: int
    T: int
    policy: dict
    seed: int | None
    first_step: int
    id: str


@dataclass(frozen=True)
class Plan:
    requests: list
    steps: list
    discarded: list


@dataclass(frozen=True)
class RowRef:
    request: int
    digest: str
    start: int
    n: int
    samples: int


def plan(registered, advice, ingress, *, seed_of=lambda advice, body: body.get("seed", 7), samples_of=None):
    """v0's planner for what the rule reads: each request's (LP, T, policy, seed, first step, id), its live rows and its discarded."""
    first: dict[int, int] = {}
    for s, st in enumerate(advice.steps):
        for row in st.rows:
            first.setdefault(row.request, s)
    units = []
    for i, q in enumerate(advice.requests):
        body = ingress[q.id]
        t = body["temperature"]
        policy = {"kind": "greedy"} if t == 0 else {"kind": "stochastic", "temperature": t, "top_p": body["top_p"], "top_k": 0,
                                                    "seeded": True, "splits": body.get("splits", 32)}
        units.append(Unit(i, "", q.prompt_len, q.output_len - 1, policy, None if t == 0 else seed_of(advice, body), first[i], q.id))
    steps, discarded = [], []
    for s, st in enumerate(advice.steps):
        live = []
        for row in st.rows:
            u = units[row.request]
            if row.start >= u.LP + u.T:
                discarded.append((s, row.request, row.start, row.n))
            else:
                k = int(row.start + row.n >= u.LP) if samples_of is None else samples_of(st, row)
                live.append((row.request, row.start, row.n, k))
        steps.append(live)
    return Plan(units, steps, discarded)


def _digest(u: Unit) -> str:
    return "P" + json.dumps([u.LP, u.T, u.policy], sort_keys=True)


@dataclass(frozen=True)
class Component:
    program_digest: str
    params: tuple
    groups: tuple


@dataclass(frozen=True)
class WRequest:
    index: int
    component: Component


@dataclass(frozen=True)
class Workload:
    fn: object
    requests: tuple
    shared_by_request: tuple


V16 = Value(16)
MUL = PrimitiveDefinition("BatchTestWlMul16", 1, (("a", V16), ("b", V16)), V16, lambda a, b: (a * b) & 0xFFFF, register=False)


def _workload(units: list[Unit], groups: tuple[str, ...] = ("weights", "prompt"), cross: bool = False) -> Workload:
    """GP-01's shape over ``units``: the shared weight once, each request's own Input, one output each (request k's reads request
    k − 1's Input under ``cross``)."""
    n = len(units)
    params = (("shared.w", V16),) + tuple((f"request[r{u.request}].{g}", V16) for u in units for g in groups[1:])
    per = len(groups) - 1

    def body(B, S, w, *xs):
        return tuple_of(*(B.call(MUL, w, xs[(k - 1 if cross and k else k) * per + per - 1]) for k in range(n)))

    fn = bind(CompositeDefinition("BatchTestWorkload", 1, (), lambda S: (params, Tuple(*([V16] * n))), body, register=False))
    comps = [Component(u.digest, tuple((g, V16) for g in groups), groups) for u in units]
    return Workload(fn, tuple(WRequest(u.request, c) for u, c in zip(units, comps)), tuple((0,) for _ in units))


@dataclass
class Compiled:
    requests: list
    steps: list
    discarded: list
    workload: Workload


def _compile(advice, ingress, planner=plan, **kw) -> Compiled:
    p = planner(None, advice, ingress)
    units = [replace(u, digest=_digest(u)) for u in p.requests]
    steps = [[RowRef(r, units[r].digest, start, n, k) for r, start, n, k in rows] for rows in p.steps]
    return Compiled(units, steps, p.discarded, _workload(units, **kw))


GREEDY = {"temperature": 0.0, "top_p": 1.0}
TOPP = {"temperature": 0.8, "top_p": 0.95, "seed": 3}


def _session(reqs: list[tuple], steps: list[list[tuple]]) -> tuple[Advice, dict]:
    """Requests (prompt_len, output_len, body), id `q<i>`; rows (request, start, n), kind as I1 derives it; one release bucket each."""
    requests = tuple(RequestAdvice(f"q{i}", lp, out) for i, (lp, out, _) in enumerate(reqs))
    adv = Advice(requests, tuple(StepAdvice(tuple(Row(r, "decode" if n == 1 and s >= reqs[r][0] else "prefill", s, n) for r, s, n in st))
                                 for st in steps), ())
    adv = replace(adv, release=tuple(range(len(outputs(adv)))))
    return adv, {f"q{i}": dict(body) for i, (_, _, body) in enumerate(reqs)}


def _mixed(body1: dict = TOPP) -> tuple[Advice, dict]:
    """r0 (LP 4, greedy) from step 0, r1 (LP 3) prefilled beside r0's first decode, r2 (r0's shape) beside two decodes; r0's last
    row runs one past its last kept output (discarded)."""
    return _session([(4, 3, GREEDY), (3, 2, body1), (4, 3, GREEDY)], [
        [(0, 0, 4)],
        [(0, 4, 1), (1, 0, 3)],
        [(0, 5, 1), (1, 3, 1), (2, 0, 4)],
        [(0, 6, 1), (2, 4, 1)],
        [(2, 5, 1)],
    ])


def test_a_batch_invariant_compilation_passes():
    adv, ing = _mixed()
    c = _compile(adv, ing)
    assert c.discarded == [(3, 0, 6, 1)]
    assert check_compiled(None, adv, ing, c, plan) is None


def test_r3_a_derived_split_count_is_refused():
    adv, ing = _mixed({**TOPP, "splits": "derived"})
    with pytest.raises(BatchDependent, match=r"request 1: its top-p sampler's split count is derived .* SplitsFor_v1\(1, num_SMs\)"):
        check_compiled(None, adv, ing, _compile(adv, ing), plan)


def test_r2_a_planner_that_reads_the_batch_is_refused():
    adv, ing = _mixed({k: v for k, v in TOPP.items() if k != "seed"})
    by_batch = lambda a, ad, i: plan(a, ad, i, seed_of=lambda advice, body: len(advice.requests))  # noqa: E731
    with pytest.raises(BatchDependent, match=r"request 1: compiled alone it's .*1, 'q1'\), in its batch .*3, 'q1'\)"):
        check_compiled(None, adv, ing, _compile(adv, ing, by_batch), by_batch)


def test_r2_rows_placed_by_the_batch_are_refused():
    adv, ing = _mixed()
    alone_samples = lambda a, ad, i: plan(a, ad, i, samples_of=lambda st, row: int(len(st.rows) == 1))  # noqa: E731
    with pytest.raises(BatchDependent, match=r"request 0: its rows \(start, n, samples\) are \[\(0, 4, 1\), \(4, 1, 1\), \(5, 1, 1\)\] "
                                             r"alone, \[\(0, 4, 1\), \(4, 1, 0\), \(5, 1, 0\)\] in its batch"):
        check_compiled(None, adv, ing, _compile(adv, ing, alone_samples), alone_samples)


def test_r2_a_request_refused_alone_is_refused():
    adv, ing = _mixed()

    def needs_company(a, ad, i):
        if len(ad.requests) < 2:
            raise Unexpressible("a request needs a batch-mate")
        return plan(a, ad, i)

    with pytest.raises(BatchDependent, match="request 0 compiles in its batch, and alone it's refused: a request needs a batch-mate"):
        check_compiled(None, adv, ing, _compile(adv, ing, needs_company), needs_company)


def test_r2_a_cache_hit_on_another_requests_kv_has_no_solo_form():
    adv, ing = _mixed()
    hit = Row(2, "prefill", 2, 2, cache_hit=2, sources=(Source(0, 2),))
    steps = list(adv.steps)
    steps[2] = StepAdvice((steps[2].rows[0], steps[2].rows[1], hit))
    adv = replace(adv, steps=tuple(steps))
    with pytest.raises(BatchDependent, match="request 2: step 2's cache hit reads KV another request computed"):
        check_compiled(None, adv, ing, _compile(adv, ing), plan)


def test_r1_a_row_naming_another_requests_program_is_refused():
    adv, ing = _mixed()
    c = _compile(adv, ing)
    c.steps[2][1] = replace(c.steps[2][1], digest=c.requests[0].digest)
    with pytest.raises(BatchDependent, match="step 2: a row of request 1 names Program"):
        check_compiled(None, adv, ing, c, plan)


def test_r1_a_composition_reading_another_requests_input_is_refused():
    adv, ing = _mixed()
    with pytest.raises(BatchDependent, match=r"GP-01: the outputs of owner 1 read request\[r0\]\.prompt .* held by owner 0 \(\+1 more\)"):
        check_compiled(None, adv, ing, _compile(adv, ing, cross=True), plan)


def test_r1_a_batch_provenance_input_is_refused():
    adv, ing = _mixed()
    with pytest.raises(BatchDependent, match=r"GP-01: the outputs of owner 0 read request\[r0\]\.schedule .* batch-derived"):
        check_compiled(None, adv, ing, _compile(adv, ing, groups=("weights", "prompt", "schedule")), plan)


def test_workload_labels_by_provenance():
    adv, ing = _mixed()
    c = _compile(adv, ing, groups=("weights", "prompt", "seed", "splits"))
    labels, ret = workload_labels(c.workload)
    assert labels == (SHARED, own(0), own(0), BATCH, own(1), own(1), BATCH, own(2), own(2), BATCH)
    assert ret.labels == (own(0), own(1), own(2))
    bad = replace(c.workload.requests[0], component=replace(c.workload.requests[0].component, groups=("weights",)))
    with pytest.raises(BatchDependent, match="names no group"):
        workload_labels(replace(c.workload, requests=(bad, *c.workload.requests[1:])))


# --- on the real compiler --------------------------------------------------------------------------------------------------------


def test_compile_steps_passes_its_own_rule():
    steps = pytest.importorskip("verity_vllm.predict.steps")
    I1 = pytest.importorskip("verity.core.protocols.certifier.advice")
    cfg = {**TINY, "max_position_embeddings": 64}
    reg = steps.Registered.of(Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, target=dict(RTX)), default_seed=7)
    greedy = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": None, "max_tokens": 3}
    topp = {"temperature": 0.8, "top_p": 0.95, "top_k": 0, "seed": None, "max_tokens": 2}

    def session(b1: dict):
        reqs = [(4, 3, greedy), (3, 2, b1)]
        rows = [[(0, 0, 4)], [(0, 4, 1), (1, 0, 3)], [(0, 5, 1), (1, 3, 1)]]
        requests = tuple(I1.RequestAdvice(f"q{i}", lp, out, "length") for i, (lp, out, _) in enumerate(reqs))
        adv = I1.Advice(requests, tuple(I1.StepAdvice(tuple(I1.Row(r, "decode" if n == 1 and s >= reqs[r][0] else "prefill", s, n)
                                                            for r, s, n in st)) for st in rows), (0,) * 5)
        ing = {f"q{i}": {"prompt": [(31 * i + 7 * k + 1) % 256 for k in range(lp)], "model": "m", "user": f"q{i}", **b}
               for i, (lp, _, b) in enumerate(reqs)}
        return adv, ing

    adv, ing = session({**greedy, "max_tokens": 2})
    check_compiled(reg, adv, ing, steps.compile_steps(reg, adv, ing), steps.plan_steps)
    adv, ing = session(topp)
    with pytest.raises(BatchDependent, match="request 1: its top-p sampler's split count is derived"):
        check_compiled(reg, adv, ing, steps.compile_steps(reg, adv, ing), steps.plan_steps)
