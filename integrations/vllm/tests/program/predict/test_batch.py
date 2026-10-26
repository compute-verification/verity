"""verity_vllm.program.predict.batch: the request-major compiler's templates hold their declared batch-invariance verdicts, and
`check_compiled` refuses by name a compilation in which a request's outputs could depend on its batch-mates.

`steps.plan_steps` isn't on main yet, so the rule's tests run it on I1 v0's advice against a stand-in planner with v0's shapes.
`test_compile_steps_passes_its_own_rule` runs it on the real compiler once that is importable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from itertools import accumulate

import pytest

from verity.core.protocols.certifier import advice as I1
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Tuple, Value
from verity_vllm.check.batch import BATCH, SHARED, check, own, report
from verity_vllm.program.predict.batch import BatchDependent, check_compiled, registered_phi, templates, workload_labels
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.phi import Phi

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
    assert _got(vs, "GP-01 over 2 requests (top-p, registered S)") == (True, None)
    assert _got(vs, "GP-01 over 2 requests (top-p, derived S)") == (False, None)


def test_the_registered_split_count_keeps_one_nucleus_at_every_batch_size_and_position():
    """The top-p template's keep set (`topp_split.topp_keep_row`, which `TopPMaskWordx{V}s{S}` evaluates) for a row served at
    batch size B and position i.  The template reads each row by itself, so the batch reaches the keep set only through S.  At
    the S vLLM derives from the batch (SplitsFor_v1(B, 188) on the RTX PRO 6000) the keep set moves for some rows; the pinned
    engine (`engine.topp_pin`: vLLM's loop at num_sm = registered S x B) launches the registered S at every B, and the keep set
    is the same at every (B, i)."""
    import numpy as np

    from verity_vllm.pipeline.topp_split_probe import gen_random
    from verity.ml.definitions.registry.topp_geometry import registered_splits, splits_for
    from verity.ml.definitions.registry.topp_split import topp_keep_row

    nsm, V = RTX["num_sms"], 4096
    S_reg = registered_splits(nsm)
    rows = [gen_random(seed, V) for seed in range(6)]
    keep = {}

    def kept(r: int, p: float, S: int) -> np.ndarray:
        if (r, p, S) not in keep:
            keep[r, p, S] = topp_keep_row(rows[r], p, S)
        return keep[r, p, S]

    moved = 0
    for r in range(len(rows)):
        for p in (0.9, 0.95):
            pinned, derived = [], []
            for B in (1, 6, 12, 24, 48, 64):
                for i in sorted({0, B // 2, B - 1}):
                    assert splits_for(B, S_reg * B) == S_reg
                    pinned.append(kept(r, p, splits_for(B, S_reg * B)))
                    derived.append(kept(r, p, splits_for(B, nsm)))
            assert all((k == pinned[0]).all() for k in pinned)
            moved += any((k != derived[0]).any() for k in derived)
    assert moved >= 1


@pytest.mark.slow
def test_the_frozen_v0_models_templates_hold_their_declared_verdicts():
    """The frozen v0 registration's own kinds (Llama-3.2-1B, V 128,256, on the RTX PRO 6000): a vocabulary change that moves one of
    them (a launch static on a GEMM at its N, a sampler construction at its V) shows only at the model's shapes (about 60 s)."""
    vs = _verdicts(registered_phi("LLAMA32_1B", "rtxpro6000"))
    assert sum(1 for v in vs if v.got == (True, True)) == 13


# --- the compile-unit rule on I1 v0's advice, against a stand-in planner ---------------------------------------------------------


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
    C: int = 0


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


def plan(registered, advice, ingress, *, seed_of=lambda advice, body: body.get("seed", 7), samples_of=None,
         C_of=lambda advice, row: row.cache_hit):
    """v0's planner for what the rule reads: each request's (LP, T, policy, seed, first step, id, C), its live rows and its
    discarded. It doesn't check a hit's tokens, so the rule's own check shows."""
    first: dict[int, tuple[int, I1.Row]] = {}
    for s, st in enumerate(advice.steps):
        for row in st.rows:
            first.setdefault(row.request, (s, row))
    units = []
    for i, q in enumerate(advice.requests):
        body, (s0, row0) = ingress[q.id], first[i]
        t = body["temperature"]
        policy = {"kind": "greedy"} if t == 0 else {"kind": "stochastic", "temperature": t, "top_p": body["top_p"], "top_k": 0,
                                                    "seeded": True, "splits": body.get("splits", 32)}
        units.append(Unit(i, "", q.prompt_len, q.output_len - 1, policy, None if t == 0 else seed_of(advice, body), s0, q.id,
                          C_of(advice, row0)))
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
    return "P" + json.dumps([u.LP, u.T, u.policy, u.C], sort_keys=True)


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
    """GP-01's shape over ``units``: the shared weight once, each request's own Inputs (``groups`` past the weight, and `prefix`
    with a hit), one output each, which reads its request's last Input (request k's reads request k − 1's under ``cross``)."""
    gs = [groups + ("prefix",) * bool(u.C) for u in units]
    params = (("shared.w", V16),) + tuple((f"request[r{u.request}].{g}", V16) for u, ug in zip(units, gs) for g in ug[1:])
    last = list(accumulate(len(ug) - 1 for ug in gs))

    def body(B, S, w, *xs):
        return tuple_of(*(B.call(MUL, w, xs[last[k - 1 if cross and k else k] - 1]) for k in range(len(units))))

    fn = bind(CompositeDefinition("BatchTestWorkload", 1, (), lambda S: (params, Tuple(*([V16] * len(units)))), body, register=False))
    comps = [Component(u.digest, tuple((g, V16) for g in ug), ug) for u, ug in zip(units, gs)]
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


def _tokens(i: int, n: int) -> list[int]:
    return [(31 * i + 7 * k + 1) % 256 for k in range(n)]


def _session(reqs: list[tuple], steps: list[list[tuple]]) -> tuple[I1.Advice, dict]:
    """Requests (prompt_len, output_len, body[, prompt]), id `q<i>`, finished on length, the prompt `_tokens(i, prompt_len)` unless
    given; rows (request, start, n[, a hit's fields]), kind as I1 derives it; the kept outputs released in order."""
    requests = tuple(I1.RequestAdvice(f"q{i}", lp, out, "length") for i, (lp, out, *_) in enumerate(reqs))
    adv = I1.Advice(requests, tuple(I1.StepAdvice(tuple(I1.Row(r, "decode" if n == 1 and s >= reqs[r][0] else "prefill", s, n, **(x[0] if x else {}))
                                                        for r, s, n, *x in st)) for st in steps), tuple(range(sum(q[1] for q in reqs))))
    return adv, {f"q{i}": {**q[2], "prompt": list(q[3]) if len(q) > 3 else _tokens(i, q[0])} for i, q in enumerate(reqs)}


def _mixed(body1: dict = TOPP) -> tuple[I1.Advice, dict]:
    """r0 (LP 4, greedy) from step 0, r1 (LP 3) prefilled beside r0's first decode, r2 (r0's shape) beside two decodes; r0's last
    row runs one past its last kept output (discarded)."""
    return _session([(4, 3, GREEDY), (3, 2, body1), (4, 3, GREEDY)], [
        [(0, 0, 4)],
        [(0, 4, 1), (1, 0, 3)],
        [(0, 5, 1), (1, 3, 1), (2, 0, 4)],
        [(0, 6, 1), (2, 4, 1)],
        [(2, 5, 1)],
    ])


def _hit(shared: int) -> tuple[I1.Advice, dict]:
    """r1 (LP 4) admitted at step 1 on a 2-token prefix-cache hit of r0's KV; its prompt shares r0's first ``shared`` tokens."""
    p0 = _tokens(0, 4)
    return _session([(4, 2, GREEDY, p0), (4, 2, GREEDY, p0[:shared] + _tokens(1, 4)[shared:])], [
        [(0, 0, 4)],
        [(0, 4, 1), (1, 2, 2, {"cache_hit": 2, "sources": (I1.Source(0, 2),)})],
        [(1, 4, 1)],
    ])


def _past_a_prompt() -> tuple[I1.Advice, dict]:
    """r1's 2-token hit reads r0's KV at position 1, which r0's first decode computed: past r0's 1-token prompt."""
    return _session([(1, 2, GREEDY), (4, 1, GREEDY, _tokens(0, 1) + _tokens(1, 3))], [
        [(0, 0, 1)],
        [(0, 1, 1), (1, 2, 2, {"cache_hit": 2, "sources": (I1.Source(0, 2),)})],
    ])


def _prompt_as_text() -> tuple[I1.Advice, dict]:
    """`_hit(2)` with r0's prompt certified as text, not token ids."""
    adv, ing = _hit(2)
    return adv, {**ing, "q0": {**ing["q0"], "prompt": " ".join(map(str, ing["q0"]["prompt"]))}}


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
    with pytest.raises(BatchDependent, match=r"request 1: compiled alone it's .*1, 'q1', 0\), in its batch .*3, 'q1', 0\)"):
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


def test_r2_a_cache_hit_on_its_own_tokens_is_kept():
    """r1's hit holds the KV of its own first 2 tokens (r0's prompt starts with them): alone, a stand-in of those tokens computes it
    and r1 plans the same at C = 2, and in GP-01 its `prefix` Input is its own."""
    adv, ing = _hit(2)
    c = _compile(adv, ing)
    assert c.requests[1].C == 2 and c.workload.requests[1].component.groups == ("weights", "prompt", "prefix")
    assert check_compiled(None, adv, ing, c, plan) is None


@pytest.mark.parametrize("session,why", [
    (lambda: _hit(1), "at positions 0 … 1, and their tokens aren't its own"),
    (_past_a_prompt, "at positions 0 … 1, and their tokens aren't its own"),
    (_prompt_as_text, "their tokens can't be checked equal"),
])
def test_r2_a_cache_hit_whose_tokens_arent_checked_its_own_is_refused(session, why):
    """The stand-in planner doesn't check a hit's tokens, and the rule does: a prompt that leaves the source's inside the hit, a hit
    past the source's prompt (the KV of a token it sampled), and a source whose prompt isn't token ids are each refused."""
    adv, ing = session()
    with pytest.raises(BatchDependent, match=f"request 1: step 1's cache hit reads request 0's KV.*{why}"):
        check_compiled(None, adv, ing, _compile(adv, ing), plan)


def test_r2_c_planned_from_the_batch_is_refused():
    """A planner that takes r1's C from the batch (its first request's prompt length, r0's 4) plans it alone at the stand-in's 2."""
    adv, ing = _hit(2)
    by_batch = lambda a, ad, i: plan(a, ad, i, C_of=lambda advice, row: advice.requests[0].prompt_len if row.cache_hit else 0)  # noqa: E731
    with pytest.raises(BatchDependent, match=r"request 1: compiled alone it's \(LP, T, policy, seed, id, C\) = .*'q1', 2\), in its "
                                             r"batch .*'q1', 4\)"):
        check_compiled(None, adv, ing, _compile(adv, ing, by_batch), by_batch)


def test_r2_a_resume_on_another_requests_kv_has_no_solo_form():
    """r0, preempted after its prefill, resumes on r1's KV of the same prompt: I1 v0 allows it, and v0 has no solo form for it."""
    p = _tokens(0, 4)
    adv, ing = _session([(4, 3, GREEDY, p), (4, 2, GREEDY, p)], [
        [(0, 0, 4)],
        [(1, 0, 4)],
        [(0, 4, 1, {"cache_hit": 4, "preempted": True, "sources": (I1.Source(1, 4),)}), (1, 4, 1)],
        [(0, 5, 1)],
    ])
    with pytest.raises(BatchDependent, match="request 0: step 2 resumes it on KV another request computed"):
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


@pytest.mark.parametrize("group", ["schedule", "prefix"])
def test_r1_a_batch_provenance_input_is_refused(group):
    """A `schedule` Input is the batch's, and so is a `prefix` on a request with no checked hit."""
    adv, ing = _mixed()
    with pytest.raises(BatchDependent, match=rf"GP-01: the outputs of owner 0 read request\[r0\]\.{group} .* batch-derived"):
        check_compiled(None, adv, ing, _compile(adv, ing, groups=("weights", "prompt", group)), plan)


def test_workload_labels_by_provenance():
    adv, ing = _mixed()
    c = _compile(adv, ing, groups=("weights", "prompt", "seed", "splits", "prefix"))
    labels, ret = workload_labels(c.workload)
    assert labels == (SHARED, *(x for r in range(3) for x in (own(r), own(r), BATCH, BATCH)))
    assert workload_labels(c.workload, frozenset({1}))[0][5:9] == (own(1), own(1), BATCH, own(1))
    assert ret.labels == (own(0), own(1), own(2))
    bad = replace(c.workload.requests[0], component=replace(c.workload.requests[0].component, groups=("weights",)))
    with pytest.raises(BatchDependent, match="names no group"):
        workload_labels(replace(c.workload, requests=(bad, *c.workload.requests[1:])))


# --- on the real compiler --------------------------------------------------------------------------------------------------------


def test_compile_steps_passes_its_own_rule():
    """At top_p 1 and below: the compiler emits the registered split count, so R3 holds at any top_p."""
    from verity.ml.definitions.registry.topp_geometry import registered_splits
    steps = pytest.importorskip("verity_vllm.program.predict.steps")
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

    for b1 in ({**greedy, "max_tokens": 2}, {**topp, "top_p": 1.0}, topp):
        adv, ing = session(b1)
        c = steps.compile_steps(reg, adv, ing)
        check_compiled(reg, adv, ing, c, steps.plan_steps)
        if b1["temperature"]:
            assert c.requests[1].policy == {"kind": "stochastic", "temperature": 0.8, "top_p": b1["top_p"], "top_k": 0, "seeded": True,
                                            "splits": registered_splits(RTX["num_sms"])}
