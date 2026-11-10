"""verity_vllm.program.predict.reads: the compiler reads of the advice and of a certified body only the fields `READS` declares,
each a checked circuit output, certified or committed advice, and reads the advice through the gate that refuses any other by
name; and prefix stability, changing y at or after a step changing nothing the circuit computes before it. A window cut after
any step plans and compiles those steps as the whole does, and another length for a request in flight moves only the unit
fields `READS` declares its length read ahead for: `T` on a batch-invariant target, which sizes only its Program's later
decodes, `stop`, its length rule (`RequestUnit.stop`, whose config holds how many outputs its Program has), and `edge` in
class G, what the window's close returns, for which the Program also computes its indexer's keys and scales for the rows it
ran, the one read ahead that isn't prefix stable (accounted only as the advice it is)."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import fields, replace

import pytest

from verity.ml.circuits.types import Array, field_types
from verity_vllm.pipeline import window_advice as WA
from verity_vllm.program.predict import reads as R
from verity_vllm.program.predict.common import Unexpressible
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import Registered, RequestUnit, compile_steps, plan_steps, request_program
from tests.program.predict.test_dsa_long_rows import I32, TOPK, _canonical, _short, _step
from tests.program.predict.test_steps import (
    A_BODY,
    A_CHUNKED,
    A_IN,
    A_MID_IN,
    A_WHOLE,
    CACHE_ON,
    GREEDY,
    HF_CONFIG_DIR,
    _b8_window,
    _class_g,
    _hit,
    _hit_chunked,
    _mixed,
    _registered,
)

pytestmark = pytest.mark.usefixtures("code_versions_stood_in")

#: the advice records' fields `READS` declares, the ones the gate passes
GATED = {(f.record, f.name) for f in R.READS if f.record in R.RECORDS}
#: a unit's fields the window's end shapes for a request it leaves with outputs to keep: how far it reaches and what its edge returns
REACH = {"T", "carried_out", "launch", "chunks", "edge", "stop"}
AHEAD = {n for f in R.READS for n in f.ahead}


def test_every_field_the_compiler_reads_is_an_output_certified_or_advice():
    assert R.refusals() == []
    assert [(f.record, f.name) for f in R.READS if f.ahead] == [("request", "output_len"), ("request", "finish")]
    assert AHEAD <= {f.name for f in fields(RequestUnit)}
    assert ("advice", "release") not in GATED, "the compiler never reads when an output was released"


@pytest.mark.parametrize("field,why", [
    (R.Field("served", "token_ids", "output"), "served.token_ids: of no record the compiler receives"),
    (R.Field("program", "samples", "certified"), "program.samples: of no record the compiler receives"),
    (R.Field("advice", "release", "served"), "advice.release: account 'served', none of output, certified, advice"),
    (R.Field("request", "stop", "advice"), "request.stop: not a field of the advice's request record"),
    (R.Field("request", "output_len", "output"), "request.output_len: a field of the advice, not a circuit output"),
    (R.Field("request", "prompt_len", "certified", ahead=("T",)), "request.prompt_len: read ahead of the step that outputs it "
                                                                   "(it shapes T), so it must be committed advice"),
    (R.Field("body", "temperature", "advice", role="policy"), "body.temperature: a certified body's field, account 'advice'"),
    (R.Field("body", "min_p", "certified", role="policy", inert=(0,)), "body.min_p: role 'policy' with inert values (0,)"),
    (R.Field("body", "seed", "advice", role="policy"), "body.seed: a stochastic request's seed is certified, its body's seed or "
                                                       "derived from its id and the window's seed salt"),
    (R.Field("served", "seed", "output"), "served.seed: a stochastic request's seed is certified"),
    (R.Field("row", "cache_hit", "advice"), "row.cache_hit: a prefix-cache hit is a row's advice, charged at its width, under "
                                            "enable_prefix_caching: never derived by verifier-side Python"),
    (R.Field("row", "sources", "certified", switch=R.PREFIX_CACHE), "row.sources: a prefix-cache hit is a row's advice"),
    (R.Field("served", "cache_hit", "output"), "served.cache_hit: a prefix-cache hit is a row's advice"),
    (R.Field("body", "stream", "certified", role="transport", switch=R.PREFIX_CACHE),
     "body.stream: switched by enable_prefix_caching, so it is advice the registration can switch off, not 'certified'"),
    (R.Field("window", "salt", "certified"), "window.salt: of the window the compiler reads only its seed salt"),
    (R.Field("window", "coin", "certified"), "window.coin: of the window the compiler reads only its seed salt, certified"),
    (R.Field("window", "seed_salt", "advice"), "window.seed_salt: a stochastic request's seed is certified"),
])
def test_a_field_the_accounting_cant_take_is_refused_by_name(field, why):
    table = tuple(f for f in R.READS if (f.record, f.name) != (field.record, field.name)) + (field,)
    assert any(m.startswith(why) for m in R.refusals(table)), R.refusals(table)
    assert R.refusals(R.READS + R.READS[:1]) == ["advice.requests: declared twice"]


def test_the_gate_refuses_a_field_reads_doesnt_declare_by_name():
    g = R.gate(A_WHOLE)
    for read, at in ((lambda: g.release, "advice.release"), (lambda: g.version, "advice.version"),
                     (lambda: g.requests[0].i1, "request.i1")):
        with pytest.raises(R.UndeclaredRead, match=re.escape(f"the compiler reads {at}, which `predict.reads.READS` doesn't declare")):
            read()
    assert (g.requests[0].output_len, g.steps[1].rows[0].start, g.steps is g.steps) == (4, 3, True)
    with pytest.raises(AttributeError, match="read-only"):
        g.requests = ()
    with pytest.raises(TypeError, match="a window's Advice, not a Advice"):
        R.gate(A_WHOLE.i1())


def test_the_table_is_the_compilers_read_path(monkeypatch):
    """Without its declaration a field the planner reads fails it by name, and the planner's body roles are the table's."""
    from verity_vllm.program.predict import steps
    monkeypatch.setattr(R, "_DECLARED", R._DECLARED - {("request", "finish")})
    with pytest.raises(R.UndeclaredRead, match=re.escape("the compiler reads request.finish")):
        plan_steps(_registered(), *_mixed())
    assert steps._INERT == {f.name: f.inert for f in R.READS if f.role == "inert"}
    assert set(steps._KNOWN) | set(steps._TRANSPORT) == {f.name for f in R.READS if f.record == "body"}


def test_the_compiler_reads_every_field_the_gate_passes(monkeypatch, tmp_path):
    """No stale declaration: over sessions with a cache hit, a preemption-free resume of a carried request and chunked
    prefills, planned on a batch-invariant target and in class G and compiled, the compiler reads each declared advice field."""
    seen: set[tuple[str, str]] = set()
    get = R._View.__getattr__

    def spy(self, name):
        if not name.startswith("_"):
            seen.add((R._KIND[type(self._record)], name))
        return get(self, name)

    monkeypatch.setattr(R._View, "__getattr__", spy)
    reg = _registered()
    compile_steps(reg, *_hit(16))
    plan_steps(reg, A_IN, A_BODY)
    g = _class_g(tmp_path, engine=CACHE_ON)
    for session in ((A_MID_IN, A_BODY), _hit_chunked()):
        plan_steps(g, *session)
    assert seen == GATED


def test_a_prefix_cache_hit_is_advice_under_the_switch_and_absent_where_the_registration_states_it_off(tmp_path):
    """Both modes behind `enable_prefix_caching`: on, or unstated (vLLM's default), a hit compiles as the advice it is; stated off,
    the registration's advice has no hit, and one is refused by name on a batch-invariant target and in class G alike, a row's
    and a carried request's."""
    hit = frozenset({("row", "cache_hit"), ("row", "sources"), ("carried", "C")})
    assert R.off({}) == R.off({R.PREFIX_CACHE: True}) == frozenset() and R.off({R.PREFIX_CACHE: False}) == hit
    assert compile_steps(_registered(), *_hit(16)).requests[1].C == 16
    assert plan_steps(_registered(engine={R.PREFIX_CACHE: True}), *_hit(16)).requests[1].C == 16
    off = _registered(engine={R.PREFIX_CACHE: False})
    plan_steps(off, *_mixed())
    why = "request 1: a prefix-cache hit of 16 tokens, and the registration states the prefix cache off (enable_prefix_caching False)"
    with pytest.raises(Unexpressible, match=re.escape(why)):
        plan_steps(off, *_hit(16))
    assert plan_steps(off, *_hit(16), partial=True).refused[1].startswith(why)
    g = _class_g(tmp_path, engine={R.PREFIX_CACHE: False})
    with pytest.raises(Unexpressible, match=re.escape("request 1: a prefix-cache hit of 16 tokens, and the registration states")):
        plan_steps(g, *_hit_chunked())
    carried = WA.Advice((WA.RequestAdvice("a", 20, 4, "length"),),
                        tuple(WA.StepAdvice((WA.Row(0, "decode", p, 1, 0, False, ()),)) for p in (21, 22)), (0, 1),
                        (WA.Carried(16, 21, 2, ((16, 21),)),))
    body = {"a": {**GREEDY, "prompt": list(range(1, 21))}}
    assert plan_steps(_class_g(tmp_path, engine=CACHE_ON), carried, body).requests[0].carried_in
    with pytest.raises(Unexpressible, match=re.escape("request 0: carried in on a prefix-cache hit of 16 tokens, and the registration "
                                                      "states the prefix cache off")):
        plan_steps(g, carried, body)


def _window(advice) -> WA.Advice:
    return advice if type(advice) is WA.Advice else WA.Advice(advice.requests, advice.steps, advice.release)


def _cut(advice: WA.Advice, s: int) -> WA.Advice:
    """``advice``'s window closed after step ``s``: its steps to there, the requests admitted by then (the carried ones first) and
    the release buckets of the outputs kept by then; it leaves each request with outputs still to keep to the next window."""
    n = max(len(advice.carried), 1 + max(row.request for step in advice.steps[:s + 1] for row in step.rows))
    release = tuple(b for o, b in zip(WA.outputs(advice), advice.release) if o.step <= s)
    return WA.Advice(advice.requests[:n], advice.steps[:s + 1], release, advice.carried)


def _bodies(advice: WA.Advice, ingress: dict) -> dict:
    return {q.id: ingress[q.id] for q in advice.requests}


def _diff(u: RequestUnit | None, w: RequestUnit | None) -> set[str]:
    if u is None or w is None:
        return set() if u is w else {"unit"}
    return {f.name for f in fields(u) if getattr(u, f.name) != getattr(w, f.name)}


def _lengths(cut: WA.Advice, ingress: dict) -> list[tuple[int, int, str]]:
    """Each request ``cut`` leaves with outputs to keep, at each other length its body allows past the outputs it kept (the
    least and the cap), with a finish that length can have: the cap's, else an abort, or a stop token unless ``ignore_eos``."""
    out = []
    for c in WA.close(cut):
        q, body = cut.requests[c.request], ingress[cut.requests[c.request].id]
        if not c.must:
            continue
        m = body["max_tokens"]
        for L in sorted({c.state.kept + 1, m}):
            for finish in ("length",) if L == m else ("abort",) + ("eos",) * (not body.get("ignore_eos")):
                if (L, finish) != (q.output_len, q.finish):
                    out.append((c.request, L, finish))
    return out


def _prefix_stable(reg: Registered, advice, ingress: dict) -> set[str]:
    """Hold ``advice``'s plan prefix stable at every step s. Cut after s, the window plans each step to there as the whole does:
    its live and discarded rows, and each unit but for how far a request the cut leaves reaches, its launch a prefix of the
    whole's. And under each other length a request the cut leaves could have (y changed after s), the cut plans the same rows,
    its units differing only in fields `READS` declares that length read ahead for. Returns the fields some length moved."""
    advice = _window(advice)
    whole = plan_steps(reg, advice, ingress)
    moved: set[str] = set()
    for s in range(len(advice.steps) - 1):
        cut = _cut(advice, s)
        plan, left = plan_steps(reg, cut, _bodies(cut, ingress)), {c.request for c in WA.close(cut) if c.must}
        assert (plan.steps, plan.discarded, plan.refused) == (whole.steps[:s + 1], [d for d in whole.discarded if d[0] <= s], {})
        for i, u in enumerate(plan.requests):
            if i in plan.idle:
                continue
            assert _diff(u, whole.requests[i]) <= (REACH if i in left else set()), (s, i)
            if u is not None and u.launch is not None:
                assert u.launch == whole.requests[i].launch[:len(u.launch)], (s, i)
        for r, L, finish in _lengths(cut, ingress):
            requests = tuple(replace(q, output_len=L, finish=finish) if i == r else q for i, q in enumerate(cut.requests))
            other = plan_steps(reg, WA.Advice(requests, cut.steps, cut.release, cut.carried), _bodies(cut, ingress))
            assert (other.steps, other.discarded) == (plan.steps, plan.discarded), (s, r, L)
            for i, (u, w) in enumerate(zip(other.requests, plan.requests)):
                d = _diff(u, w)
                assert d <= (AHEAD if i == r else set()), (s, r, L, d)
                moved |= d
    return moved


def _calls(program) -> list[str]:
    """``program``'s calls but its length rule's (`RequestUnit.stop`), the last call, whose config `READS` charges as advice."""
    return [n.fn.id for n in program.fn.body.nodes if not n.fn.id.startswith("StopCheck_")]


def _members(program) -> list[str]:
    """`_calls`, a batch's once per member: the Calls a batched Program makes (`prune.batched_form`)."""
    return [i for n in program.fn.body.nodes if not n.fn.id.startswith("StopCheck_") for i in [n.fn.id] * n.n]


def test_a_batch_invariant_plan_is_prefix_stable_and_a_length_sets_only_its_t():
    """Batch-invariant: a request's Program is its whole request's in every window, so its length sets its T ahead of its stop,
    and nothing else in the plan moves."""
    reg = _registered()
    for session in (_mixed(), _hit(16), (A_WHOLE, A_BODY)):
        assert _prefix_stable(reg, *session) == {"T", "stop"}


@pytest.mark.slow
def test_a_batch_invariant_program_is_prefix_stable_in_its_length():
    """At every cut each request compiles to the Program it has whole, and its Programs at each length its body allows are,
    call for call but its length rule's check, prefixes of one another, returning its token ids first, ``Array(L, I32)``: its
    length sizes only its later decodes and its length rule (about 6 s)."""
    reg = _registered()
    advice, ingress = _mixed()
    whole = compile_steps(reg, advice, ingress)
    for s in range(len(advice.steps) - 1):
        cut = _cut(_window(advice), s)
        c = compile_steps(reg, cut, _bodies(cut, ingress))
        assert [u.digest for u in c.requests] == [u.digest for u in whole.requests[:len(c.requests)]], s
    for u in whole.requests[:2]:
        m = ingress[u.id]["max_tokens"]
        built = [request_program(reg, replace(u, T=L - 1)).program for L in range(1, m + 1)]
        assert [field_types(p.fn.signature()[1])[0] for p in built] == [Array(L, I32) for L in range(1, m + 1)]
        assert all(_calls(p) == _calls(built[-1])[:len(_calls(p))] for p in built), u.id


def test_a_class_g_window_is_prefix_stable_at_every_step(tmp_path):
    """Class G: a request's Program is its rows in the window, at the launches they ran in; cut after any step, each unit is the
    whole's up to there, and no length moves a unit while INDEX_TOPK (2,048 keys) is past every row: the B8 row's window, a
    cache hit with a chunked prefill, and a request prefilled in chunks, whole and carried in mid-prefill."""
    reg = _class_g(tmp_path, engine=CACHE_ON)
    for session in (_b8_window()[:2], _hit_chunked(), (A_CHUNKED, A_BODY), (A_MID_IN, A_BODY)):
        assert _prefix_stable(reg, *session) == set()


def _past_topk(tmp_path) -> tuple[Registered, WA.Advice, dict]:
    """Class G at INDEX_TOPK 4 (GLM-5.3 cut to its first 4 layers, `TOPK`): a (3 prompt tokens, 5 outputs, its rows past
    INDEX_TOPK from position 4) and b (2 and 3, stopping before its cap of 5, within INDEX_TOPK unless it runs on) in one
    window."""
    hf = _short(json.loads((HF_CONFIG_DIR / "GLM53.config.json").read_text()))
    reg = _class_g(tmp_path, engine=CACHE_ON, model={"hf_config": hf})
    reg = Registered.of(Phi.from_json({**reg.phi.to_json(), "engine": _canonical(reg.phi.engine)}), default_seed=7)
    assert reg.phi.hf_config["index_topk"] == TOPK
    steps = (_step((0, 3, 0, 3), (1, 2, 0, 2)), _step((0, 3, 3, 1), (1, 2, 2, 1)), _step((0, 3, 4, 1)), _step((0, 3, 5, 1)),
             _step((1, 2, 3, 1)), _step((0, 3, 6, 1)))
    advice = WA.Advice((WA.RequestAdvice("a", 3, 5, "length"), WA.RequestAdvice("b", 2, 3, "eos")), steps, (0,) * 8)
    return reg, advice, {"a": {**GREEDY, "max_tokens": 5, "prompt": [1, 2, 3]}, "b": {**GREEDY, "max_tokens": 5, "prompt": [4, 5]}}


def _running_on(cut: WA.Advice, r: int, L: int) -> WA.Advice:
    return replace(cut, requests=tuple(replace(q, output_len=L) if i == r else q for i, q in enumerate(cut.requests)))


def test_a_class_g_length_past_index_topk_sizes_only_what_the_windows_close_returns(tmp_path):
    """At INDEX_TOPK 4, a window's close returns a request it leaves with outputs to keep with its indexer's keys and scales on
    its edge where its last row reads past INDEX_TOPK keys: its length, read ahead of its stop, sets its edge and nothing else."""
    reg, advice, ingress = _past_topk(tmp_path)
    assert _prefix_stable(reg, advice, ingress) == {"edge"}
    cut = _cut(advice, 1)
    assert [u.edge for u in plan_steps(reg, cut, ingress).requests] == [("prefix", "prefix_ik", "prefix_iks"), ("prefix",)]
    assert plan_steps(reg, _running_on(cut, 1, 4), ingress).requests[1].edge == ("prefix", "prefix_ik", "prefix_iks")


@pytest.mark.slow
def test_a_class_g_program_is_prefix_stable_at_every_cut_but_for_its_edges_indexer_history(tmp_path):
    """Class G at INDEX_TOPK 4 on GLM-5.3's first four layers (about 35 s). A (3 prompt tokens, 4 outputs, prefilled in chunks
    of 2 and 1, its rows at positions 4 and 5 past INDEX_TOPK) compiled in a window cut after each step is, call for call, its
    whole Program up to there, and its edge carries its indexer's history from the first cut on, before any row has read past
    INDEX_TOPK. But b, left after two outputs, computes its indexer's keys for the 3 rows it ran (a key row and its projection
    in each of the 3 indexer layers) only where it runs on to 4 outputs, so that the close returns them, its calls stopping at
    3 a subsequence of those: the one read of y ahead that isn't prefix stable, accounted as the advice ``output_len`` is.
    What no cheaper test can: that the Programs built from the units are prefix stable but there, a request's Program at a step
    making the calls it makes whole whatever y holds after it; the split tests hold one cut a session, and the plan-level tests
    the units, not the Programs built from them."""
    reg, advice, ingress = _past_topk(tmp_path)
    calls, edges = [], []
    for s in range(len(A_CHUNKED.steps)):
        c = compile_steps(reg, A_CHUNKED if s == len(A_CHUNKED.steps) - 1 else _cut(A_CHUNKED, s), A_BODY)
        (p,) = c.programs.values()
        assert c.batch_rule is None
        calls.append(_calls(p))
        edges.append(c.requests[0].edge)
    assert all(cut == calls[-1][:len(cut)] for cut in calls) and len({len(cut) for cut in calls}) == len(calls)
    assert edges == [("prefix", "prefix_ik", "prefix_iks")] * (len(calls) - 1) + [()]
    cut = _cut(advice, 1)
    stops, runs = (request_program(reg, plan_steps(reg, a, ingress).requests[1]).program for a in (cut, _running_on(cut, 1, 4)))
    left = iter(_calls(runs))
    assert all(f in left for f in _calls(stops))
    assert Counter(_members(runs)) - Counter(_members(stops)) == Counter(
        {"CublasSplitKGemm_v1{K=6144,N=160,SPLIT_K=4,DOT=HopperBF16WgmmaDot16_v1}": 9, "DsaIndexerKRow_v1{D=128,ROPE=64,EPS=1e-06}": 9})
    ret = [field_types(p.fn.signature()[1]) for p in (stops, runs)]
    assert ret[1][:len(ret[0])] == ret[0] and len(ret[1]) == len(ret[0]) + 2
