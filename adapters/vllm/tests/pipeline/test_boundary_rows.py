"""A served window's boundary commitment (`pipeline.boundary_rows`): each span's `b.ingress` and `b.egress` rows word for
word, the size classes, one class's commitment under core's window domains registering by both verifiers, its files
recommitting to its roots, the serving path's hook on a stand-in for vLLM's V1 scheduler committing each window's
finished requests by class as it closes, and a serve's stop rule: a stop class's rows, its stop word the stop block's
output over the committed rows, the stop blocks' own registration over the class's boundary record, and a response one
id shorter or longer than the rule's refused."""

from __future__ import annotations

import json
import struct
from types import SimpleNamespace as NS

import numpy as np
import pytest

from verity.protocols.spot.verifier.one_stage import partition as P, registration as R, verifier as V
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.evaluation import evaluate
from verity.circuits.population import population
from verity.circuits.program import Program
from verity.circuits.types import Array, Value as Bits
from verity.commitments.windows import rows as CS
from verity.definitions.registry import stop_check, stop_rule
from verity_vllm.engine.hooks import Hooks
from verity_vllm.pipeline import boundary_rows as BR

XOR8 = PrimitiveDefinition("BoundaryRowsTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
#: the check template: a proof block of one gate's worth
PAIR = bind(CompositeDefinition("BoundaryRowsTestPair", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))
NOT8 = PrimitiveDefinition("BoundaryRowsTestNot8", 1, (("a", Bits(8)),), Bits(8), lambda a: a ^ 0xFF, register=False)
NOT = bind(CompositeDefinition("BoundaryRowsTestNot", 0, (), lambda S_: ((("a", Bits(8)),), Bits(8)),
                               lambda B, S_, a: B.call(NOT8, a), register=False))
#: a check template whose body calls two templates, as one too large for a proof block calls its pieces
CHAIN = bind(CompositeDefinition("BoundaryRowsTestChain", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                 lambda B, S_, x: B.call(NOT, B.call(PAIR, x)), register=False))
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": 4, "min_tokens": 0,
          "ignore_eos": True, "seed": None, "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
LAW = {"law": "subset", "k": 2}


def _spans(n: int = 5, prompt: int = 3, served: int = 4) -> list[BR.Span]:
    return [BR.Span(f"r{i}", tuple(range(10 * i, 10 * i + prompt)), GREEDY, tuple(range(100 + i, 100 + i + served)), "length")
            for i in range(n)]


def _layout(n: int = 5, run: str = "a" * 128) -> BR.Layout:
    return BR.Layout(PAIR, 0, 1, 4, 4, n, run)


def _u32s(words: np.ndarray) -> list[int]:
    return [int(lo) | int(hi) << 16 for lo, hi in words.reshape(-1, 2)]


def test_each_span_is_one_ingress_row_and_one_egress_row_laid_out_as_the_spec_says():
    """`b.ingress`: lp, the sampling inputs (floats as f32 bits, a negative top_k two's complement, the seed a u64 low
    first), then the prompt's ids at the class's cap, zero past it; `b.egress`: the ids served n, the finish's code, then
    the ids, zero past them; every u32 two u16 words, low first."""
    s = BR.Span("e", (70000, 5), {**GREEDY, "temperature": 0.75, "top_k": -1, "seed": (1 << 40) + 3, "max_tokens": 9},
                (1, 2, 3), "stop")
    ing = _u32s(BR.ingress_row(s, 4))
    f32 = struct.unpack("<I", struct.pack("<f", 0.75))[0]
    assert ing == [2, 9, 0, 1, f32, struct.unpack("<I", struct.pack("<f", 1.0))[0], 0xFFFFFFFF, 0, 0, 0,
                   struct.unpack("<I", struct.pack("<f", 1.0))[0], 1, 3, 1 << 8, 70000, 5, 0, 0]
    assert len(ing) == BR.HEAD_U32 + 4 and BR.ingress_row(s, 4)[-8:-6].tolist() == [70000 & 0xFFFF, 70000 >> 16]
    assert _u32s(BR.egress_row(s, 4)) == [3, 1, 1, 2, 3, 0]
    assert _u32s(BR.egress_row(BR.Span("u", (1,), GREEDY), 2)) == [0, 0, 0, 0]
    with pytest.raises(ValueError, match="past the class"):
        BR.ingress_row(s, 1)
    with pytest.raises(ValueError, match="u32"):
        BR.egress_row(BR.Span("n", (1,), GREEDY, (-1,), "stop"), 2)


def test_a_span_is_classed_by_its_prompt_and_its_outputs_each_capped_at_a_power_of_two():
    assert [BR.cap(n) for n in (0, 1, 2, 3, 4, 5, 255, 256, 257)] == [2, 2, 2, 4, 4, 8, 256, 256, 512]
    spans = [BR.Span("a", (1,) * 3, GREEDY, (2,) * 4), BR.Span("b", (1,) * 9, GREEDY, (2,) * 2),
             BR.Span("c", (1,) * 4, GREEDY, (2,) * 3)]
    assert {k: [s.engine_id for s in v] for k, v in BR.filed(spans).items()} == {(4, 4, False): ["a", "c"],
                                                                                 (16, 2, False): ["b"]}
    with pytest.raises(ValueError, match="two proof blocks or more"):
        BR.Layout(PAIR, 0, 0, 4, 4, 1, "a" * 128)


def test_a_class_commits_under_core_window_domains_and_registers_by_both_verifiers():
    """One class's record: `population_program(replay, N)` under the replay grain, each value's root under core's window
    domain (`registration.window_domain`), accepted by one-stage's R1-R7 and by the Lean verifier against the
    challenger's copies from public items alone (`layout_of` over ``commit.json``), and refused under another class's
    copies. Fresh salts give fresh roots over the same rows."""
    spans, L = _spans(), _layout()
    c = BR.commit(spans, L, 0)
    for p in BR.PORTS:
        assert L.ports()[p]["domain"] == R.window_domain(L.partition["program"], P.digest(L.partition), p, BR.SCHEMA, 5, L.run)
    assert (L.n_rho, P.units(L.program, L.grain)) == (4, 5)
    pub = json.loads(json.dumps(c.public()))
    mine = BR.layout_of(pub, PAIR)
    rec = BR.record(pub, PAIR, LAW)
    assert rec == c.record(LAW) and rec["window"]["kind"] == "served" and rec["window"]["index"] == c.index_leaf
    assert R.check(rec, mine.program, mine.grain, LAW, CS.SCHEME, mine.ports()).codes == ()
    assert V.check(rec, mine.program, mine.grain, LAW, CS.SCHEME, mine.ports()).codes == ()
    other = _layout(n=6)
    assert R.check(rec, other.program, other.grain, LAW, CS.SCHEME, other.ports()).codes != ()
    assert BR.commit(spans, L, 0).roots != c.roots
    with pytest.raises(ValueError, match="not 5 of class"):
        BR.commit(spans[:4], L, 0)


def test_a_class_reopens_from_its_files_and_refuses_a_changed_one(tmp_path):
    c = BR.commit(_spans(), _layout(), 3)
    d = BR.write(c, tmp_path / "w3" / "p4-o4")
    assert sorted(x.name for x in d.iterdir()) == sorted([BR.COMMIT, BR.INDEX, BR.SALTS, CS.INDEX_SALT])
    again = BR.reopen(d, PAIR)
    assert again.roots == c.roots and again.index_leaf == c.index_leaf and again.spans == c.spans
    idx = json.loads((d / BR.INDEX).read_text())
    idx["spans"][0]["served"][0] += 1
    (d / BR.INDEX).write_text(json.dumps(idx) + "\n")
    with pytest.raises(ValueError, match="recommit to another values"):
        BR.reopen(d, PAIR)


def test_an_inlined_check_is_checked_by_the_templates_its_body_calls(tmp_path):
    """``inline``: the replay block is the check template's instances with each body in place, computing what their
    population computes and holding only their outputs; its proof blocks are the templates the body calls, in the order of
    their first call; and the class commits, registers and reopens with ``inline`` in its record, under other domains than
    the population's. A primitive has no body to inline."""
    L = BR.Layout(CHAIN, 0, 1, 4, 4, 5, "a" * 128, inline=True)
    assert [P.template_id(t) for t in L.templates] == [P.template_id(PAIR), P.template_id(NOT)]
    assert L.query == P.template_instances_query([PAIR, NOT]) and P.units(Program(L.replay), L.query) == 8
    assert L.replay.signature()[1] == Array(4, Bits(8))
    xs = [3, 5, 7, 7, 0, 255, 1, 2]
    want = tuple(0xFF ^ a ^ b for a, b in zip(xs[::2], xs[1::2]))
    assert evaluate(L.replay, xs) == evaluate(population(CHAIN, 4), xs) == want
    c = BR.commit(_spans(), L, 0)
    pub = json.loads(json.dumps(c.public()))
    assert pub["inline"] is True and BR.layout_of(pub, CHAIN) == L
    assert V.check(BR.record(pub, CHAIN, LAW), L.program, L.grain, LAW, CS.SCHEME, L.ports()).codes == ()
    plain = BR.Layout(CHAIN, 0, 1, 4, 4, 5, "a" * 128)
    assert plain.templates == (CHAIN,) and plain.ports()[BR.INGRESS]["domain"] != L.ports()[BR.INGRESS]["domain"]
    assert BR.reopen(BR.write(c, tmp_path / "w0" / "p4-o4"), CHAIN).roots == c.roots
    with pytest.raises(ValueError, match="composite"):
        BR.inlined(XOR8, 2)
    with pytest.raises(ValueError, match="calls of templates"):
        BR.Layout(PAIR, 0, 1, 4, 4, 5, "a" * 128, inline=True)
    bw = BR.BoundaryWindows.of({"check": P.template_id(CHAIN), "inline": True}, tmp_path / "of")
    bw.stop()
    assert bw.inline and not BR.BoundaryWindows.of({"check": "c"}, tmp_path / "of2").inline


class Request:
    def __init__(self, rid: str, prompt: list[int], max_tokens: int = 4, **sampling):
        self.request_id, self.prompt_token_ids = rid, prompt
        self.sampling_params = NS(**{**GREEDY, "max_tokens": max_tokens, **sampling})


class Scheduler:
    """vLLM's V1 scheduler as the hook wraps it."""

    def __init__(self, caching: bool = False):
        self.cache_config = NS(enable_prefix_caching=caching)
        self.added: list[str] = []
        self.outputs: dict = {}

    def add_request(self, request):
        self.added.append(request.request_id)

    def schedule(self):
        return NS()

    def update_from_output(self, so, mro):
        out, self.outputs = self.outputs, {}
        return out

    def finish_requests(self, request_ids, finished_status):
        return []


def _step(sched: Scheduler, served: dict[str, tuple[list[int], str | None]]) -> None:
    sched.outputs = {0: NS(outputs=[NS(request_id=k, new_token_ids=ids, finish_reason=None if f is None else NS(name=f.upper()))
                                    for k, (ids, f) in served.items()])}
    sched.schedule()
    sched.update_from_output(None, None)


def test_the_hook_commits_each_windows_finished_requests_by_class_as_it_closes(tmp_path):
    """Three requests: at the first schedule a window's length after it opened, window 0 closes with the two finished,
    by class, and carries the third; the third's finish, the only one running, closes window 1; each class's files
    reopen to the roots its sink line holds; an abort finishes a request; and prefix caching is refused."""
    now, lines = [0.0], []
    sched, hooks = Scheduler(), Hooks(tag="test-boundary-rows")
    bw = BR.BoundaryWindows(tmp_path, PAIR, 0, 1, 10.0, sink=lines.append, clock=lambda: now[0], serve="s").install(sched, hooks)
    try:
        sched.add_request(Request("a", [1, 2, 3]))
        sched.add_request(Request("b", [4] * 9))
        sched.add_request(Request("c", [5, 6, 7, 8]))
        _step(sched, {"a": ([7, 8, 9], "length"), "b": ([1], None), "c": ([2, 3, 4, 5], "stop")})
        assert bw.window == 0 and lines == []
        now[0] = 10.0
        _step(sched, {"b": ([2], None)})
        assert bw.window == 1 and list(bw.open) == ["b"]
        _step(sched, {"b": ([3], "length")})
        assert bw.window == 2 and bw.open == {}
        sched.add_request(Request("d", [1]))
        sched.finish_requests(["d"], NS(name="FINISHED_ABORTED"))
        assert bw.window == 3
        bw.stop()
        bw.stop()
        got = {(x["window"], x["prompt"], x["outputs"]): x for x in lines}
        assert sorted(got) == [(0, 4, 4), (1, 16, 4), (2, 2, 2)] and all("error" not in x for x in lines)
        assert got[(0, 4, 4)]["spans"] == 2 and got[(0, 4, 4)]["carried"] == 1 and got[(1, 16, 4)]["carried"] == 0
        for (w, p, o), x in got.items():
            c = BR.reopen(tmp_path / f"w{w}" / f"p{p}-o{o}", PAIR)
            assert {k: v.hex() for k, v in c.roots.items()} == {k: v["root"] for k, v in x["values"].items()}
            assert x["run"] == BR.run_root("s", w)
        a, c_ = BR.reopen(tmp_path / "w0" / "p4-o4", PAIR).spans
        assert (a.engine_id, a.served, a.finish, c_.engine_id, c_.finish) == ("a", (7, 8, 9), "length", "c", "stop")
        assert BR.reopen(tmp_path / "w2" / "p2-o2", PAIR).spans[0].finish == "abort"
    finally:
        bw.stop()
        hooks.uninstall()
    with pytest.raises(ValueError, match="prefix caching off"):
        BR.BoundaryWindows(tmp_path / "x", PAIR, 0, 1, 10.0, clock=lambda: 0.0).install(Scheduler(caching=True), Hooks())


#: a serve's stop rule: two stop ids, a model of 64 ids, responses of at most 4
RULE = BR.Stop((3, 7), 64, 4)
#: a model that states no stop id
NO_IDS = BR.Stop((), 64, 4)
#: a law over a class's stop blocks
STOP_LAW = {"law": "subset", "k": 2}


def _ended(rid: str, served: tuple[int, ...], finish: str | None, max_tokens: int = 4,
           prompt: tuple[int, ...] = (1, 2, 5), **sampling) -> BR.Span:
    return BR.Span(rid, prompt, {**GREEDY, "ignore_eos": False, "max_tokens": max_tokens, **sampling}, served, finish)


#: spans the rule ends as they were served: at a stop id, at ``max_tokens`` twice, and at the model's length
HELD = [_ended("stop", (9, 9, 7), "stop"), _ended("length", (9, 9, 9, 9), "length"), _ended("short", (9, 9), "length", 2),
        _ended("model", (9, 9), "length", 4, tuple(range(10, 72)))]


def _stop_layout(n: int = 3, prompt: int = 4) -> BR.Layout:
    return BR.Layout(PAIR, 0, 1, prompt, RULE.outputs, n, "a" * 128, stop=RULE)


def _word(n: int, finish: int) -> int:
    return n + (finish << stop_rule.N_BITS)


def _block(rule: BR.Stop, span: BR.Span, served: tuple[int, ...] | None = None) -> int:
    """The stop block's output on the rows a class under ``rule`` holds of ``span``, its `b.egress` row of ids ``served``."""
    ids = span.served if served is None else served
    head = _u32s(BR.ingress_row(span, BR.cap(len(span.prompt))))[:BR.HEAD_U32]
    (word,) = evaluate(rule.template, head, [*ids, *[0] * (rule.outputs - len(ids))])
    return word


def _flat(rows: np.ndarray) -> list[int]:
    return [x for r in rows for x in _u32s(r)]


def test_the_stop_block_is_the_stop_rule_over_a_spans_request_row_and_served_ids():
    """`egress_stop` is `StopRule16_v1` at the registered stop ids and model length, reading ``max_tokens`` and the
    prompt's length off the span's `b.request` row and returning the rule's word, n + finish * 8192: a stop id ends a
    response at it and counts, else ``max_tokens``, else the model's length; no stop ids, and no id stops it; and past
    the cap it runs on, finish 0, where `Stop.word` holds no span. A cap the word's n can't hold is refused."""
    params, ret = RULE.template.signature()
    assert [(k, t.leaves) for k, t in params] == [("request", BR.HEAD_U32), ("ids", 4)] and ret == Bits(16)
    assert (BR.FIELD_AT["lp"], BR.FIELD_AT["max_tokens"]) == (0, 1)
    assert [_block(RULE, s) for s in HELD] == [_word(3, 1), _word(4, 2), _word(2, 2), _word(2, 3)]
    assert [RULE.word(s) for s in HELD] == [_block(RULE, s) for s in HELD]
    for ids, mt, lp in [((9, 3, 7, 9), 4, 3), ((9, 9, 9, 9), 3, 1), ((7, 9, 9, 9), 4, 62), ((9, 9, 9, 9), 4, 61),
                        ((0, 0, 0, 0), 9, 0), ((9, 9, 9, 3), 0, 70)]:
        s = _ended("x", ids, "length", mt, tuple(range(lp)))
        assert _block(RULE, s) == stop_rule.reference16(list(ids), mt, RULE.ids, lp, RULE.max_model_len)
    assert NO_IDS.template.id != RULE.template.id
    assert _block(NO_IDS, HELD[0], (9, 9, 7, 9)) == _word(4, 2)
    assert _word(4, 2) == stop_rule.reference16([9, 9, 7, 9], 4, (stop_check.NOT_AN_ID,), 3, 64)
    long = _ended("long", (9, 9, 9, 9), "length", 9)
    assert _block(RULE, long) == _word(4, 0) and RULE.word(long) is None
    assert BR.Stop.of(json.loads(json.dumps(RULE.to_json()))) == RULE and BR.Stop.of(None) is None
    assert BR.Stop.of(NO_IDS.to_json()) == NO_IDS
    for bad in ((7, 3), (3, 3), (-1,), (1 << 32,)):
        with pytest.raises(ValueError, match="increasing u32 ids"):
            BR.Stop(bad, 64, 4)
    with pytest.raises(ValueError, match="a power of two"):
        BR.Stop((3,), 64, 5)
    with pytest.raises(ValueError, match="which the stop word's n holds"):
        BR.Stop((3,), 1 << 20, 1 << stop_rule.N_BITS)
    for bad in ({"ids": [3]}, {**RULE.to_json(), "ignore_eos": True}):
        with pytest.raises(ValueError, match="a stop rule is"):
            BR.Stop.of(bad)


def test_the_stop_block_reads_ignore_eos_off_the_request_row_and_takes_no_stop_id_for_that_request():
    """A request that sets ``ignore_eos`` has no stop id, and the stop block reads that off its `b.request` row: the
    same served ids end at a stop id without it and run to ``max_tokens`` with it, under one registered rule, so a
    class holds requests of both kinds and nothing is refused as a serve starts. Any word but 0 there sets it. The
    block's output on such a row is the rule's over no stop id, and `Stop.word` is that output."""
    heeds, ignores = _ended("h", (9, 7, 9, 9), "stop"), _ended("i", (9, 7, 9, 9), "length", ignore_eos=True)
    assert _block(RULE, heeds, (9, 7, 9, 9)) == _word(2, 1) and _block(RULE, ignores) == _word(4, 2)
    assert RULE.word(ignores) == _word(4, 2) == stop_rule.reference16([9, 7, 9, 9], 4, (stop_check.NOT_AN_ID,), 3, 64)
    held = _ended("h", (9, 7), "stop")
    assert RULE.word(heeds) is None and RULE.word(held) == _word(2, 1)
    assert RULE.word(_ended("i", (9, 7), "stop", ignore_eos=True)) is None
    head = _u32s(BR.ingress_row(heeds, 4))[:BR.HEAD_U32]
    at = BR.FIELD_AT["ignore_eos"]
    assert head[at] == 0 and _u32s(BR.ingress_row(ignores, 4))[at] == 1
    for word, want in ((0, _word(2, 1)), (1, _word(4, 2)), (2, _word(4, 2)), (1 << 31, _word(4, 2)),
                       (stop_check.NOT_AN_ID, _word(4, 2))):
        assert evaluate(RULE.template, [*head[:at], word, *head[at + 1:]], [9, 7, 9, 9]) == (want,)
    for served, mt, lp in (((3, 9, 9, 9), 4, 3), ((9, 9, 9, 7), 3, 1), ((7, 9, 9, 9), 4, 62), ((9, 9, 9, 3), 9, 5)):
        s = _ended("x", served, "length", mt, tuple(range(lp)), ignore_eos=True)
        assert _block(RULE, s) == _block(NO_IDS, s) == stop_rule.reference16(list(served), mt, (stop_check.NOT_AN_ID,), lp, 64)
    rows = BR.port_rows([held, ignores], BR.Layout(PAIR, 0, 1, 4, RULE.outputs, 2, "a" * 128, stop=RULE))
    assert rows[BR.STOP][:, 0].tolist() == [_word(2, 1), _word(4, 2)]
    assert evaluate(population(RULE.template, 2), _flat(rows[BR.REQUEST]), _flat(rows[BR.EGRESS])) == \
        (_word(2, 1), _word(4, 2))


def test_a_stop_class_takes_a_span_only_as_the_rule_ends_it():
    """`Stop.word` is None, and the span in no stop class, for an abort, an error, a request still running, an end that
    isn't the rule's and a ``min_tokens`` other than 0, which the rule doesn't read."""
    assert [RULE.word(_ended("f", (9, 9, 7), f)) for f in ("abort", "error", None, "length")] == [None] * 4
    assert RULE.word(_ended("m", (9, 9, 7), "stop", min_tokens=1)) is None
    assert RULE.word(_ended("m", (9, 9, 7), "stop", min_tokens=0)) == _word(3, 1)


def test_a_class_under_a_stop_rule_commits_its_rows_in_parts_and_a_stop_class_its_stop_word(tmp_path):
    """Under a stop rule a class's boundary values are `b.ingress`'s row as `b.request` and `b.prompt`, and `b.egress`
    with no n and no finish; a stop class also commits `b.stop`, each span's word, which is the stop block's output on
    its rows; no word twice. Its boundary record holds the three, names the rule and that it is a stop class,
    registers by both verifiers against the challenger's copies from public items alone, is refused under the same
    class without a rule, and reopens from its files. A class of the rule's serve without a stop block commits the
    three alone."""
    L = _stop_layout()
    c = BR.commit(HELD[:3], L, 0)
    rows = BR.port_rows(HELD[:3], L)
    assert list(rows) == [BR.REQUEST, BR.PROMPT, BR.EGRESS, BR.STOP] == list(L.committed) and L.values == BR.PARTS
    for i, s in enumerate(HELD[:3]):
        assert np.concatenate([rows[BR.REQUEST][i], rows[BR.PROMPT][i]]).tolist() == BR.ingress_row(s, 4).tolist()
        assert _u32s(rows[BR.EGRESS][i]) == [*s.served, *[0] * (4 - len(s.served))] == _u32s(BR.egress_row(s, 4))[2:]
        assert rows[BR.STOP][i].tolist() == [RULE.word(s)] == [_block(RULE, s)]
    pub = json.loads(json.dumps(c.public()))
    assert pub["stop"] == RULE.to_json() and pub["stopped"] is True and (pub["prompt"], pub["outputs"]) == (4, 4)
    assert {k: v["words"] for k, v in pub["values"].items()} == {BR.REQUEST: 2 * BR.HEAD_U32, BR.PROMPT: 8, BR.EGRESS: 8,
                                                                  BR.STOP: 1}
    mine = BR.layout_of(pub, PAIR)
    rec = BR.record(pub, PAIR, LAW)
    assert mine == L and rec == c.record(LAW) and [r["port"] for r in rec["roots"]] == list(BR.PARTS)
    assert (rec["window"]["stop"], rec["window"]["stopped"]) == (RULE.to_json(), True)
    assert R.check(rec, mine.program, mine.grain, LAW, CS.SCHEME, mine.ports()).codes == ()
    assert V.check(rec, mine.program, mine.grain, LAW, CS.SCHEME, mine.ports()).codes == ()
    plain = _layout(n=3)
    assert R.check(rec, plain.program, plain.grain, LAW, CS.SCHEME, plain.ports()).codes != ()
    assert not {"stop", "stopped"} & set(BR.commit(_spans(3), plain, 0).public())
    again = BR.reopen(BR.write(c, tmp_path / "w0" / "p4-o4"), PAIR)
    assert again.roots == c.roots and again.layout == L and again.spans == c.spans
    with pytest.raises(ValueError, match="not 3 of class"):
        BR.commit([*HELD[:2], HELD[3]], L, 0)
    with pytest.raises(ValueError, match="filed at its rule's cap"):
        BR.Layout(PAIR, 0, 1, 4, 8, 3, "a" * 128, stop=RULE)
    for doc, why in (({k: v for k, v in pub.items() if k != "stopped"}, "exactly under a stop rule"),
                     (pub | {"stopped": False}, "are not its layout's"),
                     (pub | {"values": {k: v for k, v in pub["values"].items() if k != BR.STOP}}, "are not its layout's")):
        with pytest.raises(ValueError, match=why):
            BR.layout_of(doc, PAIR)
    unheld = [_ended("abort", (9, 9), "abort"), _ended("running", (9,), None)]
    U = BR.Layout(PAIR, 0, 1, 4, 2, 2, "a" * 128, stop=RULE, held=False)
    u = BR.commit(unheld, U, 0)
    upub = json.loads(json.dumps(u.public()))
    assert list(upub["values"]) == list(BR.PARTS) == list(U.committed) and upub["stopped"] is False
    assert BR.layout_of(upub, PAIR) == U and U.stop_ports() == {}
    assert _flat(BR.port_rows(unheld, U)[BR.EGRESS]) == [9, 9, 9, 0]
    assert BR.reopen(BR.write(u, tmp_path / "w0" / "p4-o2-nostop"), PAIR).roots == u.roots
    with pytest.raises(ValueError, match="only a stop class has stop blocks"):
        BR.stop_record(upub, PAIR, STOP_LAW, u.record(LAW))


def test_the_stop_blocks_register_as_a_population_of_their_own_over_the_class_boundary_record():
    """A stop class's stop record: `population_program(stop block, N)`, one block a span; its registered values the
    block's two parameters, each the boundary record's port of that value taken as an output value, under that port's
    root and domain; its one port `b.stop`, under the window domain of the stop population's own program; its window
    the boundary record's, named the stop population's. With the boundary record in its log it is accepted by one-stage's
    R1-R7 and by the Lean verifier, against the challenger's copies from public items alone. Refused: without the
    boundary record, an input under another root than that record's, `b.stop` under one of that record's domains, and a
    second stop record of the window. The population's outputs on the committed rows are the committed words."""
    L = _stop_layout()
    c = BR.commit(HELD[:3], L, 0)
    pub = json.loads(json.dumps(c.public()))
    mine = BR.layout_of(pub, PAIR)
    rec = BR.record(pub, PAIR, LAW)
    reg, srec = BR.stop_registration(pub, PAIR, rec), BR.stop_record(pub, PAIR, STOP_LAW, rec)
    assert P.units(mine.stop_program, mine.stop_query) == 3 and srec["previous"] == R.digest(rec)
    assert srec["roots"] == [{"port": BR.STOP, "root": pub["values"][BR.STOP]["root"], **mine.stop_ports()[BR.STOP]}]
    part = mine.stop_partition
    assert mine.stop_ports()[BR.STOP] == {"leaves": 3, "domain": R.window_domain(part["program"], P.digest(part), BR.STOP,
                                                                                BR.SCHEMA, 3, mine.run)}
    by_port = {r["port"]: r for r in rec["roots"]}
    assert ({e["value"]: (e["root"], e["domain"], e["leaves"], e["words"], e["word_bits"]) for e in srec["registered"]}
            == {v: (by_port[p]["root"], by_port[p]["domain"], 3, w, 32)
                for (v, p), w in zip(BR.STOP_READS.items(), (BR.HEAD_U32, 4))})
    assert reg.outputs == {v: {"record": R.digest(rec), "port": p} for v, p in BR.STOP_READS.items()}
    assert srec["window"] == {**rec["window"], "population": "stop"} and srec["registered"] == reg.registered

    def codes(record: dict, log: list[dict], verifier=R) -> tuple[str, ...]:
        return verifier.check(record, mine.stop_program, mine.stop_query, STOP_LAW, CS.SCHEME, mine.stop_ports(), log=log,
                              registered=reg.values, outputs=reg.outputs, schemas=reg.schemas).codes

    assert codes(srec, [rec]) == () == codes(srec, [rec], V)
    assert "R7-output" in codes(srec, [])
    forged = {**srec, "registered": [{**srec["registered"][0], "root": "1" * 128}, *srec["registered"][1:]]}
    assert codes(forged, [rec]) == ("R7-held",)
    moved = {**srec, "roots": [{**srec["roots"][0], "domain": by_port[BR.EGRESS]["domain"]}]}
    assert codes(moved, [rec]) == ("R3-domain",)
    assert "R6-window" in codes(srec, [rec, srec])
    rows = BR.port_rows(HELD[:3], L)
    outputs = evaluate(population(RULE.template, 3), _flat(rows[BR.REQUEST]), _flat(rows[BR.EGRESS]))
    assert list(outputs) == rows[BR.STOP].ravel().tolist() == [RULE.word(s) for s in HELD[:3]]
    with pytest.raises(ValueError, match="not this commitment's values"):
        BR.stop_registration(pub, PAIR, BR.commit(HELD[:3], L, 0).record(LAW))


def test_a_response_one_id_shorter_or_longer_than_the_stop_rules_is_refused():
    """For each way the rule ends a response (a stop id, ``max_tokens``, the model's length): the stop block's output on
    the span's committed rows is its `b.stop` word, whose n is the ids it was served. With its `b.egress` row one id
    shorter or one id longer, the block's n is not that row's count, so the word such a response needs, whatever finish
    it states, is not the block's output on its rows, and a stop block sampled on it fails. The serve commits neither in
    a stop class (`commit` refuses it, `filed` files it without a stop block), and a response past the cap has no row in
    one at all. In a class of three, a `b.egress` row forged one id shorter changes the stop population's output at that
    span alone, to a word that is not the forged count's."""
    n_of = (1 << stop_rule.N_BITS) - 1
    for s in HELD:
        P_ = BR.cap(len(s.prompt))
        L = _stop_layout(1, prompt=P_)
        word = RULE.word(s)
        assert BR.port_rows([s], L)[BR.STOP].tolist() == [[word]] and _block(RULE, s) == word
        assert word & n_of == len(s.served)
        for ids in (s.served[:-1], (*s.served, 9)):
            cheat = BR.Span(s.engine_id, s.prompt, s.sampling, ids, s.finish)
            assert RULE.word(cheat) is None
            assert list(BR.filed([cheat], RULE)) == [(P_, BR.cap(len(ids)), False)]
            if len(ids) > RULE.outputs:
                continue
            assert _block(RULE, s, ids) & n_of != len(ids)
            assert all(_block(RULE, s, ids) != _word(len(ids), finish) for finish in range(8))
            with pytest.raises(ValueError, match="doesn't end this span as it was served"):
                BR.commit([cheat], L, 0)
        assert list(BR.filed([s], RULE)) == [(P_, RULE.outputs, True)]
    rows = BR.port_rows(HELD[:3], _stop_layout())
    forged = rows[BR.EGRESS].copy()
    assert _u32s(forged[0]) == [9, 9, 7, 0]
    forged[0, 4:6] = 0
    outputs = evaluate(population(RULE.template, 3), _flat(rows[BR.REQUEST]), _flat(forged))
    assert outputs[0] & n_of != 2 and list(outputs[1:]) == rows[BR.STOP].ravel().tolist()[1:]


def test_the_hook_files_the_spans_the_stop_rule_ends_in_stop_classes_and_the_rest_without_a_stop_block(tmp_path):
    """A serve with a stop rule: a response the rule ends (a stop id, ``max_tokens``, and ``max_tokens`` past a stop id
    for a request that set ``ignore_eos``) is committed in the stop class of its prompt cap, at the rule's cap whatever
    its count, with its stop word; one it doesn't (an end at an id that is no stop id of the rule's, an abort) by its
    size class under ``-nostop``, without one, beside a stop class of the same sizes; every class names the rule and
    whether it is a stop class, and reopens to its sink line's roots; and a serve's configuration names the rule."""
    lines: list[dict] = []
    sched, hooks = Scheduler(), Hooks(tag="test-boundary-rows-stop")
    bw = BR.BoundaryWindows(tmp_path, PAIR, 0, 1, 10.0, sink=lines.append, clock=lambda: 0.0, serve="s", stop=RULE)
    bw.install(sched, hooks)
    try:
        for rid, prompt in (("a", [1, 2, 5]), ("b", [4] * 9), ("c", [5, 6, 8, 8]), ("d", [1]), ("f", [1, 2, 5])):
            sched.add_request(Request(rid, prompt, ignore_eos=False))
        sched.add_request(Request("e", [1, 2, 5], ignore_eos=True))
        sched.finish_requests(["d"], NS(name="FINISHED_ABORTED"))
        _step(sched, {"a": ([9, 7], "stop"), "b": ([9, 9, 9, 9], "length"), "c": ([9, 9], "stop"),
                      "e": ([9, 7, 9, 9], "length"), "f": ([9, 9, 9, 9], "stop")})
        bw.stop()
        got = {(x["prompt"], x["outputs"], x["stopped"]): x for x in lines}
        assert sorted(got) == [(2, 2, False), (4, 2, False), (4, 4, False), (4, 4, True), (16, 4, True)]
        assert all("error" not in x and x["window"] == 0 and x["stop"] == RULE.to_json() for x in lines)
        for (p, o, held), x in got.items():
            c = BR.reopen(tmp_path / "w0" / f"p{p}-o{o}{'' if held else '-nostop'}", PAIR)
            assert {k: v.hex() for k, v in c.roots.items()} == {k: v["root"] for k, v in x["values"].items()}
            assert list(x["values"]) == [*BR.PARTS, *([BR.STOP] if held else [])]
        both = BR.reopen(tmp_path / "w0" / "p4-o4", PAIR)
        assert [s.engine_id for s in both.spans] == ["a", "e"]
        assert BR.port_rows(both.spans, both.layout)[BR.STOP][:, 0].tolist() == [_word(2, 1), _word(4, 2)]
        assert [s.engine_id for s in BR.reopen(tmp_path / "w0" / "p4-o4-nostop", PAIR).spans] == ["f"]
        assert BR.reopen(tmp_path / "w0" / "p4-o2-nostop", PAIR).spans[0].engine_id == "c"
    finally:
        bw.stop()
        hooks.uninstall()
    of, plain = (BR.BoundaryWindows.of({"check": "c", **cfg}, tmp_path / "of") for cfg in ({"stop": RULE.to_json()}, {}))
    of.stop()
    plain.stop()
    assert of.stop_rule == RULE and plain.stop_rule is None and of.release is None


def test_a_request_the_certifier_did_not_release_is_in_no_stop_class(tmp_path):
    """A serve that knows its requests' releases (``release``): each span keeps its own, (run, N), in the prover's
    index, and a request without one is filed without a stop block, however it ended. A serve that knows none files by
    the rule alone, and its spans' index entries name no release."""
    lines: list[dict] = []
    sched, hooks = Scheduler(), Hooks(tag="test-boundary-rows-release")
    bw = BR.BoundaryWindows(tmp_path, PAIR, 0, 1, 10.0, sink=lines.append, clock=lambda: 0.0, serve="s", stop=RULE,
                            release={"a": ("run", 0), "c": ("run", 1)}.get)
    bw.install(sched, hooks)
    try:
        for rid in "abc":
            sched.add_request(Request(rid, [1, 2, 5], ignore_eos=False))
        _step(sched, {rid: ([9, 7], "stop") for rid in "abc"})
        bw.stop()
        assert sorted((x["prompt"], x["outputs"], x["stopped"], x["spans"]) for x in lines) == [(4, 2, False, 1),
                                                                                               (4, 4, True, 2)]
        held = BR.reopen(tmp_path / "w0" / "p4-o4", PAIR).spans
        assert [(s.engine_id, s.release) for s in held] == [("a", ("run", 0)), ("c", ("run", 1))]
        assert [(s.engine_id, s.release) for s in BR.reopen(tmp_path / "w0" / "p4-o2-nostop", PAIR).spans] == [("b", None)]
    finally:
        bw.stop()
        hooks.uninstall()
    span = _ended("x", (9, 9, 7), "stop")
    assert list(BR.filed([span], RULE)) == [(4, 4, True)] and list(BR.filed([span], RULE, released=True)) == [(4, 4, False)]
    released = BR.Span("x", span.prompt, span.sampling, span.served, "stop", ("run", 7))
    assert list(BR.filed([released], RULE, released=True)) == [(4, 4, True)]
    assert BR.Span.from_json(json.loads(json.dumps(released.to_json()))) == released and "release" not in span.to_json()
