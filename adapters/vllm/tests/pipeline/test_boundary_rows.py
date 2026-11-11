"""A served window's boundary commitment (`pipeline.boundary_rows`): each span's `b.ingress` and `b.egress` rows word for
word, the size classes, one class's commitment under core's window domains registering by both verifiers, its files
recommitting to its roots, and the serving path's hook on a stand-in for vLLM's V1 scheduler committing each window's
finished requests by class as it closes."""

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
    assert {k: [s.engine_id for s in v] for k, v in BR.classes(spans).items()} == {(4, 4): ["a", "c"], (16, 2): ["b"]}
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
    def __init__(self, rid: str, prompt: list[int], max_tokens: int = 4):
        self.request_id, self.prompt_token_ids = rid, prompt
        self.sampling_params = NS(**{**GREEDY, "max_tokens": max_tokens})


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
