"""A two-stage window's select inputs (`pipeline.boundary_logits`, `boundary_rows.Layout.logits`): the class's layout
lists them after its return at the rows round 2's links read; a span's rows are its tokens' logits as the select's units
read them, its last token's again past its ids; rank 0's sampler hook on a stand-in for Model Runner V2 keeps each
emitting request's row; the builder commits and keeps a class's rows, refused for a span whose rows don't match its ids
or after a call the hook couldn't read; the channel between the two processes; and the window's boundary commitment
(`boundary_windows.BoundaryWindows`) waiting for each class's root and listing it after its ids, registered against the
challenger's copy of the layout, with or without a stop rule, round 2 opening the kept rows at that root."""

from __future__ import annotations

import json
import stat
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from verity.protocols.spot.verifier import second_stage as SS
from verity.protocols.spot.verifier.one_stage import partition as P, registration as R
from verity.circuits.codec import descriptor_id
from verity.commitments.windows import rows as CS
from verity_vllm.engine.hooks import Hooks
from verity_vllm.pipeline import boundary_logits as BL, boundary_rows as BR, boundary_windows as BW
from verity_vllm.pipeline.boundary_stop import Stop

from tests.pipeline.test_boundary_tables import DESIGN

V, N = 160, 8
K = V // N
CHECK = f"GreedySelect_v1{{V={V},N={N}}}"
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": 4, "min_tokens": 0,
          "ignore_eos": False, "seed": None, "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
RULE = Stop((7,), 64, 4)


@pytest.fixture(scope="module")
def check():
    return BR.check_template(CHECK)


def logit(rid: str, j: int) -> np.ndarray:
    """The stand-in's row of logits for request ``rid``'s id j: finite bf16 bit patterns that differ by request and id."""
    seed = (sum(map(ord, rid)) * 131 + j * 17) % 200
    return (0x3F00 + (seed + np.arange(V)) % 128).astype(np.uint16)


def test_a_class_lists_its_select_inputs_after_its_return_at_the_rows_round_2s_links_read(check):
    """Each of the select's K units a token reads its lanes of the token's logits, unit k of token j at row j·K + k, as
    round 2 derives the links (`second_stage.replay_rows`); the value is listed after the return, ``logits`` or, beside
    tables, under the replay block's id, at O·K rows a span of N words, under the row values' window domain."""
    L = BR.Layout(check, 0, 1, 8, 4, 3, "ab" * 64, True, ids=True, logits=True)
    assert L.logit_rows == (4 * K, N) and L.logits_port == BR.LOGITS
    assert list(L.ports()) == [*BR.PORTS, BR.RETURN, BR.LOGITS] and L.rank0 == (BR.LOGITS,)
    assert (L.leaves(BR.LOGITS), L.words(BR.LOGITS), L.schemas()[BR.LOGITS]) == (3 * 4 * K, N, BR.SCHEMA)
    assert L.domain(BR.LOGITS) == BR.window_domain(L.partition["program"], P.digest(L.partition), BR.LOGITS, 3 * 4 * K,
                                                   L.run)
    rows = {(r.template, r.port): r for r in SS.replay_rows(L.program, descriptor_id(L.replay))}
    reads = [rows[(P.template_id(t), "logits")] for t in L.templates]
    assert [(r.per, [SS.row_at(r.runs, j) for j in range(4)]) for r in reads] == [
        (4 * K, [j * K + k for j in range(4)]) for k in range(K)]
    T = BR.Layout(check, 0, 1, 8, 4, 3, "ab" * 64, True, (0,), 16, ids=True, logits=True)
    block = descriptor_id(T.replay)
    assert list(T.ports())[2:4] == [f"{block}/{BR.RETURN}", f"{block}/{BR.LOGITS}"] and T.rank0[1:] == T.tables
    assert L.provenance()["logits"] is True and "logits" not in BR.Layout(check, 0, 1, 8, 4, 3, "ab" * 64, True,
                                                                         ids=True).provenance()


def test_a_class_commits_select_inputs_only_beside_its_ids_of_the_greedy_select_inlined(check):
    with pytest.raises(ValueError, match="only beside those ids"):
        BR.Layout(check, 0, 1, 8, 4, 3, "ab" * 64, True, logits=True)
    for c, inline in ((BR.check_template(f"TokenSelect_v4{{V={V}}}"), True), (check, False)):
        with pytest.raises(ValueError, match="GreedySelect_v1 inlined"):
            _ = BR.Layout(c, 0, 1, 8, 4, 3, "ab" * 64, inline, ids=True, logits=True).logit_rows


def test_a_spans_rows_are_its_tokens_logits_as_its_units_read_them_and_its_last_tokens_again_past_its_ids():
    x = np.stack([logit("a", j) for j in range(3)])
    got = BL.span_rows(x, 4, N)
    assert got.shape == (4 * K, N)
    assert all(np.array_equal(got[j * K + k], x[min(j, 2), k * N:(k + 1) * N]) for j in range(4) for k in range(K))
    assert not BL.span_rows(np.zeros((0, V), np.uint16), 4, N).any()
    with pytest.raises(ValueError):
        BL.span_rows(np.stack([logit("a", j) for j in range(5)]), 4, N)


class Sampler:
    """vLLM's sampler as the runner calls it, ``(logits, input_batch)``: it samples each row's first maximum."""

    def __call__(self, logits, input_batch):
        return NS(sampled_token_ids=logits.float().argmax(dim=1, keepdim=True))


def batch(**reqs: tuple[int, int, int]) -> NS:
    """A batch of Model Runner V2's input: each request's (computed tokens, scheduled tokens, prompt length)."""
    ids = list(reqs)
    return NS(req_ids=ids, num_reqs=len(ids), num_draft_tokens=0,
              num_computed_tokens_np=np.array([reqs[r][0] for r in ids]),
              num_scheduled_tokens=np.array([reqs[r][1] for r in ids]),
              prefill_len_np=np.array([reqs[r][2] for r in ids]))


def call(hook: BL.SamplerHook, b: NS, rows: dict[str, np.ndarray]) -> None:
    hook(torch.from_numpy(np.stack([rows[r] for r in b.req_ids]).view(np.int16)).view(torch.bfloat16), b)


def test_the_sampler_hook_keeps_each_emitting_requests_row_in_order_and_nothing_of_a_prefill_chunk():
    """Request a's prompt ends in the first call and its two decodes follow; b's first prefill chunk stops short of its
    prompt's end, so its row there is no id's, and its second reaches it. Each kept row is the logits the sampler took,
    bit for bit."""
    hook = BL.SamplerHook(Sampler(), capturing=lambda: False)
    call(hook, batch(a=(0, 5, 5), b=(0, 4, 6)), {"a": logit("a", 0), "b": logit("b", 9)})
    call(hook, batch(a=(5, 1, 5), b=(4, 2, 6)), {"a": logit("a", 1), "b": logit("b", 0)})
    call(hook, batch(a=(6, 1, 5)), {"a": logit("a", 2)})
    a, b = hook.logits.take("a"), hook.logits.take("b")
    assert np.array_equal(a, np.stack([logit("a", j) for j in range(3)])) and np.array_equal(b, logit("b", 0)[None])
    assert hook.logits.requests() == [] and hook.logits.failed is None


def test_a_call_the_hook_cant_read_fails_the_logits_and_every_later_build_says_why():
    hook = BL.SamplerHook(Sampler(), capturing=lambda: False)
    b = batch(a=(0, 5, 5))
    b.num_draft_tokens = 1
    call(hook, b, {"a": logit("a", 0)})
    assert "draft tokens" in hook.logits.failed
    got = BL.Builder(hook.logits).build({"window": 0, "run": "r" * 128, "prompt": 8, "outputs": 4, "stopped": None,
                                         "program": "cd" * 64, "partition": "ef" * 64, "value": BR.LOGITS,
                                         "rows": 4 * K, "words": N, "spans": [["a", 1, "stop"]]})
    assert got["entry"] is None and "incomplete" in got["error"]


def test_no_row_is_kept_while_a_cuda_graph_is_captured():
    hook = BL.SamplerHook(Sampler(), capturing=lambda: True)
    call(hook, batch(a=(0, 5, 5)), {"a": logit("a", 0)})
    assert hook.logits.requests() == [] and hook.logits.failed is None


def doc(spans, **kw):
    return {"window": 0, "run": "r" * 128, "prompt": 8, "outputs": 4, "stopped": None, "program": "cd" * 64,
            "partition": "ef" * 64, "value": BR.LOGITS, "rows": 4 * K, "words": N, "spans": spans, **kw}


@pytest.mark.parametrize("held, served, finish, why", [(3, 2, "stop", "3 rows"), (1, 2, "length", "1 rows"),
                                                       (5, 5, "stop", "not whole rows")])
def test_the_builder_refuses_a_span_whose_rows_dont_match_its_ids(held, served, finish, why):
    logits = BL.Logits()
    for j in range(held):
        logits.add("a", logit("a", j))
    got = BL.Builder(logits).build(doc([["a", served, finish]]))
    assert got["entry"] is None and why in got["error"] and logits.requests() == []


def test_an_aborted_span_may_hold_a_row_past_its_ids_served():
    logits = BL.Logits()
    for j in range(3):
        logits.add("a", logit("a", j))
    assert BL.Builder(logits).build(doc([["a", 2, "abort"]]))["error"] is None


def test_the_channel_carries_an_assignment_to_rank_0_and_its_answer_back_and_tells_a_stop_class_from_its_sibling(tmp_path):
    logits = BL.Logits()
    for rid in ("a", "c"):
        logits.add(rid, logit(rid, 0))
    ch = BL.Channel(tmp_path / "ch", timeout_s=1.0, sleep=lambda _s: None)
    held, sibling = doc([["a", 1, "stop"]], stopped=True), doc([["c", 1, "abort"]], stopped=False)
    for d in (held, sibling):
        ch.assign(d)
    assert ch.serve(BL.Builder(logits)) == 2
    assert len({ch.answer(d)["entry"]["root"] for d in (held, sibling)}) == 2
    gone = BL.Channel(tmp_path / "gone", timeout_s=0.0, sleep=lambda _s: None)
    with pytest.raises(TimeoutError):
        gone.answer(held)


class Scheduler:
    """vLLM's V1 scheduler as `BoundaryWindows` wraps it: the step's outputs are ``outputs``."""

    def __init__(self, max_model_len: int = 64) -> None:
        self.cache_config, self.outputs, self.max_model_len = NS(enable_prefix_caching=False), {}, max_model_len

    def add_request(self, request):
        return None

    def schedule(self):
        return NS()

    def update_from_output(self, so, mro):
        out, self.outputs = self.outputs, {}
        return out

    def finish_requests(self, request_ids, finished_status):
        return []


def serve(out, check, logits: BL.Logits, keep: BL.Kept, lines: list, stop: Stop | None = None,
          served: dict | None = None) -> tuple[BW.BoundaryWindows, BL.Logits]:
    """A window of ``served`` (request: (prompt, ids served, finish)), each id's logits kept on rank 0 as the sampler
    hook keeps them, closed and committed."""
    sched, hooks = Scheduler(), Hooks(tag="test-boundary-logits")
    bw = BW.BoundaryWindows(out, check, 0, 1, 10.0, sink=lines.append, clock=lambda: 0.0, serve="s", inline=True,
                            ids=True, stop=stop, logits=BL.LocalChannel(BL.Builder(logits, keep=keep))).install(sched, hooks)
    try:
        for rid, (prompt, ids, _f) in served.items():
            sched.add_request(NS(request_id=rid, prompt_token_ids=prompt, sampling_params=NS(**GREEDY)))
            for j in range(len(ids)):
                logits.add(rid, logit(rid, j))
        sched.outputs = {0: NS(outputs=[NS(request_id=rid, new_token_ids=ids, finish_reason=NS(name=f))
                                        for rid, (_p, ids, f) in served.items()])}
        sched.schedule()
        sched.update_from_output(None, None)
        bw.stop()
    finally:
        hooks.uninstall()
    return bw, logits


def modes(root) -> set[tuple[str, int]]:
    return {(p.name, stat.S_IMODE(p.stat().st_mode)) for p in [root, *root.rglob("*")]}


def test_each_classs_commitment_waits_for_its_select_inputs_and_round_2_opens_them_at_its_root(tmp_path, check):
    """A serve of ids with no stop rule: requests a (5 prompt ids, 4 served, stopped) and b (3 prompt ids, 3 served, at
    its max_tokens) close one window in classes (8, 4) and (4, 4). Each class lists ``logits`` after ``o0``, at O·K rows of N
    words, its root rank 0's; the challenger's copy of the layout from ``commit.json`` alone registers it; its files
    reopen to the same roots; and rank 0's kept rows, readable by their owner only, recommit to that root and are each
    token's logits as its units read them, b's last token's again past its 3 ids."""
    lines, keep = [], BL.Kept(tmp_path / "private" / "logits")
    served = {"a": ([1, 2, 3, 4, 5], [9, 9, 9, 9], "STOP"), "b": ([6, 7, 8], [9, 9, 9], "LENGTH")}
    _bw, logits = serve(tmp_path / "out", check, BL.Logits(), keep, lines, served=served)
    assert all("error" not in x for x in lines) and logits.requests() == []
    run = BR.run_root("s", 0)
    for rid, (p, o) in (("a", (8, 4)), ("b", (4, 4))):
        d = tmp_path / "out" / "w0" / f"p{p}-o{o}"
        pub = json.loads((d / BR.COMMIT).read_text())
        mine = BR.layout_of(pub, check)
        assert pub["logits"] is True and list(pub["values"]) == [*BR.PORTS, BR.RETURN, BR.LOGITS] == list(mine.ports())
        assert (pub["values"][BR.LOGITS]["leaves"], pub["values"][BR.LOGITS]["words"]) == (o * K, N)
        rec = BR.record(pub, check, DESIGN)
        assert [r["port"] for r in rec["roots"]] == [*BR.PORTS, BR.RETURN, BR.LOGITS]
        assert R.check(rec, mine.program, mine.grain, DESIGN, CS.SCHEME, mine.ports()).codes == ()
        assert BR.reopen(d, check).roots == {k: bytes.fromhex(v["root"]) for k, v in pub["values"].items()}
        port = BL.load(tmp_path / "private", 0, run, p, o, None)
        assert port.root.hex() == pub["values"][BR.LOGITS]["root"] and port.domain == mine.domain(BR.LOGITS)
        n = len(served[rid][1])
        assert np.array_equal(port.rows, BL.span_rows(np.stack([logit(rid, j) for j in range(n)]), o, N))
    assert {m for name, m in modes(tmp_path / "private" / "logits") if "." in name} == {0o600}
    assert {m for name, m in modes(tmp_path / "private" / "logits") if "." not in name} == {0o700}


def test_under_a_stop_rule_the_stop_class_and_its_sibling_each_commit_and_keep_their_own(tmp_path, check):
    """Under a stop rule with T = 4: a ends on a stop id after 3 ids, so it is in the stop class p8-o4; c is aborted
    after 3 ids, so the rule doesn't end it as served and it is in that class's sibling p8-o4-nostop. Each lists
    ``logits`` after ``o0`` (and the stop class `b.stop` after them), each registers against its layout under the
    verifier's copy of the rule, and rank 0 keeps each class's rows apart, a's last token's again at its fourth."""
    lines, keep = [], BL.Kept(tmp_path / "private" / "logits")
    served = {"a": ([1, 2, 3, 4, 5], [9, 9, 7], "STOP"), "c": ([1, 2, 3, 4, 5, 6], [9, 9, 9], "ABORTED")}
    serve(tmp_path / "out", check, BL.Logits(), keep, lines, stop=RULE, served=served)
    assert all("error" not in x for x in lines), lines
    run = BR.run_root("s", 0)
    for rid, sub, held in (("a", "p8-o4", True), ("c", "p8-o4-nostop", False)):
        pub = json.loads((tmp_path / "out" / "w0" / sub / BR.COMMIT).read_text())
        mine = BR.layout_of(pub, check, RULE)
        assert list(pub["values"]) == [*BR.PARTS, BR.RETURN, BR.LOGITS, *([BR.STOP] if held else [])]
        assert R.check(BR.record(pub, check, DESIGN, stop=RULE), mine.program, mine.grain, DESIGN, CS.SCHEME,
                       mine.ports()).codes == ()
        port = BL.load(tmp_path / "private", 0, run, 8, 4, held)
        assert port.root.hex() == pub["values"][BR.LOGITS]["root"]
        assert np.array_equal(port.rows, BL.span_rows(np.stack([logit(rid, j) for j in range(3)]), 4, N))
    assert sorted(p.name for p in (tmp_path / "private" / "logits").iterdir()) == [
        BL.stem(0, run, 8, 4, True), BL.stem(0, run, 8, 4, False)]


def test_a_class_registers_nothing_when_rank_0_committed_no_logits(tmp_path, check):
    lines = []
    logits = BL.Logits()
    logits.fail("step 3: TypeError: draft tokens")
    sched, hooks = Scheduler(), Hooks(tag="test-boundary-logits")
    bw = BW.BoundaryWindows(tmp_path, check, 0, 1, 10.0, sink=lines.append, clock=lambda: 0.0, serve="s", inline=True,
                            ids=True, logits=BL.LocalChannel(BL.Builder(logits))).install(sched, hooks)
    try:
        sched.add_request(NS(request_id="a", prompt_token_ids=[1, 2, 3], sampling_params=NS(**GREEDY)))
        sched.outputs = {0: NS(outputs=[NS(request_id="a", new_token_ids=[9], finish_reason=NS(name="STOP"))])}
        sched.update_from_output(None, None)
        bw.stop()
    finally:
        hooks.uninstall()
    assert len(lines) == 1 and "rank 0 committed no logits" in lines[0]["error"]
    assert not (tmp_path / "w0" / "p4-o2" / BR.COMMIT).exists()


@pytest.mark.parametrize("cfg", [{"ids": False}, {"inline": False}, {"check": f"TokenSelect_v4{{V={V}}}"},
                                 {"per_prompt": 1}])
def test_a_serve_of_select_inputs_is_refused_as_it_starts_without_ids_of_the_greedy_select_inlined(tmp_path, cfg):
    base = {"check": CHECK, "inline": True, "per_prompt": 0, "per_output": 1, "ids": True,
            "logits": {"private": str(tmp_path / "private")}}
    with pytest.raises(ValueError, match="GreedySelect_v1 inlined"):
        BW.BoundaryWindows.of({**base, **cfg}, tmp_path / "out")
