"""A two-stage window's history tables (`pipeline.boundary_tables`): their names from public items alone; rank 0's step
hook on a stand-in for vLLM's Model Runner V2 reading each MLA layer's rows at the step's slots, by request and position,
across chunked prefill, decode, padding and a recompute after a preemption; the builder's tables over them under the
class's names and domains, refused for a request whose rows don't match its lengths or after a step the hook couldn't
read; a cache that doesn't hold bf16 latent rows refused; the channel between the two processes; and the window's boundary
commitment (`boundary_windows.BoundaryWindows`) waiting for each class's tables' roots and listing them beside its values,
registered by both verifiers from public items alone; every table of a class filled to its C rows, so that no published
string shows a request's length; and what round 2 opens, kept on the node, a served table loaded by its name."""

from __future__ import annotations

import json
import stat
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from verity.protocols.spot.verifier.one_stage import registration as R, verifier as V
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.types import Array, Value as Bits
from verity.commitments import history as HT
from verity.commitments.windows import rows as CS
from verity_vllm.commit import history_tables as H
from verity_vllm.engine.hooks import Hooks
from verity_vllm.pipeline import boundary_rows as BR, boundary_tables as BT, boundary_windows as BW
from verity_vllm.records.boundary_tables import ROOTS

RUN = "a" * 128
PROGRAM = "cd" * 64
CAP = 16
BLOCK = 4


def test_a_tables_name_binds_the_run_the_class_the_slot_and_the_layer_and_nothing_else():
    base = BT.name(RUN, 4, 8, 2, 7)
    assert base == H.name(f"{RUN}/p4-o8/s2", 7) == f"history/{RUN}/p4-o8/s2/model.layers.7.self_attn.latent"
    others = {BT.name("b" * 128, 4, 8, 2, 7), BT.name(RUN, 8, 8, 2, 7), BT.name(RUN, 4, 16, 2, 7), BT.name(RUN, 4, 8, 3, 7),
              BT.name(RUN, 4, 8, 2, 8)}
    assert len(others) == 5 and base not in others


@pytest.mark.parametrize("args", [("", 4, 8, 0), (RUN, -1, 8, 0), (RUN, 4, 8, -1), (RUN, 4, 8, 1.0), (RUN, 4, True, 0)])
def test_a_key_is_refused_without_a_run_or_with_a_class_or_slot_that_isnt_a_natural_number(args):
    with pytest.raises(ValueError):
        BT.key(*args)


class Mla(torch.nn.Module):
    """A stand-in for an MLA layer the engine bound a KV cache to: [blocks, block size, 576] bf16."""

    def __init__(self, i: int, blocks: int = 16, dtype: str = "auto", words: int = H.WORDS) -> None:
        super().__init__()
        self.layer_name, self.kv_lora_rank, self.qk_rope_head_dim = f"model.layers.{i}.self_attn.attn", words - 64, 64
        self.kv_cache_dtype, self.kv_cache = dtype, torch.zeros(blocks, BLOCK, words, dtype=torch.bfloat16)


class IndexerCache(torch.nn.Module):
    """A bound cache with no MLA latent (DSA's indexer keys): not a table's."""

    def __init__(self) -> None:
        super().__init__()
        self.layer_name, self.kv_cache = "model.layers.0.self_attn.indexer.k_cache", torch.zeros(16, BLOCK, 132, dtype=torch.uint8)


class Model(torch.nn.Module):
    def __init__(self, layers: tuple[int, ...] = (0, 3)) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([Mla(i) for i in layers] + [IndexerCache()])


def row(request: str, position: int, layer: int, gen: int = 0) -> np.ndarray:
    """The stand-in's latent row: finite bf16 bit patterns that differ by request, position, layer and computation."""
    seed = (sum(map(ord, request)) * 131 + position * 17 + layer * 7 + gen * 3) % 200
    return (0x3F00 + (seed + np.arange(H.WORDS)) % 256).astype(np.uint16)


class Runner:
    """A stand-in for Model Runner V2: `execute_model` writes each scheduled position's rows into every MLA cache at its
    slot (a block a request at a time) and leaves the step's batch and slots in ``execute_model_state``, padded past the
    batch as the runner pads its buffers."""

    def __init__(self, model: Model) -> None:
        self.model, self.blocks, self.free, self.gen = model, {}, list(range(16)), {}
        self.execute_model_state = None

    def preempt(self, request: str) -> None:
        self.free += self.blocks.pop(request)
        self.gen[request] = self.gen.get(request, 0) + 1

    def execute_model(self, so, dummy_run: bool = False):
        ids, qsl, nct, slots = list(so.num_scheduled_tokens), [0], [], []
        for rid in ids:
            start, n = so.starts[rid], so.num_scheduled_tokens[rid]
            for p in range(start, start + n):
                bs = self.blocks.setdefault(rid, [])
                while len(bs) <= p // BLOCK:
                    bs.append(self.free.pop(0))
                slot = bs[p // BLOCK] * BLOCK + p % BLOCK
                for m in self.model.layers:
                    if isinstance(m, Mla):
                        i = int(m.layer_name.split(".")[2])
                        m.kv_cache.view(torch.int16).view(-1, H.WORDS)[slot] = torch.from_numpy(
                            row(rid, p, i, self.gen.get(rid, 0)).view(np.int16))
                slots.append(slot)
            qsl.append(qsl[-1] + n)
            nct.append(start)
        sm = torch.tensor(slots + [-1, -1], dtype=torch.int32)
        names = [m.layer_name for m in self.model.layers]
        self.execute_model_state = NS(
            input_batch=NS(req_ids=ids, query_start_loc_np=np.array(qsl + [qsl[-1]] * 3, dtype=np.int32),
                           num_computed_tokens_np=np.array(nct + [0, 0], dtype=np.int32)),
            slot_mappings_by_layer=dict.fromkeys(names, sm))
        return None


def step(**sched: tuple[int, int]) -> NS:
    """A scheduler output: each request's (start, tokens), in batch order."""
    return NS(num_scheduled_tokens={r: n for r, (_s, n) in sched.items()}, starts={r: s for r, (s, _n) in sched.items()},
              total_num_scheduled_tokens=sum(n for _s, n in sched.values()), scheduled_spec_decode_tokens={})


def serve() -> tuple[Runner, BT.StepHook, Hooks]:
    """Request a: a 5-id prompt in two prefill chunks, then three decodes; b joins at a's second chunk, is preempted after
    four positions and recomputed from 0."""
    model, hooks = Model(), Hooks(tag="test")
    runner = Runner(model)
    hook = BT.StepHook(model).install(runner, hooks)
    runner.execute_model(step(a=(0, 3)))
    runner.execute_model(step(b=(0, 3), a=(3, 2)))
    runner.execute_model(step(a=(5, 1), b=(3, 1)))
    runner.execute_model(step(), dummy_run=True)
    runner.preempt("b")
    runner.execute_model(step(a=(6, 1), b=(0, 4)))
    runner.execute_model(step(a=(7, 1), b=(4, 1)))
    return runner, hook, hooks


def test_the_step_hook_reads_each_mla_layers_rows_at_the_steps_slots_by_request_and_position():
    runner, hook, hooks = serve()
    assert list(hook.layers) == [0, 3] and hook.latents.failed is None
    want = {("a", 0): 8, ("b", 1): 5}
    for (rid, gen), n in want.items():
        got = hook.latents.take(rid)
        for x in (0, 3):
            assert np.array_equal(got[x], np.stack([row(rid, p, x, gen) for p in range(n)]))
    hooks.uninstall()
    runner.execute_model(step(a=(8, 1)))
    assert hook.latents.requests() == []


def built(make_tables: list | None = None):
    tables = [] if make_tables is None else make_tables

    def make(program, value, words, capacity):
        t = HT.HistoryTable(program, value, words, capacity)
        tables.append(t)
        return t
    return make, tables


def test_the_builder_builds_each_spans_tables_under_the_classs_names_in_slot_order():
    _runner, hook, _hooks = serve()
    make, tables = built()
    doc = BT.assignment(3, RUN, PROGRAM, 8, 4, [0, 3], CAP, [("b", 3, 3, "length"), ("a", 5, 4, "stop")])
    out = BT.Builder(hook.latents, make).build(doc)
    assert "error" not in out and (out["window"], out["prompt"], out["outputs"]) == (3, 8, 4)
    names = [BT.name(RUN, 8, 4, r, x) for r in (0, 1) for x in (0, 3)]
    assert [e["value"] for e in out["tables"]] == [t.value for t in tables] == names
    for e, t in zip(out["tables"], tables, strict=True):
        assert e == t.entry() and e["leaves"] == CAP and e["words"] == H.WORDS
        assert e["domain"] == HT.domain(PROGRAM, e["value"], H.WORDS, CAP).domain_id.hex()
    assert [len(t.rows) for t in tables] == [BT.class_rows(8, 4, CAP)] * 4 == [12] * 4
    assert np.array_equal(np.stack(tables[3].rows[:8]), np.stack([row("a", p, 3) for p in range(8)]))
    assert hook.latents.requests() == []


def test_two_spans_of_a_class_with_other_lengths_publish_as_many_strings_and_a_read_past_l_is_the_zero_row():
    """b holds 5 positions and a 8, in class (8, 4): each table holds C = 12 rows, its own and then zero rows, each under a
    fresh salt, so the two publish 12 strings each, no placeholder among them, and no entry differs but in its name and root.
    The root covers the fill: it is not the root over L rows with placeholders past them."""
    _runner, hook, _hooks = serve()
    make, tables = built()
    out = BT.Builder(hook.latents, make).build(BT.assignment(3, RUN, PROGRAM, 8, 4, [0, 3], CAP,
                                                            [("b", 3, 3, "length"), ("a", 5, 4, "stop")]))
    assert "error" not in out
    b3, a3 = tables[1], tables[3]
    placeholder = HT.placeholder(H.WORDS)
    for t, n in ((b3, 5), (a3, 8)):
        assert len(t.bcs) == 12 and placeholder not in t.bcs and len(set(t.salts)) == 12
        assert not np.stack(t.rows[n:]).any() and np.stack(t.rows[:n]).any()
        bare = HT.HistoryTable(PROGRAM, t.value, H.WORDS, CAP)
        bare.append(np.stack(t.rows[:n]), t.salts[:n])
        assert bare.root != t.root
        HT.check(PROGRAM, t.value, H.WORDS, t.bcs, t.root, CAP)
    drop = ("value", "root", "domain")
    assert {k: v for k, v in b3.entry().items() if k not in drop} == {k: v for k, v in a3.entry().items() if k not in drop}
    assert BT.class_rows(256, 256) == 512 and BT.class_rows(1 << 17, 16) == BT.CAPACITY == 1 << 17
    assert np.array_equal(BT.fill(np.ones((2, 3), np.uint16), 4, 4, 8), np.concatenate([np.ones((2, 3)), np.zeros((6, 3))]))
    with pytest.raises(ValueError, match="at most the 8 its class"):
        BT.fill(np.ones((9, 3), np.uint16), 4, 4)
    for args in ((0, 4), (4, 4.0)):
        with pytest.raises(ValueError, match="cap is a positive integer"):
            BT.class_rows(*args)


@pytest.mark.parametrize("b, why", [(("b", 3, 2, "length"), "not the 4 to 4"), (("b", 3, 1, "stop"), "not the 3 to 3"),
                                    (("b", 2, 1, "abort"), "not the 0 to 3")])
def test_the_builder_refuses_a_span_whose_rows_dont_match_its_lengths(b, why):
    """b holds five positions (a three-id prompt and three ids served, the last never computed)."""
    _runner, hook, _hooks = serve()
    out = BT.Builder(hook.latents, built()[0]).build(BT.assignment(0, RUN, PROGRAM, 8, 4, [0, 3], CAP, [b, ("a", 5, 4, "stop")]))
    assert f"slot 0: rank 0 holds 5 positions of it, {why}" in out["error"] and "tables" not in out
    assert hook.latents.requests() == []


def test_an_aborted_span_may_hold_any_prefix_of_its_positions():
    assert BT.computed(5, 2, "abort") == (0, 7) and BT.computed(5, 2, None) == (0, 7) and BT.computed(5, 2, "stop") == (6, 6)


def test_a_step_with_draft_tokens_fails_the_rows_and_every_later_build_says_why():
    model, hooks = Model(), Hooks(tag="test")
    runner = Runner(model)
    hook = BT.StepHook(model).install(runner, hooks)
    runner.execute_model(step(a=(0, 2)))
    runner.execute_model(NS(**{**vars(step(a=(2, 2))), "scheduled_spec_decode_tokens": {"a": [7]}}))
    out = BT.Builder(hook.latents, built()[0]).build(BT.assignment(0, RUN, PROGRAM, 2, 2, [0, 3], CAP, [("a", 2, 2, "stop")]))
    assert "step 1: ValueError: a step with draft tokens" in out["error"]


def test_the_builder_refuses_a_class_at_other_layers_than_rank_0_holds():
    _runner, hook, _hooks = serve()
    out = BT.Builder(hook.latents, built()[0]).build(BT.assignment(0, RUN, PROGRAM, 8, 4, [0], CAP, [("a", 5, 4, "stop")]))
    assert "the class's tables are at layers [0], rank 0 holds [0, 3]" in out["error"]


@pytest.mark.parametrize("layer", [Mla(0, dtype="fp8_ds_mla"), Mla(0, dtype="fp8"), Mla(0, words=512)])
def test_a_cache_that_doesnt_hold_bf16_latent_rows_is_refused(layer):
    model = Model()
    model.layers[0] = layer
    with pytest.raises(ValueError, match="doesn't hold DsaLatentRow_v1's 576 bf16 words a row"):
        BT.StepHook(model)


def test_the_step_hook_reads_only_the_layers_it_is_given_and_refuses_one_the_model_hasnt():
    assert list(BT.StepHook(Model(), layers=[3]).layers) == [3] and BT.StepHook(Model(), layers=[3]).latents.layers == (3,)
    with pytest.raises(ValueError, match=r"layers \[1\] aren't MLA layers of the model, whose are \[0, 3\]"):
        BT.StepHook(Model(), layers=[0, 1])


def test_latents_replace_a_recomputed_requests_rows_and_refuse_a_gap():
    lat = BT.Latents([1])
    lat.add("r", 0, {1: np.stack([row("r", p, 1) for p in range(4)])})
    lat.add("r", 2, {1: np.stack([row("r", p, 1, 1) for p in range(2, 5)])})
    assert lat.length("r") == 5
    with pytest.raises(ValueError, match="rows at 7, past the 5 it holds"):
        lat.add("r", 7, {1: row("r", 7, 1)[None]})
    with pytest.raises(ValueError, match="one \\(k, 576\\) block of rows at each layer of \\[1\\]"):
        lat.add("r", 5, {2: row("r", 5, 1)[None]})
    got = lat.take("r")[1]
    assert np.array_equal(got, np.stack([row("r", 0, 1), row("r", 1, 1)] + [row("r", p, 1, 1) for p in range(2, 5)]))


def test_the_channel_carries_an_assignment_to_rank_0_and_its_answer_back(tmp_path):
    _runner, hook, _hooks = serve()
    ch = BT.Channel(tmp_path / "private", sleep=lambda s: pytest.fail("the answer was there"))
    doc = BT.assignment(1, RUN, PROGRAM, 8, 4, [0, 3], CAP, [("a", 5, 4, "stop")])
    ch.assign(doc)
    assert ch.serve(BT.Builder(hook.latents, built()[0])) == 1 and ch.serve(BT.Builder(hook.latents)) == 0
    out = ch.roots(1, RUN, 8, 4)
    assert [e["value"] for e in out["tables"]] == [BT.name(RUN, 8, 4, 0, x) for x in (0, 3)]
    for f in (tmp_path / "private").rglob("*.json"):
        assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert {stat.S_IMODE(d.stat().st_mode) for d in (tmp_path / "private", tmp_path / "private" / "assign")} == {0o700}


def test_the_channel_gives_up_on_an_answer_after_its_timeout():
    now = [0.0]
    ch = BT.Channel("/nonexistent/private", timeout_s=1.0, poll_s=0.4, clock=lambda: now[0],
                    sleep=lambda s: now.__setitem__(0, now[0] + s))
    with pytest.raises(TimeoutError, match="rank 0 answered no tables for window 2's class \\(8, 4\\) in 1 s"):
        ch.roots(2, RUN, 8, 4)


def test_the_local_channel_builds_at_once():
    _runner, hook, _hooks = serve()
    ch = BT.LocalChannel(BT.Builder(hook.latents, built()[0]))
    ch.assign(BT.assignment(1, RUN, PROGRAM, 8, 4, [0, 3], CAP, [("a", 5, 4, "stop")]))
    assert len(ch.roots(1, RUN, 8, 4)["tables"]) == 2


XOR8 = PrimitiveDefinition("BoundaryTablesTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("BoundaryTablesTestPair", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": 4, "min_tokens": 0,
          "ignore_eos": True, "seed": None, "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
LAW = {"law": "subset", "k": 1}


class Scheduler:
    """vLLM's V1 scheduler as `BoundaryWindows` wraps it: the step's outputs are ``outputs``."""

    def __init__(self) -> None:
        self.cache_config, self.outputs = NS(enable_prefix_caching=False), {}

    def add_request(self, request):
        return None

    def schedule(self):
        return NS()

    def update_from_output(self, so, mro):
        out, self.outputs = self.outputs, {}
        return out

    def finish_requests(self, request_ids, finished_status):
        return []


def windows(out, latents: BT.Latents, make, lines: list, keep: BT.Kept | None = None):
    sched, hooks = Scheduler(), Hooks(tag="test-boundary-tables")
    bw = BW.BoundaryWindows(out, PAIR, 0, 1, 10.0, sink=lines.append, clock=lambda: 0.0, serve="s",
                            tables=BT.LocalChannel(BT.Builder(latents, make, keep=keep)), layers=(0, 3), capacity=CAP)
    return bw.install(sched, hooks), sched, hooks


def finish(sched: Scheduler) -> None:
    """Requests a (a 5-id prompt, 4 ids served, stopped) and b (a 3-id prompt, 3 served, at its cap), as `serve` computed
    them, finish in one step: the window closes with a in class (8, 4) and b in class (4, 4)."""
    for rid, prompt in (("a", [1, 2, 3, 4, 5]), ("b", [6, 7, 8])):
        sched.add_request(NS(request_id=rid, prompt_token_ids=prompt, sampling_params=NS(**GREEDY)))
    sched.outputs = {0: NS(outputs=[NS(request_id="a", new_token_ids=[9, 9, 9, 9], finish_reason=NS(name="STOP")),
                                    NS(request_id="b", new_token_ids=[9, 9, 9], finish_reason=NS(name="LENGTH"))])}
    sched.schedule()
    sched.update_from_output(None, None)


def test_each_classs_commitment_waits_for_its_tables_and_lists_them_beside_its_values(tmp_path):
    """The window's two classes each get their span's tables from rank 0 once it closes, at slot 0 of the class, and list
    their roots after ``b.ingress`` and ``b.egress``; the challenger's copy of each class's layout from ``commit.json``
    alone names and places every table, and both verifiers accept the registration against it, which lists every root
    before any coin. The class's files reopen to the same roots."""
    _runner, hook, _hooks = serve()
    make, tables, lines = *built(), []
    bw, sched, hooks = windows(tmp_path, hook.latents, make, lines)
    try:
        finish(sched)
        bw.stop()
    finally:
        hooks.uninstall()
    assert all("error" not in x for x in lines) and hook.latents.requests() == []
    got = {(x["prompt"], x["outputs"]): x for x in lines}
    assert sorted(got) == [(4, 4), (8, 4)]
    made = {t.value: t for t in tables}
    for (p, o), line in got.items():
        run, pub = BR.run_root("s", 0), json.loads((tmp_path / "w0" / f"p{p}-o{o}" / BR.COMMIT).read_text())
        assert pub["values"] == line["values"]
        names = [BT.name(run, p, o, 0, x) for x in (0, 3)]
        assert list(pub["values"]) == [*BR.PORTS, *names] and pub["tables"] == {"layers": [0, 3], "capacity": CAP}
        assert [pub["values"][n]["root"] for n in names] == [made[n].root.hex() for n in names]
        mine = BR.layout_of(pub, PAIR)
        assert mine.tables == tuple(names) and mine.ports()[names[1]] == {k: made[names[1]].entry()[k] for k in
                                                                           ("leaves", "domain")}
        rec = BR.record(pub, PAIR, LAW)
        assert [r["port"] for r in rec["roots"]] == [*BR.PORTS, *names]
        assert R.check(rec, mine.program, mine.grain, LAW, CS.SCHEME, mine.ports()).codes == ()
        assert V.check(rec, mine.program, mine.grain, LAW, CS.SCHEME, mine.ports()).codes == ()
        bare = BR.Layout(PAIR, 0, 1, p, o, 1, run)
        assert R.check(rec, bare.program, bare.grain, LAW, CS.SCHEME, bare.ports()).codes != ()
        assert BR.reopen(tmp_path / "w0" / f"p{p}-o{o}", PAIR).roots == {k: bytes.fromhex(v["root"])
                                                                          for k, v in pub["values"].items()}
    assert [len(made[BT.name(BR.run_root("s", 0), *k, 0, 3)].rows) for k in ((8, 4), (4, 4))] == [12, 8]


def test_a_class_registers_nothing_when_rank_0_built_no_tables_or_answered_other_ones(tmp_path):
    _runner, hook, _hooks = serve()
    hook.latents.fail("step 9: ValueError: no slot")
    lines: list = []
    bw, sched, hooks = windows(tmp_path, hook.latents, built()[0], lines)
    try:
        finish(sched)
        bw.stop()
    finally:
        hooks.uninstall()
    assert [x["error"] for x in lines] == ["ValueError: rank 0 built no history tables: ValueError: rank 0's latent rows "
                                           "are incomplete: step 9: ValueError: no slot"] * 2
    assert not list(tmp_path.rglob(BR.COMMIT))
    L = BR.Layout(PAIR, 0, 1, 8, 4, 1, RUN, layers=(0, 3), capacity=CAP)
    entries = [HT.HistoryTable(L.partition["program"], n, H.WORDS, CAP).entry() for n in L.tables]

    def answer(tables):
        return {"format": ROOTS, "window": 0, "run": RUN, "prompt": 8, "outputs": 4, "tables": tables}
    assert BR.table_roots(answer(entries), L) == {e["value"]: bytes.fromhex(e["root"]) for e in entries}
    with pytest.raises(ValueError, match="not the class's"):
        BR.table_roots(answer(entries[::-1]), L)
    for other in ({"leaves": 8}, {"domain": HT.domain(PROGRAM, L.tables[0], H.WORDS, CAP).domain_id.hex()}, {"root": "00"}):
        with pytest.raises(ValueError, match="isn't at the class's leaves, words and domain"):
            BR.table_roots(answer([entries[0] | other, entries[1]]), L)
    with pytest.raises(ValueError, match="'format' is None"):
        BR.table_roots({"tables": entries}, L)
    with pytest.raises(ValueError, match="committed with their roots, not 0"):
        BR.commit([BR.Span("a", (1,) * 5, GREEDY, (2,) * 4, "stop")], L, 0)


def test_history_tables_refuse_a_scheduler_that_caches_prefixes(tmp_path):
    """A request served from the prefix cache computes no latent row before its hit's end, and rank 0 refuses rows that
    start past what it holds: every class committed after the first hit would register nothing. So the hook is refused
    on a caching scheduler before it serves a request, and nothing is wrapped; without tables it is installed
    (`test_boundary_rows`)."""
    sched, hooks, lines = Scheduler(), Hooks(tag="test-boundary-tables-caching"), []
    sched.cache_config.enable_prefix_caching = True
    bw = BW.BoundaryWindows(tmp_path, PAIR, 0, 1, 10.0, sink=lines.append, clock=lambda: 0.0, serve="s",
                            tables=BT.LocalChannel(BT.Builder(BT.Latents([0, 3]))), layers=(0, 3))
    try:
        with pytest.raises(ValueError, match="history tables need prefix caching off"):
            bw.install(sched, hooks)
        assert "add_request" not in vars(sched) and "schedule" not in vars(sched)
    finally:
        bw.stop()
        hooks.uninstall()
    assert lines == []


def test_the_tables_private_directory_is_never_under_what_leaves_the_node(tmp_path):
    cfg = {"check": "c", "tables": {"private": str(tmp_path / "node"), "layers": 2, "capacity": CAP, "timeout_s": 5}}
    bw = BW.BoundaryWindows.of(cfg, tmp_path / "out", public=[tmp_path / "run"])
    bw.stop()
    assert (bw.layers, bw.capacity, bw.tables.timeout_s, bw.tables.dir) == ((0, 1), CAP, 5.0, tmp_path / "node" / "channel")
    for under in (tmp_path / "out" / "node", tmp_path / "run" / "x"):
        with pytest.raises(ValueError, match="which leaves the node"):
            BW.BoundaryWindows.of({**cfg, "tables": {**cfg["tables"], "private": str(under)}}, tmp_path / "out",
                                  public=[tmp_path / "run"])
    assert BT.layers_of([3, 61]) == (3, 61)
    with pytest.raises(ValueError, match="both rank 0's channel and their layers"):
        BW.BoundaryWindows(tmp_path / "x", PAIR, 0, 1, 10.0, layers=(0,))


def modes(root) -> set[tuple[str, int]]:
    return {("dir" if q.is_dir() else "file", stat.S_IMODE(q.stat().st_mode)) for q in [root, *root.rglob("*")]}


def kept_window(tmp_path) -> tuple[BT.Kept, list]:
    """`finish`'s window served with rank 0 keeping its tables under ``tmp_path``/node: the kept files and the tables."""
    _runner, hook, _hooks = serve()
    make, tables, lines = *built(), []
    kept = BT.Kept(tmp_path / "node" / "tables")
    bw, sched, hooks = windows(tmp_path / "out", hook.latents, make, lines, kept)
    try:
        finish(sched)
        bw.stop()
    finally:
        hooks.uninstall()
    assert all("error" not in x for x in lines)
    return kept, tables


def test_rank_0_keeps_each_tables_rows_and_salts_on_the_node_and_round_2_reads_a_span_at_a_range(tmp_path):
    """Each table's rows and salts as built go to rank 0's private files, owner-only, before its entry is given; round 2
    reads a span's rows at a range of positions with their salts and public strings, rebuilds the table to its root and
    paths, and the class reopens with every table rebuilt from them; a changed file, or a range past what a table holds,
    is refused."""
    kept, tables = kept_window(tmp_path)
    assert modes(tmp_path / "node" / "tables") == {("dir", 0o700), ("file", 0o600)}
    assert not list((tmp_path / "out").rglob("*.rows")) and not list((tmp_path / "out").rglob("*.salts"))
    run, made = BR.run_root("s", 0), {t.value: t for t in tables}
    a3 = made[BT.name(run, 8, 4, 0, 3)]
    got = kept.span(0, run, 8, 4, 0, 3, range(2, 5))
    assert got["entry"] == a3.entry() and got["positions"] == range(2, 5)
    assert np.array_equal(got["rows"], np.stack([row("a", p, 3) for p in range(2, 5)]))
    assert got["salts"] == a3.salts[2:5] and got["strings"] == a3.bcs[2:5]
    again = kept.table(0, run, 8, 4, 0, 3)
    assert again.root == a3.root and again.path(7) == a3.path(7)
    c = BR.reopen(tmp_path / "out" / "w0" / "p8-o4", PAIR, kept)
    assert c.roots[a3.value] == a3.root
    past = kept.span(0, run, 8, 4, 0, 3, range(8, 12))
    assert not past["rows"].any() and past["strings"] == a3.bcs[8:12] and HT.placeholder(H.WORDS) not in past["strings"]
    with pytest.raises(ValueError, match="aren't a range within the 12 it holds"):
        kept.span(0, run, 8, 4, 0, 3, range(4, 13))
    f = tmp_path / "node" / "tables" / f"w0-{run[:16]}-p8-o4" / "s0" / "layer-0.rows"
    raw = bytearray(f.read_bytes())
    raw[0] ^= 1
    f.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="rebuild to another table than the one kept"):
        BR.reopen(tmp_path / "out" / "w0" / "p8-o4", PAIR, kept)


def test_the_interior_loads_a_served_table_by_its_name_and_commits_it_again_to_round_1s_root(tmp_path):
    """`load` from the serve's private directory and a table's name alone: the table's entry, its program and its C rows and
    salts, which committed again under round 1's program, name, words and capacity give the root round 1's record lists for
    that port; a name that isn't a served table's, or one no table kept here has, is refused."""
    kept, tables = kept_window(tmp_path)
    run = BR.run_root("s", 0)
    for (p, o), slot_rows in (((8, 4), 8), ((4, 4), 5)):
        pub = json.loads((tmp_path / "out" / "w0" / f"p{p}-o{o}" / BR.COMMIT).read_text())
        for x in (0, 3):
            n = BT.name(run, p, o, 0, x)
            got = BT.load(tmp_path / "node", n)
            made = next(t for t in tables if t.value == n)
            assert got.entry() == made.entry() and got.root == pub["values"][n]["root"]
            assert got.program == BR.layout_of(pub, PAIR).partition["program"]
            assert got.rows.shape == (BT.class_rows(p, o, CAP), H.WORDS) and not got.rows[slot_rows:].any()
            assert got.salts == b"".join(made.salts) and got.strings() == made.bcs
            again = HT.HistoryTable(got.program, got.value, got.words, got.leaves)
            again.append(got.rows, [got.salts[i * 192:(i + 1) * 192] for i in range(len(got.rows))])
            assert again.root.hex() == pub["values"][n]["root"]
    for bad in ("history/x", BT.name(run, 8, 4, 0, 3) + "x", f"history/{run}/p08-o4/s0/model.layers.3.self_attn.latent"):
        with pytest.raises(ValueError, match="is no served table's name"):
            BT.load(tmp_path / "node", bad)
    for absent in (BT.name(run, 8, 4, 1, 3), BT.name("b" * 128, 8, 4, 0, 3), BT.name(run, 8, 4, 0, 1)):
        with pytest.raises(ValueError, match="rank 0 kept 0 tables named"):
            BT.load(tmp_path / "node", absent)


def test_a_class_that_registers_nothing_keeps_nothing(tmp_path):
    """A class whose second span's rows don't match its lengths: the first span's files were written, and are dropped."""
    _runner, hook, _hooks = serve()
    kept = BT.Kept(tmp_path / "tables")
    doc = BT.assignment(1, RUN, PROGRAM, 8, 4, [0, 3], CAP, [("a", 5, 4, "stop"), ("b", 3, 1, "stop")])
    out = BT.Builder(hook.latents, built()[0], keep=kept).build(doc)
    assert "slot 1: rank 0 holds 5 positions" in out["error"] and not list((tmp_path / "tables").rglob("*.json"))
    with pytest.raises(FileNotFoundError):
        kept.span(1, RUN, 8, 4, 0, 0, range(0, 1))
    assert BT.Kept.of({"private": str(tmp_path / "node")}, tmp_path / "out").dir == tmp_path / "node" / "tables"
    with pytest.raises(ValueError, match="which leaves the node"):
        BT.Kept.of({"private": str(tmp_path / "out" / "node")}, tmp_path / "out")
