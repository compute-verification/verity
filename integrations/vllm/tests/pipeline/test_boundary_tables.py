"""A two-stage window's history tables (`pipeline.boundary_tables`): their names from public items alone; rank 0's step
hook on a stand-in for vLLM's Model Runner V2 reading each MLA layer's rows at the step's slots, by request and position,
across chunked prefill, decode, padding and a recompute after a preemption; the builder's tables over them under the
class's names and domains, refused for a request whose rows don't match its lengths or after a step the hook couldn't
read; a cache that doesn't hold bf16 latent rows refused; and the channel between the two processes."""

from __future__ import annotations

import stat
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from verity.ml.commitments import history as HT
from verity_vllm.commit import history_tables as H
from verity_vllm.engine.hooks import Hooks
from verity_vllm.pipeline import boundary_tables as BT

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
    assert [len(t.rows) for t in tables] == [5, 5, 8, 8]
    assert np.array_equal(np.stack(tables[3].rows), np.stack([row("a", p, 3) for p in range(8)]))
    assert hook.latents.requests() == []


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
