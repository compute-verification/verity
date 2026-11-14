"""`boundary_tables` on rank 0: a table's key and its class's row count, a request's rows in `Latents`, the steps `StepHook`
reads of a model runner, the answer `Builder` gives a class, and the thread `Channel` serves on."""
from __future__ import annotations

import re
import threading
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import pytest
import torch
from verity.commitments import history as HT

from verity_vllm.engine.hooks import Hooks
from verity_vllm.pipeline import boundary_tables as BT

RUN, PROGRAM, LEAVES = "a" * 128, "cd" * 64, 16
LAYER = "model.layers.3.self_attn.attn"


@pytest.mark.parametrize("call, why", [
    (lambda: BT.key("", 4, 8, 0), "a table's name binds its window's run: give it"),
    (lambda: BT.key(RUN, -1, 8, 0), "a table's prompt is a natural number, not -1"),
    (lambda: BT.key(RUN, 4, True, 0), "a table's outputs is a natural number, not True"),
    (lambda: BT.key(RUN, 4, 8, 1.0), "a table's slot is a natural number, not 1.0"),
    (lambda: BT.class_rows(4, 8, 3), "a table's capacity is a power of two of at least 2, not 3"),
    (lambda: BT.fill(np.zeros((5, 2), np.uint16), 2, 2, LEAVES),
     "a span's rows are at most the 4 its class (2, 2) holds at capacity 16, not shape (5, 2)"),
])
def test_a_key_a_class_and_a_fill_say_what_they_refuse(call, why):
    with pytest.raises(ValueError, match=f"^{re.escape(why)}$"):
        call()


@pytest.mark.parametrize("args, which, given", [((0, 8), "prompt", "0"), ((4, 8.0), "outputs", "8.0")])
def test_a_class_names_which_of_its_two_lengths_is_no_positive_integer(args, which, given):
    with pytest.raises(ValueError, match=rf"^a class's {which} \w+ is a positive integer, not {re.escape(given)}$"):
        BT.class_rows(*args)


def test_a_class_holds_its_prompts_and_its_outputs_positions_from_two_rows_up_to_the_tables_capacity():
    assert BT.class_rows(1, 1) == 2 and BT.class_rows(3, 2) == 5
    assert BT.class_rows(4, 8, 8) == 8 and BT.class_rows(BT.CAPACITY, 1) == BT.CAPACITY


def test_latents_hold_nothing_of_a_request_they_never_saw():
    lat = BT.Latents((0, 3), words=4)
    assert lat.length("a") == 0 and lat.requests() == []
    assert {x: (r.shape, r.dtype) for x, r in lat.take("a").items()} == {0: ((0, 4), np.uint16), 3: ((0, 4), np.uint16)}


def test_latents_keep_their_own_copy_of_a_steps_rows():
    lat = BT.Latents((0,), words=4)
    rows: np.ndarray = np.arange(8, dtype=np.uint16).reshape(2, 4)
    lat.add("a", 0, {0: rows})
    rows[:] = 0
    assert lat.take("a")[0].tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_latents_refuse_a_step_whose_layers_hold_blocks_of_other_lengths():
    lat = BT.Latents((0, 3), words=4)
    why = "request a: one (k, 4) block of rows at each layer of [0, 3], not [0, 3] of shapes [(1, 4), (2, 4)]"
    with pytest.raises(ValueError, match=f"^{re.escape(why)}$"):
        lat.add("a", 0, {0: np.zeros((1, 4), np.uint16), 3: np.zeros((2, 4), np.uint16)})
    assert lat.length("a") == 0


def served(blocks: int = 2, block: int = 4) -> BT.StepHook:
    """The step hook of a model of one MLA layer, at index 3, whose cache row at slot s holds the word s + 1 throughout."""
    m = torch.nn.Module()
    m.layer_name, m.kv_lora_rank, m.qk_rope_head_dim = LAYER, BT.WORDS - 64, 64
    words = (1 + torch.arange(blocks * block, dtype=torch.int16)).repeat_interleave(BT.WORDS)
    m.kv_cache = words.view(blocks, block, BT.WORDS).view(torch.bfloat16)
    return BT.StepHook(m)


def state(slots: list[int], start: int = 0) -> NS:
    """A model runner's state after a step of request a alone: its tokens' slots, at positions ``start`` on."""
    batch = NS(req_ids=["a"], query_start_loc_np=np.array([0, len(slots)]), num_computed_tokens_np=np.array([start]))
    return NS(input_batch=batch, slot_mappings_by_layer={LAYER: torch.tensor(slots, dtype=torch.int32)})


def step(slots: list[int], **more: Any) -> tuple[NS, NS]:
    """A runner after that step, and the step's scheduler output."""
    return NS(execute_model_state=state(slots)), NS(total_num_scheduled_tokens=len(slots), **more)


@pytest.mark.parametrize("blocks, block", [(2, 4), (8, 1)])
def test_the_hook_reads_slot_s_as_row_s_mod_the_block_size_of_block_s_over_it_and_counts_the_step(blocks, block):
    hook = served(blocks, block)
    hook.after(*step([5, 0, 2]))
    assert hook.steps == 1 and hook.latents.failed is None
    assert hook.latents.take("a")[3].tolist() == [[6] * BT.WORDS, [1] * BT.WORDS, [3] * BT.WORDS]


@pytest.mark.parametrize("so", [NS(), NS(total_num_scheduled_tokens=0), NS(total_num_scheduled_tokens=None)])
def test_a_step_that_scheduled_no_token_is_neither_read_nor_counted(so):
    hook = served()
    hook.after(NS(), so)
    assert hook.steps == 0 and hook.latents.failed is None


@pytest.mark.parametrize("slots, more, why", [
    ([0, -1], {}, "a scheduled token has no slot (-1): its row went to no cache position"),
    ([0, 1], {"scheduled_spec_decode_tokens": {"a": [7]}},
     "a step with draft tokens: a rejected draft's row is no position the request keeps"),
])
def test_a_step_the_hook_cannot_read_fails_the_rows_and_no_later_step_is_read(slots, more, why):
    hook = served()
    hook.after(*step(slots, **more))
    assert hook.latents.failed == f"step 0: ValueError: {why}" and hook.steps == 1
    hook.after(*step([2, 3]))
    assert hook.steps == 1 and hook.latents.requests() == []


def test_a_dummy_run_is_not_a_step_though_it_schedules_tokens():
    hook, hooks = served(), Hooks(tag="test")
    runner = NS(execute_model=lambda so, dummy_run=False: "out", execute_model_state=state([0, 1]))
    hook.install(runner, hooks)
    assert runner.execute_model(NS(total_num_scheduled_tokens=2), dummy_run=True) == "out" and hook.steps == 0
    assert runner.execute_model(NS(total_num_scheduled_tokens=2)) == "out" and hook.steps == 1
    hooks.uninstall()


def one_layer(positions: int) -> tuple[BT.Latents, dict[str, Any]]:
    """Rank 0's rows at layer 3 alone, ``positions`` of request a, and class (2, 2)'s assignment of a, which stopped after
    a two-id prompt and two ids served: three positions."""
    lat = BT.Latents((3,))
    lat.add("a", 0, {3: np.full((positions, BT.WORDS), 7, np.uint16)})
    return lat, BT.assignment(0, RUN, PROGRAM, 2, 2, [3], LEAVES, [("a", 2, 2, "stop")])


def test_the_builder_builds_a_class_whose_tables_are_at_one_layer():
    lat, doc = one_layer(3)
    out = BT.Builder(lat, HT.HistoryTable).build(doc)
    assert "error" not in out
    assert [(e["value"], e["leaves"], e["words"]) for e in out["tables"]] == [(BT.name(RUN, 2, 2, 0, 3), LEAVES, BT.WORDS)]


def test_a_class_the_builder_refuses_is_answered_with_the_reason_though_nothing_of_it_was_kept(tmp_path):
    lat, doc = one_layer(1)
    out = BT.Builder(lat, HT.HistoryTable, keep=BT.Kept(tmp_path / "tables")).build(doc)
    why = "ValueError: slot 0: rank 0 holds 1 positions of it, not the 3 to 3 its lengths and finish (stop) give"
    assert out == {"format": BT.ROOTS, "window": 0, "run": RUN, "prompt": 2, "outputs": 2, "error": why}
    assert lat.requests() == [] and not (tmp_path / "tables").exists()


def test_the_channel_serves_on_a_daemon_thread_every_poll_until_its_event_is_set(tmp_path, monkeypatch):
    """The thread is a stand-in that only keeps what it was made with, and the event one that its first wait sets, so the
    thread's loop runs here and ends after one pass."""
    made: list[dict[str, Any]] = []
    waits: list[float | None] = []

    class Once(threading.Event):
        def wait(self, timeout: float | None = None) -> bool:
            waits.append(timeout)
            self.set()
            return True

    def thread(**kw: Any) -> NS:
        made.append(kw)
        return NS(start=lambda: None)

    def unanswered(_s: float) -> None:
        raise AssertionError("rank 0 answered nothing")

    monkeypatch.setattr(BT, "threading", NS(Event=Once, Thread=thread))
    ch, builder = BT.Channel(tmp_path, poll_s=0.25, sleep=unanswered), NS(build=lambda doc: {"answered": doc["window"]})
    ch.assign({"window": 0, "run": RUN, "prompt": 2, "outputs": 2})
    stop = ch.start(builder)
    (kw,) = made
    assert kw["daemon"] is True and kw["name"] == "boundary-tables" and not stop.is_set()
    kw["target"]()
    assert stop.is_set() and waits == [0.25] and ch.roots(0, RUN, 2, 2) == {"answered": 0}
    ch.assign({"window": 1, "run": RUN, "prompt": 2, "outputs": 2})
    kw["target"]()
    assert waits == [0.25] and ch.serve(builder) == 1
