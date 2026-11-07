"""`tp_edge` on a stand-in engine core whose ``step_with_batch_queue`` keeps two steps in flight as vLLM d9105ea80's
does, and on stand-in block pools; the snapshot itself on stand-in ranks holding CPU tensors (`test_tp_state`'s)."""
from __future__ import annotations

import json
from collections import deque
from types import SimpleNamespace as NS

import pytest

import tp_edge as E


class Block:
    def __init__(self, block_id, ref_cnt=0, block_hash=None, is_null=False):
        self.block_id, self.ref_cnt, self.block_hash, self.is_null = block_id, ref_cnt, block_hash, is_null


class Pool:
    def __init__(self, blocks):
        self.blocks = blocks
        self.null_block = blocks[0]

    def get_num_free_blocks(self):
        return sum(1 for b in self.blocks[1:] if b.ref_cnt == 0)


def pool():
    return Pool([Block(0, 1, is_null=True), Block(1, 2, "h1"), Block(2), Block(3, 0, "h3"), Block(4, 1), Block(5)])


def test_the_live_blocks_are_the_null_held_and_cached_free_ones():
    p = pool()
    assert E.live_blocks(p) == [0, 1, 3, 4]
    assert E.pool_counts(p) == {"blocks": 6, "held": 2, "cached_free": 1, "free": 3}


class Sched:
    def __init__(self, work: int):
        self.work, self.current_step, self.processed = work, 0, []
        self.sched_step_seq = self.processed_step_seq = 0
        self.deferred_frees = deque()

    def has_requests(self):
        return self.work > 0

    def schedule(self):
        self.current_step += 1
        self.work -= 1
        return self.current_step

    def update_from_output(self, step):
        self.processed.append(step)
        return {0: step}


class Core:
    """vLLM's ``step_with_batch_queue`` reduced to its queue: schedule into the queue while it has room and the scheduler
    has work, else block on the oldest step and process its output."""

    def __init__(self, work: int, queue: int | None = 2):
        self.scheduler = Sched(work)
        self.batch_queue = None if queue is None else deque(maxlen=queue)
        self.size = queue
        self.step_fn = self.step if queue is None else self.step_with_batch_queue

    def step(self):
        if not self.scheduler.has_requests():
            return {}, False
        return self.scheduler.update_from_output(self.scheduler.schedule()), True

    def step_with_batch_queue(self):
        q = self.batch_queue
        if self.scheduler.has_requests():
            q.appendleft(self.scheduler.schedule())
            if len(q) < self.size:
                return None, True
        elif not q:
            return None, False
        return self.scheduler.update_from_output(q.pop()), True


def run(core: Core, edges: E.Edges, calls: int, seen: list):
    edges.snapshot = lambda c, edge: seen.append((edge, c.scheduler.current_step, list(c.scheduler.processed),
                                                 len(c.batch_queue or ())))
    E.install(core, edges)
    for _ in range(calls):
        core.step_fn()


def test_an_edge_drains_the_batch_queue_then_snapshots_then_schedules_on():
    core, seen = Core(work=10), []
    run(core, E.Edges("/unused", steps=[4, 7]), 30, seen)
    assert seen == [(4, 4, [1, 2, 3, 4], 0), (7, 7, [1, 2, 3, 4, 5, 6, 7], 0)]
    assert core.scheduler.processed == list(range(1, 11)) and "has_requests" not in vars(core.scheduler)


def test_under_synchronous_scheduling_the_edge_follows_its_step():
    core, seen = Core(work=6, queue=None), []
    run(core, E.Edges("/unused", steps=[2, 5]), 10, seen)
    assert seen == [(2, 2, [1, 2], 0), (5, 5, [1, 2, 3, 4, 5], 0)]


def test_an_undrained_edge_is_refused():
    core = Core(work=3)
    core.scheduler.sched_step_seq = 2
    core.scheduler.kv_cache_manager = NS(block_pool=pool())
    with pytest.raises(RuntimeError, match="edge 1 is not drained: sched_step_seq 2, processed_step_seq 0"):
        E.Edges("/unused", steps=[1]).snapshot(core, 1)


def rec(rank, digests, rng="r", layers=("a", "b")):
    return {"rank": {"rank": rank, "host": "h", "device": {"uuid": f"GPU-{rank}"}, "allreduce": ["CUSTOM"]},
            "layers": [{"name": n} for n in layers], "layer_digests": list(digests), "rng_digest": rng}


def test_agree_names_the_ranks_and_layers_that_differ():
    same = [rec(r, ["x", "y"]) for r in (2, 0, 1)]
    assert E.agree(same) == {"ranks": [0, 1, 2], "layers_equal": True, "kv_equal": True, "rng_equal": True}
    got = E.agree([rec(0, ["x", "y"]), rec(1, ["x", "z"], rng="q"), rec(2, ["x", "y"])])
    assert got["kv_equal"] is False and got["kv_first_difference"] == {1: {"layer": 1, "name": "b"}}
    assert got["rng_differ"] == [1]
    assert E.agree([rec(0, ["x"]), rec(1, ["x"], layers=("a", "c"))])["layers_differ"] == [1]


def test_a_rank_map_that_moved_is_reported_per_rank_and_field():
    served = E.rank_map([rec(r, []) for r in (0, 1)])
    moved = E.rank_map([rec(0, []), rec(1, []) | {"rank": {"rank": 1, "host": "h", "device": {"uuid": "GPU-7"},
                                                           "allreduce": ["CUSTOM"]}}])
    assert E.check_rank_map(served, served) == []
    assert E.check_rank_map(served, moved) == ["rank 1: device {'uuid': 'GPU-1'} served, {'uuid': 'GPU-7'} restored"]
    assert E.check_rank_map(served, served[:1]) == ["2 served ranks, 1 restored"]


def test_the_plugin_config_needs_out_and_an_edge_and_savers_resolve():
    assert E.config({}) is None
    assert E.config({E.ENV: json.dumps({"out": "/o", "steps": [3]})}) == {"out": "/o", "steps": [3]}
    assert E.config({E.ENV: json.dumps({"out": "/o", "at_s": [60]})}) == {"out": "/o", "at_s": [60]}
    with pytest.raises(ValueError, match="needs out, and steps or at_s"):
        E.config({E.ENV: json.dumps({"out": "/o"})})
    assert E.resolve("tp_edge:live_blocks") is E.live_blocks
    with pytest.raises(ValueError, match="module:function"):
        E.resolve("tp_edge.live_blocks")
    with pytest.raises(ValueError, match="no edge"):
        E.Edges("/o")


def test_an_edge_at_a_time_falls_on_the_first_step_after_it():
    now = [0.0]
    core, seen = Core(work=12), []
    edges = E.Edges("/unused", at_s=[2.5, 5.0], clock=lambda: now[0])
    edges.snapshot = lambda c, edge: seen.append((edge, list(c.scheduler.processed)))
    E.install(core, edges)
    for _ in range(30):
        core.step_fn()
        now[0] += 1.0
    assert seen == [(4, [1, 2, 3, 4]), (5, [1, 2, 3, 4, 5])]


def test_a_snapshot_checks_the_ranks_dumps_one_copy_and_records_the_edge(tmp_path):
    pytest.importorskip("torch")
    import test_tp_state as T

    workers = [T.worker(1, rank=r) for r in range(3)]

    def collective_rpc(fn, args=()):
        return [fn(w, *args) for w in workers]

    core = Core(work=0)
    core.scheduler.kv_cache_manager = NS(block_pool=Pool([Block(0, 1, is_null=True)] +
                                                         [Block(i, i % 2, f"h{i}" if i % 3 else None)
                                                          for i in range(1, T.BLOCKS)]))
    core.scheduler.running, core.scheduler.waiting = [1, 2], [3]
    core.model_executor = NS(collective_rpc=collective_rpc)
    saved = []
    edges = E.Edges(tmp_path, steps=[5], savers=[lambda c, d: saved.append(d.name)], clock=iter(range(100)).__next__)
    got = edges.snapshot(core, 5)
    assert got["check"] == {"ranks": [0, 1, 2], "layers_equal": True, "kv_equal": True, "rng_equal": True}
    assert got["ids"] == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11] and got["pool"]["cached_free"] == 4
    assert [w["rank"] for w in got["dump"]] == [0] and saved == ["edge-000005"]
    assert json.loads((tmp_path / "edge-000005" / "edge.json").read_text())["ids"] == got["ids"]
    assert got["seconds"] == {"digest": 1, "dump": 1, "savers": 1} and got["trace_step"] is None
    core.scheduler._verity_servetax = [object(), NS(scheduled=41)]
    assert E.trace_step(core.scheduler) == 41
    assert edges.snapshot(core, 9)["dump"] == []

    T.tensors(workers[2])[2][7, 1, 1] += 1
    got = E.Edges(tmp_path, steps=[6], dump=1).snapshot(core, 6)
    assert got["check"]["kv_equal"] is False
    assert got["check"]["kv_first_difference"] == {2: {"layer": 2, "name": "model.layers.10.self_attn.attn", "block": 7,
                                                       "index": 6, "blocks_differing": 1}}
    assert sorted(w["rank"] for w in got["dump"]) == [0, 1, 2]
