"""The boundary state's capture and restore (`boundary`) on stand-ins for vLLM d9105ea80's block pool, requests, engine
core and model runner (the attributes they read and write, as vLLM has them), and the restore's comparisons."""
from __future__ import annotations

import pickle
from collections import defaultdict, deque
from dataclasses import dataclass
from types import SimpleNamespace as NS

import boundary as B
import pytest


@dataclass(slots=True)
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    _block_hash: bytes | None = None
    _block_hash_num_tokens: int | None = None
    prev_free_block: KVCacheBlock | None = None
    next_free_block: KVCacheBlock | None = None
    is_null: bool = False


class FreeQueue:
    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)
        self.fake_free_list_head, self.fake_free_list_tail = KVCacheBlock(-1), KVCacheBlock(-1)
        prev = self.fake_free_list_head
        for b in blocks:
            prev.next_free_block, b.prev_free_block, prev = b, prev, b
        prev.next_free_block, self.fake_free_list_tail.prev_free_block = self.fake_free_list_tail, prev

    def popleft(self) -> KVCacheBlock:
        b = self.fake_free_list_head.next_free_block
        self.remove(b)
        return b

    def remove(self, b: KVCacheBlock) -> None:
        b.prev_free_block.next_free_block, b.next_free_block.prev_free_block = b.next_free_block, b.prev_free_block
        b.prev_free_block = b.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, b: KVCacheBlock) -> None:
        last = self.fake_free_list_tail.prev_free_block
        last.next_free_block, b.prev_free_block = b, last
        b.next_free_block, self.fake_free_list_tail.prev_free_block = self.fake_free_list_tail, b
        self.num_free_blocks += 1


class Pool:
    def __init__(self, n: int) -> None:
        self.num_gpu_blocks, self.enable_caching = n, True
        self.blocks = [KVCacheBlock(i) for i in range(n)]
        self.free_block_queue = FreeQueue(self.blocks)
        self.cached_block_hash_to_block = NS(_cache={})
        self.cached_block_hashes_by_block: dict[int, set[bytes]] = {}
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
        self.kv_event_queue: list = []


def served_pool() -> Pool:
    """Blocks 1-3 held (2 and 3 cached under a hash), 4 freed after caching, then 2 freed back: an order unlike the ids'."""
    p = Pool(8)
    for i in (1, 2, 3, 4):
        p.free_block_queue.remove(p.blocks[i])
        p.blocks[i].ref_cnt = 1
    for i, h in ((2, b"h2"), (3, b"h3"), (4, b"h4")):
        p.blocks[i]._block_hash, p.blocks[i]._block_hash_num_tokens = h, 16
        p.cached_block_hash_to_block._cache[h] = p.blocks[i]
        p.cached_block_hashes_by_block[i] = {h}
    p.cached_block_hash_to_block._cache[b"h5"] = {0: p.blocks[3], 1: p.blocks[4]}
    for i in (4, 2):
        p.blocks[i].ref_cnt = 0
        p.free_block_queue.append(p.blocks[i])
    return p


def test_pool_state_round_trips_into_a_fresh_pool():
    p, q = served_pool(), Pool(8)
    st = B.pool_state(p)
    assert st["free"] == [5, 6, 7, 4, 2] and st["num_free"] == 5 and st["null_block"] == 0
    B.put_pool(q, pickle.loads(pickle.dumps(st)))
    assert B.pool_state(q) == st
    assert B.pool_diff(st, B.pool_state(q)) == {}
    assert q.cached_block_hash_to_block._cache[b"h5"][1] is q.blocks[4]
    assert q.free_block_queue.popleft() is q.blocks[5]


def test_pool_diff_names_the_blocks_and_order_that_differ():
    p, q = served_pool(), served_pool()
    q.blocks[6].ref_cnt = 2
    q.free_block_queue.remove(q.blocks[4])
    q.free_block_queue.append(q.blocks[4])
    d = B.pool_diff(B.pool_state(p), B.pool_state(q))
    assert d["ref_cnt"] == {"blocks": 1, "first": [6]}
    assert d["free"] == {"len": [5, 5], "first_position": 3}
    assert "num_free" not in d and "cache" not in d


def test_put_pool_refuses_another_configuration():
    with pytest.raises(ValueError, match="not the served configuration"):
        B.put_pool(Pool(9), B.pool_state(served_pool()))


def test_attrs_name_requests_and_blocks_and_skip_objects():
    p = served_pool()
    req = Request(0, 0.0)
    req.request_id = "r1"
    m = NS(req_to_blocks=defaultdict(list, {"r1": [p.blocks[1], p.blocks[3]]}), num_cached_block={"r1": 1},
           _partial_hit_reqs={"r1": (16, p.blocks[3])}, running=[req], waiting=deque([req]), ids={"a"},
           config=object(), step=7)
    st, skipped = B.attrs(m)
    assert st["req_to_blocks"] == {"r1": [B.Ref("B", 1), B.Ref("B", 3)]}
    assert st["_partial_hit_reqs"] == {"r1": (16, B.Ref("B", 3))}
    assert st["running"] == [B.Ref("R", "r1")] and st["waiting"] == [B.Ref("R", "r1")]
    assert skipped == {"config": "object"}
    fresh = NS(req_to_blocks=defaultdict(list), num_cached_block={}, _partial_hit_reqs={}, running=[], waiting=deque(),
               ids=set(), config=None, step=0)
    keep = fresh.req_to_blocks
    B.put_attrs(fresh, st, lambda r: p.blocks[r.key] if r.kind == "B" else req)
    assert fresh.req_to_blocks is keep and fresh.req_to_blocks["r1"] == [p.blocks[1], p.blocks[3]]
    assert isinstance(fresh.waiting, deque) and fresh.waiting[0] is req
    assert (fresh.ids, fresh.step, fresh.config) == ({"a"}, 7, None)


def test_requests_pickle_with_their_block_hasher_named():
    seen = []
    hasher = lambda r: seen.append(r) or []   # noqa: E731 -- a closure pickle can't hold
    reqs = [NS(request_id="a", _block_hasher=hasher, tokens=[1, 2]), NS(request_id="b", _block_hasher=hasher, tokens=[3])]
    with pytest.raises(Exception):
        pickle.dumps(reqs)
    fresh = lambda r: []   # noqa: E731
    back = B.unpickle_requests(B.pickle_requests(reqs, {"block_hasher": hasher}), {"block_hasher": fresh})
    assert [(r.request_id, r.tokens) for r in back] == [("a", [1, 2]), ("b", [3])]
    assert all(r._block_hasher is fresh for r in back)


# -------------------------------------------------------------------------------------------------- the runner


class RequestState:
    __module__ = "vllm.v1.worker.gpu.states"

    def __init__(self) -> None:
        self.req_id_to_index: dict[str, int] = {}
        self.free_indices = [3, 2, 1, 0]
        self.max_seq_len = [0, 0, 0, 0]
        self.block_table_ptrs = [0xdead]
        self.inner = Inner()


class Inner:
    __module__ = "vllm.v1.worker.gpu.sample.states"

    def __init__(self) -> None:
        self.seeds_set = [False] * 4
        self.stream = object()


def test_runner_state_walks_the_state_objects_and_passes_by_addresses():
    r = NS(req_states=RequestState(), model=object())
    r.req_states.req_id_to_index["x"] = r.req_states.free_indices.pop()
    r.req_states.inner.seeds_set[0] = True
    st = B.runner_state(r)
    assert set(st["leaves"]) == {"req_states.req_id_to_index", "req_states.free_indices", "req_states.max_seq_len",
                                 "req_states.inner.seeds_set"}
    assert st["skipped"] == {"req_states.block_table_ptrs": "address", "req_states.inner.stream": "object"}
    fresh = NS(req_states=RequestState(), model=object())
    keep = fresh.req_states.free_indices
    assert B.put_runner(fresh, pickle.loads(pickle.dumps(st))) == []
    assert fresh.req_states.free_indices is keep and keep == [3, 2, 1]
    assert fresh.req_states.req_id_to_index == {"x": 0} and fresh.req_states.inner.seeds_set[0] is True


def test_runner_state_copies_arrays_and_uva_backed_tensors():
    np = pytest.importorskip("numpy")

    class UvaBackedTensor:
        def __init__(self, n: int) -> None:
            self.np, self.copies = np.zeros(n, dtype=np.int32), 0

        def copy_to_uva(self) -> None:
            self.copies += 1

    holder = type("SamplingStates", (), {"__module__": "vllm.v1.worker.gpu.sample.states"})
    a, b = holder(), holder()
    for h in (a, b):
        h.seeds, h.num_logprobs = UvaBackedTensor(4), np.zeros(4, dtype=np.int64)
    a.seeds.np[1], a.num_logprobs[2] = 99, -1
    st = B.runner_state(NS(sampler=a))
    assert {p: k for p, (k, _v) in st["leaves"].items()} == {"sampler.seeds": "UvaBackedTensor", "sampler.num_logprobs": "array"}
    B.put_runner(NS(sampler=b), st)
    assert b.seeds.np.tolist() == [0, 99, 0, 0] and b.seeds.copies == 1 and b.num_logprobs.tolist() == [0, 0, -1, 0]
    assert B.put_runner(NS(), st) == ["sampler.seeds", "sampler.num_logprobs"]


# --------------------------------------------------------------------------------------------------- dump and diff

class Request:
    __module__ = "vllm.v1.request"

    def __init__(self, n: int, t: float) -> None:
        self.request_id, self.num_computed_tokens, self.arrival_time = "a", n, t
        self.block_hash = b"\x01" * 80
        self.tokens = list(range(40))
        self.hasher = lambda r: []


class Scheduler:
    __module__ = "vllm.v1.core.sched.scheduler"

    def __init__(self, req: Request) -> None:
        self.requests, self.running = {req.request_id: req}, [req]


def test_dump_and_diff_find_what_a_restore_missed_and_set_clock_readings_apart():
    a = B.dump(Scheduler(Request(5, 1.0)), "sched")
    assert a["sched.running[0]"] == "@sched.requests['a']" and a["sched.requests['a'].hasher"] == "obj:function"
    req = Request(5, 2.0)
    d = B.diff(a, B.dump(Scheduler(req), "sched"))
    assert (d["differing"], d["clock_differing"]) == (0, 1)
    req.num_computed_tokens, req.tokens[39] = 6, -1
    d = B.diff(a, B.dump(Scheduler(req), "sched"))
    assert sorted(x["path"] for x in d["first"]) == ["sched.requests['a'].num_computed_tokens", "sched.requests['a'].tokens"]
    assert d["patterns"] == {"sched.requests[*].num_computed_tokens": 1, "sched.requests[*].tokens": 1}


def test_diff_sets_paths_on_one_side_apart_and_dump_leaves_out_the_harness():
    s = Scheduler(Request(5, 1.0))
    a = B.dump(s, "sched")
    s._verity_trace = ["hooks"]
    s.extra = 3
    b = B.dump(s, "sched")
    assert not any("_verity" in p for p in b)
    d = B.diff(a, b)
    assert (d["differing"], d["only_served"], d["only_restored"]) == (0, [], ["sched.extra"])
    assert B.attrs(s)[0].keys() == {"requests", "running", "extra"}


# ------------------------------------------------------------------------------------------------------- the edges

class Sched:
    def __init__(self, work: int) -> None:
        self.work, self.scheduled = work, 0

    def has_requests(self) -> bool:
        return self.scheduled < self.work


class Core:
    """A stand-in engine core whose ``step_with_batch_queue`` schedules while its scheduler has requests, and finishes the
    oldest batch once two are in flight or nothing more is scheduled, as vLLM's does."""

    def __init__(self, work: int) -> None:
        self.batch_queue: deque = deque()
        self.scheduler = Sched(work)

    def step(self) -> tuple[str, int] | None:
        s = self.scheduler
        if s.has_requests():
            self.batch_queue.appendleft(s.scheduled)
            s.scheduled += 1
            if len(self.batch_queue) < 2:
                return None
        if not self.batch_queue:
            return None
        return ("out", self.batch_queue.pop())


def test_an_edge_drains_the_batches_in_flight_then_takes_the_state_between_steps():
    core = Core(work=10)
    taken = []
    now = [0.0]
    due = lambda e: e.edge == 0 and core.scheduler.scheduled >= 3   # noqa: E731
    edges = B.Edges(core, due, lambda e, info: taken.append((dict(info), len(core.batch_queue), core.scheduler.scheduled)),
                    lambda: now[0])
    orig = Core.step
    assert [edges.step(orig) for _ in range(3)] == [None, ("out", 0), ("out", 1)]
    assert core.scheduler.scheduled == 3 and list(core.batch_queue) == [2]
    now[0] = 1.0
    assert edges.step(orig) == ("out", 2) and core.scheduler.scheduled == 3 and not taken
    now[0] = 2.0
    assert edges.step(orig) is None
    assert taken == [({"edge": 0, "due_t": 1.0, "drained_t": 2.0, "drain_calls": 1}, 0, 3)]
    assert core.scheduler.scheduled == 4 and edges.edge == 1
    assert "has_requests" not in vars(core.scheduler)


def test_drain_one_puts_back_a_has_requests_the_scheduler_held_itself():
    core = Core(work=5)
    own = lambda: True   # noqa: E731
    core.scheduler.has_requests = own
    core.batch_queue.appendleft(0)
    B.drain_one(lambda c: c.scheduler.has_requests(), core)
    assert core.scheduler.has_requests is own


def test_edges_fall_at_times_after_the_start_and_at_steps():
    now = [0.0]
    due, start = B.at_times([10.0, 20.0], lambda: now[0])
    e = NS(edge=0)
    now[0] = 50.0
    assert not due(e)
    start()
    now[0] = 59.0
    assert not due(e)
    now[0] = 60.0
    assert due(e)
    start()
    e.edge = 1
    assert not due(e)
    now[0] = 70.0
    assert due(e)
    e.edge = 2
    assert not due(e)
    count = [4]
    at = B.at_steps([5], lambda: count[0])
    assert not at(NS(edge=0))
    count[0] = 5
    assert at(NS(edge=0)) and not at(NS(edge=1))


class Event:
    def __init__(self, clock: list[float]) -> None:
        self.clock, self.t, self.done = clock, None, False

    def record(self) -> None:
        self.t = self.clock[0]

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.done = True

    def elapsed_time(self, other: Event) -> float:
        return (other.t - self.t) * 1e3


def test_busy_counts_each_steps_gpu_time_by_the_traces_step_index():
    clock = [0.0]
    made: list[Event] = []
    busy = B.Busy(lambda: made.append(Event(clock)) or made[-1], first=7)
    busy.started(0)
    busy.sampled()
    for t0, t1 in ((1.0, 1.5), (2.0, 2.25)):
        clock[0] = t0
        busy.started(16)
        clock[0] = t1
        busy.sampled()
    made[1].done = True
    busy.reap()
    assert busy.done == {7: 0.5} and len(busy.open) == 1
    busy.reap(wait=True)
    assert busy.total(7, 9) == (0.75, 2) and busy.total(8, 9) == (0.25, 1)


# ------------------------------------------------------------------------------------------------------- outputs

def test_step_outputs_and_their_diff():
    def eco(rid: str, toks: list[int]) -> NS:
        return NS(request_id=rid, new_token_ids=toks, new_logprobs=None)

    served = {5: B.step_outputs({0: NS(outputs=[eco("a", [1]), eco("b", [2]), eco("c", [])])}), 6: {"a": [[3], [7]]}}
    assert served[5] == {"a": [[1], None], "b": [[2], None]}
    assert B.outputs_diff(served, served)["first"] is None
    d = B.outputs_diff(served, {5: served[5], 6: {"a": [[3], [8]]}})
    assert (d["request_steps_tokens_differing"], d["request_steps_logprobs_differing"]) == (0, 1)
    assert d["first"] == {"step": 6, "request": "a", "served": [[3], [7]], "replayed": [[3], [8]]}
    d = B.outputs_diff(served, {5: {"a": [[1], None]}, 6: served[6]})
    assert d["request_steps_tokens_differing"] == 1 and d["first"]["request"] == "b"


def test_logprob_bits_are_the_float32_patterns():
    np = pytest.importorskip("numpy")
    lp = NS(logprobs=np.array([[-0.5], [-1.25]], dtype=np.float32))
    assert B.logprob_bits(lp) == np.array([-0.5, -1.25], dtype=np.float32).view(np.uint32).tolist()


def test_rng_state_puts_back_what_the_next_draws_read():
    np = pytest.importorskip("numpy")
    import random

    st = B.rng_state()
    want = (np.random.randint(0, 1 << 62), random.random())
    np.random.randint(0, 9, size=100)
    random.random()
    B.set_rng(pickle.loads(pickle.dumps(st)))
    assert (np.random.randint(0, 1 << 62), random.random()) == want
