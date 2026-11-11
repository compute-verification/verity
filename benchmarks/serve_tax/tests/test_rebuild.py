"""The rebuild's bookkeeping on stand-ins for vLLM's engine: the K/V file's layout, the hits the record forces and when
their pages load, the blocks a re-added request shares, the start's refusal of async scheduling's extra row, and the
per-request K/V roots and the mirror's save and load (numpy, skipped without it), and where ``--capture`` is
refused."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType
from types import SimpleNamespace as NS
from typing import Any

import pytest
import rebuild as RB


def test_slot_major_takes_mla_and_refuses_heads_first() -> None:
    assert RB.slot_major([64, 1, 16, 576], 16)
    assert RB.slot_major([64, 16, 576], 16)
    assert not RB.slot_major([64, 2, 16, 64], 16)
    assert not RB.slot_major([64], 16)


def test_layer_offsets_add_storage_starts_and_refuse_a_pool_without_block_rows() -> None:
    layout = {"block_row_bytes": 8, "storages": [100, 50, 20],
              "views": [{"storage": 0, "offset": 0}, {"storage": 1, "offset": 4}, {"storage": 2, "offset": 0}]}
    assert RB.layer_offsets(layout) == [0, 104, 150]
    with pytest.raises(SystemExit, match="block-major"):
        RB.layer_offsets({"block_row_bytes": 0, "storages": [1], "views": []})


def test_recorded_starts_list_each_new_or_resumed_start_from_the_first_step_on() -> None:
    steps = {
        3: {"rows": [["a", 0, 8]], "new": ["a"]},
        4: {"rows": [["a", 8, 1], ["b", 32, 4]], "new": ["b"]},
        6: {"rows": [["a", 9, 1], ["b", 0, 40]], "resumed": ["b"]},
    }
    assert RB.recorded_starts(steps, 4) == {"b": [(4, 32), (6, 0)]}
    assert RB.recorded_starts(steps, 0) == {"a": [(3, 0)], "b": [(4, 32), (6, 0)]}


def test_step_diff_drops_a_readded_request_at_its_first_fresh_step_only() -> None:
    want = {5: {"rows": [["a", 7, 1]], "new": [], "cache_hit": {}},
            6: {"rows": [["a", 8, 1], ["c", 0, 4]], "new": ["c"], "cache_hit": {"c": 0}}}
    got = {5: {"rows": [["a", 7, 1]], "new": ["a"], "cache_hit": {"a": 7}},
           6: {"rows": [["a", 8, 1], ["c", 0, 4]], "new": ["c"], "cache_hit": {"c": 0}}}
    assert RB.step_diff(want, got, 5, {"a"})["steps_differing"] == 0
    got[6] = {**got[6], "new": ["c", "a"]}
    d = RB.step_diff(want, got, 5, {"a"})
    assert d["steps_differing"] == 1 and d["first"]["step"] == 6


def test_first_of_takes_the_first_source_that_has_the_page() -> None:
    get = RB.first_of(lambda h: None, lambda h: [h], lambda h: ["late"])
    assert get(b"x") == [b"x"]
    assert RB.first_of(lambda h: None)(b"x") is None


class Blocks:
    def __init__(self, blocks: tuple[list[int], ...]) -> None:
        self.blocks = blocks


def hits(starts: dict[str, list[tuple[int, int]]], now: list[int]) -> Any:
    kvm = NS(block_pool=None, create_kv_cache_blocks=Blocks)
    return RB.Hits(kvm, [], lambda h: None, starts, 4, lambda: now[0])


def test_hits_due_seeds_a_request_at_the_step_the_record_schedules_it() -> None:
    h = hits({"a": [(3, 8), (9, 0)], "b": [(3, 4)]}, [0])
    seeded: list[tuple[str, int]] = []
    h.seed = lambda q, n: seeded.append((q, n))
    h.due({"a": "qa"}, 3)
    h.due({"a": "qa", "b": "qb"}, 5)
    h.due({"a": "qa", "b": "qb"}, 9)
    assert seeded == [("qa", 8), ("qa", 0)]
    assert h.want_at("a", 4) == 0 and h.want_at("b", 4) is None


def test_hits_lookup_cuts_a_longer_hit_and_notes_a_shorter_one() -> None:
    now = [3]
    h = hits({"a": [(3, 8)], "b": [(3, 12)]}, now)
    look = h.lookup(lambda q: (Blocks(([10, 11, 12, 13],)), 16))
    blocks, n = look(NS(request_id="a"))
    assert n == 8 and blocks.blocks == ([10, 11],)
    blocks, n = h.lookup(lambda q: (Blocks(([10],)), 4))(NS(request_id="b"))
    assert n == 4
    now[0] = 4
    assert look(NS(request_id="a"))[1] == 16
    assert h.report() | {"short_first": None} == {"requests_with_hits": 2, "pages_loaded": 0, "pages_not_found": 0,
                                                  "lookups_cut": 1, "lookups_short": 1, "short_first": None}
    assert h.short == [["b", 4, 12]]


@dataclass
class KVCacheBlocks:
    blocks: tuple[list[Any], ...]


class Kvm:
    """`allocate_slots` as the fresh pool answers it: the hit blocks first, then new ids for the rest."""

    def __init__(self, bs: int, room: bool = True) -> None:
        self.bs, self.room, self.next = bs, room, 10
        self.req: dict[str, list[int]] = {}
        self.calls: list[dict[str, Any]] = []
        self.cached: list[tuple[str, int]] = []

    def allocate_slots(self, q: Any, num_new_tokens: int, **kw: Any) -> Any:
        self.calls.append(kw)
        if not self.room:
            return None
        hit = [b.block_id for b in kw["new_computed_blocks"].blocks[0]] if kw.get("new_computed_blocks") else []
        k = -(-(kw["num_new_computed_tokens"] + kw["num_external_computed_tokens"]) // self.bs)
        new = list(range(self.next, self.next + k - len(hit)))
        self.next += len(new)
        self.req[q.request_id] = hit + new
        return NS()

    def cache_blocks(self, q: Any, n: int) -> None:
        self.cached.append((q.request_id, n))

    def get_block_ids(self, rid: str) -> tuple[list[int], ...]:
        return (self.req[rid],)


@pytest.fixture
def kv_cache_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    mods = {name: ModuleType(name) for name in ("vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.kv_cache_manager")}
    mods["vllm.v1.core.kv_cache_manager"].KVCacheBlocks = KVCacheBlocks
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)


def test_alloc_prefix_shares_the_leading_blocks_the_served_engine_shared(kv_cache_manager: None) -> None:
    kvm, pool, fresh_of = Kvm(4), NS(blocks=[NS(block_id=i) for i in range(32)]), {}
    a, b, c = (NS(request_id=r, num_computed_tokens=0) for r in "abc")
    assert RB.alloc_prefix(kvm, pool, a, 10, [100, 101, 102], fresh_of, 4) == ([10, 11, 12], 0)
    assert fresh_of == {100: 10, 101: 11} and a.num_computed_tokens == 10
    assert RB.alloc_prefix(kvm, pool, b, 9, [100, 101, 200], fresh_of, 4) == ([10, 11, 13], 2)
    assert kvm.calls[-1] | {"new_computed_blocks": None} == {
        "num_new_computed_tokens": 8, "new_computed_blocks": None, "num_external_computed_tokens": 1,
        "delay_cache_blocks": True, "has_scheduled_reqs": False}
    # a prefix that is all shared full blocks keeps its last one unshared: allocate_slots must place a token
    assert RB.alloc_prefix(kvm, pool, c, 8, [100, 101], fresh_of, 4) == ([10, 14], 1)
    assert fresh_of == {100: 10, 101: 11}
    assert kvm.cached == [("a", 10), ("b", 9), ("c", 8)]


def test_alloc_prefix_refuses_a_full_pool(kv_cache_manager: None) -> None:
    q = NS(request_id="a", num_computed_tokens=0)
    with pytest.raises(SystemExit, match="no room"):
        RB.alloc_prefix(Kvm(4, room=False), NS(blocks=[]), q, 8, [1, 2], {}, 4)


def test_start_state_refuses_a_step_that_runs_a_request_finished_the_step_before() -> None:
    def req(rid: str) -> Any:
        return NS(request_id=rid, prompt_token_ids=[1], output_token_ids=[2], num_computed_tokens=3,
                  num_output_placeholders=1, status=NS(name="RUNNING"), num_preemptions=0)
    sched = NS(running=[req("a"), req("x")], waiting=[], requests={"a": req("a"), "x": req("x")})
    served = {9: {"finished": {"x": {"reason": "length"}}}, 10: {"rows": [["a", 3, 1], ["x", 3, 1]]}}
    arrived = {"a": {"sampling": {}}, "x": {"sampling": {}}}
    with pytest.raises(SystemExit, match="start elsewhere"):
        RB.start_state(sched, None, None, None, arrived, served, 8, 10, 4, None)


def test_served_seeds_read_each_held_request_s_runner_slot_and_refuse_a_v1_edge() -> None:
    st = {"runner": {"leaves": {"req_states.req_id_to_index": ("plain", {"a": 2, "b": 0}),
                                "sampler.sampling_states.seeds": ("UvaBackedTensor", [11, -5, 7])}}}
    assert RB.served_seeds(st) == {"a": 7, "b": 11}
    with pytest.raises(SystemExit, match="V2_MODEL_RUNNER"):
        RB.served_seeds({"runner": {"leaves": {}}})
    with pytest.raises(SystemExit, match="no generators"):
        RB.carry_rng(None, start={"step": 3})


def test_roots_read_each_request_s_computed_tokens_only() -> None:
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    pool = [rng.integers(-128, 128, (8, 4 * 3), dtype=np.int8) for _ in range(2)]
    moved = [np.zeros_like(x) for x in pool]
    for x, y in zip(pool, moved, strict=True):
        y[[5, 6, 7]] = x[[1, 2, 3]]
    spans = {"a": ([1, 2], 6), "b": ([1, 3], 4)}
    want = RB.roots(pool, spans, 4)
    assert RB.roots(moved, {"a": ([5, 6], 6), "b": ([5, 7], 4)}, 4) == want
    moved[0][6, 2 * 3:] = 99
    assert RB.roots(moved, {"a": ([5, 6], 6), "b": ([5, 7], 4)}, 4) == want
    moved[1][6, 0] ^= 1
    d = RB.roots_diff(want, RB.roots(moved, {"a": ([5, 6], 6), "b": ([5, 7], 4)}, 4))
    assert d["differing"] == 1 and d["first"] == ["a"] and not d["roots_equal"]


def test_mirror_save_and_saved_read_back_rows_and_pages(tmp_path: Any) -> None:
    np = pytest.importorskip("numpy")
    m = RB.Mirror.__new__(RB.Mirror)
    rng = np.random.default_rng(1)
    m.bs, m.slot, m.flat = 2, 3, [None, None]
    m.store = {"a": rng.integers(-128, 128, (6, 2, 3), dtype=np.int8), "b": rng.integers(-128, 128, (6, 2, 3), dtype=np.int8)}
    m.written = {"a": 5, "b": 5}
    m.hashes = {b"\x01": ("a", 1), b"\x02": ("b", 1)}
    got = m.save(tmp_path / "mirror", {"a"})
    assert got == {"tokens": 9, "bytes": 9 * 2 * 3, "requests": 2, "pages": 2}
    s = RB.Saved(tmp_path / "mirror")
    assert (s.rows("a", 5) == m.store["a"][:5]).all() and (s.rows("b", 4) == m.store["b"][:4]).all()
    for h in (b"\x01", b"\x02"):
        assert all((x == y).all() for x, y in zip(s.page(h), m.page(h), strict=True))
    assert s.page(b"\x03") is None
    with pytest.raises(SystemExit, match="holds 4 tokens"):
        s.rows("b", 5)
    with pytest.raises(SystemExit, match="none of its tokens"):
        s.rows("c", 1)


def test_substitute_keeps_equal_pages_equal_and_the_target_s_pages_as_they_are() -> None:
    a = list(range(10, 22))
    b = list(range(10, 18)) + [7, 7, 7, 7, 5]
    sa, kept = RB.substitute(a, 4, set())
    sb, _ = RB.substitute(b, 4, set())
    assert kept == 0 and len(sa) == len(a) and len(sb) == len(b)
    assert sa[:8] == sb[:8] and sa[8:12] != sb[8:12]
    assert all(RB.SUBSTITUTE_IDS[0] <= x < RB.SUBSTITUTE_IDS[1] for x in sa + sb)
    assert sa[:8] != a[:8] and RB.substitute(a, 4, set())[0] == sa
    keep = set(RB.page_chain(list(range(10, 18)) + [0, 0, 0, 0], 4))
    got, kept = RB.substitute(b, 4, keep)
    assert kept == 2 and got[:8] == b[:8] and got[8:] == sb[8:]


def test_served_finish_counts_outputs_at_the_start_and_since_for_the_requests_the_record_finishes() -> None:
    outs = {5: {"a": [[1, 2], None], "b": [[3], None]}, 6: {"a": [[4], None], "c": [[9], None]}}
    steps = {5: {"finished": {}}, 6: {"finished": {"a": {"reason": "stop"}, "c": {"reason": "length"}}}}
    assert RB.served_finish({"a": 10, "b": 1}, outs, steps) == {"a": 13, "c": 1}


def test_shape_view_drops_only_the_dummies_finish_reasons() -> None:
    steps = {4: {"rows": [["t", 0, 1]], "finished": {"t": {"reason": "stop"}, "d": {"reason": "stop"}}}}
    got = RB.shape_view(steps, "t")
    assert got[4]["finished"] == {"t": {"reason": "stop"}, "d": None} and got[4]["rows"] == steps[4]["rows"]
    assert steps[4]["finished"]["d"] == {"reason": "stop"}


class Pool:
    """The fresh pool's prefix cache as `Hits.seed` reads and fills it."""

    def __init__(self, cached: dict[Any, int]) -> None:
        self.cache, self.next, self.freed = dict(cached), 50, []
        self.cached_block_hash_to_block = NS(get_one_block=lambda key: NS(block_id=self.cache[key]) if key in self.cache else None)

    def get_new_blocks(self, n: int) -> list[Any]:
        self.next += 1
        return [NS(block_id=self.next)]

    def _insert_block_hash(self, key: Any, blk: Any, bs: int) -> None:
        self.cache[key] = blk.block_id

    def free_blocks(self, blocks: list[Any]) -> None:
        self.freed += [b.block_id for b in blocks]


@pytest.fixture
def kv_cache_utils(monkeypatch: pytest.MonkeyPatch) -> None:
    mods = {name: ModuleType(name) for name in ("torch", "vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.kv_cache_utils")}
    mods["vllm.v1.core.kv_cache_utils"].make_block_hash_with_group_id = lambda h, g: (h, g)
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)


def test_hits_seed_poisons_a_dummy_s_missing_pages_and_never_lets_one_stand_in_for_the_target(kv_cache_utils: None) -> None:
    pool = Pool({(b"p0", 0): 7, (b"t1", 0): 8})
    kv = Writes()
    taken = {"pages_seeded_poisoned": 0, "target_pages_unavailable": 0, "target_pages_own": 0}
    shape = NS(dummy=lambda rid: rid != "t", mode="zero", taken=taken)
    h = RB.Hits(NS(block_pool=pool), kv, lambda h: None, {}, 4, lambda: 0, shape, lambda h: None)
    h.seed(NS(request_id="d", block_hashes=[b"p0", b"p1"]), 8)
    h.put()
    assert kv.calls == [({51: RB.FILL_PAGE}, "zero")] and pool.cache[(b"p1", 0)] == 51 and pool.freed == [51]
    assert taken["pages_seeded_poisoned"] == 1 and h.loaded == 0
    h.seed(NS(request_id="t", block_hashes=[b"t0", b"t1"]), 8)
    h.put()
    assert taken["target_pages_unavailable"] == 2 and h.loaded == 0 and (b"t0", 0) not in pool.cache and len(kv.calls) == 1


class Writes:
    """A K/V that records `Hits`' writes."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[int, Any], str]] = []

    def write(self, ops: dict[int, Any], mode: str) -> None:
        self.calls.append((dict(ops), mode))


def test_hits_writes_a_step_s_pages_at_once_each_block_s_last_and_takes_the_target_s_own_page_only_if_cached(
        kv_cache_utils: None) -> None:
    pool = Pool({(b"t1", 0): 8})
    kv = Writes()
    taken = {"pages_seeded_poisoned": 0, "target_pages_unavailable": 0, "target_pages_own": 0, "target_pages_loaded": 0,
             "target_pages_overwritten": 0}
    shape = NS(dummy=lambda rid: rid != "t", mode="zero", taken=taken)
    served = {b"t0": ("near", 3), b"t1": ("far", 4)}
    h = RB.Hits(NS(block_pool=pool), kv, served.get, {"t": [(5, 16)]}, 4, lambda: 5, shape,
                lambda x: RB.OWN_PAGE if x in (b"t2", b"t3") else None)
    pool.cache[(b"t2", 0)] = 9
    h.due({"t": NS(request_id="t", block_hashes=[b"t0", b"t1", b"t2", b"t3"])}, 5)
    assert kv.calls == [({51: ("near", 3), 8: ("far", 4)}, "zero")]
    assert taken | {"pages_seeded_poisoned": None} == {"pages_seeded_poisoned": None, "target_pages_unavailable": 1,
                                                       "target_pages_own": 1, "target_pages_loaded": 1,
                                                       "target_pages_overwritten": 1}
    h.ops = {7: ("near", 1)}
    h.ops[7] = RB.FILL_PAGE
    h.put()
    assert kv.calls[-1] == ({7: RB.FILL_PAGE}, "zero") and h.ops == {}


def test_rank_kv_groups_a_step_s_loads_by_dump_fills_the_rest_and_refills_a_dummy_s_slots_by_block() -> None:
    calls: list[tuple[str, tuple]] = []

    def rpc(fn: Any, args: tuple = ()) -> list[Any]:
        calls.append((fn.__name__, args))
        if fn.__name__ == "block_roots":
            return [{"rank": r, "roots": {"a": "x" if r != 2 else "y"}} for r in (1, 0, 2)]
        return [None]

    kv = RB.RankKv(rpc, 4, RB.Path("/e1"), [{"block_bytes": 10}, {"block_bytes": 6}])
    assert kv.block_bytes == 16
    kv.write({5: ("/e1", 50), 6: RB.FILL_PAGE, 7: ("/e2", 70), 8: ("/e1", 80)}, "nan")
    assert calls == [("copy_blocks", ("/e1", [[50, 5], [80, 8]])), ("copy_blocks", ("/e2", [[70, 7]])),
                     ("fill_blocks", ([6], RB.T.FILL))]
    assert kv.loaded_from == {"/e1": 2, "/e2": 1}
    calls.clear()
    kv.fill_slots([9, 10, 3, 17], "zero")
    assert calls == [("fill_blocks", ([0, 2, 4], 0))]
    calls.clear()
    kv.load([(11, 1), (12, 2)], [(2, 3)], "copy")
    assert calls == [("copy_blocks", ("/e1", [[11, 1], [12, 2]]))]
    with pytest.raises(SystemExit, match="--tail copy"):
        kv.load([], [], "zero")
    calls.clear()
    assert kv.roots({"a": ([1, 2, 3], 9)}) == {"a": "x"}
    assert calls == [("block_roots", ({"a": [1, 2]},))] and kv.ranks_disagree == [{"requests": 1, "ranks": [2]}]


def test_served_pins_are_set_before_vllm_loads_and_refused_once_inductor_has_read_its_own(tmp_path, monkeypatch) -> None:
    import json

    pins = {"PYTHONHASHSEED": "0", "TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK": "0", "VLLM_USE_V2_MODEL_RUNNER": "1"}
    (tmp_path / "meta.json").write_text(json.dumps({"pins": pins}))
    env: dict[str, str] = {}
    monkeypatch.setattr(RB, "os", NS(environ=env))
    monkeypatch.delitem(sys.modules, "torch._inductor.config", raising=False)
    monkeypatch.setattr(RB, "sys", NS(flags=NS(hash_randomization=0), modules=sys.modules))
    assert RB.served_pins(tmp_path) == pins
    assert env == {"TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK": "0", "VLLM_USE_V2_MODEL_RUNNER": "1"}
    monkeypatch.setitem(sys.modules, "torch._inductor.config", ModuleType("torch._inductor.config"))
    with pytest.raises(SystemExit, match="Inductor"):
        RB.served_pins(tmp_path)
    (tmp_path / "meta.json").write_text(json.dumps({"pins": pins | {"PYTHONHASHSEED": "1"}}))
    with pytest.raises(SystemExit, match="PYTHONHASHSEED=0"):
        RB.served_pins(tmp_path)


def test_the_perturbed_row_is_sampled_as_the_block_s_first_arrival_and_never_stops_early() -> None:
    arrivals = [{"engine_id": "a", "sampling": {"top_p": 0.95, "max_tokens": 9}}, {"engine_id": "b", "sampling": {}}]
    row = RB.perturb_row(64, 7, arrivals)
    assert row["before_step"] == 7 and row["prompt_ids"] == [0] * 64
    assert row["sampling"] == {"top_p": 0.95, "max_tokens": 9, "ignore_eos": True} and arrivals[0]["sampling"] == {
        "top_p": 0.95, "max_tokens": 9}
    assert RB.perturb_row(8, 0, [])["sampling"] == {"ignore_eos": True}


def test_a_dummy_served_stopped_on_a_token_gets_the_next_row_and_is_finished_where_serving_finished_it(monkeypatch) -> None:
    steps = {5: {"finished": {"a": {"reason": "length"}, "b": {"reason": "stop", "stop_reason": 7}}},
             6: {"finished": {"t": {"reason": "stop"}}}, 7: {}}
    assert RB.served_stops(steps) == {"b": 5, "t": 6}
    finished = []
    sched = NS(requests={"a": 1, "b": 2, "t": 3}, finish_requests=lambda ids, status: finished.append((list(ids), status)))
    d = RB.Dummies("t", "zero", sched, None, None, 4, [], {"a": 3, "b": 4, "t": 9}, RB.served_stops(steps))
    assert d.stops == {"b": 5}
    assert d.params("a", {"sampling": {}})["max_tokens"] == 3 and d.params("b", {"sampling": {}})["max_tokens"] == 5
    vllm = ModuleType("vllm")
    vllm.v1 = ModuleType("vllm.v1")
    vllm.v1.request = ModuleType("vllm.v1.request")
    vllm.v1.request.RequestStatus = NS(FINISHED_STOPPED="stopped")
    for name, mod in (("vllm", vllm), ("vllm.v1", vllm.v1), ("vllm.v1.request", vllm.v1.request)):
        monkeypatch.setitem(sys.modules, name, mod)
    so4, so5 = object(), object()
    trace = NS(_pending={id(so4): 4, id(so5): 5}, records=[{"kind": "step", "step": 4, "finished": {}},
                                                           {"kind": "step", "step": 5, "finished": {"a": {"reason": "length"}}}])
    wrapped = d.stopper(trace)(lambda so, mro: "outs")
    assert wrapped(so4, None) == "outs" and finished == []
    assert wrapped(so5, None) == "outs" and finished == [(["b"], "stopped")]
    assert d.taken["stops_forced"] == 1
    assert trace.records[0]["finished"] == {} and set(trace.records[1]["finished"]) == {"a", "b"}


def test_the_perturbed_row_jumps_the_waiting_queue() -> None:
    from collections import deque

    class Queue(deque):
        def remove_request(self, r):
            self.remove(r)

        def prepend_request(self, r):
            self.appendleft(r)

    sched = NS(requests={"a": "A", "b": "B", RB.PERTURB_ID: "P"}, waiting=Queue(["A", "B", "P"]))
    RB.jump_queue(sched, RB.PERTURB_ID)
    assert list(sched.waiting) == ["P", "A", "B"]


def test_full_pages_are_the_blocks_a_request_s_computed_tokens_fill() -> None:
    assert RB.full_pages({"a": ([4, 5, 6], 9), "b": ([7], 3), "c": ([1, 2], 8)}, 4) == {"a": [4, 5], "b": [], "c": [1, 2]}


def test_a_served_preempted_request_is_resumed_and_any_other_stays_new() -> None:
    import enum

    class Status(enum.IntEnum):
        WAITING = 1
        PREEMPTED = 3

    q, w = NS(status=Status.WAITING), NS(status=Status.WAITING)
    RB.keep_preempted(q, "PREEMPTED"), RB.keep_preempted(w, "RUNNING")
    assert (q.status, w.status) == (Status.PREEMPTED, Status.WAITING)


@pytest.mark.parametrize("args", [["--served", "S", "--scratch", "D"],
                                  ["--arm-dir", "A", "--kv-dir", "K", "--shape-only", "r0"]])
def test_capture_takes_only_a_served_shape_only_target_s_rows(args: list[str]) -> None:
    with pytest.raises(SystemExit, match="--capture takes the sampler rows of --served's --shape-only target"):
        RB.main([*args, "--capture", "TokenSelect_v4{V=8}", "--out", "O"])
