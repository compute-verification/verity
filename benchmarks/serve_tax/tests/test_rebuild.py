"""The rebuild's bookkeeping on stand-ins for vLLM's engine: the K/V file's layout, the hits the record forces and when
their pages load, the blocks a re-added request shares, the start's refusal of async scheduling's extra row, and the
per-request K/V roots and the mirror's save and load (numpy, skipped without it)."""
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
        RB.start_state(sched, None, None, None, arrived, served, 8, 10, 4)


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
