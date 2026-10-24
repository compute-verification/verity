"""The DeepSeek-V4 attention source on CPU (`verity_vllm/acquire/sources/dsv4_attention_source.py`) against the DeepSeek-V4-Pro
capture map: the stand-in of `tests/acquire/dsv4_model.py` runs the pinned vLLM's attention call structure (its methods, the
call-time globals, the paged caches in their layouts, the compressors' masks and slots) on a short decode, a scored decode, a
prefill, a mixed step, a decode that crosses no 128 boundary and a prefill in two chunks, the first ending mid-window, on and off
the target, and a real `NativeHostCommitter`, whose `_acquire` records a copy on CPU, commits what the source binds.

It catches a name the map lists that a step does not bind, binds twice, or binds off its route or off the target; a value from the
wrong call, layer, rows or cache layout (a padded paged cache's data and scale bytes regrouped per row); a compressed row of a token
the kernels' mask skips; a global called outside a layer's forward; a layer attending at other positions than the first; a
compressor whose ratio is not its row count's; a step the route source admits without its names; and the row-count rule: a step on
which a compressor writes no row commits without that name, every other compressed-rows name holds the rows its committed
positions give, and a step fails whose names, whose rows under one of them (one short, one extra), or whose source's record of the
names it found empty, are not what its positions give.  A step one row short is a fault the test injects (a boundary token with
no compressed slot, `dsv4_model.step_metadata`'s `unslotted`), which the pinned vLLM does not give."""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import sys
import types
from collections import Counter
from typing import Any

import pytest
import torch

from tests.acquire.dsv4_model import (CMAP, CSA, CSA_R, ENV, HCA_R, HD_S, IKB, L, LONG, NHL, NHP, SW, SWAB, TOPK, _paged, dsv4_model,
                                      metadata, run_step)
from tests.acquire.test_dsv4_route_source import COUNTED, Run, map_names
from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.acquire.sources import dsv4_attention_source as DS
from verity_vllm.acquire.sources import route_source as RS
from verity_vllm.acquire.sources import taps
from verity_vllm.commit import capture_map as CM
from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES
from verity_vllm.commit.committer.plan_cache import PlanMetas
from verity_vllm.query.manifest.bind import Binder

#: nearly every test here builds and serves a DeepSeek-V4 stand-in, seconds to minutes each, so the whole module is slow
pytestmark = pytest.mark.slow

OTHERS = tuple(b for b in RS.BINDERS if b != DS.BINDER)
TAPS_ = [t for t in CMAP["taps"] if (t.get("commit") or {}).get("binder") == DS.BINDER]
#: committed name -> (role, layer)
ROLE_OF = {CM.at_layer(t["commit"]["name"], i): (DS.role_of(t), i) for t in TAPS_ for i in CM.tap_layers(CMAP, t, ENV)}
HCA = [i for i in range(L) if i not in CSA]
DN, DR = ENV["DN"], ENV["DR"]
#: each layer's compressed rows, as their parts' names
ROWS = [f"compress.{k}" for k in range(len(DS.MLA_PARTS))]


def compressed(layers: Any) -> list[str]:
    return [f"model.layers.{i}.attn/{m}" for i in layers for m in ROWS]


#: (metadata, first position, routes, phase)
STEPS = [(metadata(decodes=4), 124, ("decode-tokens", "hc-fused", "indexer-short"), "decode"),
         (metadata(decodes=4, max_seq_len=LONG), 252, ("decode-tokens", "hc-fused", "indexer-scored"), "decode"),
         (metadata(prefill_tokens=20, max_seq_len=LONG), 108, ("hc-split", "indexer-scored", "prefill-tokens"), "prefill"),
         (metadata(decodes=2, prefill_tokens=18, max_seq_len=LONG), 236,
          ("decode-tokens", "hc-split", "indexer-scored", "prefill-tokens"), "prefill"),
         (metadata(decodes=4), 0, ("decode-tokens", "hc-fused", "indexer-short"), "decode")]
#: the prefill at positions 108..127: five CSA rows (111, 115, ..., 127) and one HCA row (127)
PREFILL = 2
#: the step on which no HCA compressor writes a row (positions 0..3)
NO_C128 = 4


def as_bytes(t: torch.Tensor) -> torch.Tensor:
    """A float tensor's bytes: a row part read as bf16 out of a byte cache can hold NaN patterns."""
    return t.view(torch.uint8) if t.is_floating_point() else t


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(as_bytes(a), as_bytes(b))


#: an fp8_ds_mla row's parts: (first byte, end byte, dtype)
PARTS = {"nope": (0, DN, torch.uint8), "rope": (DN, DN + 2 * DR, torch.bfloat16), "scale": (DN + 2 * DR, SWAB, torch.uint8)}


def value(model: Any, role: str, i: int, d: int) -> torch.Tensor:
    """What the stand-in computed for `role` at layer `i` on its last step, with `d` decode rows: a row part its slice of the row, a
    state its score half."""
    cache, _, part = role.rpartition(".")
    if part in PARTS:
        lo, hi, dtype = PARTS[part]
        return value(model, cache, i, d)[:, lo:hi].view(dtype)
    a = model.model.layers[i].attn
    s = a.seen

    def half(rows: torch.Tensor) -> torch.Tensor:
        return rows[:, rows.shape[1] // 2:]

    return {
        "attn.qr": lambda: s["qr"], "attn.kv": lambda: s["kv"], "attn.q": lambda: s["q"], "attn.swa_rows": lambda: s["swa_rows"],
        "compressor.kv_score": lambda: s["kv_score"], "compressor.state": lambda: half(a.compressor.seen["state"]),
        "attn.compressed_rows": lambda: a.compressor.seen["compressed"], "indexer.compressor.kv_score": lambda: s["indexer_kv_score"],
        "indexer.compressor.state": lambda: half(a.indexer.compressor.seen["state"]), "attn.gather": lambda: s["gather"],
        "indexer.k_cache": lambda: a.indexer.compressor.seen["compressed"], "indexer.topk_short": lambda: a.indexer.seen["topk"],
        "indexer.q_fp8": lambda: a.indexer.seen["q"], "indexer.weights": lambda: a.indexer.seen["weights"],
        "indexer.topk": lambda: a.indexer.indexer_op.seen["topk"],
        "attn.swa_indices": lambda: s["swa_indices"][:d].reshape(d, -1), "attn.swa_lens": lambda: s["swa_lens"][:d],
        "attn.extra_indices": lambda: s["extra_indices"][:d].reshape(d, -1), "attn.extra_lens": lambda: s["extra_lens"][:d],
        "attn.out.decode": lambda: s["out"][:d, :NHL], "attn.out.prefill": lambda: s["out"][d:, :NHL],
        "attn.o_fp8": lambda: a.wo_a.seen["o_fp8"], "attn.o_scale": lambda: a.wo_a.seen["o_scale"], "attn.o_einsum": lambda: a.wo_a.seen["z"],
        "attn.positions": lambda: s["positions"],
    }[role]()


class AttnRun(Run):
    """`Run` with the attention source attached, recording each step's bound copies, the committer's acquire/flush order and, per
    bound name, the value the stand-in computed for it."""

    def __init__(self, target: bool = True) -> None:
        super().__init__(target)
        self.attn = DS.Dsv4AttentionSource(self.com, self.src, self.model)
        self.src.rows = self.attn
        self.attn.install()
        self.rec: dict[int, list[tuple[str, torch.Tensor]]] = {}
        self.log: list[tuple[str, str]] = []
        self.want: dict[int, dict[str, torch.Tensor]] = {}
        self.com._acquire = self._record
        self.com.flush_after_plane = self._flush

    def _record(self, name: str, t: torch.Tensor, job: Any) -> None:
        self.rec.setdefault(int(job.step), []).append((name, t.clone()))
        self.log.append(("acquire", name))

    def _flush(self, what: str) -> bool:
        self.log.append(("flush", what))
        return False

    def step(self, k: int, md: Any, first: int = 0, unslotted: tuple[int, ...] = ()) -> None:
        self.com._cur = types.SimpleNamespace(step=k)
        self.attn.on_step_begin()
        try:
            run_step(self.model, md, first, unslotted=unslotted)
        finally:
            self.com._cur = None
        self.want[k] = {n: value(self.model, *ROLE_OF[n], md.num_decode_tokens) for n, _t in self.rec.get(k, [])}

    def names(self, k: int) -> list[str]:
        return [n for n, _t in self.rec.get(k, [])]

    def check(self, steps: dict[int, list[str]], *, sources: list[Any] | None = None,
              shapes: dict[int, dict[str, tuple[int, ...]]] | None = None) -> dict[str, Any]:
        """`Run.check` with every name the source bound at the shape it bound, and the names `shapes` gives."""
        bound = {k: {n: tuple(t.shape) for n, t in self.rec.get(k, [])} for k in steps}
        return super().check(steps, sources=sources, shapes={k: {**bound[k], **(shapes or {}).get(k, {})} for k in steps})

    def bound(self, k: int, name: str) -> torch.Tensor:
        return next(t for n, t in self.rec[k] if n == name)

    def close(self) -> None:
        self.attn.uninstall()
        self.src.uninstall()


@pytest.fixture(scope="module")
def runs() -> dict[bool, AttnRun]:
    out = {}
    for target in (True, False):
        r = AttnRun(target)
        try:
            for k, (md, first, _routes, _phase) in enumerate(STEPS):
                r.step(k, md, first)
        finally:
            r.close()
        out[target] = r
    return out


def test_the_source_wraps_every_attention_layer_and_its_globals():
    r = AttnRun()
    home = sys.modules["tests.acquire.dsv4_model"]
    a = r.model.model.layers[4].attn
    wrapped = {DS.FORWARD, DS.PROJECTIONS, DS.INSERT, DS.ATTEND}
    try:
        assert r.attn.homes == {f: [home] for f in DS.GLOBALS} and sorted(r.attn.layers) == list(range(L))
        assert wrapped <= set(vars(a)) and all(getattr(home, f).__name__ == "call" for f in DS.GLOBALS)
        assert {k: v for k, v in r.attn.ratio.items() if k[1] in (3, 4)} == {
            ("attn.compressed_rows", 3): HCA_R, ("attn.compressed_rows", 4): CSA_R, ("indexer.k_cache", 4): CSA_R}
        assert r.attn.owners[("indexer.k_cache", 4)] is a.indexer.k_cache and r.attn.owners[("attn.compressed_rows", 4)] is a
    finally:
        r.close()
    assert all(getattr(home, f).__name__ == f for f in DS.GLOBALS) and not wrapped & set(vars(a))
    off = AttnRun(target=False)
    try:
        assert set(off.attn.homes) == set(DS.GLOBALS) - {DS.EINSUM, DS.SPARSE}, "off the target no o_einsum or gather is bound"
    finally:
        off.close()


def test_every_step_binds_the_map_names_of_its_routes_once(runs, capsys):
    for target, r in runs.items():
        assert r.attn.empty == {NO_C128: compressed(HCA)}, target
        for k, (_md, _first, routes, phase) in enumerate(STEPS):
            want = set(CM.committer_names(CMAP, ENV, phase=phase, routes=routes, rank=3, binders=(DS.BINDER,), on_target=target))
            assert Counter(r.names(k)) == Counter(want - set(r.attn.empty.get(k, ()))), (target, k)
            if target:
                assert set(map_names(capsys, routes, phase, binders=(DS.BINDER,))) == want
    on, off = runs[True], runs[False]
    assert len(on.names(0)) == 19 * L + 4 * len(CSA) + 1 == 1280 and len(on.names(NO_C128)) == 1280 - 3 * len(HCA)
    assert len(on.names(PREFILL)) - len(off.names(PREFILL)) == 3 * L + 3 * len(CSA), "the einsum, the gather and the scored indexer's"
    assert all(r.attn.positions == {k: tuple(range(first, first + md.num_actual_tokens)) for k, (md, first, *_x) in enumerate(STEPS)}
               for r in runs.values())
    assert len(on.names(1)) - len(off.names(1)) == 2 * L + 3 * len(CSA), "FlashMLA's output, the einsum and the scored indexer's"
    assert on.attn.stats["unpaired"] == on.attn.stats["layout_refused"] == on.attn.stats["unattributed"] == 0


def test_each_bound_value_is_the_value_it_names(runs):
    for r in runs.values():
        for k in range(len(STEPS)):
            for name, t in r.rec[k]:
                assert same(t, r.want[k][name]), (k, name)
    r = runs[True]
    rows = {k: (r.bound(k, "model.layers.2.attn/compress.0").shape[0], r.bound(k, "model.layers.2.attn.indexer.k_cache/rows").shape[0],
                r.bound(k, "model.layers.3.attn/compress.0").shape[0]) for k in range(NO_C128)}
    assert rows == {0: (1, 1, 1), 1: (1, 1, 1), PREFILL: (5, 5, 1), 3: (5, 5, 1)}, "the prefill writes a CSA row at 111, 115, ..., 127"
    parts = [r.bound(PREFILL, f"model.layers.2.attn/compress.{k}") for k in range(3)]
    assert [(t.shape[1], t.dtype) for t in parts] == [(DN, torch.uint8), (DR, torch.bfloat16), (SWAB - DN - 2 * DR, torch.uint8)]
    assert r.bound(2, "model.layers.2.attn.indexer.k_cache/rows").shape[1] == IKB
    assert r.bound(0, "model.layers.2.attn.compressor/out").shape[1] == 2 * ENV["HD"], "a state row's score half"
    swa = r.model.model.layers[5].attn.swa_cache_layer.kv_cache
    assert swa.stride(0) > swa.shape[1] * SWAB, "the stand-in's caches have a padded block stride"
    s = int(7 * (STEPS[NO_C128][1] + 3) + 3)
    naive = swa[s // swa.shape[1], s % swa.shape[1]]
    assert not torch.equal(naive[:DN], r.bound(NO_C128, "model.layers.5.attn/kv_insert.nope")[3]), "a row is not a [slot] read"
    assert r.bound(3, "model.layers.7.attn/out_decode").shape == (2, NHL, 8) and r.bound(3, "model.layers.7.attn/out_prefill").shape[0] == 18
    assert r.bound(0, "model.layers.4.attn/kv_insert.q").shape[1] == NHP and r.bound(0, "model.layers.3.attn/extra_indices").shape == (4, 128)
    gathered = {i: r.bound(PREFILL, f"model.layers.{i}.attn/gather").shape for i in (2, 3)}
    assert gathered == {2: (SW + 128 // CSA_R, HD_S), 3: (SW + 128 // HCA_R, HD_S)}, "positions 108..127: the window 0..127 and its rows"
    o_fp8 = r.model.model.layers[4].attn.wo_a.seen["o_fp8"]
    assert not DS.pitched_2d(o_fp8) and not DS.pitched_2d(r.model.model.layers[4].attn.wo_a.seen["o_scale"])
    assert r.bound(NO_C128, "model.layers.4.attn/o_fp8").is_contiguous(), "a value the native collector cannot stage is bound copied"
    assert all(r.attn.stats["copied_by_tap"][role] == len(STEPS) * L for role in ("attn.o_fp8", "attn.o_scale"))
    assert set(r.attn.stats["copied_by_tap"]) == {"attn.o_fp8", "attn.o_scale"}, r.attn.stats["copied_by_tap"]


def test_split_rows_and_state_rows_read_their_layout_only():
    c = _paged(2, 4, SWAB)
    flat = c.view(2, 4 * SWAB)
    flat[1, 2 * 576:3 * 576] = 7
    flat[1, 4 * 576 + 2 * 8:4 * 576 + 3 * 8] = 9
    got = DS.split_rows(c, torch.tensor([6]), SWAB, 8, block_size=4)
    assert got.shape == (1, SWAB) and got[0, :576].eq(7).all() and got[0, 576:].eq(9).all()
    s = torch.tensor([1, 6])
    assert DS.split_rows(c, s, SWAB, 8, block_size=8) is None and DS.split_rows(c[:, :, :-1], s, SWAB, 8) is None
    assert DS.split_rows(c.float(), s, SWAB, 8) is None and DS.split_rows(c.transpose(1, 2), s, SWAB, 8) is None
    st = _paged(2, 4, 16, torch.float32)
    st[1, 2] = 3.0
    assert DS.state_rows(st, s, 16, 4).tolist()[1] == [3.0] * 16
    assert DS.state_rows(st, s, 16, 8) is None and DS.state_rows(st.double(), s, 16, 4) is None and DS.state_rows(st, s, 8, 4) is None


def test_a_cache_in_another_layout_is_not_bound_and_the_step_fails_by_its_name(capsys):
    r = AttnRun()
    try:
        r.model.model.layers[5].attn.swa_cache_layer.kv_cache = _paged(6, 16, SWAB, torch.float32)
        md, first, routes, phase = STEPS[0]
        r.step(0, md, first)
        gone = [f"model.layers.5.attn/kv_insert.{p}" for p in DS.MLA_PARTS]
        assert not set(gone) & set(r.names(0)) and r.attn.stats["layout_refused"] == 1
        got = r.check({0: map_names(capsys, routes, phase, binders=OTHERS) + r.names(0)})
        assert got["ok"] is False and got["violations"][0]["routes_bad"] == dict.fromkeys(sorted(gone), (0, 1))
    finally:
        r.close()


def test_the_top_k_buffer_is_flushed_after_each_csa_layer(runs):
    r = runs[True]
    topk = [j for j, (what, n) in enumerate(r.log) if what == "acquire" and n.endswith(("/topk", "/topk_short"))]
    assert len(topk) == len(STEPS) * len(CSA)
    assert all(r.log[j + 1] == ("flush", "indexer.topk" if r.log[j][1].endswith("/topk") else "indexer.topk_short") for j in topk)
    assert sum(1 for e in r.log if e[0] == "flush") == len(topk)
    assert len([j for j, (w, n) in enumerate(runs[False].log) if w == "acquire" and n.endswith("/topk")]) == 0


def test_a_global_called_outside_every_layer_s_forward_fails_the_step():
    r = AttnRun()
    try:
        home = sys.modules["tests.acquire.dsv4_model"]
        r.com._cur = types.SimpleNamespace(step=0)
        before = r.com.x03_expected
        home.fused_q_kv_rmsnorm(torch.zeros(2, 6), torch.zeros(2, 8), torch.ones(6), torch.ones(8), 1e-6)
        r.com._cur = None
        assert r.attn.stats["unattributed"] == 1 and r.rec == {} and r.com.x03_expected == before + 1
        assert r.com.errors[-1].startswith("dsv4_attention: fused_q_kv_rmsnorm called outside")
    finally:
        r.close()


def committed(r: AttnRun, capsys) -> dict[int, list[str]]:
    """Each step's committed names: the other binders' names on its routes, then what the attention source bound."""
    return {k: map_names(capsys, routes, phase, binders=OTHERS) + r.names(k) for k, (_md, _f, routes, phase) in enumerate(STEPS)}


def test_an_hca_decode_step_with_no_compressed_row_commits_without_its_nc128_names(runs, capsys):
    """The decode at positions 0..3 crosses no 128 boundary: the positions it committed give NC128 = 0, so the identity check holds
    it to the map's names less every HCA layer's compressed rows, which its compressors did not write, and admits it."""
    r = runs[True]
    hca_rows = set(compressed(HCA))
    positions = r.attn.positions[NO_C128]
    assert positions == (0, 1, 2, 3) and CM.row_counts(CMAP, ENV, positions) == {"NC4": 1, "NC128": 0}
    _md, _f, routes, phase = STEPS[NO_C128]
    assert r.src.empty_at(tuple(sorted(routes)), phase, positions) == hca_rows == set(r.attn.empty[NO_C128])
    steps = committed(r, capsys)
    assert not hca_rows & set(steps[NO_C128]) and hca_rows <= set(steps[0])
    got = r.check(steps)
    assert got["ok"] is True, got["violations"][:2]
    gone = "model.layers.9.attn/o_scale"
    got = r.check({**steps, 0: [n for n in steps[0] if n != gone]})
    assert [x["step"] for x in got["violations"]] == [0] and got["violations"][0]["routes_bad"] == {gone: (0, 1)}


def test_a_step_the_source_found_empty_where_the_positions_give_rows_fails(capsys):
    """A decode at positions 124..127 whose last token, the step's only 4 and 128 boundary, has no compressed slot (a fault): every
    compressor writes no row, so the source finds every compressed-rows name empty, while the committed positions give one row of
    each.  The identity check fails the step by the reason and by every such name."""
    r = AttnRun()
    try:
        md, first, routes, phase = STEPS[0]
        r.step(0, md, first, unslotted=(3,))
        rows = set(compressed(range(L))) | {f"model.layers.{i}.attn.indexer.k_cache/rows" for i in CSA}
        assert set(r.attn.empty[0]) == rows and CM.row_counts(CMAP, ENV, r.attn.positions[0]) == {"NC4": 1, "NC128": 1}
        got = r.check({0: map_names(capsys, routes, phase, binders=OTHERS) + r.names(0)})
        v = got["violations"][0]
        assert got["ok"] is False and v["routes_bad_n"] == len(rows) + 1
        assert v["routes_bad"][RS.EMPTY_DIFFERS] == (len(rows), 0)
        assert all(c == (0, 1) for n, c in v["routes_bad"].items() if n != RS.EMPTY_DIFFERS)
    finally:
        r.close()


def test_a_step_with_rows_the_positions_give_none_for_fails(runs, capsys):
    """The reverse: a name the step's positions give no row for is refused when present, whether the source bound it (its record
    then differs from the positions' too) or the commit holds it beside the source's record; and a step whose positions were not
    recorded fails by that reason."""
    r = runs[True]
    steps = committed(r, capsys)
    extra = "model.layers.3.attn/compress.0"
    got = r.check({**steps, NO_C128: steps[NO_C128] + [extra]})
    assert [v["step"] for v in got["violations"]] == [NO_C128] and got["violations"][0]["routes_bad"] == {extra: (1, 0)}
    saved = r.attn.positions[0]
    try:
        r.attn.positions[0] = (0, 1, 2, 3)
        got = r.check(steps)
        v = got["violations"][0]
        assert [x["step"] for x in got["violations"]] == [0] and v["routes_bad_n"] == 3 * len(HCA) + 1
        assert v["routes_bad"][RS.EMPTY_DIFFERS] == (0, 3 * len(HCA))
        assert {n: c for n, c in v["routes_bad"].items() if n != RS.EMPTY_DIFFERS} == {n: (1, 0) for n in sorted(compressed(HCA))[:7]}
        del r.attn.positions[0]
        got = r.check(steps)
        assert got["violations"][0]["routes_bad"] == {RS.NO_POSITIONS: (len(steps[0]), 0)}
    finally:
        r.attn.positions[0] = saved


def test_every_compressed_rows_name_holds_its_positions_row_count_and_the_prefill_commits(runs, capsys):
    """Each step binds under every compressed-rows name the rows its committed positions give (`CM.row_counts`), the rows the
    pinned kernels write: the prefill at 108..127 five CSA rows and one HCA row.  The identity check reads each name's committed
    leading dimension and admits every step."""
    for target, r in runs.items():
        for k in range(len(STEPS)):
            counts = CM.row_counts(CMAP, ENV, r.attn.positions[k])
            got = {n: int(t.shape[0]) for n, t in r.rec[k] if n in COUNTED}
            assert got == {n: counts[s] for n, s in COUNTED.items() if counts[s]}, (target, k)
    r = runs[True]
    assert CM.row_counts(CMAP, ENV, r.attn.positions[PREFILL]) == {"NC4": 5, "NC128": 1}
    got = r.check(committed(r, capsys))
    assert got["ok"] is True, got["violations"][:2]


def test_a_prefill_one_row_short_under_its_compressed_rows_names_fails_by_layer_and_name(capsys):
    """The prefill at 108..127 with its first CSA boundary token (111) given no compressed slot, a fault: every CSA compressor
    writes four rows where the positions give five, the source binds the four it wrote and finds nothing empty, and the identity
    check fails the step under `ROW_COUNT_DIFFERS` at every CSA layer's names (its compressed rows' parts and the indexer's rows),
    (4, 5); the HCA rows, at 127, commit."""
    r = AttnRun()
    try:
        md, first, routes, phase = STEPS[PREFILL]
        r.step(PREFILL, md, first, unslotted=(3,))
        short = {n for n, s in COUNTED.items() if s == "NC4"}
        assert {n: int(t.shape[0]) for n, t in r.rec[PREFILL] if n in short} == dict.fromkeys(short, 4) and not r.attn.empty
        got = r.check({PREFILL: map_names(capsys, routes, phase, binders=OTHERS) + r.names(PREFILL)})
        want = {RS.row_count_differs(ROLE_OF[n][1], n): (4, 5) for n in short}
        v = got["violations"][0]
        assert got["ok"] is False and [x["step"] for x in got["violations"]] == [PREFILL] and v["routes_bad_n"] == len(want) == 4 * len(CSA)
        assert v["routes_bad"] == dict(sorted(want.items())[:8])
        assert RS.row_count_differs(10, "model.layers.10.attn/compress.0") in v["routes_bad"]
    finally:
        r.close()


def test_a_compressed_rows_name_one_row_extra_or_with_no_leading_dimension_fails(runs, capsys):
    """A commit that holds one row more than the prefill's positions give under a CSA and an HCA name fails under
    `ROW_COUNT_DIFFERS`, by layer and name, (committed, count); and the commit side fails closed: `split_step_names` given no
    metas fails every row-counted name the step commits, (-1, count).  It reads the leading dimensions from `TensorMeta`s and from
    a cached plan's columns (`PlanMetas`) alike."""
    r = runs[True]
    steps = committed(r, capsys)
    csa, hca = "model.layers.2.attn/compress.0", "model.layers.3.attn/compress.0"
    got = r.check(steps, shapes={PREFILL: {csa: (6, DN), hca: (2, DN)}})
    assert [v["step"] for v in got["violations"]] == [PREFILL]
    assert got["violations"][0]["routes_bad"] == {RS.row_count_differs(2, csa): (6, 5), RS.row_count_differs(3, hca): (2, 1)}
    names = Counter(steps[PREFILL])
    _rest, bad = r.src.split_step_names(names, PREFILL, False)
    assert bad == {RS.row_count_differs(ROLE_OF[n][1], n): (-1, 5 if s == "NC4" else 1) for n, s in COUNTED.items()}
    lead = r.leads(PREFILL)
    shapes = [(lead[n], SWAB) if n in lead else (4, 8) for n in steps[PREFILL]]
    plan = PlanMetas(steps[PREFILL], ["uint8"] * len(shapes), [64] * len(shapes), [64 * i for i in range(len(shapes))], shapes, GPU_CHUNK_BYTES)
    assert RS.leading_dims(plan) == RS.leading_dims(plan.scan()) and RS.leading_dims(plan)[csa] == 5
    assert r.src.split_step_names(names, PREFILL, False, plan)[1] == r.src.split_step_names(names, PREFILL, False, plan.scan())[1] == {}
    assert r.src.split_step_names(Counter(steps[0]), 9, True)[1] == {}, "a padding-only step is held to step 0's set of names only"


def test_a_chunked_prefill_whose_first_chunk_ends_mid_window_holds_the_rows_the_kernels_write(capsys):
    """A 158-token prefill in chunks within the stand-in's 32-token batch, positions 0..29, then 30..61, 62..93, 94..125 and
    126..157: the first ends mid-window of both ratios, so every later chunk starts on compressor state the one before left, which
    the pinned kernels read for the window and never for the mask (`CM.row_counts`).  The first writes 7 CSA rows (3, ..., 27),
    each later one 8 (31, ..., 59; ...; 127, ..., 155), and only the last an HCA row (127): the source binds what the stand-in
    wrote, those counts, and every chunk commits."""
    r = AttnRun()
    chunks = ((0, 30), (30, 32), (62, 32), (94, 32), (126, 32))
    try:
        for k, (first, n) in enumerate(chunks):
            r.step(k, metadata(prefill_tokens=n, max_seq_len=first + n), first)
        assert [CM.row_counts(CMAP, ENV, r.attn.positions[k]) for k in range(len(chunks))] == [
            {"NC4": 7, "NC128": 0}, *[{"NC4": 8, "NC128": 0}] * 3, {"NC4": 8, "NC128": 1}]
        assert r.attn.positions[4] == tuple(range(126, 158)) and set(r.attn.empty) == {0, 1, 2, 3}
        for k in range(len(chunks)):
            counts = CM.row_counts(CMAP, ENV, r.attn.positions[k])
            assert all(same(t, r.want[k][n]) for n, t in r.rec[k])
            assert {n: int(t.shape[0]) for n, t in r.rec[k] if n in COUNTED} == {n: counts[s] for n, s in COUNTED.items() if counts[s]}
            assert r.src.step_routes(k) == (("hc-split", "indexer-short", "prefill-tokens"), "prefill")
        got = r.check({k: map_names(capsys, *r.src.step_routes(k), binders=OTHERS) + r.names(k) for k in range(len(chunks))})
        assert got["ok"] is True, got["violations"][:2]
    finally:
        r.close()


def test_every_layer_attends_at_the_first_layers_positions():
    r = AttnRun()
    try:
        job = types.SimpleNamespace(step=0)
        p = torch.arange(124, 128)
        r.attn.on_step_begin()
        r.attn._bind_positions(0, p, 4, job)
        r.attn._bind_positions(1, p, 4, job)
        assert r.com.errors == [] and r.attn.positions == {0: (124, 125, 126, 127)} and r.attn.stats["by_tap"] == {DS.POSITIONS: 1}
        r.attn._bind_positions(2, p.clone(), 4, job)
        assert r.com.errors == ["dsv4_attention: layer 2 attended at other positions than the step's first layer"]
    finally:
        r.close()


def test_the_source_refuses_a_compressor_whose_ratio_is_not_its_row_counts():
    model = dsv4_model()
    model.model.layers[3].attn.compressor.compress_ratio = CSA_R
    with pytest.raises(ValueError, match="layer 3's attn.compressed_rows compressor has ratio 4, not the ratio of its row count NC128"):
        DS.Dsv4AttentionSource(None, types.SimpleNamespace(cmap=CMAP, env=ENV, on_target=True), model)
    cmap = {**CMAP, "row_counts": {**CMAP["row_counts"], "positions": "attn.q"}}
    with pytest.raises(ValueError, match="row counts read 'attn.q'"):
        DS.Dsv4AttentionSource(None, types.SimpleNamespace(cmap=cmap, env=ENV, on_target=True), dsv4_model())


def test_the_arena_bound_counts_every_name_with_nx_at_the_envelope_s_max_model_len():
    e = ENV

    def per_layer(csa: bool, target: bool, nx: int) -> int:
        width = (4 if csa else 2) * e["HD"]
        b = 2 * e["QL"] + 2 * e["HD"] + 2 * NHP * e["HD"] + SWAB + 4 * width + 4 * width // 2 + SWAB
        b += 4 * e["SW"] + 4 + 4 * (TOPK if csa else nx) + 4 + e["OGL"] * e["HPG"] * e["HD"] + 4 * e["OGL"] * e["HPG"] * e["HD"] // 128
        if csa:
            b += 4 * 4 * e["ID"] + 4 * 2 * e["ID"] + IKB + 4 * TOPK
        if target:
            b += 2 * 2 * NHL * e["HD"] + 2 * e["OGL"] * e["OL"] + 2 * e["HD"] + (e["IH"] * e["ID"] + 4 * e["IH"] + 4 * TOPK if csa else 0)
        return b

    for target in (True, False):
        r = AttnRun(target)
        try:
            for envelope, nx, mml in (({"max_model_len": 1 << 20}, 8192, 1 << 20), ({"max_prompt": 900, "max_new": 100}, 128, 1000)):
                got = r.attn.extra_step_bytes(envelope, r.model)
                want = sum(per_layer(i in CSA, target, nx) for i in range(L)) + 8
                seq = sum(2 * e["HD"] * (e["SW"] - 1 + -(-mml // (CSA_R if i in CSA else HCA_R))) for i in range(L)) if target else 0
                n = (21 * L + 7 * len(CSA) if target else 17 * L + 4 * len(CSA)) + 1
                assert got == {"per_token_bytes": want, "per_seq_bytes": seq, "tensors_per_step": n}, (target, envelope)
        finally:
            r.close()


def test_a_layer_that_gathers_two_prefill_chunks_at_one_step_fails_it():
    """The gathered rows are the rows the tokens read below their lengths, each once, in first-read order; the Program gathers one
    chunk's workspace, so a layer's second `flash_mla_sparse_fwd` at a step fails the step and binds nothing more."""
    r = AttnRun()
    try:
        home = sys.modules["tests.acquire.dsv4_model"]
        r.com._cur = types.SimpleNamespace(step=0)
        r.attn.on_step_begin()
        r.attn._layer = 2
        kv, indices = torch.arange(12.0).view(6, 1, 2), torch.tensor([[[3, 1, 5]], [[1, 4, 3]]])
        for _ in range(2):
            home.flash_mla_sparse_fwd(q=torch.zeros(2, 1), kv=kv, indices=indices, sm_scale=1.0, topk_length=torch.tensor([2, 3]),
                                      out=torch.zeros(2, 1))
        r.com._cur = None
        assert r.names(0) == ["model.layers.2.attn/gather"] and torch.equal(r.bound(0, "model.layers.2.attn/gather"), kv[[3, 1, 4], 0])
        assert r.com.errors == [f"dsv4_attention: layer 2 ran {DS.SPARSE} on two prefill chunks at one step; the Program gathers one"]
    finally:
        r.close()


def test_the_source_declares_the_ports_of_the_calls_it_binds_and_claims_their_call_boundaries():
    ports = DS.ports_of(CMAP)
    assert set(ports) == {"kv_insert.q", "kv_insert.nope", "kv_insert.rope", "kv_insert.scale", "compressor.kv_score", "out_decode",
                          "out_prefill", "o_fp8", "o_scale", "o_einsum"} and all(p.call_boundary for p in ports.values())
    assert [ports[f"kv_insert.{p}"].port for p in ("q", *DS.MLA_PARTS)] == ["0", "1", "3", "2"]
    r = AttnRun()
    try:
        assert r.attn.ports() == ports
    finally:
        r.close()
    row = {"family": CB.FAMILY, "spec": "FlashMlaDecode_v1{NH=16}"}
    assert CB.claimed(row) == DS.BINDER and CB.claimed({**row, "spec": "GemmF32Out_v1{K=7168}"}) is None
    assert CB.claimed({**row, "family": "instance_outputs"}) is None
    man = {"ranks": 8, "fixture": {"model": CMAP["pins"]["checkpoint"]["repo"]}, "identities": [row]}
    assert DS.BINDER in taps.attached(man) and CB.unattached(man, taps.attached(man)) == []


def test_a_layer_s_names_bind_the_identities_of_record(runs):
    """Layer 2's prefill commits bound against its rows as the manifest of record spells them (a Call of the attention's body at
    `<attn>/<fx node>`, a hinted value at `<attn>`): each row binds once, and the rest of what the layer commits is an extra."""
    a = "model.layers.2.attn"
    spec = {f"{a}/dsv4_qnorm_rope_kv_insert_2": ("Dsv4QNormRopeKvInsert_v1", "0123"), a: ("Dsv4MlaCompressC4_v1", "012"),
            f"{a}/mm_4": ("GemmF32Out_v1", ["out"]), f"{a}/dsv4_sparse_attn_2": ("FlashMlaPrefill_v1", ["out"]),
            f"{a}/fused_inv_rope_fp8_quant_kernel_2": ("Dsv4InvRopeQuantHead_v1", "01"),
            f"{a}/deep_gemm_fp8_einsum_2": ("ScaledMmFp8BlockGrouped_v1", ["out"])}
    rows = [{"op_path": op, "output_member": f"compress.{p}" if op == a else p, "spec": s, "port": p, "engine_step": PREFILL}
            for op, (s, ports) in spec.items() for p in ports]
    rows += [{"op_path": a, "output_member": m, "spec": s, "port": "out", "engine_step": PREFILL}
             for m, s in (("gather", "FlashMlaGatherRow_v1"), ("q", "TritonBlockRMSNorm_v1"), ("kv", "TritonBlockRMSNorm_v1"))]
    rows.append({"op_path": f"{a}.compressor", "output_member": "out", "spec": "Dsv4CompressorScore_v1", "port": "out",
                 "engine_step": PREFILL})
    names = [n for n in runs[True].names(PREFILL) if n.startswith((f"{a}/", f"{a}."))]
    got = Binder(rows, ports=DS.ports_of(CMAP)).bind(PREFILL, names)
    bound = Counter((op, m) for op, m, _k, how in got if how != "unbound")
    want = Counter((r["op_path"], r["output_member"]) for r in rows)
    assert bound == want, sorted(set(bound) ^ set(want))
    assert {n for n, (*_x, how) in zip(names, got) if how == "unbound"} == {
        f"{a}.indexer/{m}" for m in ("q_fp8", "weights")} | {f"{a}.indexer.indexer_op/topk", f"{a}.indexer.compressor/kv_score",
                                                             f"{a}.indexer.compressor/out", f"{a}.indexer.k_cache/rows"}


def test_route_source_attaches_the_attention_source():
    assert DS.BINDER in RS.BINDERS and DS.binds(CMAP) and RS.unbuilt(CMAP) == set()
    assert {DS.role_of(t) for t in TAPS_} == DS.ROLES and len(DS.ROLES) == 29
    with pytest.raises(ValueError, match="binds no"):
        DS.Dsv4AttentionSource(None, types.SimpleNamespace(cmap={**CMAP, "taps": [*CMAP["taps"], {**TAPS_[0], "id": "attn.other"}]},
                                                           env=ENV, on_target=True), dsv4_model())
