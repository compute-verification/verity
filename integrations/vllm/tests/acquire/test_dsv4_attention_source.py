"""The DeepSeek-V4 attention source on CPU (`verity_vllm/acquire/sources/dsv4_attention_source.py`) against the DeepSeek-V4-Pro
capture map: the stand-in of `tests/acquire/dsv4_model.py` runs the pinned vLLM's attention call structure (its methods, the
call-time globals, the paged caches in their layouts, the compressors' masks) on a short decode, a scored decode, a prefill with a
token that has no compressed slot, a mixed step and a decode that crosses no 128 boundary, on and off the target, and a real
`NativeHostCommitter`, whose `_acquire` records a copy on CPU, commits what the source binds.

It catches a name the map lists that a step does not bind, binds twice, or binds off its route or off the target; a value from the
wrong call, layer, rows or cache layout (a padded paged cache's data and scale bytes regrouped per row); a compressed row of a token
the kernels' mask skips; a global called outside a layer's forward; and a step the route source admits without its names, among
them a step on which a compressor wrote no row."""
from __future__ import annotations

import sys
import types
from collections import Counter
from typing import Any

import pytest
import torch

from tests.acquire.dsv4_model import (CMAP, CSA, CSA_R, ENV, HCA_R, IKB, L, LONG, NHL, NHP, SWAB, TOPK, _paged, dsv4_model, metadata,
                                      run_step)
from tests.acquire.test_dsv4_route_source import Run, map_names
from verity_vllm.acquire.sources import dsv4_attention_source as DS
from verity_vllm.acquire.sources import route_source as RS
from verity_vllm.commit import capture_map as CM

OTHERS = tuple(b for b in RS.BINDERS if b != DS.BINDER)
TAPS_ = [t for t in CMAP["taps"] if (t.get("commit") or {}).get("binder") == DS.BINDER]
#: committed name -> (role, layer)
ROLE_OF = {CM.at_layer(t["commit"]["name"], i): (DS.role_of(t), i) for t in TAPS_ for i in CM.tap_layers(CMAP, t, ENV)}
HCA = [i for i in range(L) if i not in CSA]
#: (metadata, first position, tokens with no compressed slot, routes, phase)
STEPS = [(metadata(decodes=4), 124, (), ("decode-tokens", "hc-fused", "indexer-short"), "decode"),
         (metadata(decodes=4, max_seq_len=LONG), 252, (), ("decode-tokens", "hc-fused", "indexer-scored"), "decode"),
         (metadata(prefill_tokens=20, max_seq_len=LONG), 108, (3,), ("hc-split", "indexer-scored", "prefill-tokens"), "prefill"),
         (metadata(decodes=2, prefill_tokens=18, max_seq_len=LONG), 236, (),
          ("decode-tokens", "hc-split", "indexer-scored", "prefill-tokens"), "prefill"),
         (metadata(decodes=4), 0, (), ("decode-tokens", "hc-fused", "indexer-short"), "decode")]
#: the step on which no HCA compressor writes a row (positions 0..3)
NO_C128 = 4


def as_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else t


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(as_bytes(a), as_bytes(b))


def value(model: Any, role: str, i: int, d: int) -> torch.Tensor:
    """What the stand-in computed for `role` at layer `i` on its last step, with `d` decode rows."""
    a = model.model.layers[i].attn
    s = a.seen
    return {
        "attn.qr": lambda: s["qr"], "attn.kv": lambda: s["kv"], "attn.q": lambda: s["q"], "attn.swa_rows": lambda: s["swa_rows"],
        "compressor.kv_score": lambda: s["kv_score"], "compressor.state": lambda: a.compressor.seen["state"],
        "attn.compressed_rows": lambda: a.compressor.seen["compressed"], "indexer.compressor.kv_score": lambda: s["indexer_kv_score"],
        "indexer.compressor.state": lambda: a.indexer.compressor.seen["state"],
        "indexer.k_cache": lambda: a.indexer.compressor.seen["compressed"], "indexer.topk_short": lambda: a.indexer.seen["topk"],
        "indexer.q_fp8": lambda: a.indexer.seen["q"], "indexer.weights": lambda: a.indexer.seen["weights"],
        "indexer.topk": lambda: a.indexer.indexer_op.seen["topk"],
        "attn.swa_indices": lambda: s["swa_indices"][:d].reshape(d, -1), "attn.swa_lens": lambda: s["swa_lens"][:d],
        "attn.extra_indices": lambda: s["extra_indices"][:d].reshape(d, -1), "attn.extra_lens": lambda: s["extra_lens"][:d],
        "attn.out.decode": lambda: s["out"][:d, :NHL], "attn.out.prefill": lambda: s["out"][d:, :NHL],
        "attn.o_fp8": lambda: a.wo_a.seen["o_fp8"], "attn.o_scale": lambda: a.wo_a.seen["o_scale"], "attn.o_einsum": lambda: a.wo_a.seen["z"],
    }[role]()


class AttnRun(Run):
    """`Run` with the attention source attached, recording each step's bound copies, the committer's acquire/flush order and, per
    bound name, the value the stand-in computed for it."""

    def __init__(self, target: bool = True) -> None:
        super().__init__(target)
        self.attn = DS.Dsv4AttentionSource(self.com, self.src, self.model)
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
            for k, (md, first, unslotted, _routes, _phase) in enumerate(STEPS):
                r.step(k, md, first, unslotted)
        finally:
            r.close()
        out[target] = r
    return out


def test_the_source_wraps_every_attention_layer_and_the_four_globals():
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
        assert set(off.attn.homes) == set(DS.GLOBALS) - {DS.EINSUM}, "off the target no o_einsum is bound"
    finally:
        off.close()


def test_every_step_binds_the_map_names_of_its_routes_once(runs, capsys):
    hca_rows = {f"model.layers.{i}.attn/compressed_rows" for i in HCA}
    for target, r in runs.items():
        assert r.attn.empty == {NO_C128: sorted(hca_rows, key=lambda n: int(n.split(".")[2]))}, target
        for k, (_md, _first, _unslotted, routes, phase) in enumerate(STEPS):
            want = set(CM.committer_names(CMAP, ENV, phase=phase, routes=routes, rank=3, binders=(DS.BINDER,), on_target=target))
            assert Counter(r.names(k)) == Counter(want - set(r.attn.empty.get(k, ()))), (target, k)
            if target:
                assert set(map_names(capsys, routes, phase, binders=(DS.BINDER,))) == want
    on, off = runs[True], runs[False]
    assert len(on.names(0)) == 15 * L + 4 * len(CSA) == 1035 and len(on.names(NO_C128)) == 1035 - len(HCA)
    assert len(on.names(1)) - len(off.names(1)) == 2 * L + 3 * len(CSA), "FlashMLA's output, the einsum and the scored indexer's"
    assert on.attn.stats["unpaired"] == on.attn.stats["layout_refused"] == on.attn.stats["unattributed"] == 0


def test_each_bound_value_is_the_value_it_names(runs):
    for r in runs.values():
        for k in range(len(STEPS)):
            for name, t in r.rec[k]:
                assert same(t, r.want[k][name]), (k, name)
    r = runs[True]
    rows = {k: (r.bound(k, "model.layers.2.attn/compressed_rows").shape[0], r.bound(k, "model.layers.2.attn.indexer.k_cache/rows").shape[0],
                r.bound(k, "model.layers.3.attn/compressed_rows").shape[0]) for k in range(NO_C128)}
    assert rows == {0: (1, 1, 1), 1: (1, 1, 1), 2: (4, 4, 1), 3: (5, 5, 1)}, "the prefill's token 3 has no compressed slot"
    assert r.bound(2, "model.layers.2.attn/compressed_rows").shape[1] == SWAB and r.bound(2, "model.layers.2.attn.indexer.k_cache/rows").shape[1] == IKB
    swa = r.model.model.layers[5].attn.swa_cache_layer.kv_cache
    assert swa.stride(0) > swa.shape[1] * SWAB, "the stand-in's caches have a padded block stride"
    s = int(7 * (STEPS[NO_C128][1] + 3) + 3)
    naive = swa[s // swa.shape[1], s % swa.shape[1]]
    assert not torch.equal(naive, r.bound(NO_C128, "model.layers.5.attn.swa_cache_layer/rows")[3]), "a row is not a [slot] read"
    assert r.bound(3, "model.layers.7.attn/out_decode").shape == (2, NHL, 8) and r.bound(3, "model.layers.7.attn/out_prefill").shape[0] == 18
    assert r.bound(0, "model.layers.4.attn/q").shape[1] == NHP and r.bound(0, "model.layers.3.attn/extra_indices").shape == (4, 128)
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
        md, first, unslotted, routes, phase = STEPS[0]
        r.step(0, md, first, unslotted)
        gone = "model.layers.5.attn.swa_cache_layer/rows"
        assert gone not in r.names(0) and r.attn.stats["layout_refused"] == 1
        got = r.check({0: map_names(capsys, routes, phase, binders=OTHERS) + r.names(0)})
        assert got["ok"] is False and got["violations"][0]["routes_bad"] == {gone: (0, 1)}
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


def test_the_route_source_refuses_a_step_whose_compressor_wrote_no_row(runs, capsys):
    """Proposal B of the DSV4-PREP report would hold such a step to the map's names less the ones `empty` records; until then the
    identity check refuses it by those names, and admits every other step."""
    r = runs[True]
    steps = {k: map_names(capsys, routes, phase, binders=OTHERS) + r.names(k) for k, (_md, _f, _u, routes, phase) in enumerate(STEPS)}
    got = r.check(steps)
    assert got["ok"] is False and [v["step"] for v in got["violations"]] == [NO_C128]
    v = got["violations"][0]
    assert v["routes_bad_n"] == len(HCA) and v["routes_bad"] == {n: (0, 1) for n in sorted(r.attn.empty[NO_C128])[:8]}
    got = r.check({k: names for k, names in steps.items() if k != NO_C128})
    assert got["ok"] is True, got["violations"][:2]
    gone = "model.layers.9.attn/o_scale"
    got = r.check({**steps, 0: [n for n in steps[0] if n != gone]})
    assert [x["step"] for x in got["violations"]][:1] == [0] and got["violations"][0]["routes_bad"] == {gone: (0, 1)}


def test_the_arena_bound_counts_every_name_with_nx_at_the_envelope_s_max_model_len():
    e = ENV

    def per_layer(csa: bool, target: bool, nx: int) -> int:
        width = (4 if csa else 2) * e["HD"]
        b = 2 * e["QL"] + 2 * e["HD"] + 2 * NHP * e["HD"] + SWAB + 2 * 4 * width + SWAB
        b += 4 * e["SW"] + 4 + 4 * (TOPK if csa else nx) + 4 + e["OGL"] * e["HPG"] * e["HD"] + 4 * e["OGL"] * e["HPG"] * e["HD"] // 128
        if csa:
            b += 2 * 4 * 4 * e["ID"] + IKB + 4 * TOPK
        if target:
            b += 2 * 2 * NHL * e["HD"] + 2 * e["OGL"] * e["OL"] + (e["IH"] * e["ID"] + 4 * e["IH"] + 4 * TOPK if csa else 0)
        return b

    for target in (True, False):
        r = AttnRun(target)
        try:
            for envelope, nx in (({"max_model_len": 1 << 20}, 8192), ({"max_prompt": 900, "max_new": 100}, 128)):
                got = r.attn.extra_step_bytes(envelope, r.model)
                want = sum(per_layer(i in CSA, target, nx) for i in range(L))
                n = (16 * L + 7 * len(CSA)) if target else (13 * L + 4 * len(CSA))
                assert got == {"per_token_bytes": want, "per_seq_bytes": 0, "tensors_per_step": n}, (target, envelope)
        finally:
            r.close()


def test_route_source_attaches_the_attention_source():
    assert DS.BINDER in RS.BINDERS and DS.binds(CMAP) and RS.unbuilt(CMAP) == set()
    assert {DS.role_of(t) for t in TAPS_} == DS.ROLES and len(DS.ROLES) == 23
    with pytest.raises(ValueError, match="binds no"):
        DS.Dsv4AttentionSource(None, types.SimpleNamespace(cmap={**CMAP, "taps": [*CMAP["taps"], {**TAPS_[0], "id": "attn.other"}]},
                                                           env=ENV, on_target=True), dsv4_model())
