"""The H200 capture of DeepSeek-V4-Pro's SWA and HCA decode index rows (`dsv4_decode_metadata_capture_gpu`) on the CPU: its steps'
operands, its launch checks and comparison jobs on what the twin says the kernels write, its PTX check, its verdicts and its main
loop with the GPU stood in for."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from tests.properties import dsv4_decode_metadata_capture_gpu as X
from verity_vllm.program.kernels import dsv4_decode_metadata_twin as T
from verity_vllm.properties.record import REGISTRY, digest_of

SMALL = {"swa": "swa-4096", "c128a": "c128-4096"}


def _swa_twin(case: str, o: dict):
    req = o["req"]
    rows, lens, declined = T.swa_indices(X._w(np.arange(req.size)), o["qsl"][req], o["qsl"][req + 1], o["seq"][req],
                                         o["table"][req], o["valid"], **X.statics(case))
    assert not declined.any()
    return rows, lens


def _c128_twin(case: str, kind: str, o: dict):
    nd = o["req"].size
    rows, lens, declined = T.c128a_global_topk(o["pos"][:nd], o["slot"][:nd], o["table"][o["req"]], **X.statics(case, kind))
    assert not declined.any()
    return rows, lens


def simulate_launch(case: str, kind: str, o: dict) -> dict:
    """What `launch` would return were the kernels the twin."""
    S = X.SENTINEL
    if X.PHASE[case] == "swa":
        SW, n = X.statics(case)["SW"], o["req"].size
        rows, lens = _swa_twin(case, o)
        run = {"ws": np.full((n + X.ROWS_PAD, SW), S, np.uint32), "wl": np.full(n + X.LENS_PAD, S, np.uint32)}
        run["ws"][:n], run["wl"][:n] = rows, lens
        d = {"ws": np.full((n + X.WS_OFF + 1, SW), S, np.uint32), "wl": np.full(n + X.LENS_OFF + 4, S, np.uint32)}
        d["ws"][X.WS_OFF:X.WS_OFF + n], d["wl"][X.LENS_OFF:X.LENS_OFF + n] = rows, lens
        return {"runs": [run, copy.deepcopy(run)], "direct": d, "operands": True}
    st, W = X.statics(case, kind), X.statics(case, kind)["W"]
    width, nd, n = X.stride(case), o["req"].size, o["pos"].size
    rows, lens = _c128_twin(case, kind, o)
    pre = X.prefill_rows(o["pos"][nd:], W, st["R"])
    run = {"gd": np.full((n + X.ROWS_PAD, width), S, np.uint32), "dl": np.full(n + X.LENS_PAD, S, np.uint32),
           "pf": np.full((X.PREFILL + X.ROWS_PAD, width), S, np.uint32)}
    run["gd"][:nd, :W], run["dl"][:nd], run["pf"][:n - nd, :W] = rows, lens, pre
    d = {"gd": np.full((n + X.WS_OFF + 1, width), S, np.uint32), "dl": np.full(n + X.LENS_OFF + 4, S, np.uint32),
         "pf": np.full((X.PREFILL + X.WS_OFF + 1, width), S, np.uint32)}
    d["gd"][X.WS_OFF:X.WS_OFF + nd, :W], d["dl"][X.LENS_OFF:X.LENS_OFF + nd] = rows, lens
    d["pf"][X.WS_OFF:X.WS_OFF + n - nd, :W] = pre
    return {"runs": [run, copy.deepcopy(run)], "views": True, "direct": d, "operands": True}


def simulate(case: str, seeds=(7, 8)):
    ops = {kind: X.requests(case, kind, seed) for kind, seed in zip(("served", "edge"), seeds)}
    return ops, {kind: simulate_launch(case, kind, ops[kind]) for kind in ops}


@pytest.fixture(scope="module")
def observed(tmp_path_factory):
    """Each case's operands, the twin's words as the GPU's, and `observe`'s records and quick jobs on them."""
    out = {}
    for case in X.CASES_L:
        ops, rows = simulate(case)
        work = tmp_path_factory.mktemp(case)
        recs, jobs = X.observe(work, case, ops, rows, quick=True)
        out[case] = SimpleNamespace(ops=ops, rows=rows, recs=recs, jobs=jobs)
    return out


def test_the_statics_are_dsv4s() -> None:
    """The window of 128 at the SWA cache's block 64 over cdiv(max_model_len, 64) blocks; ratio 128 at block 256 / 128 over
    cdiv(max_model_len, 256), at the step's width (8192 at 2^20, 128 at 4096, the buffers' c128a_max_compressed) and the edge's
    1024; the registry's served statics; one PTX count per kernel, the sources, 16 served requests (max_num_seqs)."""
    for case, max_len in X.MAX_LEN.items():
        assert X.statics(case) == X.R.DSV4_DECODE_METADATA[case]
        assert len(X.POSITIONS[case]) == 16 and X.POSITIONS[case][0] == 0 and X.POSITIONS[case][-1] == max_len - 1
        assert list(X.POSITIONS[case]) == sorted(set(X.POSITIONS[case]))
        if X.PHASE[case] == "swa":
            assert X.statics(case) == {"SW": 128, "BS": X.SWA_BLOCK, "NB": -(-max_len // X.SWA_BLOCK)}
        else:
            st = X.statics(case)
            assert st == {"W": X.stride(case), "R": X.HCA_R, "BS": X.KERNEL_BLOCK // X.HCA_R, "NB": -(-max_len // X.KERNEL_BLOCK)}
            assert st["W"] == min(max(1 << (max(max_len // X.HCA_R, 1) - 1).bit_length(), 128), X.stride(case))
    assert X.stride("c128") == 8192 and X.stride("c128-4096") == 128
    assert X.statics("c128", "edge")["W"] == 1024 and X.statics("c128-4096", "edge") == X.statics("c128-4096")
    assert set(X.PTX_COUNTS) == set(X.KERNELS) and len(X.SOURCES) == 5 and X.TRITON_BLOCK == 1024


@pytest.mark.parametrize("case", ["swa", "swa-4096"])
def test_the_swa_served_step_holds_what_it_means(case: str) -> None:
    """16 requests of one token each (query_start_loc 0..16, seq_lens position + 1) and two padding tokens (byte 0, request 0);
    every position's window inside its block row, the full window, the row's last block and the empty window's single lane."""
    st = X.statics(case)
    o = X.requests(case, "served", 1)
    nreq = len(X.POSITIONS[case])
    assert o["table"].shape == (nreq, st["NB"]) and o["qsl"].tolist() == list(range(nreq + 1))
    assert o["req"].tolist() == list(range(nreq)) + [0] * X.PADDING and o["valid"].tolist() == [1] * nreq + [0] * X.PADDING
    assert (T.s32(o["seq"]) - 1).tolist() == list(X.POSITIONS[case])
    rows, lens = _swa_twin(case, o)
    lens = T.s32(lens)
    assert lens[:nreq].tolist() == [min(p + 1, st["SW"]) for p in X.POSITIONS[case]] and (lens[nreq:] == 0).all()
    assert (T.s32(rows[nreq:]) == -1).all() and (T.s32(rows[nreq - 1]) >= 0).all()


@pytest.mark.parametrize("case", ["swa", "swa-4096"])
def test_the_swa_edge_step_holds_what_it_means(case: str) -> None:
    """Twelve requests of 1 to 4 tokens in a shuffled map, each request's newest position an edge, so its others are below it
    (negative ones among them, whose lengths are negative); query_start_loc words that wrap int32 between requests; every byte;
    block rows whose products by 64 wrap int32 read by a live lane; no token the twin declines."""
    st = X.statics(case)
    o = X.requests(case, "edge", 2)
    n = sum(X.SWA_EDGE_QLEN)
    assert o["req"].size == n and (np.diff(o["req"]) < 0).any() and set(o["valid"].tolist()) == set(X.EDGE_BYTES)
    qsl = T.s32(o["qsl"])
    assert (np.diff(qsl) < 0).any()
    newest = [int(o["pos"][o["req"] == r].max()) for r in range(len(X.SWA_EDGE_QLEN))]
    assert newest == list(X.swa_edges(case)) and o["pos"].min() < 0
    req = o["req"]
    pos = T.s32(X._w(T.s32(o["seq"][req]) - (T.s32(o["qsl"][req + 1]) - T.s32(o["qsl"][req])) + np.arange(n) - T.s32(o["qsl"][req])))
    assert pos.tolist() == o["pos"].tolist()
    _, lens = _swa_twin(case, o)
    live = (np.arange(st["SW"]) < T.s32(lens)[:, None]) & (o["valid"] != 0)[:, None]
    assert (T.s32(lens) < 0).any() and live.any()
    p = np.maximum(pos - (st["SW"] - 1), 0)[:, None] + np.arange(st["SW"])
    bn = T.s32(o["table"][req][np.arange(n)[:, None], np.where(live, p // st["BS"], 0)])[live]
    assert (bn * st["BS"] > X.I32_MAX).any() and (bn * st["BS"] < X.I32_MIN).any() and (bn < 0).any()


@pytest.mark.parametrize("case", ["c128", "c128-4096"])
def test_the_c128_served_step_holds_what_it_means(case: str) -> None:
    """16 requests at positions from 0 to max_model_len - 1 (an empty row, one one short of the width and a full one), two
    padding tokens (slot -1) and two prefill tokens; every live lane's block among its request's allocated ones."""
    st = X.statics(case)
    o = X.requests(case, "served", 1)
    nreq = len(X.POSITIONS[case])
    assert o["req"].tolist() == list(range(nreq)) + [0] * X.PADDING and o["pos"].size == nreq + X.PADDING + X.PREFILL
    slot = T.s64(o["slot"])
    assert (slot[:nreq] >= 0).all() and (slot[nreq:nreq + X.PADDING] == -1).all()
    _, lens = _c128_twin(case, "served", o)
    lens = lens.astype(np.int64)
    want = [min((p + 1) // st["R"], st["W"]) for p in X.POSITIONS[case]]
    assert lens[:nreq].tolist() == want and (lens[nreq:] == 0).all() and want[0] == 0
    assert max(want) == min(st["W"], max(X.POSITIONS[case]) // st["R"] + 1)


@pytest.mark.parametrize("case", ["c128", "c128-4096"])
def test_the_c128_edge_step_holds_what_it_means(case: str) -> None:
    """Six requests' 42 tokens in a shuffled map; positions at the lane counts' edges and the int64 ends (INT64_MAX's + 1 wrapping
    negative; full rows past the width only where the row holds the width); slots of every sign; block rows whose products by 2
    wrap int32 read by a live lane; no token the twin declines."""
    st = X.statics(case, "edge")
    o = X.requests(case, "edge", 2)
    nd = X.EDGE_REQUESTS * X.EDGE_PER_REQUEST
    assert o["req"].size == nd and (np.diff(o["req"]) < 0).any()
    pos, slot = T.s64(o["pos"][:nd]), T.s64(o["slot"][:nd])
    assert {X.I64_MIN, X.I64_MAX, -1, -st["R"] - 1} <= set(pos.tolist()) and (slot < 0).any() and (slot >= 0).any()
    assert {X.I64_MIN, X.I64_MAX, -1, 0} <= set(slot.tolist())
    _, lens = _c128_twin(case, "edge", o)
    q = np.minimum(np.where(pos + 1 < 0, -1, (pos + 1) // st["R"]), st["W"])
    live = np.arange(st["W"]) < q[:, None]
    assert (q <= 0).any() and (q == min(st["W"], st["BS"] * st["NB"])).any()
    assert (lens.astype(np.int64) == np.where(slot >= 0, np.maximum(q, 0), 0)).all()
    bn = T.s32(o["table"][o["req"]][np.arange(nd)[:, None], np.where(live, np.arange(st["W"]) // st["BS"], 0)])[live]
    assert (bn * st["BS"] > X.I32_MAX).any() and (bn * st["BS"] < X.I32_MIN).any() and (bn < 0).any()


def test_the_prefill_rows_are_the_kernels_prefill_branch() -> None:
    """Lane j < W is j where j < (position + 1) // R (int64, truncating), else -1."""
    pos = X._w64([-1, 0, 126, 127, 128 * 5 - 1, X.I64_MAX, X.I64_MIN, 1 << 40])
    got = T.s32(X.prefill_rows(pos, 8, 128))
    assert got.tolist() == [[-1] * 8, [-1] * 8, [-1] * 8, [0] + [-1] * 7, list(range(5)) + [-1] * 3, [-1] * 8, [-1] * 8,
                            list(range(8))]


def test_observe_passes_the_twins_words(observed) -> None:
    """On the twin's words every launch check holds: the runs agree, the direct launch wrote the launcher's words and nothing
    around them, the HCA views and prefill rows are the kernel's, the operands are unchanged."""
    for case, ob in observed.items():
        assert sorted(x["case"] for x in ob.recs) == sorted([case, f"{case}-edge"])
        for x in ob.recs:
            assert X._launch_ok(x), x
        served = next(x for x in ob.recs if x["case"] == case)
        assert served["padding"] == X.PADDING and served["requests"] == 16


SLIPS = {"swa": ["sentinel", "run_sentinel", "lens_sentinel", "second_run", "direct", "operands"],
         "c128a": ["sentinel", "run_sentinel", "lens_sentinel", "second_run", "direct", "views", "prefill", "operands"]}


@pytest.mark.parametrize("phase,what", [(p, w) for p, ws in SLIPS.items() for w in ws])
def test_the_launch_checks_catch_each_slip(observed, tmp_path, phase: str, what: str) -> None:
    """A word written past the token rows (by the direct launch, or by both launches alike), a length past the tokens, a second launch that differs, a direct launch that differs
    from the launcher, a returned view that is not the buffer's, a prefill row that is not the kernel's prefill branch, and an
    operand written each fail their check."""
    case = SMALL[phase]
    ob = observed[case]
    rows = copy.deepcopy(ob.rows)
    s = rows["served"]
    rk, lk = ("ws", "wl") if phase == "swa" else ("gd", "dl")
    if what == "sentinel":
        s["direct"][rk][0, 3] = 0
    elif what == "run_sentinel":
        for r in s["runs"]:
            r[rk][-1, 3] = 0
    elif what == "lens_sentinel":
        s["direct"][lk][-1] = 0
    elif what == "second_run":
        s["runs"][1][lk][0] ^= 1
    elif what == "direct":
        s["direct"][rk][X.WS_OFF + 2, 5] ^= 1
    elif what == "views":
        s["views"] = False
    elif what == "prefill":
        for r in s["runs"]:
            r["pf"][0, 0] ^= 1
        s["direct"]["pf"][X.WS_OFF, 0] ^= 1
    else:
        s["operands"] = False
    recs, _ = X.observe(tmp_path, case, ob.ops, rows, quick=True)
    failed = {(x["case"], k) for x in recs for k, v in x["checks"].items() if v is not True}
    want = {"sentinel": "untouched", "run_sentinel": "untouched", "lens_sentinel": "untouched", "second_run": "deterministic", "direct": "direct",
            "views": "views", "prefill": "prefill", "operands": "operands"}[what]
    assert failed == {(case, want)}


def test_the_hca_width_check_reads_past_an_adaptive_width(observed, tmp_path) -> None:
    """At the edge's width 1024 of a row 8192 wide, a word written at lane 1024 of a decode row fails `untouched`."""
    ob = observed["c128"]
    rows = copy.deepcopy(ob.rows)
    for r in rows["edge"]["runs"]:
        r["gd"][0, 1024] = 0
    recs, _ = X.observe(tmp_path, "c128", ob.ops, rows, quick=True)
    assert {(x["case"], k) for x in recs for k, v in x["checks"].items() if v is not True} == {("c128-edge", "untouched")}


def _stand_in_vllm(monkeypatch, calls: list) -> None:
    """vLLM's `sparse_swa` and `sparse_mla` stood in for on the CPU: the launchers and the kernels the twin over the tensors they
    are handed, each call's arguments recorded."""
    torch = pytest.importorskip("torch")

    def u32(t):
        return t.contiguous().view(torch.int32).numpy().view(np.uint32)

    def swa_stores(out, lens, W, qsl, seq, req, valid, table, bs, n):
        nb = table.shape[1]
        r = req[:n].numpy()
        rows, ln, declined = T.swa_indices(X._w(np.arange(n)), u32(qsl)[r], u32(qsl)[r + 1], u32(seq)[r], u32(table)[r],
                                           valid.view(torch.uint8).numpy(), W, bs, nb)
        assert not declined.any()
        out.copy_(torch.from_numpy(rows.view(np.int32)).reshape(out.shape))
        lens.copy_(torch.from_numpy(ln.view(np.int32)))

    class SwaLauncher:
        def __call__(self, ws, wl, window, width, left, right, qsl, seq, req, valid, table, bs, *, num_tokens, token_offset):
            calls.append({"op": "swa", "ws": (tuple(ws.shape), ws.stride(0)), "window": window, "width": width,
                          "visible": left is wl and right is wl, "table": (tuple(table.shape), table.stride(0)), "bs": bs,
                          "valid": (valid.dtype, tuple(valid.shape)), "n": num_tokens, "offset": token_offset})
            swa_stores(ws[:num_tokens, 0], wl[:num_tokens], window, qsl, seq, req, valid, table, bs, num_tokens)

    class SwaKernel:
        def __getitem__(self, grid):
            def run(out, out_stride, lens, window, width, left, right, qsl, seq, req, valid, table, table_stride, bs, offset,
                    HAS_IMAGE, TRITON_BLOCK_SIZE):
                calls.append({"op": "swa-kernel", "grid": grid, "strides": (out_stride, table_stride), "window": window,
                              "width": width, "bs": bs, "offset": offset, "image": HAS_IMAGE, "block": TRITON_BLOCK_SIZE})
                swa_stores(out, lens, window, qsl, seq, req, valid, table, bs, grid[0])
            return run

    def c128_stores(gd, gstride, dl, pf, pstride, pos, R, W, nd, req, table, bs, slot):
        n = pos.shape[0]
        nb = table.shape[1]
        p, s = pos.numpy().view(np.uint64), slot.numpy().view(np.uint64)
        rows, ln, declined = T.c128a_global_topk(p[:nd], s[:nd], u32(table)[req[:nd].numpy()], W, R, bs, nb)
        assert not declined.any()
        gd[:nd, :W] = torch.from_numpy(rows.view(np.int32))
        dl[:nd] = torch.from_numpy(ln.view(np.int32))
        pf[:n - nd, :W] = torch.from_numpy(X.prefill_rows(p[nd:], W, R).view(np.int32))

    def c128_launcher(pos, R, nd, req, table, bs, slot, gd, dl, pf, max_compressed_tokens):
        calls.append({"op": "c128", "pos": (pos.dtype, tuple(pos.shape)), "R": R, "nd": nd, "table": (tuple(table.shape),
                      table.stride(0)), "bs": bs, "slot": slot.dtype, "W": max_compressed_tokens, "stride": gd.stride(0)})
        c128_stores(gd, gd.stride(0), dl, pf, pf.stride(0), pos, R, max_compressed_tokens, nd, req, table, bs, slot)
        W, n = max_compressed_tokens, pos.shape[0]
        return gd[:nd, :W], dl[:nd], pf[:n - nd, :W]

    class C128Kernel:
        def __getitem__(self, grid):
            def run(gd, gstride, dl, pf, pstride, pos, R, W, nd, req, table, table_stride, bs, slot, BLOCK_SIZE):
                calls.append({"op": "c128-kernel", "grid": grid, "strides": (gstride, pstride, table_stride), "R": R, "W": W,
                              "nd": nd, "bs": bs, "block": BLOCK_SIZE})
                c128_stores(gd, gstride, dl, pf, pstride, pos, R, W, nd, req, table, bs, slot)
            return run

    sw = types.ModuleType("vllm.v1.attention.backends.mla.sparse_swa")
    sw._COMPUTE_SWA_INDICES_AND_LENS_KERNEL = SwaLauncher()
    sw._compute_swa_indices_and_lens_kernel = SwaKernel()
    sm = types.ModuleType("vllm.models.deepseek_v4.sparse_mla")
    sm.build_c128a_topk_metadata = c128_launcher
    sm._build_c128a_topk_metadata_kernel = C128Kernel()
    for name in ("vllm", "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.backends", "vllm.v1.attention.backends.mla",
                 "vllm.models", "vllm.models.deepseek_v4"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, sw.__name__, sw)
    monkeypatch.setitem(sys.modules, sm.__name__, sm)
    monkeypatch.setattr(X, "DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


def test_launch_hands_the_kernels_the_decodes_operands(tmp_path, monkeypatch) -> None:
    """`launch` on CPU tensors with the kernels the twin: the SWA launcher gets a (tokens + 3, 1, 128) row buffer, the window 128
    as the width, the lengths' buffer as both visible arguments, the [:requests] table of stride NB, block 64, a bool valid of the
    tokens' bytes and token_offset 0, and the kernel one program per token at the launcher's constexprs; the HCA launcher int64
    positions of every token, the decode-token count, the [:requests] table, block 2, int64 slots, the step's width and buffers of
    stride c128a_max_compressed, and its kernel one program per token; `observe` passes every check and every word on what comes
    back."""
    calls: list = []
    _stand_in_vllm(monkeypatch, calls)
    torch = sys.modules["torch"]
    for case in ("swa-4096", "c128-4096", "c128"):
        calls.clear()
        ops = {kind: X.requests(case, kind, seed) for kind, seed in zip(("served", "edge"), (3, 4))}
        rows = {kind: X.launch(case, kind, ops[kind]) for kind in ops}
        for kind, o in ops.items():
            st = X.statics(case, kind)
            nd, nreq = o["req"].size, o["table"].shape[0]
            mine = calls[:3] if kind == "served" else calls[3:]
            if X.PHASE[case] == "swa":
                assert [c["op"] for c in mine] == ["swa", "swa", "swa-kernel"]
                assert mine[0] == {"op": "swa", "ws": ((nd + X.ROWS_PAD, 1, st["SW"]), st["SW"]), "window": st["SW"],
                                   "width": st["SW"], "visible": True, "table": ((nreq, st["NB"]), st["NB"]), "bs": st["BS"],
                                   "valid": (torch.bool, (nd,)), "n": nd, "offset": 0}
                assert mine[2] == {"op": "swa-kernel", "grid": (nd,), "strides": (st["SW"], st["NB"]), "window": st["SW"],
                                   "width": st["SW"], "bs": st["BS"], "offset": 0, "image": False, "block": X.TRITON_BLOCK}
            else:
                n, S = o["pos"].size, X.stride(case)
                assert [c["op"] for c in mine] == ["c128", "c128", "c128-kernel"]
                assert mine[0] == {"op": "c128", "pos": (torch.int64, (n,)), "R": st["R"], "nd": nd,
                                   "table": ((nreq, st["NB"]), st["NB"]), "bs": st["BS"], "slot": torch.int64, "W": st["W"],
                                   "stride": S}
                assert mine[2] == {"op": "c128-kernel", "grid": (n,), "strides": (S, S, st["NB"]), "R": st["R"], "W": st["W"],
                                   "nd": nd, "bs": st["BS"], "block": X.TRITON_BLOCK}
                assert nd != 1 and nd % 16
        recs, jobs = X.observe(tmp_path / case, case, ops, rows, quick=False)
        assert all(X._launch_ok(x) for x in recs) and len(recs) == 2, recs
        for j in ([] if case == "c128" else jobs[:2] + jobs[-2:]):
            r = X.compare(j)
            assert "error" not in r and r["mismatch"] == 0 and r["twin_mismatch"] == 0 and r["reference"] == j["rows"][1] - j["rows"][0], r


def _no_refs(monkeypatch) -> None:
    monkeypatch.setattr(X, "_ref_rows", lambda lo, hi, every: [])


def _flipped(tmp_path, job: dict, key: str, at: tuple) -> dict:
    a = {k: np.array(v) for k, v in X._load(job["paths"]).items()}
    a[key][at] ^= 1
    return {**job, "paths": X._save(tmp_path, **a)}


@pytest.mark.parametrize("case", X.CASES_L)
def test_the_jobs_take_every_word_and_find_a_flipped_one(observed, case: str, tmp_path, monkeypatch) -> None:
    """Every job against the twin (the references off): every token's row and length words, 0 mismatches; a word flipped in a row
    and a length each found at its token and word."""
    _no_refs(monkeypatch)
    ob = observed[case]
    tot: dict = {}
    for j in ob.jobs:
        r = X.compare(j)
        assert "error" not in r, r
        t = tot.setdefault(r["case"], [0, 0])
        t[0] += r["words"]
        t[1] += r["mismatch"]
    width = {kind: X.statics(case, kind)["SW" if X.PHASE[case] == "swa" else "W"] for kind in ("served", "edge")}
    assert tot == {x["case"]: [x["tokens"] * (width["edge" if x["case"].endswith("-edge") else "served"] + 1), 0] for x in ob.recs}
    j = [j for j in ob.jobs if j["case"] == f"{case}-edge"][1]
    lo = j["rows"][0]
    r = X.compare(_flipped(tmp_path / "rows", j, "got_rows", (lo + 1, 7)))
    assert r["mismatch"] == 1 and r["first_mismatch"] == {"token": lo + 1, "word": 7, "want": r["first_mismatch"]["want"],
                                                          "kernel": r["first_mismatch"]["want"] ^ 1}
    r = X.compare(_flipped(tmp_path / "lens", j, "got_lens", (lo + 2,)))
    assert r["mismatch"] == 1 and r["first_mismatch"]["word"] == width["edge"] and r["first_mismatch"]["token"] == lo + 2


def test_the_references_agree_with_the_twin(observed) -> None:
    """The references `Dsv4SwaIndices_v1` and `Dsv4C128aGlobalTopk_v1` on every edge token and on the served steps' first, last
    and padding tokens (the HCA row at 8192 on its first and last) agree with the twin on every word."""
    for case, ob in observed.items():
        tokens = {x["case"]: x["tokens"] for x in ob.recs}
        n = tokens[case]
        picks = {case: [0, n - 1] if case == "c128" else [0, len(X.POSITIONS[case]) - 1, n - 1],
                 f"{case}-edge": list(range(tokens[f"{case}-edge"]))}
        for j in ob.jobs:
            lo, hi = j["rows"]
            for i in (i for i in picks.get(j["case"], []) if lo <= i < hi):
                r = X.compare({**j, "rows": [i, i + 1], "ref_every": 1})
                assert r["reference"] == 1 and r["twin_mismatch"] == 0 and r["mismatch"] == 0, (j["case"], i, r)


def test_the_jobs_reference_every_token_or_a_sample(observed, tmp_path) -> None:
    """A full run references every token; --quick every 16th served one and each job's first, and every edge one."""
    for case, ob in observed.items():
        for j in ob.jobs:
            assert j["ref_every"] == (1 if j["case"].endswith("-edge") else X.QUICK_REF_EVERY), j
        _, full = X.observe(tmp_path / case, case, ob.ops, ob.rows, quick=False)
        assert {j["ref_every"] for j in full} == {1}


@pytest.mark.parametrize("phase", ["swa", "c128a"])
def test_a_token_outside_the_domain_is_an_error(observed, tmp_path, phase: str) -> None:
    """A job holding a token whose live lane reads past its block row (the kernel would read the next request's) is an error, not
    a pass: a window at the row's end, an HCA row wider than the row holds."""
    case = SMALL[phase]
    j = next(j for j in observed[case].jobs if j["case"] == case)
    a = {k: np.array(v) for k, v in X._load(j["paths"]).items()}
    st = X.statics(case)
    if phase == "swa":
        a["seq"][a["req"][j["rows"][0]]] = st["NB"] * st["BS"] + 1
    else:
        a["pos"][j["rows"][0]] = st["W"] * st["R"] + 5
    r = X.compare({**j, "paths": X._save(tmp_path, **a)})
    assert "error" in r and "block row" in r["error"]


def _ptx(counts: dict, target: str = "sm_90a") -> str:
    body = "\n".join(f"\t{op} \t%r{i}, %r1, %r2;" for op, k in counts.items() for i in range(k))
    return f".version 8.7\n.target {target}\n.address_size 64\n{body}\n\tadd.s32 \t%r9, %r1, 4;\n\tld.global.b32 \t%r3, [%rd1];\n"


@pytest.mark.parametrize("slip", [None, "missing", "rem", "swa_no_div", "c128a_no_popc", "md1", "target"])
def test_the_ptx_check(tmp_path, slip) -> None:
    """A file of each kernel with its PTX_COUNTS passes; a kernel's compile missing, a remainder in it, the window's division
    gone, the count without its popc, the HCA kernel's num_decode_tokens == 1 specialization or another target fail."""
    cache = tmp_path / "triton_cache"
    files = {"swa": [dict(X.PTX_COUNTS["swa"])], "c128a": [dict(X.PTX_COUNTS["c128a"])]}
    if slip == "missing":
        files["c128a"] = []
    elif slip == "rem":
        files["swa"].append({**X.PTX_COUNTS["swa"], "rem.s32": 1})
    elif slip == "swa_no_div":
        files["swa"] = [{k: v for k, v in X.PTX_COUNTS["swa"].items() if k != "div.s32"}]
    elif slip == "c128a_no_popc":
        files["c128a"] = [{k: v for k, v in X.PTX_COUNTS["c128a"].items() if k != "popc.b32"}]
    elif slip == "md1":
        files["c128a"].append({**X.PTX_COUNTS["c128a"], "mad.wide.s32": 12, "mul.lo.s32": 10, "sub.s32": 8, "shl.b64": 0})
    i = 0
    for key, counts in files.items():
        for cnt in counts:
            (cache / f"h{i}").mkdir(parents=True)
            text = _ptx({k: v for k, v in cnt.items() if v}, "sm_80" if slip == "target" and i == 0 else "sm_90a")
            (cache / f"h{i}" / f"{X.KERNELS[key]}.ptx").write_text(text)
            i += 1
    rec = X.ptx(cache, tmp_path / "out")
    failed = {k for k, v in rec["checks"].items() if not v}
    want = {None: set(), "missing": {"c128a"}, "rem": {"swa"}, "swa_no_div": {"swa"}, "c128a_no_popc": {"c128a"},
            "md1": {"c128a"}, "target": {"target"}}[slip]
    assert failed == want
    assert len(list((tmp_path / "out" / "ptx").glob("*.ptx"))) == i


def test_the_ptx_counts_are_the_kernels_ops() -> None:
    """The counted ops are the ones the module docstring reads: the window's and the row's `//` by the runtime block size as
    div.s32 and `%` without a remainder; the window's byte as one 8-bit load, its start's max; the row's int64 division, min and
    loads, its count as a popc, a warp reduction and two shuffles."""
    swa, hca = X.PTX_COUNTS["swa"], X.PTX_COUNTS["c128a"]
    assert swa["div.s32"] == hca["div.s32"] == 8 and not {k for k in {**swa, **hca} if k.startswith("rem.")}
    assert (swa["ld.global.b8"], swa["and.b16"], swa["setp.ne.b16"], swa["max.s32"]) == (1, 1, 1, 2)
    assert (hca["div.s64"], hca["div.u32"], hca["min.s64"], hca["ld.global.b64"]) == (1, 1, 1, 2)
    assert (hca["popc.b32"], hca["redux.sync.add.s32"], hca["shfl.sync.bfly.b32"]) == (1, 1, 2)
    assert "popc.b32" not in swa and "ld.global.b8" not in hca
    for c in (swa, hca):
        assert X.ptx_counts(_ptx(c)) == c


def test_the_sources_are_pinned_by_hash(tmp_path) -> None:
    rel = list(X.SOURCES)
    (tmp_path / rel[0]).parent.mkdir(parents=True)
    (tmp_path / rel[0]).write_text("x")
    rec = X.sources(tmp_path)
    assert rec["ok"] is False and rec["files"][rel[0]] is False and rec["files"][rel[1]] is None
    assert hashlib.sha256(b"x").hexdigest() not in X.SOURCES.values()


def test_the_programs_bind() -> None:
    rec = X.bind_check()
    assert rec["ok"], rec
    assert len(rec["bindings"]) == len(X.CASES_L) + len(X.EDGE_W)


def _doc() -> dict:
    launches = [{"phase": p, "case": c, "checks": {"a": True, "b": True}} for p, c in X.CASES]
    comps = [{"phase": p, "case": c, "words": 10, "mismatch": 0, "reference": 1, "twin_mismatch": 0, "first_mismatch": None}
             for p, c in X.CASES]
    return {"schema": X.SCHEMA, "gate": {"ok": True}, "errors": {}, "selection": {"ok": True}, "ptx": {"ok": True},
            "bind": {"ok": True}, "launches": launches, "comparisons": comps, "totals": {k: sum(r[k] for r in comps) for k in X.TOTALS}}


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("selection", "ok"), False), (("ptx", "ok"), False),
                                        (("bind", "ok"), False), (("errors", "stack"), "boom"),
                                        (("launches", 0, "checks", "b"), False), (("launches", 6, "checks", "a"), False),
                                        (("comparisons", 1, "mismatch"), 1), (("comparisons", 5, "twin_mismatch"), 3),
                                        (("comparisons", 3, "reference"), 0), (("comparisons", 7, "words"), 0),
                                        (("totals", "mismatch"), 1)])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert X._verdict(_doc())
    doc = _doc()
    node = doc
    for p in path[:-1]:
        node = node[p]
    node[path[-1]] = value
    assert not X._verdict(doc)


def test_each_case_has_its_own_verdict() -> None:
    """A case is its phase and its name: a case's launch error fails its served and edge cases and no other, a comparison's error
    its own case, a missing launch record its case."""
    want = [f"{p}/{c}" for p, c in X.CASES]
    assert want == ["swa/swa", "swa/swa-edge", "swa/swa-4096", "swa/swa-4096-edge", "c128a/c128", "c128a/c128-edge",
                    "c128a/c128-4096", "c128a/c128-4096-edge"]
    assert list(X.cases(_doc())) == want and all(c["ok"] for c in X.cases(_doc()).values())
    doc = _doc()
    doc["errors"]["c128-4096"] = "boom"
    assert {k for k, c in X.cases(doc).items() if not c["ok"]} == {"c128a/c128-4096", "c128a/c128-4096-edge"}
    doc = _doc()
    doc["errors"]["compare/swa/swa-edge"] = "boom"
    doc["launches"] = [x for x in doc["launches"] if x["case"] != "c128"]
    assert {k for k, c in X.cases(doc).items() if not c["ok"]} == {"swa/swa-edge", "c128a/c128"}


def _stub_config(monkeypatch) -> None:
    for k in ("TRITON_CACHE_DIR", "VLLM_ENABLE_V1_MULTIPROCESSING"):
        monkeypatch.setenv(k, "")
    vc = types.ModuleType("vllm.config")
    vc.CompilationConfig = vc.VllmConfig = lambda **kw: kw
    vc.set_current_vllm_config = lambda cfg: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", vc)
    monkeypatch.setattr(X.G, "_environment", dict)


def test_main_end_to_end_on_stubbed_launches(observed, tmp_path, monkeypatch, capsys) -> None:
    """`main` with the GPU stood in for (the small rows' twin-made words: each one's first served job and an edge job with one
    word flipped in the HCA one) and vLLM's config module stubbed: the jobs through the spawned worker pool, the per-case verdicts,
    the sealed record and the exit status, so a slip in the orchestration shows here and not on the H200."""
    _stub_config(monkeypatch)
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True})
    monkeypatch.setattr(X, "bind_check", lambda: {"bindings": {}, "ok": True})
    monkeypatch.setattr(X, "ptx", lambda cache, outdir: {"ok": True})
    by = {}
    for case in SMALL.values():
        for j in observed[case].jobs:
            by.setdefault(j["case"], j)
    edge = _flipped(tmp_path / "bad", {**by["c128-4096-edge"], "rows": [0, 2]}, "got_rows", (1, 100))
    jobs = [{**by["swa-4096"], "rows": [0, 3]}, {**by["swa-4096-edge"], "rows": [0, 2]}, {**by["c128-4096"], "rows": [0, 3]}, edge]

    def gpu(work, outdir, quick, doc):
        doc["selection"] = {"ok": True}
        doc["launches"] += [x for case in SMALL.values() for x in observed[case].recs]
        return jobs

    monkeypatch.setattr(X, "gpu", gpu)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1", "--quick"]) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"]
    good = {"swa/swa-4096", "swa/swa-4096-edge", "c128a/c128-4096"}
    assert {c: v["ok"] for c, v in doc["cases"].items()} == {c: c in good for c in X.cases(_doc())}
    words = X.statics("c128-4096")["W"] + 1
    assert doc["cases"]["c128a/c128-4096-edge"]["totals"] == {"words": 2 * words, "mismatch": 1, "reference": 2, "twin_mismatch": 0}
    assert doc["cases"]["swa/swa-4096"]["totals"] == {"words": 3 * (X.statics("swa-4096")["SW"] + 1), "mismatch": 0, "reference": 1,
                                                      "twin_mismatch": 0}
    printed = capsys.readouterr().out
    assert "DSV4-DECODE-METADATA NO-GO" in printed
    assert f"no-go cases={[c for c in X.cases(_doc()) if c not in good]}" in printed


def test_a_failed_gate_writes_a_sealed_record_and_exits_3(tmp_path, monkeypatch, capsys) -> None:
    """The device gate failing writes the sealed NO-GO record, prints GATE-FAIL and exits 3 before any launch."""
    _stub_config(monkeypatch)
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": False, "sms": 114})
    monkeypatch.setattr(X, "gpu", lambda *a: pytest.fail("launched past a failed gate"))
    with pytest.raises(SystemExit) as e:
        X.main(["--out", str(tmp_path / "out"), "--scratch", str(tmp_path)])
    assert e.value.code == 3
    doc = json.loads((tmp_path / "out" / X.RECORD).read_text())
    assert doc["ok"] is False and doc["digest"] == digest_of(doc) and doc["gate"] == {"ok": False, "sms": 114}
    assert "DSV4-DECODE-METADATA GATE-FAIL" in capsys.readouterr().out


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["dsv4_decode_metadata_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)
