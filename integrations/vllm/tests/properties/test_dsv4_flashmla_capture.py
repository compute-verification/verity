"""The H200 capture of DeepSeek-V4-Pro's FlashMLA decode and prefill (`dsv4_flashmla_capture_gpu`) on the CPU: its cache layout
against FlashMLA's quant.py, its operands, the prefill's workspace restated in torch, the rows it hands each token's Definition,
its comparison jobs on twin-made kernel words, its verdicts and its main loop with the GPU stood in for."""
from __future__ import annotations

import contextlib
import copy
import json
import sys
import types

import numpy as np
import pytest
import torch

from tests.properties import dsv4_flashmla_capture_gpu as X
from verity_vllm.program.kernels import flashmla_prefill_twin as PT
from verity_vllm.program.kernels import flashmla_twin as T
from verity_vllm.program.registry import flashmla as F
from verity_vllm.program.registry import flashmla_prefill as P
from verity_vllm.properties.record import REGISTRY, digest_of


FLIP = (6, 3, 100)
PFLIP, WFLIP = (2, 5, 300), (1, 3, 77)
DECODE = [c for p, c in X.CASES if p == "decode"]
PREFILL = [c for p, c in X.CASES if p == "prefill"]


@pytest.fixture(scope="module")
def swa(tmp_path_factory):
    """The `swa` case's saved arrays with the twin's words as the kernel's: (num_splits, paths, paths with word FLIP =
    (token, head, word) flipped)."""
    d = tmp_path_factory.mktemp("swa")
    o = X.case_operands("swa")
    m, ns = X.twin_sched(o)
    b = o["q"].shape[0]
    a = X._load(X._save(d / "pre", **X.case_arrays(o, np.zeros((b, X.H_Q, X.D), np.uint16), m, ns)))
    metas, nsl = X._metas(a)
    out = np.stack([X.twin_out(a, t, metas, nsl)[1] for t in range(b)])
    good = X._save(d / "good", **X.case_arrays(o, out, m, ns))
    out[FLIP] ^= 1
    return ns, good, X._save(d / "bad", **X.case_arrays(o, out, m, ns))


@pytest.fixture(scope="module")
def pswa(tmp_path_factory):
    """The prefill `swa` case's saved arrays with the twin's workspace and words as the GPU's: (lens, paths, paths with output word
    PFLIP = (token, head, word) flipped, paths with workspace word WFLIP = (request, row, word) flipped)."""
    d = tmp_path_factory.mktemp("pswa")
    o = X.prefill_operands("swa")
    ws = X.workspace(o, X.SENTINELS[0])
    a = X.prefill_arrays(o, ws, np.zeros((o["q"].shape[0], X.H_Q, X.D), np.uint16))
    out = np.stack([X.prefill_twin_out(a, t)[1] for t in range(o["q"].shape[0])])
    good = X._save(d / "good", **X.prefill_arrays(o, ws, out))
    out[PFLIP] ^= 1
    bad_out = X._save(d / "bad-out", **X.prefill_arrays(o, ws, out))
    out[PFLIP] ^= 1
    ws[WFLIP] ^= 1
    return o["lens"], good, bad_out, X._save(d / "bad-kv", **X.prefill_arrays(o, ws, out))


def test_the_statics_are_dsv4s() -> None:
    """DeepSeek-V4-Pro at TP8 (config.json at b5968e91): 128 / 8 heads padded to 64, head_dim 512 = 448 | 64, window 128, the
    indexer's top 1024; FlashMLA's MODEL1 page of 64 tokens is 37440 bytes (quant.py:142), and the planner's 132 partitions."""
    assert (X.NH, X.H_Q, X.DN, X.DR, X.D, X.QT, X.BN) == (16, 64, 448, 64, 512, 64, 64)
    assert X.SCALE == 512 ** -0.5 == F.FLASHMLA_DSV4["SCALE"] and X.WINDOW == 128 and X.C4_TOPK == 1024
    assert X.page_bytes(64) == 37440 and X.page_bytes(2) == 1728 and T.num_sm_parts(X.H_Q) == 132
    assert DECODE == list(X.CASE_LAYERS) and PREFILL == list(X.PREFILL_CASE_LAYERS) and len(X.EDGE_SINKS) == X.NH


def test_the_prefill_statics_are_dsv4s() -> None:
    """`_forward_prefill`'s index rows round_up(top_k + 128, 128) wide (128, 1152, 256), NP = ceil(ceil(n / 64) / 2) and at least
    one, the compressed caches' page blocks the decode's, the C4 tokens over every NP of a 1152-wide row and both parities of their
    block counts, and the Definitions' statics the capture's."""
    assert [X.prefill_width(k) for k in ("swa", "c4", "c128")] == [128, 1152, 256]
    assert [X.prefill_pairs(n) for n in (0, 1, 64, 65, 128, 129, 513, 1088, 1152)] == [1, 1, 1, 1, 1, 2, 5, 9, 9]
    assert all(X.PREFILL_LAYERS[k][1] == X.LAYERS[k][0] for k in ("c4", "c128")) and X.PREFILL_LAYERS["c4"][2] == X.C4_TOPK
    n = np.array([kc + w for _, kc, w in X.PREFILL_LENS["c4"]])
    assert sorted({X.prefill_pairs(int(v)) for v in n}) == list(range(1, X.prefill_pairs(X.prefill_width("c4")) + 1))
    assert set((-(-n // X.BN) % 2).tolist()) == {0, 1} and n.max() == X.prefill_width("c4")
    st = P.FLASHMLA_PREFILL_DSV4
    assert (st["NH"], st["D"], st["BN"], st["SCALE"]) == (X.NH, X.D, X.BN, X.SCALE) and st["DOT"] is F.FLASHMLA_DSV4["DOT"]
    assert P.FLASHMLA_GATHER_DSV4 == {"DN": X.DN, "DR": X.DR, "QT": X.QT}


def test_every_prefill_token_reads_its_requests_gathered_rows() -> None:
    """Each token's indices as `PREFILL_LENS` lays them out: distinct compressed rows of its request, then consecutive window rows
    within its request's, every one a row the gathers write, -1 past its length; no block table onto a reserved block; and
    `prefill_token` the token's 2 NP 64 rows at those indices."""
    for case in ("swa", "c4", "c128"):
        o = X.prefill_operands(case)
        nc, m = (int(v) for v in o["geom"])
        r_ = len(X.PREFILL_REQS)
        assert nc == int(o["comp_lens"].max()) and m == nc + int(o["gather_lens"].max())
        assert not o["q"][:, X.NH:].any() and np.all(o["sink"][X.NH:] == X.F_NINF)
        assert np.all(o["swa_table"] < X.CACHE_BLOCKS["swa"] - 1)
        assert "extra_table" not in o or np.all(o["extra_table"] < X.CACHE_BLOCKS[o["layer"]] - 1)
        ws = np.repeat(np.arange(r_ * m, dtype=np.uint16).reshape(r_, m, 1), X.D, axis=2)
        a = X.prefill_arrays(o, ws, np.zeros((o["q"].shape[0], X.H_Q, X.D), np.uint16))
        w_ = X.written(o)
        for t, (r, kc, w) in enumerate(X.PREFILL_LENS[case]):
            idx, n = o["idx"][t], kc + w
            rows = idx[:n] - r * m
            assert int(o["lens"][t]) == n and np.all(idx[n:] == -1) and np.all((rows >= 0) & (rows < m)) and w_[r, rows].all()
            assert np.unique(rows[:kc]).size == kc and np.array_equal(np.diff(rows[kc:]), np.ones(max(w - 1, 0), np.int32))
            assert w == 0 or nc <= rows[kc] and rows[-1] < nc + int(o["gather_lens"][r])
            x = X.prefill_token(a, t)
            k = 2 * x["NP"] * X.BN
            ok = idx[:k] >= 0
            assert x["NP"] == X.prefill_pairs(n) and x["rows"].shape == (k, X.D) and (x["len"], x["skv"]) == (n, r_ * m)
            assert np.array_equal(x["rows"][ok, 0], idx[:k][ok]) and not x["rows"][~ok].any()


def _dequantised(cache: np.ndarray, table: np.ndarray, r: int, positions: np.ndarray, pbs: int) -> np.ndarray:
    blocks = np.asarray(table)[r, positions // pbs].astype(np.int64)
    nope, scale, rope = X.gather_rows(cache, blocks * pbs + positions % pbs, pbs)
    x = torch.from_numpy(nope).view(torch.float8_e4m3fn).to(torch.bfloat16)
    s = torch.from_numpy(scale.astype(np.int16) << 7).view(torch.bfloat16).repeat_interleave(X.QT, dim=-1)
    return np.concatenate([(x * s).view(torch.int16).numpy().view(np.uint16), rope], axis=-1)


def test_the_workspace_is_each_requests_rows_dequantised() -> None:
    """`workspace`, the two gathers' rows, restated in torch: request r's compressed positions 0.. and its last gather_len window
    positions through their block tables, each e4m3 byte as bf16 times its tile's UE8M0 byte shifted into a bf16's exponent, then
    the bf16 RoPE, at rows 0.. and N_c..; the sentinel everywhere else."""
    for case in ("c4", "c128"):
        o = X.prefill_operands(case)
        nc, m = (int(v) for v in o["geom"])
        want = np.full((len(X.PREFILL_REQS), m, X.D), X.SENTINELS[1], np.uint16)
        for r, (s, g) in enumerate(X.PREFILL_REQS):
            n = int(o["comp_lens"][r])
            want[r, :n] = _dequantised(o["extra_cache"], o["extra_table"], r, np.arange(n), int(o["extra_pbs"][0]))
            want[r, nc:nc + g] = _dequantised(o["swa_cache"], o["swa_table"], r, np.arange(s - g, s), X.SWA_PBS)
        assert np.array_equal(X.workspace(o, X.SENTINELS[1]), want)


def test_the_cache_is_flashmlas_model1_layout() -> None:
    """`pack_cache` read back through quant.py's own views (tests/quant.py:143-147, `dequantize_k_cache`:212-216) on a uint8
    tensor, the eighth scale byte and the page padding untouched, and `gather_rows` the rows by slot with -1 rows zero."""
    rng = np.random.default_rng(0)
    for pbs in (64, 2):
        nb = 3
        n = nb * pbs
        nope = rng.integers(0, 256, (n, X.DN), dtype=np.uint8)
        scale = rng.integers(0, 256, (n, X.DN // X.QT), dtype=np.uint8)
        rope = rng.integers(0, 1 << 16, (n, X.DR), dtype=np.uint16)
        fill = rng.integers(0, 256, (nb, X.page_bytes(pbs)), dtype=np.uint8)
        c = X.pack_cache(nope, scale, rope, pbs, fill)
        res = torch.from_numpy(c)[:, :pbs * X.TOKEN_BYTES]
        nope_rope = res[:, :pbs * (X.DN + 2 * X.DR)].reshape(nb, pbs, X.DN + 2 * X.DR)
        assert np.array_equal(nope_rope[:, :, :X.DN].numpy().reshape(n, X.DN), nope)
        assert np.array_equal(nope_rope[:, :, X.DN:].contiguous().view(torch.int16).numpy().view(np.uint16).reshape(n, X.DR), rope)
        sc = res[:, pbs * (X.DN + 2 * X.DR):].reshape(nb, pbs, 8)
        assert np.array_equal(sc[:, :, :7].numpy().reshape(n, 7), scale)
        assert np.array_equal(sc[:, :, 7].numpy(), fill[:, pbs * 576:pbs * 584].reshape(nb, pbs, 8)[:, :, 7])
        assert np.array_equal(c[:, pbs * X.TOKEN_BYTES:], fill[:, pbs * X.TOKEN_BYTES:])
        idx = np.array([0, n - 1, -1, pbs + 1, 5 % n])
        g = X.gather_rows(c, idx, pbs)
        for got, want in zip(g, (nope, scale, rope)):
            assert np.array_equal(got[[0, 1, 3, 4]], want[idx[[0, 1, 3, 4]]]) and not got[2].any()


def test_the_rows_quantize_as_quant_py() -> None:
    """`quantize_rows` is quant.py's MODEL1 quantization (:150-157) bit for bit: the UE8M0 scale of each 64-element tile and the
    E4M3 bytes of x / scale, here through torch's own casts."""
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((32, X.DN)) * np.exp2(rng.uniform(-8, 8, (32, 1)))).astype(np.float32)
    x[0] = 0
    x[1, :64] = 1e-9
    r = rng.standard_normal((32, X.DR)).astype(np.float32)
    nope, scale, rope = X.quantize_rows(x, r)
    t = torch.from_numpy(x).view(32, 7, 64)
    inv = torch.pow(2, torch.clamp_min(t.abs().amax(-1) / 448.0, 1e-4).log2().ceil())
    assert np.array_equal(scale, inv.to(torch.float8_e8m0fnu).view(torch.uint8).numpy())
    want = (t / inv.unsqueeze(-1)).to(torch.float8_e4m3fn).view(torch.uint8).numpy().reshape(32, X.DN)
    assert np.array_equal(nope, want)
    assert np.array_equal(rope, torch.from_numpy(r).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))


def test_every_case_plans_and_hands_each_token_its_rows() -> None:
    """Every case's tokens through the planner port and `token_inputs`: the window's NW rows from the SWA cache then
    ceil64(extra_len) compressed rows, the indices -1 past each length, the lengths, and SPLITS tiling the token's blocks; no
    token but `edge`'s reads a reserved block."""
    for case in DECODE:
        o = X.case_operands(case)
        m, ns = X.twin_sched(o)
        a = X.case_arrays(o, np.zeros((o["q"].shape[0], X.H_Q, X.D), np.uint16), m, ns)
        metas, nsl = X._metas(a)
        w = X.extra_width(o)
        assert not o["q"][:, X.NH:].any() and np.all(o["sink"][X.NH:] == X.F_NINF)
        for t in range(o["q"].shape[0]):
            x = X.token_inputs(a, t, metas, nsl)
            s = int(o["swa_lens"][t])
            e = int(o["extra_lens"][t]) if w else 0
            assert x["NW"] == max(-(-s // 64), 1) * 64 and x["N"] == x["NW"] + -(-e // 64) * 64
            assert x["SPLITS"][0] == 0 and x["SPLITS"][-1] == x["N"] // 64 and len(x["SPLITS"]) - 1 == nsl[t + 1] - nsl[t]
            assert [int(v) for v in x["lens"]] == [s] + ([e] if x["N"] > x["NW"] else [])
            idx = x["idx"].view(np.int32)
            assert np.array_equal(idx[:x["NW"]], o["swa_idx"][t, :x["NW"]])
            assert np.array_equal(idx[x["NW"]:], o["extra_buf"][t, :x["N"] - x["NW"]]) if w else x["N"] == x["NW"]
            if case != "edge":
                assert np.all(idx[:x["NW"]][s:] == -1) and np.all(idx[x["NW"]:][e:] == -1)
                assert np.all(o["swa_idx"][t] < (X.CACHE_BLOCKS["swa"] - 1) * X.SWA_PBS)
                if w:
                    assert np.all(o["extra_buf"][t, :w] < (X.CACHE_BLOCKS[o["layer"]] - 1) * X.LAYERS[o["layer"]][0])
            nope, scale, rope = X.gather_rows(o["swa_cache"], idx[:x["NW"]], X.SWA_PBS)
            assert np.array_equal(x["nope"][:x["NW"]], nope) and np.array_equal(x["rope"][:x["NW"]], rope)


def test_the_cases_cover_the_planners_regimes() -> None:
    """The lone token is cut into 18 splits, `batch` shares partitions between requests (some split, some whole), and the
    window-only tokens of length 0 and 1 are one block each."""
    per = {c: np.diff(X.twin_sched(X.case_operands(c))[1]) for c in DECODE}
    assert per["lone"].tolist() == [18] and set(per["batch"].tolist()) == {1, 2} and per["swa"][:2].tolist() == [1, 1]
    assert per["c4"].max() == 18 and per["c128"].min() >= 2


def test_the_edge_words_land_where_they_are_meant_to(tmp_path) -> None:
    """`edge`'s token 2 reads -1 within its lengths and rows of NaN scale past them, and only its NaN-sink head is NaN (the
    rows past a length are masked); token 5 is an empty request, every real head +0."""
    o = X.case_operands("edge")
    m, ns = X.twin_sched(o)
    a = X.case_arrays(o, np.zeros((8, X.H_Q, X.D), np.uint16), m, ns)
    metas, nsl = X._metas(a)
    x, y = X.twin_out(a, 2, metas, nsl)
    valid = T.row_valid(x["idx"], x["lens"], x["NW"])
    assert int(np.count_nonzero(valid)) == 99 + 199 and np.all(x["scale"][~valid][-20:] == 0xFF)
    assert [h for h in range(X.NH) if np.all(y[h] == 0x7FFF)] == [2] and not y[0].any()      # sink NaN, sink +inf
    assert not X.twin_out(a, 5, metas, nsl)[1][:X.NH].any()


def test_the_twin_jobs_take_the_kernel_words(swa) -> None:
    ns, good, bad = swa
    jobs = [j for j in X.case_jobs("swa", good, ns, quick=True) if j["kind"] == "twin"]
    assert [j["tokens"] for j in jobs] == [[0, 4], [4, 8]]
    res = [X.compare(j) for j in jobs]
    assert sum(r["words"] for r in res) == 8 * X.H_Q * X.D and not any(r["mismatch"] for r in res)
    r = X.compare({**jobs[1], "paths": bad})
    assert r["mismatch"] == 1 and [r["first_mismatch"][k] for k in ("token", "head", "word")] == list(FLIP)


def test_the_reference_job_is_the_registered_definition(swa) -> None:
    """`FlashMlaDecode_v1{NH=16,...}` itself on a window token of one block (its 16 heads) against the twin."""
    ns, good, _ = swa
    r = X.compare({"phase": "decode", "case": "swa", "kind": "ref", "paths": good, "token": 2})
    assert {k: r[k] for k in X.TOTALS} == {"words": 0, "mismatch": 0, "reference": 1, "twin_mismatch": 0}


def test_the_jobs_reference_the_extreme_tokens() -> None:
    ns = np.cumsum([0, 1, 3, 2, 1, 5, 2])
    refs = lambda case, quick: [j["token"] for j in X.case_jobs(case, {}, ns, quick) if j["kind"] == "ref"]  # noqa: E731
    assert refs("c4", False) == [0, 4] and refs("c4", True) == [0] and refs("edge", False) == list(range(6))
    assert [j["tokens"] for j in X.case_jobs("c4", {}, ns, False) if j["kind"] == "twin"] == [[0, 4], [4, 6]]


def test_a_bad_planner_word_is_an_error_of_its_case(tmp_path) -> None:
    o = X.case_operands("c128")
    m, ns = X.twin_sched(o)
    m = m.copy()
    m[1, 3] += 1
    paths = X._save(tmp_path, **X.case_arrays(o, np.zeros((6, X.H_Q, X.D), np.uint16), m, ns))
    r = X.compare({"phase": "decode", "case": "c128", "kind": "twin", "paths": paths, "tokens": [0, 6]})
    assert "do not tile" in r["error"]


def test_the_prefill_jobs_take_the_gpus_words(pswa) -> None:
    """The workspace job over every word of the four requests' 144 rows and six `FlashMlaGatherRow_v1` references, the twin jobs
    over every output word; a flipped workspace word and a flipped output word each found where they are."""
    lens, good, bad_out, bad_kv = pswa
    jobs = X.prefill_jobs("swa", good, lens, quick=True)
    assert [j["kind"] for j in jobs] == ["gather", "prefill-twin", "prefill-twin", "prefill-ref"]
    assert [j["tokens"] for j in jobs[1:3]] == [[0, 4], [4, 8]] and jobs[3]["token"] == 0
    g, *tw = (X.compare(j) for j in jobs[:3])
    assert {k: g[k] for k in X.TOTALS} == {"words": 4 * 144 * X.D, "mismatch": 0, "reference": 6, "twin_mismatch": 0}
    assert sum(r["words"] for r in tw) == 8 * X.H_Q * X.D and not any(r["mismatch"] for r in tw)
    r = X.compare({**jobs[1], "paths": bad_out})
    assert r["mismatch"] == 1 and [r["first_mismatch"][k] for k in ("token", "head", "word")] == list(PFLIP)
    r = X.compare({**jobs[0], "paths": bad_kv})
    assert r["mismatch"] == 1 and [r["first_mismatch"][k] for k in ("request", "row", "word")] == list(WFLIP)


def test_the_prefill_jobs_reference_the_extreme_tokens() -> None:
    refs = lambda case, lens, quick: [j["token"] for j in X.prefill_jobs(case, {}, lens, quick) if j["kind"] == "prefill-ref"]  # noqa: E731
    lens = [5, 1152, 1, 300, 1, 1152]
    assert refs("c4", lens, False) == [1, 2] and refs("c4", lens, True) == [2] and refs("edge", lens, False) == list(range(6))
    assert [j["tokens"] for j in X.prefill_jobs("c4", {}, lens, False) if j["kind"] == "prefill-twin"] == [[0, 4], [4, 6]]


@pytest.mark.slow
def test_the_prefill_reference_job_is_the_registered_definition(pswa) -> None:
    """`FlashMlaPrefill_v1{NH=16,NP=1,...}` itself on a token of one valid row against the twin."""
    _, good, _, _ = pswa
    r = X.compare({"phase": "prefill", "case": "swa", "kind": "prefill-ref", "paths": good, "token": 0})
    assert {k: r[k] for k in X.TOTALS} == {"words": 0, "mismatch": 0, "reference": 1, "twin_mismatch": 0}


def _finite(words: np.ndarray) -> np.ndarray:
    return (np.asarray(words) & 0x7F80) != 0x7F80


def _edge() -> dict:
    o = X.prefill_operands("edge")
    return X.prefill_arrays(o, X.workspace(o, X.SENTINELS[0]), np.zeros((o["q"].shape[0], X.H_Q, X.D), np.uint16))


def test_the_prefill_edge_words_land_where_they_are_meant_to() -> None:
    """`edge`'s rows: token 0's 1152 within its length all valid and finite (the scales 0x00 / 0x01 / 0x7F / 0x80, the subnormal
    and the +-0 rows among them); token 1's rows 1 and 2 and its two SWA rows valid and not finite; token 2's six bad indices
    invalid within its length, and every row it reads past its length a valid index onto a non-finite row; token 5 of length 0
    over valid indices; token 7 one valid index of 1152."""
    a = _edge()

    def token(t: int):
        x = X.prefill_token(a, t)
        return x, PT.row_valid(x["idx"], x["len"], x["skv"]), _finite(x["rows"]).all(axis=1)

    x, valid, fin = token(0)
    assert valid.sum() == 1152 and fin[valid].all() and set(x["idx"][:3].tolist()) == {0, 3, 4}
    x, valid, fin = token(1)
    assert valid[[0, 1, 210, 220]].all() and not fin[[0, 1, 210, 220]].any() and fin[2:200].all()
    x, valid, fin = token(2)
    n = x["len"]
    assert valid.sum() == n - 6 and not valid[7:13].any() and not fin[n:].any()
    assert np.all((x["idx"][n:].view(np.int32) >= 0) & (x["idx"][n:].view(np.int32) < x["skv"]))
    x, valid, _ = token(5)
    idx = x["idx"].view(np.int32)
    assert x["len"] == 0 and not valid.any() and np.all((idx >= 0) & (idx < x["skv"]))
    x, valid, _ = token(7)
    assert x["len"] == 1152 and valid.sum() == 1


@pytest.mark.slow
def test_the_prefill_edge_outputs() -> None:
    """The twin on `edge`: tokens 0 and 2 are finite on every head but the NaN-sink one (token 2's bad and past-length rows are
    masked and zero-filled), token 1's rows that overflow or are NaN make every head NaN, token 5 of length 0 is +0 on every
    real head."""
    a = _edge()
    out = {t: X.prefill_twin_out(a, t)[1][:X.NH] for t in (0, 1, 2, 5)}
    for t in (0, 2):
        assert [h for h in range(X.NH) if not _finite(out[t][h]).all()] == [2] and np.all(out[t][2] == 0x7FFF)
    assert np.all(out[1] == 0x7FFF) and not out[5].any()


def test_the_programs_bind_at_dsv4s_statics() -> None:
    rec = X.bind_check()
    assert rec["ok"], rec


LAUNCH = {"decode": {"b": 8, "checks": {"shape": True, "sched": True, "rerun": True, "planner": True, "returned": True}},
          "prefill": {"b": 8, "checks": {"shape": True, "gather_rerun": True, "rerun": True, "returned": True}}}


def _doc() -> dict:
    launches = [{"phase": p, "case": c, **copy.deepcopy(LAUNCH[p])} for p, c in X.CASES]
    comps = [{"phase": p, "case": c, "words": 10, "mismatch": 0, "reference": 1, "twin_mismatch": 0, "first_mismatch": None}
             for p, c in X.CASES]
    return {"schema": X.SCHEMA, "gate": {"ok": True}, "errors": {}, "selection": {"ok": True}, "bind": {"ok": True},
            "launches": launches, "comparisons": comps,
            "totals": {k: sum(r[k] for r in comps) for k in X.TOTALS}}


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("selection", "ok"), False), (("bind", "ok"), False),
                                        (("errors", "stack"), "boom"), (("launches", 2, "checks", "sched"), False),
                                        (("launches", 0, "checks", "rerun"), False), (("comparisons", 1, "mismatch"), 1),
                                        (("comparisons", 4, "twin_mismatch"), 3), (("comparisons", 5, "reference"), 0),
                                        (("launches", 7, "checks", "gather_rerun"), False), (("comparisons", 6, "words"), 0),
                                        (("comparisons", 9, "twin_mismatch"), 1), (("totals", "mismatch"), 1)])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert X._verdict(_doc())
    doc = _doc()
    node = doc
    for p in path[:-1]:
        node = node[p]
    node[path[-1]] = value
    assert not X._verdict(doc)


def test_each_case_has_its_own_verdict() -> None:
    """A case is its phase and its name: a prefill case's launch, comparison or error fails it and not the decode case of that
    name, and the other way round."""
    want = [f"decode/{c}" for c in X.CASE_LAYERS] + [f"prefill/{c}" for c in X.PREFILL_CASE_LAYERS]
    assert list(X.cases(_doc())) == want and all(c["ok"] for c in X.cases(_doc()).values())
    doc = _doc()
    doc["launches"] = [x for x in doc["launches"] if x["case"] != "lone"]
    doc["errors"]["compare/decode/edge"] = "boom"
    doc["comparisons"][2]["mismatch"] = 1
    doc["errors"]["prefill/swa"] = "boom"
    doc["launches"][-1]["checks"]["returned"] = False
    assert {k for k, c in X.cases(doc).items() if not c["ok"]} == {"decode/lone", "decode/edge", "decode/c128", "prefill/swa",
                                                                   "prefill/edge"}


@pytest.mark.slow
def test_main_end_to_end_on_stubbed_launches(swa, pswa, tmp_path, monkeypatch, capsys) -> None:
    """`main` with the GPU's launches stood in for (the decode `swa` case's twin-made words, the same words under `c128` with one
    flipped, and the prefill `swa` case's twin-made workspace and words) and vLLM's config module stubbed: the jobs through the
    spawned worker pool, the per-case verdicts, the sealed record and the exit status, so a slip in the orchestration shows here
    and not in the H200's first hour."""
    ns, good, bad = swa
    plens, pgood, _, _ = pswa
    vc = types.ModuleType("vllm.config")
    vc.CompilationConfig = vc.VllmConfig = lambda **kw: kw
    vc.set_current_vllm_config = lambda cfg: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", vc)
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    monkeypatch.setattr(X, "bind_check", lambda: {"bindings": {}, "ok": True})

    def gpu(work, outdir, quick, doc):
        doc["selection"] = {"ok": True}
        doc["launches"] += [{"phase": "decode", "case": c, **copy.deepcopy(LAUNCH["decode"])} for c in ("swa", "c128")]
        doc["launches"].append({"phase": "prefill", "case": "swa", **copy.deepcopy(LAUNCH["prefill"])})
        twin = [j for j in X.case_jobs("swa", good, ns, quick) if j["kind"] == "twin"]
        prefill = [j for j in X.prefill_jobs("swa", pgood, plens, quick) if j["kind"] != "prefill-ref"]
        return twin + [{"phase": "decode", "case": "swa", "kind": "ref", "paths": good, "token": 1}] + \
            [{**j, "case": "c128", "paths": bad} for j in twin] + prefill

    monkeypatch.setattr(X, "gpu", gpu)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1", "--quick"]) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"]
    go = {"decode/swa", "prefill/swa"}
    assert {c: v["ok"] for c, v in doc["cases"].items()} == {c: c in go for c in X.cases(_doc())}
    assert doc["cases"]["decode/c128"]["totals"]["mismatch"] == 1
    assert doc["cases"]["prefill/swa"]["totals"] == {"words": 4 * 144 * X.D + 8 * X.H_Q * X.D, "mismatch": 0, "reference": 6,
                                                     "twin_mismatch": 0}
    assert doc["totals"]["words"] == 2 * 8 * X.H_Q * X.D + 4 * 144 * X.D + 8 * X.H_Q * X.D
    printed = capsys.readouterr().out
    assert "DSV4-FLASHMLA NO-GO" in printed
    assert f"no-go cases={[c for c in X.cases(_doc()) if c not in go]}" in printed


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["dsv4_flashmla_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)
