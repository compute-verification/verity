"""The H200 capture of DeepSeek-V4-Pro's q-norm/RoPE/SWA insert and indexer q quant (`dsv4_qnorm_insert_capture_gpu`) on the
CPU: its table and operands, the edge words where they are meant to be, the insert's bytes as FlashMLA's MODEL1 page, its
comparison jobs on twin-made kernel words, the spy on the CuTe dispatch, its verdicts and its main loop with the GPU stood in
for."""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import contextlib
import copy
import functools
import json
import sys
import types

import numpy as np
import pytest

from tests.properties import dsv4_qnorm_insert_capture_gpu as X
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.definitions.numerics import dsv4_qnorm_insert_twin as W
from verity.ml.definitions.registry import dsv4_qnorm_insert as R
from verity_vllm.properties.record import REGISTRY, digest_of

M = X.M
#: (token, head, word) of q_out: a pad head, and a real head of a DP-padded row (past the slots, normed and not inserted)
QFLIPS = ((2, 63, 0), (45, 3, 300))
IQFLIP, IWFLIP = (150, 7, 99), (3, 60)


def _flip(a: np.ndarray, at: tuple) -> np.ndarray:
    a = np.array(a)
    a[at] ^= 1
    return a


@pytest.fixture(scope="module")
def padded(tmp_path_factory):
    """The insert `padded` case (48 tokens, 40 slots, some -1) saved with the twin's words as the kernel's: (operands, paths, paths
    with QFLIPS flipped in q_out, with a written nope byte flipped, with the written pad scale byte flipped, with an unwritten
    byte flipped) and the flipped bytes' (block, byte)."""
    d = tmp_path_factory.mktemp("padded")
    o = X.insert_operands("padded")
    q_out = X.insert_twin(o, 0, o["q"].shape[0])[0]
    cache, _ = X.twin_cache(o)
    base = {"q": o["q"], "kv": o["kv"], "cs": o["cs"], "slots": o["slots"], "fills": o["fills"][:1]}
    s = int(o["slots"][o["slots"] >= 0][0])
    b, off = s // X.PBS, s % X.PBS
    at = {"nope": (b, off * M.DATA_BYTES + 10), "pad": (b, X.PBS * M.DATA_BYTES + off * M.SCALE_BYTES + 7),
          "unwritten": tuple(int(v) for v in np.argwhere(~X.written(o["slots"], cache.shape[0]))[0])}
    bad_q = q_out.copy()
    for f in QFLIPS:
        bad_q[f] ^= 1
    paths = {"good": X._save(d / "good", **base, q_out=q_out, cache=cache), "q": X._save(d / "q", **base, q_out=bad_q, cache=cache)}
    for k, v in at.items():
        paths[k] = X._save(d / k, **base, q_out=q_out, cache=_flip(cache, v))
    return o, paths, at


@pytest.fixture(scope="module")
def short(tmp_path_factory):
    """The indexer `short` case (200 tokens) saved with the twin's words as the kernel's: (paths, paths with q_fp8 byte IQFLIP and
    weights_out word IWFLIP flipped)."""
    d = tmp_path_factory.mktemp("short")
    o = X.indexer_operands("short")
    q8, w = X.indexer_twin(o, 0, o["q"].shape[0])
    base = {"q": o["q"], "w": o["w"], "cs": o["cs"]}
    return X._save(d / "good", **base, q_fp8=q8, w_out=w), X._save(d / "bad", **base, q_fp8=_flip(q8, IQFLIP),
                                                                   w_out=_flip(w, IWFLIP))


def test_the_statics_are_dsv4s() -> None:
    """DeepSeek-V4-Pro at TP8: 128 / 8 query heads padded to FlashMLA's 64, head_dim 512 = 448 | 64 in 64-element scale blocks,
    the SWA page of 64 tokens (37440 bytes); the indexer's 64 heads of 128 and its weight scale 128^-0.5 64^-0.5; the cases on
    both sides of the insert's grid cutoff and the CuTe kernel's coarsen cutoff."""
    assert (X.NH, X.NHP, X.DN, X.DR, X.D, X.QB, X.EPS, X.PBS) == (16, 64, 448, 64, 512, 64, 1e-6, 64)
    assert M.page_bytes(X.PBS) == 37440 and (M.TOKEN_BYTES, M.DATA_BYTES, M.SCALE_BYTES) == (584, 576, 8)
    assert (X.IH, X.ID, X.IDR) == (64, 128, 64) and X.SOFTMAX_SCALE * X.HEAD_SCALE == X.ISCALE == R.DSV4_INDEXER["SCALE"]
    T = {c: v[0] for c, v in X.INSERT_CASES.items()}
    assert T["chunk"] == X.INSERT_CUTOFF - 1 and T["reduced"] == X.INSERT_CUTOFF and T["long"] > X.INSERT_CUTOFF
    assert all(n <= t for t, n, _, _ in X.INSERT_CASES.values()) and X.INSERT_CASES["padded"][1] < T["padded"]
    TI = {c: v[0] for c, v in X.INDEXER_CASES.items()}
    assert TI["last1"] == X.COARSEN_CUTOFF - 1 and TI["coarse"] == X.COARSEN_CUTOFF and TI["odd"] > X.COARSEN_CUTOFF
    assert set(X.QUICK_INSERT) <= set(X.INSERT_CASES) and set(X.QUICK_INDEXER) <= set(X.INDEXER_CASES)
    assert len(X.EDGE_K) == X.DN // X.QB and X.EDGE_POS + len(X.EDGE_CS) <= X.MAX_POS


def test_the_sumsq_probe_separates_fused_from_unfused() -> None:
    """The probe is a separate non-served-eps claim.  Its repeated lane leaves three different bf16 words per lane after the
    butterfly and rsqrt; the fused model is the bound Definition, while FMUL-then-FADD differs on 96 output words."""
    o = X.sumsq_probe_operands()
    assert X.SUMSQ_PROBE_EPS == 2.0 ** -140 and o["q"].shape == (1, 1, X.D)
    assert o["q"][0, 0].tolist() == list(X.SUMSQ_PROBE_LANE) * W.LANES
    assert np.all(o["cs"][0, :X.DR // 2] == X.F32_ONE) and not o["cs"][0, X.DR // 2:].any()
    fused = X.sumsq_probe_reference(o["q"][0, 0], o["cs"][0], fused=True)
    unfused = X.sumsq_probe_reference(o["q"][0, 0], o["cs"][0], fused=False)
    fn = bind(R.Dsv4QNormRope, D=X.D, DR=X.DR, EPS=X.SUMSQ_PROBE_EPS)
    definition = np.asarray(X._flat(evaluate(fn, o["q"][0, 0].tolist(), o["cs"][0].tolist())), dtype=np.uint16)
    bad = np.flatnonzero(fused != unfused)
    assert np.array_equal(fused, definition)
    assert bad.tolist() == [16 * lane + i for lane in range(W.LANES) for i in (3, 9, 15)]
    assert {(int(fused[i]), int(unfused[i])) for i in bad} == {(0x3CF4, 0x3CF5), (0x3F4E, 0x3F4F), (0x3E28, 0x3E29)}
    assert "sumsq-probe" not in X.INSERT_CASES and ("insert", "sumsq-probe") not in X.cases_of(False)


def test_the_table_is_ropes_and_its_special_rows() -> None:
    """Below EDGE_POS row p is cos | sin of p t_i in f32 (row 0 the identity); from EDGE_POS the rows of EDGE_CS: the given words,
    a full-mantissa cos with sin = +-cos, and a row of edge words."""
    cs = X.cos_sin_cache()
    h = X.DR // 2
    assert cs.shape == (X.MAX_POS, X.DR) and cs.dtype == np.uint32
    assert np.all(cs[0, :h] == X.F32_ONE) and not cs[0, h:].any()
    f = cs[:X.EDGE_POS].view(np.float32).astype(np.float64)
    assert np.abs(f[:, :h] ** 2 + f[:, h:] ** 2 - 1).max() < 1e-6
    for j, (c, s) in enumerate(X.EDGE_CS):
        row = cs[X.EDGE_POS + j]
        if c == "full":
            assert np.all(row[h:] == (row[:h] if s > 0 else row[:h] | np.uint32(X.SIGN))) and np.all(row[:h] & 1)
            assert np.all((row[:h].view(np.float32) >= 0.5) & (row[:h].view(np.float32) < 1))
        elif c == "words":
            assert np.isin(row, W._F32_EDGE).all()
        else:
            assert np.all(row[:h] == c) and np.all(row[h:] == s)


@pytest.mark.parametrize("case", list(X.INSERT_CASES))
def test_every_insert_case_has_its_operands(case: str) -> None:
    """Each case's q, kv, positions and the table's rows at them; slots distinct where >= 0, within the blocks, the second -1;
    two fills of the cache's shape that differ."""
    T_, n, _, _ = X.INSERT_CASES[case]
    o = X.insert_operands(case)
    assert o["q"].shape == (T_, X.NH, X.D) and o["kv"].shape == (T_, X.D) and o["q"].dtype == o["kv"].dtype == np.uint16
    assert np.all((o["positions"] >= 0) & (o["positions"] < X.MAX_POS))
    assert np.array_equal(o["cs"], X.cos_sin_cache()[o["positions"]])
    s, nb = o["slots"], o["fills"].shape[1]
    v = s[s >= 0]
    assert s.shape == (n,) and s[1] == -1 and np.unique(v).size == v.size and v.max() < nb * X.PBS
    assert o["fills"].shape == (2, nb, M.page_bytes(X.PBS)) and (o["fills"][0] != o["fills"][1]).mean() > 0.99


@pytest.mark.parametrize("case", list(X.INDEXER_CASES))
def test_every_indexer_case_has_its_operands(case: str) -> None:
    T_, _, _ = X.INDEXER_CASES[case]
    o = X.indexer_operands(case)
    assert o["q"].shape == (T_, X.IH, X.ID) and o["w"].shape == (T_, X.IH) and o["q"].dtype == o["w"].dtype == np.uint16
    assert np.array_equal(o["cs"], X.cos_sin_cache()[o["positions"]])


def _abs(x: np.ndarray) -> np.ndarray:
    return np.asarray(x) & 0x7FFF


def test_the_insert_edge_words_land_where_they_are_meant_to() -> None:
    """`edge`: token 0 zero (its q and rope +-0, its nope +0, every scale byte the amax floor's 105, the eighth 0); token 1's infinities
    and NaNs; tokens 2-4 each KV block's amax 448 2^k exactly and one bf16 ulp either side, where the twin's UE8M0 byte k + 127
    turns to k + 128 only above; bf16's largest word leading each head and block; subnormals; the RoPE pairs (x, x) on the
    rows s = +-c; one token on every special row."""
    o = X.insert_operands("edge")
    q, kv, pos = o["q"], o["kv"], o["positions"]
    q_out, nope, scale, rope = X.insert_twin(o, 0, q.shape[0])
    assert not q[0].any() and not kv[0].any() and not nope[0].any()
    assert not _abs(q_out[0]).any() and not _abs(rope[0]).any() and (q_out[0] == 0x8000).any()   # the rotation's fma(0, -s, -0)
    assert scale[0].tolist() == [105] * 7 + [0] and not scale[:, 7].any()
    assert [{int(w) for w in q[1, h] if w & 0x7F80 == 0x7F80} for h in range(3)] == [{0x7F80}, {0xFF80}, {0x7FC0}]
    assert np.all(q[1, 3] == 0xFFC1) and np.all(kv[1, 3 * X.QB:4 * X.QB] == 0x7FC0)
    k = np.array(X.EDGE_K)
    for t, ulp in ((2, 0), (3, 1), (4, -1)):
        blocks = _abs(kv[t, :X.DN]).reshape(-1, X.QB)
        assert blocks.max(axis=1).tolist() == [X.amax_word(int(e), ulp) for e in k]
        assert scale[t, :7].tolist() == (k + 127 + (ulp > 0)).tolist()
    assert np.all(_abs(q[5, :, 0]) == 0x7F7F) and np.all(_abs(kv[5, :X.DN:X.QB]) == 0x7F7F)
    assert np.all((q[6] & 0x7F80) == 0) and np.all(_abs(q[6]) > 0) and np.all((kv[6] & 0x7F80) == 0)
    h = X.DR // 2
    for t in range(7, 11):
        assert np.array_equal(q[t, :, X.DN + 1::2], q[t, :, X.DN::2]) and np.array_equal(kv[t, X.DN + 1::2], kv[t, X.DN::2])
        c, s = o["cs"][t, :h], o["cs"][t, h:]
        assert np.all((s & np.uint32(0x7FFFFFFF)) == c)
    assert pos[11:23].tolist() == list(range(X.EDGE_POS, X.EDGE_POS + len(X.EDGE_CS)))


def test_the_indexer_edge_words_land_where_they_are_meant_to() -> None:
    """`edge`: token 0 zero; token 1's specials and edge weights; tokens 2-4 head h's amax 448 2^(h - 40) and one ulp either side
    on the identity row, where the twin's largest e4m3 byte is 448's (0x7E) exactly and below and 224's (0x76) above, from the
    heads whose amax clears the floor 1e-4; NoPE +0 and the RoPE pairs (x, x) on the rows s = +-c; one token on every special
    row."""
    o = X.indexer_operands("edge")
    q, w, pos = o["q"], o["w"], o["positions"]
    q8, w_out = X.indexer_twin(o, 0, q.shape[0])
    assert not q[0].any() and not w[0].any() and not (q8[0] & 0x7F).any() and not w_out[0].any()
    assert np.all(q[1, 3] == 0xFFC1) and np.isin(w[1], W._BF16_EDGE).all()
    floor = int(np.ceil(np.log2(1e-4 / 448)))
    for t, ulp, top in ((2, 0, 0x7E), (3, 1, 0x76), (4, -1, 0x7E)):
        assert pos[t] == X.EDGE_POS
        assert _abs(q[t]).max(axis=1).tolist() == [X.amax_word(hh - 40, ulp) for hh in range(X.IH)]
        assert np.all((q8[t, 40 + floor:] & 0x7F).max(axis=1) == top)
    for t in range(5, 9):
        assert not q[t, :, :X.ID - X.IDR].any() and np.array_equal(q[t, :, X.ID - X.IDR + 1::2], q[t, :, X.ID - X.IDR::2])
        assert pos[t] == X.EDGE_POS + t - 4
    assert np.all((q[9] & 0x7F80) == 0) and np.all(_abs(q[9]) > 0)
    assert pos[11:23].tolist() == list(range(X.EDGE_POS, X.EDGE_POS + len(X.EDGE_CS)))


def test_the_insert_writes_flashmlas_model1_rows() -> None:
    """The twin's rows scattered at every slot of three pages, in a random order, are `pack_cache`'s MODEL1 pages of them
    (quant.py's layout, test_dsv4_flashmla_capture) with the eighth scale byte 0; `written` is each valid slot's 576 + 8 bytes
    and never the page's padding; `gather_rows`, FlashMLA's read, gives each token's nope, seven scale bytes and rope back."""
    rng = np.random.default_rng(3)
    nb = 3
    n = nb * X.PBS
    kv = W._bf16(rng, (n, X.D))
    nope, scale, rope = W.kv_rows(kv, X.cos_sin_cache()[rng.integers(0, X.EDGE_POS, n)], DR=X.DR, QB=X.QB)
    fill = rng.integers(0, 256, (nb, M.page_bytes(X.PBS)), dtype=np.uint8)
    slots = rng.permutation(n).astype(np.int64)
    cache = W.scatter(fill, slots, *W.cache_rows(nope, scale, rope), X.PBS)
    tok = np.argsort(slots)
    want = M.pack_cache(nope[tok], scale[tok, :7], rope[tok], X.PBS, fill)
    want[:, X.PBS * M.DATA_BYTES + 7:X.PBS * M.TOKEN_BYTES:M.SCALE_BYTES] = 0
    assert np.array_equal(cache, want)
    mask = X.written(slots, nb)
    assert mask[:, :X.PBS * M.TOKEN_BYTES].all() and not mask[:, X.PBS * M.TOKEN_BYTES:].any()
    part = slots.copy()
    part[::3] = -1
    assert X.written(part, nb).sum() == M.TOKEN_BYTES * int((part >= 0).sum())
    for got, w in zip(M.gather_rows(cache, slots, X.PBS), (nope, scale[:, :7], rope)):
        assert np.array_equal(got, w)


def test_the_launch_checks_both_caches(padded) -> None:
    """The twin's insert on both fills passes; a byte written outside the slots (into the page padding, or a skipped slot's) fails
    `untouched`, and a written byte that differs between the runs fails `rerun`."""
    o, _, at = padded
    rows = W.cache_rows(*W.kv_rows(o["kv"][:o["slots"].size], o["cs"][:o["slots"].size], DR=X.DR, QB=X.QB))
    g1, g2 = (W.scatter(f, o["slots"], *rows, X.PBS) for f in o["fills"])
    assert X.cache_checks(o["slots"], o["fills"], g1, g2) == {"rerun": True, "untouched": True}
    for where in ((0, M.page_bytes(X.PBS) - 1), at["unwritten"]):
        assert X.cache_checks(o["slots"], o["fills"], g1, _flip(g2, where)) == {"rerun": True, "untouched": False}
    assert X.cache_checks(o["slots"], o["fills"], g1, _flip(g2, at["nope"])) == {"rerun": False, "untouched": True}


def test_the_insert_jobs_take_the_kernel_words(padded) -> None:
    """The q job over every word of 48 tokens' 64 heads (the pad heads and the DP-padded rows among them), the cache job over every
    byte of the cache and each slot's row as FlashMLA reads it; both flipped q words found, the first where it is; a flipped
    nope byte found in the cache and in FlashMLA's nope, the pad scale byte and an unwritten byte in the cache alone."""
    o, paths, at = padded
    jobs = X.insert_jobs("padded", paths["good"], o["q"].shape[0], quick=True)
    assert [j["kind"] for j in jobs] == ["cache", "q", "insert-ref"] and jobs[1]["tokens"] == [0, 48]
    cache, qj = X.compare(jobs[0]), X.compare(jobs[1])
    nb, valid = o["fills"].shape[1], int((o["slots"] >= 0).sum())
    assert cache["words"] == nb * M.page_bytes(X.PBS) + valid * (X.DN + 7 + X.DR) and not cache["mismatch"]
    assert qj["words"] == 48 * X.NHP * X.D and not qj["mismatch"]
    r = X.compare({**jobs[1], "paths": paths["q"]})
    assert r["mismatch"] == 2 and r["first_mismatch"]["at"] == list(QFLIPS[0]) and r["first_mismatch"]["what"] == "q"
    r = X.compare({**jobs[0], "paths": paths["nope"]})
    assert r["mismatch"] == 2 and r["first_mismatch"]["what"] == "cache" and r["first_mismatch"]["at"] == list(at["nope"])
    for k in ("pad", "unwritten"):
        r = X.compare({**jobs[0], "paths": paths[k]})
        assert r["mismatch"] == 1 and r["first_mismatch"]["at"] == list(at[k]), k


def test_the_insert_reference_job_is_the_registered_definition(padded) -> None:
    """`Dsv4QNormRopeKvInsert_v1{NH=16,...}` itself on a token against the twin, every one of its 33288 output words."""
    _, paths, _ = padded
    r = X.compare({"phase": "insert", "case": "padded", "kind": "insert-ref", "paths": paths["good"], "token": 0})
    assert {k: r[k] for k in X.TOTALS} == {"words": 0, "mismatch": 0, "reference": 1, "twin_mismatch": 0}


def test_the_indexer_jobs_take_the_kernel_words(short) -> None:
    good, bad = short
    jobs = X.indexer_jobs("short", good, 200, quick=True)
    assert [j["kind"] for j in jobs] == ["iq", "iq", "indexer-ref"] and [j["tokens"] for j in jobs[:2]] == [[0, 128], [128, 200]]
    res = [X.compare(j) for j in jobs[:2]]
    assert sum(r["words"] for r in res) == 200 * X.IH * (X.ID + 1) and not any(r["mismatch"] for r in res)
    r = X.compare({**jobs[1], "paths": bad})
    assert r["mismatch"] == 1 and r["first_mismatch"]["what"] == "q_fp8" and r["first_mismatch"]["at"] == list(IQFLIP)
    r = X.compare({**jobs[0], "paths": bad})
    assert r["mismatch"] == 1 and r["first_mismatch"]["what"] == "weights_out" and r["first_mismatch"]["at"] == list(IWFLIP)
    r = X.compare(jobs[2])
    assert {k: r[k] for k in X.TOTALS} == {"words": 0, "mismatch": 0, "reference": 1, "twin_mismatch": 0}


def test_the_jobs_split_and_reference_the_extreme_tokens() -> None:
    def of(jobs, kind):
        return [j.get("tokens", j.get("token")) for j in jobs if j["kind"] == kind]

    jobs = X.insert_jobs("long", {}, 300, False)
    assert jobs[0]["kind"] == "cache" and of(jobs, "q") == [[0, 128], [128, 256], [256, 300]] and of(jobs, "insert-ref") == [0, 299]
    assert of(X.insert_jobs("long", {}, 300, True), "insert-ref") == [0]
    assert of(X.insert_jobs("edge", {}, 40, False), "insert-ref") == list(range(40))
    assert of(X.insert_jobs("edge", {}, 40, True), "insert-ref") == [0] and of(X.insert_jobs("decode", {}, 1, False), "insert-ref") == [0]
    ij = X.indexer_jobs("odd", {}, 777, False)
    assert len(of(ij, "iq")) == 7 and of(ij, "iq")[-1] == [768, 777] and of(ij, "indexer-ref") == [0, 776]


def test_a_job_that_raises_is_an_error_of_its_case(padded) -> None:
    _, paths, _ = padded
    r = X.compare({"phase": "insert", "case": "padded", "kind": "q", "paths": {**paths["good"], "q_out": "/nonexistent.npy"},
                   "tokens": [0, 8]})
    assert (r["phase"], r["case"]) == ("insert", "padded") and "nonexistent" in r["error"]


def test_the_spy_counts_a_cached_static_method_and_restores_it() -> None:
    """`IndexerQFp8Kernel.compile` is `@cache` over `@staticmethod` (fused_indexer_q_cutedsl.py:567-568): the spy sees each call's
    positional arguments, the cache still answers, and the class gets its own attribute back; likewise a module's function."""
    class K:
        @functools.cache
        @staticmethod
        def compile(a, b=0):
            return (a, b, object())

    raw, first = vars(K)["compile"], K.compile(1, 2)
    with X.Spy(K, "compile", static=True) as s:
        assert K.compile(1, 2) is first and K.compile(3)[:2] == (3, 0)
    assert s.calls == [(1, 2), (3,)] and vars(K)["compile"] is raw and K.compile(1, 2) is first
    f = lambda x: x + 1  # noqa: E731
    mod = types.SimpleNamespace(f=f)
    with X.Spy(mod, "f") as s:
        assert mod.f(1) == 2 and mod.f is not f
    assert s.calls == [(1,)] and mod.f is f


def test_the_programs_bind_at_dsv4s_statics() -> None:
    rec = X.bind_check()
    assert rec["ok"] and len(rec["bindings"]) == 7, rec


LAUNCH = {"insert": {"b": 8, "checks": {"shape": True, "rerun": True, "untouched": True, "inputs": True}},
          "indexer": {"b": 8, "checks": {"shape": True, "rerun": True, "route": True, "coarsen": True, "inputs": True}}}


def _doc(quick: bool = False) -> dict:
    launches = [{"phase": p, "case": c, **copy.deepcopy(LAUNCH[p])} for p, c in X.cases_of(quick)]
    comps = [{"phase": p, "case": c, "words": 10, "mismatch": 0, "reference": 1, "twin_mismatch": 0, "first_mismatch": None}
             for p, c in X.cases_of(quick)]
    return {"schema": X.SCHEMA, "quick": quick, "gate": {"ok": True}, "errors": {}, "selection": {"ok": True},
            "bind": {"ok": True}, "sumsq_probe": {"case": "insert/sumsq-probe", "served_statics": False,
                                                  "scope": "diagnostic non-served eps; excluded from served case totals", "ok": True},
            "launches": launches, "comparisons": comps,
            "totals": {k: sum(r[k] for r in comps) for k in X.TOTALS}}


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("selection", "ok"), False), (("bind", "ok"), False),
                                        (("sumsq_probe", "ok"), False),
                                        (("errors", "stack"), "boom"), (("launches", 0, "checks", "untouched"), False),
                                        (("launches", 3, "checks", "rerun"), False), (("launches", 8, "checks", "coarsen"), False),
                                        (("launches", 12, "checks", "route"), False), (("comparisons", 1, "mismatch"), 1),
                                        (("comparisons", 9, "twin_mismatch"), 3), (("comparisons", 5, "reference"), 0),
                                        (("comparisons", 13, "words"), 0), (("totals", "mismatch"), 1)])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert X._verdict(_doc())
    doc = _doc()
    node = doc
    for p in path[:-1]:
        node = node[p]
    node[path[-1]] = value
    assert not X._verdict(doc)


def test_each_case_has_its_own_verdict() -> None:
    """A case is its phase and its name: the insert's `decode` fails without failing the indexer's, and the other way round; a
    quick record holds the quick cases only."""
    want = [f"insert/{c}" for c in X.INSERT_CASES] + [f"indexer/{c}" for c in X.INDEXER_CASES]
    assert list(X.cases(_doc())) == want and all(c["ok"] for c in X.cases(_doc()).values())
    doc = _doc()
    doc["launches"] = [x for x in doc["launches"] if (x["phase"], x["case"]) != ("insert", "decode")]
    doc["errors"]["compare/indexer/edge"] = "boom"
    doc["comparisons"][2]["mismatch"] = 1
    doc["errors"]["indexer/short"] = "boom"
    doc["launches"][-2]["checks"]["coarsen"] = False
    assert {k for k, c in X.cases(doc).items() if not c["ok"]} == {"insert/decode", "insert/chunk", "indexer/short",
                                                                   "indexer/wide", "indexer/edge"}
    quick = _doc(quick=True)
    assert list(X.cases(quick)) == [f"insert/{c}" for c in X.QUICK_INSERT] + [f"indexer/{c}" for c in X.QUICK_INDEXER]
    assert X._verdict(quick)


def test_main_end_to_end_on_stubbed_launches(padded, short, tmp_path, monkeypatch, capsys) -> None:
    """`main --quick` with the GPU's launches stood in for (the insert `padded` case's twin-made words as `decode`, the same with q
    flipped as `reduced`, the indexer `short` case's as `decode`) and vLLM's config module stubbed: the jobs through the spawned
    worker pool with both references, the per-case verdicts, the sealed record and the exit status, so a slip in the
    orchestration shows here and not on the H200."""
    o, paths, _ = padded
    good, _ = short
    vc = types.ModuleType("vllm.config")
    vc.CompilationConfig = vc.VllmConfig = lambda **kw: kw
    vc.set_current_vllm_config = lambda cfg: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", vc)
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    monkeypatch.setattr(X, "bind_check", lambda: {"bindings": {}, "ok": True})
    T_ = o["q"].shape[0]

    def gpu(work, outdir, quick, doc):
        doc["selection"] = {"ok": True}
        doc["sumsq_probe"] = copy.deepcopy(_doc(quick=True)["sumsq_probe"])
        doc["launches"] += [{"phase": "insert", "case": c, **copy.deepcopy(LAUNCH["insert"])} for c in ("decode", "reduced")]
        doc["launches"].append({"phase": "indexer", "case": "decode", **copy.deepcopy(LAUNCH["indexer"])})
        ins = [{**j, "case": "decode"} for j in X.insert_jobs("padded", paths["good"], T_, quick)]
        bad = [{**j, "case": "reduced", "paths": paths["q"]} for j in X.insert_jobs("padded", paths["good"], T_, quick)
               if j["kind"] != "insert-ref"]
        return ins + bad + [{**j, "case": "decode"} for j in X.indexer_jobs("short", good, 200, quick)]

    monkeypatch.setattr(X, "gpu", gpu)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1", "--quick"]) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"]
    go = {"insert/decode", "indexer/decode"}
    assert {c: v["ok"] for c, v in doc["cases"].items()} == {c: c in go for c in X.cases(_doc(quick=True))}
    assert doc["cases"]["insert/reduced"]["totals"]["mismatch"] == 2
    assert doc["cases"]["insert/decode"]["totals"]["reference"] == 1 and doc["cases"]["indexer/decode"]["totals"]["reference"] == 1
    assert doc["cases"]["indexer/decode"]["totals"]["words"] == 200 * X.IH * (X.ID + 1)
    printed = capsys.readouterr().out
    assert "DSV4-QNORM-INSERT NO-GO" in printed
    assert f"no-go cases={[c for c in X.cases(_doc(quick=True)) if c not in go]}" in printed


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["dsv4_qnorm_insert_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)
    assert doc["sumsq_probe"]["served_statics"] is False and "insert/sumsq-probe" not in X.cases(doc)
