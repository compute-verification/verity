"""The DeepSeek-V4-Pro indexer capture without a GPU: the numpy twins' word steps are `dsv4_indexer.WORD`'s primitives on special
and random words, each twin is its Definition (the score row, the cache entry of the state rows the kernel reads, the head, the
fill), the state rows' and the head layout's wiring, the operands' edges, each comparison job takes the kernel's words and finds a
flipped one, the top-k checks at 1024, the PTX scan, the bindings, the statics, the verdict and `main` end to end (the GPU run is
`tests/properties/dsv4_indexer_capture_gpu.py --out ...` on an H200)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.properties import dsv4_indexer_capture_gpu as D4
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity_vllm.program.registry import dsa
from verity_vllm.properties.record import REGISTRY, digest_of

I, X = D4.I, D4.X
SPECIAL = (0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x3F800000, 0xBF800000, 0x3F000000, 0x7F7FFFFF,
           0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7FFFFFFF, 0xFFC00001, 0x7F800001, 0x43E00000, 0x43E00001, 0xC3E00000,
           0x2EDBE6FF, 0x38D1B717, 0xC2FC0000, 0xC2FC0001, 0x42FE0000, 0x4B000000, 0xCB000001)


def _words(seed: int, n: int = 64) -> np.ndarray:
    """SPECIAL, uniform words and f32 values of every exponent."""
    rng = np.random.default_rng(seed)
    vals = rng.standard_normal(n).astype(np.float32) * np.exp2(rng.integers(-140, 128, n)).astype(np.float32)
    return np.concatenate([np.array(SPECIAL, np.uint32), rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32),
                           vals.view(np.uint32)])


def _f32(rng: np.random.Generator, *shape: int, scale: float = 1.0) -> np.ndarray:
    return (rng.standard_normal(shape).astype(np.float32) * np.float32(scale)).view(np.uint32)


def _bf16(rng: np.random.Generator, *shape: int, scale: float = 1.0) -> np.ndarray:
    return (_f32(rng, *shape, scale=scale) >> 16).astype(np.uint16)


def _cs(rng: np.random.Generator, n: int) -> np.ndarray:
    ang = rng.uniform(0, 2 * np.pi, (n, D4.ROPE // 2))
    return np.concatenate([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32).view(np.uint32)


def _total(results: list[dict]) -> dict:
    assert not [r for r in results if "error" in r], results
    return {k: sum(r[k] for r in results) for k in D4.TOTALS}


# ------------------------------------------------------------------------------------------------------------- the word steps
def _operands(name: str, w: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Each step's operands: every word for one operand, every pair of SPECIAL and random pairs for two, random triples for three."""
    arity = len(I.WORD[name].signature()[0])
    if name == "cvt":
        return [np.unique(np.concatenate([w >> 16, w & 0xFFFF])).astype(np.uint32)]
    if name == "fsel":
        a, b = rng.choice(w, 400), rng.choice(w, 400)
        return [np.arange(400, dtype=np.uint32) & 1, a, b]
    if arity == 1:
        return [w]
    if arity == 2:
        s = np.array(SPECIAL, np.uint32)
        a, b = np.meshgrid(s, s)
        return [np.concatenate([a.ravel(), rng.choice(w, 600)]), np.concatenate([b.ravel(), rng.choice(w, 600)])]
    return [rng.choice(w, 2000) for _ in range(3)]


@pytest.mark.parametrize("name", sorted(I.WORD) + ["neg"])
def test_the_numpy_steps_are_the_word_primitives(name: str) -> None:
    """`NumpyOps` runs the Definitions' bodies over arrays: each step is its primitive word for word (NaN, inf, subnormals, the
    MUFU tables and div.full's range scaling, e4m3 saturation, the i32 wraps and comparisons)."""
    rng = np.random.default_rng(sorted(I.WORD).index(name) if name in I.WORD else 99)
    prim = I.WORD.get(name, I.F32Neg)
    args = _operands(name if name != "neg" else "fabs", _words(1), rng)
    with np.errstate(all="ignore"):
        got = np.broadcast_to(getattr(D4.OPS, name)(*args), args[0].shape).astype(np.int64)
    want = np.array([evaluate(prim, *(int(a[i]) for a in args))[0] for i in range(args[0].size)], dtype=np.int64)
    bad = np.flatnonzero(got != want)
    assert bad.size == 0, [(name, [hex(int(a[i])) for a in args], hex(int(want[i])), hex(int(got[i]))) for i in bad[:4]]


# ------------------------------------------------------------------------------------------------------------- the twins
def test_the_score_twin_is_the_definition() -> None:
    rng = np.random.default_rng(2)
    score = _f32(rng, 3, D4.STATE_W, scale=4.0)
    score[1] = np.resize(_words(3), D4.STATE_W)
    ape = _f32(rng, 3, D4.STATE_W, scale=0.5)
    ape[2, :len(SPECIAL)] = SPECIAL
    want = D4.score_twin(score, ape)
    fn = bind(I.Dsv4CompressorScore, W=D4.STATE_W)
    assert [list(evaluate(fn, score[i].tolist(), ape[i].tolist())) for i in range(3)] == want.astype(np.int64).tolist()


def _entry_operands(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Five cache entries' state rows, k_norm's weight and rotary rows: random; the rows before position 0 (kv 0, score -inf);
    kv x 2^70 with a NaN, an inf and a -inf, every score -inf; bf16-subnormal kv with a one-hot score; every score +inf, the rotary
    row cos[2] inf and sin[5] NaN."""
    n = 5
    kv = _f32(rng, n, D4.ROWS, D4.D, scale=2.0)
    sc = _f32(rng, n, D4.ROWS, D4.D, scale=4.0)
    kv[1, :4], sc[1, :4] = 0, D4.NEG_INF
    kv[2] = (D4._f(kv[2]) * np.float32(2.0 ** 70)).view(np.uint32)
    kv[2, 5, 3], kv[2, 6, 5], kv[2, 7, 7] = 0x7FC00000, 0x7F800000, D4.NEG_INF
    sc[2] = D4.NEG_INF
    kv[3] = (D4._f(kv[3]) * np.float32(2.0 ** -140)).view(np.uint32)
    sc[3] = np.float32(-1e30).view(np.uint32)
    sc[3, 3] = 0
    sc[4] = 0x7F800000
    w = (1 + 0.1 * rng.standard_normal(D4.D)).astype(np.float32).view(np.uint32) >> 16
    w[3], w[100] = 0, 0x4280
    cs = _cs(rng, n)
    cs[4, 2], cs[4, D4.ROPE // 2 + 5] = 0x7F800000, 0x7FC00000
    return kv, sc, w.astype(np.uint16), cs


def test_the_entries_twin_is_the_definition() -> None:
    kv, sc, w, cs = _entry_operands(np.random.default_rng(4))
    q, s = D4.entries_twin(kv, sc, w, cs)
    fn = bind(I.Dsv4IndexerCompress, D=D4.D, ROPE=D4.ROPE, EPS=D4.EPS)
    for i in range(kv.shape[0]):
        y = evaluate(fn, kv[i].reshape(-1).tolist(), sc[i].reshape(-1).tolist(), w.tolist(), cs[i].tolist())
        assert list(y[:D4.D]) == q[i].tolist() and y[D4.D] == s[i], i
    floor = 0x34800000
    assert s[2] == s[3] == s[4] == floor and (q[2] == 0x7E).all() and (q[4] == 0x7E).all() and set(q[3].tolist()) <= {0, 0x80}
    assert len(set(q[0].tolist())) > 32 and s[0] != floor


def _head_operands(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Seven heads: random; zero; x 2^100; bf16 subnormals; +inf at 5, NaN at 450, -inf at 500; multiples of 448; one outlier on
    a rotary row of cos 0 and sin -1."""
    x = _bf16(rng, 7, D4.HD, scale=3.0)
    x[1] = 0
    x[2] = (np.asarray(D4._f(x[2].astype(np.uint32) << 16), np.float64) * 2.0 ** 100).astype(np.float32).view(np.uint32) >> 16
    x[3] = _bf16(rng, D4.HD, scale=2.0 ** -130)
    x[4, 5], x[4, 450], x[4, 500] = 0x7F80, 0x7FC0, 0xFF80
    x[5] = (448.0 * np.exp2(rng.integers(-8, 8, D4.HD))).astype(np.float32).view(np.uint32) >> 16
    x[6] = _bf16(rng, D4.HD, scale=2.0 ** -20)
    x[6, 130] = 0x4980
    cs = _cs(rng, 7)
    cs[6, :D4.ROPE // 2], cs[6, D4.ROPE // 2:] = 0, np.float32(-1).view(np.uint32)
    cs[4, 2] = 0x7F800000
    return x.astype(np.uint16), cs


def test_the_heads_twin_is_the_definition() -> None:
    x, cs = _head_operands(np.random.default_rng(5))
    q, s = D4.heads_twin(x, cs)
    fn = bind(I.Dsv4InvRopeQuantHead, CHUNKS=D4.CHUNKS, QG=D4.QG, ROPE=D4.ROPE)
    for i in range(x.shape[0]):
        y = evaluate(fn, x[i].tolist(), cs[i].tolist())
        assert list(y[:D4.HD]) == q[i].tolist() and list(y[D4.HD:]) == s[i].tolist(), i
    assert (s[1] == 0x2A800000).all() and set(q[1].tolist()) <= {0, 0x80}
    assert s[4, 0] == s[4, 3] == 0x7F800000 and s[4, 1] != 0x7F800000


def test_the_fill_twin_is_the_definition() -> None:
    p = D4.fill_positions()
    assert p.size == len(D4.FILL_POSITIONS) + 2 * D4.FILL_RANDOM and (p < 0).sum() > 8
    sel = np.concatenate([np.arange(len(D4.FILL_POSITIONS)), len(D4.FILL_POSITIONS) + np.arange(0, 2 * D4.FILL_RANDOM, 8)])
    want = D4.fill_twin(p[sel])
    fn = bind(I.Dsv4ShortContextTopK, TOPK=D4.TOPK, RATIO=D4.RATIO)
    for i, pos in zip(sel, p[sel]):
        assert list(evaluate(fn, int(pos) & D4.M64)) == (want[list(sel).index(i)].astype(np.int64) & 0xFFFFFFFF).tolist(), int(pos)
    n = (want >= 0).sum(axis=1)
    assert n[list(p[sel]).index(2 ** 63 - 1)] == 0 and n[list(p[sel]).index(4095)] == D4.TOPK and n[list(p[sel]).index(11)] == 3


# --------------------------------------------------------------------------------------------------------------- the wiring
def test_the_state_rows_are_the_kernel_reads() -> None:
    """Row r of the token at position p is position p - 7 + r of its request through the block table, rows 0-3 the overlap halves
    [0:128] / [256:384], rows 4-7 [128:256] / [384:512]; rows before position 0 kv 0 and score -inf."""
    nb = 6
    b, r, c = np.meshgrid(np.arange(nb), np.arange(D4.STATE_BLOCK), np.arange(2 * D4.STATE_W), indexing="ij")
    state = (b * 100000 + r * 1000 + c).astype(np.uint32)
    bt = np.array([[4, 1, 5, 0], [2, 3, 0, 0]], dtype=np.int32)
    req, pos = np.array([0, 0, 1]), np.array([3, 11, 7])
    kv, sc = D4.state_rows(state, bt, req, pos)
    assert kv.shape == sc.shape == (3, D4.ROWS, D4.D)
    for t in range(3):
        for row in range(D4.ROWS):
            p = pos[t] - 7 + row
            if p < 0:
                assert (kv[t, row] == 0).all() and (sc[t, row] == D4.NEG_INF).all()
                continue
            lo = 0 if row < D4.RATIO else D4.D
            words = state[bt[req[t], p // 4], p % 4]
            assert kv[t, row].tolist() == words[lo:lo + D4.D].tolist()
            assert sc[t, row].tolist() == words[D4.STATE_W + lo:D4.STATE_W + lo + D4.D].tolist()


def test_the_layout_writes_each_slot_once() -> None:
    o = D4.layout(20261013)
    pos, req, sslot, kslot = o["pos"], o["req"], o["sslot"], o["kslot"]
    written = sslot >= 0
    assert np.unique(sslot[written]).size == written.sum() == pos.size - D4.EXTRA // 2
    assert (sslot[written] == o["sbt"][req[written], pos[written] // 4] * 4 + pos[written] % 4).all()
    entries = (kslot >= 0) & written
    assert [int(entries[req == k].sum()) for k in range(3)] == [D4.SEQ // 4, D4.EDGE_SEQ // 4, 0]
    assert np.unique(kslot[entries]).size == entries.sum() and (kslot[entries] < D4.NBK * D4.BLOCK).all()
    assert ((pos[entries] + 1) % 4 == 0).all()
    pre = {tuple(x) for x in o["prefilled"].tolist()}
    first = np.flatnonzero((req == 1) & entries)[0]
    assert {(int(o["sbt"][1, (pos[first] - 7 + r) // 4]), int((pos[first] - 7 + r) % 4)) for r in range(4)} <= pre
    assert not ({(int(s) // 4, int(s) % 4) for s in sslot[(req == 0) & written]} & pre)
    for c in D4.CS_SPECIAL:
        t = np.flatnonzero(pos == c + 3)
        assert t.size == 1 and entries[t[0]] and req[t[0]] == 1


def test_the_operands_carry_their_edges() -> None:
    o = D4.operands(20261013, dev="cpu")
    ks = o["kv_score"].numpy()
    sc, kv = ks[D4.SEQ:, D4.STATE_W:], ks[D4.SEQ:, :D4.STATE_W]
    assert np.isinf(sc[50]).all() and np.isnan(sc[60]).all() and (sc[64:72] == -np.inf).all() and (sc[32:40] == 0).all()
    assert np.isnan(kv[110, 3]) and kv[111, 5] == np.inf and kv[112, 7] == -np.inf and kv[150, 9] == np.float32(3e38)
    assert (np.abs(kv[88:96]) > 2.0 ** 60).mean() > 0.9 and (np.abs(kv[96:104]) < 2.0 ** -126).all()
    assert (sc[131] == 0).all() and (sc[[128, 129, 130, 132]] == np.float32(-1e30)).all()
    assert np.isfinite(ks[:D4.SEQ]).all()
    state = o["state0"].view(torch.int32).numpy().view(np.uint32)
    b, r = o["prefilled"][:, 0], o["prefilled"][:, 1]
    mask = np.ones(state.shape[:2], dtype=bool)
    mask[b, r] = False
    assert (state[mask] == D4.SENTINEL_STATE).all() and (state[b, r] != D4.SENTINEL_STATE).all()
    assert o["w"].dtype == torch.bfloat16 and o["w"][3] == 0 and o["w"][100] == 64
    cs = o["cs"].numpy()
    a, b2, c = D4.CS_SPECIAL
    assert cs[a, 2] == np.inf and np.isnan(cs[b2, D4.ROPE // 2 + 5]) and (cs[c, :32] == 0).all() and (cs[c, 32:] == -1).all()
    assert np.isfinite(np.delete(cs, [a, b2], axis=0)).all()


def test_the_head_and_query_operands_carry_their_edges() -> None:
    x, pos = D4.head_operands(77, dev="cpu")
    u = x.view(torch.int16).numpy().view(np.uint16)
    assert x.shape == (D4.INV_TOKENS, D4.NH, D4.HD) and (u[0, 0] == 0).all() and pos[1:4].tolist() == list(D4.CS_SPECIAL)
    assert u[0, 3, 5] == 0x7F80 and (u[0, 4, 450] & 0x7FFF) > 0x7F80 and u[0, 5, 500] == 0xFF80
    assert ((u[0, 2] & 0x7F80) == 0).mean() > 0.9 and (pos >= 0).all() and (pos < D4.MAXPOS).all()
    q, w = D4.queries(5, 16, dev="cpu")
    assert q.dtype == torch.float8_e4m3fn and q.shape == (16, D4.H, D4.D) and (q[6, 0].float() == 0).all() and (w[7] == 0).all()
    assert not torch.isnan(q.float()).any()


def test_the_head_words_layout() -> None:
    """Head 8 g + i of `fused_inv_rope_fp8_quant`'s (T, G, 8 * 512) bytes and (T, G, 8 * 4) scales is [t, g, 512 i:512 (i + 1)]
    and [t, g, 4 i:4 i + 4]."""
    T = 3
    t, g, j = np.meshgrid(np.arange(T), np.arange(D4.GROUPS), np.arange(D4.HPG * D4.HD), indexing="ij")
    fp8 = (t * 7 + g * 3 + j // D4.HD).astype(np.uint8)
    t, g, j = np.meshgrid(np.arange(T), np.arange(D4.GROUPS), np.arange(D4.HPG * D4.CHUNKS), indexing="ij")
    scale = (t * 1000 + g * 100 + j).astype(np.uint32)
    q, s = D4.head_words(fp8, scale)
    assert q.shape == (T, D4.NH, D4.HD) and s.shape == (T, D4.NH, D4.CHUNKS)
    for tt in range(T):
        for h in range(D4.NH):
            gg, i = divmod(h, D4.HPG)
            assert (q[tt, h] == tt * 7 + gg * 3 + i).all()
            assert s[tt, h].tolist() == [tt * 1000 + gg * 100 + 4 * i + c for c in range(D4.CHUNKS)]


# ---------------------------------------------------------------------------------------------------------- the comparisons
def _score_job(tmp_path: Path, flip: bool = False) -> dict:
    rng = np.random.default_rng(6)
    n = 40
    score = _f32(rng, n, D4.STATE_W, scale=4.0)
    score[3] = np.resize(_words(7), D4.STATE_W)
    ape = _f32(rng, n, D4.STATE_W, scale=0.5)
    got = D4.score_twin(score, ape)
    if flip:
        got[17, 200] ^= 1
    paths = D4._save(tmp_path / "score", score=score, ape=ape, got=got, tokens=np.arange(n) + 1000)
    return {"phase": "score", "case": "served", "kind": "score", "paths": paths, "ref_every": 16}


def test_the_score_jobs_take_the_kernel_words(tmp_path) -> None:
    job = _score_job(tmp_path)
    t = _total([D4.compare({**job, "rows": [a, b]}) for a, b in D4._chunks(40, 16)])
    assert t == {"words": 40 * D4.STATE_W, "mismatch": 0, "unresolved": 0, "reference": 3, "twin_mismatch": 0}
    r = D4.compare({**_score_job(tmp_path, flip=True), "rows": [0, 40]})
    assert r["mismatch"] == 1 and r["first_mismatch"] == {"token": 1017, "word": 200, "want": r["first_mismatch"]["want"],
                                                          "kernel": r["first_mismatch"]["want"] ^ 1}


def _entries_job(tmp_path: Path, flip: bool = False) -> dict:
    kv, sc, w, cs = _entry_operands(np.random.default_rng(8))
    q, s = D4.entries_twin(kv, sc, w, cs)
    if flip:
        q[2, 77] ^= 1
    paths = D4._save(tmp_path / "entries", kv=kv, score=sc, w=w, cs=cs, q=q, s=s, tokens=np.array([3, 7, 6147, 6151, 6155]))
    return {"phase": "compress", "case": "edge", "kind": "entries", "paths": paths, "ref_every": 1}


def test_the_entries_jobs_take_the_kernel_bytes(tmp_path) -> None:
    job = _entries_job(tmp_path)
    t = _total([D4.compare({**job, "rows": [a, b]}) for a, b in D4._chunks(5, 3)])
    assert t == {"words": 5 * (D4.D + 1), "mismatch": 0, "unresolved": 0, "reference": 5, "twin_mismatch": 0}
    r = D4.compare({**_entries_job(tmp_path, flip=True), "rows": [1, 3]})
    assert r["mismatch"] == 1 and r["first_mismatch"]["position"] == 6147 and r["first_mismatch"]["word"] == 77
    assert r["twin_mismatch"] == 0 and r["reference"] == 2


def _heads_job(tmp_path: Path, flip: bool = False) -> dict:
    x, cs = _head_operands(np.random.default_rng(9))
    q, s = D4.heads_twin(x, cs)
    if flip:
        s[4, 3] ^= 1 << 23
    paths = D4._save(tmp_path / "heads", x=x, cs=cs, q=q, s=s, ids=np.arange(7) + 32)
    return {"phase": "inv_rope", "case": "served", "kind": "heads", "paths": paths, "ref_every": 1}


def test_the_heads_jobs_take_the_kernel_words(tmp_path) -> None:
    t = _total([D4.compare({**_heads_job(tmp_path), "rows": [a, b]}) for a, b in D4._chunks(7, 4)])
    assert t == {"words": 7 * (D4.HD + D4.CHUNKS), "mismatch": 0, "unresolved": 0, "reference": 7, "twin_mismatch": 0}
    r = D4.compare({**_heads_job(tmp_path, flip=True), "rows": [0, 7]})
    assert r["mismatch"] == 1 and r["first_mismatch"]["head_id"] == 36 and r["first_mismatch"]["word"] == D4.HD + 3


def _fill_job(tmp_path: Path, flip: bool = False) -> dict:
    p = D4.fill_positions()[:48]
    got = D4.fill_twin(p)
    if flip:
        got[20, 5] = 5 if got[20, 5] < 0 else -1
    paths = D4._save(tmp_path / "fill", pos=p, got=got)
    return {"phase": "fill", "case": "served", "kind": "fill", "paths": paths, "ref_every": 16}


def test_the_fill_jobs_take_the_kernel_words(tmp_path) -> None:
    t = _total([D4.compare({**_fill_job(tmp_path), "rows": [a, b]}) for a, b in D4._chunks(48, 32)])
    assert t == {"words": 48 * D4.TOPK, "mismatch": 0, "unresolved": 0, "reference": 3, "twin_mismatch": 0}
    r = D4.compare({**_fill_job(tmp_path, flip=True), "rows": [16, 32]})
    assert r["mismatch"] == 1 and r["first_mismatch"]["position"] == int(D4.fill_positions()[20]) and r["first_mismatch"]["word"] == 5


def _e4m3(rng: np.random.Generator, *shape: int) -> np.ndarray:
    b = rng.integers(0, 256, shape).astype(np.uint8)
    return np.where((b & 0x7F) == 0x7F, b ^ 1, b).astype(np.uint8)


def _logits_job(tmp_path: Path, flip: bool = False) -> list[dict]:
    """A 6-token prefill over 3 keys (row r's window [0, (r + 1) // 2)) and a decode of the last two tokens on every key, at 64
    heads, as `phase_logits` saves them."""
    rng = np.random.default_rng(10)
    T, K = 6, 3
    q, k = _e4m3(rng, T, D4.H, D4.D), _e4m3(rng, K, D4.D)
    q[0] = 0
    w = (rng.standard_normal((T, D4.H)).astype(np.float32) * 0.1).view(np.uint32)
    ks = np.exp2(rng.integers(-8, 0, K)).astype(np.float32).view(np.uint32)
    r = np.concatenate([np.full((i + 1) // 2, i) for i in range(T)])
    c = np.concatenate([np.arange((i + 1) // 2) for i in range(T)])
    got = X.logit_rows(q[r], w[r], k[c], ks[c])[0]
    rd, cd = np.array([4, 4, 4, 5, 5, 5]), np.array([0, 1, 2, 0, 1, 2])
    gotd = X.logit_rows(q[rd], w[rd], k[cd], ks[cd])[0]
    if flip:
        gotd[4] ^= 1
    paths = D4._save(tmp_path / "logits", q=q, w=w, k=k, ks=ks, r=r, c=c, got=got, rd=rd, cd=cd, gotd=gotd)
    return [{"phase": "logits", "case": "prefill", "kind": "logits", "paths": paths, "keys": ["r", "c", "got"], "range": [0, r.size]},
            {"phase": "logits", "case": "decode", "kind": "logits", "paths": paths, "keys": ["rd", "cd", "gotd"], "range": [0, rd.size]}]


def test_the_logits_twin_is_the_reference_at_64_heads() -> None:
    rng = np.random.default_rng(11)
    n = 6
    q, k = _e4m3(rng, n, D4.H, D4.D), _e4m3(rng, n, D4.D)
    q[1, 3] = 0
    w = (rng.standard_normal((n, D4.H)).astype(np.float32) * 0.1).view(np.uint32)
    ks = np.exp2(rng.integers(-8, 0, n)).astype(np.float32).view(np.uint32)
    want, nonfin = X.logit_rows(q, w, k, ks)
    fn = bind(dsa.DsaMqaLogitV3, H=D4.H, D=D4.D)
    assert not nonfin.any() and len(set(want.tolist())) == n
    assert want.tolist() == [evaluate(fn, q[i].reshape(-1).tolist(), w[i].tolist(), k[i].tolist(), int(ks[i]))[0] for i in range(n)]


def test_the_logits_jobs_take_the_kernel_words(tmp_path) -> None:
    res = [D4.compare(j) for j in _logits_job(tmp_path)]
    assert _total(res) == {"words": 9 + 6, "mismatch": 0, "unresolved": 0, "reference": 4, "twin_mismatch": 0}
    res = [D4.compare(j) for j in _logits_job(tmp_path, flip=True)]
    assert (res[0]["mismatch"], res[1]["mismatch"]) == (0, 1) and res[1]["first_mismatch"]["row"] == 5
    assert res[1]["first_mismatch"]["key"] == 1


def test_a_failing_job_records_its_error(tmp_path) -> None:
    job = {**_score_job(tmp_path), "rows": [0, 4]}
    job["paths"] = {**job["paths"], "ape": str(tmp_path / "missing.npy")}
    r = D4.compare(job)
    assert set(r) == {"phase", "case", "error"} and "missing.npy" in r["error"]


# ------------------------------------------------------------------------------------------------------------------ top-k
def test_the_short_rows_reference_is_the_closed_form_at_1024() -> None:
    fn = bind(dsa.DsaTopKShortRow, TOPK=D4.TOPK)
    for n in (0, 1, 2, D4.TOPK - 1, D4.TOPK, -3):
        want = X.topk_short(np.array([max(n, 0)]), D4.TOPK)[0]
        assert list(evaluate(fn, n & 0xFFFFFFFF)) == (want.astype(np.int64) & 0xFFFFFFFF).tolist(), n


def test_the_topk_checks() -> None:
    vals, lens, names = D4.topk_rows(11)
    assert names[-1] == "negative" and lens[-1] == -3 and np.isinf(vals[0]).all() and vals.shape[1] == D4.MAX_MODEL_LEN
    out = X.topk_short(np.maximum(lens, 0), D4.TOPK)
    for i in np.flatnonzero(lens > D4.TOPK):
        out[i] = np.sort(np.argsort(-vals[i, :lens[i]], kind="stable")[:D4.TOPK])
    got = D4.check_topk(vals, lens, out)
    assert (got["short_rows"], got["short_mismatch"], got["long_rows"], got["long_valid"]) == (6, 0, 4, 4)
    assert got["long_ascending"] == 4 and got["long_with_ties"] == 2
    bad = out.copy()
    bad[2, 1] = -1
    bad[lens.tolist().index(2048), 7] = bad[lens.tolist().index(2048), 8]
    got = D4.check_topk(vals, lens, bad)
    assert got["short_mismatch"] == 1 and got["long_valid"] == 3
    ones = lens.tolist().index(1100)
    sel = out[ones]
    worse = sel.copy()
    worse[np.flatnonzero(vals[ones, sel] == 1)[0]] = np.setdiff1d(np.flatnonzero(vals[ones, :1100] == 0), sel)[0]
    assert D4.topk_valid(vals[ones], 1100, sel)["valid"] and D4.topk_valid(vals[ones], 1100, worse)["valid"] is False
    wide = np.full((len(lens), D4.MAX_MODEL_LEN + 13), np.inf, dtype=np.float32)
    wide[:, 5:5 + D4.MAX_MODEL_LEN] = vals
    assert D4.check_topk(wide, np.maximum(lens, 0), out, 5)["long_valid"] == 4


def test_the_logits_rerun_compares_the_words_each_row_writes() -> None:
    """With clean_logits=False the words past row r's window [0, ke[r]) are whatever the buffer held."""
    a = torch.randn(5, 7)
    b = a.clone()
    ke = torch.tensor([1, 2, 7, 0, 3], dtype=torch.int32)
    b[0, 6], b[3, 0] = float("nan"), 3.0
    assert D4.logits_rerun(a, b, ke) == (True, 2)
    b[2, 4] = torch.nextafter(a[2, 4], torch.tensor(float("inf")))
    assert D4.logits_rerun(a, b, ke) == (False, 2)


# ------------------------------------------------------------------------------------------------------------ ptx, bind, statics
def test_the_ptx_scan_wants_one_specialization_per_kernel(tmp_path) -> None:
    body = ("fma.rn.f32 %f1, %f2, %f3, %f4;\ndiv.full.f32 %f5, %f2, %f3;\nex2.approx.ftz.f32 %f6, %f1;\n"
            "min.xorsign.abs.f32 %f7, %f6, %f5;\nmax.f32 %f8, %f7, %f1;\n.maxntid 32, 1, 1\n")
    for i, k in enumerate(D4.KERNELS):
        (tmp_path / "c" / str(i)).mkdir(parents=True)
        (tmp_path / "c" / str(i) / f"{k}.ptx").write_text(body)
    got = D4.ptx(tmp_path / "c", tmp_path / "out")
    assert got["ok"] and got["kernels"][D4.KERNELS[2]][0]["counts"] == {"fma.rn.f32": 1, "div.full.f32": 1, "ex2.approx.ftz.f32": 1,
                                                                        "min.xorsign.abs.f32": 1, "max.f32": 1}
    assert got["kernels"][D4.KERNELS[0]][0]["maxntid"] == ["32, 1, 1"]
    assert len(list((tmp_path / "out" / "ptx").glob("*.ptx"))) == len(D4.KERNELS)
    (tmp_path / "c" / "9").mkdir()
    (tmp_path / "c" / "9" / f"{D4.KERNELS[1]}.ptx").write_text(body + "add.f32 %f9, %f1, %f2;\n")
    assert D4.ptx(tmp_path / "c", tmp_path / "out")["ok"] is False


def test_the_programs_bind_at_dsv4s_statics() -> None:
    got = D4.bind_check()
    assert got["ok"] and len(got["bindings"]) == 12
    assert "DsaMqaLogit_v3{H=64,D=128}" in got["bindings"] and "Dsv4IndexerCompress_v1{D=128,ROPE=64,EPS=1e-06}" in got["bindings"]


FIXTURE = Path(__file__).parents[1] / "check" / "fixtures" / "deepseek_v4" / "config.json"


@pytest.mark.skipif(not FIXTURE.exists(), reason="DeepSeek-V4-Pro's config fixture lands with cursor/dsv4-prep-8c79")
def test_the_statics_are_dsv4s() -> None:
    cfg = json.loads(FIXTURE.read_text())
    assert (cfg["index_n_heads"], cfg["index_head_dim"], cfg["index_topk"], cfg["qk_rope_head_dim"]) == (D4.H, D4.D, D4.TOPK, D4.ROPE)
    assert D4.RATIO in cfg["compress_ratios"] and cfg["rms_norm_eps"] == D4.EPS and cfg["compress_rope_theta"] == D4.THETA
    assert (cfg["num_attention_heads"] // 8, cfg["head_dim"], cfg["o_groups"] // 8) == (D4.NH, D4.HD, D4.GROUPS)
    assert cfg["head_dim"] - cfg["qk_rope_head_dim"] == D4.NOPE


# ----------------------------------------------------------------------------------------------------------------- verdict
TOT = {"words": 10, "mismatch": 0, "unresolved": 0, "reference": 1, "twin_mismatch": 0}
LAUNCHES = [{"phase": "score", "case": "served", "checks": {"deterministic": True, "kv_copied": True}},
            {"phase": "compress", "case": "served", "checks": {"state_unchanged": True}},
            {"phase": "compress", "case": "edge", "checks": {"entries": True}},
            {"phase": "fill", "case": "served", "checks": {"deterministic": True}},
            {"phase": "inv_rope", "case": "served", "checks": {"scale_padding_zero": True}},
            {"phase": "logits", "case": "prefill", "checks": {"gather": True}},
            {"phase": "logits", "case": "decode", "checks": {"shape": True}}]


def _doc() -> dict:
    return {"schema": D4.SCHEMA, "gate": {"ok": True}, "errors": {}, "selection": {"ok": True}, "ptx": {"ok": True},
            "bind": {"ok": True}, "topk": {"ok": True}, "launches": copy.deepcopy(LAUNCHES),
            "comparisons": [{"phase": p, "case": c, **TOT} for p, c in D4.CASES],
            "totals": {k: v * len(D4.CASES) for k, v in TOT.items()}}


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("errors", "fill"), "boom"), (("selection", "ok"), False),
                                        (("ptx", "ok"), None), (("bind", "ok"), False), (("topk", "ok"), False),
                                        (("totals", "mismatch"), 1), (("totals", "unresolved"), 1), (("totals", "reference"), 0),
                                        (("totals", "twin_mismatch"), 1), (("totals", "words"), 0),
                                        (("launches", 0, "checks", "kv_copied"), False), (("launches", 4, "checks"), {}),
                                        (("comparisons", 5, "mismatch"), 1), (("comparisons", 1, "reference"), 0),
                                        (("errors", "compare/inv_rope/served"), "boom")])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert D4._verdict(_doc())
    doc = _doc()
    node = doc
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    assert not D4._verdict(doc)


def test_each_case_has_its_own_verdict() -> None:
    assert all(c["ok"] for c in D4.cases(_doc()).values())
    assert list(D4.cases(_doc())) == ["score/served", "compress/served", "compress/edge", "fill/served", "inv_rope/served",
                                      "logits/prefill", "logits/decode"]
    for drop in ("launches", "comparisons"):
        doc = _doc()
        doc[drop] = [x for x in doc[drop] if x["phase"] != "fill" and x["case"] != "edge"]
        assert {k for k, c in D4.cases(doc).items() if not c["ok"]} == {"fill/served", "compress/edge"}, drop
    doc = _doc()
    doc["errors"]["compress"] = "boom"
    assert {k for k, c in D4.cases(doc).items() if not c["ok"]} == {"score/served", "compress/served", "compress/edge"}
    doc = _doc()
    doc["errors"]["compare/logits/decode"] = "boom"
    doc["comparisons"][0]["twin_mismatch"] = 1
    assert {k for k, c in D4.cases(doc).items() if not c["ok"]} == {"score/served", "logits/decode"}


def _stub_vllm(monkeypatch, gate: dict) -> None:
    """vLLM's config module, the device gate and the environment stood in for; the env `run` sets restored after the test."""
    import contextlib
    import sys
    import types

    vc = types.ModuleType("vllm.config")
    vc.CompilationConfig = vc.VllmConfig = lambda **kw: kw
    vc.set_current_vllm_config = lambda cfg: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", vc)
    for k in ("VLLM_ENABLE_V1_MULTIPROCESSING", "TRITON_CACHE_DIR", "DG_JIT_CACHE_DIR"):
        monkeypatch.setenv(k, "")
    monkeypatch.setattr(D4.C, "gate", lambda *a: gate)
    monkeypatch.setattr(D4.G, "_environment", lambda: {})


def test_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys) -> None:
    """`main` with the GPU's phases stood in for (the comparison jobs of the tests above, one word flipped in the decode logits,
    no compress/served launch) and vLLM's config module stubbed: the jobs through the spawned worker pool, the per-case verdicts,
    the sealed record and the exit status, so a slip in the orchestration shows here and not on the H200."""
    _stub_vllm(monkeypatch, {"ok": True})
    monkeypatch.setattr(D4, "ptx", lambda cache, outdir: {"kernels": {}, "ok": True})
    monkeypatch.setattr(D4, "bind_check", lambda: {"bindings": {}, "ok": True})
    jobs = [{**_score_job(tmp_path / "s"), "rows": [0, 20]}, {**_fill_job(tmp_path / "f"), "rows": [0, 48]},
            {**_heads_job(tmp_path / "h"), "rows": [0, 3]}, {**_entries_job(tmp_path / "e"), "rows": [0, 2]},
            *_logits_job(tmp_path / "l", flip=True)]

    def gpu(work, outdir, quick, doc):
        doc["selection"] = {"ok": True}
        doc["topk"] = {"ok": True, "cases": {}}
        doc["launches"] += [x for x in copy.deepcopy(LAUNCHES) if x["phase"] != "compress" or x["case"] == "edge"]
        return jobs

    monkeypatch.setattr(D4, "gpu", gpu)
    out = tmp_path / "out"
    assert D4.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "2"]) == 1
    doc = json.loads((out / D4.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"] and doc["schema"] == D4.SCHEMA
    assert {c: v["ok"] for c, v in doc["cases"].items()} == {"score/served": True, "compress/served": False, "compress/edge": True,
                                                             "fill/served": True, "inv_rope/served": True, "logits/prefill": True,
                                                             "logits/decode": False}
    assert doc["totals"]["mismatch"] == 1 and doc["totals"]["words"] == (20 * D4.STATE_W + 48 * D4.TOPK + 3 * (D4.HD + D4.CHUNKS)
                                                                         + 2 * (D4.D + 1) + 9 + 6)
    assert doc["env"]["DG_JIT_CACHE_DIR"].endswith("deep_gemm") and doc["env"]["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    printed = capsys.readouterr().out
    assert f"DSV4-INDEXER NO-GO {doc['digest']}" in printed and "no-go cases=['compress/served', 'logits/decode']" in printed


def test_the_gate_failure_stops_the_run(tmp_path, monkeypatch, capsys) -> None:
    _stub_vllm(monkeypatch, {"ok": False, "why": "cc 8.9"})
    monkeypatch.setattr(D4, "gpu", lambda *a: pytest.fail("the GPU phases ran after a failed gate"))
    with pytest.raises(SystemExit) as e:
        D4.main(["--out", str(tmp_path / "out"), "--scratch", str(tmp_path)])
    doc = json.loads((tmp_path / "out" / D4.RECORD).read_text())
    assert e.value.code == 3 and doc["ok"] is False and doc["digest"] == digest_of(doc)
    assert capsys.readouterr().out.rstrip().endswith("DSV4-INDEXER GATE-FAIL")


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), D4._verdict(_doc()))
    assert REGISTRY["dsv4_indexer_capture"].schema == D4.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)
