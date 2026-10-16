"""The CPU half of DeepSeek-V4's MoE capture (`tests/properties/dsv4_moe_capture_gpu.py`; the GPU run is `--out DIR` on an H200): the
twins the run compares every launch with are `registry.dsv4_moe`'s Programs, the rows reach the routers' and the clamp's edges, the
MUFU probe's expectations are the Definitions' models, and `--check` finds a changed output."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from verity.ml.circuits import library
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops import mufu as MU
from verity_vllm.program.registry import dsv4_moe as D
from verity_vllm.program.registry import prims as P

from tests.properties import dsv4_moe_capture_gpu as S

U32 = np.uint32


@pytest.fixture(scope="module")
def rows():
    return S.all_rows(2026, 8192, 2048)


def _words(rng) -> np.ndarray:
    """f32 words around every boundary the element functions branch on, specials and random words."""
    edges = (np.array([S.HALVING, S.Y_TWO, S.Y_ONE, 0x41A00000, 0x42AE0000, 0x3F800000, 0x00800000], np.int64)[:, None]
             + np.arange(-4, 5)).reshape(-1).astype(U32)
    return np.concatenate([S.EXTREMES, S.NONFINITE, edges, edges | U32(S.SIGN), rng.integers(0, 1 << 32, 1500, dtype=np.uint64).astype(U32),
                           S._w(np.concatenate([rng.uniform(-120, 120, 1500), rng.standard_normal(300) * 1e-6]))]).astype(U32)


def _one(d, *xs: int) -> int:
    r = evaluate(d, list(xs))
    return int(r[0]) if isinstance(r, (list, tuple)) else int(r)


def test_router_input_intactness_compares_f32_words():
    """NaNs with unchanged payloads are intact; one changed payload bit is not."""
    import torch

    words = np.array([0x7FC00000, 0xFFFFFFFF, 0x7F800001, 0x3F800000], U32)
    before = torch.from_numpy(words.view(np.int32)).view(torch.float32)
    same = before.clone()
    assert not torch.equal(same, before) and S._equal_f32_words(same, before)
    changed = same.view(torch.int32).clone()
    changed[0] ^= 1
    assert not S._equal_f32_words(changed.view(torch.float32), before)


def test_lg2_words_are_the_measured_unit_and_the_library_table():
    w = S.lg2_words(S.lg2_inputs())
    raw = w.astype("<u4").tobytes()
    lib = library.table("lg2")
    assert hashlib.sha256(raw).hexdigest() == D.LG2_MEASURED["sha256"] == lib.sha256
    assert hashlib.sha512(raw).hexdigest() == lib.sha512 and len(w) == lib.words == D.LG2_MEASURED["words"]
    m = np.random.default_rng(1).integers(0, 1 << 23, 400)
    assert w[m].tolist() == [D.mufu_lg2_mantissa_bits(int(v)) for v in m]


@pytest.mark.parametrize("name", ["ex2", "hash_weight", "score", "lg2"])
def test_the_element_twins_are_the_word_definitions(name):
    fn, d = {"ex2": (S.MC.ex2_approx, bind(D.Ex2Approx)), "hash_weight": (S.hash_weight_words, bind(D.Dsv4HashWeight)),
             "score": (S.score_words, bind(D.Dsv4SqrtSoftplus)), "lg2": (S.lg2_words, D.MufuLg2Mantissa)}[name]
    words = _words(np.random.default_rng(3))
    assert fn(words).tolist() == [_one(d, int(x)) for x in words]


def test_the_twins_are_the_programs_at_small_widths(rows):
    """The twins restate the Programs at any width; at 16 experts and 8 or 9 clamp columns the Programs evaluate fast, on rows cut
    from every kind."""
    R = rows["router"]
    idx = S.sample("router", R, 120, 5)
    x, b = R["x"][idx, :16], S.row_bias(R)[idx, :16]
    w, ids = S.router_twin(x, b)
    d = bind(D.MoeRouterSqrtSoftplus, E=16, SCALE=S.SCALE)
    assert [list(evaluate(d, x[i].tolist() + b[i].tolist())) for i in range(len(idx))] == np.concatenate([w, ids], 1).tolist()
    H = S.hash_rows(9, 160, 16)
    hid = S.hash_ids(H)
    d = bind(D.MoeRouterHash, E=16, SCALE=S.SCALE)
    assert [list(evaluate(d, H["x"][i].tolist() + hid[i].tolist())) for i in range(160)] == S.hash_twin(H["x"], hid).tolist()
    C = rows["clamp384"]
    ci = S.sample("clamp384", C, 120, 5)
    gu = np.concatenate([C["gu"][ci, :8], C["gu"][ci, 384:392]], 1)
    d = bind(D.SiluMulClamp, I=8, LIMIT=S.LIMIT)
    assert [list(evaluate(d, r.tolist())) for r in gu] == S.clamp_twin(gu).tolist()
    d = bind(D.SiluMulClamp, I=9, LIMIT=S.LIMIT)
    C9 = rows["clamp9"]["gu"][:60]
    assert [list(evaluate(d, r.tolist())) for r in C9] == S.clamp_twin(C9).tolist()


@pytest.mark.slow
def test_the_twins_are_the_served_programs(rows):
    """At the served statics (E = 384, I = 384 and 9): one row of each row kind under the cancelling and the non-finite bias, and
    one of each hash and clamp kind."""
    for path in S.PATHS:
        P_ = rows[path]
        if path == "router":
            idx = [g * S.GROUP + k for g in (4, 6) for k in range(len(S.ROW_KINDS))]
        else:
            kind = S._row_kind(path, P_)
            idx = [int(np.flatnonzero(kind == k)[0]) for k in np.unique(kind)]
        want = S.expect(path, {k: v for k, v in P_.items()})
        got = [S.program_row((path, S.program_inputs(path, P_, i))) for i in idx]
        assert got == [S.program_outputs(path, want, i) for i in idx], path


def _router_trace(R):
    """Each router row's scores, currents in the order the kernel takes them (rows, E) and its unbiased top six."""
    x, b = R["x"], S.row_bias(R)
    s = S.score_words(x)
    cur = S.MC.add(s, b)
    cur = S.MC.f32(np.where(S.MC.is_nan(cur), U32(D.NAN_SCORE), cur))
    ranked = -np.sort(-cur, axis=1, kind="stable")
    unbiased = np.argsort(-S.MC.f32(s), axis=1, kind="stable")[:, :6]
    return s, cur, ranked, unbiased


@pytest.mark.slow
def test_the_router_rows_reach_the_routers_edges(rows):
    R = rows["router"]
    again = S.router_rows(2026, 8192)
    assert R["x"].shape == (8192, S.E) and R["b"].shape == (128, S.E) and S.digest(R["x"], R["b"]) == S.digest(again["x"], again["b"])
    kinds = {(int(R["bias_kind"][i // S.GROUP]), int(R["row_kind"][i])) for i in range(len(R["x"]))}
    assert kinds == {(a, c) for a in range(len(S.BIAS_KINDS)) for c in range(len(S.ROW_KINDS))}
    s, cur, ranked, unbiased = _router_trace(R)
    w, ids = S.router_twin(R["x"], S.row_bias(R))
    above = (cur > -np.inf).sum(axis=1)
    assert {0, 1, 5} <= set(above.tolist()) and np.count_nonzero(above < 6) > 300            # fewer than six currents above -inf
    assert np.count_nonzero((ranked[:, 5] == ranked[:, 6]) & (ranked[:, 5] > -np.inf)) > 300   # a tie decides the last slot
    assert np.count_nonzero((ranked[:, :6] == 0).any(axis=1) & (above >= 6)) > 60             # score + bias == +0 in a slot
    assert np.count_nonzero(ranked[:, :6] == np.float32(-1e30)) > 100                         # a NaN current takes a slot
    assert np.count_nonzero((np.sort(unbiased, 1) != np.sort(ids, 1)).any(axis=1)) > 1000     # the bias decides the selection
    sel = np.take_along_axis(s, ids.astype(np.int64), 1)
    assert np.count_nonzero((sel == 0).all(axis=1)) > 20                                     # a zero sum: the divisor 1
    assert np.count_nonzero((sel == S.PINF).any(axis=1)) > 100                               # an infinite sum: +0 and NaN weights
    assert np.count_nonzero(S.MC.is_nan(sel).any(axis=1)) > 20 and np.count_nonzero(w == S.QNAN) > 100
    for edge in (0x41A00000, S.HALVING | S.SIGN):                                             # the select x > 20; the ex2 halving
        for d in (-1, 0, 1):
            assert np.count_nonzero(R["x"] == U32(edge + d))
    t = S.MC.f32(S.MC.mul(R["x"], U32(D.LOG2E)))
    assert np.count_nonzero((t < -126) & (t > -150)) > 10_000 and np.count_nonzero((t >= -126) & (t < -120)) > 10_000
    a = R["x"] & U32(0x7FFFFFFF)
    assert np.count_nonzero((a > 0) & (a < 0x00800000)) > 1000 and np.count_nonzero(a > S.PINF) > 10_000


def test_the_hash_rows_reach_the_weights_edges(rows):
    H = rows["hash"]
    again = S.hash_rows(2027, 8192)
    assert S.digest(*H.values()) == S.digest(*again.values())
    ids = S.hash_ids(H)
    assert ids.min() >= 0 and ids.max() < S.E and np.array_equal(np.sort(H["tokens"]), np.arange(len(H["tokens"])))
    assert np.count_nonzero([len(set(r)) < 6 for r in ids.tolist()]) > 1000                   # repeated table ids
    x = np.take_along_axis(H["x"], ids.astype(np.int64), 1)
    a = x & U32(0x7FFFFFFF)
    w = S.hash_weight_words(x)
    y = S.MC.add(S.MC.ex2_approx(S.MC.mul(a, U32(D.NEG_LOG2E))), U32(S.ONE))
    assert np.count_nonzero(y == U32(S.TWO)) > 500 and np.count_nonzero(y == U32(S.ONE)) > 2000       # lg2(2) = 1; lg2(1) = +0
    assert np.count_nonzero((y > U32(S.ONE)) & (y < U32(S.ONE + 64))) > 200                           # lg2 of 1 + a few ulps
    for d in (-1, 0, 1):
        assert np.count_nonzero(a == U32(S.HALVING + d))
    assert np.count_nonzero(S.MC.is_nan(x)) > 500 and np.count_nonzero(x == U32(S.PINF)) > 500
    assert np.count_nonzero((w == 0).all(axis=1)) > 900                                       # a zero sum: the divisor 1
    out = S.hash_twin(H["x"], ids)
    assert np.count_nonzero((out == S.QNAN).any(axis=1)) > 900                                 # an infinite sum


def test_the_clamp_rows_reach_the_limits(rows):
    for path, d in zip(S.PATHS[2:], S.WIDTHS):
        gu = rows[path]["gu"]
        assert gu.shape[1] == 2 * d
        g, u = gu[:, :d], gu[:, d:]
        f = S.MC.f32(S.DR.bf16_to_f32_words(gu))
        nan = S.MC.is_nan(S.DR.bf16_to_f32_words(gu))
        lim = U32(S.LIMIT_BF16)
        for half, sl in (("gate", slice(0, d)), ("up", slice(d, 2 * d))):
            h, hf, hn = gu[:, sl], f[:, sl], nan[:, sl]
            assert np.count_nonzero(hf > S.LIMIT) > 100 and np.count_nonzero(hn) > 100, (path, half)
            assert np.count_nonzero(h == lim) and np.count_nonzero(h == lim + 1) and np.count_nonzero(h == lim - 1), (path, half)
            assert np.count_nonzero(h == 0x8000) and np.count_nonzero(h == 0xFF80), (path, half)
        assert np.count_nonzero(f[:, d:] < -S.LIMIT) > 100 and np.count_nonzero(u == (lim | 0x8000)), path
        assert np.count_nonzero(f[:, :d] < -88.0) > 100, path                                  # expf overflows: silu -0
    s = rows["clamp384"]
    sweep = s["gu"][s["kind"] == S.CLAMP_KINDS.index("sweep")]
    assert len(np.unique(sweep[:, :384])) == 1 << 16 and len(np.unique(sweep[:, 384:])) == 1 << 16


def test_the_clamp_inputs_are_the_definitions_clamps():
    """`clamp_inputs` is `SiluMulClampBf16_v1`'s ``bf16(fminf(g, LIMIT))`` and ``bf16(fmaxf(fminf(u, LIMIT), -LIMIT))`` on every
    word, as each half of an I = 1 row."""
    every = np.arange(1 << 16, dtype=np.uint16)
    gc = S.clamp_inputs(np.stack([every, every], 1))
    f, lim = S.MC.f32(S.DR.bf16_to_f32_words(every)), np.float32(S.LIMIT)
    g, u = np.fmin(f, lim), np.fmax(np.fmin(f, lim), -lim)
    assert np.array_equal(gc[:, 0], (g.view(U32) >> 16).astype(np.uint16))
    assert np.array_equal(gc[:, 1], (u.view(U32) >> 16).astype(np.uint16))


@pytest.mark.slow
def test_the_mufu_probes_expectations_are_the_models():
    want = S.mufu_expect()
    assert len(S.SQRT_IN) == 1 << 24 and S.SQRT_IN[0] == S.ONE and S.SQRT_IN[1 << 23] == S.TWO
    assert np.array_equal(MU.mufu_sqrt_words(S.SQRT_IN, MU.table("sqrt")), want["sqrt"])
    assert [_one(P.MufuSqrtFtz, int(v)) for v in want["sqrt_sweep_in"][:300]] == want["sqrt_sweep"][:300].tolist()
    x = S.ex2_inputs()
    assert x[0] == S.EX2_LO and U32(D.MINUS_126) in x and U32(D.MINUS_126 + 1) in x and U32(D.MINUS_126 - 1) in x
    pick = np.concatenate([np.flatnonzero(np.abs(x.astype(np.int64) - D.MINUS_126) < 3), np.arange(0, len(x), 9973)])
    ex2 = bind(D.Ex2Approx)
    assert [_one(ex2, int(x[i])) for i in pick] == want["ex2"][pick].tolist()
    dv = S.div_divisors()
    assert {S.ONE, S.PINF, 0x39800000, 0x007FFFFF, 0x00800000} <= set(dv.tolist()) and (dv > 0).all() and (dv <= U32(S.PINF)).all()
    pick = np.concatenate([np.arange(38), np.arange(38, len(dv), 4099)])
    two5 = P.bits_of(S.SCALE)
    assert [P.F32Mul.evaluate(P.DivFullScaleA.evaluate(two5, int(v)), P.DivFullRcp.evaluate(int(v))) for v in dv[pick]] == want["div"][pick].tolist()
    assert want["lg2_specials"][0] == S.ONE and (want["lg2_specials"][1:] == S.QNAN).all()


def test_the_rows_file_round_trips(rows, tmp_path):
    got = {p: S.expect(p, P_) for p, P_ in rows.items()}
    np.savez_compressed(tmp_path / S.ROWS_FILE, **S._flat(rows, got))
    r2, g2 = S._unflat(np.load(tmp_path / S.ROWS_FILE))
    assert set(r2) == set(g2) == set(S.PATHS)
    for p in S.PATHS:
        assert set(r2[p]) == set(rows[p]) and all(np.array_equal(r2[p][k], rows[p][k]) for k in rows[p])
        assert set(g2[p]) == set(got[p]) and all(np.array_equal(g2[p][k], got[p][k]) for k in got[p])


def test_check_finds_a_changed_output(tmp_path):
    """`--check` on a capture written from the twins: the Programs agree, except on the one output changed."""
    rows = S.all_rows(7, 64, 8)
    got = {p: S.expect(p, P_) for p, P_ in rows.items()}
    got["hash"]["w"][1, 0] ^= 1
    np.savez_compressed(tmp_path / S.ROWS_FILE, **S._flat(rows, got))
    rec = S.check(str(tmp_path), workers=1, limit=2)
    assert rec["hash"] == {"rows": 2, "mismatch": 1, "first": [1]} and rec["ok"] is False
    assert all(rec[p] == {"rows": 2, "mismatch": 0, "first": []} for p in S.PATHS if p != "hash")
    assert json.loads((tmp_path / S.CHECK_FILE).read_text())["hash"]["mismatch"] == 1
