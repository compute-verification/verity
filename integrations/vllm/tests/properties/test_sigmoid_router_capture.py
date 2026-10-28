"""The CPU half of the sigmoid-router capture (`tests/properties/sigmoid_router_capture_gpu.py`; the GPU run is `--out DIR` on a
node): the twin the run compares every launch with is `MoeRouterSigmoidBias_v1`, the rows reach the router's edges, the MUFU sweep's
expectation is the twin's, and `--check` finds a changed output."""
from __future__ import annotations

import json

import numpy as np
import pytest
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import moe
from verity.ml.definitions.numerics import moe_twins

from tests.properties import sigmoid_router_capture_gpu as S

U32 = np.uint32


@pytest.fixture(scope="module")
def R():
    return S.rows(2026, 10_240)


def _scores(R):
    """Each row's selection scores, sigmoids and finite experts (rows, E), as `twin` computes them."""
    x, b = R["x"], S.row_bias(R)
    fin = S._finite(x) & S._finite(b)
    sig = S.sigmoid_words(x)
    biased = S._add(sig, b)
    return np.where(fin, np.where((biased & U32(0x7FFFFFFF)) == 0, U32(0), biased), U32(S.NEG_INF)).astype(U32), sig, fin


def _top8(scores: np.ndarray) -> np.ndarray:
    return np.argsort(~S._keys(scores, np.arange(scores.shape[1])), axis=1, kind="stable")[:, :8]


def test_sigmoid_words_is_the_sass_model():
    rng = np.random.default_rng(11)
    edges = (np.array([S.MUFU_LO, S.MUFU_HI, 0x3F19999A, 0x41102CB4], np.int64)[:, None] + np.arange(-3, 4)).reshape(-1).astype(U32)
    words = np.concatenate([S.EXTREMES, S.NONFINITE, edges, edges | U32(S.SIGN),
                            rng.integers(0, 1 << 32, 3000, dtype=np.uint64).astype(U32),
                            S._w(np.concatenate([rng.uniform(-20, 20, 3000), rng.standard_normal(500) * 1e-3]))]).astype(U32)
    assert S.sigmoid_words(words).tolist() == [moe_twins.nv_sigmoid_bits(int(w)) for w in words]


@pytest.mark.slow
@pytest.mark.parametrize("launch", S.LAUNCHES)
def test_the_twin_is_the_program_at_16_experts(R, launch):
    """`twin` restates the Program at any width with 8 slots; at 16 experts the Program evaluates fast, on rows cut from every kind."""
    d = bind(moe.MoeRouterSigmoidBias, E=16, TOPK=8, SCALE=2.5, KERNEL=launch)
    idx = S.sample(R, 160, 5)
    x, b = R["x"][idx, :16], S.row_bias(R)[idx, :16]
    w, ids = S.twin(x, b, launch)
    for i in range(len(idx)):
        assert list(evaluate(d, x[i].tolist() + b[i].tolist())) == w[i].tolist() + ids[i].tolist(), (launch, int(idx[i]))


@pytest.mark.slow
@pytest.mark.parametrize("launch", S.LAUNCHES)
def test_the_twin_is_the_served_program(R, launch):
    """At the served width: one row of each row kind under the cancelling and the non-finite bias."""
    idx = [g * S.GROUP + k for g in (4, 6) for k in range(len(S.ROW_KINDS))]
    w, ids = S.twin(R["x"][idx], S.row_bias(R)[idx], launch)
    got = [S.program_row((launch, R["x"][i], S.row_bias(R)[i])) for i in idx]
    assert got == [w[j].tolist() + ids[j].tolist() for j in range(len(idx))]


def test_the_rows_reach_the_routers_edges(R):
    again = S.rows(2026, 10_240)
    assert R["x"].shape == (10_240, S.E) and R["b"].shape == (160, S.E)
    assert S.digest(R["x"], R["b"]) == S.digest(again["x"], again["b"])
    kinds = {(int(R["bias_kind"][i // S.GROUP]), int(R["row_kind"][i])) for i in range(len(R["x"]))}
    assert kinds == {(a, c) for a in range(len(S.BIAS_KINDS)) for c in range(len(S.ROW_KINDS))}
    sel, sig, fin = _scores(R)
    picked = _top8(sel)
    ranked = np.take_along_axis(sel, np.argsort(~S._keys(sel, np.arange(S.E)), axis=1, kind="stable"), 1)
    nfin = fin.sum(axis=1)
    assert {0, 1, 7, 8, 9} <= set(nfin.tolist())                                         # fewer finite experts than slots, or none
    assert np.count_nonzero((ranked[:, :8] == 0).any(axis=1) & (nfin >= 8)) > 60          # score + bias == 0 in a slot
    assert np.count_nonzero((ranked[:, 7] == ranked[:, 8]) & (ranked[:, 7] != S.NEG_INF)) > 500   # a tie decides the last slot
    unbiased = _top8(np.where(fin, sig, U32(S.NEG_INF)).astype(U32))
    assert np.count_nonzero((np.sort(unbiased, 1) != np.sort(picked, 1)).any(axis=1)) > 1000     # the bias decides the selection
    a = R["x"] & U32(0x7FFFFFFF)
    for lo, hi in ((1, 0x00800000), (0x00800000, S.MUFU_LO), (S.MUFU_LO, S.MUFU_HI), (S.MUFU_HI, S.PINF)):
        assert np.count_nonzero((a >= lo) & (a < hi)) > 1000                               # subnormal, polynomial, MUFU, saturated
    for edge in (S.MUFU_LO, S.MUFU_HI):
        for d in (-1, 0, 1):
            assert np.count_nonzero(R["x"] == U32(edge + d)) and np.count_nonzero(R["x"] == U32((edge + d) | S.SIGN))


def test_the_sweep_is_every_mufu_logit_and_its_expectation_is_the_twins():
    words = S.sweep_words()
    span = S.MUFU_HI - S.MUFU_LO
    assert len(words) % 8 == 0 and 0 <= len(words) - 2 * span < 8
    assert words[0] == S.MUFU_LO and words[span - 1] == S.MUFU_HI - 1 and words[span] == S.MUFU_LO | S.SIGN
    assert words[2 * span - 1] == (S.MUFU_HI - 1) | S.SIGN
    for row0 in (0, 31, 32 * 1000 + 5):
        vals = words[row0 * 8:(row0 + 256) * 8].reshape(-1, 8)
        x, pos = S.sweep_chunk(vals, row0)
        assert np.count_nonzero(S._finite(x)) == vals.size and (np.take_along_axis(x, pos, 1) == vals).all()
        ew, eids = S.sweep_expect(vals, pos)
        for launch in S.LAUNCHES:
            w, ids = S.twin(x, np.zeros(S.E, U32), launch, renormalize=False, scale=1.0)
            assert np.array_equal(w, ew) and np.array_equal(ids, eids)


def test_the_run_is_at_vllms_defaults(monkeypatch):
    """The run clears every switch the process inherited, and its verdict needs vLLM to read the served ones back: a batch-invariant
    run, or one whose grouped top-k is not the fused kernel, is not a GO however its rows compare."""
    from types import SimpleNamespace

    from verity_vllm.engine.env import SERVED_DEFAULTS, served_defaults

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")
    assert served_defaults() == {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_DEEP_GEMM": "0"}
    assert not [k for k in SERVED_DEFAULTS if k in __import__("os").environ]
    assert S.served(SimpleNamespace(VLLM_BATCH_INVARIANT=False, VLLM_USE_FUSED_MOE_GROUPED_TOPK=True))["ok"]
    doc = {"gate": {"ok": True}, "served": S.served(SimpleNamespace(VLLM_BATCH_INVARIANT=False, VLLM_USE_FUSED_MOE_GROUPED_TOPK=True)),
           "host_subnormals": True, "mufu": {"ok": True}, "kernels": {lnch: {"ok": True} for lnch in S.LAUNCHES},
           "capture": {"deterministic": True, "tiles_equal": True, "logits_intact": True},
           "rows": {lnch: {"mismatch": 0} for lnch in S.LAUNCHES}, "program": {"ok": True}, "sweep": {"ok": True}}
    assert S._verdict(doc)
    for bi, fused in ((True, True), (False, False)):
        doc["served"] = S.served(SimpleNamespace(VLLM_BATCH_INVARIANT=bi, VLLM_USE_FUSED_MOE_GROUPED_TOPK=fused))
        assert not S._verdict(doc)


def test_check_finds_a_changed_output(tmp_path):
    """`--check` on a capture written from the twin: the Program agrees, except on the one output changed."""
    R = S.rows(7, S.GROUP)
    got = {}
    for launch in S.LAUNCHES:
        got[f"{launch}_w"], got[f"{launch}_ids"] = S.twin(R["x"], S.row_bias(R), launch)
    got["block_w"][1, 0] ^= 1
    np.savez_compressed(tmp_path / S.ROWS_FILE, **R, **got)
    rec = S.check(str(tmp_path), workers=1, limit=2)
    assert rec["block"] == {"mismatch": 1, "first": [1]} and rec["warp"] == {"mismatch": 0, "first": []} and rec["ok"] is False
    assert json.loads((tmp_path / S.CHECK_FILE).read_text())["rows"] == 2
