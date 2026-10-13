"""The GLM-5.3 DSA capture without a GPU: the RoPE probe's operands separate every pair of candidate contractions and name each
pair of forms, the index-K probe's bytes are `DsaIndexerKRow_v1`'s form and tell the others apart, the logits twin is the
reference, each comparison job takes the kernel's words and finds a flipped one, the index conversion and top-k restatements, the
PTX scan, the bindings, the verdict and `main` end to end (the GPU run is `tests/properties/glm53_dsa_capture_gpu.py --out ...` on
an H200)."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.properties import glm53_dsa_capture_gpu as X
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.defs import bind
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.registry import dsa
from verity_vllm.program.registry import ref_prims as RP
from verity_vllm.program.registry import sparse_mla as M
from verity_vllm.program.kernels.fp8_twins import np_f32_to_e4m3_sat
from verity_vllm.properties.record import REGISTRY, digest_of

PAIRS = [(a, b) for fs in (X.L_FORMS, X.H_FORMS) for i, a in enumerate(fs) for b in fs[i + 1:]]


def _bf16(rng: np.random.Generator, *shape: int, scale: float = 1.0) -> np.ndarray:
    return ((rng.standard_normal(shape).astype(np.float32) * np.float32(scale)).view(np.uint32) >> 16).astype(np.uint16)


def _e4m3(rng: np.random.Generator, *shape: int) -> np.ndarray:
    b = rng.integers(0, 256, shape).astype(np.uint8)
    return np.where((b & 0x7F) == 0x7F, b ^ 1, b).astype(np.uint8)


def _total(results: list[dict]) -> dict:
    assert not [r for r in results if "error" in r], results
    return {k: sum(r[k] for r in results) for k in X.TOTALS}


# ------------------------------------------------------------------------------------------------------------------- RoPE
def _quads() -> np.ndarray:
    return np.array(X.rope_probe_quads(), dtype=np.uint16)


def test_the_rope_probe_separates_every_pair_of_forms() -> None:
    q = _quads()
    f = X.rope_forms(q[:, 0], q[:, 1], q[:, 2], q[:, 3])
    exact = [X.rope_forms_exact(*map(int, r)) for r in q]
    for k in X.L_FORMS + X.H_FORMS:
        assert np.array_equal(f[k], np.array([e[k] for e in exact], dtype=np.uint16)), k
        assert (f[k] != f[k + "_ftz"]).any(), k
    assert all(int((f[a] != f[b]).sum()) >= 16 for a, b in PAIRS)


@pytest.mark.parametrize("L", X.L_FORMS + tuple(f + "_ftz" for f in X.L_FORMS))
def test_each_pair_of_forms_is_named_by_the_probe_alone(L: str) -> None:
    q = _quads()
    f = X.rope_forms(q[:, 0], q[:, 1], q[:, 2], q[:, 3])
    for Hf in X.H_FORMS:
        m = X.rope_matches(q[:, 0], q[:, 1], q[:, 2], q[:, 3], f[L], f[Hf])
        assert X.rope_settled(m) and Hf in m["H"] and L in m["L"]
        if not L.endswith("_ftz"):
            assert (m["L"], m["H"]) == ([L], [Hf])
    assert X.rope_name({"L": ["L_a"], "H": ["H_a"]}) == "RoPEHeadGptJ_v1"
    assert X.rope_name({"L": ["L_a"], "H": ["H_b"]}).startswith("RoPEHeadGptJ_v2")
    assert X.rope_name({"L": ["L_n"], "H": ["H_a"]}).startswith("no Definition")
    assert not X.rope_settled({"L": ["L_a", "L_b"], "H": ["H_a"]}) and not X.rope_settled({"L": [], "H": ["H_a"]})


def test_the_registered_rope_is_the_l_a_h_a_pair() -> None:
    """`RoPEHeadGptJ_v1` (the map's `fused-rope-triton`) on probe rows is (L_a, H_a): the name `rope_name` gives those words."""
    x, cs = X.rope_probe_inputs(X.PROBE_TOKENS)
    fn = bind(RP.RoPEHeadGptJ, D=X.ROPE)
    f = X.rope_forms(x[:, 0::2], x[:, 1::2], cs[:, :32], cs[:, 32:])
    for t in range(x.shape[0]):
        y = np.array(evaluate(fn, x[t].tolist(), cs[t].tolist()), dtype=np.uint16)
        assert np.array_equal(y[0::2], f["L_a"][t]) and np.array_equal(y[1::2], f["H_a"][t])


def _rope_state(rng: np.random.Generator, L: str, Hf: str, flip: bool = False):
    """The served tokens' and the probes' k_pe and q_pe as `phase_rope` reads them, the kernel's words the (L, H) pair."""
    n = 5

    def out(x, cs):
        f = X.rope_forms(x[:, 0::2], x[:, 1::2], cs[:, :32], cs[:, 32:])
        y = np.empty_like(x)
        y[:, 0::2], y[:, 1::2] = f[L], f[Hf]
        return y

    def t(a):
        return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(torch.bfloat16)

    cs = _bf16(rng, n, X.ROPE, scale=0.5)
    k_pe, q_pe = _bf16(rng, n, X.ROPE), _bf16(rng, n * X.NH, X.ROPE)
    px, pcs = X.rope_probe_inputs(X.PROBE_TOKENS)
    kprobe = out(px, pcs)
    if flip:
        kprobe[1, 6] ^= 1
    qx, qcs = np.repeat(px, X.NH, axis=0), np.repeat(pcs, X.NH, axis=0)
    o = {"pos": torch.arange(n), "cs": t(cs), "k_pe": t(k_pe), "q_pe": t(q_pe.reshape(n, X.NH, X.ROPE))}
    kst = {"k_pe_out": t(out(k_pe, cs)), "k_pe_probe": {"x": px, "cs": pcs, "out": kprobe}}
    qst = {"q_pe_out": t(out(q_pe, np.repeat(cs, X.NH, axis=0)).reshape(n, X.NH, X.ROPE)),
           "q_pe_probe": {"x": px, "cs": pcs, "out": out(qx, qcs).reshape(-1, X.NH, X.ROPE)}}
    return o, kst, qst


@pytest.mark.parametrize("L,Hf,name", [("L_a", "H_a", "RoPEHeadGptJ_v1"), ("L_a", "H_b", "RoPEHeadGptJ_v2"),
                                       ("L_b", "H_n", "no Definition")])
def test_the_rope_phase_names_the_contraction(L: str, Hf: str, name: str) -> None:
    got = X.phase_rope(*_rope_state(np.random.default_rng(4), L, Hf))
    assert got["ok"]
    for k in ("k_pe", "q_pe"):
        assert (got[k]["L"], got[k]["H"]) == ([L], [Hf]) and got[k]["definition"].startswith(name)
        assert got[k]["reference_rows"] > 2 and got[k]["twin_mismatch"] == 0 and got[k]["probe_rows"] > 0
    assert (got["k_pe"]["mismatch_v1"] == 0) == (name == "RoPEHeadGptJ_v1")


def test_a_word_no_form_gives_leaves_the_rope_unsettled() -> None:
    got = X.phase_rope(*_rope_state(np.random.default_rng(4), "L_a", "H_a", flip=True))
    assert not got["ok"] and not got["k_pe"]["settled"] and got["q_pe"]["settled"]


# --------------------------------------------------------------------------------------------------------------- index K
def _index_k_probe():
    fn = bind(dsa.DsaIndexerKRow, D=X.D, ROPE=X.ROPE, EPS=X.K_EPS)
    bs, css, ys = [], [], []
    for j in range(X.PROBE_LAUNCHES):
        b, cs = X.index_k_probe(j)
        bs.append(b)
        css.append(cs)
        ys.append(evaluate(fn, [0] * X.D, [0] * X.D, b, cs))
    y = np.array(ys, dtype=np.int64)
    return np.array(bs, dtype=np.uint32), np.array(css, dtype=np.uint16), y[:, :X.D].astype(np.uint8), y[:, X.D].astype(np.uint32)


def test_the_index_k_probe_is_the_definitions_form_and_tells_the_others_apart() -> None:
    b, cs, q, s = _index_k_probe()
    assert (s == 0x3F800000).all()
    got = X.index_k_forms(b, cs, q)
    assert (got["L"], got["H"]) == (["L_A"], ["H_A"])
    assert all(got["separating"][k] >= 32 for k in ("L_B", "L_N", "H_B", "H_N"))
    x = b[:, :X.ROPE]
    cos, sin = DR.bf16_to_f32_words(cs[:, :32]), DR.bf16_to_f32_words(cs[:, 32:])
    other = q.copy()
    other[:, 0:X.ROPE:2] = np_f32_to_e4m3_sat(DR.as_f32(DR.add_rn(DR.mul_rn(x[:, 0::2], cos),
                                                                   DR.mul_rn(x[:, 1::2] ^ X.NEG, sin)))).reshape(16, 32)
    other[:, 1:X.ROPE:2] = np_f32_to_e4m3_sat(DR.as_f32(DR.fma_rn(x[:, 0::2], sin, DR.mul_rn(x[:, 1::2], cos)))).reshape(16, 32)
    got = X.index_k_forms(b, cs, other)
    assert (got["L"], got["H"]) == (["L_N"], ["H_B"])


def _kslots_job(tmp_path, *, flip: bool = False) -> dict:
    """The index-K probe's slots and three served-like rows (one zero, one of bf16 subnormals), their bytes the reference's."""
    rng = np.random.default_rng(6)
    b, cs, q, s = _index_k_probe()
    n = 3
    k = _bf16(rng, n, X.D, scale=2.0)
    k[1] = 0
    k[2] = _bf16(rng, X.D, scale=2.0 ** -130)
    w = (1 + 0.1 * rng.standard_normal((n, X.D))).astype(np.float32).view(np.uint32)
    bb = (0.1 * rng.standard_normal((n, X.D))).astype(np.float32).view(np.uint32)
    css = _bf16(rng, n, X.ROPE, scale=0.5)
    fn = bind(dsa.DsaIndexerKRow, D=X.D, ROPE=X.ROPE, EPS=X.K_EPS)
    y = np.array([evaluate(fn, k[r].tolist(), w[r].tolist(), bb[r].tolist(), css[r].tolist()) for r in range(n)], dtype=np.int64)
    K = np.concatenate([np.zeros((16, X.D), np.uint16), k])
    Q = np.concatenate([q, y[:, :X.D].astype(np.uint8)])
    if flip:
        Q[17, 40] ^= 1
    paths = X._save(tmp_path / "k", k=K, w=np.concatenate([np.zeros((16, X.D), np.uint32), w]), b=np.concatenate([b, bb]),
                    cs=np.concatenate([cs, css]), q=Q, s=np.concatenate([s, y[:, X.D].astype(np.uint32)]), tokens=np.arange(19) + 100)
    return {"phase": "k", "case": "served", "kind": "kslots", "paths": paths}


def test_the_kslot_jobs_take_the_kernel_bytes(tmp_path) -> None:
    job = _kslots_job(tmp_path)
    t = _total([X.compare({**job, "rows": [a, b]}) for a, b in X._chunks(19, 8)])
    assert t == {"words": 19 * (X.D + 1), "mismatch": 0, "unresolved": 0, "reference": 19, "twin_mismatch": 0}
    job = _kslots_job(tmp_path, flip=True)
    r = X.compare({**job, "rows": [0, 19]})
    assert r["mismatch"] == 1 and r["first_mismatch"]["token"] == 117 and r["first_mismatch"]["word"] == 40


# --------------------------------------------------------------------------------------------------------------- index Q
def test_the_qtok_jobs_take_the_kernel_words(tmp_path) -> None:
    rng = np.random.default_rng(7)
    n = 2
    q = _bf16(rng, n, X.H, X.D, scale=3.0)
    q[1, 0] = 0
    q[1, 3] = _bf16(rng, X.D, scale=2.0 ** -130)
    cs, w = _bf16(rng, n, X.ROPE, scale=0.5), _bf16(rng, n, X.H)
    fn = bind(dsa.DsaIndexerQ, H=X.H, D=X.D, ROPE=X.ROPE)
    y = np.array([evaluate(fn, q[r].reshape(-1).tolist(), cs[r].tolist(), w[r].tolist()) for r in range(n)], dtype=np.int64)
    outq, outw = y[:, :X.H * X.D].astype(np.uint8).reshape(n, X.H, X.D), y[:, X.H * X.D:].astype(np.uint32)
    for flip in (False, True):
        if flip:
            outw[1, 5] ^= 1
        paths = X._save(tmp_path / f"q{flip}", q=q, cs=cs, w=w, outq=outq, outw=outw, tokens=np.array([3, 9]))
        r = X.compare({"phase": "q", "case": "served", "kind": "qtok", "paths": paths, "rows": [0, n]})
        assert (r["words"], r["reference"], r["mismatch"]) == (n * (X.H * X.D + X.H), n, int(flip))
    assert r["first_mismatch"] == {"token": 9, "word": X.H * X.D + 5, "want": int(y[1, X.H * X.D + 5]), "kernel": int(outw[1, 5])}


# ---------------------------------------------------------------------------------------------------------------- logits
def _logit_operands(rng: np.random.Generator, n: int):
    q, k = _e4m3(rng, n, X.H, X.D), _e4m3(rng, n, X.D)
    w = (rng.standard_normal((n, X.H)).astype(np.float32) * 0.1).view(np.uint32)
    ks = np.exp2(rng.integers(-8, 0, n)).astype(np.float32).view(np.uint32)
    q[0] = 0
    k[1] = 0
    return q, w, k, ks


def test_the_logits_twin_is_the_reference() -> None:
    q, w, k, ks = _logit_operands(np.random.default_rng(8), 12)
    want, nonfin = X.logit_rows(q, w, k, ks)
    fn = bind(dsa.DsaMqaLogitV3, H=X.H, D=X.D)
    assert not nonfin.any() and want[0] == want[1] == 0 and len(set(want.tolist())) > 8
    assert want.tolist() == [evaluate(fn, q[i].reshape(-1).tolist(), w[i].tolist(), k[i].tolist(), int(ks[i]))[0] for i in range(12)]
    k[3, 17] = 0x7F
    assert X.logit_rows(q, w, k, ks)[1].tolist() == [i == 3 for i in range(12)]


def _logits_job(tmp_path, *, flip: bool = False, nan: bool = False) -> list[dict]:
    """A 6-token prefill (every key i <= r) and a decode of 3 keys of the last two tokens' queries, as `phase_logits` saves them."""
    rng = np.random.default_rng(9)
    T = 6
    q, w, _, _ = _logit_operands(rng, T)
    k = _e4m3(rng, T, X.D)
    ks = np.exp2(rng.integers(-8, 0, T)).astype(np.float32).view(np.uint32)
    if nan:
        k[2, 0] = 0xFF
    r = np.concatenate([np.full(i + 1, i) for i in range(T)])
    c = np.concatenate([np.arange(i + 1) for i in range(T)])
    got = X.logit_rows(q[r], w[r], k[c], ks[c])[0]
    rd, cd = np.array([4, 4, 4, 5, 5, 5]), np.array([0, 1, 2, 0, 1, 2])
    gotd = X.logit_rows(q[rd], w[rd], k[cd], ks[cd])[0]
    if flip:
        got[7] ^= 1
    paths = X._save(tmp_path / "logits", q=q, w=w, k=k, ks=ks, r=r, c=c, got=got, rd=rd, cd=cd, gotd=gotd)
    jobs = [{"phase": "logits", "case": "prefill", "kind": "logits", "paths": paths, "keys": ["r", "c", "got"], "range": [a, b]}
            for a, b in X._chunks(r.size, 8)]
    return jobs + [{"phase": "logits", "case": "decode", "kind": "logits", "paths": paths, "keys": ["rd", "cd", "gotd"],
                    "range": [0, rd.size]}]


def test_the_logits_jobs_take_the_kernel_words(tmp_path) -> None:
    res = [X.compare(j) for j in _logits_job(tmp_path)]
    assert _total(res) == {"words": 21 + 6, "mismatch": 0, "unresolved": 0, "reference": 2 * len(res), "twin_mismatch": 0}
    res = [X.compare(j) for j in _logits_job(tmp_path, flip=True)]
    assert _total(res)["mismatch"] == 1 and res[0]["first_mismatch"]["row"] == 3 and res[0]["first_mismatch"]["key"] == 1
    res = [X.compare(j) for j in _logits_job(tmp_path, nan=True)]
    assert _total(res)["unresolved"] == 4 + 2 and _total(res)["mismatch"] == 0


def test_the_logits_rerun_compares_the_words_each_row_writes() -> None:
    """With clean_logits=False the words past row r's window [0, r] are whatever the buffer held: they differ between reruns
    without making the kernel nondeterministic, and a differing word inside a window does."""
    a = torch.randn(5, 7)
    b = a.clone()
    b[1, 2], b[0, 6] = float("nan"), 3.0
    assert X.logits_rerun(a, b) == (True, 2)
    b[3, 3] = torch.nextafter(a[3, 3], torch.tensor(float("inf")))
    assert X.logits_rerun(a, b) == (False, 2)


# ------------------------------------------------------------------------------------------------------------- attention
def test_the_attn_jobs_take_the_kernel_words(tmp_path) -> None:
    rng = np.random.default_rng(10)
    T = 3
    qn, qr = _bf16(rng, X.NH, X.DV), _bf16(rng, X.NH, X.DR)
    kv = _bf16(rng, T, X.DV + X.DR, scale=0.3)
    fn = bind(M.SparseMlaHead, T=T, **{k: v for k, v in M.SPARSE_MLA_GLM53.items() if k != "NH"})
    out = np.array([evaluate(fn, qn[h].tolist(), qr[h].tolist(), kv.reshape(-1).tolist()) for h in range(2)], dtype=np.uint16)
    out = np.concatenate([out, np.zeros((X.NH - 2, X.DV), np.uint16)])
    out[1, 300] ^= 1
    paths = X._save(tmp_path / "attn", qn=qn, qr=qr, kv=kv, out=out)
    res = [X.compare({"phase": "attn", "case": "token7", "kind": "attn", "paths": paths, "T": T, "head": h}) for h in (0, 1)]
    assert [r["mismatch"] for r in res] == [0, 1] and res[1]["first_mismatch"]["word"] == 300
    assert _total(res)["words"] == 2 * X.DV and _total(res)["reference"] == 2


def test_the_index_conversion_compacts_valid_entries_in_order() -> None:
    bt = np.array([5, 2, 7, 0], dtype=np.int32)
    topk = np.array([[0, 63, 64, -1, 130, 255, 256, 300, -1, 1], [-1] * 10], dtype=np.int32)
    out, n = X.convert_rows(topk, bt, 64)
    assert out[0].tolist() == [320, 383, 128, 450, 63, 321, -1, -1, -1, -1] and out[1].tolist() == [-1] * 10
    assert n.tolist() == [6, 0]
    out, _ = X.convert_rows(topk, bt, 100)
    assert out[0, :3].tolist() == [500, 563, 200]


# ------------------------------------------------------------------------------------------------------------------ top-k
def test_the_short_rows_reference_is_the_closed_form() -> None:
    """`DsaTopKShortRow_v1{2048}` on the lengths `phase_topk` evaluates (a negative length the kernels clamp to 0) is `topk_short`."""
    fn = bind(dsa.DsaTopKShortRow, TOPK=X.TOPK)
    for n in (0, 1, 2, X.TOPK - 1, X.TOPK, -3):
        want = X.topk_short(np.array([max(n, 0)]))[0]
        assert list(evaluate(fn, n & 0xFFFFFFFF)) == (want.astype(np.int64) & 0xFFFFFFFF).tolist(), n


def test_the_topk_checks() -> None:
    vals, lens, names = X._topk_rows(11)
    assert names[-1] == "negative" and lens[-1] == -3 and np.isinf(vals[0]).all()
    out = X.topk_short(np.maximum(lens, 0))
    for i in np.flatnonzero(lens > X.TOPK):
        out[i] = np.sort(np.argsort(-vals[i, :lens[i]], kind="stable")[:X.TOPK])
    got = X._check_topk(vals, lens, out)
    assert (got["short_rows"], got["short_mismatch"], got["long_rows"], got["long_claimed"], got["long_claimed_valid"]) == (6, 0, 4, 4, 4)
    assert got["long_ascending"] == 4 and got["long_with_ties"] == 2 and X._topk_ok(got)
    coop = X._check_topk(vals, lens, out, kernel="cooperative", launch=len(lens), names=names)
    assert [r["row"] for r in coop["long"] if r["overflow"]] == ["ties_all_equal"] and coop["long_overflow_valid"] == 1
    bad = out.copy()
    bad[2, 1] = -1
    bad[lens.tolist().index(4096), 7] = bad[lens.tolist().index(4096), 8]
    got = X._check_topk(vals, lens, bad)
    assert got["short_mismatch"] == 1 and got["long_claimed_valid"] == 3 and not X._topk_ok(got)
    ones = lens.tolist().index(2100)
    sel = out[ones]
    worse = sel.copy()
    worse[np.flatnonzero(vals[ones, sel] == 1)[0]] = np.setdiff1d(np.flatnonzero(vals[ones, :2100] == 0), sel)[0]
    assert X.topk_long(vals[ones], 2100, sel)["valid"] and X.topk_long(vals[ones], 2100, worse)["valid"] is False


def _exact_top(vals: np.ndarray, lens: np.ndarray, topk: int) -> np.ndarray:
    """A top-``topk`` set of each row under `DsaTopKSelection_v1`'s order, ascending (short rows the shortcut)."""
    out = X.topk_short(np.maximum(lens, 0), topk)
    for i in np.flatnonzero(lens > topk):
        out[i] = np.sort(np.argsort(-dsa.order_keys(vals[i, :lens[i]]), kind="stable")[:topk])
    return out


def test_the_coarse_bins_follow_the_order() -> None:
    """Every coarse histogram's bins ascend with `DsaTopKSelection_v1`'s order (so a bin above the threshold holds only selected
    keys), +0 and -0 never share a bin (so no kernel's float compare sees the pair), and an f32 whose half is -inf shares the bin of
    hist2048's padding."""
    rng = np.random.default_rng(3)
    special = np.array([0.0, -0.0, np.inf, -np.inf, 65504, 65519.99, 65520, -65520, -65504, 1e-8, -1e-8, 6e-5, -6e-5, 1.0, 1.0009766,
                        3.4e38, -3.4e38, 1e-45, -1e-45], dtype=np.float32)
    v = np.concatenate([special, (rng.standard_normal(20000) * 10.0 ** rng.integers(-8, 6, 20000)).astype(np.float32),
                        (np.float32(1.0).view(np.uint32) + np.arange(5000, dtype=np.uint32)).view(np.float32)])
    assert v.dtype == np.float32
    v = v[np.argsort(dsa.order_keys(v), kind="stable")]
    for bits in (8, 10, 11, 12):
        b = X.half_bins(v, bits)
        assert (np.diff(b) >= 0).all(), bits
        assert X.half_bins(np.float32([0.0]), bits)[0] == X.half_bins(np.float32([-0.0]), bits)[0] + 1
    assert X.half_bins(np.float32([-np.inf, -65520.0, -3.4e38]), 11).tolist() == [31, 31, 31]
    assert X.half_bins(np.float32([-65504.0]), 11)[0] == 32
    assert X.threshold_bin(np.array([0, 1, 1, 2, 2, 2]), 4) == (1, 3, 2)
    assert X.cooperative_blocks(16385, 8)[-1] == (14340, 16385) and X.cooperative_blocks(16385, 8)[0] == (0, 2052)
    assert [X.cooperative_cluster(n) for n in (1, 8, 9, 33, 34, 64)] == [8, 8, 4, 4, 2, 2]


def test_the_long_rows_and_their_claim() -> None:
    """The long rows at each top-k: an exact top set passes on every row at every kernel's launches, and the claim
    (`topk_overflow`) leaves out exactly the rows whose threshold bin overflows that path's tie capacity, the NaN row and the -inf
    row; a +0 swapped for a -0 fails; the kernels' tie handling run past its capacity (the first ties by index kept) misses a crowd's
    top keys, so a crowd the claim leaves out does tell an overflow."""
    for topk in X.LONG_TOPKS:
        vals, lens, names = X._long_rows(topk, 13 + topk)
        R = len(lens)
        assert R > 33 and (lens > topk).all() and lens.max() <= X.LONG_WIDTH and X.LONG_WIDTH % 4 == 0
        assert np.isinf(vals[0, lens[0]:]).all()
        out = _exact_top(vals, lens, topk)
        at = {n: i for i, n in enumerate(names)}
        over = {}
        for kernel, launch in (("prefill", R), ("cooperative", 8), ("cooperative", 16), ("cooperative", R), ("persistent", 32),
                               ("persistent", R)):
            got = X._check_topk(vals, lens, out, kernel=kernel, launch=launch, names=names)
            assert X._topk_ok(got) and got["long_overflow_valid"] == got["long_overflow"] and got["long_claimed"] > 0, (kernel, launch)
            over[kernel, launch] = {r["row"]: r["overflow"] for r in got["long"] if r["overflow"]}
        outside = {"nan_len8193": "nan", "ninf_len4099": "half -inf"}
        assert over["prefill", R] == outside
        assert over["persistent", 32] == {**outside, "ties5000_len8192": "hist2048: 5000 ties > 3708",
                                          "ties5000_len16385": "hist256: 5000 ties > 4096"}
        middle = {n: (int(m[2]), int(m[3])) for n in names if (m := re.fullmatch(r"(ties|crowd)(\d+)_len(\d+)", n))}
        assert over["persistent", R] == {**outside, **{n: f"hist4096: {m} ties > {topk}" for n, (m, k) in middle.items()
                                                       if m > topk and k <= 32768},
                                         "crowd20000_len32769": "filtered: 20000 ties > 16384"}
        for launch in (8, 16, R):
            assert {n for n in over["cooperative", launch] if n in middle} == {n for n, (m, k) in middle.items()
                                                                               if m > max(topk, 1024 if k > 16384 else 0)}
            assert over["cooperative", launch].keys() - middle.keys() == outside.keys()
        assert over["cooperative", 8]["crowd3000_len32769"] == f"cluster: 3000 ties > {max(topk, 1024)}"
        assert re.fullmatch(r"cluster: 1[45]\d\d ties in one block > 1024", over["cooperative", R]["crowd3000_len32769"])
        z = at["zeros_len4097"]
        swapped = out.copy()
        plus, minus = np.flatnonzero(vals[z, :lens[z]] == 0.0), None
        minus = [j for j in plus if np.signbit(vals[z, j])]
        plus = [j for j in plus if not np.signbit(vals[z, j])]
        assert len(plus) == 3 and len(minus) == 4 and set(plus) <= set(out[z]) and not set(minus) & set(out[z])
        swapped[z, list(out[z]).index(plus[0])] = minus[0]
        assert X._check_topk(vals, lens, swapped)["long_claimed_valid"] == X._check_topk(vals, lens, out)["long_claimed_valid"] - 1
        c = at["crowd3000_len8192"]
        v = vals[c, :lens[c]]
        bins = X.half_bins(v, 12)
        b, above, _ = X.threshold_bin(bins, topk)
        kept = np.flatnonzero(bins == b)[:topk]
        pick = kept[np.argsort(-dsa.order_keys(v[kept]), kind="stable")[:topk - above]]
        assert not dsa.selection_ok(v, lens[c], np.concatenate([np.flatnonzero(bins > b), pick]))


def _stub_topk_ops(monkeypatch, calls: list, persistent_dup: bool = False) -> None:
    """vLLM's four top-k ops stood in for by exact numpy ones on CPU tensors, with their calling conventions: prefill reads each row
    from the buffer's start plus rowStarts (stride0 0 here) and writes indices relative to it, the decode ops take a row per
    launch row; each call recorded.  ``persistent_dup`` makes the persistent kernel (launches to 32 rows) repeat an index in its
    first row past top-k."""
    import sys
    import types

    def top(row: np.ndarray, n: int, topk: int) -> np.ndarray:
        if n <= topk:
            return np.where(np.arange(topk) < max(n, 0), np.arange(topk), -1).astype(np.int32)
        return np.sort(np.argsort(-dsa.order_keys(row[:n]), kind="stable")[:topk]).astype(np.int32)

    def rows(name, d, lens, out, topk, dup=False):
        calls.append((name, d.shape[0], topk))
        v, ls = d.numpy(), lens.reshape(-1).tolist()
        for i, n in enumerate(ls):
            out[i] = torch.from_numpy(top(v[i], n, topk))
        long = [i for i, n in enumerate(ls) if n > topk]
        if dup and long:
            out[long[0], 1] = out[long[0], 0]

    def prefill(logits, starts, ends, out, nrows, stride0, stride1, topk):
        assert (stride0, stride1) == (0, 1) and out.shape == (nrows, topk)
        calls.append(("prefill", nrows, topk))
        flat = logits.reshape(-1).numpy()
        for r, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
            out[r] = torch.from_numpy(top(flat[s:], e - s, topk))

    ops = types.ModuleType("vllm._custom_ops")
    ops.top_k_per_row_prefill = prefill
    ops.top_k_per_row_decode = lambda d, next_n, sl, out, nr, s0, s1, topk: rows("decode", d, sl, out, topk)
    vllm = types.ModuleType("vllm")
    vllm._custom_ops = ops
    sai = types.ModuleType("vllm.model_executor.layers.sparse_attn_indexer")
    sai.RADIX_TOPK_WORKSPACE_SIZE = 1 << 20
    for name, mod in (("vllm", vllm), ("vllm._custom_ops", ops), ("vllm.model_executor", types.ModuleType("vllm.model_executor")),
                      ("vllm.model_executor.layers", types.ModuleType("vllm.model_executor.layers")),
                      ("vllm.model_executor.layers.sparse_attn_indexer", sai)):
        monkeypatch.setitem(sys.modules, name, mod)

    def persistent(d, sl, out, ws, topk, max_len):
        assert ws.numel() == 1 << 20 and max_len == d.shape[1]
        rows("persistent", d, sl, out, topk, persistent_dup and d.shape[0] <= 32)

    def cooperative(d, sl, out, ws, topk, max_len):
        assert ws.numel() == 1 << 20 and d.stride(0) % 4 == 0 and max_len == int(sl.max())
        rows("cooperative", d, sl, out, topk)

    monkeypatch.setattr(torch.ops, "_C", types.SimpleNamespace(persistent_topk=persistent, cooperative_topk=cooperative), raising=False)


def test_the_long_launches_on_stubbed_kernels(monkeypatch) -> None:
    """`_long_topk` with the ops stood in for (`_stub_topk_ops`): every launch at both top-k values, prefill's two variants in one
    launch of PREFILL_RADIX_ROW + R rows, cooperative in launches of 8, 16 and R rows, persistent in 32 and R; exact ops pass
    `_long_ok` with the claim's overflow rows recorded valid, and a persistent kernel that repeats an index fails it there only."""
    monkeypatch.setattr(X, "PREFILL_RADIX_ROW", 48)
    calls: list = []
    _stub_topk_ops(monkeypatch, calls)
    long = {}
    for topk in X.LONG_TOPKS:
        vals, lens, names = X._long_rows(topk, 13 + topk)
        long[str(topk)] = r = X._long_topk(vals, lens, names, topk, True, device="cpu")
        assert not r["errors"] and set(r["cases"]) == set(X.LONG_GATING + X.LONG_GATING_COOPERATIVE) | {"decode_fallback"}
        assert all(c["deterministic"] and c["same_sets"] and c["long_overflow_valid"] == c["long_overflow"] for c in r["cases"].values())
        assert r["cases"]["prefill_radix"]["long"] == r["cases"]["prefill_insertion"]["long"]
        R = len(lens)
        sizes = lambda name: [n for c, n, t in calls if c == name and t == topk]  # noqa: E731
        assert sizes("prefill") == [48 + R] * 2 and sizes("decode") == [R] * 2
        assert sizes("cooperative") == [8] * 5 + [1] + [8] * 5 + [1] + [16, 16, 9] * 2 + [R] * 2 and R == 41
        assert sizes("persistent") == [32, R - 32] * 2 + [R] * 2
    assert X._long_ok(long, True)
    _stub_topk_ops(monkeypatch, calls, persistent_dup=True)
    vals, lens, names = X._long_rows(X.TOPK, 13 + X.TOPK)
    bad = X._long_topk(vals, lens, names, X.TOPK, True, device="cpu")
    assert not X._long_ok({**long, str(X.TOPK): bad}, True)
    failing = {n for n, c in bad["cases"].items() if not X._topk_ok(c)}
    assert failing == {"persistent"} and bad["cases"]["persistent"]["long_claimed_valid"] == bad["cases"]["persistent"]["long_claimed"] - 2


# ------------------------------------------------------------------------------------------------------------ slots, ptx
def test_the_slot_layout_reads_the_block_rows_then_the_scales() -> None:
    nb = 2
    cache = np.zeros((nb, X.BLOCK, X.STRIDE), dtype=np.uint8)
    flat = cache.reshape(-1)
    for slot in (0, 70, 127):
        blk, off = divmod(slot, X.BLOCK)
        base = blk * X.BLOCK * X.STRIDE
        flat[base + off * X.D: base + off * X.D + X.D] = (np.arange(X.D) + slot) % 256
        flat[base + X.BLOCK * X.D + off * 4: base + X.BLOCK * X.D + off * 4 + 4] = np.array([slot + 1], np.uint32).view(np.uint8)
    for slot in (0, 70, 127):
        q, s = X.slot_bytes(cache, slot)
        assert q.tolist() == ((np.arange(X.D) + slot) % 256).tolist() and s == slot + 1
    m = X.slot_mask(nb, [0, 70, 127])
    assert m.sum() == 3 * (X.D + 4) and (flat[~m] == 0).all()


def test_the_ptx_scan_wants_one_specialization_per_kernel(tmp_path) -> None:
    body = "fma.rn.f32 %f1, %f2, %f3, %f4;\nmul.rn.f32 %f5, %f2, %f3;\ncvt.rn.bf16.f32 %rs1, %f1;\n.maxntid 128, 1, 1\n"
    for i, k in enumerate(X.KERNELS):
        (tmp_path / "c" / str(i)).mkdir(parents=True)
        (tmp_path / "c" / str(i) / f"{k}.ptx").write_text(body)
    got = X.ptx(tmp_path / "c", tmp_path / "out")
    assert got["ok"] and got["kernels"]["_fused_q_kernel"][0]["counts"] == {"fma.rn.f32": 1, "mul.rn.f32": 1, "cvt.rn.bf16.f32": 1}
    assert len(list((tmp_path / "out" / "ptx").glob("*.ptx"))) == 2
    (tmp_path / "c" / "2").mkdir()
    (tmp_path / "c" / "2" / "_fused_q_kernel.ptx").write_text(body + "add.f32 %f9, %f1, %f2;\n")
    assert X.ptx(tmp_path / "c", tmp_path / "out")["ok"] is False


def test_the_programs_bind_at_glm53s_statics() -> None:
    got = X.bind_check()
    assert got["ok"] and "SparseMla_v1{T=2048,NH=8,...}" in got["bindings"] and "RoPEHeadGptJ_v2{D=64}" in got["bindings"]
    assert "DsaMqaLogits_v3{N=64,H=32,D=128}" in got["bindings"]


def test_the_statics_are_glm53s() -> None:
    cfg = json.loads((Path(__file__).parents[1] / "check" / "fixtures" / "glm_moe_dsa" / "config.json").read_text())
    assert (cfg["index_n_heads"], cfg["index_head_dim"], cfg["index_topk"], cfg["qk_rope_head_dim"]) == (X.H, X.D, X.TOPK, X.ROPE)
    assert cfg["indexer_rope_interleave"] and cfg["rope_interleave"]
    assert (cfg["num_attention_heads"] // 8, cfg["kv_lora_rank"], cfg["qk_nope_head_dim"], cfg["q_lora_rank"]) == (X.NH, X.DV, X.NOPE, X.Q_LORA)
    assert cfg["rope_parameters"] == {"rope_theta": X.THETA, "rope_type": "default"} and cfg["max_position_embeddings"] == X.MAXPOS
    assert cfg["rms_norm_eps"] == X.RMS_EPS and M.SPARSE_MLA_GLM53["SCALE"] == X.SCALE and M.SPARSE_MLA_GLM53["NH"] == X.NH


# ----------------------------------------------------------------------------------------------------------------- verdict
TOT = {"words": 10, "mismatch": 0, "unresolved": 0, "reference": 1, "twin_mismatch": 0}
LAUNCHES = [{"phase": "k", "case": "served", "checks": {"deterministic": True, "topk_cleared": True}},
            {"phase": "k", "case": "probe", "checks": {"scale_one": True}},
            {"phase": "q", "case": "served", "checks": {"deterministic": True}},
            {"phase": "logits", "case": "prefill", "checks": {"gather": True}},
            {"phase": "logits", "case": "decode", "checks": {"shape": True}},
            {"phase": "attn", "case": "fa3_sparse", "checks": {"conversion": True}}]
COMPS = [("k", "served"), ("k", "probe"), ("q", "served"), ("logits", "prefill"), ("logits", "decode"), ("attn", "token0"),
         ("attn", "token65")]


def _doc() -> dict:
    return {"schema": X.SCHEMA, "gate": {"ok": True}, "errors": {}, "selection": {"ok": True}, "ptx": {"ok": True}, "bind": {"ok": True},
            "rope": {"ok": True}, "topk": {"ok": True}, "launches": copy.deepcopy(LAUNCHES),
            "comparisons": [{"phase": p, "case": c, **TOT} for p, c in COMPS], "totals": {k: v * len(COMPS) for k, v in TOT.items()}}


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("errors", "logits"), "boom"), (("selection", "ok"), False),
                                        (("ptx", "ok"), None), (("bind", "ok"), False), (("rope", "ok"), False), (("topk", "ok"), False),
                                        (("totals", "mismatch"), 1), (("totals", "unresolved"), 1), (("totals", "reference"), 0),
                                        (("totals", "twin_mismatch"), 1), (("totals", "words"), 0),
                                        (("launches", 0, "checks", "topk_cleared"), False), (("launches", 5, "checks"), {}),
                                        (("comparisons", 5, "mismatch"), 1), (("comparisons", 1, "reference"), 0),
                                        (("errors", "compare/attn/token65"), "boom")])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert X._verdict(_doc())
    doc = _doc()
    node = doc
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    assert not X._verdict(doc)


def test_each_case_has_its_own_verdict() -> None:
    assert all(c["ok"] for c in X.cases(_doc()).values()) and list(X.cases(_doc())) == ["k/served", "k/probe", "q/served",
                                                                                         "logits/prefill", "logits/decode", "attn"]
    for drop in ("launches", "comparisons"):
        doc = _doc()
        doc[drop] = [x for x in doc[drop] if x["phase"] != "attn" and x["case"] != "probe"]
        assert {k for k, c in X.cases(doc).items() if not c["ok"]} == {"attn", "k/probe"}, drop
    doc = _doc()
    doc["errors"]["compare/k/probe"] = "boom"
    doc["comparisons"][3]["mismatch"] = 1
    assert {k for k, c in X.cases(doc).items() if not c["ok"]} == {"k/probe", "logits/prefill"}


def test_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys) -> None:
    """`main` with the GPU's phases stood in for (the comparison jobs of the tests above, one word flipped in the decode logits)
    and vLLM's config module stubbed: the jobs through the spawned worker pool, the per-case verdicts, the sealed record, the
    ROPE line and the exit status, so a slip in the orchestration shows here and not in the H200's first hour."""
    import contextlib
    import sys
    import types

    vc = types.ModuleType("vllm.config")
    vc.CompilationConfig = vc.VllmConfig = lambda **kw: kw
    vc.set_current_vllm_config = lambda cfg: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", vc)
    for k in ("VLLM_USE_DEEP_GEMM", "VLLM_ENABLE_V1_MULTIPROCESSING", "TRITON_CACHE_DIR", "DG_JIT_CACHE_DIR"):
        monkeypatch.setenv(k, "")
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    monkeypatch.setattr(X, "ptx", lambda cache, outdir: {"kernels": {}, "ok": True})
    monkeypatch.setattr(X, "bind_check", lambda: {"bindings": {}, "ok": True})
    k = _kslots_job(tmp_path / "k")
    logits = _logits_job(tmp_path / "l")
    gotd = np.array(X._load(logits[-1]["paths"])["gotd"])
    gotd[1] ^= 4
    logits[-1] = {**logits[-1], "paths": {**logits[-1]["paths"], **X._save(tmp_path / "l2", gotd=gotd)}}

    def gpu(work, outdir, quick, doc):
        doc["selection"] = {"ok": True}
        doc["rope"] = {"ok": True, "k_pe": {"definition": "RoPEHeadGptJ_v1", "L": ["L_a"], "H": ["H_a"]},
                       "q_pe": {"definition": "RoPEHeadGptJ_v1", "L": ["L_a"], "H": ["H_a"]}}
        doc["topk"] = {"ok": True, "cases": {}}
        doc["launches"] += [x for x in copy.deepcopy(LAUNCHES) if x["phase"] in ("k", "logits") and x["case"] != "probe"]
        return [{**k, "rows": [16, 19]}] + logits

    monkeypatch.setattr(X, "gpu", gpu)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"]) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"]
    assert {c: v["ok"] for c, v in doc["cases"].items()} == {"k/served": True, "k/probe": False, "q/served": False,
                                                             "logits/prefill": True, "logits/decode": False, "attn": False}
    assert doc["totals"]["mismatch"] == 1 and doc["totals"]["words"] == 3 * (X.D + 1) + 21 + 6
    assert doc["env"]["VLLM_USE_DEEP_GEMM"] == "0" and doc["env"]["DG_JIT_CACHE_DIR"].endswith("deep_gemm")
    printed = capsys.readouterr().out
    assert "ROPE k_pe=RoPEHeadGptJ_v1 q_pe=RoPEHeadGptJ_v1" in printed and "GLM53-DSA NO-GO" in printed
    assert "no-go cases=['k/probe', 'q/served', 'logits/decode', 'attn']" in printed


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["glm53_dsa_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)
