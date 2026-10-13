"""The H200 capture of DeepSeek-V4-Pro's FlashMLA decode (`dsv4_flashmla_capture_gpu`) on the CPU: its cache layout against
FlashMLA's quant.py, its operands, the rows it hands each token's Definition, its comparison jobs on twin-made kernel words, its
verdicts and its main loop with the GPU stood in for."""
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
from verity_vllm.program.kernels import flashmla_twin as T
from verity_vllm.program.registry import flashmla as F
from verity_vllm.properties.record import REGISTRY, digest_of


FLIP = (6, 3, 100)


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


def test_the_statics_are_dsv4s() -> None:
    """DeepSeek-V4-Pro at TP8 (config.json at b5968e91): 128 / 8 heads padded to 64, head_dim 512 = 448 | 64, window 128, the
    indexer's top 1024; FlashMLA's MODEL1 page of 64 tokens is 37440 bytes (quant.py:142), and the planner's 132 partitions."""
    assert (X.NH, X.H_Q, X.DN, X.DR, X.D, X.QT, X.BN) == (16, 64, 448, 64, 512, 64, 64)
    assert X.SCALE == 512 ** -0.5 == F.FLASHMLA_DSV4["SCALE"] and X.WINDOW == 128 and X.C4_TOPK == 1024
    assert X.page_bytes(64) == 37440 and X.page_bytes(2) == 1728 and T.num_sm_parts(X.H_Q) == 132
    assert {c for _, c in X.CASES} == set(X.CASE_LAYERS) and len(X.EDGE_SINKS) == X.NH


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
    for _, case in X.CASES:
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
    per = {c: np.diff(X.twin_sched(X.case_operands(c))[1]) for _, c in X.CASES}
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


def test_the_programs_bind_at_dsv4s_statics() -> None:
    rec = X.bind_check()
    assert rec["ok"], rec


LAUNCH = {"b": 8, "checks": {"shape": True, "sched": True, "rerun": True, "planner": True, "returned": True}}


def _doc() -> dict:
    launches = [{"phase": "decode", "case": c, **copy.deepcopy(LAUNCH)} for _, c in X.CASES]
    comps = [{"phase": "decode", "case": c, "words": 10, "mismatch": 0, "reference": 1, "twin_mismatch": 0, "first_mismatch": None}
             for _, c in X.CASES]
    return {"schema": X.SCHEMA, "gate": {"ok": True}, "errors": {}, "selection": {"ok": True}, "bind": {"ok": True},
            "launches": launches, "comparisons": comps,
            "totals": {k: sum(r[k] for r in comps) for k in X.TOTALS}}


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("selection", "ok"), False), (("bind", "ok"), False),
                                        (("errors", "stack"), "boom"), (("launches", 2, "checks", "sched"), False),
                                        (("launches", 0, "checks", "rerun"), False), (("comparisons", 1, "mismatch"), 1),
                                        (("comparisons", 4, "twin_mismatch"), 3), (("comparisons", 5, "reference"), 0),
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
    assert list(X.cases(_doc())) == [f"decode/{c}" for c in X.CASE_LAYERS] and all(c["ok"] for c in X.cases(_doc()).values())
    doc = _doc()
    doc["launches"] = [x for x in doc["launches"] if x["case"] != "lone"]
    doc["errors"]["compare/decode/edge"] = "boom"
    doc["comparisons"][2]["mismatch"] = 1
    assert {k for k, c in X.cases(doc).items() if not c["ok"]} == {"decode/lone", "decode/edge", "decode/c128"}


@pytest.mark.slow
def test_main_end_to_end_on_stubbed_launches(swa, tmp_path, monkeypatch, capsys) -> None:
    """`main` with the GPU's launches stood in for (the `swa` case's twin-made words, and the same words under `c128` with one
    flipped) and vLLM's config module stubbed: the jobs through the spawned worker pool, the per-case verdicts, the sealed
    record and the exit status, so a slip in the orchestration shows here and not in the H200's first hour."""
    ns, good, bad = swa
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
        doc["launches"] += [{"phase": "decode", "case": c, **copy.deepcopy(LAUNCH)} for c in ("swa", "c128")]
        twin = [j for j in X.case_jobs("swa", good, ns, quick) if j["kind"] == "twin"]
        return twin + [{"phase": "decode", "case": "swa", "kind": "ref", "paths": good, "token": 1}] + \
            [{**j, "case": "c128", "paths": bad} for j in twin]

    monkeypatch.setattr(X, "gpu", gpu)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1", "--quick"]) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"]
    assert {c: v["ok"] for c, v in doc["cases"].items()} == {"decode/swa": True, "decode/c4": False, "decode/c128": False,
                                                             "decode/lone": False, "decode/batch": False, "decode/edge": False}
    assert doc["cases"]["decode/c128"]["totals"]["mismatch"] == 1 and doc["totals"]["words"] == 2 * 8 * X.H_Q * X.D
    printed = capsys.readouterr().out
    assert "DSV4-FLASHMLA NO-GO" in printed
    assert "no-go cases=['decode/c4', 'decode/c128', 'decode/lone', 'decode/batch', 'decode/edge']" in printed


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["dsv4_flashmla_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)
