"""The capture of GLM-5.3's sparse read of the FP8 cache (`glm53_flashmla_fp8_capture_gpu`) on the CPU: its row layout, its
operands, the hand example through the two reads, the listing's split into functions, and its comparison on twin-made kernel
words."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import json

import numpy as np
import pytest

from tests.properties import glm53_flashmla_fp8_capture_gpu as X
from verity.definitions.numerics import flashmla_twin as T
from verity.definitions.numerics import flashmla_v32_twin as V
from verity_vllm.properties.record import digest_of

#: a row width the twin runs in a fraction of a second: two blocks a token
SMALL_TOPK = 128
FLIP = (1, 3, 100)


def _planner(t: int, topk: int):
    metas, ns = T.sched_meta([topk], topk=topk, parts=T.num_sm_parts(X.H_Q, s_q=t))
    return metas, ns, T.request_splits(metas, ns, 0, topk // X.BN)


def _fabricate(root, lay, cases: dict[str, dict[str, np.ndarray]], topk: int) -> None:
    """A capture directory as `launch` leaves it, the twin's words standing for the kernel's and `dense_rows` for the
    read-back's."""
    d = root / "capture"
    d.mkdir(parents=True)
    cache = X.host_cache(lay)
    np.save(d / "cache.npy", cache)
    for k in ("lens", "table", "numpy"):
        np.save(d / f"layout_{k}.npy", lay[k])
    launches = []
    for case, o in cases.items():
        t = o["q"].shape[0]
        metas, ns, splits = _planner(t, topk)
        g = X.global_indices(lay, o)
        out = np.zeros((t, X.H_Q, X.DV), dtype=np.uint16)
        for i in range(t):
            q = np.zeros((X.NH + 1, X.DK), dtype=np.uint16)
            q[:X.NH] = o["q"][i]
            y = V.decode(q, *X.rows_at(cache, g[i]), g[i], SPLITS=splits, QT=X.QT, BN=X.BN, DV=X.DV, SCALE=X.SCALE)
            out[i, :X.NH + 1], out[i, X.NH + 1:] = y, y[X.NH]
        (d / case).mkdir()
        for k, a in (("q", o["q"]), ("idx", o["idx"]), ("req", o["req"]), ("gidx", g), ("out", out),
                     ("metas", np.array([m.words() for m in metas], dtype=np.int32)), ("num_splits", np.array(ns, dtype=np.int32))):
            np.save(d / case / f"{k}.npy", a)
        launches.append({"case": case, "tokens": t, "checks": {"rerun": True}})
    blocks = sorted({int(lay["table"][r, p // X.PBS]) for r, p in X.probe_rows(lay)})
    nope, scale, rope = X.unpack_rows(cache[blocks].reshape(-1, X.ROW_BYTES))
    (d / "readback").mkdir()
    np.save(d / "readback" / "rows.npy", np.concatenate([X.dense_rows(nope, scale), rope], axis=1))
    np.save(d / "readback" / "blocks.npy", np.array(blocks, dtype=np.int32))
    (d / "launch.json").write_text(json.dumps({
        "schema": X.SCHEMA, "gate": {"ok": True}, "errors": {}, "selection": {"ok": True}, "write": {"ok": True},
        "readback": {"rerun": True}, "cases": list(cases), "launches": launches}))
    (root / "sass").mkdir()
    (root / "sass" / "index.json").write_text(json.dumps({"ok": True, "instance": ["000.sm_90a.sass.gz"], "library": {}, "kept": []}))


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    """A capture at `SMALL_TOPK` indices a token: `d3` (three decode tokens, two splits each) and `probe` cut to the hand row, a
    row of tie scales and two finite rows.  Yields its directory with `TOPK` still patched."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(X, "TOPK", SMALL_TOPK)
        lay = X.layout(True)
        probe = X.case_operands("probe", lay)
        keep = [X.HAND_ROW, 1, 40, X.PLANT_ROWS + 5]
        cases = {"d3": X.case_operands("d3", lay), "probe": {k: v[keep] for k, v in probe.items()}}
        root = tmp_path_factory.mktemp("made")
        _fabricate(root, lay, cases, SMALL_TOPK)
        yield root


def test_the_row_is_flashmlas_656_bytes() -> None:
    rng = np.random.default_rng(0)
    nope, scale, rope = X.numpy_rows(rng, 5)
    rows = X.pack_rows(nope, scale, rope)
    assert rows.shape == (5, 656) and (X.DN, X.DR, X.QT, X.DK, X.DV) == (512, 64, 128, 576, 512)
    assert np.array_equal(rows[:, :512], nope)
    assert np.array_equal(rows[:, 512:528].copy().view("<f4"), scale.view(np.float32))
    assert np.array_equal(rows[:, 528:].copy().view("<u2"), rope)
    for a, b in zip(X.unpack_rows(rows), (nope, scale, rope)):
        assert np.array_equal(a, b)
    assert not np.any((nope & 0x7F) == 0x7F)


def test_the_hand_example_reads_two_ways() -> None:
    """Scale 1 + 2^-8 + 2^-20 on 1.75: the sparse kernel rounds the scale to bf16 first (1.0078125) and stages 1.765625; the
    read-back multiplies by the f32 scale and stages 1.7578125."""
    nope = np.full((1, X.DN), X.HAND_BYTE, dtype=np.uint8)
    scale = np.full((1, 4), X.HAND_SCALE, dtype=np.uint32)
    sparse = V.kv_rows(nope, scale, np.zeros((1, X.DR), np.uint16), [True], X.QT)
    assert sparse[0, 0] == 0x3FE2 and X.dense_rows(nope, scale)[0, 0] == 0x3FE1
    assert not V.kv_rows(nope, scale, np.ones((1, X.DR), np.uint16), [False], X.QT).any()


def test_a_row_is_valid_unless_its_index_is_minus_one() -> None:
    assert V.row_valid(np.array([0, -1, 5, 2**31 - 1, -2], dtype=np.int32)).tolist() == [True, False, True, True, True]


@pytest.mark.parametrize("quick", [True, False])
def test_every_index_names_a_row_of_its_request(quick: bool) -> None:
    lay = X.layout(quick)
    used = lay["table"][lay["table"] > 0]
    assert used.size == np.unique(used).size == lay["blocks"] - 1 and used.max() == lay["blocks"] - 1
    assert lay["numpy"][-1] and int(lay["lens"][-1]) == X.PLANT_ROWS + 4096
    for case in X.case_names(quick):
        o = X.case_operands(case, lay)
        g = X.global_indices(lay, o)
        t = o["req"].size
        assert o["q"].shape == (t, X.NH, X.DK) and o["idx"].shape == (t, X.TOPK) and o["idx"].dtype == np.int32
        assert np.array_equal(g == -1, o["idx"] == -1) and int(g[g != -1].min()) >= X.PBS
        assert all(int(row.max()) < int(lay["lens"][r]) for r, row in zip(o["req"], o["idx"]))
        if case.startswith("d"):
            assert t == int(case[1:]) == np.unique(o["req"]).size
        if case.startswith("probe"):
            assert ((g != -1).sum(axis=1) == 1).all()


def test_the_decode_cases_stand_on_every_cut_of_the_planner() -> None:
    cut = {t: _planner(t, X.TOPK)[2] for t in range(1, 140)}
    firsts = [t for t in cut if t == 1 or cut[t] != cut[t - 1]]
    assert len(firsts) == 12 and set(firsts) <= set(X.DECODE_T) and {t - 1 for t in firsts[1:]} <= set(X.DECODE_T)
    assert cut[1] == tuple(range(33)) and cut[67] == cut[139] == (0, 32) and cut[16] == (0, 5, 10, 15, 20, 25, 30, 32)
    assert {len(cut[t]) - 1 for t in X.QUICK_T} == {32, 11, 3, 1}


def test_the_planted_rows_are_where_edge_reads_them() -> None:
    lay = X.layout(True)
    plant = len(lay["lens"]) - 1
    nope, scale, rope = X.rows_at(X.host_cache(lay), X.slots(lay, plant, np.arange(X.PLANT_ROWS)))
    assert scale[X.HAND_ROW].tolist() == [X.HAND_SCALE] * 4 and nope[X.HAND_ROW, 0] == X.HAND_BYTE
    assert nope[7, 17] == 0x7F and nope[7, 300] == 0xFF and set(nope[8].tolist()) == {0x7E, 0xFE}
    assert set((scale[13:32] & 0xFFFF).reshape(-1).tolist()) == {0x7FFF, 0x8000, 0x8001}
    assert rope[11, :3].tolist() == [0x7F80, 0xFF80, 0x7FC0] and all(r in X.PLANTED for r in X.PLANTED_SCALES)
    poison = X.rows_at(X.host_cache(lay), [X.PBS - 1])
    assert poison[0][0, 0] == 0x7F and poison[1][0, 0] == 0x7FC00000 and poison[2][0, 0] == 0x7FC0
    o = X.case_operands("edge", lay)
    assert (o["req"] == plant).all() and X.HAND_ROW in o["idx"][0] and (o["idx"][6] == -1).all()
    assert (o["idx"][7] != -1).sum() == 1 and np.unique(o["idx"][8]).size == 1 and not o["q"][10].any()
    assert (o["idx"][12, :X.BN] == -1).all() and (o["idx"][13, :5 * X.BN] == -1).all() and (o["idx"][13, 5 * X.BN:] != -1).all()
    assert set(range(X.PLANT_ROWS)) <= set(o["idx"][15].tolist())
    flat = X.case_operands("edgeflat", lay)
    assert flat["req"].size == 80 and _planner(80, X.TOPK)[2] == (0, 32) and len(_planner(16, X.TOPK)[2]) == 8


def test_the_listing_splits_into_functions() -> None:
    listing = """
Fatbin elf code:
================
arch = sm_90a
code version = [1,7]

	code for sm_90a
		Function : _Z3foov
	.headerflags	@"EF_CUDA_SM90"
        /*0000*/                   MOV R1, c[0x0][0x28] ;        /* 0x00000a0000017a02 */
                                                                 /* 0x000fc40000000f00 */
        /*0010*/                   FMUL R0, R0, R2 ;             /* 0x0000000200007220 */
		Function : _Z3barv
        /*0000*/                   EXIT ;                        /* 0x000000000000794d */

Fatbin ptx code:
================
arch = sm_90a
""".splitlines(keepends=True)
    got = [(a, n, "".join(b)) for a, n, b in X.split_sass(listing)]
    assert [(a, n) for a, n, _ in got] == [("sm_90a", "_Z3foov"), ("sm_90a", "_Z3barv")]
    assert "FMUL" in got[0][2] and "EXIT" not in got[0][2] and "ptx" not in got[1][2]
    assert X.SASS_KEEP.search("_ZN4sm906decode10sparse_fp839flash_fwd_splitkv_mla_fp8_sparse_kernelINS1_14KernelTemplateIL"
                              "NS1_9ModelTypeE0ELi64EEENS1_9TMAParamsEEEvx")
    assert X.SASS_INSTANCE.search("flash_fwd_splitkv_mla_fp8_sparse_kernelINS1_14KernelTemplateILNS1_9ModelTypeE0ELi64EEE")
    assert not X.SASS_INSTANCE.search("flash_fwd_splitkv_mla_fp8_sparse_kernelINS1_14KernelTemplateILNS1_9ModelTypeE1ELi64EEE")
    assert not X.SASS_INSTANCE.search("flash_fwd_splitkv_mla_fp8_sparse_kernelINS1_14KernelTemplateILNS1_9ModelTypeE0ELi128EEE")


def test_the_comparison_takes_the_kernel_words(made) -> None:
    d = made / "capture"
    job = {"dir": str(d), "case": "d3", "tokens": [0, 3]}
    r = X.compare_tokens(job)
    assert "error" not in r and r["differ"] == 0 and r["words"] == 3 * X.H_Q * X.DV and r["twin_words"] == 3 * (X.NH + 1) * X.DV
    out = np.load(d / "d3" / "out.npy")
    good = out.copy()
    try:
        out[FLIP] ^= 1
        out[2, 40, 7] ^= 1
        np.save(d / "d3" / "out.npy", out)
        r = X.compare_tokens(job)
        assert (r["twin_differ"], r["pad_differ"], r["differ"]) == (1, 1, 2)
        assert r["first"]["token"] == FLIP[0] and r["first"]["head"] == FLIP[1] and r["first"]["word"] == FLIP[2]
        assert r["first"]["kernel"] == r["first"]["twin"] ^ 1 and r["first"]["splits"] == [0, 1, 2]
    finally:
        np.save(d / "d3" / "out.npy", good)


def test_the_probes_count_the_rows_the_two_reads_stage_differently(made) -> None:
    r = X.compare_probes(made / "capture", ["probe"])
    assert r["hand"] == {"byte": X.HAND_BYTE, "scale": X.HAND_SCALE, "sparse": 0x3FE2, "readback": 0x3FE1,
                         "row_words_differ": r["hand"]["row_words_differ"]} and r["hand"]["row_words_differ"] > 0
    assert r["rows"] == 4 and r["words"] == 4 * X.DN and r["rows_differ"] >= 2 and r["special_rows"] == 0
    assert r["by_writer"]["planted"]["rows"] == 3 and r["by_writer"]["vllm"]["rows"] == 1
    assert (r["rope_differ"], r["sparse_model_differ"], r["dense_model_differ"]) == (0, 0, 0)
    assert r["cases"]["probe"] == {"tokens": 4, "pad_heads_disagree": 0} and r["examples"][0]["sparse"] != r["examples"][0]["readback"]


def test_the_record_holds_only_when_no_word_differs(made, capsys) -> None:
    doc = X.compare(str(made), 0)
    assert doc["ok"] and doc["digest"] == digest_of(doc) and doc["totals"]["differ"] == 0
    assert doc["totals"]["words"] == 7 * X.H_Q * X.DV and json.loads((made / X.RECORD).read_text()) == doc
    assert "GLM53-FLASHMLA-FP8 HOLDS" in capsys.readouterr().out
    launch = made / "capture" / "launch.json"
    good = launch.read_text()
    try:
        for path in (("gate", "ok"), ("selection", "ok"), ("write", "ok"), ("readback", "rerun")):
            bad = json.loads(good)
            bad[path[0]][path[1]] = False
            launch.write_text(json.dumps(bad))
            assert not X.compare(str(made), 0)["ok"], path
        bad = json.loads(good)
        bad["launches"][0]["checks"]["rerun"] = "error"
        launch.write_text(json.dumps(bad))
        assert not X.compare(str(made), 0)["ok"]
        bad = json.loads(good)
        bad["cases"].append("edge")
        launch.write_text(json.dumps(bad))
        assert not X.compare(str(made), 0)["ok"]
    finally:
        launch.write_text(good)
    assert "FAILS" in capsys.readouterr().out
