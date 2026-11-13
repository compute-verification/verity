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
from verity.definitions.numerics import flashmla_dsv32_twin as V
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
        "schema": X.SCHEMA, "gate": {"ok": True}, "selection": {"ok": True}, "write": {"ok": True},
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
    by_section = ['\t.section\t.text._Z3bazv,"ax",@progbits\n', "        /*0000*/                   EXIT ;\n"]
    assert [n for _, n, _ in X.split_sass(by_section)] == ["_Z3bazv"]
    assert X.SASS_KEEP.search("_ZN4sm906decode10sparse_fp839flash_fwd_splitkv_mla_fp8_sparse_kernelINS1_14KernelTemplateIL"
                              "NS1_9ModelTypeE0ELi64EEENS1_9TMAParamsEEEvx")
    assert X.SASS_INSTANCE.search("flash_fwd_splitkv_mla_fp8_sparse_kernelINS1_14KernelTemplateIL9ModelType0ELi64EEE")
    assert X.SASS_INSTANCE.search("flash_fwd_splitkv_mla_fp8_sparse_kernelINS1_14KernelTemplateILNS1_9ModelTypeE0ELi64EEE")
    assert not X.SASS_INSTANCE.search("flash_fwd_splitkv_mla_fp8_sparse_kernelINS1_14KernelTemplateIL9ModelType1ELi64EEE")
    assert not X.SASS_INSTANCE.search("flash_fwd_splitkv_mla_fp8_sparse_kernelINS1_14KernelTemplateILNS1_9ModelTypeE1ELi64EEE")
    assert not X.SASS_INSTANCE.search("flash_fwd_splitkv_mla_fp8_sparse_kernelINS1_14KernelTemplateIL9ModelType0ELi128EEE")


def test_the_listing_is_made_image_by_image(tmp_path, monkeypatch) -> None:
    """Each sm_90a image is disassembled alone, so an image the disassembler cannot read (another architecture's) stops
    nothing; the images that hold a kept function stay beside the listings, and one that fails to list fails the stage."""
    import gzip
    import subprocess

    instance = "_ZN4sm906decode10sparse_fp839flash_fwd_splitkv_mla_fp8_sparse_kernelINS1_14KernelTemplateIL9ModelType0ELi64EEEEEvv"
    planner = "_ZN4smxx6decode23get_mla_metadata_kernelE24GetDecodeSchedMetaParams"
    body = "        /*0000*/                   FMUL R0, R0, R2 ;             /* 0x0000000200007220 */\n"
    listings = {"lib.1.sm_90a.cubin": f"\t\tFunction : {planner}\n{body}\t\tFunction : _Z5otherv\n{body}",
                "lib.3.sm_90a.cubin": f"\t\tFunction : _Z5otherv\n{body}",
                "lib.10.sm_90a.cubin": f"\t\tFunction : {instance}\n{body}{body}"}
    failing: set[str] = set()

    def run(cmd, **kw):
        if cmd[1:3] == ["-xelf", "all"]:
            for name in (*listings, "lib.2.sm_100.cubin"):
                (kw["cwd"] / name).write_bytes(name.encode())
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[1] == "-sass":
            name = cmd[2].rsplit("/", 1)[-1]
            return subprocess.CompletedProcess(cmd, int(name in failing), listings[name], "")
        return subprocess.CompletedProcess(cmd, 0, "demangled\n", "")

    monkeypatch.setattr(X, "flashmla_library", lambda: "lib.so")
    monkeypatch.setattr(X.C, "_archs", lambda so: {"so": so, "sha256": "0" * 64, "sass_archs": ["100", "90a"]})
    monkeypatch.setattr(X.C, "_run", run)
    rec = X.sass(str(tmp_path))
    assert rec["ok"] and rec["instance"] == ["001.sm_90a.sass.gz"]
    assert [(e["image"], e["function"], e["instructions"]) for e in rec["kept"]] == [("lib.1.sm_90a.cubin", planner, 1),
                                                                                  ("lib.10.sm_90a.cubin", instance, 2)]
    assert [(i["file"], i["functions"], i["kept"]) for i in rec["images"]] == [("lib.1.sm_90a.cubin", 2, 1), ("lib.3.sm_90a.cubin", 1, 0),
                                                                              ("lib.10.sm_90a.cubin", 1, 1)]
    assert sorted(p.name for p in (tmp_path / "sass" / "elf").iterdir()) == ["lib.1.sm_90a.cubin", "lib.10.sm_90a.cubin"]
    with gzip.open(tmp_path / "sass" / "001.sm_90a.sass.gz", "rt") as f:
        assert f.read() == f"\t\tFunction : {instance}\n{body}{body}"
    assert json.loads((tmp_path / "sass" / "index.json").read_text()) == rec
    failing.add("lib.3.sm_90a.cubin")
    assert not X.sass(str(tmp_path / "again"))["ok"]


def test_the_comparison_takes_the_kernel_words(made) -> None:
    d = made / "capture"
    job = {"dir": str(d), "case": "d3", "tokens": [0, 3]}
    r = X.compare_tokens(job)
    assert r["differ"] == 0 and r["words"] == 3 * X.H_Q * X.DV and r["twin_words"] == 3 * (X.NH + 1) * X.DV
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


def test_the_tokens_held_to_the_definition_are_spread_over_a_case() -> None:
    from verity.definitions.registry import flashmla_dsv32 as R

    S = R.FLASHMLA_GLM53
    assert (S["NH"], S["DN"], S["DR"], S["QT"], S["BN"], S["SCALE"]) == (X.NH, X.DN, X.DR, X.QT, X.BN, X.SCALE)
    assert R.GLM53_TOPK == 2048 and V.step_splits(14, topk=2048) == _planner(14, 2048)[2]
    assert X.definition_tokens(24, 2, False) == [0, 23] and X.definition_tokens(24, 3, False) == [0, 12, 23]
    assert X.definition_tokens(16, 1, True) == list(range(16)) and X.definition_tokens(16, 0, True) == []
    assert X.definition_tokens(1, 2, False) == [0] and X.definition_tokens(5, 1, False) == [0]


@pytest.mark.slow
def test_the_definitions_evaluator_takes_the_kernel_words(made) -> None:
    """A token through `FlashMlaV32Decode_v1` itself, bound at the capture's statics and the step's SPLITS: with one kernel
    word flipped, that word and no other differs, so the Definition, not only its twin, gives the fabricated words.  One
    evaluation of 8 heads over 128 rows of 512 bytes, which no smaller binding stands for: the statics are the capture's."""
    d = made / "capture"
    out = np.load(d / "d3" / "out.npy")
    good = out.copy()
    try:
        out[FLIP] ^= 1
        np.save(d / "d3" / "out.npy", out)
        r = X.definition_token({"dir": str(d), "case": "d3", "token": FLIP[0]})
    finally:
        np.save(d / "d3" / "out.npy", good)
    assert (r["words"], r["differ"]) == (X.NH * X.DV, 1)
    assert r["target"].startswith("FlashMlaV32Decode_v1{NH=8,SPLITS=(0,1,2),DN=512,DR=64,QT=128,BN=64,")
    assert r["first"] == {"token": FLIP[0], "head": FLIP[1], "word": FLIP[2], "definition": int(good[FLIP]),
                          "kernel": int(good[FLIP]) ^ 1}


def test_the_probes_count_the_rows_the_two_reads_stage_differently(made) -> None:
    r = X.compare_probes(made / "capture", ["probe"])
    assert r["hand"] == {"byte": X.HAND_BYTE, "scale": X.HAND_SCALE, "sparse": 0x3FE2, "readback": 0x3FE1,
                         "row_words_differ": r["hand"]["row_words_differ"]} and r["hand"]["row_words_differ"] > 0
    assert r["rows"] == 4 and r["words"] == 4 * X.DN and r["rows_differ"] >= 2 and r["special_rows"] == 0
    assert r["by_writer"]["planted"]["rows"] == 3 and r["by_writer"]["vllm"]["rows"] == 1
    assert r["words_differ"] == sum(w["words_differ"] for w in r["by_writer"].values()) >= r["one_ulp"] > 0
    assert r["by_writer"]["planted"]["most_in_a_row"] >= r["hand"]["row_words_differ"] >= r["by_writer"]["planted"]["least_in_a_row"]
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
        bad["errors"] = {"stack": "boom"}
        launch.write_text(json.dumps(bad))
        assert not X.compare(str(made), 0)["ok"]
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
    index = made / "sass" / "index.json"
    listing = index.read_text()
    try:
        index.unlink()
        none = X.compare(str(made), 0)
        assert not none["ok"] and none["sass"]["ok"] is False and none["sass"]["error"].startswith("FileNotFoundError")
        index.write_text(json.dumps({"ok": False, "instance": [], "library": {}, "kept": []}))
        assert not X.compare(str(made), 0)["ok"]
    finally:
        index.write_text(listing)
    assert doc["definition"] == {"asked": 0, "tokens": 0, "words": 0, "differ": 0, "targets": [], "cases": {}, "first": None}


def test_the_record_counts_the_tokens_held_to_the_definition(made, monkeypatch) -> None:
    """`compare`'s count of the tokens held to the Definition, the evaluator stood in for: one token a case, a word that
    differs fails the record, and so does none at all; an evaluation that raises is raised, and leaves no record."""
    def stub(differ: int, error: bool = False):
        def run(job):
            if error and job["case"] == "probe":
                raise RuntimeError("boom")
            return {"case": job["case"], "token": job["token"], "words": X.NH * X.DV, "differ": differ if job["case"] == "d3" else 0,
                    "first": {"token": job["token"], "head": 2, "word": 9, "definition": 5, "kernel": 4} if differ else None,
                    "target": "FlashMlaV32Decode_v1{...}"}
        return run

    monkeypatch.setattr(X, "definition_token", stub(0))
    doc = X.compare(str(made), 0, definition=1)
    held = doc["definition"]
    assert doc["ok"] and (held["asked"], held["tokens"], held["words"], held["differ"]) == (1, 2, 2 * X.NH * X.DV, 0)
    assert {c: v["tokens"] for c, v in held["cases"].items()} == {"d3": [0], "probe": [0]}
    assert held["targets"] == ["FlashMlaV32Decode_v1{...}"] and held["first"] is None
    monkeypatch.setattr(X, "definition_token", stub(1))
    doc = X.compare(str(made), 0, definition=1)
    assert not doc["ok"] and doc["totals"]["differ"] == 0 and doc["definition"]["differ"] == 1
    assert doc["definition"]["first"] == {"case": "d3", "token": 0, "head": 2, "word": 9, "definition": 5, "kernel": 4}
    monkeypatch.setattr(X, "definition_token", stub(0, error=True))
    assert (made / X.RECORD).exists()
    with pytest.raises(RuntimeError, match="boom"):
        X.compare(str(made), 0, definition=1)
    assert not (made / X.RECORD).exists()
    monkeypatch.setattr(X, "definition_tokens", lambda tokens, n, whole: [])
    assert not X.compare(str(made), 0, definition=1)["ok"]
