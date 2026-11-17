import gzip
import json
import re

import numpy as np
import pytest

import methodology as M
import models
import windows


def test_contexts_windows_and_tokens():
    assert M.CONTEXTS == {2048: 16, 8192: 16, 32768: 4} and M.SEEDS == (0, 1, 2)
    assert M.windows(8192) == tuple(range(16)) and M.scored_tokens(2048) == 16 * 2047
    with pytest.raises(ValueError, match="the bench's contexts are"):
        M.windows(4096)


def test_calibration_is_disjoint_from_every_eval_window():
    c = M.CALIBRATION
    start = c["windows"][0] * (c["ctx"] - 1)
    for ctx, n in M.CONTEXTS.items():
        assert n * ctx <= start, ctx


def test_sign_threshold():
    assert M.sign_threshold(16) == 13
    assert M.sign_threshold(6) == 6
    assert M.sign_threshold(5) is None and M.sign_threshold(4) is None


def test_reproduces_is_range_overlap_with_a_floor():
    assert M.REPRODUCE_FLOOR == 0.001
    assert M.reproduces(9.0, 9.0, 9.0085, 9.0085) and not M.reproduces(9.0, 9.0, 9.0095, 9.0095)
    assert M.reproduces(9.0, 9.0, 8.9915, 8.9915) and not M.reproduces(9.0, 9.0, 8.9905, 8.9905)
    assert M.reproduces(9.0, 9.1, 9.05, 9.3) and not M.reproduces(9.0, 9.1, 9.2, 9.3)


def test_compare_needs_disjoint_seeds_and_the_windows():
    a, b = [9.20, 9.21, 9.22], [9.30, 9.31, 9.32]
    better, worse = [1.0] * 16, [1.1] * 16
    assert M.compare(a, b, better, worse) == "<"
    assert M.compare(b, a, worse, better) == ">"
    assert M.compare(a, [9.215, 9.3, 9.31], better, worse) == "≈"
    split = [1.0] * 12 + [1.2] * 4
    assert M.compare(a, b, split, worse) == "≈"
    assert M.compare(a, b, [1.0] * 13 + [1.2] * 3, worse) == "<"
    assert M.compare(a, b, [1.0] * 4, [1.1] * 4) == "< (weak)"
    assert M.compare(a, b, [1.0] * 3 + [1.2], [1.1] * 4) == "≈"
    with pytest.raises(ValueError, match="different windows"):
        M.compare(a, b, [1.0], [1.0, 2.0])


def test_cut_is_tonights_recipe():
    ids = np.arange(100, dtype=np.int64)
    w = windows.cut(ids, 10, 7, range(3))
    assert w.tolist() == [[7] + list(range(0, 9)), [7] + list(range(9, 18)), [7] + list(range(18, 27))]
    w = windows.cut(ids, 10, None, [2])
    assert w.tolist() == [list(range(20, 30))]
    with pytest.raises(ValueError, match="only 11 windows"):
        windows.cut(ids, 10, 7, [11])


def test_the_lock_covers_every_family_set_and_context():
    lock = windows.lock()
    fams = {m["tokenizer"] for m in models.MODELS.values()}
    assert set(lock) == fams
    for fam in fams:
        want = {windows.key(s, c) for s in M.EVAL_SETS for c in M.CONTEXTS}
        want.add(windows.key(M.CALIBRATION["set"], M.CALIBRATION["ctx"], True))
        assert set(lock[fam]) == want, fam
        for k, rec in lock[fam].items():
            assert re.fullmatch(r"[0-9a-f]{64}", rec["sha256"]), (fam, k)
            if k.endswith("calibration"):
                assert rec["windows"] == [M.CALIBRATION["windows"][0], M.CALIBRATION["windows"][-1]]
            else:
                assert rec["windows"] == [0, M.CONTEXTS[int(k.split("/")[1])] - 1]
        assert len({r["bos_lead"] for r in lock[fam].values()}) == 1


def test_llama_c4_2048_is_tonights_window_list():
    """The earlier linears harness's c4val2048x16.npy and c4cal2048x16.npy (c4_windows.py), by their sha256."""
    lock = windows.lock()["llama3"]
    assert lock["c4/2048"]["sha256"] == "1ca575b1129dc9262d4e9882795f5cfb1decb7952fb8b9c383ea9691399a0664"
    assert lock["c4/2048/calibration"]["sha256"] == "d9788b2e4d3c48bd5fa425cf3cee1049cfe6088ec57c9fe9b37b86b0bfe42331"


def test_check_refuses_another_list():
    rec = windows.lock()["qwen2.5"]
    w = np.zeros((16, 2048), dtype=np.int64)
    with pytest.raises(ValueError, match="locked " + rec["wikitext2/2048"]["sha256"]):
        windows.check("qwen2.5", "wikitext2", 2048, w)
    with pytest.raises(ValueError, match="the methodology's"):
        windows.check("qwen2.5", "wikitext2", 2048, w[:8])
    with pytest.raises(ValueError, match="has no gpt2"):
        windows.check("gpt2", "wikitext2", 2048, w)


def test_datasets_are_pinned():
    assert set(windows.DATASETS) == set(M.EVAL_SETS)
    for d in windows.DATASETS.values():
        assert re.fullmatch(r"[0-9a-f]{64}", d["sha256"]) and re.fullmatch(r"art:[0-9a-f]{64}", d["art"])
        assert d["format"] in ("jsonl.gz", "json")


class Tok:
    """A tokenizer stand-in: one id per character."""
    bos_token_id = 1

    def __call__(self, s, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in s]}


def test_build_reads_the_pinned_file(tmp_path, monkeypatch):
    docs = [{"text": "x" * 3000 + str(i)} for i in range(M.C4_DOCS + 5)]
    path = tmp_path / "c4.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(d) + "\n" for d in docs)
    lines = tmp_path / "wt.json"
    lines.write_text(json.dumps(["a" * 50] * 6000))
    sets = {"c4": {"file": path.name, "format": "jsonl.gz", "sha256": windows.file_sha256(path)},
            "wikitext2": {"file": lines.name, "format": "json", "sha256": "0" * 64}}
    monkeypatch.setattr(windows, "DATASETS", sets)
    encode, bos = windows.encoder(Tok())
    texts = windows.read_texts("c4", path)
    assert len(texts) == M.C4_DOCS and texts[-1].endswith(str(M.C4_DOCS - 1))
    w, rec = windows.build("c4", 2048, encode, bos, tmp_path)
    stream = windows.stream(texts, encode)
    assert w.shape == (16, 2048) and (w[:, 0] == 1).all() and (w[3, 1:] == stream[3 * 2047:4 * 2047]).all()
    assert rec == {"sha256": windows.digest(w), "windows": [0, 15], "stream_tokens": len(stream), "bos_lead": True}
    cal, _ = windows.build("c4", 2048, encode, bos, tmp_path, calibration=True)
    assert (cal[0, 1:] == stream[100 * 2047:101 * 2047]).all()
    with pytest.raises(ValueError, match="calibration is C4 at 2048"):
        windows.build("c4", 8192, encode, bos, tmp_path, calibration=True)
    with pytest.raises(ValueError, match="pinned " + "0" * 64):
        windows.build("wikitext2", 2048, encode, bos, tmp_path)
