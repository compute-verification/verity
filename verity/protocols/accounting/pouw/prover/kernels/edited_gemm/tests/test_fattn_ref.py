"""fattn_ref.py, the CPU judge of fattn.cu's gate (the softmax of one query's scores, by table): on a gate directory
written from the definition in its docstring it passes, and it fails by the one flag whose dumped value differs, on a
wrong table, or when the device's own verdicts are negative. The bytes of P past those apv reads are not judged."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import fattn_ref as R
import numpy as np
import pytest

S, H, A = 256, 2, (1 << 32) // 100    # a score 100 below the largest is one step of the table
LUT = [round(127 * 2 ** (-t / 16)) for t in range(256)]
#: (the keys' head, the query's column: its position is the column modulo S)
COLS = [(0, 0), (0, 5), (1, S + 127), (1, 128), (0, S + 255), (1, 200)]
FLAGS = ("S", "m", "P", "e0", "sp", "sz")


def softmax(scores: list[int], pos: int) -> tuple[int, list[int]]:
    """The largest score of keys 0 .. pos, and e of each of those keys."""
    m = max(scores[:pos + 1])
    return m, [LUT[min((m - x) * A >> 32, 255)] for x in scores[:pos + 1]]


@pytest.fixture(scope="module")
def right(tmp_path_factory) -> Path:
    d, rng, cols = tmp_path_factory.mktemp("fattn"), np.random.default_rng(29), []
    X = rng.integers(-100, 101, (2, S, 128))
    for k in range(2):
        X[k].astype(np.int8).tofile(d / f"X_{k}.bin")
    for k, col in COLS:
        wh = rng.integers(-20, 21, 128)
        scores, pos = [int(x) for x in X[k] @ wh], col % S
        m, e = softmax(scores, pos)
        P = np.full(S, 77, np.int8)               # past the bytes apv reads, anything
        P[:128 * (pos // 128 + 1)] = 0
        P[1:pos + 1] = e[1:]
        wh.astype(np.int8).tofile(d / f"Wh_{k}_{col}.bin")
        P.tofile(d / f"P_{k}_{col}.bin")
        np.array(scores, np.int32).tofile(d / f"S_{k}_{col}.bin")
        f32 = np.float32
        cols.append([k, col, float(f32(1) / f32(sum(e))), float(f32(e[0]) / f32(sum(e))), m, e[0]])
    (d / "meta.json").write_text(json.dumps({"s": S, "H": H, "rep": 1, "A": A, "lut": LUT, "cols": cols,
                                             "device_ok": True, "control_rejected": True}))
    return d


@pytest.fixture
def gate(right, tmp_path) -> Path:
    return shutil.copytree(right, tmp_path / "gate")


def wrong(r: dict) -> dict[int, list[str]]:
    """Each column's flags that are false, by its place in the gate, the columns with none left out."""
    out = {i: [flag for flag in FLAGS if not c[flag]] for i, c in enumerate(r["cols"])}
    return {i: flags for i, flags in out.items() if flags}


def edit_meta(gate: Path, edit) -> None:
    meta = json.loads((gate / "meta.json").read_text())
    edit(meta)
    (gate / "meta.json").write_text(json.dumps(meta))


def test_a_right_gate_passes(right):
    r = R.check(right)
    assert r["ok"] and r["lut"] and wrong(r) == {}, r
    assert [(c["kvh"], c["col"], c["pos"]) for c in r["cols"]] == [(k, col, col % S) for k, col in COLS]
    # position 0 sees key 0 alone: e(0) = 127 is the whole sum
    assert (r["cols"][0]["L"], r["cols"][0]["nonzero"]) == (127, 1)


def test_the_made_up_gate_reaches_the_tables_end_and_a_larger_score_past_the_query(right):
    """Some key's e is 0 (a score far below the largest), and in some column a key after the query scores higher
    than every key up to it: a judge that took the largest over all keys would differ there."""
    r = R.check(right)
    assert any(c["nonzero"] < c["pos"] + 1 for c in r["cols"])
    past = []
    for (k, col), c in zip(COLS, r["cols"], strict=True):
        scores = np.fromfile(right / f"S_{k}_{col}.bin", np.int32)
        past.append(int(scores.max()) > int(scores[:c["pos"] + 1].max()))
    assert any(past)


@pytest.mark.parametrize(("i", "file", "at", "flag"), [
    (5, "S", 3, "S"), (1, "S", 250, "S"),      # the stored column is judged whole, past the query too
    (5, "P", 2, "P"), (2, "P", 0, "P"),        # key 0 is no part of P
    (1, "P", 100, "P"),                        # nor a key past the query, on the bytes apv reads
])
def test_one_wrong_dumped_word_fails_by_its_flag(gate, i, file, at, flag):
    k, col = COLS[i]
    path = gate / f"{file}_{k}_{col}.bin"
    a = np.fromfile(path, np.int32 if file == "S" else np.int8)
    a[at] ^= 1
    a.tofile(path)
    r = R.check(gate)
    assert not r["ok"] and wrong(r) == {i: [flag]}


def test_a_byte_of_p_that_apv_does_not_read_is_not_judged(gate):
    k, col = COLS[1]
    path = gate / f"P_{k}_{col}.bin"
    a = np.fromfile(path, np.int8)
    a[128] ^= 1
    a.tofile(path)
    assert R.check(gate)["ok"]


@pytest.mark.parametrize(("field", "flag"), [(2, "sp"), (3, "sz"), (4, "m"), (5, "e0")])
def test_one_wrong_value_of_the_devices_fails_by_its_flag(gate, field, flag):
    def edit(meta: dict) -> None:
        v = meta["cols"][4][field]
        meta["cols"][4][field] = float(np.nextafter(np.float32(v), np.float32(1))) if field < 4 else v + 1

    edit_meta(gate, edit)
    r = R.check(gate)
    assert not r["ok"] and wrong(r) == {4: [flag]}


def test_a_wrong_table_fails(gate):
    edit_meta(gate, lambda meta: meta["lut"].__setitem__(200, meta["lut"][200] + 1))
    r = R.check(gate)
    assert not r["ok"] and not r["lut"] and wrong(r) == {}


@pytest.mark.parametrize("verdict", ["device_ok", "control_rejected"])
def test_the_devices_own_verdicts_count(gate, verdict):
    edit_meta(gate, lambda meta: meta.__setitem__(verdict, False))
    r = R.check(gate)
    assert not r["ok"] and r["lut"] and wrong(r) == {}


def test_main_is_exit_1_when_any_gate_fails(right, gate, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fattn_ref.py", str(right)])
    assert R.main() == 0 and json.loads(capsys.readouterr().out)["ok"]
    edit_meta(gate, lambda meta: meta["cols"][0].__setitem__(4, meta["cols"][0][4] - 1))
    monkeypatch.setattr(sys, "argv", ["fattn_ref.py", str(gate), str(right)])    # a right gate after it clears nothing
    assert R.main() == 1
    assert [json.loads(line)["ok"] for line in capsys.readouterr().out.splitlines()] == [False, True]
