"""cwgemm_ref.py, the CPU judge of cwgemm.cu's gate: on cgemm's gate directory with cw_l7qs's dumps beside it (the
same served words and receipts, cplane's plane of c and its U + 64 bytes, written from the definition in the judge's
docstring) it passes, and it fails by the one flag whose dumped bytes differ, when cgemm's own gate fails, or when
cw_l7qs's verdicts are negative or name other tiles."""
from __future__ import annotations

import itertools
import json
import shutil
import struct
import sys
from pathlib import Path

import cwgemm_ref as W
import numpy as np
import pytest
import test_cgemm_ref as G

C, T, SFX = W.cr, W.T, G.SFX
DUMPS = {"O": ("Y_cw_l7qs", np.int32), "R": ("R_cw_l7qs", np.uint32), "plane": ("plane", np.uint32),
         "UB": ("UB", np.uint8)}


def write_gate(d: Path) -> None:
    """cgemm's gate in Dither-C's numerics, and what cw_l7qs dumps on the same tile: it serves the same words, and
    cplane's word (kt 128 + row) 4 + j holds c(128 tn + row, 128 kt + 32 j + 4 w + b) at bit 8 b + w."""
    G.write_full(d, "co32c")
    k = json.loads((d / "meta.json").read_text())["k"]
    key = b"".join(struct.pack(">Q", w) for w in G.KEY)
    c = C.patch_bits(key, C.C_TAG, G.TN * T, T, k)
    plane = np.zeros((k // 128, T, 4), np.uint32)
    for kt, row, j, b, w in itertools.product(range(k // 128), range(T), range(4), range(4), range(8)):
        plane[kt, row, j] |= np.uint32(int(c[row, 128 * kt + 32 * j + 4 * w + b]) << (8 * b + w))
    plane.tofile(d / ("plane" + SFX))
    (C.u_rows(key, G.TN * T, T) + 64).astype(np.uint8).tofile(d / ("UB" + SFX))
    shutil.copy(d / ("Y_c_l7qs" + SFX), d / ("Y_cw_l7qs" + SFX))
    shutil.copy(d / ("R_c_l7qs" + SFX), d / ("R_cw_l7qs" + SFX))
    (d / "cw_meta.json").write_text(json.dumps(
        {"device_ok": True, "control_rejected": True, "device": "made up", "tiles": [[G.TM, G.TN]]}))


@pytest.fixture(scope="module")
def right(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("cw")
    write_gate(d)
    return d


@pytest.fixture
def gate(right, tmp_path) -> Path:
    return shutil.copytree(right, tmp_path / "gate")


def wrong(r: dict) -> list[str]:
    """cw_l7qs's flags of the tile that are false."""
    return [name for name, v in r["cw_l7qs"]["tiles"][f"{G.TM},{G.TN}"].items() if v is False]


def test_a_right_gate_passes(right):
    r = W.check(right)
    t = r["cw_l7qs"]["tiles"][f"{G.TM},{G.TN}"]
    assert r["ok"] and wrong(r) == [] and G.wrong(r) == [], r
    assert set(DUMPS) <= set(t) and t["Wp_formula"] and t["sum_max"] <= 255


@pytest.mark.parametrize("flag", list(DUMPS))
def test_one_wrong_dumped_word_fails_by_its_flag(gate, flag):
    file, dtype = DUMPS[flag]
    G.flip(gate / (file + SFX), 300, dtype)
    r = W.check(gate)
    assert not r["ok"] and wrong(r) == [flag] and G.wrong(r) == []


def test_a_gate_cgemm_ref_fails_is_failed(gate):
    G.flip(gate / ("PB" + SFX), 300, np.int8)
    r = W.check(gate)
    assert not r["ok"] and wrong(r) == [] and G.wrong(r) == ["PB"]


@pytest.mark.parametrize("change", [{"device_ok": False}, {"control_rejected": False},
                                    {"tiles": [[G.TM, G.TN], [G.TM, G.TN]]}])
def test_cw_l7qs_own_verdicts_and_tiles_count(gate, change):
    """A negative verdict of cw_l7qs's, or tiles that are not cgemm's list, fails a gate whose every dump is right."""
    cw = json.loads((gate / "cw_meta.json").read_text())
    (gate / "cw_meta.json").write_text(json.dumps(cw | change))
    r = W.check(gate)
    assert not r["ok"] and wrong(r) == [] and G.wrong(r) == []


def test_main_is_exit_1_when_any_gate_fails(right, gate, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["cwgemm_ref.py", str(right)])
    assert W.main() == 0 and json.loads(capsys.readouterr().out)["ok"]
    G.flip(gate / ("UB" + SFX), 0, np.uint8)
    monkeypatch.setattr(sys, "argv", ["cwgemm_ref.py", str(gate), str(right)])    # a right gate after it clears nothing
    assert W.main() == 1
    assert [json.loads(line)["ok"] for line in capsys.readouterr().out.splitlines()] == [False, True]
