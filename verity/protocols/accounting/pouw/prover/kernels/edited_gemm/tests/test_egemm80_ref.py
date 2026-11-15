"""egemm80_ref.py, the CPU judge of egemm80.cu's gate: on a gate directory written from the definition (each arm's
Y = X W^T, and each receipted arm's receipts over X W'^T's partial sums at its own commit spacing) it passes, and it fails
the arm, and only that arm, whose served word or receipt word differs. A judge that passes everything would let a wrong
kernel through the GPU run's gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import egemm80_ref as R
import numpy as np
import pytest

M, N, K, SEED = 2 * R.TM, 2 * R.BN, 128, 0x80A1
TILES_M, TILES_N = M // R.TM, N // R.BN
#: each receipted arm's commit spacing in columns, egemm80.cu's L of the arm. Written out, so the spacings a gate directory
#: is made with are not the ones the judge under test binds
SPACING = {"e16": 16, "e32": 32, "e64": 64}
ARMS = ["plain", *SPACING]


def write_gate(d: Path) -> None:
    """Four 128 x 128 tiles, two rows of two, as the device would dump them if it were right. An arm's receipts absorb
    the words of X[:, :j] W'[:, :j]^T at j = its spacing, twice its spacing, ..., K, and the tile in row tm and column tn
    is the call's tile tm TILES_N + tn."""
    rng = np.random.default_rng(80)
    X = rng.integers(-127, 128, (M, K), dtype=np.int8)
    W = rng.integers(-96, 97, (N, K), dtype=np.int8)
    U = rng.integers(-31, 32, (N, 32), dtype=np.int8)
    for name, a in (("X", X), ("W", W), ("U", U)):
        a.tofile(d / f"{name}.bin")
    x, w = X.astype(np.int64), W.astype(np.int64)
    wp = w + np.tile(U.astype(np.int64), (1, K // 32))
    for arm in ARMS:
        (x @ w.T).astype(np.int32).tofile(d / f"Y_{arm}.bin")
    for arm, spacing in SPACING.items():
        rec = []
        for tm in range(TILES_M):
            for tn in range(TILES_N):
                a, b = x[tm * R.TM:(tm + 1) * R.TM], wp[tn * R.BN:(tn + 1) * R.BN]
                words = [((a[:, :j] @ b[:, :j].T) & 0xFFFFFFFF).astype(np.uint32) for j in range(spacing, K + 1, spacing)]
                rec.append(R.tile_receipts(SEED, words, tm * TILES_N + tn))
        np.stack(rec).astype(np.uint32).tofile(d / f"R_{arm}.bin")
    (d / "meta.json").write_text(json.dumps({"M": M, "N": N, "K": K, "seed": SEED, "arms": ARMS}))


def flip(path: Path, at: int, dtype) -> None:
    a = np.fromfile(path, dtype)
    a[at] ^= 1
    a.tofile(path)


def failed(r: dict) -> list[str]:
    return [arm for arm, v in r["arms"].items() if not v["ok"]]


def test_a_right_gate_passes(tmp_path):
    write_gate(tmp_path)
    r = R.check(tmp_path)
    assert r["ok"] and failed(r) == [], r
    assert list(r["arms"]) == ARMS and "receipt_words" not in r["arms"]["plain"]
    for arm in SPACING:
        v = r["arms"][arm]
        assert v["receipt_words"] == TILES_M * TILES_N * R.NCT * 8
        assert v["receipt_words_wrong"] == 0 and v["control_rejected"]


@pytest.mark.parametrize("arm", ARMS)
def test_a_wrong_served_word_fails_its_arm(tmp_path, arm):
    write_gate(tmp_path)
    flip(tmp_path / f"Y_{arm}.bin", M * N - 1, np.int32)
    r = R.check(tmp_path)
    assert not r["ok"] and failed(r) == [arm] and not r["arms"][arm]["y_equal"]


@pytest.mark.parametrize("arm", list(SPACING))
def test_a_wrong_receipt_word_fails_its_arm(tmp_path, arm):
    write_gate(tmp_path)
    flip(tmp_path / f"R_{arm}.bin", R.NCT * 8 + 5, np.uint32)   # a word of the second tile
    r = R.check(tmp_path)
    assert not r["ok"] and failed(r) == [arm]
    assert r["arms"][arm]["receipt_words_wrong"] == 1 and r["arms"][arm]["y_equal"]


def test_an_arm_is_judged_at_its_own_spacing(tmp_path):
    """e16 holding e32's receipts fails: the judge absorbs the partial sums at each arm's own commit boundaries."""
    write_gate(tmp_path)
    (tmp_path / "R_e16.bin").write_bytes((tmp_path / "R_e32.bin").read_bytes())
    r = R.check(tmp_path)
    assert failed(r) == ["e16"] and r["arms"]["e16"]["receipt_words_wrong"] > R.NCT


def test_the_command_exits_1_on_a_difference(tmp_path, monkeypatch, capsys):
    write_gate(tmp_path)
    monkeypatch.setattr(sys, "argv", ["egemm80_ref.py", str(tmp_path)])
    assert R.main() == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    flip(tmp_path / "Y_e64.bin", 0, np.int32)
    assert R.main() == 1
    line = json.loads(capsys.readouterr().out)
    assert line["ok"] is False and line["dir"] == str(tmp_path)
    monkeypatch.setattr(sys, "argv", ["egemm80_ref.py"])
    assert R.main() == 2
