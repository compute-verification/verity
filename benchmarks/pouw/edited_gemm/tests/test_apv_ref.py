"""apv_ref.py, the CPU judge of apv.cu's gate (P·V with the probabilities as the 7-bit operand, key 0 its own block):
on a gate directory of two tiles written from the definition in its docstring (the first query tile, where key 0 is
masked out of the first block, and a later one) it passes, and it fails by the one flag whose dumped bytes differ, on
a wrong scale, when a warp row's receipts leave a word out or are another tile's, or when the device's own verdicts
are negative."""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

import apv_ref as A
import numpy as np
import pytest
import test_cgemm_ref as G
import test_cpv_ref as PV
from cgemm_ref import DITHER_TAG

S, H, KEY = PV.S, 2, PV.KEY
#: (the tile's index in the call, head, query tile, the keys' head)
FIRST, LATER = (4, 0, 0, 1), (9, 1, 2, 0)
TILES = {"first": FIRST, "later": LATER}
#: each flag a dump answers for, with the dump's file and the type its words are flipped in
DUMPS = {"Pt": ("Pt", np.int8), "X": ("X", np.int8), "Um": ("Um", np.int8), "PA": ("PA", np.int8),
         "DG": ("DG", np.int8), "Sq": ("Sq", np.int32), "X0": ("X0", np.int32), "O": ("O", np.int32),
         "F": ("F", np.uint32), "R": ("R", np.uint32)}


def words_of(tile: tuple, P: np.ndarray, Vt: np.ndarray) -> dict:
    """What apv.cu holds on one tile if it is right: P is the probabilities, queries x keys, and Vt channels x keys,
    over keys 0 .. 128 (qt + 1) - 1. Query q's unit is keys 1 .. q, ending in its own 32-key block b_l = q // 32."""
    t, _, qt, kvh = tile
    nk = P.shape[1]
    q, keys, rho = 128 * qt + np.arange(128), np.arange(nk), np.arange(32)
    M = ((keys[None, :] >= 1) & (keys[None, :] <= q[:, None])).astype(np.int64)
    Md = M[np.arange(128)[:, None], 32 * (q // 32)[:, None] + rho[None, :]]    # the mask on each query's own block
    X = Vt + A.patch_cols(KEY, DITHER_TAG, 128 * kvh, 128, S, 0, nk)
    Xz = X * (keys >= 1)                                                       # key 0 is in no unit
    U = A.u_rows(KEY, 128 * qt, 128)
    Pt = M * (P - 64 + np.tile(U, (1, nk // 32)))
    rng = np.random.default_rng(t)
    sp, sz, sv = (rng.uniform(0.01, 2, 128).astype(np.float32) for _ in range(3))
    DG, Sq = np.zeros((4, 128, 128), np.int64), np.zeros((4, 128), np.int64)
    served, F, words = np.zeros((128, 128), np.int64), np.zeros((128, 128), np.float32), {}
    for w in range(4):
        bl = 4 * qt + w
        rows, ku = slice(32 * w, 32 * w + 32), 32 * (bl + 1)
        Y = sum((Xz[:, 32 * j:32 * j + 32] for j in range(bl)), np.zeros((128, 32), np.int64))    # channels x residues
        Y0 = (Y + 128) % 256 - 128
        Y1 = ((Y - Y0) // 256 + 128) % 256 - 128
        Y2 = ((Y - Y0) // 256 - Y1) // 256
        assert (Y0 + 256 * Y1 + 65536 * Y2 == Y).all()
        DG[w] = np.hstack([Y0, Y1, X[:, ku - 32:ku], Y2])
        Sq[w] = Xz[:, :ku - 32].sum(1)
        T = [Pt[rows, :j] @ X[:, :j].T for j in range(32, ku + 1, 32)]
        served[rows] = T[-1] - U[rows] @ Y.T - (U * Md)[rows] @ X[:, ku - 32:ku].T    # U's part goes: one digit
        sums = M[rows] @ X.T
        body = (served[rows] + 64 * sums).astype(np.float32) * sp[rows, None]
        F[rows] = (body + sz[rows, None] * X[:, 0].astype(np.float32)[None, :]) * sv[None, :]
        words[w] = [*T, served[rows]]
    assert (served == (M * (P - 64)) @ X.T).all()    # the masked product of P - 64 and X, and with 64 S of P itself
    return {"P": P.astype(np.int8), "V": Vt.astype(np.int8), "Pt": Pt.astype(np.int8), "X": X.astype(np.int8),
            "Um": (U - 64).astype(np.int8), "PA": np.hstack([-U, -U, -U * Md, 64 * Md]).astype(np.int8),
            "DG": DG.astype(np.int8), "Sq": Sq.astype(np.int32), "X0": X[:, 0].astype(np.int32), "sp": sp, "sz": sz,
            "sv": sv, "O": served.astype(np.int32), "F": F, "words": words}


@pytest.fixture(scope="module")
def made(tmp_path_factory) -> tuple[Path, dict]:
    """The gate directory, and each tile's words by warp row."""
    d, words = tmp_path_factory.mktemp("apv"), {}
    for name, tile in TILES.items():
        rng = np.random.default_rng(41 + tile[0])
        nk = 128 * (tile[2] + 1)
        files = words_of(tile, rng.integers(0, 128, (128, nk)), rng.integers(-100, 101, (128, nk)))
        words[name] = files.pop("words")
        for file, a in (files | {"R": PV.receipts(tile[0], words[name])}).items():
            a.tofile(d / f"{file}_{tile[0]}.bin")
    (d / "meta.json").write_text(json.dumps(
        {"s": S, "H": H, "seed": PV.SEED, "key": [f"{w:016x}" for w in G.KEY], "rep": 1, "ntiles": len(TILES),
         "tiles": [list(tile) for tile in TILES.values()], "device_ok": True, "control_rejected": True,
         "device": "made up"}))
    return d, words


@pytest.fixture
def gate(made, tmp_path) -> Path:
    return shutil.copytree(made[0], tmp_path / "gate")


def test_u_is_laid_out_as_the_docstring_says():
    """Every entry of U from SHA-512(key, tag 14, q // 2): a byte's low six bits plus one, negative where bit 6 is."""
    row0 = 384
    u = A.u_rows(KEY, row0, 128)
    assert u.shape == (128, 32) and abs(u).min() == 1 and abs(u).max() == 64
    for i in range(128):
        q = row0 + i
        digest = hashlib.sha512(KEY + struct.pack(">Q", 14 << 56 | q // 2)).digest()
        for rho in range(32):
            byte = digest[(q % 2) * 32 + rho]
            assert u[i, rho] == ((byte & 63) + 1) * (-1 if byte & 64 else 1)


def test_a_right_gate_passes(made):
    r = A.check(made[0])
    assert r["ok"] and PV.wrong(r) == {}, r
    for tile in TILES.values():
        res = r["tiles"][str(tile[0])]
        assert set(DUMPS) <= set(res) and res["O_is_served"] and res["DG_s8"]
        assert (res["h"], res["qt"], res["kvh"]) == tile[1:]


def test_the_made_up_gate_has_a_second_digit(made):
    """The later tile's sums before a query's block pass one signed byte, so DG's Y1 is not all zero there."""
    DG = np.fromfile(made[0] / f"DG_{LATER[0]}.bin", np.int8).reshape(4, 128, 128)
    assert (DG[:, :, 32:64] != 0).mean() > 0.2 and not DG[:, :, 96:].any()


@pytest.mark.parametrize("name", list(TILES))
@pytest.mark.parametrize("flag", list(DUMPS))
def test_one_wrong_dumped_word_fails_by_its_flag(gate, name, flag):
    file, dtype = DUMPS[flag]
    G.flip(gate / f"{file}_{TILES[name][0]}.bin", 101, dtype)
    r = A.check(gate)
    assert not r["ok"] and PV.wrong(r) == {TILES[name][0]: [flag]}


@pytest.mark.parametrize("scale", ["sp", "sz", "sv"])
def test_a_wrong_scale_fails_the_float_output(gate, scale):
    path = gate / f"{scale}_{LATER[0]}.bin"
    a = np.fromfile(path, np.float32)
    a[77] *= 2
    a.tofile(path)
    r = A.check(gate)
    assert not r["ok"] and PV.wrong(r) == {LATER[0]: ["F"]}


@pytest.mark.parametrize("w", range(4))
def test_every_warp_rows_receipts_count(gate, w):
    G.flip(gate / f"R_{FIRST[0]}.bin", 8 * (128 * w + 5) + 2, np.uint32)
    r = A.check(gate)
    assert not r["ok"] and PV.wrong(r) == {FIRST[0]: ["R"]}


@pytest.mark.parametrize(("name", "w", "word"), [("first", 0, 1), ("first", 3, 0), ("later", 1, 10), ("later", 2, 4)])
def test_receipts_without_one_word_fail(made, gate, name, w, word):
    """The row's O (after its b_l + 1 words T) or a T left out."""
    words = dict(made[1][name])
    words[w] = words[w][:word] + words[w][word + 1:]
    PV.receipts(TILES[name][0], words).tofile(gate / f"R_{TILES[name][0]}.bin")
    r = A.check(gate)
    assert not r["ok"] and PV.wrong(r) == {TILES[name][0]: ["R"]}


def test_another_tiles_receipts_fail(made, gate):
    PV.receipts(LATER[0], made[1]["later"], tile=LATER[0] + 1).tofile(gate / f"R_{LATER[0]}.bin")
    r = A.check(gate)
    assert not r["ok"] and PV.wrong(r) == {LATER[0]: ["R"]}


@pytest.mark.parametrize("verdict", ["device_ok", "control_rejected"])
def test_the_devices_own_verdicts_count(gate, verdict):
    meta = json.loads((gate / "meta.json").read_text())
    (gate / "meta.json").write_text(json.dumps(meta | {verdict: False}))
    r = A.check(gate)
    assert not r["ok"] and PV.wrong(r) == {}


def test_main_is_exit_1_when_any_gate_fails(made, gate, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["apv_ref.py", str(made[0])])
    assert A.main() == 0 and json.loads(capsys.readouterr().out)["ok"]
    G.flip(gate / f"X0_{FIRST[0]}.bin", 0, np.int32)
    monkeypatch.setattr(sys, "argv", ["apv_ref.py", str(gate), str(made[0])])    # a right gate after it clears nothing
    assert A.main() == 1
    assert [json.loads(line)["ok"] for line in capsys.readouterr().out.splitlines()] == [False, True]
