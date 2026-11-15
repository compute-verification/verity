"""cpv_ref.py, the CPU judge of cpv.cu's gate (P·V under the causal mask): on a gate directory of three tiles written
from the definition in its docstring (a full unit, and the diagonal unit at an even and at an odd query tile) it
passes, and it fails by the one flag whose dumped bytes differ, when a warp row's receipts leave a word out or are
another tile's, or when the device's own verdicts are negative."""
from __future__ import annotations

import functools
import hashlib
import itertools
import json
import shutil
import struct
import sys
from pathlib import Path

import cpv_ref as C
import numpy as np
import pytest
import test_cgemm_ref as G
from cgemm_ref import C_TAG, DITHER_TAG, NCT, patch_bits, tile_receipts

S, NB, H, SEED = 512, 2, 2, 0x5EED7
KEY = b"".join(struct.pack(">Q", w) for w in G.KEY)
#: (the tile's index in the call, head, query tile, key block, the keys' head, whether the block holds the queries)
FULL, EVEN, ODD = (3, 1, 3, 0, 1, 0), (6, 1, 2, 1, 0, 1), (11, 0, 3, 1, 1, 1)
TILES = {"full": FULL, "even": EVEN, "odd": ODD}
#: each flag a dump answers for, with the dump's file and the type its words are flipped in
DUMPS = {"Pt": ("Pt", np.int8), "X": ("X", np.int8), "PA": ("PA", np.int8), "DG": ("DG", np.int8),
         "Sb": ("Sb", np.int32), "O": ("O", np.int32), "F": ("F", np.uint32), "R": ("R", np.uint32)}


def words_of(tile: tuple, P: np.ndarray, Vt: np.ndarray) -> dict:
    """What cpv.cu holds on one tile if it is right: P is queries x keys and Vt channels x keys, both in the key
    block's own columns. A query's unit ends at its own 32-key block b_l, whose keys after the query are masked."""
    t, h, qt, B, kvh, diag = tile
    nk = P.shape[1]
    q = 128 * qt + np.arange(128) - 256 * B                       # each query, counted from the block's first key
    M = (np.arange(nk)[None, :] <= q[:, None]).astype(np.int64)
    Md = (np.arange(32)[None, :] <= (q % 32)[:, None]).astype(np.int64) if diag else np.ones((128, 32), np.int64)
    c = C.patch_cols(KEY, C_TAG, h * S + 128 * qt, 128, S, 256 * B, nk)
    X = Vt + C.patch_cols(KEY, DITHER_TAG, 128 * kvh, 128, S, 256 * B, 256)
    U = C.u_pa(KEY, (h * NB + B) * S + 128 * qt, 128)
    Pt = M * (P + c + np.tile(U, (1, nk // 32)))
    zero = np.zeros((128, 32), np.int64)
    DG, Sb, A = np.zeros((8, 128, 128), np.int64), np.zeros((9, 128), np.int64), {}
    for bl in range(8):
        Y = sum((X[:, 32 * j:32 * j + 32] for j in range(bl)), zero)   # channels x residues, the blocks before b_l
        z = C.z_unit(KEY, kvh * NB + B, bl).T
        A[bl] = ((Y & 127) - 128 * z, (Y >> 7) + z) if bl >= 2 else (Y, zero)   # Y = A0 + 128 A1 from b_l = 2 on
        DG[bl] = np.hstack([A[bl][0], A[bl][1], X[:, 32 * bl:32 * bl + 32], zero])
        Sb[bl] = X[:, :32 * bl].sum(1)
    Sb[8] = X.sum(1)
    rng = np.random.default_rng(t)
    sp, sv = rng.uniform(0.01, 2, 128).astype(np.float32), rng.uniform(0.01, 2, 128).astype(np.float32)
    served, F, words = np.zeros((128, 128), np.int64), np.zeros((128, 128), np.float32), {}
    for w in range(4):
        bl = 4 * (qt % 2) + w if diag else 7
        rows, ku = slice(32 * w, 32 * w + 32), 32 * (bl + 1)
        A0, A1 = A[bl]
        T = [Pt[rows, :j] @ X[:, :j].T for j in range(32, ku + 1, 32)]
        Op = T[-1] - U[rows] @ A0.T - (U * Md)[rows] @ X[:, ku - 32:ku].T    # U's part goes, but for 128 D_1
        D1 = U[rows] @ A1.T
        served[rows] = Op - 128 * D1
        sums = M[rows, :ku] @ X[:, :ku].T
        F[rows] = (2 * (served[rows] + 64 * sums) - sums).astype(np.float32) * (sp[rows, None] * sv[None, :])
        words[w] = [*T, Op] + ([D1] if bl >= 2 else [])
    assert (served == (M * (P + c)) @ X[:, :nk].T).all()    # the masked product of P + c and X, U's part gone
    return {"P": P.astype(np.int8), "V": Vt.astype(np.int8), "Pt": Pt.astype(np.int8), "X": X.astype(np.int8),
            "PA": np.hstack([-U, U, -U * Md, 127 * Md]).astype(np.int8), "DG": DG.astype(np.int8),
            "Sb": Sb.astype(np.int32), "sp": sp, "sv": sv, "O": served.astype(np.int32), "F": F, "words": words}


def receipts(t: int, words: dict, tile: int | None = None) -> np.ndarray:
    """A tile's receipts: warp row w's 128 threads absorb that row's words, which fill its 32 queries alone."""
    R = np.zeros((NCT, 8), np.uint32)
    for w, row in words.items():
        full = np.zeros((len(row), 128, 128), np.uint32)
        full[:, 32 * w:32 * w + 32] = [G.u32(word) for word in row]
        R[128 * w:128 * w + 128] = tile_receipts(SEED, list(full), t if tile is None else tile)[128 * w:128 * w + 128]
    return R


@pytest.fixture(scope="module")
def made(tmp_path_factory) -> tuple[Path, dict]:
    """The gate directory, and each tile's words by warp row."""
    d, words = tmp_path_factory.mktemp("cpv"), {}
    for name, tile in TILES.items():
        rng = np.random.default_rng(31 + tile[0])
        nk = 256 if name != "even" else 128     # an even diagonal tile's queries see the block's first 128 keys
        files = words_of(tile, rng.integers(-63, 63, (128, nk)), rng.integers(-100, 101, (128, 256)))
        words[name] = files.pop("words")
        for file, a in (files | {"R": receipts(tile[0], words[name])}).items():
            a.tofile(d / f"{file}_{tile[0]}.bin")
    (d / "meta.json").write_text(json.dumps(
        {"s": S, "nB": NB, "H": H, "seed": SEED, "key": [f"{w:016x}" for w in G.KEY], "rep": 1, "ntiles": len(TILES),
         "tiles": [list(tile) for tile in TILES.values()], "device_ok": True, "control_rejected": True,
         "device": "made up"}))
    return d, words


@pytest.fixture
def gate(made, tmp_path) -> Path:
    return shutil.copytree(made[0], tmp_path / "gate")


def wrong(r: dict) -> dict[int, list[str]]:
    """Each tile's flags that are false, the tiles with none left out."""
    out = {int(t): [name for name, v in res.items() if v is False] for t, res in r["tiles"].items()}
    return {t: flags for t, flags in out.items() if flags}


def test_the_keys_bits_are_laid_out_as_the_docstring_says():
    """c and b, U and z, every entry from SHA-512(key, tag, block)."""
    @functools.cache
    def bits(tag: int, j: int) -> list[int]:
        digest = hashlib.sha512(KEY + struct.pack(">Q", tag << 56 | j)).digest()
        return [digest[i // 8] >> (i % 8) & 1 for i in range(512)]

    row0, col0, unit, bl = S + 384, 256, 3, 5
    c, u, z = C.patch_cols(KEY, C_TAG, row0, 128, S, col0, 256), C.u_pa(KEY, row0, 128), C.z_unit(KEY, unit, bl)
    assert (c.shape, u.shape, z.shape) == ((128, 256), (128, 32), (32, 128))
    assert (c == patch_bits(KEY, C_TAG, row0, 128, S)[:, col0:col0 + 256]).all()
    for i, col in itertools.product(range(128), range(256)):
        row = row0 + i
        assert c[i, col] == bits(C_TAG, (row // 4) * (S // 128) + (col0 + col) // 128)[(row % 4) * 128 + col % 128]
    for i, rho in itertools.product(range(128), range(32)):
        row = row0 + i
        beta = bits(12, row // 2)[8 * ((row % 2) * 32 + rho):][:7]
        assert u[i, rho] == -64 + sum(p * w for p, w in zip(beta, (1, 2, 4, 8, 16, 32, 65), strict=True))
    for rho, ch in itertools.product(range(32), range(128)):
        assert z[rho, ch] == bits(13, (8 * unit + bl) * 8 + ch // 16)[(ch % 16) * 32 + rho]


def test_a_right_gate_passes(made):
    r = C.check(made[0])
    assert r["ok"] and wrong(r) == {}, r
    tiles = {name: r["tiles"][str(tile[0])] for name, tile in TILES.items()}
    assert [t["kind"] for t in tiles.values()] == ["full", "diagonal", "diagonal"]
    assert all(set(DUMPS) <= set(t) and t["O_is_served"] and t["DG_s8"] for t in tiles.values())
    # D_1 is a word where the unit has two blocks or more before its own: every warp row but an even tile's first two
    assert [len(t["D1_nonzero"]) for t in tiles.values()] == [4, 2, 4]
    assert all(share > 0.5 for t in tiles.values() for share in t["D1_nonzero"])


@pytest.mark.parametrize("name", list(TILES))
@pytest.mark.parametrize("flag", list(DUMPS))
def test_one_wrong_dumped_word_fails_by_its_flag(gate, name, flag):
    file, dtype = DUMPS[flag]
    G.flip(gate / f"{file}_{TILES[name][0]}.bin", 8 * 100 + 1, dtype)
    r = C.check(gate)
    assert not r["ok"] and wrong(r) == {TILES[name][0]: [flag]}


@pytest.mark.parametrize("scale", ["sp", "sv"])
def test_a_wrong_scale_fails_the_float_output(gate, scale):
    G.flip(gate / f"{scale}_{ODD[0]}.bin", 77, np.uint32)
    r = C.check(gate)
    assert not r["ok"] and wrong(r) == {ODD[0]: ["F"]}


@pytest.mark.parametrize("w", range(4))
def test_every_warp_rows_receipts_count(gate, w):
    G.flip(gate / f"R_{EVEN[0]}.bin", 8 * (128 * w + 5) + 2, np.uint32)
    r = C.check(gate)
    assert not r["ok"] and wrong(r) == {EVEN[0]: ["R"]}


@pytest.mark.parametrize(("name", "w", "word"), [("full", 2, 8), ("full", 0, 9), ("even", 3, 4), ("even", 0, 0),
                                                  ("odd", 1, 7), ("odd", 3, 0)])
def test_receipts_without_one_word_fail(made, gate, name, w, word):
    """The row's O' (after its b_l + 1 words T), its D_1 (the last, where it has one) or a T left out."""
    words = dict(made[1][name])
    words[w] = words[w][:word] + words[w][word + 1:]
    receipts(TILES[name][0], words).tofile(gate / f"R_{TILES[name][0]}.bin")
    r = C.check(gate)
    assert not r["ok"] and wrong(r) == {TILES[name][0]: ["R"]}


def test_another_tiles_receipts_fail(made, gate):
    receipts(ODD[0], made[1]["odd"], tile=ODD[0] + 1).tofile(gate / f"R_{ODD[0]}.bin")
    r = C.check(gate)
    assert not r["ok"] and wrong(r) == {ODD[0]: ["R"]}


@pytest.mark.parametrize("verdict", ["device_ok", "control_rejected"])
def test_the_devices_own_verdicts_count(gate, verdict):
    meta = json.loads((gate / "meta.json").read_text())
    (gate / "meta.json").write_text(json.dumps(meta | {verdict: False}))
    r = C.check(gate)
    assert not r["ok"] and wrong(r) == {}


def test_main_is_exit_1_when_any_gate_fails(made, gate, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["cpv_ref.py", str(made[0])])
    assert C.main() == 0 and json.loads(capsys.readouterr().out)["ok"]
    G.flip(gate / f"Sb_{FULL[0]}.bin", 0, np.int32)
    monkeypatch.setattr(sys, "argv", ["cpv_ref.py", str(gate), str(made[0])])    # a right gate after it clears nothing
    assert C.main() == 1
    assert [json.loads(line)["ok"] for line in capsys.readouterr().out.splitlines()] == [False, True]
