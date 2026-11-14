"""cgemm_ref.py, the CPU judge of cgemm.cu's gate: on a gate directory written from the definition in its docstring
(the full kernel and the cohalf one, each in Dither-C's numerics and in H1's) it passes, and it fails by the one flag
whose dumped bytes differ, when fx1 holds duo's receipts or drops one multiply, or when the device's own verdicts are
negative. A judge that passes everything would let a wrong kernel through the GPU run's gate."""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

import cgemm_ref as C
import numpy as np
import pytest

T, SEED, INV_S = C.T, 0xC6E3, 31.75
KEY = [0x0123456789ABCDEF, 0xFEDCBA9876543210, 0x0F1E2D3C4B5A6978, 0x8796A5B4C3D2E1F0]
TM, TN, TILES_N = 1, 2, 4   # row tile 1 and column tile 2 of a 2 x 4 grid: the call's tile 6
#: b's and c's SHA-512 tags (egemm.cu's DITHER_TAG, cgemm.cu's C_TAG). Written out, so the bits a gate directory is made
#: with are not under the tags the judge under test binds
DITHER_TAG, C_TAG = 7, 8
SFX = f"_{TM}_{TN}.bin"
FULL, HALF = ("co32c", "h1"), ("cohalf", "cohalf-h1")
#: each flag a dump answers for, with the dump's file and type
FULL_DUMPS = {"X": ("X", np.int8), "PA": ("PA", np.int8), "Wp": ("Wp", np.int8), "PB": ("PB", np.int8),
              "O": ("Y_c_l7qs", np.int32), "Op": ("Y_c_l7qs_o", np.int32), "R": ("R_c_l7qs", np.uint32),
              "O_fx1": ("Y_c_fx1", np.int32), "R_fx1": ("R_c_fx1", np.uint32)}
HALF_DUMPS = {"X": ("X", np.int8), "PA": ("PA", np.int8), "Wp": ("Wp", np.int8), "PB": ("PB", np.int8),
              "O": ("Y_h", np.int32), "R": ("R_h", np.uint32)}
#: floored_gemm/fgemm.cu's key_mul(1, 32 j), j = 0 .. 31: mix32(0xA0000000 + 2 (16 + 32 j)) | 1. Written out, so the
#: keys a gate directory is made with are not the ones the judge under test binds
FX1_KEY = (0x647DC62B, 0xDDEEABEF, 0x54C8DC17, 0xB07020F7, 0x027AEF5F, 0x859151DF, 0xAE776B81, 0xB1A33B77,
           0x07D80027, 0x834F6FD3, 0x7BA58CE1, 0x589DA773, 0x40BC2F31, 0xEE4B213F, 0x20C581D7, 0xC5F2BCCF,
           0xC5A846F3, 0x44DE23DD, 0x2FFECFD3, 0x786228E9, 0x53AD659F, 0x63DA4E8B, 0xCD90C4EB, 0x2231DD93,
           0x7852EF93, 0xBC7C7457, 0xE6C89781, 0x5A8E8A7F, 0x5C20233B, 0xD48D2535, 0xB4FA689B, 0x3DA91697)


def u32(a: np.ndarray) -> np.ndarray:
    return (a & 0xFFFFFFFF).astype(np.uint32)


def fx1_keys(drop: tuple[int, int] | None = None) -> np.ndarray:
    """fx1's multiplier of each word of a tile: word e of tile position t = 4 mi + ni sits at row 32 wm + 16 mi + g +
    8 (e >> 1), column 32 wn + 8 ni + 2 q + (e & 1), and is multiplied by key_mul(1, (4 t + e) 32), `FX1_KEY`'s word
    4 t + e; drop leaves one multiply out."""
    K = np.zeros((T, T), np.uint64)
    for mi in range(2):
        for ni in range(4):
            for e in range(4):
                rows = [32 * wm + 16 * mi + g + 8 * (e >> 1) for wm in range(4) for g in range(8)]
                cols = [32 * wn + 8 * ni + 2 * q + (e & 1) for wn in range(4) for q in range(4)]
                K[np.ix_(rows, cols)] = 1 if drop == (4 * mi + ni, e) else FX1_KEY[4 * (4 * mi + ni) + e]
    return K


def write(d: Path, files: dict[str, np.ndarray], meta: dict, k: int) -> None:
    for name, a in files.items():
        a.tofile(d / (name + SFX))
    (d / "meta.json").write_text(json.dumps(
        {"m": 2 * T, "n": TILES_N * T, "k": k, "seed": SEED, "inv_s": INV_S, "key": [f"{w:016x}" for w in KEY],
         "tiles": [[TM, TN]], "device_ok": True, "control_rejected": True, "device": "made up"} | meta))


def write_full(d: Path, numerics: str, drop: tuple[int, int] | None = None) -> None:
    """One 128 x 128 tile at k = 256 (two base-128 digits) as cgemm.cu would dump it if it were right. H1 scales each
    token by its largest magnitude over 126 and has the narrow U; Dither-C has one scale and the 7-bit U."""
    h1, k = numerics == "h1", 256
    rng = np.random.default_rng(17)
    key = b"".join(struct.pack(">Q", w) for w in KEY)
    xb = (rng.normal(0, 2.5, (T, k)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    x = (xb.astype(np.uint32) << 16).view(np.float32)
    if h1:
        V = np.clip(np.floor(x / (np.abs(x).max(axis=1) / np.float32(126)).astype(np.float32)[:, None]), -126, 126)
    else:
        V = np.clip(np.floor(x * np.float32(INV_S)), -127, 126)
    X = V.astype(np.int64) + C.patch_bits(key, DITHER_TAG, TM * T, T, k)
    W = rng.integers(-110, 111, (T, k)) if h1 else rng.integers(-63, 63, (T, k))
    Wc = W + C.patch_bits(key, C_TAG, TN * T, T, k)
    U = (C.u_rows16 if h1 else C.u_rows)(key, TN * T, T)
    Wp = Wc + np.tile(U, (1, k // 32))
    Y = sum(X[:, 32 * t:32 * t + 32] for t in range(k // 32))    # a row's 32 residue sums
    z = C.z_rows(key, TM * T, T, 2)[0].T
    A0, A1 = (Y & 127) - 128 * z, (Y >> 7) + z                   # Y = A0 + 128 A1, the coin z picking A0's sign
    D1 = A1 @ U.T
    Op = X @ Wp.T - A0 @ U.T
    words = [u32(X[:, :j] @ Wp[:, :j].T) for j in range(32, k + 1, 32)] + [u32(Op), u32(D1)]
    zero = np.zeros((T, 32), np.int64)
    served = (Op - 128 * D1).astype(np.int32)
    write(d, {"x": xb, "W": W.astype(np.int8), "X": X.astype(np.int8),
              "PA": np.hstack([A0, A1, zero, zero]).astype(np.int8), "Wp": Wp.astype(np.int8),
              "PB": np.hstack([-U, U, U, zero]).astype(np.int8), "Y_c_l7qs": served, "Y_c_l7qs_o": Op.astype(np.int32),
              "Y_c_fx1": served, "R_c_l7qs": C.tile_receipts(SEED, words, TM * TILES_N + TN).astype(np.uint32),
              "R_c_fx1": C.tile_receipts(SEED, [u32(w.astype(np.uint64) * fx1_keys(drop)) for w in words],
                                         TM * TILES_N + TN).astype(np.uint32)},
          {"numerics": numerics}, k)
    assert (served == X @ Wc.T).all()    # what the kernel serves is the product of the two dithered operands


def write_half(d: Path, numerics: str) -> None:
    """One tile of the cohalf kernel (QK^T at k = 128, the two 64-halves of a row summed). H1 scales each row to
    [-112, 111] with no clamp, has the narrow U and splits Y in two digits; Dither-C has the 7-bit scale and U."""
    h1, k = numerics == "cohalf-h1", 128
    rng = np.random.default_rng(19)
    key = b"".join(struct.pack(">Q", w) for w in KEY)
    xb = (rng.normal(0, 1.5, (T, k)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    x = (xb.astype(np.uint32) << 16).view(np.float32)
    if h1:
        s = np.maximum(x.max(axis=1) / np.float32(111), (-x).max(axis=1) / np.float32(112)) * np.float32(1 + 2.0 ** -20)
        V = np.floor(x / s.astype(np.float32)[:, None])
    else:
        V = np.clip(np.floor(x * np.float32(INV_S)), -64, 62)
    X = V.astype(np.int64) + C.patch_bits(key, DITHER_TAG, TM * T, T, k)
    W = rng.integers(-110, 111, (T, k)) if h1 else rng.integers(-63, 63, (T, k))
    Wc = W + C.patch_bits(key, C_TAG, TN * T, T, k)
    U = (C.u_rows16_half if h1 else C.u_rows_half)(key, TN * T, T)
    Wp = Wc + np.tile(U, (1, 2))
    Y = X[:, :64] + X[:, 64:]
    zero = np.zeros((T, 32), np.int64)
    if h1:
        Y0 = ((Y + 64) & 127) - 64
        PA, PB = np.hstack([Y0, (Y - Y0) >> 7]), np.hstack([-U, -U])
    else:
        PA, PB = np.hstack([Y, zero, zero]), np.hstack([-U, zero, zero])
    served = X @ Wc.T
    words = [u32(X[:, :j] @ Wp[:, :j].T) for j in range(32, k + 1, 32)] + [u32(served)]
    write(d, {"x": xb, "W": W.astype(np.int8), "X": X.astype(np.int8), "PA": PA.astype(np.int8),
              "Wp": Wp.astype(np.int8), "PB": PB.astype(np.int8), "Y_h": served.astype(np.int32),
              "R_h": C.tile_receipts(SEED, words, TM * TILES_N + TN).astype(np.uint32)},
          {"numerics": "h1" if h1 else "co32c", "kinds": {"h": "cohalf"}}, k)


def test_the_keys_bits_are_laid_out_as_the_docstring_says():
    """b and c, U and z entry by entry from SHA-512(key, tag, block), at rows that are not a block's first."""
    key = b"".join(struct.pack(">Q", w) for w in KEY)

    def bits(tag: int, j: int) -> list[int]:
        digest = hashlib.sha512(key + struct.pack(">Q", tag << 56 | j)).digest()
        return [digest[i // 8] >> (i % 8) & 1 for i in range(512)]

    row0, k = 2 * T, 256
    b, c = C.patch_bits(key, DITHER_TAG, row0, T, k), C.patch_bits(key, C_TAG, row0, T, k)
    u, u16, z = C.u_rows(key, row0, T), C.u_rows16(key, row0, T), C.z_rows(key, row0, T, 3)
    uh, uh16 = C.u_rows_half(key, row0, T), C.u_rows16_half(key, row0, T)
    assert (b.shape, u.shape, z.shape, uh.shape) == ((T, k), (T, 32), (2, 32, T), (T, 64))
    for i, col in ((0, 0), (5, 131), (38, 255), (127, 64)):
        row, r = row0 + i, col % 32
        patch = (row // 4) * (k // 128) + col // 128
        assert b[i, col] == bits(DITHER_TAG, patch)[(row % 4) * 128 + col % 128]
        assert c[i, col] == bits(C_TAG, patch)[(row % 4) * 128 + col % 128]
        beta = bits(10, row // 2)[8 * ((row % 2) * 32 + r):][:7]
        assert u[i, r] == -64 + sum(p * w for p, w in zip(beta, (1, 2, 4, 8, 16, 32, 65), strict=True))
        assert u16[i, r] == -16 + sum(p * w for p, w in zip(beta, (1, 2, 4, 8, 17), strict=False))
        assert [z[d, r, i] for d in range(2)] == [bits(9, row // 4)[8 * 16 * (row % 4) + 32 * d + r] for d in range(2)]
        half = bits(11, row)[8 * (col % 64):][:7]
        assert uh[i, col % 64] == -64 + sum(p * w for p, w in zip(half, (1, 2, 4, 8, 16, 32, 65), strict=True))
        assert uh16[i, col % 64] == -16 + sum(p * w for p, w in zip(half, (1, 2, 4, 8, 17), strict=False))


@pytest.fixture(scope="module")
def right(tmp_path_factory) -> dict[str, Path]:
    gates = {name: tmp_path_factory.mktemp(name) for name in FULL + HALF}
    for name, d in gates.items():
        (write_full if name in FULL else write_half)(d, name)
    return gates


def flip(path: Path, at: int, dtype) -> None:
    a = np.fromfile(path, dtype)
    a[at] ^= 1
    a.tofile(path)


def wrong(r: dict) -> list[str]:
    """The flags of the tile that are false."""
    return [name for name, v in r["tiles"][f"{TM},{TN}"].items() if v is False]


@pytest.mark.parametrize("name", FULL + HALF)
def test_a_right_gate_passes(right, name):
    r = C.check(right[name])
    t = r["tiles"][f"{TM},{TN}"]
    assert r["ok"] and wrong(r) == [], r
    assert r["numerics"] == ("h1" if name.endswith("h1") else "co32c")
    if name in FULL:
        assert r["digits"] == 2 and set(FULL_DUMPS) <= set(t) and t["O_is_served"] and t["D1_nonzero"] > 0.5
        assert t["R_fx1_rejects_duo"] and t["R_fx1_rejects_dropped_imad"]
    else:
        assert r["design"] == "cohalf" and set(HALF_DUMPS) <= set(t) and t["O_is_strip"]


@pytest.mark.parametrize(("name", "flag"), [(name, flag) for name in FULL for flag in FULL_DUMPS]
                         + [(name, flag) for name in HALF for flag in HALF_DUMPS])
def test_one_wrong_dumped_word_fails_by_its_flag(right, tmp_path, name, flag):
    d = shutil.copytree(right[name], tmp_path / "gate")
    file, dtype = (FULL_DUMPS if name in FULL else HALF_DUMPS)[flag]
    flip(d / (file + SFX), 8 * 300 + 1, dtype)
    r = C.check(d)
    assert not r["ok"] and wrong(r) == [flag]


@pytest.mark.parametrize("name", FULL)
def test_fx1_holding_duos_receipts_fails(right, tmp_path, name):
    d = shutil.copytree(right[name], tmp_path / "gate")
    shutil.copy(d / ("R_c_l7qs" + SFX), d / ("R_c_fx1" + SFX))
    r = C.check(d)
    assert not r["ok"] and wrong(r) == ["R_fx1", "R_fx1_rejects_duo"]


def test_fx1_without_one_multiply_fails(tmp_path):
    write_full(tmp_path, "h1", drop=(7, 3))
    r = C.check(tmp_path)
    assert not r["ok"] and wrong(r) == ["R_fx1", "R_fx1_rejects_dropped_imad"]


def test_fx1_without_another_multiply_fails_by_its_receipts_alone(tmp_path):
    write_full(tmp_path, "h1", drop=(2, 1))
    r = C.check(tmp_path)
    assert not r["ok"] and wrong(r) == ["R_fx1"]


@pytest.mark.parametrize("name", ["h1", "cohalf-h1"])
@pytest.mark.parametrize("verdict", ["device_ok", "control_rejected"])
def test_the_devices_own_verdicts_count(right, tmp_path, name, verdict):
    d = shutil.copytree(right[name], tmp_path / "gate")
    meta = json.loads((d / "meta.json").read_text())
    (d / "meta.json").write_text(json.dumps(meta | {verdict: False}))
    r = C.check(d)
    assert not r["ok"] and wrong(r) == []


def test_main_is_exit_1_when_any_gate_fails(right, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["cgemm_ref.py", str(right["h1"]), str(right["cohalf"])])
    assert C.main() == 0
    assert [json.loads(line)["ok"] for line in capsys.readouterr().out.splitlines()] == [True, True]
    d = shutil.copytree(right["cohalf"], tmp_path / "gate")
    flip(d / ("PB" + SFX), 5, np.int8)
    monkeypatch.setattr(sys, "argv", ["cgemm_ref.py", str(d), str(right["h1"])])    # a right gate after it clears nothing
    assert C.main() == 1
    assert [json.loads(line)["ok"] for line in capsys.readouterr().out.splitlines()] == [False, True]
