"""d2sgemm_ref.py, the CPU judge of d2sgemm.cu's gate (Dither-2S's linear): on a gate directory written from its own
recomputation it passes, and it fails when any one dumped byte the device could get wrong differs (X, W', the served Y, a
receipt) or when the device's own verdicts are negative. A judge that passes everything would let a wrong kernel through
the GPU run's gate."""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("d2sgemm_ref", HERE.parent / "d2sgemm_ref.py")
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)

T = R.T
KEY = [0x0123456789ABCDEF, 0xFEDCBA9876543210, 0x0F1E2D3C4B5A6978, 0x8796A5B4C3D2E1F0]


def write_gate(d: Path, k: int = 256, seed: int = 0x9ABC) -> None:
    """One 128 x 128 tile (m = n = 128) as the device would dump it if it were right."""
    rng = np.random.default_rng(7)
    inv_s = np.float32(31.75)
    x32 = rng.normal(0, 2.5, (T, k)).astype(np.float32)
    xb = (x32.view(np.uint32) >> 16).astype(np.uint16)
    x = (xb.astype(np.uint32) << 16).view(np.float32)
    W0 = rng.integers(-R.TOP, R.TOP + 1, (T, k)).astype(np.int64)
    W0[0], W0[1] = R.TOP, -R.TOP
    key = b"".join(struct.pack(">Q", w) for w in KEY)
    X, Wp = R.served(x, inv_s, W0, key, 0, 0)
    Y = X @ Wp.T
    rec = R.tile_receipts(seed, list(R.prefix_words(X, Wp)), 0)
    sfx = "_0_0.bin"
    xb.tofile(d / ("x" + sfx))
    W0.astype(np.int8).tofile(d / ("W" + sfx))
    X.astype(np.int8).tofile(d / ("X" + sfx))
    Wp.astype(np.int8).tofile(d / ("Wp" + sfx))
    Y.astype(np.int32).tofile(d / ("Y_d_l7qs" + sfx))
    for name in ("d_l7qs", "d_l2q"):
        rec.astype(np.uint32).tofile(d / ("R_" + name + sfx))
    meta = {"m": T, "n": T, "k": k, "seed": seed, "inv_s": float(inv_s), "key": [f"{w:016x}" for w in KEY],
            "tiles": [[0, 0]], "kinds": {"d_l7qs": "duo32", "d_l2q": "duo32"}, "device": {}, "device_ok": True,
            "control_rejected": True}
    (d / "meta.json").write_text(json.dumps(meta))


def flip(path: Path, at: int, dtype) -> None:
    a = np.fromfile(path, dtype)
    a[at] ^= 1
    a.tofile(path)


def test_a_right_gate_passes(tmp_path):
    write_gate(tmp_path)
    r = R.check(tmp_path)
    t = r["tiles"]["0,0"]
    assert r["ok"], r
    assert t["words"] == 256 // 32 and t["Y_is_last_word"] and t["Y_s32"]
    assert t["X_range"][0] >= -R.TOP and t["X_range"][1] <= R.TOP + 1 and t["Wp_range"][1] <= R.TOP + 1


@pytest.mark.parametrize("name,at,dtype", [("X", 4097, np.int8), ("Wp", 129, np.int8), ("Y_d_l7qs", 300, np.int32),
                                           ("R_d_l7qs", 17, np.uint32), ("R_d_l2q", 4095, np.uint32)])
def test_one_wrong_dump_fails(tmp_path, name, at, dtype):
    write_gate(tmp_path)
    flip(tmp_path / (name + "_0_0.bin"), at, dtype)
    assert not R.check(tmp_path)["ok"]


@pytest.mark.parametrize("field", ["device_ok", "control_rejected"])
def test_a_negative_device_verdict_fails(tmp_path, field):
    write_gate(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta[field] = False
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    assert not R.check(tmp_path)["ok"]


def test_the_dither_bits_are_the_patch_layout():
    """b and c are bit e % 8 of digest byte e / 8 of block (row / 4) (k / 128) + col / 128, e = (row % 4) 128 + col % 128
    (sha_noise.cuh's layout, which d2quant and d2wedit read), and the two tags draw different bits."""
    key = b"".join(struct.pack(">Q", w) for w in KEY)
    k = 384
    b = R.patch_bits(key, R.DITHER_TAG, 4, 8, k)
    for row, col in [(4, 0), (5, 131), (7, 383), (11, 200)]:
        e = (row % 4) * 128 + col % 128
        dg = R.block(key, R.DITHER_TAG, (row // 4) * (k // 128) + col // 128)
        assert b[row - 4, col] == (dg[e // 8] >> (e % 8)) & 1
    assert (b != R.patch_bits(key, R.C_TAG, 4, 8, k)).any()
