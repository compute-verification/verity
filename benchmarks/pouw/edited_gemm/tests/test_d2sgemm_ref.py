"""d2sgemm_ref.py, the CPU judge of d2sgemm.cu's gate (Dither-2S's linear): on a gate directory written from the
definition in its docstring it passes, and it fails when any one dumped byte the device could get wrong differs (X, W', the
served Y, a receipt) or when the device's own verdicts are negative. A judge that passes everything would let a wrong
kernel through the GPU run's gate.

The gate is written without the judge's arithmetic, as test_egemm_ref_gate.py's is and with its helpers, so a judge that
computes a tile wrongly fails on it too: b under tag 7 and c under tag 8 by sha_noise.cuh's rule, and both schedules'
receipts, fgemm.cu's duo over the running sums, a thread at a time by the receipt folds' reference."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import d2sgemm_ref as R
import numpy as np
import pytest
import test_egemm_ref_gate as G

T, TOP = R.T, 126
KEY = G.KEY


def floored(x: np.ndarray, inv_s: np.float32) -> np.ndarray:
    """clamp(floor(x inv_s), -126, 126): X before b, and Q K^T's K0."""
    return np.clip(np.floor(x * inv_s), -TOP, TOP).astype(np.int64)


def write_gate(d: Path, k: int = 256, seed: int = 0x9ABC) -> None:
    """One 128 x 128 tile (m = n = 128) as the device would dump it if it were right."""
    rng = np.random.default_rng(7)
    inv_s = np.float32(31.75)
    x32 = rng.normal(0, 2.5, (T, k)).astype(np.float32)
    xb = (x32.view(np.uint32) >> 16).astype(np.uint16)
    x = (xb.astype(np.uint32) << 16).view(np.float32)
    W0 = rng.integers(-TOP, TOP + 1, (T, k)).astype(np.int64)
    W0[0], W0[1] = TOP, -TOP
    X, Wp = floored(x, inv_s) + G.bits(7, 0, T, k), W0 + G.bits(8, 0, T, k)
    Y = X @ Wp.T
    rec = G.threads(G.duo, seed, G.running_sums(X, Wp), 0)
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
    assert t["X_range"][0] >= -TOP and t["X_range"][1] <= TOP + 1 and t["Wp_range"] == [-TOP, TOP + 1]


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


def write_qk_gate(d: Path, head_rows: bool = True, seed: int = 0x9ABC) -> None:
    """Q K^T's tile (3, 1) of H = 2 heads at 256 tokens (head 1's second query tile against its second key tile), K'
    sampled on head 1's rows (or, head_rows False, on head 0's, as a kernel that lost the head offset would)."""
    rng = np.random.default_rng(11)
    tokens, tm, tn, k = 256, 3, 1, 128
    inv_s = np.float32(1.0)

    def bf(a: np.ndarray) -> np.ndarray:
        return (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)

    def up(b: np.ndarray) -> np.ndarray:
        return (b.astype(np.uint32) << 16).view(np.float32)

    xb, kb = bf(rng.uniform(-127.5, 127.5, (T, k))), bf(rng.uniform(-127.5, 127.5, (T, k)))
    wrow = (tm // (tokens // T)) * tokens * head_rows + tn * T
    X, Wp = floored(up(xb), inv_s) + G.bits(7, tm * T, T, k), floored(up(kb), inv_s) + G.bits(8, wrow, T, k)
    rec = G.threads(G.duo, seed, G.running_sums(X, Wp), tm * (tokens // T) + tn)
    sfx = f"_{tm}_{tn}.bin"
    xb.tofile(d / ("x" + sfx))
    kb.tofile(d / ("xk" + sfx))
    X.astype(np.int8).tofile(d / ("X" + sfx))
    Wp.astype(np.int8).tofile(d / ("Wp" + sfx))
    (X @ Wp.T).astype(np.int32).tofile(d / ("Y_qk_l7qs" + sfx))
    for name in ("qk_l7qs", "qk_l2q"):
        rec.astype(np.uint32).tofile(d / ("R_" + name + sfx))
    meta = {"mode": "qk", "served": "qk_l7qs", "m": 2 * tokens, "n": tokens, "k": k, "H": 2, "Hkv": 1, "T": tokens,
            "seed": seed, "inv_s": float(inv_s), "key": [f"{w:016x}" for w in KEY], "tiles": [[tm, tn]],
            "kinds": {"qk_l7qs": "duo32", "qk_l2q": "duo32"}, "device": {}, "device_ok": True, "control_rejected": True}
    (d / "meta.json").write_text(json.dumps(meta))


def test_qk_gate_passes_and_needs_the_head_offset(tmp_path):
    """Q K^T's mode: K'_h's bits are on head h's rows (h T + t), so a K' sampled on another head's rows fails."""
    (tmp_path / "right").mkdir()
    (tmp_path / "wrong").mkdir()
    write_qk_gate(tmp_path / "right")
    write_qk_gate(tmp_path / "wrong", head_rows=False)
    r = R.check(tmp_path / "right")
    assert r["ok"], r
    assert r["tiles"]["3,1"]["words"] == 4
    assert not R.check(tmp_path / "wrong")["ok"]


def test_the_dither_bits_are_the_patch_layout():
    """b and c are in sha_noise.cuh's layout, which d2quant and d2wedit read, at a k of three patches a row too, and the
    two tags sample different bits."""
    key = b"".join(struct.pack(">Q", w) for w in KEY)
    b, c = R.patch_bits(key, R.DITHER_TAG, 4, 8, 384), R.patch_bits(key, R.C_TAG, 4, 8, 384)
    assert (b == G.bits(7, 4, 8, 384)).all() and (c == G.bits(8, 4, 8, 384)).all() and (b != c).any()
