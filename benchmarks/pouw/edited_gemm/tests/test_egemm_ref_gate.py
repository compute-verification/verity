"""egemm_ref.py, the CPU judge of egemm.cu's gate: on a gate directory written from the definition in its docstring it
passes, and it fails by the one flag whose dumped bytes differ (an operand, a patch, the served Y, a receipt of any kind, a
key-rule answer) or when the device's own verdicts are negative. A judge that passes everything would let a wrong kernel
through the GPU run's gate."""
from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import egemm_ref as R
import numpy as np
import pytest

T, K, SEED, INV_S = R.T, 128, 0x5EED, 31.75
KEY = [0x0123456789ABCDEF, 0xFEDCBA9876543210, 0x0F1E2D3C4B5A6978, 0x8796A5B4C3D2E1F0]
TM, TN, TILES_N = 1, 2, 4   # row tile 1 and column tile 2 of a 2 x 4 grid: the call's tile 6
SFX = f"_{TM}_{TN}.bin"
KINDS = {"k_" + kind: kind for kind in ("duo32", "duo64", "xin", "xin64", "xin13", *R.TB_KIND)}
DUMPS = {"X": np.int8, "Xq": np.int8, "PA": np.int8, "Wp": np.int8, "PB": np.int8, "Y": np.int32}
#: three words the plain key rule admits and r9 refuses (3 a = 5, 5 a = 7 and 7 a = 3 mod 2^30), one both admit, one neither
R9_WORDS = [5 * pow(3, -1, 1 << 30) % (1 << 30), 7 * pow(5, -1, 1 << 30) % (1 << 30), 3 * pow(7, -1, 1 << 30) % (1 << 30),
            0x9E3779B1, 2]


def write_gate(d: Path) -> None:
    """One 128 x 128 tile as the device would dump it if it were right, with a receipt of every kind."""
    rng = np.random.default_rng(13)
    key = b"".join(struct.pack(">Q", w) for w in KEY)
    xb = (rng.normal(0, 2.5, (T, K)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    f = (xb.astype(np.uint32) << 16).view(np.float32) * np.float32(INV_S)
    X = np.clip(np.floor(f), -127, 126).astype(np.int64) + R.dither_rows(key, TM * T, T, K)
    W = rng.integers(-63, 64, (T, K)).astype(np.int64)
    U = R.u_rows(key, TN * T, T)
    Wp = W + np.tile(U, (1, K // 32))
    s1, s2, s0 = R.limbs(sum(X[:, 32 * t:32 * t + 32] for t in range(K // 32)))
    z = np.zeros((T, 32), np.int64)
    dumps = {"X": X, "Xq": np.clip(np.rint(f), -127, 127), "PA": np.hstack([s1, s2, s0, z]), "Wp": Wp,
             "PB": np.hstack([U, U, U, z]), "Y": X @ W.T}
    xb.tofile(d / ("x" + SFX))
    W.astype(np.int8).tofile(d / ("W" + SFX))
    for name, a in dumps.items():
        a.astype(DUMPS[name]).tofile(d / (name + SFX))
    words = [((X[:, :j] @ Wp[:, :j].T) & 0xFFFFFFFF).astype(np.uint32) for j in range(32, K + 1, 32)]
    for name, kind in KINDS.items():
        R.receipts(kind, SEED, words, TM * TILES_N + TN).astype(np.uint32).tofile(d / ("R_" + name + SFX))
    np.array(R9_WORDS, np.uint32).tofile(d / "r9_words.bin")
    np.array([R.rf.key_ok_r9(a) for a in R9_WORDS], np.uint8).tofile(d / "r9_ok.bin")
    meta = {"m": 2 * T, "n": TILES_N * T, "k": K, "seed": SEED, "inv_s": INV_S, "key": [f"{w:016x}" for w in KEY],
            "tiles": [[TM, TN]], "kinds": KINDS, "device_ok": True, "control_rejected": True}
    (d / "meta.json").write_text(json.dumps(meta))


@pytest.fixture(scope="module")
def right(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("gate")
    write_gate(d)
    return d


@pytest.fixture
def gate(right, tmp_path) -> Path:
    return shutil.copytree(right, tmp_path / "gate")


def flip(path: Path, at: int, dtype) -> None:
    a = np.fromfile(path, dtype)
    a[at] ^= 1
    a.tofile(path)


def tile(r: dict) -> dict:
    return r["tiles"][f"{TM},{TN}"]


def test_a_right_gate_passes(right):
    r = R.check(right)
    t = tile(r)
    assert r["ok"], r
    assert all(t[name] for name in DUMPS) and t["peel"] and t["PA_s8"] and t["R"] == dict.fromkeys(KINDS, True)
    assert r["r9_probe"] == {"n": 5, "key_ok": 4, "r9_rejected": 3, "mismatch": 0}


@pytest.mark.parametrize("name", list(DUMPS))
def test_one_wrong_dumped_word_fails_by_its_flag(gate, name):
    flip(gate / (name + SFX), 77, DUMPS[name])
    r = R.check(gate)
    assert not r["ok"] and [n for n in DUMPS if not tile(r)[n]] == [name] and all(tile(r)["R"].values())


@pytest.mark.parametrize("name", list(KINDS))
def test_one_wrong_receipt_word_fails_its_kernel(gate, name):
    flip(gate / ("R_" + name + SFX), 8 * 300 + 1, np.uint32)
    r = R.check(gate)
    assert not r["ok"] and [n for n, v in tile(r)["R"].items() if not v] == [name] and all(tile(r)[n] for n in DUMPS)


@pytest.mark.parametrize("name,other", [("k_duo64", "k_duo32"), ("k_xin64", "k_xin"), ("k_xin13", "k_xin"),
                                        ("k_tb4c", "k_tb4"), ("k_tb1w", "k_tb2w")])
def test_a_kernel_is_judged_by_its_own_kind(gate, name, other):
    """A kernel holding another kind's receipts fails: every kind has its own words, fold or spacing."""
    shutil.copy(gate / ("R_" + other + SFX), gate / ("R_" + name + SFX))
    assert [n for n, v in tile(R.check(gate))["R"].items() if not v] == [name]


@pytest.mark.parametrize("change", [{"device_ok": False}, {"control_rejected": False}, {"kinds": {}}])
def test_the_devices_own_verdicts_count(gate, change):
    meta = json.loads((gate / "meta.json").read_text())
    (gate / "meta.json").write_text(json.dumps(meta | change))
    assert not R.check(gate)["ok"]


def test_one_wrong_key_rule_answer_fails(gate):
    flip(gate / "r9_ok.bin", 0, np.uint8)
    r = R.check(gate)
    assert not r["ok"] and r["r9_probe"]["mismatch"] == 1 and all(tile(r)["R"].values())


def test_a_probe_with_no_word_r9_refuses_fails(gate):
    """Words the plain key rule and r9 judge alike show nothing of the device's r9 rule."""
    np.array(R9_WORDS[3:], np.uint32).tofile(gate / "r9_words.bin")
    np.array([1, 0], np.uint8).tofile(gate / "r9_ok.bin")
    r = R.check(gate)
    assert not r["ok"] and r["r9_probe"] == {"n": 2, "key_ok": 1, "r9_rejected": 0, "mismatch": 0}
