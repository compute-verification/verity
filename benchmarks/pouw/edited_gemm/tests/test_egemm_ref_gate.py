"""egemm_ref.py, the CPU judge of egemm.cu's gate: on a gate directory written from the definition in its docstring it
passes, and it fails by the one flag whose dumped bytes differ (an operand, a patch, the served Y, a receipt of any kind, a
key-rule answer) or when the device's own verdicts are negative. A judge that passes everything would let a wrong kernel
through the GPU run's gate.

The gate is written without the judge's arithmetic, so a judge that computes a tile wrongly fails on it too: the key's
blocks, the dither bits and U by the kernels' rules (sha_noise.cuh, lpgemm.cu), the limbs by the docstring's, the duo and
XOR-in receipts a thread at a time by the receipt folds' reference, and the tb receipts by floored_gemm's reference, the
kind by that reference's own table and the unchained ones a running sum at a time. test_d2sgemm_ref.py writes its gates
with the same blocks, bits, running sums and duo."""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
from collections.abc import Callable
from functools import cache
from pathlib import Path

import egemm_ref as R
import numpy as np
import pytest

from verity.protocols.pouw.receipts import receipt_folds as F

T, K, INV_S = R.T, 128, 31.75
#: a seed whose first key candidate the plain key rule admits and r9 refuses: XIN's keys are candidates 0 and 1, XIN13's
#: 1 and 2, so the receipts tell the two rules apart and say where the key stream starts
SEED = 0x10483
KEY = [0x0123456789ABCDEF, 0xFEDCBA9876543210, 0x0F1E2D3C4B5A6978, 0x8796A5B4C3D2E1F0]
TM, TN, TILES_N = 1, 2, 4   # row tile 1 and column tile 2 of a 2 x 4 grid: the call's tile 6
SFX = f"_{TM}_{TN}.bin"
#: the tb kernels' kinds under the names floored_gemm/ref.py gives fgemm.cu's
TB = {"tb4": "tb4_32", "tb4c": "tb4c_32", "tb2w": "tb2w_32", "tb1w": "tb1w_32"}
KINDS = {"k_" + kind: kind for kind in ("duo32", "duo64", "xin", "xin64", "xin13", *TB)}
DUMPS = {"X": np.int8, "Xq": np.int8, "PA": np.int8, "Wp": np.int8, "PB": np.int8, "Y": np.int32}
#: three words the plain key rule admits and r9 refuses (3 a = 5, 5 a = 7 and 7 a = 3 mod 2^30), one both admit, one neither
R9_WORDS = [5 * pow(3, -1, 1 << 30) % (1 << 30), 7 * pow(5, -1, 1 << 30) % (1 << 30), 3 * pow(7, -1, 1 << 30) % (1 << 30),
            0x9E3779B1, 2]
#: r9's answer on each of them
R9_OK = [0, 0, 0, 1, 0]


@cache
def block(tag: int, j: int) -> bytes:
    """sha_noise.cuh's block (tag, j) of the key: SHA-512 of the key and the big-endian counter tag << 56 | j."""
    return hashlib.sha512(b"".join(struct.pack(">Q", w) for w in KEY) + bytes([tag]) + j.to_bytes(7, "big")).digest()


def bits(tag: int, row0: int, rows: int, k: int) -> np.ndarray:
    """A stacked operand's bits on its rows from row0, sha_noise.cuh's NS 1 (the dither under tag 7): at (row, col) the
    patch's block (row / 4) (k / 128) + col / 128, entry e = (row % 4) 128 + col % 128, bit e % 8 of digest byte e / 8."""
    def bit(row: int, col: int) -> int:
        e = (row % 4) * 128 + col % 128
        return (block(tag, (row // 4) * (k // 128) + col // 128)[e // 8] >> (e % 8)) & 1

    return np.array([[bit(row0 + i, c) for c in range(k)] for i in range(rows)])


def u_rows(row0: int) -> np.ndarray:
    """U's T rows from row0, lpgemm.cu's LP_UW 64 under tag 6: at (row, col) digest byte v at (row % 2) 32 + col of block
    row / 2, (v & 63) + 1 and negated if v & 64."""
    def u(row: int, col: int) -> int:
        v = block(6, row // 2)[(row % 2) * 32 + col]
        return -(v & 63) - 1 if v & 64 else (v & 63) + 1

    return np.array([[u(row0 + i, c) for c in range(32)] for i in range(T)])


def limbs(S: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """S = 256 (S_1 + S_2) + S_0: S_0 the low byte as s8, S_1 the rest held to ±127, S_2 what that left over."""
    s0 = (S + 128) % 256 - 128
    rest = (S - s0) // 256
    s1 = np.minimum(np.maximum(rest, -127), 127)
    return s1, rest - s1, s0


def running_sums(X: np.ndarray, Wp: np.ndarray) -> list[np.ndarray]:
    """T_1 .. T_{k / 32}: T_s is X W'^T over the first 32 s columns, mod 2^32."""
    return [((X[:, :j] @ Wp[:, :j].T) & 0xFFFFFFFF).astype(np.uint32) for j in range(32, X.shape[1] + 1, 32)]


def thread_words(Tw: list[list[int]], ctid: int) -> list[int]:
    """Thread ctid's 32 accumulator words of one running sum, in the order the folds absorb them."""
    return [Tw[r][c] for r, c in (F.served_entry(ctid, q) for q in range(32))]


def duo(seed: int, sums: list[list[list[int]]], tile: int, ctid: int) -> list[int]:
    """fgemm.cu's duo over the running sums given: a thread's eight chain words."""
    y, z = F.served_start(seed, tile, ctid)
    for Tw in sums:
        y, z = F.served_duo_absorb(y, z, thread_words(Tw, ctid))
    return [F.duo_final(a, b) for a, b in zip(y, z)]


def xin(seed: int, sums: list[list[list[int]]], tile: int, ctid: int) -> list[int]:
    """egemm.cu's XIN, the receipt folds' xin7 step (rotations 7 and 19): keys the first two of the key stream that pass
    pfill's rule, chain c from mix32(seed ^ mix32((tile NCT + ctid) 8 + c)), warp tile t's four words into chain t % 4."""
    a0, a1, _ = F.egemm_xin_keys(seed, F.key_ok)
    y = [F.mix32(seed ^ F.mix32((tile * F.NCT + ctid) * 8 + c)) for c in range(4)]
    for Tw in sums:
        p = thread_words(Tw, ctid)
        for t in range(8):
            y[t % 4] = F.FOLDS["xin7"]["step"](y[t % 4], *p[4 * t:4 * t + 4], a0, a1)
    return y + [0] * 4


def xin13(seed: int, sums: list[list[list[int]]], tile: int, ctid: int) -> list[int]:
    """egemm.cu's XIN13, the receipt folds' egemm schedule: keys under r9's rule, chain c from (tile NCT + ctid) 4 + c."""
    a0, a1, _ = F.egemm_xin_keys(seed)
    y = F.egemm_xin13_start(tile, ctid)
    for Tw in sums:
        y = F.egemm_xin13_absorb(y, thread_words(Tw, ctid), a0, a1)
    return y + [0] * 4


def threads(fold: Callable, seed: int, sums: list[np.ndarray], tile: int) -> np.ndarray:
    """A fold's receipts over the running sums given, [thread, 8]."""
    own = [Tw.tolist() for Tw in sums]
    return np.array([fold(seed, own, tile, ctid) for ctid in range(F.NCT)], np.uint32)


def tb(kind: str, sums: list[np.ndarray]) -> np.ndarray:
    """A tb kernel's receipts by floored_gemm/ref.py, which the judge loads: a chained kind over the running sums in
    order; an unchained one is the sum over them of A_t K_t, so the sum of each one's receipts."""
    ref, shape = R.fref, (T, F.BN, F.WM, F.NCT, 1, 1)   # one tile: its rows, its columns, a warp's rows, its threads
    if TB[kind] in ref.TBC:
        return ref.tbc_receipts(ref.TBC[TB[kind]], sums, *shape)[0]
    return np.sum([ref.tb_receipts(ref.TB[TB[kind]], ref.u8bytes(Tw), *shape)[0] for Tw in sums], axis=0, dtype=np.uint32)


def receipts(kind: str, sums: list[np.ndarray], tile: int) -> np.ndarray:
    """A kernel's receipts over the tile's running sums T_1, T_2, ..., [thread, 8]: the 64-column kinds over every
    second one, T_2, T_4, ..."""
    if kind in TB:
        return tb(kind, sums)
    fold, own = {"duo32": (duo, sums), "duo64": (duo, sums[1::2]), "xin": (xin, sums), "xin64": (xin, sums[1::2]),
                 "xin13": (xin13, sums)}[kind]
    return threads(fold, SEED, own, tile)


def write_gate(d: Path) -> None:
    """One 128 x 128 tile as the device would dump it if it were right, with a receipt of every kind."""
    rng = np.random.default_rng(13)
    xb = (rng.normal(0, 2.5, (T, K)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    f = (xb.astype(np.uint32) << 16).view(np.float32) * np.float32(INV_S)
    X = np.clip(np.floor(f), -127, 126).astype(np.int64) + bits(7, TM * T, T, K)
    W = rng.integers(-63, 64, (T, K)).astype(np.int64)
    U = u_rows(TN * T)
    Wp = W + np.tile(U, (1, K // 32))
    s1, s2, s0 = limbs(sum(X[:, 32 * t:32 * t + 32] for t in range(K // 32)))
    z = np.zeros((T, 32), np.int64)
    dumps = {"X": X, "Xq": np.clip(np.rint(f), -127, 127), "PA": np.hstack([s1, s2, s0, z]), "Wp": Wp,
             "PB": np.hstack([U, U, U, z]), "Y": X @ W.T}
    xb.tofile(d / ("x" + SFX))
    W.astype(np.int8).tofile(d / ("W" + SFX))
    for name, a in dumps.items():
        a.astype(DUMPS[name]).tofile(d / (name + SFX))
    sums = running_sums(X, Wp)
    for name, kind in KINDS.items():
        receipts(kind, sums, TM * TILES_N + TN).astype(np.uint32).tofile(d / ("R_" + name + SFX))
    np.array(R9_WORDS, np.uint32).tofile(d / "r9_words.bin")
    np.array(R9_OK, np.uint8).tofile(d / "r9_ok.bin")
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
    assert r["r9_probe"] == {"n": 5, "key_ok": 4, "r9_rejected": 3, "mismatch": 0} and r["r9_seed_skips"] == 1


def test_the_limbs_are_the_docstrings_past_s1s_bound():
    """One tile's sums stay inside S_1's bound (|S| is at most 127 k / 32), so the split is held here on sums past it."""
    S = np.array([0, 127, 128, -128, -129, 255, 256, 127 * 256 + 127, 127 * 256 + 128, -127 * 256 - 128, -127 * 256 - 129,
                  40000, -40000, 1 << 20])
    s1, s2, s0 = R.limbs(S)
    assert all((mine == judges).all() for mine, judges in zip(limbs(S), (s1, s2, s0)))
    assert (256 * (s1 + s2) + s0 == S).all() and (s0 >= -128).all() and (s0 <= 127).all()
    assert (s1 == np.clip(s1 + s2, -127, 127)).all() and (s2 != 0).sum() == 5


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
