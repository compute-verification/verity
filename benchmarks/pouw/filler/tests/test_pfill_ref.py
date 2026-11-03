"""pfill's Python twin (pfill_ref.py, hashlib's SHA-512) against pfill.cuh's host `dfill_ref` built with g++
(pfill_host.cpp), the reference `dcap dgate` holds the GPU kernel to bit for bit, in every tile kind (s8
on ±1, e2m1 on ±1 under -DPFILL_E2M1, s8 on {0,1} bits under -DPFILL_BITS, bU under -DPFILL_BU); and the properties the
filler's proved floor reads: the operands are uniform noise with nothing else in them (bU: bits on A, {±1, ..., ±64} on
B), every k32 (e2m1: k64) word is exactly its popcount identity (bU: the integer sum of U over a's bits), the bits
tile's words meet Theorem K's hypothesis (odd coefficients, F_2 rank 2K) where the ±1 tile's do not, and the receipt
depends on every fragment, on the tile id, on the coins and on the tile's kind."""

from __future__ import annotations

import ctypes
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pfill_ref", HERE / "pfill_ref.py")
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)

C0, C1 = 0x0123456789ABCDEF, 0xFEDCBA9876543210


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++ for the host build of pfill.cuh")
    so = tmp_path_factory.mktemp("pfill-host") / "pfill_host.so"
    subprocess.run([gxx, "-O2", "-std=c++17", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", "-Wno-unknown-pragmas",
                    "-o", str(so), str(HERE / "pfill_host.cpp")], check=True)
    cdll = ctypes.CDLL(str(so))
    cdll.pfill_ref.restype = ctypes.c_uint64
    cdll.pfill_ref.argtypes = [ctypes.c_uint64] * 3 + [ctypes.c_int] * 4
    cdll.pfill_width.restype = ctypes.c_int
    cdll.pfill_width.argtypes = [ctypes.c_uint64]
    cdll.pfill_e2m1x8.restype = ctypes.c_uint32
    cdll.pfill_e2m1x8.argtypes = [ctypes.c_uint32]
    cdll.pfill_s8x4.restype = ctypes.c_uint32
    cdll.pfill_s8x4.argtypes = [ctypes.c_uint32, ctypes.c_int]
    cdll.pfill_u4.restype = ctypes.c_uint32
    cdll.pfill_u4.argtypes = [ctypes.c_uint32]
    cdll.pfill_bu_hist.restype = None
    cdll.pfill_bu_hist.argtypes = [ctypes.c_uint64] * 2 + [ctypes.c_int] * 2 + [ctypes.c_void_p] * 2
    cdll.pfill_bu_uniform.restype = ctypes.c_int
    cdll.pfill_bu_uniform.argtypes = [ctypes.c_uint64] * 2 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
    return cdll


@pytest.fixture(scope="module")
def host(lib):
    return lambda c0, c1, tile, nc, steps, kind=R.S8, w=R.W: lib.pfill_ref(c0, c1, tile, nc, steps, int(kind), w)


@pytest.mark.parametrize("kind", R.KINDS)
@pytest.mark.parametrize("nc", [R.XOR, R.DUO, R.RAM])
@pytest.mark.parametrize("steps", [1, 4, 8, 13])
@pytest.mark.parametrize("tile", [0, 1, 1000003])
def test_twin_equals_host_reference(host, nc, steps, tile, kind):
    assert R.receipt(C0, C1, tile, nc, steps, kind) == host(C0, C1, tile, nc, steps, kind)


@pytest.mark.parametrize("kind", R.KINDS)
@pytest.mark.parametrize("w", [2, 4])
@pytest.mark.parametrize("nc", [R.DUO, R.RAM])
def test_narrow_phase_tile_equals_host_reference(lib, host, nc, w, kind):
    """dfill_p's tile at PF_PW = 2 or 4 warps (128 w columns) under a phase id; the width is in what a receipt binds, and
    dfill_ref takes PF_PW (8 unless built otherwise) for phase ids and the pump's 8 for every other id."""
    tile = (1 << 62) | 12345
    got = host(C0, C1, tile, nc, 16, kind, w)
    assert R.receipt(C0, C1, tile, nc, 16, kind, w) == got
    assert got != host(C0, C1, tile, nc, 16, kind, 8)
    assert lib.pfill_width(tile) == 8 and lib.pfill_width(12345) == 8


@pytest.mark.parametrize("kind", R.KINDS)
def test_twin_equals_host_reference_at_measured_depth(host, kind):
    """S = 32, dgate's short depth, at a second pair of coins."""
    c0, c1 = 0x7F88ED5DFF52F729, 0xE9FECEAE8FAABB27
    assert R.receipt(c0, c1, 77, R.RAM, 32, kind) == host(c0, c1, 77, R.RAM, 32, kind)


def test_operands_are_balanced_noise(lib):
    """Each entry is a SHA-512 output bit: over a tile's B and a warp's first fragments the two values are equally likely
    (to 5 sigma), and the s8 bytes the tile writes are exactly 1 - 2 bit (pf_pm1) or the bit itself (pf_01)."""
    for kind in R.KINDS:
        bb = np.concatenate([R.rows(C0, C1, 9, R.CTR_B, b, kind) for b in range(R.TN // 16)])
        bits = np.unpackbits((bb if kind == R.E2M1 else bb.astype(np.uint32)).view(np.uint8))
        n = bits.size
        assert n == R.TN * (64 if kind == R.E2M1 else 32)
        assert abs(int(bits.sum()) - n / 2) < 5 * np.sqrt(n) / 2
    for nib in range(16):
        for kind, val in ((R.S8, lambda b: 1 - 2 * b), (R.BITS, lambda b: b)):
            got = np.frombuffer(np.uint32(lib.pfill_s8x4(nib, kind)).tobytes(), np.int8)
            assert got.tolist() == [val((nib >> q) & 1) for q in range(4)]


def test_kinds_draw_independent_noise():
    """Each kind hashes in its own domain, so no kind's noise is another's."""
    rs = {kind: R.rows(C0, C1, 9, R.CTR_B, 0, kind) & np.uint64(R.M32) for kind in R.KINDS}
    assert len({r.tobytes() for r in rs.values()}) == len(R.KINDS)


def _e2m1(code: int) -> float:
    s, e, m = code >> 3, (code >> 1) & 3, code & 1
    return (-1.0) ** s * (m * 0.5 if e == 0 else 2.0 ** (e - 1) * (1 + m / 2))


def test_e2m1_codes_are_pm1(lib):
    """pf_e2m1x8 puts entry q of a byte in nibble q as the E2M1 code of 1 - 2 bit: +1.0 (0x2) or -1.0 (0xA)."""
    for byte in range(256):
        word = lib.pfill_e2m1x8(byte)
        assert [_e2m1((word >> (4 * q)) & 15) for q in range(8)] == [1.0 - 2 * ((byte >> q) & 1) for q in range(8)]


@pytest.mark.parametrize("kind", [R.S8, R.E2M1, R.BITS])
def test_product_is_the_popcount_identity(kind):
    """C[r][n] = sum_k A[r][k] B[k][n] for the bit rows the twin and dfill_ref use: K - 2 popcount(a_r ^ b_n) on ±1,
    popcount(a_r & b_n) on bits, and the e2m1 tile's word is that integer's f32 bits (exact: |C| <= 64)."""
    k = 64 if kind == R.E2M1 else 32
    a_rows, b_cols = R.rows(C0, C1, 3, 0, 0, kind), R.rows(C0, C1, 3, R.CTR_B, 0, kind)
    ent = (lambda b: b) if kind == R.BITS else (lambda b: 1 - 2 * b)
    val = lambda w: np.array([ent((int(w) >> i) & 1) for i in range(k)])
    a = np.stack([val(w) for w in a_rows])
    b = np.stack([val(w) for w in b_cols]).T
    x, y = a_rows[:, None], b_cols[None, :]
    want = R.popcount(x & y) if kind == R.BITS else k - 2 * R.popcount(x ^ y)
    assert (a @ b == want).all()
    got = R.words(x, y, kind)
    exp = want.astype(np.float32).view(np.uint32) if kind == R.E2M1 else (want & R.M32).astype(np.uint32)
    assert (got == exp).all()


def _rank_f2(rows: list[int]) -> int:
    rank, rows = 0, list(rows)
    while rows:
        p = rows.pop()
        if p:
            rank += 1
            low = p & -p
            rows = [r ^ p if r & low else r for r in rows]
    return rank


@pytest.mark.parametrize("kind", [R.BITS, R.S8])
def test_theorem_k_hypothesis_on_the_committed_word(kind):
    """Theorem K reads a cell's value mod 2 as a quadratic form in the
    input bits: it needs integer operands affine in the bits (the bits tile's bytes are the bits) and a target whose
    (a_k, b_k') coefficients are odd exactly when k = k' (F_2 rank 2K). A word is Σ_k A_k B_k, so its second difference
    in (a_u, b_v) is that coefficient at every base point: 1·[u = v] on bits. The ±1 tile is K's mutation: its
    coefficient is 4, 0 mod 2, and the F_2 rank is 0 (round 6 §1, "K is vacuous on even coefficients")."""
    k, rng = 32, np.random.default_rng(7)

    def word(a: int, b: int) -> int:
        w = int(R.words(np.array([a], np.uint64), np.array([b], np.uint64), kind)[0])
        return w - (1 << 32) if w >= 1 << 31 else w  # the s32 value

    for a0, b0 in rng.integers(0, 1 << 32, (3, 2), dtype=np.uint64).tolist():
        coef = np.zeros((k, k), np.int64)
        for u in range(k):
            for v in range(k):
                a1, b1 = a0 ^ (1 << u), b0 ^ (1 << v)
                d = word(a1, b1) - word(a1, b0) - word(a0, b1) + word(a0, b0)
                coef[u, v] = d * (1 if a1 > a0 else -1) * (1 if b1 > b0 else -1)  # a flip from 1 to 0 negates d
        assert (coef == (1 if kind == R.BITS else 4) * np.eye(k, dtype=np.int64)).all()
        m = [sum(int(x) << j for j, x in enumerate(row)) for row in coef % 2]
        # the form on (a, b) is [[0, M], [Mᵀ, 0]] over F_2, of rank 2 rank(M)
        assert 2 * _rank_f2(m) == (2 * k if kind == R.BITS else 0)


def test_keys_pass_the_screen():
    for tile in range(64):
        for dom in R.DOMS.values():
            a0, a1, _, _ = R.keys(R.block(C0, C1, tile, R.CTR_KEY, dom))
            assert R.key_ok(a0) and R.key_ok(a1) and a0 != a1


def test_receipt_depends_on_fragments_tile_coins_and_kind(host):
    """dgate's control (a tile cut short) differs, as do the next tile, the next coins and every other kind: a receipt
    replays nowhere."""
    for kind in R.KINDS:
        for nc in (R.XOR, R.DUO, R.RAM):
            full = host(C0, C1, 5, nc, 32, kind)
            assert full != host(C0, C1, 5, nc, 24, kind)
            assert full != host(C0, C1, 5, nc, 31, kind)
            assert full != host(C0, C1, 5 + 1, nc, 32, kind)
            assert full != host(C0 ^ 1, C1, 5, nc, 32, kind)
            for other in R.KINDS:
                if other != kind:
                    assert full != host(C0, C1, 5, nc, 32, other)


def test_bu_entries_are_bits_and_uniform_u(lib):
    """bU's s8 bytes: A's are the bits themselves (0 or 1), and U from a digest byte t is (t & 0x7F) - 64 + ((t >> 6) &
    1) = -64 + sum_{p<6} 2^p t_p + 65 t_6 (affine in the bits, odd weight on t_0 and t_6), in pf_u4's SWAR form as in
    the twin; each of the 128 values {±1, ..., ±64} comes from exactly two bytes (bit 7 unused), so U is uniform on them
    and never 0."""
    for nib in range(16):
        got = np.frombuffer(np.uint32(lib.pfill_s8x4(nib, R.BU)).tobytes(), np.int8)
        assert got.tolist() == [(nib >> q) & 1 for q in range(4)]
    t = np.arange(256)
    u = R.u_of(t)
    affine = -64 + sum((((t >> p) & 1) << p) for p in range(6)) + 65 * ((t >> 6) & 1)
    assert (u == affine).all()
    for b in range(256):  # every byte value in every byte of the word
        x = np.array([b, (b + 85) & 255, (b + 170) & 255, 255 - b], np.uint8)
        got = np.frombuffer(np.uint32(lib.pfill_u4(int(x.view("<u4")[0]))).tobytes(), np.int8)
        assert got.tolist() == R.u_of(x).tolist()
    vals, counts = np.unique(u, return_counts=True)
    assert vals.tolist() == [v for v in range(-64, 65) if v] and (counts == 2).all()


def test_bu_word_is_the_integer_product_with_affine_parity():
    """C[r][n] = sum_k A[r][k] U[k][n] over a fragment's bit rows and a tile's U columns, as the twin commits it (s32,
    |C| <= 2048), and its parity is the F2-affine form popcount(a & (t_0 ^ t_6)) mod 2 of the operand bits."""
    a_rows = R.rows(C0, C1, 3, 0, 0, R.BU)
    u = R.bu_columns(C0, C1, 3)
    a = ((a_rows[:, None] >> np.arange(32, dtype=np.uint64)) & np.uint64(1)).astype(np.int64)
    want = a @ u
    assert (np.abs(want) <= 2048).all()
    assert (R.bu_products(a_rows[None], u)[0] == (want & R.M32).astype(np.uint32)).all()
    t = np.stack([np.frombuffer(R.digest(C0, C1, 3, R.CTR_B + b, R.DOM_BU), np.uint8).reshape(2, 32)
                  for b in range(R.TN // 2)]).reshape(R.TN, 32).astype(np.int64)
    par = ((t & 1) ^ ((t >> 6) & 1)).T  # [k, n]
    assert ((want & 1) == (a @ par) % 2).all()


def test_bu_uniformity_gate(lib):
    """dgate's pf_bu_uniform: its histograms are the twin's (U's s8 bytes over each tile's 512 B blocks, A's over
    fragments 0 .. frags - 1, from hashlib), its chi-squares are those histograms', and at two pairs of coins it passes
    at dgate's sample size, every U value present and none outside {±1, ..., ±64}, A's bytes only 0 and 1."""
    tiles, frags = 2, 64
    hu, ha = (ctypes.c_uint64 * 256)(), (ctypes.c_uint64 * 256)()
    lib.pfill_bu_hist(C0, C1, tiles, frags, hu, ha)
    tw_u = np.zeros(256, np.int64)
    tw_a = np.zeros(256, np.int64)
    for tile in range(tiles):
        u = R.bu_columns(C0, C1, tile)
        np.add.at(tw_u, u.ravel() & 255, 1)
        for f in range(frags):
            tw_a[:2] += np.bincount(np.unpackbits(np.frombuffer(R.digest(C0, C1, tile, f, R.DOM_BU), np.uint8)),
                                    minlength=2)
    assert list(hu) == tw_u.tolist() and list(ha) == tw_a.tolist()
    allowed = np.array([v for v in range(-64, 65) if v]) & 255
    assert tw_u[0] == 0 and tw_u.sum() == tw_u[allowed].sum() == tiles * R.TN * 32
    for c0, c1 in ((C0, C1), (0x7F88ED5DFF52F729, 0xE9FECEAE8FAABB27)):
        chi = (ctypes.c_double * 2)()
        assert lib.pfill_bu_uniform(c0, c1, 16, 2048, chi) == 1
        assert 0 < chi[0] < 248 and 0 <= chi[1] < 36
    chi = (ctypes.c_double * 2)()
    lib.pfill_bu_uniform(C0, C1, tiles, frags, chi)
    eu, ea = tw_u.sum() / 128, tw_a.sum() / 2
    assert chi[0] == pytest.approx(float(((tw_u[allowed] - eu) ** 2 / eu).sum()), rel=1e-12)
    assert chi[1] == pytest.approx(float(((tw_a[:2] - ea) ** 2 / ea).sum()), rel=1e-12)
