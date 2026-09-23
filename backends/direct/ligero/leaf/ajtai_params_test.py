"""Tests for the Ajtai / Ring-SIS leaf parameter derivation (lane ajtai-design).

Imported by path so the test runs whether or not `leaf/__init__.py` exists on this branch."""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

_SPEC = importlib.util.spec_from_file_location("ajtai_params", pathlib.Path(__file__).with_name("ajtai_params.py"))
ap = importlib.util.module_from_spec(_SPEC)
sys.modules["ajtai_params"] = ap                     # dataclasses resolve the defining module through sys.modules
_SPEC.loader.exec_module(ap)  # type: ignore[union-attr]


def test_field_constants() -> None:
    assert ap.P == 2**31 - 2**27 + 1 == 2013265921
    assert (ap.P - 1) % 2**27 == 0                      # X^n + 1 splits completely for every power of two n <= 2^26
    assert abs(ap.LOG2P - 30.907) < 1e-3
    assert abs(ap.C_CLASSICAL - 0.2925) < 1e-3 and abs(ap.C_QUANTUM - 0.2653) < 1e-3 and abs(ap.C_PLAUSIBLE - 0.2075) < 1e-3


def test_delta_bkz_matches_published_anchors() -> None:
    # lattice-dogs LaBRADOR uses delta = 1.00444 (~ BKZ-341); Dilithium's 128-bit classical line is BKZ-438 (0.292 * 438 = 128).
    assert abs(ap.delta_bkz(341) - 1.00444) < 2e-5
    assert abs(ap.delta_bkz(438) - 1.00374) < 2e-5
    for b in range(100, 2000, 100):                     # monotone decreasing in b
        assert ap.delta_bkz(b + 100) < ap.delta_bkz(b)


def test_derive_ring_element_is_in_field_and_prefix_consistent() -> None:
    b64 = ap.derive_ring_element(3, 64)
    b128 = ap.derive_ring_element(3, 128)
    assert len(b64) == 64 and len(b128) == 128
    assert all(0 <= v < ap.P for v in b128)
    assert b128[:64] == b64                              # ajtai-n64 is the top-left block of ajtai-n128
    assert ap.derive_ring_element(3, 64) == b64          # deterministic
    assert ap.derive_ring_element(4, 64) != b64          # index in the seed
    assert ap.derive_ring_element(3, 64, b"other") != b64  # domain in the seed
    B = ap.derive_B(8, 16)
    assert len(B) == 8 and all(len(b) == 16 for b in B)
    assert [b[:16] for b in ap.derive_B(8, 32)] == B


def test_derive_ring_element_rejection_sampling_regression_vector() -> None:
    # Reference: SHAKE-256(b"verity/ajtai-babybear/B/v1" || LE32(0)), LE32 words masked to 31 bits, reject >= p.
    import hashlib
    xof = hashlib.shake_256(ap.DOMAIN + (0).to_bytes(4, "little")).digest(4096)
    ref = [v for v in (int.from_bytes(xof[k:k + 4], "little") & 0x7FFF_FFFF for k in range(0, 4096, 4)) if v < ap.P][:64]
    assert ap.derive_ring_element(0, 64) == ref
    # not all raw words were accepted (acceptance 15/16): the masked stream must contain some rejects in 4 KB
    raw = [int.from_bytes(xof[k:k + 4], "little") & 0x7FFF_FFFF for k in range(0, 4096, 4)]
    assert any(v >= ap.P for v in raw)


def test_m0_m1_ordering_and_recommended_set() -> None:
    # Proof-norm rule M0 is the most conservative, M1 above it; the recommended ajtai-n64 clears 2x block size under M1.
    b0 = ap.m0_blocksize(64, 12288)
    b1, d1 = ap.m1_blocksize(64, 12288)
    assert b0 < b1
    assert 64 < d1 < 12288
    assert ap.C_CLASSICAL * b1 >= 256 and ap.C_QUANTUM * b1 >= 128 * 1.9          # 285 / 259 in the note
    assert 100 <= ap.C_CLASSICAL * b0 <= 130                                      # 114 in the note (the M0 caveat)
    # n = 128 (bf16) satisfies even M0 at 128 quantum bits for m = 24576
    assert ap.C_QUANTUM * ap.m0_blocksize(128, 24576) >= 128
    # n = 32 has no margin: under 2x
    b32, _ = ap.m1_blocksize(32, 12288)
    assert 128 <= ap.C_CLASSICAL * b32 < 160


def test_m1_and_m2_independent_of_m_but_m0_and_wagner_not() -> None:
    assert ap.m1_blocksize(64, 12288) == ap.m1_blocksize(64, 24576)
    assert ap.m0_blocksize(64, 12288) > ap.m0_blocksize(64, 24576)
    assert ap.wagner_bits(64, 12288)[1] > ap.wagner_bits(64, 24576)[1]


def test_m2_linf_model_is_above_m1_and_swifft_crosscheck() -> None:
    # SWIFFT (64, 1024, 257): Buchmann-Lindner 2009 report ~68-bit pseudo-collisions by lattice reduction; the paper's own
    # generalised-birthday estimate is 2^106.  M1 must land near the former, M2 above M1, Wagner in the 100-135 range.
    lq = math.log2(257)
    b1, d1 = ap.m1_blocksize(64, 1024, 1, lq)
    assert 60 <= ap.C_CLASSICAL * b1 <= 80 and 200 <= d1 <= 320
    b2, c2 = ap.m2_optimize(64, 1024, 1, ap.C_CLASSICAL, 257)
    assert c2 > ap.C_CLASSICAL * b1 and 95 <= c2 <= 115
    k, wg = ap.wagner_bits(64, 1024, 1, lq)
    assert 100 <= wg <= 135
    # Buchmann-Lindner's recommended (96, 18, 389) "127-bit" set is ~118 Core-SVP classical under M1
    bq, _ = ap.m1_blocksize(96, 1728, 1, math.log2(389))
    assert 110 <= ap.C_CLASSICAL * bq <= 130


def test_bits_beat_nibbles_and_bytes() -> None:
    b_bits, _ = ap.m1_blocksize(64, 12288, 1)
    b_nib, _ = ap.m1_blocksize(64, 3072, 15)
    b_byte, _ = ap.m1_blocksize(64, 1536, 255)
    assert b_bits > b_nib > b_byte


def test_lhl_random_bits() -> None:
    assert ap.lhl_random_bits(64) == math.ceil(64 * ap.LOG2P + 256) == 2235
    assert ap.lhl_random_bits(128) == 4213


def test_estimate_row_smoke() -> None:
    e = ap.estimate(32, 6912, 1)
    assert e.n == 32 and e.m == 6912 and e.digest_bits > 900
    assert "| 1 | 32 | 6912 |" in e.row()
