"""The DRAM-free filler's Python twin (dfill_ref.py) against receipts dcap.cu's host `dfill_ref` printed,
the reference `dcap dgate` holds the GPU kernel to bit for bit; and the two properties the chain rests on: the byte map
is a bijection, and the receipt depends on the step count."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("dfill_ref", Path(__file__).resolve().parents[1] / "dfill_ref.py")
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)

C0, C1 = 0x0123456789ABCDEF, 0xFEDCBA9876543210

# (nc, steps, tile, receipt) at coins (C0, C1), from dcap.cu's dfill_ref compiled for the host.
PINNED = [
    (1, 0, 0, 0x555318184C6E8E80), (1, 0, 1, 0xE07645166BE805F5), (1, 0, 1000003, 0x53BAE75D9CDB5225),
    (1, 1, 0, 0xEA7663D9B4B28496), (1, 1, 1, 0x1B50C8A6520DFEF5), (1, 1, 1000003, 0x261E62A48D1167CD),
    (1, 8, 0, 0x1A4FB01AE6066F36), (1, 8, 1, 0xC597D161FFDC6E94), (1, 8, 1000003, 0x2F9702DE1611B5AC),
    (1, 64, 0, 0xEE6BF458BCF4A76A), (1, 64, 1, 0x4D6079178C7619E9), (1, 64, 1000003, 0x4603CC3F431C8146),
    (2, 0, 0, 0xAABCF514ADA6072C), (2, 0, 1, 0x24F1161A8A2DC66C), (2, 0, 1000003, 0xE39C6301E27C6CAC),
    (2, 1, 0, 0x087FBF0C71BE0FEB), (2, 1, 1, 0xA5C0EBC9CAD52C1B), (2, 1, 1000003, 0x239A2A993E1B0865),
    (2, 8, 0, 0xB3C92ABFC91CB2D1), (2, 8, 1, 0x6CD0D5FDE2D255AB), (2, 8, 1000003, 0x97A6A26347D02708),
    (2, 64, 0, 0x343E1CA9D6D62F9B), (2, 64, 1, 0xD38826FBA0279E75), (2, 64, 1000003, 0x6691B2D55B4BBC1D),
    (4, 0, 0, 0x66757045CB73A93A), (4, 0, 1, 0xC9E1B01B18747FE9), (4, 0, 1000003, 0xC44BCCFFE7940162),
    (4, 1, 0, 0x2EFDAAC85354038E), (4, 1, 1, 0x852E1D2DE61FABD2), (4, 1, 1000003, 0x4830C3613D11754C),
    (4, 8, 0, 0x0B6C277F1A30EDF8), (4, 8, 1, 0x81B1D31A3EED4812), (4, 8, 1000003, 0x4A4B898C297BA38C),
    (4, 64, 0, 0x9706DEF61861DE77), (4, 64, 1, 0x562175D445EB4BED), (4, 64, 1000003, 0x68020171A98FF7EF),
]


@pytest.mark.parametrize("nc,steps,tile,want", PINNED)
def test_receipt_equals_dcap_reference(nc, steps, tile, want):
    assert R.receipt(C0, C1, tile, nc, steps) == want


def test_receipt_at_measured_depth():
    """S = 2048, the depth every timed point ran, at a second pair of coins."""
    assert R.receipt(0x7F88ED5DFF52F729, 0xE9FECEAE8FAABB27, 77, 2, 2048) == 0x2F39A861AAA55C8D


def test_byte_map_is_a_bijection():
    """Every byte of D lands in exactly one byte of the next A, so a step loses nothing a later step could need."""
    ids = np.arange(16 * 8 * 4, dtype=np.int64).reshape(16, 8, 4)
    a = np.empty((16, 32), dtype=np.int64)
    for g in range(8):
        for t in range(4):
            for q in range(4):
                a[g, 4 * t + q] = ids[g, 2 * t, q]
                a[g + 8, 4 * t + q] = ids[g + 8, 2 * t, q]
                a[g, 16 + 4 * t + q] = ids[g + 8, 2 * t + 1, q]
                a[g + 8, 16 + 4 * t + q] = ids[g, 2 * t + 1, q]
    assert sorted(a.ravel().tolist()) == list(range(16 * 8 * 4))
    d = np.random.default_rng(1).integers(-(1 << 31), 1 << 31, (16, 8)).astype("<i4")
    got = R.step(np.zeros((16, 32), np.int64), np.zeros((8, 32), np.int64), d.astype(np.int64))
    by = d.view(np.uint8).reshape(16, 8, 4).astype(np.int8)
    assert got.tolist() == [[int(by.reshape(-1)[i]) for i in row] for row in a.tolist()]


def test_receipt_depends_on_every_step_block():
    """dgate's control: a tile cut short by one 8-step block writes a different receipt."""
    for nc in (1, 2, 4):
        for tile in (0, 5):
            full = R.receipt(C0, C1, tile, nc, 64)
            assert full != R.receipt(C0, C1, tile, nc, 56)
            assert full != R.receipt(C0, C1, tile, nc, 63)
