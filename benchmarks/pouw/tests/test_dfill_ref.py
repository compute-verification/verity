"""The DRAM-free filler's Python twin (decode_capacity/dfill_ref.py) against receipts dcap.cu's host `dfill_ref` printed,
the reference `dcap dgate` holds the GPU kernel to bit for bit; and the two properties the chain rests on: the byte map
is a bijection, and the receipt depends on the step count."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("dfill_ref", Path(__file__).resolve().parents[1] / "decode_capacity" / "dfill_ref.py")
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


# Map 1 (DFILL_MAP=1: the accumulator quad read in register order), the same way.
PINNED_MAP1 = [
    (1, 1, 0, 0x1CFF3491F1FF237C), (1, 1, 1, 0x7D9A7A94FDDBA199), (1, 1, 1000003, 0xC943701011E5F7D1),
    (1, 8, 0, 0x733E8C216CB02FDD), (1, 8, 1, 0x8F38190869F3A6C3), (1, 8, 1000003, 0x0E4AF9C7352FCEEE),
    (1, 64, 0, 0x592FA7AB9774A0F0), (1, 64, 1, 0x3FA855D33B182F9F), (1, 64, 1000003, 0x991284F7A46B1D74),
    (2, 1, 0, 0x015FEBF53FD50CEE), (2, 1, 1, 0x251B47D0D7C68A88), (2, 1, 1000003, 0x93FC1D11DE1863D5),
    (2, 8, 0, 0xF63C459772198BE5), (2, 8, 1, 0x6F911598E9C9EB37), (2, 8, 1000003, 0xF83C75F986A38FDC),
    (2, 64, 0, 0x2D9A584D1A532743), (2, 64, 1, 0x7CA7D5783339C428), (2, 64, 1000003, 0xED9AA19DEAF8BCD7),
    (4, 1, 0, 0xC10CC5A507FC58F1), (4, 1, 1, 0x8F77E3C51B973638), (4, 1, 1000003, 0x22F22ABAF16BD44D),
    (4, 8, 0, 0x110CAEB5EBC97F03), (4, 8, 1, 0x67F00253250FB472), (4, 8, 1000003, 0x301A1790716A7868),
    (4, 64, 0, 0x2444CD1D7AF45339), (4, 64, 1, 0x451A522722DDE4BE), (4, 64, 1000003, 0xF2E5A4621DB44B6E),
    (8, 0, 0, 0xE54F09F395158AC1), (8, 1, 1, 0x0B92403DEF87EDB7), (8, 8, 1000003, 0x00A6D2D6A88307D0),
]

# Map 2 (map 1 with a 128-bit tile seed and C in [-2^15, 2^15)), the same way.
PINNED_MAP2 = [
    (1, 1, 0, 0x1F9FFB3C6248561A), (1, 1, 1, 0x77664C2C57CC7BAF), (1, 1, 1000003, 0xDBB0FCA6999BE468),
    (1, 8, 0, 0xB1542D8E9E5DB457), (1, 8, 1, 0xF83DDEA3FA3BC5D1), (1, 8, 1000003, 0x7083D301AEA058FA),
    (1, 64, 0, 0x147D372ED2A3B46F), (1, 64, 1, 0x96B0D740670F418C), (1, 64, 1000003, 0xB84279785AF357AD),
    (2, 1, 0, 0x396EFA04345A6A1B), (2, 1, 1, 0xC9012FEF6C5699BD), (2, 1, 1000003, 0x6580A9DCC31C15CD),
    (2, 8, 0, 0x8B9233B0AC98DBFB), (2, 8, 1, 0xC04B66EB0F9A6484), (2, 8, 1000003, 0x8BE3B53625CC81B4),
    (2, 64, 0, 0x4CA4E4805E180E48), (2, 64, 1, 0x235653D2343479C8), (2, 64, 1000003, 0x0BC409AAAE6A3979),
    (4, 1, 0, 0xFACFE34FB21D66FA), (4, 1, 1, 0xB6844FE30DCAC360), (4, 1, 1000003, 0x7B12B3614B328393),
    (4, 8, 0, 0xAC9BC9D9FB683117), (4, 8, 1, 0xC5CF212E0FF64F00), (4, 8, 1000003, 0xCC25E313269D13D9),
    (4, 64, 0, 0x16FBABA3508A1290), (4, 64, 1, 0x86BD6EC8032E094F), (4, 64, 1000003, 0xFC574456BF6CA0E3),
    (8, 0, 0, 0x4CF999BE53C0B196), (8, 1, 1, 0x917215AF1F5D4638), (8, 8, 1000003, 0x582E31C63A3A1440),
]


@pytest.mark.parametrize("nc,steps,tile,want", PINNED)
def test_receipt_equals_dcap_reference(nc, steps, tile, want):
    assert R.receipt(C0, C1, tile, nc, steps) == want


@pytest.mark.parametrize("nc,steps,tile,want", PINNED_MAP1)
def test_map1_receipt_equals_dcap_reference(nc, steps, tile, want):
    assert R.receipt(C0, C1, tile, nc, steps, 1) == want


@pytest.mark.parametrize("nc,steps,tile,want", PINNED_MAP2)
def test_map2_receipt_equals_dcap_reference(nc, steps, tile, want):
    assert R.receipt(C0, C1, tile, nc, steps, 2) == want


def test_map2_seed_and_c_range():
    """Map 2's tile seed has a second 64-bit half that every operand word reads, and its C lies in [-2^15, 2^15)."""
    s = R.dseed_x(C0, C1, 7, 2)
    assert s[1] != 0 and R.dword_x(s, 2, 1, 0, 0) != R.dword_x((s[0], s[1] ^ 1), 2, 1, 0, 0)
    cs = [R.dcval_x(s, 2, r, n) for r in range(16) for n in range(8)]
    assert all(-(1 << 15) <= v < (1 << 15) for v in cs) and max(map(abs, cs)) > (1 << 13)


def test_receipt_at_measured_depth():
    """S = 2048, the depth every timed point ran, at a second pair of coins."""
    assert R.receipt(0x7F88ED5DFF52F729, 0xE9FECEAE8FAABB27, 77, 2, 2048) == 0x2F39A861AAA55C8D
    assert R.receipt(0x7F88ED5DFF52F729, 0xE9FECEAE8FAABB27, 77, 2, 2048, 1) == 0xB26B98B1DA78C728
    assert R.receipt(0x7F88ED5DFF52F729, 0xE9FECEAE8FAABB27, 77, 2, 2048, 2) == 0x712EB12C70404971


@pytest.mark.parametrize("map_", [0, 1])
def test_byte_map_is_a_bijection(map_):
    """Every byte of D lands in exactly one byte of the next A, so a step loses nothing a later step could need."""
    ids = np.arange(16 * 8 * 4, dtype=np.int64).reshape(16, 8, 4)
    a = np.empty((16, 32), dtype=np.int64)
    for g in range(8):
        for t in range(4):
            for q in range(4):
                a[g, 4 * t + q] = ids[g, 2 * t, q]
                if map_:
                    a[g + 8, 4 * t + q] = ids[g, 2 * t + 1, q]
                    a[g, 16 + 4 * t + q] = ids[g + 8, 2 * t, q]
                    a[g + 8, 16 + 4 * t + q] = ids[g + 8, 2 * t + 1, q]
                else:
                    a[g + 8, 4 * t + q] = ids[g + 8, 2 * t, q]
                    a[g, 16 + 4 * t + q] = ids[g + 8, 2 * t + 1, q]
                    a[g + 8, 16 + 4 * t + q] = ids[g, 2 * t + 1, q]
    assert sorted(a.ravel().tolist()) == list(range(16 * 8 * 4))
    d = np.random.default_rng(1).integers(-(1 << 31), 1 << 31, (16, 8)).astype("<i4")
    got = R.step(np.zeros((16, 32), np.int64), np.zeros((8, 32), np.int64), d.astype(np.int64), map_)
    by = d.view(np.uint8).reshape(16, 8, 4).astype(np.int8)
    assert got.tolist() == [[int(by.reshape(-1)[i]) for i in row] for row in a.tolist()]


@pytest.mark.parametrize("map_", [0, 1, 2])
def test_receipt_depends_on_every_step_block(map_):
    """dgate's control: a tile cut short by one 8-step block writes a different receipt."""
    for nc in (1, 2, 4):
        for tile in (0, 5):
            full = R.receipt(C0, C1, tile, nc, 64, map_)
            assert full != R.receipt(C0, C1, tile, nc, 56, map_)
            assert full != R.receipt(C0, C1, tile, nc, 63, map_)
