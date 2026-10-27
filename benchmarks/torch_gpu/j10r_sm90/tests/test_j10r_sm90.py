"""The fixtures' inputs and their writer and reader, on the CPU: the restated served weights equal the seeded generator
(values read from its module), the step's unit positions and sampled units, and `fixtures.expected` at a shrunken linear
against `h1tile.j10r_reference` called directly, with a tampered array refused by `fixtures.load`."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shape as S

import fixtures as F
from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.pouw.circuit import seeds as SD

#: per linear: B's row 0 [:4], its last row [-4:], and its first and last channel scales' f32 words, from the generator's
#: weights module at the frozen shape
GENERATOR = {"q": ([-22, 16, -79, 90], [111, 22, -24, 35], 0x370F91F7, 0x3759C24C),
        "kv": ([-126, -43, -60, 4], [96, 122, -40, 108], 0x38D16AF1, 0x388DB6BD),
        "o": ([-63, -80, -87, 8], [12, -65, 80, -127], 0x3892C008, 0x38A3017E),
        "up": ([-14, 7, -116, 44], [-96, -8, -36, -107], 0x38A718DB, 0x38C602E9),
        "down": ([38, -72, 99, 120], [83, -89, -83, 108], 0x382B9D87, 0x3802C133),
        "head": ([103, 106, -22, 40], [38, 60, -7, 5], 0x38A48456, 0x388757D9)}


@pytest.mark.parametrize("linear", list(GENERATOR))
def test_the_served_weights_are_allp_ifaces(linear):
    n, k = S.LINEARS[linear]
    first, last, s0, s1 = GENERATOR[linear]
    b = S.weight_rows(linear, [0, n - 1])
    assert b.shape == (2, k) and b.dtype == np.int8
    assert b[0, :4].tolist() == first and b[1, -4:].tolist() == last
    sw = S.weight_scales(linear)
    assert sw.shape == (n,) and sw.dtype == np.float32
    assert (int(sw.view(np.uint32)[0]), int(sw.view(np.uint32)[-1])) == (s0, s1)


def test_a_steps_units_are_numbered_in_order_without_overlap():
    for m in S.MS:
        assert S.base(S.ORDER[0], m) == S.POS0[m]
        for prev, nxt in zip(S.ORDER, S.ORDER[1:]):
            assert S.base(nxt, m) == S.base(prev, m) + S.units(prev, m)
    last = S.ORDER[-1]
    assert S.base(last, 4096) + S.units(last, 4096) <= S.POS0[16384]


@pytest.mark.parametrize("linear", list(S.LINEARS))
@pytest.mark.parametrize("m", S.MS)
def test_the_sampled_units_hold_the_grids_corners_and_middle(linear, m):
    nrb, ncb = m // S.T, S.LINEARS[linear][0] // S.T
    got = S.picks(linear, m)
    assert got == S.picks(linear, m) and got == sorted(set(got)) and len(got) == S.PICKS
    assert all(0 <= u < nrb * ncb for u in got)
    for ri, cj in ((0, 0), (0, ncb - 1), (nrb - 1, 0), (nrb - 1, ncb - 1), (nrb // 2, ncb // 2)):
        assert ri * ncb + cj in got
    assert S.T * len({u // ncb for u in got}) <= S.T * S.PICKS


def test_the_activations_carry_the_special_rows_at_both_ends():
    a = S.activation_rows("kv", 4096, [0, 1, 2, 3, 4095, 4094, 100])
    assert (a[0] == 127).all() and (a[1] == -128).all() and (a[3] == 0).all() and (a[4] == 127).all()
    assert a[2, :2].tolist() == [-128, 127] and (a[5] == -128).all()
    assert S.special_of(4096, 100) is None and len(set(a[6].tolist())) > 100


def test_the_fixtures_are_the_definitions_and_a_changed_word_is_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(S.LINEARS, "kv", (64, 512))
    doc = F.expected(tmp_path, ["kv"], [4096], 6, 1, 1)
    c = doc["calls"]["kv-4096"]
    lay = HT.Layout(S.T, S.T, 512)
    assert c["words"] == 2 * lay.sets == 2 * (512 // 32 + lay.m) and c["rows_sampled"] <= S.T * 6
    assert doc["v6_all_equal"] and c["samples"][0]["v6_receipt_equal"]
    got_doc, arrays = F.load(tmp_path)
    assert got_doc["npz_sha256"] == doc["npz_sha256"]
    ra, rb = bytes.fromhex(c["root_a"]), bytes.fromhex(c["root_b"])
    assert arrays["kv-4096/seed_c"].tobytes() == SD.call_seed(S.SALT, c["base"], ra, rb)
    for i, u in enumerate(arrays["kv-4096/units"].tolist()):
        ri, cj = divmod(u, 64 // S.T)
        a = S.activation_rows("kv", 4096, range(S.T * ri, S.T * ri + S.T)).astype(np.int64) & 0xFF
        b = S.weight_rows("kv", range(S.T * cj, S.T * cj + S.T)).astype(np.int64) & 0xFF
        z, rec = HT.j10r_reference(a.tolist(), b.tolist(), S.SALT, c["base"] + u, (ra, rb), c["base"],
                                   HT.j10r_blocks(4096, 512, (S.T, S.T), ri, cj))
        assert arrays["kv-4096/receipts"][i].tolist() == rec
        assert arrays["kv-4096/z"][i].reshape(-1).tolist() == z
        assert arrays["kv-4096/sigma"][i].tobytes() == SD.receipt_seed(S.SALT, c["base"] + u, ra, rb)
    arrays["kv-4096/receipts"][0, 0] ^= 1
    np.savez(tmp_path / "expected.npz", **arrays)
    with pytest.raises(SystemExit, match="kv-4096/receipts"):
        F.load(tmp_path)
