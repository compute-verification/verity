"""compressions.py: v31's Fill compressions a call by component, held to dither.py's reference hashing, to the counts of
Llama 3 8B's step, and to dither's limit on a row."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "accel_v31"))
import compressions as C

from verity.core.protocols.pouw.circuit import dither


def test_the_reference_hashes_what_the_model_counts_past_the_device_midstates():
    for case in C.CHECKS:
        got, want = C.reference(*case)
        assert got == want, case


def test_a_miscounted_component_differs_from_the_reference(monkeypatch):
    monkeypatch.setattr(C, "NODE", 3)
    got, want = C.reference(*C.CHECKS[0])
    assert got["node"] != want["node"] and got["row"] == want["row"]


def test_the_8b_filler_on_a_given_A_at_8_KiB_rows():
    st = C.step("8b", row=8192, kind="filler-a")
    layers = 32
    assert (st["leaves"], st["nodes"], st["pads"]) == (layers * 221184, layers * 245756, layers * 24576)
    assert st["macs"] == layers * 3573412790272
    assert st["parts"]["bits"] * 512 == layers * 654311424
    assert st["parts"]["t_S"] + st["parts"]["t_s"] == layers * 221184 * 65
    assert "O" not in st["parts"] or st["parts"]["O"] == 0
    assert C.total(st["parts"]) / st["macs"] * 1e6 == pytest.approx(6.062, abs=5e-4)


def test_a_served_call_of_one_segment_draws_no_noise_and_commits_O():
    c = C.components(16384, 4096, 4096, row=16384, kind="served")
    assert c["W"] == 0 and c["parts"]["noise"] == c["parts"]["perm"] == 0
    assert c["parts"]["O"] == 4096 * 4 * C.row_blocks(16384)


@pytest.mark.parametrize("row", [65536, 262144])
def test_a_row_past_32_KiB_is_a_format_change(row):
    assert not C.legal(row) and C.legal(32768)
    with pytest.raises(ValueError, match="dividing 32768"):
        dither.Shape(256, 256, 1024, 128, 512, row)
    big, top = C.step("8b", row=row, kind="filler-a"), C.step("8b", row=32768, kind="filler-a")
    assert big["bytes"] == top["bytes"] and big["leaves"] * row == top["leaves"] * 32768
