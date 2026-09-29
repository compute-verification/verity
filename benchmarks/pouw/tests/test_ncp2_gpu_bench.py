"""CPU contract tests for `ncp2_gpu_bench`'s accounting: the bytes each leaf hash absorbs per Qwen2.5-0.5B forward."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("ncp2_gpu_bench", Path(__file__).parents[1] / "ncp2_gpu_bench.py")
B = importlib.util.module_from_spec(spec)
sys.modules["ncp2_gpu_bench"] = B
spec.loader.exec_module(B)


def test_padding_and_the_tree():
    assert [B.absorbed("sha256", n) for n in (0, 55, 56, 119, 120)] == [64, 64, 128, 128, 192]
    assert [B.absorbed("sha512", n) for n in (111, 112)] == [128, 256]
    assert [B.absorbed("turboshake128", n) for n in (167, 168)] == [168, 336]
    assert [B.tree_nodes(p) for p in (1, 2, 3, 5, 8)] == [0, 1, 2, 4, 7]


def test_a_tiles_words_are_three_quarters_of_a_byte_per_useful_mac():
    for m in B.ROWS:
        acc = B.forward_accounting(m)
        words = sum(B.LAYERS * (m // 16) * (n // 16) * 3 * k // 16 * 256 * 4 for k, n in B.PD.QWEN05)
        assert words / acc["macs"] == 0.75
        assert 0.75 < acc["tile_bytes_per_mac"] < 0.76


def test_at_sixteen_rows_the_per_call_y_strips_hash_as_much_as_the_tile_words():
    ab = B.forward_accounting(16)["absorbed"]["sha256"]
    assert abs(ab["form_y strips (committed per call)"] / ab["tile words"] - 1) < 1e-3
    assert round(ab["total"] / 1e9, 2) == 9.57
