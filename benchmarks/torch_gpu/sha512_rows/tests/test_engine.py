"""The engine's host twin, and on a CUDA device every variant, against the CPU reference (check.py's checks)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import bench
import check
import engine as E


def _cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


ENGINES = ["host", pytest.param("cuda", marks=pytest.mark.skipif(not _cuda(), reason="no CUDA device"))]


@pytest.fixture(scope="module", params=ENGINES)
def engine(request: pytest.FixtureRequest) -> E.Engine:
    return E.Engine(request.param)


def test_row_digests_match_the_reference_at_edge_sizes(engine: E.Engine) -> None:
    r = check.check_rows(engine)
    assert set(r["mismatches"]) == set(engine.variants)
    assert all(v == 0 for v in r["mismatches"].values()), r


def test_hm96_leaves_match_under_a_salt_key_and_given_salts(engine: E.Engine) -> None:
    r = check.check_leaves(engine)
    assert r["mismatches"] == {"salt_key": 0, "salts": 0}, r


def test_frame_leaves_levels_pads_and_roots_match_tree_levels(engine: E.Engine) -> None:
    r = check.check_trees(engine)
    assert r["mismatches"] and not any(r["mismatches"].values()), r


@pytest.mark.skipif(not _cuda(), reason="no CUDA device")
def test_the_h200_layer_matches_on_sampled_slots() -> None:
    r = check.check_layout(E.Engine("cuda"), 257, [0, 31, 32, 256])
    assert r["mismatches"] == {"x": 0, "leaves": 0}, r


def test_the_h200_layer_is_commit_slots_frozen_layout() -> None:
    sources, rows = bench.h200_rows()
    sizes = [sum(n for _, _, n in r) for r in rows]
    assert len(rows) == 25 and sum(sizes) == 663_824 and max(sizes) == 32768
    assert sorted(set(sizes)) == [4, 256, 8192, 16384, 16388, 32768]
    assert all(off + n <= sources[t] for r in rows for t, off, n in r)


def test_words_and_digests_round_trip() -> None:
    d = [bytes(range(i, i + 64)) for i in range(3)]
    assert E.digests(E.words(d)) == d
    assert E.digests(np.stack([E.words(d)])) == d


def test_a_variant_name_is_four_modes_and_an_optional_straight_line_mark() -> None:
    assert E.variant_flags("2211") == ("-DSX_V=90", "-DSX_ROLL=1")
    assert E.variant_flags("0000s") == ("-DSX_V=0", "-DSX_ROLL=0")
    with pytest.raises(ValueError):
        E.variant_flags("3000")


def test_a_row_outside_its_tensor_is_refused() -> None:
    e = E.Engine("host")
    with pytest.raises(ValueError, match="outside"):
        e.rows([[(0, 8, 9)]], [np.zeros((2, 16), dtype=np.uint8)])
    with pytest.raises(ValueError, match="one or two parts"):
        e.rows([[(0, 0, 1)] * 3], [np.zeros((2, 16), dtype=np.uint8)])
