"""The engine's host twin, and on a CUDA device every variant, against the CPU reference (check.py's checks)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def test_hm96_leaves_match_under_a_salt_key_and_given_salts_and_fused(engine: E.Engine) -> None:
    r = check.check_leaves(engine)
    assert r["mismatches"] == {"salt_key": 0, "salts": 0, "fused": 0}, r


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


def _cost(e: E.Engine, fn: Any) -> dict[str, int]:
    e.counts(reset=True)
    fn()
    return e.counts(reset=True)


def test_each_call_costs_its_compressions_on_the_host_twin() -> None:
    """Counted, not timed: a row of at most 111 bytes is one compression, the empty row's over a constant schedule (its
    block is padding alone), and a row whose last block is padding alone ends with one, as do 112 and 128 bytes' second;
    an hm96 leaf five (c two, h two of which the padding block's schedule is a constant, the frame leaf one from the row's
    midstate) and three ChaCha20 blocks, and fused with its unit's slot-tree leaf six; a unit's slot-tree leaf one from its
    frame table's midstate (two from rank 2^24), against frame_leaves' two from the IV; a node two."""
    e = E.Engine("host")
    G, S = 6, 4
    t = (np.arange(S * 128) % 251).astype(np.uint8).reshape(S, 128)
    table = e.rows([[(0, 0, n)] for n in (0, 1, 16, 64, 100, 111)], [t])
    assert _cost(e, lambda: e.row_digests(table)) == {"compressions": G * S, "kw": S, "chacha": 0}
    pads = e.rows([[(0, 0, n)] for n in (110, 111, 112, 127, 128)], [t])
    assert _cost(e, lambda: e.row_digests(pads)) == {"compressions": (1 + 1 + 2 + 2 + 2) * S, "kw": 3 * S, "chacha": 0}
    x = e.row_digests(table)
    lt = e.leaf_table([bytes(64)], [0] * G, list(range(G)))
    assert _cost(e, lambda: e.leaves(lt, x, E.SaltKey(bytes(32)))) == {"compressions": 5 * G * S, "kw": G * S, "chacha": 3 * G * S}
    ft = e.frame_table([bytes(range(64))], 100, G)
    assert _cost(e, lambda: e.frame_table([bytes(range(64))], 100, G)) == {"compressions": G, "kw": 0, "chacha": 0}
    dest = e.u32s(np.arange(G))
    assert _cost(e, lambda: e.leaves(lt, x, E.SaltKey(bytes(32)), None, 0, ft, dest, e._words(S, G))) == {"compressions": 6 * G * S, "kw": G * S, "chacha": 3 * G * S}
    v = e._words(G, S)
    assert _cost(e, lambda: e.frame_units(ft, np.arange(G), v, e._words(S, G)))["compressions"] == G * S
    assert _cost(e, lambda: e.frame_leaves(v, [bytes(64)], 100))["compressions"] == 2 * G * S
    far = e.frame_table([bytes(64)], 1 << 24, G)
    assert _cost(e, lambda: e.frame_units(far, np.arange(G), v, e._words(S, G)))["compressions"] == 2 * G * S
    mids = e.node_mids([bytes(64)], 3)
    assert _cost(e, lambda: e.level(e._words(S, 2 * G), [bytes(64)], 3, (1 << 64) - G, mids))["compressions"] == 2 * G * S


def test_words_and_digests_round_trip() -> None:
    d = [bytes(range(i, i + 64)) for i in range(3)]
    assert E.digests(E.words(d)) == d
    assert E.digests(np.stack([E.words(d)])) == d


def test_a_variant_name_is_four_modes_and_an_optional_straight_line_mark() -> None:
    assert E.variant_flags("2211") == ("-DSX_V=90", "-DSX_ROLL=1")
    assert E.variant_flags("0000s") == ("-DSX_V=0", "-DSX_ROLL=0")
    assert E.variant_flags("0033") == ("-DSX_V=240", "-DSX_ROLL=1")
    with pytest.raises(ValueError):
        E.variant_flags("4000")


def test_a_row_outside_its_tensor_is_refused() -> None:
    e = E.Engine("host")
    with pytest.raises(ValueError, match="outside"):
        e.rows([[(0, 8, 9)]], [np.zeros((2, 16), dtype=np.uint8)])
    with pytest.raises(ValueError, match="one or two parts"):
        e.rows([[(0, 0, 1)] * 3], [np.zeros((2, 16), dtype=np.uint8)])
