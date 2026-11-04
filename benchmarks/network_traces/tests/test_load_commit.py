"""load_commit's host side: its streamed commitments are their references' (the certifier's hm96 root is
`Registered.commit`'s, the store's leaves and root are `pous.verifier`'s), and the frozen shape's load is the 130.10 GiB
of INT8 weights and the 68 M P2 blocks they encode into, each matrix a power of two of 256-byte rows. Nothing here times
anything (`load_commit.main` does, on a node)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_load_commit", HERE / "load_commit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lc = _load()


def test_streamed_commitments_are_their_references():
    assert lc.exact() == {"hm96_root_equal": True, "store_leaves_equal": True, "store_root_equal": True}


def test_the_frozen_shapes_load():
    sh = lc.shape()
    assert sh["weight_bytes"] == 52 * 16384 * (16384 + 64 + 64 + 16384 + 65536 + 65536)
    assert round(sh["weight_gib"], 2) == 130.10 and sh["weight_rows"] == 545_685_504
    assert sh["values_per_matrix_unpadded"] and sh["one_value_padded_rows"] == 1 << 30
    assert 68_000_000 < sh["p2_blocks"] < 68_100_000
