"""compile_slice_zk's scan of what the verifier received leaves out exactly the witness pieces with at most one nonzero
byte, which a proof's encoding holds whatever the witness, and keeps the rest held as they were. Nothing here lowers or
proves (`compile_slice_zk.prove` does, on a node)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_compile_slice_zk", HERE / "compile_slice_zk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sparse_pieces_leaves_out_one_nonzero_byte():
    Z = _load()
    zero, low, high = bytes(16), b"\x05" + bytes(15), bytes(15) + b"\x80"
    two, full = b"\x01\x02" + bytes(14), bytes(range(1, 17))
    pieces = {"wit 0 @0": low, "wit 0 @16": two, "wit 0 @32": full, "wit 8 @0": high, "wit salt 0 @0": zero}
    kept, held, out = Z.sparse_pieces(pieces, {"wit 0 @0", "wit 0 @16", "wit 0 @32", "wit 8 @0"})
    assert kept == {"wit 0 @16": two, "wit 0 @32": full}
    assert held == {"wit 0 @16", "wit 0 @32"}
    assert out == sorted(x.hex() for x in (zero, low, high))
