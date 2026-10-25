"""`p2_v1/encode_store.py`'s CPU pieces: the store's root from its segments' subtree roots is core's, shards are claimed from
either end and stop at the stop file, and the oracle a shard is checked against is the registered scheme's encode."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from verity.core.protocols.pous.schemes import p2
from verity.core.protocols.pous.verifier import MerkleTree, VerifierKey, segment_leaves

BENCH = Path(__file__).resolve().parents[1]


def _store():
    spec = importlib.util.spec_from_file_location("pous_p2_encode_store", BENCH / "p2_v1" / "encode_store.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("segments", [1, 2, 3, 5])
def test_store_root_from_segment_roots_is_cores(segments):
    E = _store()
    params = p2.P2Params(16448, 21065, 4, segments=segments)
    salt = hashlib.sha512(b"salt").digest()[:24]
    segs = [[hashlib.shake_256(b"%d/%d" % (s, v)).digest(params.block_bytes) for v in range(4)] for s in range(segments)]
    roots = [MerkleTree(segment_leaves(params, salt, s, blocks)).root for s, blocks in enumerate(segs)]
    assert E.store_root(roots) == VerifierKey.from_segments(params, salt, segs).root


def test_claims_from_either_end_stop_at_the_stop_file(tmp_path):
    E = _store()
    stop = tmp_path / "stop"
    for descending, want, at in ((True, [4, 3], "2"), (False, [0, 1, 2], "3")):
        (tmp_path / "claims").write_text("0")
        stop.write_text(at)
        order = E.shard_order(0, 5, descending)
        got = []
        while (s := E.claim(tmp_path, order, str(stop), descending)) is not None:
            got.append(s)
        assert got == want
    (tmp_path / "claims").write_text("0")
    order = E.shard_order(0, 3, True)
    assert [E.claim(tmp_path, order, None, True) for _ in range(4)] == [2, 1, 0, None]


def test_fixture_is_a_prefix_stream():
    E = _store()
    assert E.fixture_segment(1037, 4096 * 2054) == E.fixture_segment(1037, 65_536 * 2054)[:4096 * 2054]
    assert E.fixture_segment(0, 64) != E.fixture_segment(1, 64)


@pytest.mark.slow
def test_reference_oracle_is_the_registered_encode():
    """About 5 s: one block through `schemes.p2`'s pure-Python roots."""
    pytest.importorskip("gmpy2")
    sys.path.insert(0, str(BENCH))
    from p2_v1 import reference as R

    params = p2.P2Params(16448, 21065, 1, segments=3)
    salt = hashlib.sha512(b"verity/p2-encode-fb/fixture-salt/v1").digest()[:24]
    w = hashlib.shake_256(b"w").digest(params.payload_bytes)
    (c,) = p2.encode_segment(params, None, salt, 2, w)
    k, t, _ = R.keys_after(p2.expand_prefix(params, salt), R.DEPLOYMENT, 2, 0)
    assert R.encode(R.DEPLOYMENT, int.from_bytes(w, "big"), k, t).to_bytes(params.block_bytes, "big") == c
