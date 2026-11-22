"""Network attestations committed by hash, with padding elided: injective on left-packed grids, padding elided and packing
enforced, and the digest binding the execution phase, the status and the parameters."""

import itertools

import pytest

from verity.protocols.accounting.network.verifier.commitment import record_digest, row_digest
from verity.protocols.accounting.network.verifier.grid import COMPLETE, GridParams, Reason, Status, fill


def test_elided_digest_is_injective_on_left_packed_grids():
    """Every left-packed grid on T = 2, r = 2 over a 2-frame alphabet has its own digest (as
    `SecurityProofs.FillInjective`)."""
    params = GridParams(buckets=2, allocation=2, frame_bytes=1)
    seen = {}
    for counts in itertools.product(range(3), repeat=2):
        for frames in itertools.product([b"a", b"b"], repeat=sum(counts)):
            grid = fill(list(frames), counts, 2)
            d = record_digest("l", 0, grid, COMPLETE, params)
            assert seen.setdefault(d, grid) == grid
    assert len(seen) == sum(2 ** (a + b) for a, b in itertools.product(range(3), repeat=2))


def test_padding_is_elided_and_packing_enforced():
    params = GridParams(buckets=1, allocation=4, frame_bytes=1)
    assert row_digest((b"a", None, None, None), params) != row_digest((b"a", b"a", None, None), params)
    with pytest.raises(ValueError):
        row_digest((None, b"a", None, None), params)
    with pytest.raises(ValueError):
        row_digest((b"aa", None, None, None), params)


def test_digest_binds_phase_status_and_params():
    grid = ((b"x", None),)
    params = GridParams(buckets=1, allocation=2, frame_bytes=1)
    base = record_digest("l", 0, grid, COMPLETE, params)
    assert len(bytes.fromhex(base)) == 64
    assert base != record_digest("m", 0, grid, COMPLETE, params)
    assert base != record_digest("l", 1, grid, COMPLETE, params)
    assert base != record_digest("l", 0, grid, Status(Reason.OVERFLOW), params)
    assert base != record_digest("l", 0, grid, COMPLETE, GridParams(buckets=1, allocation=2, frame_bytes=1, bucket_ms=50))
