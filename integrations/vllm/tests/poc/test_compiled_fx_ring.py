"""CPU tests for the rung (c) device ring (``record_op.DeviceRing``): slot allocation per (piece, id),
row-prefix writes, oversize refusal, drain order and the op body's dispatch between ring and host path."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verity_vllm_adapter.compiled_fx import record_op


def test_ring_write_and_drain_prefix_rows() -> None:
    ring = record_op.DeviceRing(rows=8)
    x = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    y = torch.arange(3 * 2, dtype=torch.float32).reshape(3, 2) + 100
    assert ring.write(1, 7, x)
    assert ring.write(0, 3, y)
    assert set(ring.slots) == {(1, 7), (0, 3)}
    assert ring.slots[(1, 7)].shape == (8, 4)
    assert ring.eager_calls == 2 and ring.capture_calls == 0
    recs = ring.drain(3, lambda t: t.clone())
    # (piece, id) order: piece 0 first
    assert [(r.runtime_piece, r.id) for r in recs] == [(0, 3), (1, 7)]
    assert recs[0].shape == (3, 2) and torch.equal(recs[0].host, y)
    assert recs[1].shape == (3, 4) and torch.equal(recs[1].host, x)
    assert [r.seq for r in recs] == [0, 1]
    assert not ring.written


def test_ring_refuses_oversize_batches_but_still_allocates() -> None:
    ring = record_op.DeviceRing(rows=4)
    big = torch.zeros(9, 4)
    assert not ring.write(2, 5, big)
    assert (2, 5) in ring.slots and ring.oversize_calls == 1
    small = torch.ones(2, 4)
    assert ring.write(2, 5, small)
    recs = ring.drain(2, lambda t: t.clone())
    assert len(recs) == 1 and torch.equal(recs[0].host, small)


def test_ring_drain_skip_and_noncontiguous_source() -> None:
    ring = record_op.DeviceRing(rows=4)
    base = torch.arange(4 * 6, dtype=torch.float32).reshape(4, 6)
    view = base[:, ::2]  # non-contiguous (4, 3)
    assert ring.write(0, 1, view)
    assert ring.write(0, 2, torch.zeros(4, 1))
    recs = ring.drain(4, lambda t: t.clone(), skip={(0, 2)})
    assert [(r.runtime_piece, r.id) for r in recs] == [(0, 1)]
    assert torch.equal(recs[0].host, view.contiguous())
    assert recs[0].stride == (3, 1)


def test_op_body_dispatches_to_ring_then_host_fallback() -> None:
    op = record_op.install()
    sink = record_op.SINK
    sink.reset()
    sink.enabled = True
    sink.snapshot = lambda t: t.clone()
    before = sink.calls_total
    record_op.RING = record_op.DeviceRing(rows=2)
    try:
        record_op.RUNTIME["piece"] = 3
        op(torch.ones(2, 3), 11)  # fits: ring path, no host record
        assert sink.records == [] and sink.calls_total == before + 1
        op(torch.ones(5, 3), 11)  # oversize: host path
        assert len(sink.records) == 1 and sink.records[0].id == 11 and sink.records[0].runtime_piece == 3
        assert record_op.RING.oversize_calls == 1
        assert record_op.RING.stats()["slots"] == 1
    finally:
        record_op.RING = None
        sink.reset()
        sink.enabled = False
