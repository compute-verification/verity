"""Pearl-C's work law (`pearl_c_work`, statement review item 5): declared unit records, credit by real cells, draws over
credited tiles only, the verifier's rule on a drawn tile (reports, never raises), and the audit by the records."""

from __future__ import annotations

import pytest

import pearl_c_vectors as CV
from verity_pouw.audit import Epoch, Prover, Workload
from verity_pouw.protocol import Matmul, Shape, Tile
from verity_pouw.schemes import pearl_c as C
from verity_pouw.schemes import pearl_c_work as W
from verity_pouw.schemes import pearl_kw as P

S = C.PearlC(k_min=32, tile=(16, 16), forming="v1")
K = 128


def record(real=20, fail_a=(), fail_b=(), k=K, kept=K) -> W.UnitRecord:
    return W.UnitRecord(Matmul(0, Shape(32, k, 32), 0), real, kept, tuple(fail_a), tuple(fail_b))


def spiked(row: bytes) -> bytes:
    """Three entries at 10^4 among 128: the row fails `live_row` (64·3 > 128)."""
    w = list(C.parse_f32_row(row, K))
    for p in (1, 2, 3):
        w[p] = P.f32_bits(1e4)
    return C.f32_row(w)


def test_credit_follows_real_rows_and_declared_failures():
    tiles = {(t.rows[0], t.cols[0]): t for t in S.tiles(Shape(32, K, 32))}
    full, mixed = W.tile_weight(S, record(), tiles[(0, 0)]), W.tile_weight(S, record(), tiles[(16, 0)])
    assert full > 0 and mixed == full * 4 / 16
    assert W.tile_weight(S, record(real=16), tiles[(16, 0)]) == 0
    r = record(fail_a=(3,), fail_b=(20,))
    assert W.tile_weight(S, r, tiles[(0, 0)]) == 0 and W.tile_weight(S, r, tiles[(0, 16)]) == 0
    assert W.tile_weight(S, r, tiles[(16, 0)]) > 0 and W.tile_weight(S, r, tiles[(16, 16)]) == 0
    assert W.work(S, [record()]) == S.wref(Shape(20, K, 32))
    assert W.work(S, [record(kept=64)]) == S.wref(Shape(20, 64, 32))
    picks = W.draw(S, [r], b"source", bytes(32), bytes(32), bytes(32), 200)
    assert picks and all(W.credited(r, t) for _, t in picks) and {t.rows[0] for _, t in picks} == {16}
    assert W.draw(S, [record(fail_a=(0, 16))], b"source", bytes(32), bytes(32), bytes(32), 5) == []


def test_records_bind_every_field():
    base = W.records_root(b"s", b"w", [record()])
    for other in (record(real=21), record(fail_a=(1,)), record(fail_b=(2,)), record(kept=120)):
        assert W.records_root(b"s", b"w", [other]) != base
    assert W.records_root(b"t", b"w", [record()]) != base != W.records_root(b"s", b"v", [record()])
    for bad in (dict(fail_a=(25,)), dict(fail_a=(3, 3)), dict(fail_b=(32,)), dict(real=0), dict(real=33), dict(kept=129)):
        with pytest.raises(ValueError):
            record(**bad)


def test_check_drawn_reports_and_never_raises():
    rows_a, rows_b = CV.rows_f32("work", "A", 16, K), CV.rows_f32("work", "B", 16, K)
    tile = Tile(tuple(range(16)), tuple(range(16)))
    honest = W.UnitRecord(Matmul(0, Shape(16, K, 16), 0), 16, K)
    assert W.check_drawn(honest, tile, rows_a, rows_b) is None
    bad_a = rows_a[:5] + [spiked(rows_a[5])] + rows_a[6:]
    assert "fails the per-row rules" in W.check_drawn(honest, tile, bad_a, rows_b)
    assert "B row fails" in W.check_drawn(honest, tile, rows_a, rows_b[:15] + [spiked(rows_b[15])])
    declared = W.UnitRecord(Matmul(0, Shape(16, K, 16), 0), 16, K, (5,))
    assert "not creditable" in W.check_drawn(declared, tile, bad_a, rows_b)
    narrow = W.UnitRecord(Matmul(0, Shape(16, K, 16), 0), 16, 120)
    assert "past the kept width" in W.check_drawn(narrow, tile, rows_a, rows_b)
    filler = rows_a[:12] + [C.f32_row([0] * K)] * 4
    assert "per-row rules" in W.check_drawn(honest, tile, filler, rows_b)
    assert W.check_drawn(W.UnitRecord(Matmul(0, Shape(16, K, 16), 0), 12, K), tile, filler, rows_b) is None
    assert "malformed" in W.check_drawn(honest, tile, rows_a[:15] + [b"\0" * 8], rows_b)


def lifecycle():
    """32 x 128 x 16 on 16 x 16 tiles: A's rows 20 … 31 are zero filler and row 3 fails liveness; B passes."""
    rows_a = CV.rows_f32("work-life", "A", 20, K) + [C.f32_row([0] * K)] * 12
    rows_a[3] = spiked(rows_a[3])
    rows_b = CV.rows_f32("work-life", "B", 16, K)
    work = Workload((Matmul(3, Shape(32, K, 16), 9),))
    prover = Prover(S, {9: rows_b})
    epoch = Epoch.start(work, prover.weight_roots, b"beacon")
    return epoch, prover, prover.prove(epoch, {3: rows_a}), rows_a, rows_b


def test_audit_credits_real_rows_and_skips_declared_failures():
    epoch, prover, tr, rows_a, rows_b = lifecycle()
    records = W.declare(S, epoch, {3: rows_a}, {9: rows_b}, real_rows={3: 20})
    assert records == [W.UnitRecord(epoch.workload.matmuls[0], 20, K, (3,), ())]
    audit = W.audit(S, epoch, prover, tr, records, b"verifier", 8)
    assert audit.accepted and len(audit.verdicts) == 8 and set(audit.selected) == {1}
    assert audit.profile.work == int(W.work(S, records)) == S.wref(Shape(20, K, 16)) * 4 // 20


def test_audit_catches_a_lying_record():
    epoch, prover, tr, rows_a, rows_b = lifecycle()
    u = epoch.workload.matmuls[0]
    hides = W.audit(S, epoch, prover, tr, [W.UnitRecord(u, 20, K)], b"verifier", 8)
    assert not hides.accepted and any("per-row rules" in v.reason for v in hides.verdicts)
    pads = W.audit(S, epoch, prover, tr, [W.UnitRecord(u, 32, K, (3,))], b"verifier", 8)
    assert not pads.accepted and any("per-row rules" in v.reason for v in pads.verdicts)
    other = W.audit(S, epoch, prover, tr, [W.UnitRecord(Matmul(4, u.shape, 9), 20, K, (3,))], b"verifier", 8)
    assert not other.accepted and other.verdicts == ()
