"""Pearl-C's work law (`pearl_c_work`; statement review item 5, red team F1, F2, F5–F8): splits bounded by 1% of k, credit
on real, kept, unexcluded cells at Lean's accounting, the tile rule (zero filler and padding, credited rows pass and
excluded rows fail, the debit under the cap), draws keyed by a beacon round after the commitment, and the audit."""

from __future__ import annotations

import struct
from fractions import Fraction

import pytest

import quicknet as Q

import pearl_c_vectors as CV
from test_pouw_pearl_c_debit import spiked_rows
from verity_pouw.audit import Epoch, Prover, Workload
from verity_pouw.protocol import Matmul, Shape, Tile
from verity_pouw.schemes import pearl_c as C
from verity_pouw.schemes import pearl_c_work as W
from verity_pouw.schemes import pearl_kw as P

S = C.PearlC(k_min=32, tile=(16, 16), forming="v1")
K = 128
AFTER = Q.LATEST
BEFORE = Q.recorded_before(AFTER)


def record(real=20, excluded_a=(), excluded_b=(), k=K, kept=K) -> W.UnitRecord:
    return W.UnitRecord(Matmul(0, Shape(32, k, 32), 0), real, kept, tuple(excluded_a), tuple(excluded_b))


def spiked(row: bytes, k: int = K) -> bytes:
    """Three entries at 10^4 among 128: the row fails `live_row` (64·3 > 128)."""
    w = list(C.parse_f32_row(row, k))
    for p in (1, 2, 3):
        w[p] = P.f32_bits(1e4)
    return C.f32_row(w)


def test_split_is_bounded_and_canonical():
    sp = W.Split(3584, (5, 9))
    assert (sp.k_kept, sp.k_unit) == (3582, 3584) and W.Split(257, (1,)).k_unit == 256 and W.Split(300, (1,)).k_unit == 384
    W.Split(1000, tuple(range(10)))
    for bad in ((1000, tuple(range(11))), (3584, tuple(range(127))), (99, (1,)), (1000, (9, 5)), (1000, (5, 5)),
                (1000, (1000,))):
        with pytest.raises(ValueError):
            W.Split(*bad)
    assert W.Split(1000, (1,)).encode() != W.Split(1000, (2,)).encode() != W.Split(1000, (2,), b"calib").encode()


def test_credit_is_per_row_on_kept_cells_at_leans_accounting():
    tiles = {(t.rows[0], t.cols[0]): t for t in S.tiles(Shape(32, K, 32))}
    cell = W.cell_wref(S, record())
    assert cell == Fraction(S.wref(Shape(1, K, 32)), 32) and W.cell_credit(S, record()) == Fraction(S.credit_of(Shape(1, K, 32)), 32)
    r = record(excluded_a=(3,), excluded_b=(20,))
    assert W.tile_weight(S, r, tiles[(0, 0)]) == cell * 15 * 16 and W.tile_weight(S, r, tiles[(0, 16)]) == cell * 15 * 15
    assert W.tile_weight(S, r, tiles[(16, 16)]) == cell * 4 * 15 and W.tile_weight(S, record(real=16), tiles[(16, 0)]) == 0
    wref, credit = W.work(S, [r])
    assert wref == cell * 19 * 31 and credit == (1 - S.cap) * W.cell_credit(S, r) * 19 * 31
    assert W.work(S, [record(kept=64)])[0] == Fraction(S.wref(Shape(20, 64, 32)))
    picks = W.draw(S, [r], AFTER, bytes(32), bytes(32), bytes(32), 200)
    assert {(t.rows[0], t.cols[0]) for _, t in picks} == set(tiles) and all(r.credited(t)[0] for _, t in picks)
    assert W.draw(S, [r], W.Beacon(6, b"another round"), bytes(32), bytes(32), bytes(32), 200) != picks


def test_tile_cap_is_the_cells_share_of_the_units_credit():
    """The per-tile statement's cap (`Tile.lean`'s `pearlCTiles`): ρ·creditOf(unit at |K|)·tileShare, with tileShare =
    |rows|·|cols|/(m·n).  A row's own terms (forming, A′·F_B) count once across its row band, so a unit's tile caps sum to
    ρ·creditOf of the unit; ρ·creditOf of each tile's own shape would count them once per tile.  At sm_120's prices the cap
    is sm_120's credit."""
    sm120 = C.PearlC(k_min=32, tile=(16, 16), forming="v1", prices=C.SM120_PRICES)
    for scheme, kept in ((S, K), (S, 64), (sm120, K)):
        r = record(real=32, kept=kept)
        s = r.matmul.shape
        total = Fraction(0)
        for t in scheme.tiles(s):
            rows, cols = r.credited(t)
            _, cap = W.tile_cap(scheme, r, [[0] * s.k] * len(rows), [[0] * s.k] * len(cols))
            assert cap == scheme.cap * scheme.credit_of(Shape(s.m, kept, s.n)) * Fraction(len(rows) * len(cols), s.m * s.n)
            assert cap < scheme.cap * scheme.credit_of(Shape(len(rows), kept, len(cols)))
            total += cap
        assert total == scheme.cap * scheme.credit_of(Shape(s.m, kept, s.n))
    assert sm120.credit_of(Shape(32, K, 32)) < S.credit_of(Shape(32, K, 32))
    r = record(excluded_a=(3,), excluded_b=(20,))
    tile = next(t for t in S.tiles(r.matmul.shape) if (t.rows[0], t.cols[0]) == (0, 16))
    rows, cols = r.credited(tile)
    assert (len(rows), len(cols)) == (15, 15)
    assert W.tile_cap(S, r, [[0] * K] * 15, [[0] * K] * 15)[1] == S.cap * W.cell_credit(S, r) * 15 * 15


def test_records_bind_every_field_and_the_splits():
    splits = {0: W.Split(K)}
    base = W.records_root(b"s", b"w", splits, [record()])
    for other in (record(real=21), record(excluded_a=(1,)), record(excluded_b=(2,)), record(kept=120)):
        assert W.records_root(b"s", b"w", splits, [other]) != base
    assert W.records_root(b"s", b"w", {0: W.Split(K, (), b"c")}, [record()]) != base
    for bad in (dict(excluded_a=(25,)), dict(excluded_a=(3, 3)), dict(excluded_b=(32,)), dict(real=0), dict(real=33),
                dict(kept=129)):
        with pytest.raises(ValueError):
            record(**bad)


def test_weight_side_is_charged_once_per_weight():
    """F2: two units on one weight carry one weight side, outside W and the credit."""
    a, b = W.UnitRecord(Matmul(0, Shape(32, K, 32), 7), 32, K), W.UnitRecord(Matmul(1, Shape(16, K, 32), 7), 16, K)
    assert W.job_costs(S, [a, b]) == W.job_costs(S, [a]) == S.job_cost(32, K)
    assert W.work(S, [a, b])[0] == S.wref(Shape(48, K, 32))


def ops_for(rows_a, rows_b, k=K):
    ctx = CV.ctx_of("work", (16, k, 16), 0, 0)
    return S.encode(ctx, S.noise(ctx), rows_a, rows_b, Tile(tuple(range(16)), tuple(range(16))))


def test_tile_rule_reports_and_never_raises():
    rows_a, rows_b = CV.rows_f32("work", "A", 16, K), CV.rows_f32("work", "B", 16, K)
    tile = Tile(tuple(range(16)), tuple(range(16)))
    unit = lambda real=16, kept=K, ea=(), eb=(): W.UnitRecord(Matmul(0, Shape(16, K, 16), 0), real, kept, ea, eb)  # noqa: E731
    check = lambda rec, a, b: W.check_opened(S, rec, tile, a, b, ops_for(a, b))  # noqa: E731
    assert check(unit(), rows_a, rows_b) is None
    bad_a = rows_a[:5] + [spiked(rows_a[5])] + rows_a[6:]
    assert "A row 5 is credited but fails" in check(unit(), bad_a, rows_b)
    assert check(unit(ea=(5,)), bad_a, rows_b) is None
    assert "A row 7 is excluded but passes" in check(unit(ea=(5, 7)), bad_a, rows_b)
    assert "B row 15 is credited but fails" in check(unit(), rows_a, rows_b[:15] + [spiked(rows_b[15])])
    assert "past the kept width" in check(unit(kept=120), rows_a, rows_b)
    zero = C.f32_row([0] * K)
    assert check(unit(real=12), rows_a[:12] + [zero] * 4, rows_b) is None
    assert "filler row 13 is not zero" in check(unit(real=12), rows_a[:12] + [zero, rows_a[13], zero, zero], rows_b)
    assert "A row 12 is credited but fails" in check(unit(), rows_a[:12] + [zero] * 4, rows_b)
    assert "malformed" in W.check_opened(S, unit(), tile, rows_a[:15] + [b"\0" * 8], rows_b, ops_for(rows_a, rows_b))


def test_padding_stuffing_is_rejected():
    """F7: a row that fails the floor on its kept channels passes once values are written into the zero padding; the
    tile rule rejects any nonzero word past |K|."""
    k, kept = 256, 200
    rows_b = [C.f32_row(list(C.parse_f32_row(r, k))[:kept] + [0] * (k - kept)) for r in CV.rows_f32("pad", "B", 16, k)]
    base = [list(C.parse_f32_row(r, k))[:kept] + [0] * (k - kept) for r in CV.rows_f32("pad", "A", 16, k)]
    base[0][1] = P.f32_bits(1e6)
    stuffed = list(base[0])
    for p in range(kept, k, 8):
        stuffed[p] = P.f32_bits(3e4)
    assert not W.row_passes(base[0]) and W.row_passes(stuffed)
    rows_a = [C.f32_row(stuffed)] + [C.f32_row(r) for r in base[1:]]
    unit = W.UnitRecord(Matmul(0, Shape(16, k, 16), 0), 16, kept)
    tile = Tile(tuple(range(16)), tuple(range(16)))
    assert "A row 0 is not zero past the kept width" in W.check_opened(S, unit, tile, rows_a, rows_b, ops_for(rows_a, rows_b, k))


def lifecycle(rows_a=None, real=20, scheme=S):
    """32 x 128 x 16 on 16 x 16 tiles: A's rows 20 … 31 are zero filler and row 3 fails liveness; B passes."""
    if rows_a is None:
        rows_a = CV.rows_f32("work-life", "A", real, K) + [C.f32_row([0] * K)] * (32 - real)
        rows_a[3] = spiked(rows_a[3])
    rows_b = CV.rows_f32("work-life", "B", 16, K)
    work = Workload((Matmul(3, Shape(32, K, 16), 9),))
    prover = Prover(scheme, {9: rows_b})
    epoch = Epoch.unverified(work, prover.weight_roots, b"beacon")
    return epoch, prover, prover.prove(epoch, {3: rows_a}), rows_a, rows_b


SPLITS = {9: W.Split(K)}


def test_audit_credits_real_rows_and_checks_the_excluded_ones():
    epoch, prover, tr, rows_a, rows_b = lifecycle()
    records = W.declare(S, epoch, {3: rows_a}, {9: rows_b}, SPLITS, real_rows={3: 20})
    assert records == [W.UnitRecord(epoch.workload.matmuls[0], 20, K, (3,), ())]
    out = W.audit(S, epoch, prover, tr, SPLITS, records, BEFORE, AFTER, 8)
    assert out.audit.accepted and len(out.audit.verdicts) == 9 and out.audit.selected[-1] == 0
    assert out.wref == W.cell_wref(S, records[0]) * 19 * 16 and out.audit.profile.work == int(out.wref)
    assert out.credit == (1 - S.cap) * W.cell_credit(S, records[0]) * 19 * 16 and out.job == S.job_cost(16, K)


def test_r1_is_a_no_op_on_pearl_c():
    """R1, the salted per-row credit rule `check_opened` and `declare` now consult, is Pearl-C4's (its F1′ bound).
    Pearl-C (FP8) brings no hook, so an excluded row that passes the per-row rules still fails its tile, and declaring
    with the transcript gives the same records as without."""
    epoch, prover, tr, rows_a, rows_b = lifecycle()
    assert not hasattr(S, "row_over_debit") and not hasattr(S, "r1_rows")
    plain = W.declare(S, epoch, {3: rows_a}, {9: rows_b}, SPLITS, real_rows={3: 20})
    assert W.declare(S, epoch, {3: rows_a}, {9: rows_b}, SPLITS, real_rows={3: 20}, transcript=tr) == plain
    u = epoch.workload.matmuls[0]
    tile = S.tiles(u.shape)[0]
    rec = W.UnitRecord(u, 20, K, (0, 3), ())
    reason = W.check_opened(S, rec, tile, [rows_a[i] for i in tile.rows], [rows_b[j] for j in tile.cols], None)
    assert reason == "A row 0 is excluded but passes the per-row rules"


def test_pearl_c4s_per_element_term_3_is_a_no_op_on_pearl_c(monkeypatch):
    """Pearl-C4's debit (item 3 per element on the witness, `pearl_c4.tile_debit`) is its own: Pearl-C (FP8) brings no
    `tile_cap`, so its audit never reaches Pearl-C4's witness or debit and accepts as before."""
    from verity_pouw.schemes import pearl_c4

    def unreachable(*args, **kwargs):
        raise AssertionError("Pearl-C reached Pearl-C4's debit")

    monkeypatch.setattr(pearl_c4, "tile_debit", unreachable)
    monkeypatch.setattr(pearl_c4, "salt_dead", unreachable)
    epoch, prover, tr, rows_a, rows_b = lifecycle()
    assert not hasattr(S, "tile_cap")
    records = W.declare(S, epoch, {3: rows_a}, {9: rows_b}, SPLITS, real_rows={3: 20})
    assert W.audit(S, epoch, prover, tr, SPLITS, records, BEFORE, AFTER, 4).audit.accepted


def test_v_ex_is_a_no_op_on_pearl_c():
    """V-EX adds nothing to a declaration without voluntary rows: Pearl-C's records declare none, each encodes byte for
    byte in the v2 layout (so `records_root` is unchanged), and the audit accepts as before."""
    epoch, prover, tr, rows_a, rows_b = lifecycle()
    records = W.declare(S, epoch, {3: rows_a}, {9: rows_b}, SPLITS, real_rows={3: 20})
    for r in records:
        u, s, a, b = r.matmul, r.matmul.shape, r.excluded_a, r.excluded_b
        assert r.voluntary_a == r.voluntary_b == ()
        assert r.encode() == struct.pack(f"<QQIIIIIII{len(a)}I{len(b)}I", u.index, u.weight, s.m, s.k, s.n, r.real_rows,
                                         r.k_kept, len(a), len(b), *a, *b)
    assert W.audit(S, epoch, prover, tr, SPLITS, records, BEFORE, AFTER, 4).audit.accepted


def test_audit_under_hashing_h1():
    """The work law's audit is the same under hashing format -h1: its openings verify against frame-b3 roots."""
    h1 = C.PearlC(k_min=32, tile=(16, 16), forming="v1", hashing="h1")
    epoch, prover, tr, rows_a, rows_b = lifecycle(scheme=h1)
    records = W.declare(h1, epoch, {3: rows_a}, {9: rows_b}, SPLITS, real_rows={3: 20})
    assert W.audit(h1, epoch, prover, tr, SPLITS, records, BEFORE, AFTER, 4).audit.accepted
    assert not W.audit(S, epoch, prover, tr, SPLITS, records, BEFORE, AFTER, 4).audit.accepted


def test_audit_rejects_what_the_red_team_found():
    epoch, prover, tr, rows_a, rows_b = lifecycle()
    u = epoch.workload.matmuls[0]
    audit = lambda records, splits=SPLITS, **kw: W.audit(S, epoch, prover, tr, splits, records, kw.pop("before", BEFORE),  # noqa: E731
                                                         kw.pop("beacon", AFTER), 8, **kw)
    reasons = lambda out: " | ".join(v.reason for v in out.audit.verdicts if not v.ok)  # noqa: E731
    hides = audit([W.UnitRecord(u, 20, K)])
    assert not hides.audit.accepted and "A row 3 is credited but fails" in reasons(hides)
    false = audit([W.UnitRecord(u, 20, K, (3, 5))])
    assert not false.audit.accepted and "A row 5 is excluded but passes" in reasons(false)
    pads = audit([W.UnitRecord(u, 32, K, (3,))])
    assert not pads.audit.accepted and "is credited but fails" in reasons(pads)
    honest = [W.UnitRecord(u, 20, K, (3,))]
    assert "not at least 2 rounds after" in reasons(audit(honest, before=BEFORE + 3))
    assert "does not verify" in reasons(audit(honest, beacon=type(AFTER)(AFTER.round, bytes(48))))
    wide = {9: W.Split(K + 2, (7,), b"calib")}
    assert "not its weight's split" in reasons(audit(honest, splits=wide))
    narrow = {9: W.Split(K + 1, (7,), b"calib")}
    records = [W.UnitRecord(u, 20, K, (3,))]
    assert "not the pinned rule's" in reasons(audit(records, splits=narrow))
    assert "not the pinned rule's" in reasons(audit(records, splits=narrow, split_check=lambda w, sp: False))


def test_split_check_gates_an_honest_split():
    epoch, prover, tr, rows_a, rows_b = lifecycle()
    u = epoch.workload.matmuls[0]
    split = {9: W.Split(K + 1, (7,), b"calib")}
    records = [W.UnitRecord(u, 20, K, (3,))]
    seen = []
    out = W.audit(S, epoch, prover, tr, split, records, BEFORE, AFTER, 4, split_check=lambda w, sp: seen.append((w, sp)) or True)
    assert out.audit.accepted and seen == [(9, split[9])]


def test_aligned_spike_unit_is_rejected_by_the_cap():
    """F1's negative control through the audit: every row passes the domain, and the drawn tile's replayed debit (74%+
    of its atom-words) exceeds ρ·creditOf; the same unit without spikes is accepted."""
    k, t = 1024, 4
    s = C.PearlC(k_min=128, tile=(t, t), forming="v1")
    for spike, accepted in ((300.0, False), (None, True)):
        rows_a, rows_b = spiked_rows(k, t, 100, spike), spiked_rows(k, t, 900, spike)
        work = Workload((Matmul(0, Shape(t, k, t), 1),))
        prover = Prover(s, {1: rows_b})
        epoch = Epoch.unverified(work, prover.weight_roots, b"beacon")
        tr = prover.prove(epoch, {0: rows_a})
        splits = {1: W.Split(k)}
        records = W.declare(s, epoch, {0: rows_a}, {1: rows_b}, splits)
        assert records == [W.UnitRecord(work.matmuls[0], t, k)]
        out = W.audit(s, epoch, prover, tr, splits, records, BEFORE, AFTER, 1)
        assert out.audit.accepted == accepted
        if not accepted:
            assert "exceeds the cap" in out.audit.verdicts[0].reason
