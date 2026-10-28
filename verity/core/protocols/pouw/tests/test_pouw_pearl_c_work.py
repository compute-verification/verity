"""Pearl-C's work law (`pearl_c_work`): splits bounded by 1% of k, credit
on real, kept, unexcluded cells at Lean's accounting, the tile rule (zero filler and padding, credited rows pass and
excluded rows fail, the debit under the cap, and under -h3 each credited row's too), draws keyed by the epoch's draw coin
after the commitment, and the audit."""

from __future__ import annotations

import dataclasses
import struct
import sys
from fractions import Fraction
from pathlib import Path

import pytest

import epoch_coins as E

import pearl_c_vectors as CV
from test_pouw_pearl_c_debit import spiked_rows
from verity.core.protocols.pouw.audit import Epoch, Workload
from verity.core.protocols.pouw.protocol import Context, DomainError, Matmul, Shape, Tile
from verity.ml.protocols.pouw.schemes import get
from verity.ml.protocols.pouw.schemes import pearl_c as C
from verity.ml.protocols.pouw.schemes import pearl_c_work as W
from verity.ml.protocols.pouw.schemes import pearl_kw as P
from verity.ml.protocols.pouw.schemes.pearl_c_device import SM120

# the clear audit is a reference tool (benchmarks/pouw/clear), imported by name
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "benchmarks" / "pouw" / "clear"))
from clear_audit import Prover  # noqa: E402
import clear_work as CW  # noqa: E402

S = C.PearlC(k_min=32, tile=(16, 16), forming="v1")
K = 128


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
    cell = CW.cell_wref(S, record())
    assert cell == Fraction(S.wref(Shape(1, K, 32)), 32) and CW.cell_credit(S, record()) == Fraction(S.credit_of(Shape(1, K, 32)), 32)
    r = record(excluded_a=(3,), excluded_b=(20,))
    assert CW.tile_weight(S, r, tiles[(0, 0)]) == cell * 15 * 16 and CW.tile_weight(S, r, tiles[(0, 16)]) == cell * 15 * 15
    assert CW.tile_weight(S, r, tiles[(16, 16)]) == cell * 4 * 15 and CW.tile_weight(S, record(real=16), tiles[(16, 0)]) == 0
    wref, credit = CW.work(S, [r])
    assert wref == cell * 19 * 31 and credit == (1 - S.cap) * CW.cell_credit(S, r) * 19 * 31
    assert CW.work(S, [record(kept=64)])[0] == Fraction(S.wref(Shape(20, 64, 32)))
    picks = CW.draw(S, [r], CW.draw_key(b"a coin", bytes(32), bytes(32), bytes(32)), 200)
    assert {(t.rows[0], t.cols[0]) for _, t in picks} == set(tiles) and all(r.credited(t)[0] for _, t in picks)
    assert CW.draw(S, [r], CW.draw_key(b"another coin", bytes(32), bytes(32), bytes(32)), 200) != picks


def test_the_served_draw_weighs_mixed_linears_by_the_cited_statements_wref():
    """On sm_120 at its cap the tickets are `pearlCTilesDevRev1K`'s weights at `Prices.sm120Loop` and
    c = 218/25 − 8953/1000, Lean's definitions restated here in exact fractions, on both served models' four linears at
    once.  By the scheme's own `wref` they are not proportional (down is under-drawn); off sm_120 the draw is unchanged."""
    add, bf16, fs, qa, g, r, c = Fraction(1047, 125), 2, 40, 2, 4, 32, Fraction(218, 25) - Fraction(8953, 1000)

    def lean(m: int, n: int, k: int) -> Fraction:
        credit = m * k * n * (1 + add / (32 * g)) + m * k * r + 2 * bf16 * r * m * n + fs * m * k - add * m * n
        return credit + qa * m * k + c * m * k

    s = C.PearlC(forming="v1", device=SM120)
    assert CW.draw_wref(s) is CW.SM120_LOOP_CAST_8P72 and CW.draw_wref(S) is None
    linears = {"llama": [(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336)],
               "qwen": [(6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288)]}
    for nks in linears.values():
        records = [W.UnitRecord(Matmul(i, Shape(32, k, n), i), 32, k) for i, (n, k) in enumerate(nks)]
        assert [CW.ticket_cell(s, rec) for rec in records] == [lean(1, n, k) / n for n, k in nks]
        own = {CW.cell_wref(s, rec) * n / lean(1, n, k) for rec, (n, k) in zip(records, nks)}
        assert len(own) == len(nks) and min(own) == CW.cell_wref(s, records[3]) * nks[3][0] / lean(1, *nks[3])
    h = record(excluded_a=(3,), excluded_b=(20,))
    assert all(CW.tile_weight(S, h, t) == CW.cell_wref(S, h) * len(h.credited(t)[0]) * len(h.credited(t)[1])
               for t in S.tiles(h.matmul.shape))
    mixed = [W.UnitRecord(Matmul(0, Shape(64, 4096, 128), 0), 64, 4096), W.UnitRecord(Matmul(1, Shape(64, 14336, 64), 1), 40, 14336)]
    tk = CW.tickets(s, mixed)
    weight = {(p, t.rows[0], t.cols[0]): w for p, t, w in tk}
    assert Fraction(weight[(1, 0, 0)], weight[(0, 0, 0)]) == Fraction(40 * 64, 64 * 64) * lean(1, 64, 14336) / 64 / (lean(1, 128, 4096) / 128)


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
            _, cap = CW.tile_cap(scheme, r, [[0] * s.k] * len(rows), [[0] * s.k] * len(cols))
            assert cap == scheme.cap * scheme.credit_of(Shape(s.m, kept, s.n)) * Fraction(len(rows) * len(cols), s.m * s.n)
            assert cap < scheme.cap * scheme.credit_of(Shape(len(rows), kept, len(cols)))
            total += cap
        assert total == scheme.cap * scheme.credit_of(Shape(s.m, kept, s.n))
    assert sm120.credit_of(Shape(32, K, 32)) < S.credit_of(Shape(32, K, 32))
    r = record(excluded_a=(3,), excluded_b=(20,))
    tile = next(t for t in S.tiles(r.matmul.shape) if (t.rows[0], t.cols[0]) == (0, 16))
    rows, cols = r.credited(tile)
    assert (len(rows), len(cols)) == (15, 15)
    assert CW.tile_cap(S, r, [[0] * K] * 15, [[0] * K] * 15)[1] == S.cap * CW.cell_credit(S, r) * 15 * 15


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
    assert CW.job_costs(S, [a, b]) == CW.job_costs(S, [a]) == S.job_cost(32, K)
    assert CW.work(S, [a, b])[0] == S.wref(Shape(48, K, 32))


def ops_for(rows_a, rows_b, k=K):
    ctx = CV.ctx_of("work", (16, k, 16), 0, 0)
    return S.encode(ctx, S.noise(ctx), rows_a, rows_b, Tile(tuple(range(16)), tuple(range(16))))


def test_tile_rule_reports_and_never_raises():
    rows_a, rows_b = CV.rows_f32("work", "A", 16, K), CV.rows_f32("work", "B", 16, K)
    tile = Tile(tuple(range(16)), tuple(range(16)))
    unit = lambda real=16, kept=K, ea=(), eb=(): W.UnitRecord(Matmul(0, Shape(16, K, 16), 0), real, kept, ea, eb)  # noqa: E731
    check = lambda rec, a, b: CW.check_opened(S, rec, tile, a, b, ops_for(a, b))  # noqa: E731
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
    assert "malformed" in CW.check_opened(S, unit(), tile, rows_a[:15] + [b"\0" * 8], rows_b, ops_for(rows_a, rows_b))


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
    assert not CW.row_passes(base[0]) and CW.row_passes(stuffed)
    rows_a = [C.f32_row(stuffed)] + [C.f32_row(r) for r in base[1:]]
    unit = W.UnitRecord(Matmul(0, Shape(16, k, 16), 0), 16, kept)
    tile = Tile(tuple(range(16)), tuple(range(16)))
    assert "A row 0 is not zero past the kept width" in CW.check_opened(S, unit, tile, rows_a, rows_b, ops_for(rows_a, rows_b, k))


def lifecycle(rows_a=None, real=20, scheme=S, salt_key=None):
    """32 x 128 x 16 on 16 x 16 tiles: A's rows 20 … 31 are zero filler and row 3 fails liveness; B passes."""
    if rows_a is None:
        rows_a = CV.rows_f32("work-life", "A", real, K) + [C.f32_row([0] * K)] * (32 - real)
        rows_a[3] = spiked(rows_a[3])
    rows_b = CV.rows_f32("work-life", "B", 16, K)
    work = Workload((Matmul(3, Shape(32, K, 16), 9),))
    prover = Prover(scheme, {9: rows_b}, salt_key=salt_key)
    epoch = Epoch.unverified(work, prover.weight_roots, b"beacon", E.COINS)
    return epoch, prover, prover.prove(epoch, {3: rows_a}), rows_a, rows_b


SPLITS = {9: W.Split(K)}


def test_audit_credits_real_rows_and_checks_the_excluded_ones():
    epoch, prover, tr, rows_a, rows_b = lifecycle()
    records = CW.declare(S, epoch, {3: rows_a}, {9: rows_b}, SPLITS, real_rows={3: 20})
    assert records == [W.UnitRecord(epoch.workload.matmuls[0], 20, K, (3,), ())]
    out = CW.audit(S, epoch, prover, tr, SPLITS, records, E.challenger(), 8)
    assert out.audit.accepted and len(out.audit.verdicts) == 9 and out.audit.selected[-1] == 0
    assert out.wref == CW.cell_wref(S, records[0]) * 19 * 16 and out.audit.profile.work == int(out.wref)
    assert out.credit == (1 - S.cap) * CW.cell_credit(S, records[0]) * 19 * 16 and out.job == S.job_cost(16, K)


def test_an_excluded_weight_row_earns_nothing_and_its_tiles_are_accepted():
    """Where the work law and L3's tile check differ: a weight row that fails the per-row rules is excluded, and every
    tile holding its column is accepted with none of that column's cells credited, while L3's `TileCheck.Accepts` asks
    `weightOK` of each of a tile's columns and refuses the tile.  The work law is credit-sound there only while the
    excluded column's chain words (its C̃ and U words, computed and bound as any other's) earn nothing: W and the credit
    count 19 · 15 cells, not 19 · 16, and declaring the row credited fails its tile."""
    rows_a = CV.rows_f32("work-life", "A", 20, K) + [C.f32_row([0] * K)] * 12
    rows_a[3] = spiked(rows_a[3])
    rows_b = CV.rows_f32("work-life", "B", 16, K)
    rows_b[5] = spiked(rows_b[5])
    work = Workload((Matmul(3, Shape(32, K, 16), 9),))
    prover = Prover(S, {9: rows_b})
    epoch = Epoch.unverified(work, prover.weight_roots, b"beacon", E.COINS)
    tr = prover.prove(epoch, {3: rows_a})
    records = CW.declare(S, epoch, {3: rows_a}, {9: rows_b}, SPLITS, real_rows={3: 20})
    assert records == [W.UnitRecord(epoch.workload.matmuls[0], 20, K, (3,), (5,))]
    out = CW.audit(S, epoch, prover, tr, SPLITS, records, E.challenger(), 8)
    assert all(5 in t.cols for t in S.tiles(work.matmuls[0].shape))
    assert out.audit.accepted and len(out.audit.verdicts) == 9
    assert out.wref == CW.cell_wref(S, records[0]) * 19 * 15 and out.credit == (1 - S.cap) * CW.cell_credit(S, records[0]) * 19 * 15
    credited_b = W.UnitRecord(epoch.workload.matmuls[0], 20, K, (3,), ())
    rejected = CW.audit(S, epoch, prover, tr, SPLITS, [credited_b], E.challenger(), 8)
    assert not rejected.audit.accepted and any(v.reason == "B row 5 is credited but fails the per-row rules"
                                               for v in rejected.audit.verdicts)


@pytest.mark.slow
def test_a_padded_weights_last_tile_is_neither_drawn_nor_credited():
    """GLM-5.3's lm_head is 19360 rows a TP8 rank, not whole 64-column tiles; padded with zero rows to 19392 it is in
    Pearl-C's domain.  Its last tile holds real columns and zero rows, and a zero row fails the noise floor, so no tile
    check may see it (L3's `TileCheck.Accepts` asks `weightOK` of every column): the padded tile's rows are voluntary,
    which leaves it no credited cell, so no draw lands on it and it opens no exclusion tile.  Here 500 rows padded to
    504 on 8-column tiles (V-EX's ⌈n/64⌉ cap holds the tile's 8 rows): the credit is the 496 real columns of full tiles,
    under the real columns' honest credit.  Declaring the zero rows only as voluntary leaves them excluded, and their
    exclusion tile is the padded one."""
    s = C.PearlC(k_min=32, tile=(16, 8), forming="v1")
    rows_a = CV.rows_f32("work-pad", "A", 16, K)
    real = [r for r in CV.rows_f32("work-pad", "B", 504, K) if CW.row_passes(C.parse_f32_row(r, K))][:500]
    rows_b, padded = W.pad_weight(real, s.tile[1])
    assert rows_b[:500] == real and rows_b[500:] == [C.f32_row([0] * K)] * 4 and padded == tuple(range(496, 504))
    assert W.pad_weight(rows_b, 8) == (rows_b, ())
    with pytest.raises(DomainError):
        s.check(Matmul(3, Shape(16, K, 500), 9))
    with pytest.raises(ValueError):
        W.pad_weight([real[0], real[1][:8]], 8)
    work = Workload((Matmul(3, Shape(16, K, 504), 9),))
    prover = Prover(s, {9: rows_b})
    epoch = Epoch.unverified(work, prover.weight_roots, b"beacon", E.COINS)
    tr = prover.prove(epoch, {3: rows_a})
    records = CW.declare(s, epoch, {3: rows_a}, {9: rows_b}, SPLITS, padded_b={9: padded})
    assert records == [W.UnitRecord(work.matmuls[0], 16, K, (), (), (), padded)]
    assert [t.cols for _, t, w in CW.tickets(s, records) if w == 0] == [padded]
    key = CW.draw_key(b"coin", b"root", b"records", epoch.salt)
    assert all(t.cols != padded for _, t in CW.draw(s, records, key, 4096)) and CW.exclusion_tiles(s, records) == []
    out = CW.audit(s, epoch, prover, tr, SPLITS, records, E.challenger(), 16)
    assert out.audit.accepted
    assert out.credit == (1 - s.cap) * CW.cell_credit(s, records[0]) * 16 * 496
    assert out.credit < (1 - s.cap) * s.credit_of(Shape(16, K, 500))
    only_voluntary = CW.declare(s, epoch, {3: rows_a}, {9: rows_b}, SPLITS, voluntary_b={9: padded})
    assert only_voluntary[0].excluded_b == tuple(range(500, 504))
    assert [t.cols for _, t in CW.exclusion_tiles(s, only_voluntary)] == [padded]


def test_r1_is_a_no_op_on_pearl_c():
    """R1, the salted per-row credit rule `check_opened` and `declare` now consult, is Pearl-C4's (its F1′ bound).
    Pearl-C (FP8) brings no hook, so an excluded row that passes the per-row rules still fails its tile, and declaring
    with the transcript gives the same records as without."""
    epoch, prover, tr, rows_a, rows_b = lifecycle()
    assert not hasattr(S, "row_over_debit") and not hasattr(S, "r1_rows")
    plain = CW.declare(S, epoch, {3: rows_a}, {9: rows_b}, SPLITS, real_rows={3: 20})
    assert CW.declare(S, epoch, {3: rows_a}, {9: rows_b}, SPLITS, real_rows={3: 20}, transcript=tr) == plain
    u = epoch.workload.matmuls[0]
    tile = S.tiles(u.shape)[0]
    rec = W.UnitRecord(u, 20, K, (0, 3), ())
    reason = CW.check_opened(S, rec, tile, [rows_a[i] for i in tile.rows], [rows_b[j] for j in tile.cols], None)
    assert reason == "A row 0 is excluded but passes the per-row rules"


def test_pearl_c4s_per_element_term_3_is_a_no_op_on_pearl_c(monkeypatch):
    """Pearl-C4's debit (item 3 per element on the witness, `pearl_c4.tile_debit`) is its own: Pearl-C (FP8) brings no
    `tile_cap`, so its audit never reaches Pearl-C4's witness or debit and accepts as before."""
    from verity.ml.protocols.pouw.schemes import pearl_c4

    def unreachable(*args, **kwargs):
        raise AssertionError("Pearl-C reached Pearl-C4's debit")

    monkeypatch.setattr(pearl_c4, "tile_debit", unreachable)
    monkeypatch.setattr(pearl_c4, "salt_dead", unreachable)
    epoch, prover, tr, rows_a, rows_b = lifecycle()
    assert not hasattr(S, "tile_cap")
    records = CW.declare(S, epoch, {3: rows_a}, {9: rows_b}, SPLITS, real_rows={3: 20})
    assert CW.audit(S, epoch, prover, tr, SPLITS, records, E.challenger(), 4).audit.accepted


def test_v_ex_is_a_no_op_on_pearl_c():
    """V-EX adds nothing to a declaration without voluntary rows: Pearl-C's records declare none, each encodes byte for
    byte in the v2 layout (so `records_root` is unchanged), and the audit accepts as before."""
    epoch, prover, tr, rows_a, rows_b = lifecycle()
    records = CW.declare(S, epoch, {3: rows_a}, {9: rows_b}, SPLITS, real_rows={3: 20})
    for r in records:
        u, s, a, b = r.matmul, r.matmul.shape, r.excluded_a, r.excluded_b
        assert r.voluntary_a == r.voluntary_b == ()
        assert r.encode() == struct.pack(f"<QQIIIIIII{len(a)}I{len(b)}I", u.index, u.weight, s.m, s.k, s.n, r.real_rows,
                                         r.k_kept, len(a), len(b), *a, *b)
    assert CW.audit(S, epoch, prover, tr, SPLITS, records, E.challenger(), 4).audit.accepted


def test_audit_under_hashing_h3():
    """The work law's audit is the same under hashing format -h3: its openings verify against hm96-sha512 roots, once
    the honest prover has given up (`volunteer`) the A row over its row cap.  Each row's seed reads the hiding roots, so
    which rows pass the row cap changes with the salts; under this salt key row 4 is over it, and a drawn tile holding
    it fails until it is given up."""
    h3 = C.PearlC(k_min=32, tile=(16, 16), forming="v1", hashing="h3")
    epoch, prover, tr, rows_a, rows_b = lifecycle(scheme=h3, salt_key=bytes([10]) * 32)
    declared = CW.declare(h3, epoch, {3: rows_a}, {9: rows_b}, SPLITS, real_rows={3: 20})
    out = CW.audit(h3, epoch, prover, tr, SPLITS, declared, E.challenger(), 4).audit
    assert not out.accepted and any(v.reason.startswith("A row 4's replayed debit") for v in out.verdicts)
    va, vb = CW.volunteer(h3, epoch, tr, declared, {3: rows_a}, {9: rows_b})
    assert (va, vb) == ({3: [4]}, {})
    records = CW.declare(h3, epoch, {3: rows_a}, {9: rows_b}, SPLITS, real_rows={3: 20}, voluntary_a=va)
    assert CW.audit(h3, epoch, prover, tr, SPLITS, records, E.challenger(), 4).audit.accepted
    assert not CW.audit(S, epoch, prover, tr, SPLITS, records, E.challenger(), 4).audit.accepted


def test_audit_rejects_what_the_red_team_found():
    epoch, prover, tr, rows_a, rows_b = lifecycle()
    u = epoch.workload.matmuls[0]
    audit = lambda records, splits=SPLITS, **kw: CW.audit(S, epoch, prover, tr, splits, records,  # noqa: E731
                                                         kw.pop("challenger", None) or E.challenger(), 8, **kw)
    reasons = lambda out: " | ".join(v.reason for v in out.audit.verdicts if not v.ok)  # noqa: E731
    hides = audit([W.UnitRecord(u, 20, K)])
    assert not hides.audit.accepted and "A row 3 is credited but fails" in reasons(hides)
    false = audit([W.UnitRecord(u, 20, K, (3, 5))])
    assert not false.audit.accepted and "A row 5 is excluded but passes" in reasons(false)
    pads = audit([W.UnitRecord(u, 32, K, (3,))])
    assert not pads.audit.accepted and "is credited but fails" in reasons(pads)
    honest = [W.UnitRecord(u, 20, K, (3,))]
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
    out = CW.audit(S, epoch, prover, tr, split, records, E.challenger(), 4, split_check=lambda w, sp: seen.append((w, sp)) or True)
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
        epoch = Epoch.unverified(work, prover.weight_roots, b"beacon", E.COINS)
        tr = prover.prove(epoch, {0: rows_a})
        splits = {1: W.Split(k)}
        records = CW.declare(s, epoch, {0: rows_a}, {1: rows_b}, splits)
        assert records == [W.UnitRecord(work.matmuls[0], t, k)]
        out = CW.audit(s, epoch, prover, tr, splits, records, E.challenger(), 1)
        assert out.audit.accepted == accepted
        if not accepted:
            assert "exceeds the cap" in out.audit.verdicts[0].reason


def test_h3_caps_each_credited_row_as_well_as_the_tile():
    """-h3's row cap (`TTOutRowSeed.lean`'s (C2)), its seeds keyed on SHA-512 row digests: one A row with the aligned spikes
    among 31 Gaussian rows, on spiked B, is over 10x its own cap while the 32-row tile's debit is under 3/4 of the tile's.
    h0's tile rule accepts the tile and -h3's audit refuses it for that row; V-EX gives the row up, and the audit then
    accepts on the other 31 rows."""
    k, m, n = 1024, 32, 4
    h3, h0 = (C.PearlC(k_min=128, tile=(m, n), forming="v1", hashing=h) for h in ("h3", "h0"))
    assert CW.row_cap(h3) and not CW.row_cap(h0) and not CW.row_cap(S)
    rows_a = spiked_rows(k, 1, 100, 300.0) + spiked_rows(k, m - 1, 200, None)
    rows_b = spiked_rows(k, n, 900, 300.0)
    work = Workload((Matmul(0, Shape(m, k, n), 1),))
    splits = {1: W.Split(k)}
    prover = Prover(h3, {1: rows_b}, salt_key=bytes(32))
    epoch = Epoch.unverified(work, prover.weight_roots, b"beacon", E.COINS)
    tr = prover.prove(epoch, {0: rows_a})
    records = CW.declare(h3, epoch, {0: rows_a}, {1: rows_b}, splits)
    assert records == [W.UnitRecord(work.matmuls[0], m, k)]
    tile = h3.tiles(work.matmuls[0].shape)[0]
    ctx = Context(epoch.salt, work.matmuls[0], tr.activation_roots[0], epoch.weight_roots[1])
    for s in (h3, h0):
        ops = s.encode(ctx, s.noise(ctx), rows_a, rows_b, tile)
        debit, cap = CW._debit_cap(s, records[0], tile, ops, list(range(m)), list(range(n)))
        row = CW._debit_cap(s, records[0], tile, ops, [0], list(range(n)))
        assert debit < cap * 3 / 4 and row[0] > 10 * row[1] and row[1] == cap / m
    assert CW.check_opened(h0, records[0], tile, rows_a, rows_b, ops) is None
    out = CW.audit(h3, epoch, prover, tr, splits, records, E.challenger(), 1)
    assert not out.audit.accepted
    assert out.audit.verdicts[0].reason.startswith("A row 0's replayed debit") and "the row cap" in out.audit.verdicts[0].reason
    va, vb = CW.volunteer(h3, epoch, tr, records, {0: rows_a}, {1: rows_b})
    assert (va, vb) == ({0: [0]}, {})
    given = CW.declare(h3, epoch, {0: rows_a}, {1: rows_b}, splits, voluntary_a=va)
    again = CW.audit(h3, epoch, prover, tr, splits, given, E.challenger(), 1)
    assert again.audit.accepted and again.credit == (1 - h3.cap) * CW.cell_credit(h3, given[0]) * (m - 1) * n


def _census_tile(scheme, spikes: list[float | None], k: int, n: int = 4):
    """One m x n tile at depth ``k``, A row i carrying ``spikes[i]`` (`spiked_rows`), on spiked B: (record, tile, rows_a,
    rows_b, ops) under the scheme's own `encode`."""
    m = len(spikes)
    rows_a = [spiked_rows(k, 1, 100 + i, s)[0] for i, s in enumerate(spikes)]
    rows_b = spiked_rows(k, n, 900, 300.0)
    work = Workload((Matmul(0, Shape(m, k, n), 1),))
    prover = Prover(scheme, {1: rows_b}, salt_key=bytes(32))
    epoch = Epoch.unverified(work, prover.weight_roots, b"beacon", E.COINS)
    tr = prover.prove(epoch, {0: rows_a})
    tile = scheme.tiles(work.matmuls[0].shape)[0]
    ctx = Context(epoch.salt, work.matmuls[0], tr.activation_roots[0], epoch.weight_roots[1])
    return (W.UnitRecord(work.matmuls[0], m, k), tile, rows_a, rows_b,
            scheme.encode(ctx, scheme.noise(ctx), rows_a, rows_b, tile))


def test_census_k_caps_each_row_at_four_rho_and_the_tile_at_rho():
    """At the H100 record's census k = 256, -h3 caps each credited row's debit at 4ρ of its
    cells' credit (`pearlCTilesDevRev1RowCapAt` at ρr = 1/100) and the tile's at ρ.  A row at 3.66ρ in a tile at 0.30ρ is
    accepted (the ρ row cap refused it), a row at 10.4ρ is refused by the 4ρ row cap, and 16 rows each under 4ρ that
    put the tile over ρ are refused by the tile cap."""
    m = 16
    h3 = C.PearlC(tile=(m, 4), forming="v1", hashing="h3")
    assert h3.census_ks == (256,) and h3.row_cap_at(256) == 4 * h3.cap == Fraction(1, 100)
    assert CW.row_cap(h3) and CW.row_cap_multiple(h3, 256) == 4
    rec, tile, a, b, ops = _census_tile(h3, [50.0] + [None] * (m - 1), 256)
    debit, cap = CW._debit_cap(h3, rec, tile, ops, list(range(m)), [0, 1, 2, 3])
    row = CW._debit_cap(h3, rec, tile, ops, [0], [0, 1, 2, 3])
    assert debit < cap / 2 and row[1] < row[0] < 4 * row[1]
    assert CW.check_opened(h3, rec, tile, a, b, ops) is None
    rec, tile, a, b, ops = _census_tile(h3, [100.0] + [None] * (m - 1), 256)
    reason = CW.check_opened(h3, rec, tile, a, b, ops)
    assert reason.startswith("A row 0's replayed debit") and "(the row cap: 4ρ·its credited cells' credit)" in reason
    spikes = [50.0, 50.0, 50.0, 50.0, 30.0, 20.0, 20.0, 30.0, 30.0, None, 10.0, 20.0, 30.0, 20.0, None, 30.0]
    rec, tile, a, b, ops = _census_tile(h3, spikes, 256)
    rows = [CW._debit_cap(h3, rec, tile, ops, [i], [0, 1, 2, 3]) for i in range(m)]
    debit, cap = CW._debit_cap(h3, rec, tile, ops, list(range(m)), [0, 1, 2, 3])
    assert all(d <= 4 * c for d, c in rows) and debit > 2 * cap
    assert CW.check_opened(h3, rec, tile, a, b, ops).startswith("the replayed debit")


def test_census_row_cap_leaves_k_from_k_min_unchanged():
    """At every k >= k_min the row cap stays ρ, byte for byte: the multiple is 1 on the census scheme and on every scheme
    without census ks, and at k = 1,024 a row over ρ is refused with the (C2) message, while at k = 256 the same row
    shape, under 4ρ, passes."""
    m = 16
    h3 = C.PearlC(tile=(m, 4), forming="v1", hashing="h3")
    for k in (1024, 2048, 8192, 1 << 16):
        assert h3.row_cap_at(k) == h3.cap and CW.row_cap_multiple(h3, k) == 1
    for s in (S, C.PearlC(k_min=32, tile=(16, 16), forming="v1", hashing="h3"), get("pearl-c-sm120-v1-h3")):
        assert s.census_ks == () and all(CW.row_cap_multiple(s, k) == 1 for k in (128, 256, 1024))
    rec, tile, a, b, ops = _census_tile(h3, [30.0] + [None] * (m - 1), 1024)
    d, c = CW._debit_cap(h3, rec, tile, ops, [0], [0, 1, 2, 3])
    assert c < d < 4 * c
    assert CW.check_opened(h3, rec, tile, a, b, ops) == (
        f"A row 0's replayed debit {float(d):.0f} exceeds its cap {float(c):.0f} (the row cap: ρ·its credited cells' "
        "credit)")


def test_the_draw_coin_must_open_against_the_epochs_commitment_after_the_commitment():
    """The draw coin is the epoch's, asked of the challenger once the verifier recorded the transcript and records roots
    (`draw_after`): a foreign challenger's coin, a coin not under the commitment, the epoch's salt coin, a coin opened
    before the commitment, another epoch's coin and an epoch with no coin commitment are each refused before any tile
    is opened; the honest challenger's coin draws the tiles `draw_key` gives."""
    epoch, prover, tr, rows_a, rows_b = lifecycle()
    records = CW.declare(S, epoch, {3: rows_a}, {9: rows_b}, SPLITS, real_rows={3: 20})
    digest = W.records_root(epoch.salt, epoch.weights_root, SPLITS, records)
    after = CW.draw_after(tr.commitment.root, digest)
    honest = E.challenger().open("draw", after)
    out = CW.audit(S, epoch, prover, tr, SPLITS, records, E.challenger(), 8)
    want = CW.draw(S, records, CW.draw_key(honest.coin, tr.commitment.root, digest, epoch.salt), 8)
    assert out.audit.accepted and out.audit.selected[:8] == tuple(
        next(r.id for r in CW.Verifier(S, epoch).refs if (r.matmul, r.tile) == (p, t)) for p, t in want)
    refused = (
        ("coin does not open against the epoch's coin commitment", E.challenger(b"a foreign challenger")),
        ("coin does not open against the epoch's coin commitment", E.Answering(dataclasses.replace(honest, coin=bytes(32)))),
        ("coin is the epoch's 'salt' coin, not its 'draw' coin", E.Answering(E.challenger().open("salt", after))),
        ("coin was not opened after what the verifier recorded", E.Answering(E.challenger().open("draw", b"before"))),
        ("coin is epoch 8's, not epoch 7's", E.challenger(epoch=8)),
    )
    for reason, challenger in refused:
        out = CW.audit(S, epoch, prover, tr, SPLITS, records, challenger, 8)
        assert not out.audit.accepted and out.audit.verdicts == (CW.Verdict(-1, False, "the draw's " + reason),)
    bare = Epoch.unverified(epoch.workload, epoch.weight_roots, b"beacon")
    out = CW.audit(S, bare, prover, tr, SPLITS, records, E.challenger(), 8)
    assert not out.audit.accepted and "no coin commitment" in out.audit.verdicts[0].reason


def test_a_challenger_opens_each_coin_once():
    """No retry redraws: the second request for the draw coin is refused, whatever it follows."""
    ch = E.challenger()
    ch.open("draw", b"a commitment")
    with pytest.raises(DomainError, match="already open"):
        ch.open("draw", b"another commitment")
