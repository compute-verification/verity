"""Pearl-C's work sampling design and its clear audit: what the verifier credits, samples and checks, all recomputed by the verifier
(statement review item 5; red team F1, F2 and F5–F8).  It is a reference tool, not a verifier: PoUW's served path is the
served-ZK window, and its verifier is the Lean (`verity/protocols/pouw/verifier/lean/check`).  It runs the records
core keeps (`verity.protocols.pouw.prover.schemes.pearl_c_work`: `Split`, `UnitRecord`, `records_root`), which vLLM's
Pearl-C prover writes, through the clear audit (`clear_audit`).  This directory is not a package: callers put it on
``sys.path``.

* **The one exclusion rule (per row).**
  - An excluded row stays in the unit, and its C̃ and U words are computed and bound as any other's.
  - Its cells earn no credit. The verifier checks that it fails the rules: on every sampled tile that holds it, and on one
    opened tile per excluded row, every one.
  - A credited row must pass the rules.
* **Credit.**
  - A cell's sample weight is its row's share of `W_ref` at |K|, with no weight side (F2), so padding to `k_unit` and
    filler earn nothing. `W_ref` is the one the sample's Lean statement weighs by (`draw_wref`): on sm_120 at its inert-work bound
    1/1,000, `wrefDevRev1K` at `Prices.sm120Loop` and the honest cast at 8.72 (`SM120_LOOP_CAST_8P72`), so that a served
    step's mixed linears are sampled in proportion to it; elsewhere the scheme's own `wref`, Lean's
    `wrefDevRev1` at its record.  W, the profile's work, stays the scheme's own `wref`.
  - An accepted unit is credited (1 − ρ)·`creditDevRev1` over its credited cells, at the scheme's inert-work bound ρ (`scheme.cap`:
    Pearl-C's is its device record's, 1/400 on the H100 and 1/1,000 on sm_120; Pearl-C4's is its own 1/400).
* **The tile rule (`check_opened`)**, beside `clear_audit.Verifier.check_tile`'s openings and digest:
  - filler rows are zero;
  - every opened row is zero past |K| (F7);
  - credited rows pass the per-row rules, and excluded rows fail them;
  - the inert work replayed over the credited words (`pearl_c_debit`), their flagged atom-words, later +0 promotions and
    P3's forming elements (fs·|cols|/n each), is at most the inert-work share ρ times the credited cells' credit, their
    share of the unit's `creditDevRev1` (F1; `tile_cap`);
  - under a hashing format with the row bound (-h3, `row_cap`), so is each credited A row's inert work over the credited columns,
    at most ρ times its cells' credit (`TTOutRowSeed.lean`'s (C2)), and at most 4ρ at Pearl-C's census ks below k_min
    (`row_cap_multiple`, `pearlCTilesDevRev1RowCapAt`), the tile's bound staying ρ.
  A failure is a wrong tile, reported as a reason; nothing raises.
* **Voluntary rows.**
  - A record may give up up to ⌈m/64⌉ A rows (`voluntary_a`), and a weight up to ⌈n/64⌉ B rows (`voluntary_b`, the same
    in every unit of the weight). They are committed in `records_root` with the excluded rows, before the sampling coin.
  - A voluntary row earns no credit, and by the band rule (`UnitRecord.credited`) neither does the rest of its band of 64
    A rows or, for a B row, its block of 64 columns; an excluded B row takes its block's credit too. At the deployed
    64 × 64 tile, a tile holding one has no ticket and no credit, and every ticketed tile's credited cells are the game's
    passing rows × all its columns. Its words are still formed, checked and hashed on every sampled tile that holds it.
  - It is exempt from the two-sided per-row check and R1, and it opens no exclusion tile.
  - The tile's inert-work bound and R1 read credited rows only, as before. Giving up credit never helps a cheater, and the declaration
    can't adapt to the sample. `volunteer` is an honest prover's remedy for a tile over the inert-work bound.
* **The sampling guarantee is per credited cell.** The samples are weighted by the credited cells' `wref`, and a sampled
  tile's uncredited cells (filler, excluded, voluntary, or in a voluntary row's band or block) never count as coverage.
  Volunteering rows only lowers the weight of the tiles it reaches to their credited cells, which is exactly the bound
  claimed.
* **The samples (F8)** are keyed by the epoch's sampling coin, through `verity.commitments.randomness.derive`.
  - `audit` asks the challenger for the coin once the verifier has recorded the prover's commitment (the transcript root
    and the records root, `draw_after`), and checks it against the epoch's coin commitment (`verity.protocols.pouw.verifier.coins`).
    A coin that doesn't open against it, or that was opened after anything else, is refused.
  - `audit_replay` keeps the unverified sample of a recorded run (`RecordedDraw`), for replays whose samples are Fiat–Shamir
    on the commitment. It supports no claim about the sample.
  - `audit` checks the splits through `split_check`, the pinned rule that computes S (`split_channels`, with its
    tie-break and budget), which lives with the model integration.
* **Other row formats.** A scheme with its own row encoding, per-row rules and count of inert work (`pearl_c4`) brings
  `parse_row(row, k)`, `row_passes(words)` and `tile_cap(record, ops, rows, cols)`; Pearl-C v1's are this module's. Every
  scheme brings its `cap`.
"""

from __future__ import annotations

import copy
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import partial
from itertools import accumulate
from math import floor, lcm

from clear_audit import Audit, Prover, TileOpening, Transcript, Verdict, Verifier, WorkProfile, commitment_hash, draw_coin
from verity.protocols.pouw.verifier.audit import Epoch, opened_value
from verity.protocols.pouw.verifier.coins import Challenger
from verity.protocols.pouw.verifier.protocol import Context, Shape, Tile, tagged
from verity.protocols.pouw.prover.schemes import pearl_c as C
from verity.protocols.pouw.prover.schemes import pearl_c_debit as D
from verity.protocols.pouw.prover.schemes import pearl_kw as P
from verity.protocols.pouw.prover.schemes.pearl_c_device import SM120, Prices
from verity.protocols.pouw.prover.schemes.pearl_c_work import ALIGN, BAND, Split, UnitRecord, records_root
from verity.protocols.pouw.verifier.serving import HASH_FORMATS
from verity.commitments.randomness import Key, derive

DRAW_DOMAIN = "verity/pouw/pearl-c-draw/v1"
DRAW_AFTER_TAG = "verity/pouw/pearl-c/draw-after/v1"


#: V-EX's count limits: at most ⌈m/64⌉ voluntary A rows per unit and ⌈n/64⌉ voluntary B rows per weight.
VOLUNTARY_PER = 64


def voluntary_cap(rows: int) -> int:
    return -(-rows // VOLUNTARY_PER)


def check_records(epoch: Epoch, splits: Mapping[int, Split], records: Sequence[UnitRecord]) -> str | None:
    """The records against the workload and the splits: one per matmul in order, k = the split's `k_unit`, |K| its
    `k_kept`, one set of excluded and one of voluntary B rows per weight, and V-EX's count limits."""
    if [r.matmul for r in records] != list(epoch.workload.matmuls):
        return "the records are not the workload's matmuls in order"
    if set(splits) != set(epoch.workload.weights()):
        return "every weight has a split, and only those"
    excluded_b: dict[int, tuple[int, ...]] = {}
    voluntary_b: dict[int, tuple[int, ...]] = {}
    for r in records:
        sp, s = splits[r.matmul.weight], r.matmul.shape
        if (s.k, r.k_kept) != (sp.k_unit, sp.k_kept):
            return f"matmul {r.matmul.index}: k and |K| are not its weight's split"
        if excluded_b.setdefault(r.matmul.weight, r.excluded_b) != r.excluded_b:
            return f"weight {r.matmul.weight}: its units declare different excluded rows"
        if voluntary_b.setdefault(r.matmul.weight, r.voluntary_b) != r.voluntary_b:
            return f"weight {r.matmul.weight}: its units declare different voluntary rows"
        if len(r.voluntary_a) > voluntary_cap(s.m) or len(r.voluntary_b) > voluntary_cap(s.n):
            return (f"matmul {r.matmul.index}: more voluntary rows than V-EX allows (⌈m/64⌉ = {voluntary_cap(s.m)} of A, "
                    f"⌈n/64⌉ = {voluntary_cap(s.n)} of B)")
    return None


def cell_wref(scheme: C.PearlC, record: UnitRecord) -> Fraction:
    """A cell's share of its row's `wref` at |K| (Lean's `wrefDevRev1` is linear in m)."""
    s = record.matmul.shape
    return Fraction(scheme.wref(Shape(1, record.k_kept, s.n)), s.n)


def cell_credit(scheme: C.PearlC, record: UnitRecord) -> Fraction:
    s = record.matmul.shape
    return Fraction(scheme.credit_of(Shape(1, record.k_kept, s.n)), s.n)


@dataclass(frozen=True)
class DrawWref:
    """The `W_ref` a Lean statement's sample weighs by: `wrefDevRev1K (devAt d prices) c`, the scheme's rev1 `wref` at the
    statement's prices plus `c` per activation element (the honest kernel's forming beyond the record's).  `GGapSampled`,
    `HonestTileCapRate` and `CompleteSampled` take tickets proportional to a tile's `Wref`, over all of its rows and
    columns, so a sample by another price record's `wref` attains the statement's gap only on a layout of one (n, k): on a
    served step's four linears the per-linear ratios differ.  `tile_weight` counts credited cells
    only, so the tickets are the statement's on fully credited tiles; a tile with filler, excluded or voluntary rows
    gets less than its `Wref`'s ticket at the scale that fully credited tiles get theirs, so unless every tile's
    credited share is the same, no Lean statement covers that sample yet."""

    statement: str
    device: str
    cap: Fraction
    prices: Prices
    c: Fraction


#: The statement Pearl-C v1's served gap and ε cite on sm_120 at its inert-work bound 1/1,000 (`ServedMixedRev1Gap`, `Completeness`'s
#: `HonestTileCapRate`): `Prices.sm120Loop` ⟨1047/125, 2, ⟨40, 2⟩⟩ and the honest cast at 218/25 against the record's
#: 8953/1000 (the credited cast 8 and the A-only forming's measured 1.047, rounded up to qa = 2).
SM120_LOOP_CAST_8P72 = DrawWref("pearlCSampledSm120v1LoopCast8p72Rev1Cap1000", SM120.name, SM120.cap,
                                replace(SM120.prices, fadd=Fraction(1047, 125), forming_a_only=2),
                                Fraction(218, 25) - Fraction(8953, 1000))
DRAW_WREF = {(SM120.name, SM120.prices, SM120.cap): SM120_LOOP_CAST_8P72}


def draw_wref(scheme: C.PearlC) -> DrawWref | None:
    """The statement the scheme's sample weighs by, or None for the scheme's own `wref` (Lean's at its record, c = 0)."""
    if type(scheme) is not C.PearlC or scheme.forming != "v1":
        return None
    return DRAW_WREF.get((scheme.device.name, scheme.prices, scheme.cap))


def ticket_cell(scheme: C.PearlC, record: UnitRecord) -> Fraction:
    """A credited cell's sample weight: its row's share of the sampled statement's `W_ref` at |K| (`draw_wref`)."""
    at = draw_wref(scheme)
    if at is None:
        return cell_wref(scheme, record)
    priced = copy.copy(scheme)
    priced.prices = at.prices
    s = record.matmul.shape
    return (Fraction(priced.wref(Shape(1, record.k_kept, s.n))) + at.c * record.k_kept) / s.n


def tile_weight(scheme: C.PearlC, record: UnitRecord, tile: Tile) -> Fraction:
    rows, cols = record.credited(tile)
    return ticket_cell(scheme, record) * len(rows) * len(cols)


def tickets(scheme: C.PearlC, records: Sequence[UnitRecord]) -> list[tuple[int, Tile, int]]:
    """(record position, tile, integer ticket) for every tile of every record, by its credited cells' `W_ref` at the
    statement the sample cites (`draw_wref`)."""
    weights = [(p, t, tile_weight(scheme, r, t)) for p, r in enumerate(records) for t in scheme.tiles(r.matmul.shape)]
    den = lcm(*(w.denominator for _, _, w in weights)) if weights else 1
    return [(p, t, int(w * den)) for p, t, w in weights]


def work(scheme: C.PearlC, records: Sequence[UnitRecord]) -> tuple[Fraction, Fraction]:
    """(W, credit): the credited cells' `wref`, and (1 − ρ)·`creditDevRev1` over them."""
    cells = [(r, len(rows) * len(cols)) for r in records for rows, cols in (r.credited(t) for t in scheme.tiles(r.matmul.shape))]
    return (sum((cell_wref(scheme, r) * c for r, c in cells), Fraction(0)),
            (1 - scheme.cap) * sum((cell_credit(scheme, r) * c for r, c in cells), Fraction(0)))


def job_costs(scheme: C.PearlC, records: Sequence[UnitRecord]) -> int:
    """The weight side, once per weight of the epoch however many units read it (outside W and the credit)."""
    shapes = {r.matmul.weight: r.matmul.shape for r in records}
    return sum(scheme.job_cost(s.n, s.k) for s in shapes.values())


# ---- the tile rule ----------------------------------------------------------------------------------------------------

def row_passes(words: Sequence[int]) -> bool:
    """The scheme's per-row rules: the noise floor ρ·α ≥ 1 (exact product) and `pearl_c.live_row`."""
    _, rho, alpha, _ = C.s5_stats(words)
    return P.f32_of_bits(rho) * P.f32_of_bits(alpha) >= P.SIGMA_MIN and C.live_row(words)


def tile_cap(scheme: C.PearlC, record: UnitRecord, a_codes: Sequence[Sequence[int]], b_codes: Sequence[Sequence[int]],
             formed: Sequence[tuple[int, Sequence[int], int]]) -> tuple[Fraction, Fraction]:
    """(inert work, its bound) over credited words given as their A′ and B̃ code rows, ``formed`` each A row's forming (α, x, β): the
    replayed inert work, rev1's `tileDebitRev1` (`pearl_c_debit.tile_debit`), and its bound, the inert-work share ρ
    times the credited cells' credit, the unit's `creditDevRev1` at |K| times their share |rows|·|cols|/(m·n) (the
    per-tile statement's `tileShare`, `pearlCTilesDevRev1`).  A row's own terms (its forming and A′·F_B) so count once
    across its row band, a |cols|/n share in each tile, not whole in every tile: an inert forming element is
    fs·|cols|/n (`tileDebitDev`), so the test is ρ⁻¹·(n·D + fs·|cols|·E) ≤ `credit_of(1, |K|, n)`·|rows|·|cols|."""
    d, n = scheme.device, record.matmul.shape.n
    atoms, zeros, elements = D.tile_debit(a_codes, b_codes, formed, d)
    debit = D.debit_units(atoms, Fraction(len(b_codes) * elements, n), scheme.prices, d, zeros=zeros)
    return debit, scheme.cap * cell_credit(scheme, record) * len(a_codes) * len(b_codes)


def _rules(scheme) -> tuple[Callable[[bytes, int], list[int]], Callable[[Sequence[int]], bool]]:
    """(parse_row, row_passes): the scheme's own when it brings them, else Pearl-C v1's."""
    return getattr(scheme, "parse_row", C.parse_f32_row), getattr(scheme, "row_passes", row_passes)


def check_opened(scheme: C.PearlC, record: UnitRecord, tile: Tile, a_rows: Sequence[bytes], b_rows: Sequence[bytes],
                 ops: C.Ops | None) -> str | None:
    """The tile rule on an opened tile whose openings and digest have passed: None, or the reason it fails (its rows,
    `rows_failure`, then its inert-work bound, `cap_failure`)."""
    return (rows_failure(scheme, record, tile, a_rows, b_rows, ops)
            or cap_failure(scheme, record, tile, ops, getattr(scheme, "r1_tolerance", 0)))


def rows_failure(scheme: C.PearlC, record: UnitRecord, tile: Tile, a_rows: Sequence[bytes], b_rows: Sequence[bytes],
                 ops: C.Ops | None) -> str | None:
    """`check_opened`'s rows: None, or the reason one of the tile's rows fails its declaration."""
    k = record.matmul.shape.k
    parse, passes = _rules(scheme)
    ea, eb = set(record.excluded_a), set(record.excluded_b)
    va, vb = set(record.voluntary_a), set(record.voluntary_b)
    kinds = [("A", i, row, "filler" if i >= record.real_rows else "excluded" if i in ea else "voluntary" if i in va
              else "credited") for i, row in zip(tile.rows, a_rows, strict=True)]
    kinds += [("B", j, row, "excluded" if j in eb else "voluntary" if j in vb else "credited")
              for j, row in zip(tile.cols, b_rows, strict=True)]
    pos_a, pos_b = {i: p for p, i in enumerate(tile.rows)}, {j: p for p, j in enumerate(tile.cols)}
    ratio = getattr(scheme, "row_debit_ratio", None)
    tol = getattr(scheme, "r1_tolerance", 0)

    def r1(side: str, index: int, credited: bool) -> bool:
        """Whether R1, a scheme's salted per-row credit rule (Pearl-C4's F1′ bound), contradicts the declaration: a
        credited row over its bound, or an excluded row under it, beyond the scheme's tolerance band.  Pearl-C's schemes
        bring none."""
        if ratio is None or ops is None:
            return credited is False
        r = ratio(record, ops, side, (pos_a if side == "A" else pos_b)[index])
        return r > 1 + tol if credited else r <= 1 - tol

    for side, index, row, kind in kinds:
        try:
            words = parse(row, k)
        except P.Reject as e:
            return f"{side} row {index} is malformed: {e}"
        if kind == "filler":
            if any(words):
                return f"filler row {index} is not zero"
            continue
        if any(words[record.k_kept:]):
            return f"{side} row {index} is not zero past the kept width"
        if kind == "voluntary":
            continue
        ok = passes(words)
        if (kind == "credited" and not ok) or (kind == "excluded" and ok and r1(side, index, False)):
            return f"{side} row {index} is {kind} but {'fails' if kind == 'credited' else 'passes'} the per-row rules"
        if kind == "credited" and r1(side, index, True):
            return f"{side} row {index} is credited but its own debit passes R1's bound (ρ·its cell credit)"
    return None


def cap_failure(scheme: C.PearlC, record: UnitRecord, tile: Tile, ops: C.Ops | None, tol: float | Fraction = 0
                ) -> str | None:
    """`check_opened`'s inert-work bound on the tile's credited words: None, or the reason the tile's inert work or (under a row-bounded
    format) a credited row's is over its bound.  The hidden path's is `circuit.cap`'s (`cap.served`)."""
    k = record.matmul.shape.k
    rows, cols = record.credited(tile)
    if rows and cols:
        bound = 1 + Fraction(tol)
        debit, cap = _debit_cap(scheme, record, tile, ops, rows, cols)
        if debit > cap * bound:
            return f"the replayed debit {float(debit):.0f} exceeds the cap {float(cap):.0f} (ρ·the credited cells' credit)"
        widen = row_cap_multiple(scheme, k)
        for i in rows if row_cap(scheme) else ():
            debit, cap = _debit_cap(scheme, record, tile, ops, [i], cols)
            if debit > cap * widen * bound:
                return (f"A row {i}'s replayed debit {float(debit):.0f} exceeds its cap {float(cap * widen):.0f} (the row "
                        f"cap: {'' if widen == 1 else f'{widen}'}ρ·its credited cells' credit)")
    return None


def row_cap(scheme: C.PearlC) -> bool:
    """Whether the scheme's hashing format bounds every credited row's inert work as well as the tile's (`HashFormat.row_cap`,
    -h3's): (C2) of `TTOutRowSeed.lean`, `pearlCTilesDevRev1RowCap`, each row's inert work over the tile's credited columns at
    most ρ times its cells' share of the credit, or `row_cap_multiple`·ρ (`pearlCTilesDevRev1RowCapAt`)."""
    hashing = getattr(scheme, "hashing", None)
    return hashing in HASH_FORMATS and HASH_FORMATS[hashing].row_cap


def row_cap_multiple(scheme: C.PearlC, k: int) -> Fraction:
    """A credited row's inert-work bound at depth ``k`` over the tile's ρ: the scheme's `row_cap_at(k)`/ρ, 4 at Pearl-C's census ks
    (`pearl_c.CENSUS_ROW_CAP`), and 1 for a scheme without one."""
    at = getattr(scheme, "row_cap_at", None)
    return Fraction(1) if at is None else Fraction(at(k)) / Fraction(scheme.cap)


def _debit_cap(scheme: C.PearlC, record: UnitRecord, tile: Tile, ops, rows: Sequence[int], cols: Sequence[int]
               ) -> tuple[Fraction, Fraction]:
    """(inert work, its bound) over the words of the tile's rows × cols: the scheme's `tile_cap` when it brings one, else this
    module's on the A′ and B̃ code rows and the A rows' forming."""
    pos_a, pos_b = {i: p for p, i in enumerate(tile.rows)}, {j: p for p, j in enumerate(tile.cols)}
    if hasattr(scheme, "tile_cap"):
        return scheme.tile_cap(record, ops, [pos_a[i] for i in rows], [pos_b[j] for j in cols])
    a = [pos_a[i] for i in rows]
    return tile_cap(scheme, record, [ops.a.codes[p] for p in a], [ops.b.codes[pos_b[j]] for j in cols],
                    [(ops.a.alpha[p], ops.a.x[p], ops.a.beta[p]) for p in a])


class _TileRuleView:
    """The scheme with its tile-wide admissibility left to `check_opened`, which applies the per-row rules by the record."""

    def __init__(self, scheme: C.PearlC):
        self._scheme = scheme

    def __getattr__(self, name: str):
        return getattr(self._scheme, name)

    def admissible(self, ctx, ops) -> bool:
        return True


# ---- the audit --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RecordedDraw:
    """A recorded run's sample source: its round and value, for `audit_replay` alone."""

    round: int
    value: bytes


def draw_after(transcript_root: bytes, records_digest: bytes) -> bytes:
    """What the verifier records before the challenger opens the sampling coin: the transcript root and the records root."""
    return tagged(DRAW_AFTER_TAG, transcript_root, records_digest)


def draw_key(coin: bytes, transcript_root: bytes, records_digest: bytes, salt: bytes) -> Key:
    """The samples' key from the epoch's sampling coin and the commitment it was opened after."""
    return derive(coin, DRAW_DOMAIN, {"transcript": transcript_root, "records": records_digest, "salt": salt})


def replay_key(recorded: RecordedDraw, transcript_root: bytes, records_digest: bytes, salt: bytes) -> Key:
    """A recorded run's samples' key, byte for byte as the run drew them."""
    return derive(recorded.value, DRAW_DOMAIN, {"round": recorded.round, "transcript": transcript_root,
                                                "records": records_digest, "salt": salt})


def draw(scheme: C.PearlC, records: Sequence[UnitRecord], key: Key, t: int) -> list[tuple[int, Tile]]:
    """t work-weighted samples with replacement over the tiles with credited cells, under `key` (`draw_key`)."""
    tk = [x for x in tickets(scheme, records) if x[2] > 0]
    if not tk:
        return []
    cumulative = list(accumulate(w for _, _, w in tk))
    return [(tk[i][0], tk[i][1]) for i in (bisect_right(cumulative, key.uniform(cumulative[-1], j)) for j in range(t))]


def exclusion_tiles(scheme: C.PearlC, records: Sequence[UnitRecord]) -> list[tuple[int, Tile]]:
    """One tile per excluded row, which the audit opens: an excluded A row's tile in the first column block, an excluded B
    row's in the first row block of the weight's first unit."""
    out: dict[tuple[int, int, int], None] = {}
    first: dict[int, int] = {}
    for p, r in enumerate(records):
        first.setdefault(r.matmul.weight, p)
        grid = scheme.tiles(r.matmul.shape)
        for i in r.excluded_a:
            t = next(t for t in grid if t.rows[0] <= i <= t.rows[-1])
            out[(p, t.rows[0], t.cols[0])] = None
        if first[r.matmul.weight] == p:
            for j in r.excluded_b:
                t = next(t for t in grid if t.cols[0] <= j <= t.cols[-1])
                out[(p, t.rows[0], t.cols[0])] = None
    tiles = {(p, t.rows[0], t.cols[0]): t for p, r in enumerate(records) for t in scheme.tiles(r.matmul.shape)}
    return [(key[0], tiles[key]) for key in out]


@dataclass(frozen=True)
class WorkAudit:
    """An audit by the records: the verdicts, W (the credited cells' `wref`), the credit (1 − ρ)·`credit_of` over them,
    and the weight side once per weight (outside both)."""

    audit: Audit
    wref: Fraction
    credit: Fraction
    job: int


def check_drawn(scheme: C.PearlC, epoch: Epoch, transcript: Transcript, records: Sequence[UnitRecord],
                pick: tuple[int, Tile, TileOpening]) -> Verdict:
    """One opened tile, (record position, tile, opening): `Verifier.check_tile`, then `check_opened`."""
    pos, tile, opening = pick
    v = Verifier(_TileRuleView(scheme), epoch).check_tile(transcript, opening)
    if v.ok:
        u = epoch.workload.matmuls[pos]
        ctx = Context(epoch.salt, u, transcript.activation_roots[pos], epoch.weight_roots[u.weight])
        h = commitment_hash(scheme)
        a, b = [opened_value(h, o.value) for o in opening.a], [opened_value(h, o.value) for o in opening.b]
        reason = check_opened(scheme, records[pos], tile, a, b, scheme.encode(ctx, scheme.noise(ctx), a, b, tile))
        v = v if reason is None else Verdict(v.tile, False, reason)
    return v


def audit(scheme: C.PearlC, epoch: Epoch, prover: Prover, transcript: Transcript, splits: Mapping[int, Split],
          records: Sequence[UnitRecord], challenger: Challenger, t: int,
          split_check: Callable[[int, Split], bool] | None = None, tile_map: Callable = map) -> WorkAudit:
    """Check the splits and the records, sample t tiles by the records' credit with the epoch's sampling coin, open one tile
    per excluded row, and accept iff every opened tile passes `check_drawn`.
    - The verifier asks `challenger` for the sampling coin only once it has recorded the commitment (`draw_after`), and the
      coin must open against the epoch's coin commitment (`clear_audit.draw_coin`).
    - `split_check(weight, split)` recomputes S from the calibration data by the pinned rule, and is required when any S
      is nonempty.
    - `tile_map(f, picks)` applies `check_drawn` to the opened tiles in order (a process pool's map checks them in
      parallel)."""
    def key_of(digest: bytes) -> tuple[Key | None, str | None]:
        root = transcript.commitment.root
        coin, reason = draw_coin(epoch, challenger, draw_after(root, digest))
        return (None, reason) if reason is not None else (draw_key(coin, root, digest, epoch.salt), None)

    return _audit(scheme, epoch, prover, transcript, splits, records, t, split_check, tile_map, key_of)


def audit_replay(scheme: C.PearlC, epoch: Epoch, prover: Prover, transcript: Transcript, splits: Mapping[int, Split],
                 records: Sequence[UnitRecord], committed_round: int, recorded: RecordedDraw, t: int,
                 split_check: Callable[[int, Split], bool] | None = None, tile_map: Callable = map) -> WorkAudit:
    """`audit` with a recorded run's unverified sample, for replays whose samples are Fiat–Shamir on the commitment
    (`pearl_c4_replay`): the recorded round must only exceed `committed_round`.  It supports no claim about the sample."""
    def key_of(digest: bytes) -> tuple[Key | None, str | None]:
        if recorded.round <= committed_round:
            return None, "the draws' recorded round is not after the commitment"
        return replay_key(recorded, transcript.commitment.root, digest, epoch.salt), None

    return _audit(scheme, epoch, prover, transcript, splits, records, t, split_check, tile_map, key_of)


def _audit(scheme: C.PearlC, epoch: Epoch, prover: Prover, transcript: Transcript, splits: Mapping[int, Split],
           records: Sequence[UnitRecord], t: int, split_check: Callable[[int, Split], bool] | None, tile_map: Callable,
           key_of: Callable[[bytes], tuple[Key | None, str | None]]) -> WorkAudit:
    if getattr(scheme, "forming", "v1") != "v1":
        raise ValueError("the work law is stated for forming v1")
    device = getattr(scheme, "device", None)
    if device is not None and ALIGN % device.window:
        raise ValueError(f"k_unit's alignment {ALIGN} is not whole promotion windows of {scheme.name}")
    w_ref, credit = work(scheme, records)
    job = job_costs(scheme, records)

    def reject(reason: str) -> WorkAudit:
        return WorkAudit(Audit(False, (), (Verdict(-1, False, reason),), None), w_ref, credit, job)

    reason = check_records(epoch, splits, records)
    if reason is None and len(transcript.activation_roots) != len(records):
        reason = "the transcript does not have one activation root per record"
    if reason is None:
        for w, sp in sorted(splits.items()):
            if sp.side and (split_check is None or not split_check(w, sp)):
                reason = f"weight {w}: S is not the pinned rule's on its calibration data"
                break
    if reason is not None:
        return reject(reason)
    verifier = Verifier(_TileRuleView(scheme), epoch)
    digest = records_root(epoch.salt, epoch.weights_root, splits, records)
    key, reason = key_of(digest)
    if reason is not None:
        return reject(reason)
    ids = {(ref.matmul, ref.tile.rows[0], ref.tile.cols[0]): ref.id for ref in verifier.refs}
    picks = draw(scheme, records, key, t) + exclusion_tiles(scheme, records)
    opened = [(pos, tile, prover.open(epoch, ids[(pos, tile.rows[0], tile.cols[0])])) for pos, tile in picks]
    verdicts = list(tile_map(partial(check_drawn, scheme, epoch, transcript, records), opened))
    ok = all(v.ok for v in verdicts)
    profile = WorkProfile(scheme.name, int(w_ref), t, True, scheme.certificate) if ok else None
    return WorkAudit(Audit(ok, tuple(v.tile for v in verdicts), tuple(verdicts), profile), w_ref, credit, job)


def declare(scheme: C.PearlC, epoch: Epoch, activations: Mapping[int, Sequence[bytes]], weights: Mapping[int, Sequence[bytes]],
            splits: Mapping[int, Split], real_rows: Mapping[int, int] | None = None,
            transcript: Transcript | None = None, voluntary_a: Mapping[int, Sequence[int]] | None = None,
            voluntary_b: Mapping[int, Sequence[int]] | None = None,
            padded_b: Mapping[int, Sequence[int]] | None = None) -> list[UnitRecord]:
    """An honest prover's records: each real row and each weight row checked by the per-row rules on the unit's words,
    and, for a scheme with a salted per-row credit rule (`r1_rows`, Pearl-C4's R1), by that rule on the committed
    transcript's forming.  Pearl-C's schemes bring none, so `transcript` changes nothing for them.  `voluntary_a` (by
    matmul index) and `voluntary_b` (by weight) are the rows the prover gives up (V-EX, `volunteer`); a row the rules
    exclude stays excluded.  `padded_b` (by weight) is a padded tile's rows (`pearl_c_work.pad_weight`), voluntary
    whether or not they pass, so the tile opens no exclusion tile."""
    parse, passes = _rules(scheme)
    k_of = {u.weight: u.shape.k for u in epoch.workload.matmuls}
    padded = {w: set(rows) for w, rows in (padded_b or {}).items()}
    excluded_b = {w: {j for j, r in enumerate(rows) if j not in padded.get(w, ()) and not passes(parse(r, k_of[w]))}
                  for w, rows in weights.items()}
    r1 = getattr(scheme, "r1_rows", None) if transcript is not None else None
    extra_a: dict[int, set[int]] = {}
    if r1 is not None:
        done: set[int] = set()
        for p, u in enumerate(epoch.workload.matmuls):
            real = (real_rows or {}).get(u.index, u.shape.m)
            ctx = Context(epoch.salt, u, transcript.activation_roots[p], epoch.weight_roots[u.weight])
            ea, eb = r1(ctx, activations[u.index][:real], weights[u.weight], splits[u.weight].k_kept)
            extra_a[u.index] = ea
            if u.weight not in done:
                excluded_b[u.weight] |= eb
                done.add(u.weight)
    out = []
    for u in epoch.workload.matmuls:
        real = (real_rows or {}).get(u.index, u.shape.m)
        rows = activations[u.index]
        excluded_a = tuple(i for i in range(real) if not passes(parse(rows[i], u.shape.k)) or i in extra_a.get(u.index, ()))
        va = tuple(sorted(set((voluntary_a or {}).get(u.index, ())) - set(excluded_a)))
        vb = tuple(sorted((set((voluntary_b or {}).get(u.weight, ())) - excluded_b[u.weight]) | padded.get(u.weight, set())))
        out.append(UnitRecord(u, real, splits[u.weight].k_kept, excluded_a, tuple(sorted(excluded_b[u.weight])), va, vb))
    return out


def volunteer(scheme: C.PearlC, epoch: Epoch, transcript: Transcript, records: Sequence[UnitRecord],
              activations: Mapping[int, Sequence[bytes]], weights: Mapping[int, Sequence[bytes]]
              ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """V-EX's honest remedy, before the samples: (voluntary A rows by matmul index, voluntary B rows by weight) such that
    no tile's credited inert work passes its bound, for `declare`.  The prover knows the salt when it forms and commits, so it
    replays every tile's count of inert work as the verifier will.  For a tile over the bound it gives up the credited row whose removal
    leaves the least inert work over the bound, and repeats while the tile is over and the count limits allow; a tile the limits
    leave over the bound stays so, and fails if sampled.  Under the row bound (`row_cap`) a credited A row over its own bound goes
    first, the furthest over first.  Rows go up to ⌈m/64⌉ per unit and ⌈n/64⌉ per weight.  R1's rows aren't given up
    here: `records` are `declare`'s on the transcript, which excludes them.  By the band rule a row given up takes its
    band's or block's credit (`UnitRecord.credited`), so at the deployed 64 × 64 tile the first row given up leaves the
    tile no credited cell."""
    va: dict[int, set[int]] = {r.matmul.index: set(r.voluntary_a) for r in records}
    vb: dict[int, set[int]] = {}
    for r in records:
        vb.setdefault(r.matmul.weight, set(r.voluntary_b))
    tol = 1 + Fraction(getattr(scheme, "r1_tolerance", 0))
    changed = True
    while changed:
        # a B row given up changes every tile of its weight, and can lift one that passed over its bound
        changed = False
        for p, r in enumerate(records):
            u = r.matmul
            ctx = Context(epoch.salt, u, transcript.activation_roots[p], epoch.weight_roots[u.weight])
            noise = scheme.noise(ctx)
            for tile in scheme.tiles(u.shape):
                a = [activations[u.index][i] for i in tile.rows]
                b = [weights[u.weight][j] for j in tile.cols]
                ops = scheme.encode(ctx, noise, a, b, tile)
                while True:
                    cur = UnitRecord(u, r.real_rows, r.k_kept, r.excluded_a, r.excluded_b,
                                     tuple(sorted(va[u.index])), tuple(sorted(vb[u.weight])))
                    rows, cols = cur.credited(tile)
                    if not (rows and cols):
                        break
                    widen = row_cap_multiple(scheme, u.shape.k)
                    over = [(d - c * widen, i) for i in (rows if row_cap(scheme) else ())
                            for d, c in (_debit_cap(scheme, cur, tile, ops, [i], cols),) if d > c * widen * tol]
                    if over and len(va[u.index]) < voluntary_cap(u.shape.m):
                        va[u.index].add(max(over)[1])
                        changed = True
                        continue
                    debit, cap = _debit_cap(scheme, cur, tile, ops, rows, cols)
                    if debit <= cap * tol:
                        break
                    options = []
                    if len(va[u.index]) < voluntary_cap(u.shape.m):
                        options += [("A", i, _debit_cap(scheme, cur, tile, ops, [x for x in rows if x != i], cols))
                                    for i in rows if len(rows) > 1]
                    if len(vb[u.weight]) < voluntary_cap(u.shape.n):
                        options += [("B", j, _debit_cap(scheme, cur, tile, ops, rows, [x for x in cols if x != j]))
                                    for j in cols if len(cols) > 1]
                    if not options:
                        break
                    side, row, _ = min(options, key=lambda o: o[2][0] - o[2][1])
                    (va[u.index] if side == "A" else vb[u.weight]).add(row)
                    changed = True
    return {i: sorted(s) for i, s in va.items() if s}, {w: sorted(s) for w, s in vb.items() if s}


def _word_units(scheme: C.PearlC, records: Sequence[UnitRecord],
                words: Mapping[tuple[int, int], tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]]):
    """(scale, tol, per record (p, record, a cell's bound, a row's bound a cell, its tiles as (t, tile, each word's inert work))), the
    inert work and bounds times `scale`, so a word's inert work is an integer."""
    import numpy as np

    if hasattr(scheme, "tile_cap"):
        raise ValueError(f"{scheme.name} brings its own debit, not one additive over a tile's words")
    au = D.debit_units(1, 0, scheme.prices, scheme.device, zeros=0)
    fa = D.debit_units(0, 0, scheme.prices, scheme.device, zeros=1)
    scale = lcm(au.denominator, fa.denominator)
    units = []
    for p, r in enumerate(records):
        cell = scheme.cap * cell_credit(scheme, r) * scale
        tiles = []
        for t, tile in enumerate(scheme.tiles(r.matmul.shape)):
            atoms, zeros = words[(p, t)]
            tiles.append((t, tile, int(au * scale) * np.asarray(atoms, np.int64) + int(fa * scale) * np.asarray(zeros, np.int64)))
        units.append((p, r, cell, cell * row_cap_multiple(scheme, r.matmul.shape.k), tiles))
    return scale, 1 + Fraction(getattr(scheme, "r1_tolerance", 0)), units


def over_words(scheme: C.PearlC, records: Sequence[UnitRecord],
               words: Mapping[tuple[int, int], tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]]) -> list[dict]:
    """`cap_failure` on every tile from its words' counts (`volunteer_words`' ``words``): each tile whose credited inert work is
    over its bound, or (under the row bound) one of whose credited A rows is, with the tile's inert work and bound and each such row's."""
    import numpy as np

    scale, tol, units = _word_units(scheme, records, words)
    out = []
    for p, r, cell, row_cell, tiles in units:
        for t, tile, words_t in tiles:
            rows, cols = (set(x) for x in r.credited(tile))
            if not (rows and cols):
                continue
            pos_a, pos_b = [x for x, i in enumerate(tile.rows) if i in rows], [z for z, j in enumerate(tile.cols) if j in cols]
            w = words_t[np.ix_(pos_a, pos_b)]
            debit, cap = Fraction(int(w.sum()), scale), cell * len(rows) * len(cols) / scale
            over = [(tile.rows[x], Fraction(int(d), scale), row_cell * len(cols) / scale) for x, d in zip(pos_a, w.sum(axis=1))
                    if row_cap(scheme) and int(d) > row_cell * len(cols) * tol]
            if over or debit > cap * tol:
                out.append({"record": p, "tile": t, "debit": debit, "cap": cap, "rows": over})
    return out


def volunteer_words(scheme: C.PearlC, records: Sequence[UnitRecord],
                    words: Mapping[tuple[int, int], tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]]
                    ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """`volunteer` for a scheme whose inert work is additive over a tile's words (Pearl-C's chain term:
    `pearl_c_debit.chain_debit`'s counts, priced linearly by `debit_units`, and no `tile_cap` of its own), from each
    tile's counts word by word.  P3's forming term is not additive over words (an element is inert when every word of
    its row flags its block), so neither this nor `over_words` reads it: a tile it leaves within its inert-work bound
    can still be over `tile_cap`'s.
    ``words[(p, t)]`` is (flagged atoms, later +0 promotions) over the rows x columns of tile t of record p, in
    `scheme.tiles`' order (`fast_tile.word_debits` on the tile's A′ and B̃ codes).  Returns (voluntary A rows by matmul
    index, voluntary B rows by weight), for `declare`; `over_words` on the records with them is what the limits leave over.
    - Under the row bound (`row_cap`) a unit's A rows over their own bounds go first, the row furthest over summed over the
      unit's tiles first, one a band of 64 (the band rule leaves the rest of its band no credit), up to ⌈m/64⌉, which
      covers every band unless the records already spent it.  A row the A count leaves over then gives up, one at a
      time, the B row whose word
      carries most of its inert work, while that word's inert work is over the row's bound a cell, up to ⌈n/64⌉ per weight.
    - Then a tile over its own bound gives up the row whose removal leaves the least inert work over the bound, as `volunteer` does.
    - It repeats until nothing changes: a B row given up lowers the bound of every row in its column's tiles.
    Every tile's credited rows and columns are `UnitRecord.credited`'s at the rows given up so far, the band rule's."""
    import numpy as np

    _, tol, units = _word_units(scheme, records, words)
    capped = row_cap(scheme)
    va: dict[int, set[int]] = {r.matmul.index: set(r.voluntary_a) for r in records}
    vb: dict[int, set[int]] = {}
    for r in records:
        vb.setdefault(r.matmul.weight, set(r.voluntary_b))

    def credited(r: UnitRecord, tile: Tile) -> tuple[list[int], list[int]]:
        cur = replace(r, voluntary_a=tuple(sorted(va[r.matmul.index] - set(r.excluded_a))),
                      voluntary_b=tuple(sorted(vb[r.matmul.weight] - set(r.excluded_b))))
        rows, cols = (set(x) for x in cur.credited(tile))
        return [x for x, i in enumerate(tile.rows) if i in rows], [z for z, j in enumerate(tile.cols) if j in cols]

    def a_left(r: UnitRecord) -> bool:
        return len(va[r.matmul.index]) < voluntary_cap(r.matmul.shape.m)

    def b_left(r: UnitRecord) -> bool:
        return len(vb[r.matmul.weight]) < voluntary_cap(r.matmul.shape.n)

    changed = True
    while changed:
        changed = False
        for _, r, _, row_cell, tiles in units if capped else ():
            if not a_left(r):
                continue
            # an A row given up takes its band out and leaves every other band's rows' sums and bounds as they were (both
            # read the credited columns, which only a B row changes), so one scan ranks the rows given up one at a time:
            # furthest over, then the lower row, one a band
            bands = {i // BAND for i in va[r.matmul.index]}
            excess: dict[int, Fraction] = {}
            for _, tile, w in tiles:
                rows, cols = credited(r, tile)
                if rows and cols:
                    s, limit = w[np.ix_(rows, cols)].sum(axis=1), row_cell * len(cols) * tol
                    for x in np.flatnonzero(s > floor(limit)):
                        i = tile.rows[rows[x]]
                        excess[i] = excess.get(i, 0) + int(s[x]) - limit
            for i in sorted(excess, key=lambda i: (-excess[i], i)):
                if not a_left(r):
                    break
                if i // BAND not in bands:
                    va[r.matmul.index].add(i)
                    bands.add(i // BAND)
                    changed = True
        for _, r, _, row_cell, tiles in units if capped else ():
            for _, tile, w in tiles:
                while b_left(r):
                    rows, cols = credited(r, tile)
                    if not (rows and cols):
                        break
                    sub = w[np.ix_(rows, cols)]
                    x = int(np.argmax(sub.sum(axis=1)))
                    z = int(np.argmax(sub[x]))
                    if int(sub[x].sum()) <= row_cell * len(cols) * tol or int(sub[x, z]) <= row_cell * tol:
                        break
                    vb[r.matmul.weight].add(tile.cols[cols[z]])
                    changed = True
        for _, r, cell, _, tiles in units:
            for _, tile, w in tiles:
                while True:
                    rows, cols = credited(r, tile)
                    if not (rows and cols):
                        break
                    sub = w[np.ix_(rows, cols)]
                    debit, nr, nc = int(sub.sum()), len(rows), len(cols)
                    if debit <= cell * nr * nc * tol:
                        break
                    options = []
                    if a_left(r) and nr > 1:
                        options += [(debit - int(d) - cell * (nr - 1) * nc, "A", tile.rows[rows[x]])
                                    for x, d in enumerate(sub.sum(axis=1))]
                    if b_left(r) and nc > 1:
                        options += [(debit - int(d) - cell * nr * (nc - 1), "B", tile.cols[cols[z]])
                                    for z, d in enumerate(sub.sum(axis=0))]
                    if not options:
                        break
                    _, side, row = min(options, key=lambda o: o[0])
                    (va[r.matmul.index] if side == "A" else vb[r.matmul.weight]).add(row)
                    changed = True
    return {i: sorted(s) for i, s in va.items() if s}, {w: sorted(s) for w, s in vb.items() if s}
