"""Pearl-C's work law: what the verifier credits, draws and checks, all recomputed by the verifier (statement review item 5;
red team F1, F2 and F5–F8, research store ``internal/pouw/red-team-pearl-c-vllm.md``).

* **`Split`**: a weight's channel split.
  - S is a set of at most ⌊k/100⌋ input channels, computed outside the unit.
  - The unit reads the |K| = k − |S| kept channels in order, zero-padded to `k_unit`, a multiple of 128.
  - `calibration` names the data S was computed from. The pinned rule that computes S (`split_channels`, with its
    tie-break and budget) lives with the model integration; `audit` calls it through `split_check`.
* **`UnitRecord`**: what the prover declares for each matmul before the draws.
  - Its real rows. The rest are filler, which must be zero words.
  - Its kept width |K|.
  - Its excluded rows: A rows of the matmul and B rows of its weight that fail the scheme's per-row rules.
  `records_root` binds the records and the splits.
* **The one exclusion rule (per row).**
  - An excluded row stays in the unit, and its C̃ and U words are computed and bound as any other's.
  - Its cells earn no credit. The verifier checks that it fails the rules: on every drawn tile that holds it, and on one
    opened tile per excluded row, every one.
  - A credited row must pass the rules.
* **Credit.**
  - A cell's draw weight is its row's share of `W_ref` at |K|, with no weight side (F2), so padding to `k_unit` and
    filler earn nothing. `W_ref` is the one the draw's Lean statement weighs by (`draw_wref`): on sm_120 at its cap
    1/1,000, `wrefDevRev1K` at `Prices.sm120Loop` and the honest cast at 8.72 (`SM120_LOOP_CAST_8P72`), so that a served
    step's mixed linears are drawn in proportion to it (red team #919 F1); elsewhere the scheme's own `wref`, Lean's
    `wrefDevRev1` at its record.  W, the profile's work, stays the scheme's own `wref`.
  - An accepted unit is credited (1 − ρ)·`creditDevRev1` over its credited cells, at the scheme's cap ρ (`scheme.cap`:
    Pearl-C's is its device record's, 1/400 on the H100 and 1/1,000 on sm_120; Pearl-C4's is its own 1/400).
* **The tile rule (`check_opened`)**, beside `verity_pouw.audit.Verifier.check_tile`'s openings and digest:
  - filler rows are zero;
  - every opened row is zero past |K| (F7);
  - credited rows pass the per-row rules, and excluded rows fail them;
  - the debit replayed over the credited words (`pearl_c_debit`) is at most ρ times the credited cells' credit, their
    share of the unit's `creditDevRev1` (F1; `tile_cap`).
  A failure is a wrong tile, reported as a reason; nothing raises.
* **Voluntary rows (V-EX; research store ``internal/pouw/rtx-pro/theory-pearl-c4-domain.md`` §6.13, the assessor's B at
  17:32Z).**
  - A record may give up up to ⌈m/64⌉ A rows (`voluntary_a`), and a weight up to ⌈n/64⌉ B rows (`voluntary_b`, the same
    in every unit of the weight). They are committed in `records_root` with the excluded rows, before the draws' beacon.
  - A voluntary row earns no credit. Its words are still formed, checked and hashed on every drawn tile that holds it.
  - It is exempt from the two-sided per-row check and R1, and it opens no exclusion tile.
  - The tile cap and R1 read credited rows only, as before. Giving up credit never helps a cheater, and the declaration
    can't adapt to the draw. `volunteer` is an honest prover's remedy for a tile over the cap.
* **The sampling guarantee is per credited cell.** The draws are weighted by the credited cells' `wref`, and a drawn
  tile's uncredited cells (filler, excluded or voluntary) never count as coverage. Volunteering a tile's other rows only
  lowers that tile's weight to its credited cells, which is exactly the bound claimed.
* **The draws (F8)** are keyed by a drand quicknet round published after the verifier recorded the prover's commitment
  (the transcript root and the records root), through `verity.randomness.derive`.
  - `audit` verifies the round's BLS signature against the pinned key (`verity_pouw.beacon`).
  - It requires the round to be at least 2 rounds after the one due at `recorded_at`, the verifier's clock at the
    commitment.
  - `audit_replay` keeps the unverified draw for replays whose draws are Fiat–Shamir on the commitment.
* **Other row formats.** A scheme with its own row encoding, per-row rules and debit (`pearl_c4`) brings
  `parse_row(row, k)`, `row_passes(words)` and `tile_cap(record, ops, rows, cols)`; Pearl-C v1's are this module's. Every
  scheme brings its `cap`.
"""

from __future__ import annotations

import copy
import struct
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import partial
from itertools import accumulate
from math import lcm

from verity.randomness import derive

from .. import beacon as B
from ..audit import Audit, Epoch, Prover, TileOpening, Transcript, Verdict, Verifier, WorkProfile
from ..protocol import Context, Matmul, Shape, Tile, tagged
from . import pearl_c as C
from . import pearl_c_debit as D
from . import pearl_kw as P
from .pearl_c_device import SM120, Prices

RECORDS_TAG = "verity/pouw/pearl-c/records/v2"
SPLIT_TAG = "verity/pouw/pearl-c/split/v1"
DRAW_DOMAIN = "verity/pouw/pearl-c-draw/v1"
ALIGN = 128


@dataclass(frozen=True)
class Split:
    """One weight's channel split: the layer's k input channels, the side set S, and the calibration data's digest."""

    k: int
    side: tuple[int, ...] = ()
    calibration: bytes = b""

    def __post_init__(self) -> None:
        if any(not 0 <= c < self.k for c in self.side) or any(a >= b for a, b in zip(self.side, self.side[1:])):
            raise ValueError("S is strictly increasing channels of the layer")
        if len(self.side) > self.k // 100:
            raise ValueError(f"|S| = {len(self.side)} exceeds 1% of k = {self.k}")

    @property
    def k_kept(self) -> int:
        return self.k - len(self.side)

    @property
    def k_unit(self) -> int:
        return -(-self.k_kept // ALIGN) * ALIGN

    def encode(self) -> bytes:
        return struct.pack(f"<II{len(self.side)}I", self.k, len(self.side), *self.side) + tagged(SPLIT_TAG, self.calibration)


@dataclass(frozen=True)
class UnitRecord:
    """One matmul as its prover declares it before the draws."""

    matmul: Matmul                  # index, (m, k_unit, n), weight
    real_rows: int                  # rows 0 … real_rows − 1 are the call's; the rest are zero filler
    k_kept: int                     # entries 0 … k_kept − 1 of every row are the kept channels; the rest are zero
    excluded_a: tuple[int, ...] = ()  # A rows (< real_rows) that fail the per-row rules
    excluded_b: tuple[int, ...] = ()  # B rows of the weight that fail them
    voluntary_a: tuple[int, ...] = ()  # A rows (< real_rows) given up before the draws (V-EX)
    voluntary_b: tuple[int, ...] = ()  # B rows of the weight given up (V-EX)

    def __post_init__(self) -> None:
        s = self.matmul.shape
        if not (0 < self.real_rows <= s.m and 0 < self.k_kept <= s.k):
            raise ValueError("a record's real rows and kept width lie within its shape")
        for rows, bound in ((self.excluded_a, self.real_rows), (self.excluded_b, s.n), (self.voluntary_a, self.real_rows),
                            (self.voluntary_b, s.n)):
            if any(not 0 <= r < bound for r in rows) or any(a >= b for a, b in zip(rows, rows[1:])):
                raise ValueError("excluded and voluntary rows are strictly increasing and in range")
        if set(self.excluded_a) & set(self.voluntary_a) or set(self.excluded_b) & set(self.voluntary_b):
            raise ValueError("a row is excluded or voluntary, not both")

    def encode(self) -> bytes:
        """The v2 layout; a record with voluntary rows appends their counts and rows (each record is one framed part of
        `records_root`, so the two layouts can't be confused)."""
        u, s = self.matmul, self.matmul.shape
        a, b = self.excluded_a, self.excluded_b
        base = struct.pack(f"<QQIIIIIII{len(a)}I{len(b)}I", u.index, u.weight, s.m, s.k, s.n, self.real_rows,
                           self.k_kept, len(a), len(b), *a, *b)
        va, vb = self.voluntary_a, self.voluntary_b
        return base + struct.pack(f"<II{len(va)}I{len(vb)}I", len(va), len(vb), *va, *vb) if va or vb else base

    def credited(self, tile: Tile) -> tuple[list[int], list[int]]:
        """The tile's credited rows and columns: real, neither excluded nor voluntary."""
        ua, ub = set(self.excluded_a) | set(self.voluntary_a), set(self.excluded_b) | set(self.voluntary_b)
        return [i for i in tile.rows if i < self.real_rows and i not in ua], [j for j in tile.cols if j not in ub]


#: V-EX's count caps: at most ⌈m/64⌉ voluntary A rows per unit and ⌈n/64⌉ voluntary B rows per weight.
VOLUNTARY_PER = 64


def voluntary_cap(rows: int) -> int:
    return -(-rows // VOLUNTARY_PER)


def records_root(salt: bytes, weights_root: bytes, splits: Mapping[int, Split], records: Sequence[UnitRecord]) -> bytes:
    return tagged(RECORDS_TAG, salt, weights_root, *(p for w in sorted(splits) for p in (w, splits[w].encode())),
                  *(r.encode() for r in records))


def check_records(epoch: Epoch, splits: Mapping[int, Split], records: Sequence[UnitRecord]) -> str | None:
    """The records against the workload and the splits: one per matmul in order, k = the split's `k_unit`, |K| its
    `k_kept`, one set of excluded and one of voluntary B rows per weight, and V-EX's count caps."""
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
    """The `W_ref` a Lean statement's draw weighs by: `wrefDevRev1K (devAt d prices) c`, the scheme's rev1 `wref` at the
    statement's prices plus `c` per activation element (the honest kernel's forming beyond the record's).  `GγSampled`,
    `HonestTileCapRate` and `CompleteSampled` take tickets proportional to a tile's `Wref`, over all of its rows and
    columns, so a draw by another price record's `wref` attains the statement's γ only on a layout of one (n, k): on a
    served step's four linears the per-linear ratios differ (red team #919 F1).  `tile_weight` counts credited cells
    only, so the tickets are the statement's on fully credited tiles; a tile with filler, excluded or voluntary rows
    gets less than its `Wref`'s ticket at the scale that fully credited tiles get theirs, so unless every tile's
    credited share is the same, no Lean statement covers that draw yet (red team #1014)."""

    statement: str
    device: str
    cap: Fraction
    prices: Prices
    c: Fraction


#: The statement Pearl-C v1's served γ and ε cite on sm_120 at its cap 1/1,000 (`ServedMixedRev1Gamma`, `Completeness`'s
#: `HonestTileCapRate`): `Prices.sm120Loop` ⟨1047/125, 2, ⟨40, 2⟩⟩ and the honest cast at 218/25 against the record's
#: 8953/1000 (the credited cast 8 and the A-only forming's measured 1.047, rounded up to qa = 2).
SM120_LOOP_CAST_8P72 = DrawWref("pearlCSampledSm120v1LoopCast8p72Rev1Cap1000", SM120.name, SM120.cap,
                                replace(SM120.prices, fadd=Fraction(1047, 125), forming_a_only=2),
                                Fraction(218, 25) - Fraction(8953, 1000))
DRAW_WREF = {(SM120.name, SM120.prices, SM120.cap): SM120_LOOP_CAST_8P72}


def draw_wref(scheme: C.PearlC) -> DrawWref | None:
    """The statement the scheme's draw weighs by, or None for the scheme's own `wref` (Lean's at its record, c = 0)."""
    if type(scheme) is not C.PearlC or scheme.forming != "v1":
        return None
    return DRAW_WREF.get((scheme.device.name, scheme.prices, scheme.cap))


def ticket_cell(scheme: C.PearlC, record: UnitRecord) -> Fraction:
    """A credited cell's draw weight: its row's share of the drawn statement's `W_ref` at |K| (`draw_wref`)."""
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
    statement the draw cites (`draw_wref`)."""
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


def tile_cap(scheme: C.PearlC, record: UnitRecord, a_codes: Sequence[Sequence[int]], b_codes: Sequence[Sequence[int]]
             ) -> tuple[Fraction, Fraction]:
    """(debit, cap) over credited words given as their A′ and B̃ code rows: the replayed debit, rev1's `tileDebitRev1`,
    and ρ times the credited cells' credit, the unit's `creditDevRev1` at |K| times their share |rows|·|cols|/(m·n) (the
    per-tile statement's `tileShare`, `pearlCTilesDevRev1`).  A row's own terms (its forming and A′·F_B) so count once
    across its row band, a |cols|/n share in each tile, not whole in every tile."""
    d = scheme.device
    atoms, zeros = D.chain_debit(a_codes, b_codes, d)
    debit = D.debit_units(atoms, D.forming_elements(a_codes, b_codes), scheme.prices, d, zeros=zeros)
    return debit, scheme.cap * cell_credit(scheme, record) * len(a_codes) * len(b_codes)


def _rules(scheme) -> tuple[Callable[[bytes, int], list[int]], Callable[[Sequence[int]], bool]]:
    """(parse_row, row_passes): the scheme's own when it brings them, else Pearl-C v1's."""
    return getattr(scheme, "parse_row", C.parse_f32_row), getattr(scheme, "row_passes", row_passes)


def check_opened(scheme: C.PearlC, record: UnitRecord, tile: Tile, a_rows: Sequence[bytes], b_rows: Sequence[bytes],
                 ops) -> str | None:
    """The tile rule on an opened tile whose openings and digest have passed: None, or the reason it fails."""
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
    rows, cols = record.credited(tile)
    if rows and cols:
        debit, cap = _debit_cap(scheme, record, tile, ops, rows, cols)
        if debit > cap * (1 + Fraction(getattr(scheme, "r1_tolerance", 0))):
            return f"the replayed debit {float(debit):.0f} exceeds the cap {float(cap):.0f} (ρ·the credited cells' credit)"
    return None


def _debit_cap(scheme: C.PearlC, record: UnitRecord, tile: Tile, ops, rows: Sequence[int], cols: Sequence[int]
               ) -> tuple[Fraction, Fraction]:
    """(debit, cap) over the words of the tile's rows × cols: the scheme's `tile_cap` when it brings one, else this
    module's on the A′ and B̃ code rows."""
    pos_a, pos_b = {i: p for p, i in enumerate(tile.rows)}, {j: p for p, j in enumerate(tile.cols)}
    if hasattr(scheme, "tile_cap"):
        return scheme.tile_cap(record, ops, [pos_a[i] for i in rows], [pos_b[j] for j in cols])
    return tile_cap(scheme, record, [ops.a.codes[pos_a[i]] for i in rows], [ops.b.codes[pos_b[j]] for j in cols])


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
class Beacon:
    """A public beacon round: its index and value."""

    round: int
    value: bytes


def draw(scheme: C.PearlC, records: Sequence[UnitRecord], beacon: Beacon, transcript_root: bytes, records_digest: bytes,
         salt: bytes, t: int) -> list[tuple[int, Tile]]:
    """t work-weighted draws with replacement over the tiles with credited cells, keyed by the beacon round and the
    commitment."""
    tk = [x for x in tickets(scheme, records) if x[2] > 0]
    if not tk:
        return []
    key = derive(beacon.value, DRAW_DOMAIN, {"round": beacon.round, "transcript": transcript_root, "records": records_digest,
                                             "salt": salt})
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
        a, b = [o.value for o in opening.a], [o.value for o in opening.b]
        reason = check_opened(scheme, records[pos], tile, a, b, scheme.encode(ctx, scheme.noise(ctx), a, b, tile))
        v = v if reason is None else Verdict(v.tile, False, reason)
    return v


def audit(scheme: C.PearlC, epoch: Epoch, prover: Prover, transcript: Transcript, splits: Mapping[int, Split],
          records: Sequence[UnitRecord], recorded_at: float, beacon: B.Round, t: int,
          split_check: Callable[[int, Split], bool] | None = None, tile_map: Callable = map,
          chain: B.Chain = B.QUICKNET) -> WorkAudit:
    """Check the splits and the records, draw t tiles by the records' credit with a verified beacon round after the
    commitment, open one tile per excluded row, and accept iff every opened tile passes `check_drawn`.
    - `recorded_at` is the verifier's clock (UNIX s) when it recorded the commitment (the transcript root and the
      records root). The draw's round must be a verified round of `chain` at least `beacon.MARGIN` rounds after the one
      due then (`beacon.check_after`), never read against a lagging view of the latest round.
    - `split_check(weight, split)` recomputes S from the calibration data by the pinned rule, and is required when any S
      is nonempty.
    - `tile_map(f, picks)` applies `check_drawn` to the opened tiles in order (a process pool's map checks them in
      parallel)."""
    reason = B.check_after(chain, beacon, recorded_at)
    if reason is not None:
        return _audit(scheme, epoch, prover, transcript, splits, records, 0, Beacon(beacon.round, beacon.value), t,
                      split_check, tile_map, reason)
    return _audit(scheme, epoch, prover, transcript, splits, records, B.round_at(chain, recorded_at),
                  Beacon(beacon.round, beacon.value), t, split_check, tile_map)


def audit_replay(scheme: C.PearlC, epoch: Epoch, prover: Prover, transcript: Transcript, splits: Mapping[int, Split],
                 records: Sequence[UnitRecord], committed_round: int, beacon: Beacon, t: int,
                 split_check: Callable[[int, Split], bool] | None = None, tile_map: Callable = map) -> WorkAudit:
    """`audit` with an unverified draw, for replays of recorded runs whose draws are Fiat–Shamir on the commitment
    (`pearl_c4_replay`): the beacon's round must only exceed `committed_round`.  It supports no claim of
    `beacon-unpredictability`."""
    return _audit(scheme, epoch, prover, transcript, splits, records, committed_round, beacon, t, split_check, tile_map)


def _audit(scheme: C.PearlC, epoch: Epoch, prover: Prover, transcript: Transcript, splits: Mapping[int, Split],
           records: Sequence[UnitRecord], committed_round: int, beacon: Beacon, t: int,
           split_check: Callable[[int, Split], bool] | None, tile_map: Callable, beacon_reason: str | None = None
           ) -> WorkAudit:
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
    if reason is None:
        reason = beacon_reason
    if reason is None and len(transcript.activation_roots) != len(records):
        reason = "the transcript does not have one activation root per record"
    if reason is None and beacon.round <= committed_round:
        reason = "the draws' beacon round is not after the commitment"
    if reason is None:
        for w, sp in sorted(splits.items()):
            if sp.side and (split_check is None or not split_check(w, sp)):
                reason = f"weight {w}: S is not the pinned rule's on its calibration data"
                break
    if reason is not None:
        return reject(reason)
    verifier = Verifier(_TileRuleView(scheme), epoch)
    digest = records_root(epoch.salt, epoch.weights_root, splits, records)
    ids = {(ref.matmul, ref.tile.rows[0], ref.tile.cols[0]): ref.id for ref in verifier.refs}
    picks = draw(scheme, records, beacon, transcript.commitment.root, digest, epoch.salt, t) + exclusion_tiles(scheme, records)
    opened = [(pos, tile, prover.open(epoch, ids[(pos, tile.rows[0], tile.cols[0])])) for pos, tile in picks]
    verdicts = list(tile_map(partial(check_drawn, scheme, epoch, transcript, records), opened))
    ok = all(v.ok for v in verdicts)
    profile = WorkProfile(scheme.name, int(w_ref), t, True, scheme.certificate) if ok else None
    return WorkAudit(Audit(ok, tuple(v.tile for v in verdicts), tuple(verdicts), profile), w_ref, credit, job)


def declare(scheme: C.PearlC, epoch: Epoch, activations: Mapping[int, Sequence[bytes]], weights: Mapping[int, Sequence[bytes]],
            splits: Mapping[int, Split], real_rows: Mapping[int, int] | None = None,
            transcript: Transcript | None = None, voluntary_a: Mapping[int, Sequence[int]] | None = None,
            voluntary_b: Mapping[int, Sequence[int]] | None = None) -> list[UnitRecord]:
    """An honest prover's records: each real row and each weight row checked by the per-row rules on the unit's words,
    and, for a scheme with a salted per-row credit rule (`r1_rows`, Pearl-C4's R1), by that rule on the committed
    transcript's forming.  Pearl-C's schemes bring none, so `transcript` changes nothing for them.  `voluntary_a` (by
    matmul index) and `voluntary_b` (by weight) are the rows the prover gives up (V-EX, `volunteer`); a row the rules
    exclude stays excluded."""
    parse, passes = _rules(scheme)
    k_of = {u.weight: u.shape.k for u in epoch.workload.matmuls}
    excluded_b = {w: {j for j, r in enumerate(rows) if not passes(parse(r, k_of[w]))} for w, rows in weights.items()}
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
        vb = tuple(sorted(set((voluntary_b or {}).get(u.weight, ())) - excluded_b[u.weight]))
        out.append(UnitRecord(u, real, splits[u.weight].k_kept, excluded_a, tuple(sorted(excluded_b[u.weight])), va, vb))
    return out


def volunteer(scheme: C.PearlC, epoch: Epoch, transcript: Transcript, records: Sequence[UnitRecord],
              activations: Mapping[int, Sequence[bytes]], weights: Mapping[int, Sequence[bytes]]
              ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """V-EX's honest remedy, before the draws: (voluntary A rows by matmul index, voluntary B rows by weight) such that
    no tile's credited debit passes its cap, for `declare`.  The prover knows the salt when it forms and commits, so it
    replays every tile's debit as the verifier will.  For a tile over the cap it gives up the credited row whose removal
    leaves the least debit over the cap, and repeats while the tile is over and the count caps allow; a tile the caps
    leave over the cap stays so, and fails if drawn.  Rows go up to ⌈m/64⌉ per unit and ⌈n/64⌉ per weight."""
    va: dict[int, set[int]] = {r.matmul.index: set(r.voluntary_a) for r in records}
    vb: dict[int, set[int]] = {}
    for r in records:
        vb.setdefault(r.matmul.weight, set(r.voluntary_b))
    tol = 1 + Fraction(getattr(scheme, "r1_tolerance", 0))
    changed = True
    while changed:
        # a B row given up changes every tile of its weight, and can lift one that passed over its cap
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
