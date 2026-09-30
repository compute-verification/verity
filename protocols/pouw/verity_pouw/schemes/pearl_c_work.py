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
  - A cell's draw weight is its row's share of `wref` at |K|: Lean's `wrefOf`, with no weight side (F2), so padding to
    `k_unit` and filler earn nothing.
  - An accepted unit is credited (1 − ρ)·`creditOf` over its credited cells.
* **The tile rule (`check_opened`)**, beside `verity_pouw.audit.Verifier.check_tile`'s openings and digest:
  - filler rows are zero;
  - every opened row is zero past |K| (F7);
  - credited rows pass the per-row rules, and excluded rows fail them;
  - the debit replayed over the credited words (`pearl_c_debit`) is at most ρ·`creditOf` of the tile (F1).
  A failure is a wrong tile, reported as a reason; nothing raises.
* **The draws (F8)** are keyed by a beacon round published after the verifier recorded the prover's commitment (the
  transcript root and the records root), through `verity.randomness.derive`.
* **Other row formats.** A scheme with its own row encoding, per-row rules and debit (`pearl_c4`) brings
  `parse_row(row, k)`, `row_passes(words)` and `tile_cap(record, ops, rows, cols)`; Pearl-C v1's are this module's.
"""

from __future__ import annotations

import struct
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import accumulate
from math import lcm

from verity.randomness import derive

from ..audit import Audit, Epoch, Prover, Transcript, Verdict, Verifier, WorkProfile
from ..protocol import Context, Matmul, Shape, Tile, tagged
from . import pearl_c as C
from . import pearl_c_debit as D
from . import pearl_kw as P

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

    def __post_init__(self) -> None:
        s = self.matmul.shape
        if not (0 < self.real_rows <= s.m and 0 < self.k_kept <= s.k):
            raise ValueError("a record's real rows and kept width lie within its shape")
        for rows, bound in ((self.excluded_a, self.real_rows), (self.excluded_b, s.n)):
            if any(not 0 <= r < bound for r in rows) or any(a >= b for a, b in zip(rows, rows[1:])):
                raise ValueError("excluded rows are strictly increasing and in range")

    def encode(self) -> bytes:
        u, s = self.matmul, self.matmul.shape
        a, b = self.excluded_a, self.excluded_b
        return struct.pack(f"<QQIIIIIII{len(a)}I{len(b)}I", u.index, u.weight, s.m, s.k, s.n, self.real_rows, self.k_kept,
                           len(a), len(b), *a, *b)

    def credited(self, tile: Tile) -> tuple[list[int], list[int]]:
        """The tile's credited rows and columns: real, not excluded."""
        ea, eb = set(self.excluded_a), set(self.excluded_b)
        return [i for i in tile.rows if i < self.real_rows and i not in ea], [j for j in tile.cols if j not in eb]


def records_root(salt: bytes, weights_root: bytes, splits: Mapping[int, Split], records: Sequence[UnitRecord]) -> bytes:
    return tagged(RECORDS_TAG, salt, weights_root, *(p for w in sorted(splits) for p in (w, splits[w].encode())),
                  *(r.encode() for r in records))


def check_records(epoch: Epoch, splits: Mapping[int, Split], records: Sequence[UnitRecord]) -> str | None:
    """The records against the workload and the splits: one per matmul in order, k = the split's `k_unit`, |K| its
    `k_kept`, and one set of excluded B rows per weight."""
    if [r.matmul for r in records] != list(epoch.workload.matmuls):
        return "the records are not the workload's matmuls in order"
    if set(splits) != set(epoch.workload.weights()):
        return "every weight has a split, and only those"
    excluded_b: dict[int, tuple[int, ...]] = {}
    for r in records:
        sp = splits[r.matmul.weight]
        if (r.matmul.shape.k, r.k_kept) != (sp.k_unit, sp.k_kept):
            return f"matmul {r.matmul.index}: k and |K| are not its weight's split"
        if excluded_b.setdefault(r.matmul.weight, r.excluded_b) != r.excluded_b:
            return f"weight {r.matmul.weight}: its units declare different excluded rows"
    return None


def cell_wref(scheme: C.PearlC, record: UnitRecord) -> Fraction:
    """A cell's share of its row's `wref` at |K| (Lean's `wrefOf` is linear in m)."""
    s = record.matmul.shape
    return Fraction(scheme.wref(Shape(1, record.k_kept, s.n)), s.n)


def cell_credit(scheme: C.PearlC, record: UnitRecord) -> Fraction:
    s = record.matmul.shape
    return Fraction(scheme.credit_of(Shape(1, record.k_kept, s.n)), s.n)


def tile_weight(scheme: C.PearlC, record: UnitRecord, tile: Tile) -> Fraction:
    rows, cols = record.credited(tile)
    return cell_wref(scheme, record) * len(rows) * len(cols)


def tickets(scheme: C.PearlC, records: Sequence[UnitRecord]) -> list[tuple[int, Tile, int]]:
    """(record position, tile, integer ticket) for every tile of every record, by its credited cells' `wref`."""
    weights = [(p, t, tile_weight(scheme, r, t)) for p, r in enumerate(records) for t in scheme.tiles(r.matmul.shape)]
    den = lcm(*(w.denominator for _, _, w in weights)) if weights else 1
    return [(p, t, int(w * den)) for p, t, w in weights]


def work(scheme: C.PearlC, records: Sequence[UnitRecord]) -> tuple[Fraction, Fraction]:
    """(W, credit): the credited cells' `wref`, and (1 − ρ)·`creditOf` over them."""
    cells = [(r, len(rows) * len(cols)) for r in records for rows, cols in (r.credited(t) for t in scheme.tiles(r.matmul.shape))]
    return (sum((cell_wref(scheme, r) * c for r, c in cells), Fraction(0)),
            (1 - D.CAP) * sum((cell_credit(scheme, r) * c for r, c in cells), Fraction(0)))


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
    """(debit, cap) over credited words given as their A′ and B̃ code rows: `debitOf` of the replayed debit, and
    ρ·`creditOf` of the tile at |K|."""
    debit = D.debit_units(D.chain_atoms(a_codes, b_codes), D.forming_elements(a_codes, b_codes))
    return debit, D.CAP * scheme.credit_of(Shape(len(a_codes), record.k_kept, len(b_codes)))


def _rules(scheme) -> tuple[Callable[[bytes, int], list[int]], Callable[[Sequence[int]], bool]]:
    """(parse_row, row_passes): the scheme's own when it brings them, else Pearl-C v1's."""
    return getattr(scheme, "parse_row", C.parse_f32_row), getattr(scheme, "row_passes", row_passes)


def check_opened(scheme: C.PearlC, record: UnitRecord, tile: Tile, a_rows: Sequence[bytes], b_rows: Sequence[bytes],
                 ops) -> str | None:
    """The tile rule on an opened tile whose openings and digest have passed: None, or the reason it fails."""
    k = record.matmul.shape.k
    parse, passes = _rules(scheme)
    ea, eb = set(record.excluded_a), set(record.excluded_b)
    kinds = [("A", i, row, "filler" if i >= record.real_rows else "excluded" if i in ea else "credited")
             for i, row in zip(tile.rows, a_rows, strict=True)]
    kinds += [("B", j, row, "excluded" if j in eb else "credited") for j, row in zip(tile.cols, b_rows, strict=True)]
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
        if passes(words) != (kind == "credited"):
            return f"{side} row {index} is {kind} but {'fails' if kind == 'credited' else 'passes'} the per-row rules"
    rows, cols = record.credited(tile)
    if rows and cols:
        pos_a, pos_b = {i: p for p, i in enumerate(tile.rows)}, {j: p for p, j in enumerate(tile.cols)}
        if hasattr(scheme, "tile_cap"):
            debit, cap = scheme.tile_cap(record, ops, [pos_a[i] for i in rows], [pos_b[j] for j in cols])
        else:
            debit, cap = tile_cap(scheme, record, [ops.a.codes[pos_a[i]] for i in rows],
                                  [ops.b.codes[pos_b[j]] for j in cols])
        if debit > cap:
            return f"the replayed debit {float(debit):.0f} exceeds the cap {float(cap):.0f} (ρ·creditOf of the tile)"
    return None


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
    """An audit by the records: the verdicts, W (the credited cells' `wref`), the credit (1 − ρ)·`creditOf` over them, and
    the weight side once per weight (outside both)."""

    audit: Audit
    wref: Fraction
    credit: Fraction
    job: int


def audit(scheme: C.PearlC, epoch: Epoch, prover: Prover, transcript: Transcript, splits: Mapping[int, Split],
          records: Sequence[UnitRecord], committed_round: int, beacon: Beacon, t: int,
          split_check: Callable[[int, Split], bool] | None = None) -> WorkAudit:
    """Check the splits and the records, draw t tiles by the records' credit with the beacon round after the commitment,
    open one tile per excluded row, and accept iff every opened tile passes `Verifier.check_tile` and `check_opened`.
    `committed_round` is the latest beacon round when the verifier recorded the commitment; `split_check(weight, split)`
    recomputes S from the calibration data by the pinned rule, and is required when any S is nonempty."""
    if getattr(scheme, "forming", "v1") != "v1":
        raise ValueError("the work law is stated for forming v1")
    w_ref, credit = work(scheme, records)
    job = job_costs(scheme, records)

    def reject(reason: str) -> WorkAudit:
        return WorkAudit(Audit(False, (), (Verdict(-1, False, reason),), None), w_ref, credit, job)

    reason = check_records(epoch, splits, records)
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
    verdicts = []
    for pos, tile in picks:
        tid = ids[(pos, tile.rows[0], tile.cols[0])]
        opening = prover.open(epoch, tid)
        v = verifier.check_tile(transcript, opening)
        if v.ok:
            u = epoch.workload.matmuls[pos]
            ctx = Context(epoch.salt, u, transcript.activation_roots[pos], epoch.weight_roots[u.weight])
            a, b = [o.value for o in opening.a], [o.value for o in opening.b]
            reason = check_opened(scheme, records[pos], tile, a, b, scheme.encode(ctx, scheme.noise(ctx), a, b, tile))
            v = v if reason is None else Verdict(tid, False, reason)
        verdicts.append(v)
    ok = all(v.ok for v in verdicts)
    profile = WorkProfile(scheme.name, int(w_ref), t, True, scheme.certificate) if ok else None
    return WorkAudit(Audit(ok, tuple(v.tile for v in verdicts), tuple(verdicts), profile), w_ref, credit, job)


def declare(scheme: C.PearlC, epoch: Epoch, activations: Mapping[int, Sequence[bytes]], weights: Mapping[int, Sequence[bytes]],
            splits: Mapping[int, Split], real_rows: Mapping[int, int] | None = None) -> list[UnitRecord]:
    """An honest prover's records: each real row and each weight row checked by the per-row rules on the unit's words."""
    parse, passes = _rules(scheme)
    k_of = {u.weight: u.shape.k for u in epoch.workload.matmuls}
    excluded_b = {w: tuple(j for j, r in enumerate(rows) if not passes(parse(r, k_of[w])))
                  for w, rows in weights.items()}
    out = []
    for u in epoch.workload.matmuls:
        real = (real_rows or {}).get(u.index, u.shape.m)
        rows = activations[u.index]
        excluded_a = tuple(i for i in range(real) if not passes(parse(rows[i], u.shape.k)))
        out.append(UnitRecord(u, real, splits[u.weight].k_kept, excluded_a, excluded_b[u.weight]))
    return out
