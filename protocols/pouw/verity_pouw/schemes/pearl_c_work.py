"""Pearl-C's work law (statement review, item 5): what the verifier credits and draws, recomputed by the verifier.

* `UnitRecord`: what the prover declares for each matmul before the draws.
  - its real rows (the rest are filler padding the unit to the kernel's granularity);
  - its kept width |K| (the rest of k is zero padding);
  - the rows that fail the scheme's per-row rules: A rows of the matmul, and B rows of its weight.
  `records_root` binds the records, and the draw key reads it.
* Credit.
  - A tile weighs the unit's `wref` at its real rows and kept width, times the tile's share of the real cells.
  - A tile of filler only, or with a declared failing row, weighs 0 and is never drawn. Padding earns nothing, and an
    uncredited tile costs the prover its credit, not the audit.
* `check_drawn`: the verifier's rule on a drawn tile's opened rows.
  - The tile is creditable under its record.
  - The opened real A rows and the B rows are zero past |K|.
  - Each of them passes the per-row rules.
  Filler rows are unconstrained: they earn nothing, and a zero row fails the noise floor.  A failure is a wrong tile,
  reported as a reason; nothing raises.
* `declare` builds an honest prover's records.
* `audit` draws by the records and checks each drawn tile with `verity_pouw.audit.Verifier.check_tile`, whose
  admissibility over the whole tile is replaced by `check_drawn`.
"""
from __future__ import annotations

import struct
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import accumulate
from math import lcm

from verity.randomness import derive

from ..audit import AUDIT_DOMAIN, Audit, Epoch, Prover, Transcript, Verdict, Verifier, WorkProfile
from ..protocol import Matmul, Shape, Tile, tagged
from . import pearl_c as C
from . import pearl_kw as P

RECORDS_TAG = "verity/pouw/pearl-c/records/v1"


@dataclass(frozen=True)
class UnitRecord:
    """One matmul as its prover declares it before the draws."""

    matmul: Matmul                 # index, (m, k, n), weight
    real_rows: int                 # rows 0 … real_rows − 1 are the call's; the rest are filler
    k_kept: int                    # entries 0 … k_kept − 1 of every row are the call's; the rest are zero padding
    fail_a: tuple[int, ...] = ()   # A rows (< real_rows) that fail the per-row rules
    fail_b: tuple[int, ...] = ()   # B rows of the weight that fail them

    def __post_init__(self) -> None:
        s = self.matmul.shape
        if not (0 < self.real_rows <= s.m and 0 < self.k_kept <= s.k):
            raise ValueError("a record's real rows and kept width lie within its shape")
        for rows, bound in ((self.fail_a, self.real_rows), (self.fail_b, s.n)):
            if any(not 0 <= r < bound for r in rows) or any(a >= b for a, b in zip(rows, rows[1:])):
                raise ValueError("declared failing rows are strictly increasing and in range")

    def encode(self) -> bytes:
        u, s = self.matmul, self.matmul.shape
        return struct.pack(f"<QQIIIIIII{len(self.fail_a)}I{len(self.fail_b)}I", u.index, u.weight, s.m, s.k, s.n, self.real_rows,
                           self.k_kept, len(self.fail_a), len(self.fail_b), *self.fail_a, *self.fail_b)


def records_root(salt: bytes, weights_root: bytes, records: Sequence[UnitRecord]) -> bytes:
    return tagged(RECORDS_TAG, salt, weights_root, *(r.encode() for r in records))


def credited(record: UnitRecord, tile: Tile) -> bool:
    fa, fb = set(record.fail_a), set(record.fail_b)
    return tile.rows[0] < record.real_rows and not fa.intersection(tile.rows) and not fb.intersection(tile.cols)


def tile_weight(scheme: C.PearlC, record: UnitRecord, tile: Tile) -> Fraction:
    """The unit's `wref` at its real rows and kept width, times the tile's share of the real cells; 0 when uncredited."""
    if not credited(record, tile):
        return Fraction(0)
    s = record.matmul.shape
    real = sum(1 for r in tile.rows if r < record.real_rows)
    return Fraction(scheme.wref(Shape(record.real_rows, record.k_kept, s.n)) * real * len(tile.cols), record.real_rows * s.n)


def tickets(scheme: C.PearlC, records: Sequence[UnitRecord]) -> list[tuple[int, Tile, int]]:
    """(record position, tile, integer ticket) for every tile of every record; uncredited and filler-only tiles get 0."""
    weights = [(p, t, tile_weight(scheme, r, t)) for p, r in enumerate(records) for t in scheme.tiles(r.matmul.shape)]
    den = lcm(*(w.denominator for _, _, w in weights)) if weights else 1
    return [(p, t, int(w * den)) for p, t, w in weights]


def work(scheme: C.PearlC, records: Sequence[UnitRecord]) -> Fraction:
    return sum((tile_weight(scheme, r, t) for r in records for t in scheme.tiles(r.matmul.shape)), Fraction(0))


def draw(scheme: C.PearlC, records: Sequence[UnitRecord], source: bytes, transcript_root: bytes, records_digest: bytes,
         salt: bytes, t: int) -> list[tuple[int, Tile]]:
    """t work-weighted draws with replacement over the credited tiles, keyed by the transcript and the records."""
    tk = [x for x in tickets(scheme, records) if x[2] > 0]
    if not tk:
        return []
    key = derive(source, AUDIT_DOMAIN, {"transcript": transcript_root, "records": records_digest, "salt": salt})
    cumulative = list(accumulate(w for _, _, w in tk))
    return [(tk[i][0], tk[i][1]) for i in (bisect_right(cumulative, key.uniform(cumulative[-1], j)) for j in range(t))]


def row_passes(words: Sequence[int]) -> bool:
    """The scheme's per-row rules: the noise floor ρ·α ≥ 1 (exact product) and `pearl_c.live_row`."""
    _, rho, alpha, _ = C.s5_stats(words)
    return P.f32_of_bits(rho) * P.f32_of_bits(alpha) >= P.SIGMA_MIN and C.live_row(words)


def check_drawn(record: UnitRecord, tile: Tile, a_rows: Sequence[bytes], b_rows: Sequence[bytes]) -> str | None:
    """The verifier's rule on a drawn tile's opened rows: None if the record's claim holds, else the reason."""
    if not credited(record, tile):
        return "the tile is not creditable, so it cannot be drawn"
    k = record.matmul.shape.k
    real_a = [row for i, row in zip(tile.rows, a_rows, strict=True) if i < record.real_rows]
    for which, rows in (("A", real_a), ("B", b_rows)):
        for row in rows:
            try:
                words = C.parse_f32_row(row, k)
            except P.Reject as e:
                return f"an opened {which} row is malformed: {e}"
            if any(words[record.k_kept:]):
                return f"an opened {which} row is not zero past the kept width"
            if not row_passes(words):
                return f"an opened {which} row fails the per-row rules but is not declared"
    return None


class _RowsByRecord:
    """The scheme with its tile-wide admissibility left to `check_drawn`, which applies the per-row rules to real rows."""

    def __init__(self, scheme: C.PearlC):
        self._scheme = scheme

    def __getattr__(self, name: str):
        return getattr(self._scheme, name)

    def admissible(self, ctx, ops) -> bool:
        return True


def declare(scheme: C.PearlC, epoch: Epoch, activations: Mapping[int, Sequence[bytes]], weights: Mapping[int, Sequence[bytes]],
            real_rows: Mapping[int, int] | None = None, k_kept: Mapping[int, int] | None = None) -> list[UnitRecord]:
    """An honest prover's records: every real row checked by the per-row rules (filler rows are not declared)."""
    fail_b = {w: tuple(j for j, r in enumerate(rows) if not row_passes(C.parse_f32_row(r, len(r) // 4)))
              for w, rows in weights.items()}
    out = []
    for u in epoch.workload.matmuls:
        real = (real_rows or {}).get(u.index, u.shape.m)
        rows = activations[u.index]
        fail_a = tuple(i for i in range(real) if not row_passes(C.parse_f32_row(rows[i], u.shape.k)))
        out.append(UnitRecord(u, real, (k_kept or {}).get(u.weight, u.shape.k), fail_a, fail_b[u.weight]))
    return out


def audit(scheme: C.PearlC, epoch: Epoch, prover: Prover, transcript: Transcript, records: Sequence[UnitRecord], source: bytes,
          t: int) -> Audit:
    """Draw t tiles by the records' credit, and accept iff every drawn tile passes `Verifier.check_tile` and `check_drawn`."""
    verifier = Verifier(_RowsByRecord(scheme), epoch)
    if [r.matmul for r in records] != list(epoch.workload.matmuls) or len(transcript.activation_roots) != len(records):
        return Audit(False, (), (), None)
    digest = records_root(epoch.salt, epoch.weights_root, records)
    ids = {(ref.matmul, ref.tile.rows[0], ref.tile.cols[0]): ref.id for ref in verifier.refs}
    picks = draw(scheme, records, source, transcript.commitment.root, digest, epoch.salt, t)
    verdicts = []
    for pos, tile in picks:
        tid = ids[(pos, tile.rows[0], tile.cols[0])]
        opening = prover.open(epoch, tid)
        v = verifier.check_tile(transcript, opening)
        if v.ok:
            reason = check_drawn(records[pos], tile, [o.value for o in opening.a], [o.value for o in opening.b])
            v = v if reason is None else Verdict(tid, False, reason)
        verdicts.append(v)
    ok = all(v.ok for v in verdicts)
    credit = work(scheme, records)
    profile = WorkProfile(scheme.name, int(credit), len(verdicts), True, scheme.certificate) if ok else None
    return Audit(ok, tuple(v.tile for v in verdicts), tuple(verdicts), profile)
