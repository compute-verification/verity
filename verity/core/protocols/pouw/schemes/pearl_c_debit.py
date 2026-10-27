"""Pearl-C's debit, replayed by the verifier on a drawn tile: `Pouw.PearlC.Skip` and `Game.lean`'s `debitOf`.

* `word_flags(a, b)` is `Skip.wordFlags` over a word's k/32 atoms (a, b: the word's A′ row and B̃ row as E4M3 codes).
  - The chain runs in groups of four atoms (`Params.pi.G`), each from +0, and each group's word is promoted into an FP32
    total by one nearest-even add.
  - Atom t is flagged when some set of its group's atoms containing it, skipped together in place (a skipped atom passes
    its incoming word through), leaves the word's output unchanged. Each of the group's 2^4 masked words is promoted into
    the total before the group and carried through the later promotions (`jointFlags`).
  - The later promotions are monotone in the total, so the totals after group g that they carry to the output form an
    interval. It is found backward by bisection over FP32 order (`Freeze.lean`'s certificates, `jointFlagsFast_eq`).
  - A word's flags read only its two rows over the full k, so a tile's openings are enough.
* `word_debit(a, b)` is a word's flags and its later promotions into a +0 total, rev1's `laterZeroPromos`
  (`DeviceRev1.lean`): the g ≥ 1 whose total after g groups is ±0, so that promotion g + 1 returns its group's word and a
  program may skip it. The first promotion of every word adds into T_0 = +0; rev1 takes it out of the credit instead
  (`pearl_c.PearlC.first_adds`).
* `chain_debit(a_rows, b_rows)`: the flagged atoms and the later +0 promotions, each summed over the tile's words (the
  chain debit is additive over words).
* `forming_elements(...)`: **a stub that returns 0.** P3's per-tile rule is still being written: an A′ element is shared by
  every tile of its row band, and P3's owner is stating how a per-tile `needed` and the element's credit are apportioned.
  Until it is relayed, the replayed debit is the chain debit alone.
* `debit_units(atoms, elements, prices, device, zeros=…)` is rev1's `tileDebitRev1` at a device's W1 prices
  (`pearl_c.Prices`, the device's own by default): `debitDev` (`debitOf` at `Params.pi`, `Costs.adopted` on the H100),
  per flagged atom-word its 32 MACs and a quarter of a promotion add (40 on the H100, 34 on sm_120) and fs per debited
  element (64 and 40), and an add per later +0 promotion (32 and 8).
* The cap ρ is the device record's (`Device.cap`; `CAP` is the H100's 1/400, sm_120's is 1/1,000): a tile whose debit
  exceeds ρ times its credited cells' credit, their share of the unit's `creditDevRev1` (`clear_work.tile_cap`), is
  rejected (`pearlCTilesDevRev1`'s `capOK`, per tile: `Tile.lean`'s `tileShare`).

Each takes the scheme's device (`pearl_c_device`, the H100 by default): its atom, G and prices.  The module constants are
the H100's.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from verity.ml.circuits.silicon import fp32

from . import pearl_kw as P
from .pearl_c_device import H100, Device, Prices

G = H100.g                             # atoms per promotion group (`Params.pi.G`)
ATOM = H100.atom.k
CAP = H100.cap
ATOM_UNITS = ATOM + Fraction(H100.prices.fadd, G)  # `debitOf`: 32 MACs and 1/G of a promotion add per flagged atom-word
ELEMENT_UNITS = H100.prices.forming_credited       # `Costs.adopted`'s fs per debited forming element


def _promoted(device: Device) -> int:
    if device.g is None:
        raise NotImplementedError(f"{device.name}: Skip's debit is stated for promotion groups; no rule without promotion")
    return device.g


def atom_units(device: Device = H100, prices: Prices | None = None) -> Fraction:
    """`debitOf`'s price of one flagged atom-word on `device` at `prices` (its own by default): its MACs and 1/G of a
    promotion add."""
    return device.atom.k + Fraction((prices or device.prices).fadd, _promoted(device))
_TOP = 0x7F7FFFFF                      # the largest finite FP32 magnitude


def _key(word: int) -> int:
    """An FP32 word as an integer in value order; ±0 are one key."""
    return -(word & 0x7FFFFFFF) if word >> 31 else word


def _word(key: int) -> int:
    return (-key) | 0x80000000 if key < 0 else key


def _promote(total: int, word: int) -> int:
    out = fp32.add(total, word)
    if (out >> 23) & 0xFF == 0xFF:
        raise P.Reject("a promotion left the finite range")
    return 0 if out == 0x80000000 else out


def _step(word: int, a: Sequence[int], b: Sequence[int], device: Device) -> int:
    out = device.atom.step(word, a, b)
    if (out >> 23) & 0xFF == 0xFF:
        raise P.Reject("an atom left the finite range")
    return out


def _carried(x: int, w: int) -> int:
    """The key of RN(word(x) + w); an overflow keys past every finite value (±∞'s own keys), so the map stays monotone."""
    out = fp32.add(_word(x), w)
    return 0 if out == 0x80000000 else _key(out)


def _least(pred) -> int:
    """The least key x in the finite FP32 range with pred(x) (pred monotone), or one past the top if there is none."""
    if not pred(_TOP):
        return _TOP + 1
    lo, hi = -_TOP, _TOP
    while lo < hi:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _masked(atoms: Sequence[tuple[Sequence[int], Sequence[int]]], device: Device) -> dict[int, int]:
    """A group's word from +0 under each mask of skipped atoms (bit j: atom j passes its incoming word through)."""
    words = {0: 0}
    for j, (a, b) in enumerate(atoms):
        nxt = {}
        for mask, w in words.items():
            nxt[mask] = _step(w, a, b, device)
            nxt[mask | 1 << j] = w
        words = nxt
    return words


def word_flags(a: Sequence[int], b: Sequence[int], device: Device = H100) -> list[bool]:
    """`Skip.wordFlags k a b t` for t < k/32."""
    return word_debit(a, b, device)[0]


def word_debit(a: Sequence[int], b: Sequence[int], device: Device = H100) -> tuple[list[bool], int]:
    """A word's flags (`word_flags`) and its later promotions into a +0 total, rev1's `laterZeroPromos`: the g in
    [1, k/(32·G)) whose total after g groups is ±0, so that promotion g + 1 returns its group's word."""
    g_, atom = _promoted(device), device.atom.k
    if len(a) != len(b) or len(a) % (g_ * atom):
        raise P.Reject("a word's rows are whole promotion groups")
    atoms = [(a[s:s + atom], b[s:s + atom]) for s in range(0, len(a), atom)]
    groups = [_masked(atoms[g:g + g_], device) for g in range(0, len(atoms), g_)]
    totals = [0]
    for masked in groups:
        totals.append(_promote(totals[-1], masked[0]))
    lo = hi = _key(totals[-1])
    bounds = [(lo, hi)]
    for g in range(len(groups) - 1, 0, -1):
        w = groups[g][0]
        lo, hi = _least(lambda x, w=w, lo=lo: _carried(x, w) >= lo), _least(lambda x, w=w, hi=hi: _carried(x, w) > hi) - 1
        bounds.append((lo, hi))
    bounds.reverse()
    flags = []
    for g, masked in enumerate(groups):
        lo, hi = bounds[g]
        keeps = [mask for mask, w in masked.items() if mask and lo <= _key(_promote(totals[g], w)) <= hi]
        flags.extend(any(mask >> j & 1 for mask in keeps) for j in range(g_))
    return flags, sum(1 for t in totals[1:-1] if t == 0)


def chain_debit(a_rows: Sequence[Sequence[int]], b_rows: Sequence[Sequence[int]], device: Device = H100
                ) -> tuple[int, int]:
    """(atoms, zeros) over every word (a_row, b_row) of a tile, both additive over words: the flagged atoms
    (`Skip.chainDebit`'s count) and the later promotions into a +0 total (`laterZeroPromos`)."""
    atoms = zeros = 0
    for a in a_rows:
        for b in b_rows:
            flags, z = word_debit(a, b, device)
            atoms, zeros = atoms + sum(flags), zeros + z
    return atoms, zeros


def forming_elements(a_rows: Sequence[Sequence[int]], b_rows: Sequence[Sequence[int]]) -> int:
    """STUB, pending P3's per-tile rule (module docstring): the debited A′ elements of a tile.  Returns 0."""
    return 0


def debit_units(atoms: int, elements: int, prices: Prices | None = None, device: Device = H100, *, zeros: int
                ) -> Fraction:
    """rev1's debit at a device's prices (`tileDebitRev1`): `debitDev` (`debitOf Params.pi Costs.adopted` at the H100's),
    per flagged atom-word its 32 MACs and 1/G of a promotion add and fs per debited element, and an add per later
    promotion into a +0 total; `prices` the device's own by default."""
    p = prices or device.prices
    return atom_units(device, prices) * atoms + p.forming_credited * elements + p.fadd * zeros
