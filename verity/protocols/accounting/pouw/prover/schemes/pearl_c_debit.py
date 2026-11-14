"""Pearl-C's count of inert work, replayed by the verifier on a sampled tile: `Pouw.PearlC.Skip` and `Game.lean`'s `debitOf`.

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
  chain's inert work is additive over words).
* `salt_dead(alpha, x, beta)`: P3's salt-dead predicate (`SaltDead.saltDead`) on each element of a formed A row: its
  A′ code has one value at every noise pair of the support, which by `saltDeadP_iff_ends` is the support's two ends
  agreeing, E′ the row's flat code c_β = e4m3(f32(β·44)) at all 32 positions and F e4m3(±11/4) at all 32, from the
  forming's accumulator f32(α·x) (`circuit.words.ends_agree`, the hidden inert-work bound's bit).
* `forming_elements(flags, dead)`: P3's per-tile rule (`tileDebitDev`'s `elems`): the A′ elements (i, l), l < k, of the
  tile's rows that are salt-dead or whose atom block l/32 every word of the row over the tile's columns flags.  Each
  counts fs·|cols|/n of inert work, the tile's share of an element its row band shares (`clear_work.tile_cap`).
* A word's products that no salt makes count (rev1's `wordMend`, `Mend.lean`), one MAC each:
  - `end_codes(alpha, x, beta)`: each element's codes at the support's plus and minus ends, the two salts at which
    every element's line is the flat pair with F at +11/4 and at −11/4; `row_ends` adds `salt_dead` from them.
  - `cut_flags(acc, a, b)`: per product of one atom step from the accumulator word `acc`, whether its aligned
    contribution at its group's exponent is 0 (`Mend.cuts`: a zero product, or one the adder's truncating shift cuts).
  - `chain_cuts(a, b)`: those flags along a word's chain, from +0 at the first atom of each promotion group.
  - `word_mend(ends_a, ends_b)`: the elements l that are a dead pair (salt-dead in both the A row and the B row) or a
    forced zero (cut at both ends of the support), the one count.
* `tile_debit(a_rows, b_rows, formed, formed_b)`: the chain's inert work, `forming_elements` and the words' `word_mend`, each
  word's flags replayed once; ``formed_b`` is each B row's forming, as ``formed`` is each A row's.
* `debit_units(atoms, elements, prices, device, zeros=…, mends=…)` is rev1's `tileDebitRev1` at a device's W1 prices
  (`pearl_c.Prices`, the device's own by default): `debitDev` (`debitOf` at `Params.pi`, `Costs.adopted` on the H100),
  per flagged atom-word its 32 MACs and a quarter of a promotion add (40 on the H100, 34 on sm_120) and fs per inert
  element (64 and 40; a tile's elements count |cols|/n each), an add per later +0 promotion (32 and 8), and one MAC
  per mend.
* The inert-work bound ρ is the device record's (`Device.cap`; `CAP` is the H100's 1/400, sm_120's is 1/1,000): a tile whose inert work
  exceeds ρ times its credited cells' credit, their share of the unit's `creditDevRev1` (`clear_work.tile_cap`), is
  rejected (`pearlCTilesDevRev1`'s `capOK`, per tile: `Tile.lean`'s `tileShare`).

Each takes the scheme's device (`pearl_c_device`, the H100 by default): its atom, G and prices.  The module constants are
the H100's.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction

from verity.circuits.silicon import fp32
from verity.circuits.silicon.models import group_sum
from verity.circuits.silicon.term import FP32_MAX_EXPONENT, FP32_SIGNIFICAND_WIDTH, Term, fp32_term

from . import pearl_kw as P
from .pearl_c_device import H100, Device, Prices

G = H100.g                             # atoms per promotion group (`Params.pi.G`)
ATOM = H100.atom.k
CAP = H100.cap
ATOM_UNITS = ATOM + Fraction(H100.prices.fadd, G)  # `debitOf`: 32 MACs and 1/G of a promotion add per flagged atom-word


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


#: the salt-dead support's ends (`SaltDead.support`, `Cap.plusEnd` and `minusEnd`): E's flat entry e4m3(44), and F's
#: e4m3(+11/4) and e4m3(−11/4)
E_FLAT = 0x63
F_ENDS = (0x43, 0xC3)


def flat_code(beta: int) -> int:
    """c_β: the E′ code of the support's flat E entry at the row's β (`pearl_c.eprime` of 44)."""
    return P.f32_to_fp8(P.f32_of_bits(fp32.mul(beta, P.f32_bits(P.fp8_to_f32(E_FLAT)))))


def ends_agree(c_plus: int, c_minus: int) -> bool:
    """Whether the two ends' codes have one value: equal, or both zeros (`Cap.codeVal_eq_iff`)."""
    return c_plus == c_minus or not (c_plus | c_minus) & 0x7F


def end_codes(alpha: int, x: Sequence[int], beta: int, device: Device = H100) -> tuple[list[int], list[int]]:
    """Each element's codes at the support's plus and minus ends (`Mend.endRow`): the forming's atom from f32(α·x) over
    [c_β]³² and [±11/4]³², cast to E4M3."""
    cb, k = flat_code(beta), device.atom.k
    seen: dict[int, tuple[int, int]] = {}
    plus, minus = [], []
    for w in x:
        acc = fp32.mul(alpha, w)
        if acc not in seen:
            seen[acc] = tuple(P.f32_to_fp8(P.f32_of_bits(_step(acc, [cb] * k, [f] * k, device))) for f in F_ENDS)
        plus.append(seen[acc][0])
        minus.append(seen[acc][1])
    return plus, minus


def salt_dead(alpha: int, x: Sequence[int], beta: int, device: Device = H100) -> list[bool]:
    """P3's salt-dead predicate on each element of a formed row (module docstring): its two ends' codes from f32(α·x)
    agree."""
    return [ends_agree(p, m) for p, m in zip(*end_codes(alpha, x, beta, device))]


def row_ends(alpha: int, x: Sequence[int], beta: int, device: Device = H100
             ) -> tuple[list[int], list[int], list[bool]]:
    """A formed row's plus-end codes, its minus-end codes (`end_codes`), and whether each element's agree
    (`salt_dead`)."""
    plus, minus = end_codes(alpha, x, beta, device)
    return plus, minus, [ends_agree(p, m) for p, m in zip(plus, minus)]


def _code_term(c: int) -> tuple[bool, int, int]:
    """`Fp8Atom.product`'s factor: an E4M3 code's sign, its exponent or 1 for a subnormal, and its 4-bit significand."""
    e = (c >> 3) & 0xF
    return bool(c & 0x80), max(e, 1), (c & 7) | (8 if e else 0)


def cut_flags(acc: int, a: Sequence[int], b: Sequence[int], device: Device = H100) -> list[bool]:
    """`Mend.cuts`: per product of one atom step from the accumulator word `acc`, whether the adder cuts it, its aligned
    contribution at its group's exponent being 0: each group's exponent is the largest of the incoming value's and the
    group's nonzero non-NaN products', at least the floor, and a product's significand on the adder is shifted right by
    its distance below it (`models.group_sum`)."""
    atom, rescale = device.atom, device.atom.width - FP32_SIGNIFICAND_WIDTH
    value: Term | None = None if (acc >> 23) & 0xFF == 0xFF else fp32_term(acc)
    out: list[bool] = []
    start = 0
    for size in atom.groups:
        prods = []
        for i in range(start, start + size):
            (sa, ea, ma), (sb, eb, mb) = _code_term(a[i]), _code_term(b[i])
            nan = (a[i] & 0x7F) == 0x7F or (b[i] & 0x7F) == 0x7F
            prods.append((sa ^ sb, ea + eb - 14, (ma * mb) << 17, nan))
        top = atom.zero_exponent
        for _, e, mag, nan in prods:
            if mag and not nan and e > top:
                top = e
        if value is not None and value.magnitude and value.exponent > top:
            top = value.exponent
        for _, e, mag, _ in prods:
            scaled = mag << rescale if rescale >= 0 else mag >> -rescale
            out.append((scaled >> (top - e)) == 0)
        if value is not None and not any(nan for *_, nan in prods):
            value = group_sum([value, *(Term(s, e, m) for s, e, m, _ in prods)], width=atom.width,
                              zero_exponent=atom.zero_exponent)
            if value.magnitude and value.exponent > FP32_MAX_EXPONENT:
                value = None
        else:
            value = None  # NaN or an infinity carries through the later groups, and no group aligns to it
        start += size
    return out


def chain_cuts(a: Sequence[int], b: Sequence[int], device: Device = H100) -> list[bool]:
    """`Mend.chainCuts`: the cut flags of the word of code rows `a` and `b` along its chain, from +0 at the first atom of
    each promotion group (with no promotion, only at atom 0); entry 32·t + r is product r of atom t."""
    k, g_ = device.atom.k, device.g
    word, out = 0, []
    for t, s in enumerate(range(0, len(a), k)):
        if t % g_ == 0 if g_ else t == 0:
            word = 0
        out.extend(cut_flags(word, a[s:s + k], b[s:s + k], device))
        word = device.atom.step(word, a[s:s + k], b[s:s + k])
    return out


def word_mend(ends_a: tuple[Sequence[int], Sequence[int], Sequence[bool]],
              ends_b: tuple[Sequence[int], Sequence[int], Sequence[bool]], device: Device = H100) -> int:
    """rev1's `wordMend` on the word of an A row and a B row given by their `row_ends`: the elements that are a dead
    pair (salt-dead in both rows) or a forced zero (cut at both ends of the support)."""
    cp = chain_cuts(ends_a[0], ends_b[0], device)
    cm = chain_cuts(ends_a[1], ends_b[1], device)
    return sum(1 for l, (da, db) in enumerate(zip(ends_a[2], ends_b[2])) if (da and db) or (cp[l] and cm[l]))


    """`tileDebitDev`'s inert forming elements: over the tile's rows i, the l < k with row i's atom block l/32 flagged
    by every word (i, j) of the tile's columns (``flags[i][j]``, `word_flags`), or salt-dead (``dead(i)``, row i's
    `salt_dead`, read only where some word leaves the block unflagged)."""
    elements, atom = 0, device.atom.k
    for i, words in enumerate(flags):
        every = [all(col) for col in zip(*words)]
        if all(every):
            elements += atom * len(every)
            continue
        row_dead = dead(i)
        elements += sum(1 for l, d in enumerate(row_dead) if every[l // atom] or d)
    return elements


def tile_debit(a_rows: Sequence[Sequence[int]], b_rows: Sequence[Sequence[int]],
               formed: Sequence[tuple[int, Sequence[int], int]], formed_b: Sequence[tuple[int, Sequence[int], int]],
               device: Device = H100) -> tuple[int, int, int, int]:
    """(atoms, zeros, elements, mends) over a tile's words: `chain_debit`'s counts, `forming_elements` and the words'
    `word_mend`, each word's flags replayed once; ``formed`` is each A row's (α, x, β), its forming's, and ``formed_b``
    each B row's."""
    flags, atoms, zeros = [], 0, 0
    for a in a_rows:
        row = []
        for b in b_rows:
            f, z = word_debit(a, b, device)
            row.append(f)
            atoms, zeros = atoms + sum(f), zeros + z
        flags.append(row)
    elements = forming_elements(flags, lambda i: salt_dead(formed[i][0], formed[i][1], formed[i][2], device), device)
    ends_a = [row_ends(*f, device) for f in formed]
    ends_b = [row_ends(*f, device) for f in formed_b]
    return atoms, zeros, elements, sum(word_mend(ea, eb, device) for ea in ends_a for eb in ends_b)


def debit_units(atoms: int, elements: Fraction | int, prices: Prices | None = None, device: Device = H100, *, zeros: int,
                mends: int = 0) -> Fraction:
    """rev1's inert work at a device's prices (`tileDebitRev1`): `debitDev` (`debitOf Params.pi Costs.adopted` at the H100's),
    per flagged atom-word its 32 MACs and 1/G of a promotion add and fs per inert element (a tile's at |cols|/n
    each), an add per later promotion into a +0 total, and a MAC per product no salt makes count (`word_mend`);
    `prices` the device's own by default."""
    p = prices or device.prices
    return atom_units(device, prices) * atoms + p.forming_credited * elements + p.fadd * zeros + mends
