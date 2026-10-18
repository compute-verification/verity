"""RT10's repro against compute-accounting's padded-window credit rule (note:20261018T2200Z-draft-padded-credit-rule,
its Lean statement on cursor/padded-credit-stmt-ea24 at 5d09a7396, and its 21:58Z revision in thread 1792358147.145339).
Red-team evidence for note:proofs/20261018T2245Z-report-rt10-padded-credit, not a test of record: each test pins one
of its findings with the real Pearl-C v1 scheme or with exact draw probabilities.

* A padding slot's honest row is the zero row. The forming refuses it (ok 0), and every checked word of a tile over it
  is +0, so a wrong ok of 1 admits tiles that are right at their committed operands and cost no chain.
* Under the statement's value-independent draw (`DrawnLayer.draw : B → Fin s → J`), wrong row units that admit those
  tiles escape at their own count's rate, while the credit they add is n_c tiles each.
* Under the revision's draw weighted by the credit each unit gates at the committed values, a slot whose annotation is
  committed 0 is never drawn, nor is its annotation unit, nor a selection that gates egress and no credit.
* Under that draw the tally is one unit of weight Cr among about 5·Cr, so "drawn with probability ≈ 1" holds only when
  K′ is sized over the whole draw weight, not over the credited work as `window.k_prime` sizes it.
* The ok of a row is a function of its FP32 row alone, but it is committed in the side row next to codes that read the
  noise, so it is committed after every tile's noise is fixed, and the noise is a function of the prover's root.
"""
from __future__ import annotations

import hashlib
import random
import struct
from fractions import Fraction

from verity.core.protocols.pouw.protocol import Context, Matmul, Shape, Tile
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes import pearl_kw as P
from verity.core.protocols.pouw.window import k_prime

M, K, N = 16, 128, 16
EPS, DELTA = Fraction(1, 10), Fraction(1, 2 ** 20)


def _f32(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _rows(rng: random.Random, count: int, sd: float) -> list[bytes]:
    return [struct.pack(f"<{K}I", *[_f32(rng.gauss(0, sd)) for _ in range(K)]) for _ in range(count)]


def _call(salt: bytes = b"salt", root_a: bytes = b"root_a", live: int = 1):
    """One call of M batch slots, `live` of them a real activation row and the rest the zero row (STATIC-WIN's empty
    slot), times N weight rows, through Pearl-C v1's forming and tile check."""
    rng = random.Random(7)
    s = C.PearlC(k_min=32, tile=(16, 16), forming="v1")
    ctx = Context(hashlib.sha256(salt).digest(), Matmul(3, Shape(M, K, N), 9), hashlib.sha256(root_a).digest(),
                  hashlib.sha256(b"root_b").digest())
    a = _rows(rng, live, 1.0) + [bytes(4 * K)] * (M - live)
    b = _rows(rng, N, 0.05)
    noise = s.noise(ctx)
    ops = s.encode(ctx, noise, a, b, Tile(tuple(range(M)), tuple(range(N))))
    ok = [P.f32_of_bits(r) * P.f32_of_bits(al) >= P.SIGMA_MIN and C.live_row(x)
          for r, al, x in zip(ops.a.rho, ops.a.alpha, ops.a.x)]
    return s, ctx, noise, ops, ok


def test_a_zero_padding_row_is_refused_and_its_tiles_cost_nothing():
    """(c): the forming refuses the zero row, so its honest ok is 0. A wrong ok of 1 admits its tiles, and they are
    right at their committed operands with no work: the row forms to zero codes (β = ρ·α·LINE = 0 scales the noise line
    to zero), and every C̃ and U word of the row is +0. `Pc8TileSide` uses none of the index field, so drawing such a
    tile passes; only drawing the row unit catches the wrong ok."""
    s, ctx, _, ops, ok = _call()
    assert ok == [True] + [False] * (M - 1)
    assert not s.admissible(ctx, ops)
    checked = s.checked(ctx, ops)
    for i in range(1, M):
        assert set(ops.a.codes[i]) == {0}
        assert set(checked.c[i]) == {0} and set(checked.u[i]) == {0}
    assert set(checked.c[0]) != {0}


def _escape_without_replacement(population: int, wrong: int, draws: int) -> float:
    out = 1.0
    for i in range(draws):
        out *= (population - wrong - i) / (population - i)
    return out


def test_the_statements_uniform_draw_prices_a_wrong_row_unit_at_one_unit():
    """(c) under the Lean statement's draw, which reads only the draw coin: one call at batch 1 of 16 with n_c = 4096
    weight rows. Fifteen wrong Pc8RowA units (ok 1 on the fifteen zero rows) add 15·n_c credited tiles at no work: 16x
    credit. K′ is `window.k_prime` at the claimed credit over the population. The fifteen escape with probability above
    0.96; skipping the same 15·n_c tiles escapes with probability below 2^-400."""
    m, n = 16, 4096
    population = m * n + m + n
    claimed = m * n
    draws = k_prime(EPS, DELTA, claimed, population, 1)
    assert draws == 141
    assert _escape_without_replacement(population, 15, draws) > 0.96
    assert _escape_without_replacement(population, 15 * n, draws) < 2.0 ** -400


def _weighted(ok_a: list[int], ok_b: list[int], admit: list[int], w: int = 1) -> dict[str, int]:
    """The revision's draw weights at the committed values: a tile its credit when admit ∧ ok_A ∧ ok_B, an annotation
    the credit of the tiles it gates, the tally the whole credit, and a selection that gates egress and no credit 0."""
    sel = {(i, j): admit[i] & ok_a[i] & ok_b[j] for i in range(len(ok_a)) for j in range(len(ok_b))}
    cr = w * sum(sel.values())
    weights = {f"tile{i},{j}": w * v for (i, j), v in sel.items()}
    weights |= {f"okA{i}": w * sum(sel[i, j] for j in range(len(ok_b))) for i in range(len(ok_a))}
    weights |= {f"okB{j}": w * sum(sel[i, j] for i in range(len(ok_a))) for j in range(len(ok_b))}
    weights |= {f"admit{i}": w * sum(sel[i, j] for j in range(len(ok_b))) for i in range(len(ok_a))}
    weights |= {"tally": cr, "egress_gate0": 0}
    return weights


def test_the_revisions_weighted_draw_never_draws_a_slot_marked_dead():
    """(a) and (b) under the 21:58Z revision: row 0 is live and admitted, and the prover commits its ok_A as 0 (one wrong
    forming unit). At the committed values its tiles weigh 0 and so does its annotation unit, so neither is drawn with
    any number of draws. A selection that gates only egress weighs 0 too. A static annotation weight (the credit it
    would gate at ok 1) draws it."""
    admit, ok_b = [1, 1, 0, 0], [1, 1, 1, 1]
    honest = _weighted([1, 1, 0, 0], ok_b, admit)
    marked = _weighted([0, 1, 0, 0], ok_b, admit)
    total = sum(marked.values())
    dead = [f"tile0,{j}" for j in range(4)] + ["okA0"]
    for draws in (1, 141, 10 ** 6):
        assert (1 - Fraction(sum(marked[u] for u in dead), total)) ** draws == 1
        assert (1 - Fraction(marked["egress_gate0"], total)) ** draws == 1
    assert honest["okA0"] == 4 and marked["okA0"] == 0
    assert marked["tally"] == honest["tally"] - 4


def test_the_tally_is_drawn_with_probability_about_one_only_with_draws_sized_over_the_whole_weight():
    """(d) under the revision: with the annotations and the tally in the population at their weights, the draw weight is
    W = 5·Cr (tiles, ok_A, ok_B, admit, tally). K′ sized over the credited work (`window.k_prime` with n·w_max the
    credited work, p = ε) at ε = 1/2 is 20 draws: the tally escapes with probability (4/5)^20 > 2^-7, and a wrong ε share
    of the tiles escapes with probability (9/10)^20 > 2^-4, both far above δ = 2^-20. Sized over W (p = ε/5) it is 132
    draws, and both are below δ. (`k_prime` caps K at its population, so it is read at a million credited units in the
    same ratio.)"""
    eps = Fraction(1, 2)
    weights = _weighted([1] * 4, [1] * 4, [1] * 4)
    cr, total = weights["tally"], sum(weights.values())
    assert total == 5 * cr
    scale = 10 ** 6
    over_credit = k_prime(eps, DELTA, scale, scale, 1)
    over_weight = k_prime(eps, DELTA, scale, scale * total // cr, 1)
    assert (over_credit, over_weight) == (20, 132)
    tally_miss = 1 - Fraction(cr, total)
    tiles_miss = 1 - eps * Fraction(cr, total)
    assert tally_miss ** over_credit > Fraction(1, 2 ** 7) and tiles_miss ** over_credit > Fraction(1, 2 ** 4)
    assert tally_miss ** over_weight < DELTA and tiles_miss ** over_weight < DELTA


def test_ok_is_fixed_by_the_fp32_row_but_committed_after_the_noise():
    """(e): another salt leaves every row's ok unchanged (ok reads only the FP32 row: finiteness, ρ·α ≥ 1, dead lanes)
    and changes the live row's codes, which share its side row with ok (`[codes ‖ P ‖ I lo, I hi, ok, 0 × 5]`). So the
    position holding ok is committed after the tiles' noise, and in the statement's terms every tile is `Exposed` and
    `CandidatesEarly` fails on its code slots. The noise seed is a function of the prover's own root_a."""
    _, _, noise, ops, ok = _call()
    _, _, _, ops2, ok2 = _call(salt=b"another salt")
    assert ok == ok2
    assert ops.a.codes[0] != ops2.a.codes[0]
    _, _, noise3, _, _ = _call(root_a=b"another root")
    assert noise3.seed_a != noise.seed_a
