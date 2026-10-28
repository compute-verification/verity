"""Adversarial cases against CacheStep_v1 (`cache_step`), which fixes the prefix hit `ReplayStep_v1`
takes as bounded witness. Its own suite shows the hit is pinned to the cached prefix; these
cases pin down what the piece does NOT constrain.

Findings, each confirmed executably below:

1. The hit's LENGTH is pinned, but only to the committed table: the derived `wh` is the longest cached prefix of the
   table handed in, so a prover who controls the committed table controls the hit exactly. (Shown: two tables give two
   pinned hits for the same request and units.)
2. The table's and the row's token ids (`units`) are a FREE WITNESS in this piece: nothing ties them to the certified
   prompt/output. A prover fabricates a trie of invented tokens and a matching `units` window to manufacture an
   arbitrarily long "hit" that charges nothing, with canonical sources naming invented owners. The hit channel is
   closed only once the compile unit binds `units` (and the chained table) to the certified I/O and wires `wh` into
   `ReplayStep_v1`'s `hit`.
3. Past the pool bound (`sent` = 1, where eviction may have shortened the real hit) the hit is advice v1's charged
   field and is checked only to be a cached prefix: a claim LONGER than the table is refused (can't exceed the cache),
   a SHORTER one is accepted (it over-charges). So eviction is a charged residual here, not a free channel.
"""
from __future__ import annotations

import random
from dataclasses import replace
from types import SimpleNamespace

from verity.core.protocols.certifier import cache_step as CS

SMALL = CS.statics(8, 4, 8, 4, 32)
S = SimpleNamespace(**SMALL)
KU = CS.widths(S).KU            # 8 units a window
ROOT = S.N                      # 8


def _chain(tokens, owner):
    """A straight trie of `tokens` (a list of U-token units) from ROOT, each unit under the one before: entry e is
    (e - 1 or ROOT, tokens[e], owner)."""
    return tuple(CS.Entry(ROOT if e == 0 else e - 1, tuple(t), owner) for e, t in enumerate(tokens))


def _longest(table, row):
    for wh in range(KU + 1):
        if CS.reference(S, table, replace(row, wh=wh, sent=False, nins=0)).ok:
            return wh
    return None


def test_a_fabricated_table_and_units_manufacture_a_free_hit():
    """Finding 2. The tokens are the prover's invention. A fabricated chain of 7 units plus a matching `units` window
    gives a derived hit of 7 units (28 tokens) that charges nothing, with sources naming an invented owner, and the
    piece accepts it (nothing ties the tokens to any certified prompt)."""
    rng = random.Random(0)
    invented = [tuple(rng.randrange(1 << S.TW) for _ in range(S.U)) for _ in range(7)]
    table = _chain(invented, owner=3)            # request 3 "computed" these invented tokens
    tail = (tuple(rng.randrange(1 << S.TW) for _ in range(S.U)),)   # an 8th, non-hit unit
    units = tuple(invented) + tail

    row = CS.Row(r=5, units=units, admit=True, K=S.L)       # a different request, K = L so every unit is lookable
    wh = _longest(table, row)
    assert wh == 7, wh                                       # the longest cached prefix of the fabricated trie
    out = CS.reference(S, table, replace(row, wh=wh))
    assert out.ok
    assert CS.evaluate(SMALL, [(table, replace(row, wh=wh))]) == [out]
    # the hit's canonical sources name the invented owner for all 7 units, from invented tokens.
    assert CS.sources(S, table, out.trail, wh) == ((3, 7 * S.U),)
    # 7 of the window's 8 units are "free" hits here: the charge the compile unit must impose is the binding of
    # `units` to the certified tokens, absent in this piece.
    print(f"\n[cache-step] fabricated hit: {wh} of {KU} units ({wh * S.U} tokens) hit on invented tokens, "
          f"sources {CS.sources(S, table, out.trail, wh)}")


def test_the_hit_length_is_pinned_to_the_committed_table_only():
    """Finding 1. For the same request and the same `units`, two committed tables give two different pinned derived
    hits: the prover selects the hit by selecting the table it commits."""
    rng = random.Random(1)
    units_toks = [tuple(rng.randrange(1 << S.TW) for _ in range(S.U)) for _ in range(KU)]
    units = tuple(units_toks)
    row = CS.Row(r=1, units=units, admit=True, K=S.L)

    long_table = _chain(units_toks[:7], owner=2)
    short_table = _chain(units_toks[:3], owner=2)
    assert _longest(long_table, row) == 7
    assert _longest(short_table, row) == 3
    # and in each table the derived hit is exact (one unit either side refused), so the table fully determines it.
    for table, cached in ((long_table, 7), (short_table, 3)):
        for x in range(KU + 1):
            assert CS.reference(S, table, replace(row, wh=x)).ok == (x == cached)


def test_past_the_bound_the_hit_is_charged_not_free():
    """Finding 3. With `sent` = 1 the hit is advice v1's charged field: a claim longer than the committed table is
    refused, a shorter one accepted (over-charging). Eviction can only shorten the real hit, so this is a charged
    residual, not a free lengthening channel."""
    rng = random.Random(2)
    units_toks = [tuple(rng.randrange(1 << S.TW) for _ in range(S.U)) for _ in range(KU)]
    units = tuple(units_toks)
    table = _chain(units_toks[:5], owner=2)
    row = CS.Row(r=1, units=units, admit=True, K=S.L, sent=True)
    cached = _longest(table, replace(row, sent=False))
    assert cached == 5
    for x in range(KU + 1):
        assert CS.reference(S, table, replace(row, wh=x)).ok == (x <= cached), x
    # the circuit agrees on the boundary (cached, cached+1).
    cases = [(table, replace(row, wh=cached)), (table, replace(row, wh=cached + 1))]
    assert [o.ok for o in CS.evaluate(SMALL, cases)] == [True, False]
