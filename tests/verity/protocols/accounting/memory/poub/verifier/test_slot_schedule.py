"""The refresh schedule (`rota.Schedule`, the Lean's `Params.slotAt`): its sizes at the reference plan, and every slot's
kind and window fixed by the public parameters, each re-key's origin a fresh slot of a window a whole number of periods
before its own and each registered re-key the whole class run of its window."""

from __future__ import annotations

import pytest

from verity.protocols.accounting.memory.poub.verifier.rota import Schedule


@pytest.mark.parametrize(("fresh", "total", "widest"), [(1, 17, 4), (2, 29, 7), (4, 53, 13)])
def test_the_reference_plans_slots_are_draw256s(fresh: int, total: int, widest: int) -> None:
    """N = 64, W = 6, L = 2 (`refresh/DRAW256.md`'s table)."""
    s = Schedule(64, 2, 6, fresh)
    assert s.total == total and max(len(s.slots(w)) for w in range(6)) == widest


@pytest.mark.parametrize(("n", "life", "windows", "fresh"), [(6, 2, 5, 1), (64, 2, 6, 4), (5, 3, 9, 2), (2, 4, 7, 1),
                                                              (7, 1, 4, 0)])
def test_every_slot_has_one_window_and_kind_and_each_rekey_a_fresh_origin(n, life, windows, fresh) -> None:
    s = Schedule(n, life, windows, fresh)
    seen = []
    for w in range(windows):
        kinds = [s.slot_at(k) for k in s.slots(w)]
        assert all(at[0] == w for at in kinds)
        seen += [at[1] for at in kinds]
        registered = [k for _, k in kinds if k[0] == "registered"]
        lo, hi = s.class_run(w % life)
        assert registered == ([("registered", lo, hi - lo)] if w >= 1 and lo < hi else [])
        assert [k for _, k in kinds if k[0] == "fresh"] == [("fresh",)] * fresh
        for _, k in kinds:
            if k[0] == "rekey":
                b, origin = s.slot_at(k[1])
                assert origin == ("fresh",) and b < w and (w - b) % life == 0
    assert len(seen) == s.total and s.slot_at(s.total) is None and s.slot_at(-1) is None
    assert s.window_of(s.total - 1) == windows - 1
    for b in range(windows):
        for i in range(fresh):
            k = s.fresh_slot(b, i)
            due = [w for w in range(windows) for j in s.slots(w) if s.slot_at(j)[1] == ("rekey", k)]
            assert due == list(range(b + life, windows, life))
