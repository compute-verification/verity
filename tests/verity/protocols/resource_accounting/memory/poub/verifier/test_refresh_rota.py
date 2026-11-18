import itertools

import pytest

from verity.protocols.resource_accounting.memory.poub.verifier.rota import Schedule

SHAPES = [(n, life, w, f) for n, life, w, f in itertools.product((1, 3, 7, 64), (1, 2, 3, 5), (1, 4, 13), (0, 1, 4))]


def test_the_schedule_of_64_blocks_at_period_2_over_6_windows_with_4_fresh_slots_has_53_slots():
    s = Schedule(64, 2, 6, 4)
    assert [s.window_size(w) for w in range(6)] == [4, 5, 9, 9, 13, 13]
    assert s.total == 53
    assert [s.slot_in(1, m) for m in range(5)] == [("registered", 32, 32), *[("fresh",)] * 4]
    assert [s.slot_in(2, m) for m in range(9)] == [
        ("registered", 0, 32), ("rekey", 0), ("rekey", 1), ("rekey", 2), ("rekey", 3), *[("fresh",)] * 4]
    assert s.slot_in(4, 1) == ("rekey", 0) and s.slot_in(4, 5) == ("rekey", 14)
    assert s.slot_at(52) == (5, ("fresh",)) and s.slot_at(53) is None


@pytest.mark.parametrize("n, life, w, f", SHAPES)
def test_every_fresh_slot_is_rekeyed_once_in_each_later_window_of_its_class(n, life, w, f):
    s = Schedule(n, life, w, f)
    rekeys = [kind[1] for k in range(s.total) if (kind := s.slot_at(k)[1])[0] == "rekey"]
    for b in range(w):
        for i in range(f):
            k = s.fresh_slot(b, i)
            assert s.slot_at(k) == (b, ("fresh",))
            at = [s.slot_at(r)[0] for r in range(s.total) if s.slot_at(r)[1] == ("rekey", k)]
            assert at == list(range(b + life, w, life))
    assert len(rekeys) == sum(len(range(b + life, w, life)) for b in range(w)) * f


@pytest.mark.parametrize("n, life, w, f", SHAPES)
def test_a_window_holds_at_most_one_registered_slot_its_class_s_run_and_the_runs_cover_the_store_once(n, life, w, f):
    s = Schedule(n, life, w, f)
    runs = [s.class_run(c) for c in range(life)]
    assert [j for lo, hi in runs for j in range(lo, hi)] == list(range(n))
    for t in range(w):
        regs = [s.slot_in(t, m) for m in range(s.window_size(t)) if s.slot_in(t, m)[0] == "registered"]
        lo, hi = runs[t % life]
        assert regs == ([("registered", lo, hi - lo)] if t >= 1 and lo < hi else [])


@pytest.mark.parametrize("n, life, w, f", SHAPES)
def test_slots_are_laid_out_window_by_window(n, life, w, f):
    s = Schedule(n, life, w, f)
    assert [k for t in range(w) for k in s.slots(t)] == list(range(s.total))
    assert all(s.slot_at(k)[0] == t for t in range(w) for k in s.slots(t))


def test_the_schedule_reads_public_parameters_alone():
    assert Schedule(64, 3, 10, 2) == Schedule(64, 3, 10, 2)
    for bad in ((64, 0, 10, 2), (0, 2, 10, 2), (64, 2, 10, -1), (64, 2, 10, 1.5)):
        with pytest.raises(ValueError):
            Schedule(*bad)
