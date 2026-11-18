import itertools

import pytest

from verity.protocols.resource_accounting.memory.poub.prover.rota import REGISTRATION, REKEY_MAX, Rota

SHAPES = [(n, life, w) for n, life, w in itertools.product((1, 3, 7, 64), (1, 2, 3, 5), (1, 4, 13))]


def chunks(rota: Rota, appends: int):
    """Every chunk of the rota: each registered block, and `appends` fresh appends born in windows 0, 1, …"""
    for j in range(rota.blocks):
        yield 0, rota.registered_class(j), True
    for k in range(appends):
        yield k % rota.windows, rota.append_class(k), False


@pytest.mark.parametrize("n, life, w", SHAPES)
def test_every_chunk_is_rekeyed_exactly_once_a_period_after_its_base(n, life, w):
    rota = Rota(n, life, w)
    for base, c, _ in chunks(rota, 9):
        got = set(rota.rekeys(base, c))
        for start in range(base + 1, w - life + 1, life):
            assert len(got & set(range(start, start + life))) == 1, (base, c, start)
        assert all(base < r < w for r in got)
        assert rota.first(base, c) <= base + life


@pytest.mark.parametrize("n, life, w", SHAPES)
def test_no_chunk_is_planned_outside_its_keys_window_or_before_its_probation_ends(n, life, w):
    rota = Rota(n, life, w)
    for base, c, registered in chunks(rota, 9):
        rekeys = set(rota.rekeys(base, c))
        for t in range(w):
            joined = rota.key(base, c, t, registered=registered)
            if not registered and t < base + life:
                assert joined is None
                continue
            assert joined is not None, (base, c, t)
            assert joined <= t < joined + life
            assert joined in rekeys or (registered and joined == REGISTRATION)


@pytest.mark.parametrize("n, life, w", SHAPES)
def test_a_registered_block_is_plannable_in_every_window_and_a_fresh_one_from_the_end_of_its_probation(n, life, w):
    rota = Rota(n, life, w)
    for t in range(w):
        assert all(rota.key(0, rota.registered_class(j), t, registered=True) is not None for j in range(n))
        for k in range(9):
            base = k % w
            assert (rota.key(base, rota.append_class(k), t, registered=False) is not None) == (t >= base + life)


@pytest.mark.parametrize("n, life, w", SHAPES)
def test_the_due_lists_name_each_rekey_once(n, life, w):
    """Fresh appends are 2k + 1 among the session's appends (re-keys between them), so the class is the fresh ordinal
    k mod L, not the append's index."""
    rota = Rota(n, life, w)
    fresh = [(2 * k + 1, k % w, 1 + 300 * k) for k in range(9)]
    due_reg = {r: rota.registered_due(r) for r in range(w + 2)}
    for j in range(n):
        want = set(rota.rekeys(0, rota.registered_class(j)))
        assert want == {r for r, runs in due_reg.items() for a, count in runs if a <= j < a + count}
        assert all(sum(a <= j < a + count for a, count in runs) <= 1 for runs in due_reg.values())
    for f, (k, b, count) in enumerate(fresh):
        want = set(rota.rekeys(b, f % life))
        for r in range(w + 2):
            pieces = [(o, m) for kk, o, m in rota.appends_due(r, fresh) if kk == k]
            assert bool(pieces) == (r in want)
            if pieces:
                assert [j for o, m in pieces for j in range(o, o + m)] == list(range(count))
                assert all(m <= REKEY_MAX for _, m in pieces)
    for r in range(w + 2):
        assert rota.due(r, fresh) == (*((("registered", a), m) for a, m in due_reg[r]),
                                      *((("append", k, o), m) for k, o, m in rota.appends_due(r, fresh)))


def test_a_registered_class_is_rekeyed_in_appends_of_at_most_rekey_max_blocks():
    rota = Rota(3 * REKEY_MAX + 7, 2, 4)
    runs = rota.registered_due(2)
    assert runs == ((0, REKEY_MAX), (REKEY_MAX, 260))
    assert rota.registered_due(1) == ((772, REKEY_MAX), (772 + REKEY_MAX, 259))


@pytest.mark.parametrize("n, life", [(7, 3), (64, 5), (3, 5)])
def test_the_registered_classes_are_runs_covering_the_store_once(n, life):
    rota = Rota(n, life, 4)
    runs = [rota.class_runs(c) for c in range(life)]
    assert sum(count for _, count in runs) == n
    assert [j for f, count in runs for j in range(f, f + count)] == list(range(n))
    assert all(rota.registered_class(j) == c for c, (f, count) in enumerate(runs) for j in range(f, f + count))


def test_the_rota_reads_public_parameters_alone():
    assert Rota(64, 3, 10) == Rota(64, 3, 10)
    assert [Rota(64, 3, 10).registered_due(r) for r in range(10)] == [
        (), ((22, 21),), ((43, 21),), ((0, 22),), ((22, 21),), ((43, 21),), ((0, 22),), ((22, 21),), ((43, 21),),
        ((0, 22),)]
    with pytest.raises(ValueError):
        Rota(64, 0, 10)
