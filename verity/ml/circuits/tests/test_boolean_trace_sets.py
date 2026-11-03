"""`trace`'s large forms: past `trace._SMALL` bits in more than one block of ``2^trace._B`` keys a form keeps its bits as a
`_Set` its circuit interns, which changes no set and no gate. The XOR of two sets so kept is the frozensets' XOR, kept so, with
equal sets one object; and a builder traced with every set of more than one bit in two blocks a `_Set` gives the gates it gives
with frozensets only, at every block size."""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import mufu as BU, quant as BQ
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.types import Array


#: keys near each other (blocks of several keys) and far apart (prefixes that differ high up)
KEYS = [*range(300), *range(1000, 1280, 7), 1 << 20, (1 << 20) + 1, (1 << 20) + 64, 1 << 30]


def _blocks(monkeypatch, b: int) -> None:
    monkeypatch.setattr(TR, "_B", b)
    monkeypatch.setattr(TR, "_W", 1 << b)
    monkeypatch.setattr(TR, "_M", (1 << (1 << b)) - 1)


def _same(tab: dict, g, w: frozenset, probe) -> None:
    """``g`` is the set ``w`` as a form keeps it: its keys, size and members, and a `_Set` (its keys in order, ``w``'s one
    node) exactly when ``w`` has more than `_SMALL` keys over two blocks."""
    assert frozenset(g) == w and len(g) == len(w) and all(k in g for k in w)
    assert not any(k in g for k in probe if k not in w)
    if len(w) > TR._SMALL and len({k >> TR._B for k in w}) > 1:
        assert type(g) is TR._Set and list(g) == sorted(w) and g is TR._canon(tab, w)
    else:
        assert type(g) is frozenset


@pytest.mark.parametrize("b", [0, 2, 6])
def test_xors_of_sets_are_the_frozensets_xors_with_equal_sets_one_object(b, monkeypatch) -> None:
    monkeypatch.setattr(TR, "_SMALL", 3)
    _blocks(monkeypatch, b)
    R = random.Random(20261026)
    tab: dict = {}
    want = [frozenset(R.sample(KEYS, R.randrange(0, 40))) for _ in range(40)]
    want += [frozenset(range(64 * k, 64 * k + n)) for k, n in ((0, 5), (2, 40), (3, 64))]
    got = [TR._canon(tab, s) for s in want]
    for g, w in zip(got, want):
        _same(tab, g, w, KEYS)
    for _ in range(4000):
        i, j = R.randrange(len(want)), R.randrange(len(want))
        w, g = want[i] ^ want[j], TR._sx(tab, got[i], got[j])
        _same(tab, g, w, R.sample(KEYS, 30))
        want.append(w)
        got.append(g)


def _absmax():
    S = SimpleNamespace(N=40, RELU=True)
    return BQ._absmax(S), [Array(S.N, BQ.BF16)], BQ.BF16


def _ex2():
    params, ret = bind(BU.MufuEx2Ftz).signature()
    return BU.MufuEx2Ftz.build, [t for _, t in params], ret


def _lookup():
    """`fp.lookup` sums minterms into plain `forms.L`s of frozensets, which `TracingCircuit.canon` keeps as a `_Set`."""
    R = random.Random(7)
    table = [R.getrandbits(4) for _ in range(64)]
    return (lambda C, x: fp.lookup(C, x, table, 4)), [Array(6, BL.BIT)], Array(4, BL.BIT)


@pytest.mark.parametrize("b", [0, 2, 6])
@pytest.mark.parametrize("case", [_absmax, _ex2, _lookup], ids=["RowAbsMax{40,RELU}", "MufuEx2Ftz", "lookup"])
def test_a_trace_is_the_same_with_every_set_shared(case, b, monkeypatch) -> None:
    build, params, ret = case()
    monkeypatch.setattr(TR, "_SMALL", 1 << 40)
    plain = TR.trace(build, params, ret)
    monkeypatch.setattr(TR, "_SMALL", 1)
    _blocks(monkeypatch, b)
    shared = TR.trace(build, params, ret)
    assert (shared.n_inputs, shared.gates, shared.outputs) == (plain.n_inputs, plain.gates, plain.outputs)
