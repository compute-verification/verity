"""`trace`'s large forms: past `trace._SMALL` bits a form keeps its bits as a `_Set` its circuit interns, which changes no set
and no gate. The XOR of two sets so kept is the frozensets' XOR, kept so, with equal sets one object; and a builder traced with
every set of more than one bit a `_Set` gives the gates it gives with frozensets only."""
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


def test_xors_of_sets_are_the_frozensets_xors_with_equal_sets_one_object(monkeypatch) -> None:
    monkeypatch.setattr(TR, "_SMALL", 3)
    R = random.Random(20261026)
    tab: dict = {}
    want = [frozenset(R.sample(range(300), R.randrange(0, 10))) for _ in range(24)]
    got = [TR._canon(tab, s) for s in want]
    for _ in range(4000):
        i, j = R.randrange(len(want)), R.randrange(len(want))
        w, g = want[i] ^ want[j], TR._sx(tab, got[i], got[j])
        assert frozenset(g) == w and len(g) == len(w) and all(k in g for k in w)
        if len(w) > TR._SMALL:
            assert type(g) is TR._Set and list(g) == sorted(w) and g is TR._canon(tab, w)
        else:
            assert type(g) is frozenset
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


@pytest.mark.parametrize("case", [_absmax, _ex2, _lookup], ids=["RowAbsMax{40,RELU}", "MufuEx2Ftz", "lookup"])
def test_a_trace_is_the_same_with_every_set_shared(case, monkeypatch) -> None:
    build, params, ret = case()
    monkeypatch.setattr(TR, "_SMALL", 1 << 40)
    plain = TR.trace(build, params, ret)
    monkeypatch.setattr(TR, "_SMALL", 1)
    shared = TR.trace(build, params, ret)
    assert (shared.n_inputs, shared.gates, shared.outputs) == (plain.n_inputs, plain.gates, plain.outputs)
