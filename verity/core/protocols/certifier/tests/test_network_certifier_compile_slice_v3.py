"""The compile slice in output form (`compile_slice_v3.CompileSlice_v3`): CompileSlice_v2 computing its schedule, the
state it leaves and every pool derivation rather than checking them. Its only free input is the advice, ℓ_w = DA + PW
bits; the circuit's verdict, served rows and last rows are `compile_slice_v3.reference`'s (CompileSlice_v2's scan) on
both demonstration slices, on every advice of their arrivals (each decision string and step count) and on random windows
of the demo's six-block class under eviction pressure, and the served rows are the ones CompileSlice_v2 checks."""
from __future__ import annotations

import itertools
import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import compile_slice as CS, compile_slice_v2 as V2, compile_slice_v3 as V3
from verity.core.protocols.certifier import replay_chain as RC
from verity.ml.circuits import boolean as BL

FIXTURES = Path(__file__).parent / "fixtures"


def book() -> dict:
    streams = json.loads((FIXTURES / "arith_encode_vectors.json").read_text())["streams"]
    return next(s for s in streams if s["source"].startswith("async-light"))["decisions"]["book"]


def _x(f):
    st, sl = f()
    return SimpleNamespace(st=st, S=SimpleNamespace(**st), sl=sl)


@pytest.fixture(scope="module")
def synth():
    return _x(V3.demo)


@pytest.fixture(scope="module")
def asynth():
    return _x(V3.demo_async)


def _agrees(st: dict, slices: list[CS.Slice]) -> list:
    """The circuit's outputs on `slices` against the reference's: the verdict, and where it accepts, srv and cout."""
    S = SimpleNamespace(**st)
    refs = [V3.reference(S, sl) for sl in slices]
    for sl, ref, got in zip(slices, refs, V3.evaluate(st, slices)):
        assert (got.verdict == V3.ACCEPT) == (ref is not None), sl
        if ref:
            assert (got.srv, got.cout) == (ref.srv, ref.cout), sl
    return refs


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_the_free_input_is_the_advice(request, which):
    """Every input row is classified, and the unit's input gates that are neither ingress nor wired (another member's
    committed output) are the advice's: DA decision bits and the PW = 3 bits of the step count, 8 at the demo's class,
    PW alone at DA = 0 and D + PW at DA = D. The traced circuit has no input past its signature's."""
    x = request.getfixturevalue(which)
    Z = RC.widths(CS.chain(x.S))
    params, _ret = V3.unit(**x.st).signature()
    widths = {name: sum(BL.elements(t)) for name, t in params}
    assert set(widths) == set(V3.CLASSES) and Z.PW == 3
    assert V3.free_width(x.S, widths) == widths["adv"] == x.S.DA + Z.PW == 8
    assert widths["ing"] == x.S.K * (1 + x.S.IW + Z.LW)
    assert V3._counted(tuple((k, x.st[k]) for k in V3.STATICS))["inputs"] == sum(widths.values())
    for da in (0, x.S.D):
        assert V3.free_width(SimpleNamespace(**{**x.st, "DA": da})) == da + Z.PW
    with pytest.raises(ValueError, match="at most D"):
        V3.shape(SimpleNamespace(**{**x.st, "DA": x.S.D + 1}))


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_the_demo_is_its_reference(request, which):
    """The demo compiles (verdict 0x7) to the reference's served rows and last rows, and its served rows are the
    ``srv`` row CompileSlice_v2's honest case takes as input and checks, less that row's padding to whole words."""
    x = request.getfixturevalue(which)
    (ref,) = _agrees(x.st, [x.sl])
    assert ref.verdict == V3.ACCEPT == 0x7
    S2 = SimpleNamespace(**V3._v2(x.st))
    srv = list(V2.case(S2, x.sl, book()).rows["srv"])
    assert ref.srv == srv[:len(ref.srv)] and not any(srv[len(ref.srv):])
    assert len(ref.cout) == V3.out_bits(x.S)["cout"]


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_every_advice_is_its_reference(request, which):
    """On the demo's arrivals, each of the 32 decision strings at each step count 0 to 7 is advice the circuit compiles
    exactly where the reference does, to its outputs: a step count past P is refused by the load alone (0x6)."""
    x = request.getfixturevalue(which)
    slices = [replace(x.sl, decisions=d, steps=n) for d in itertools.product((0, 1), repeat=x.S.DA) for n in range(8)]
    refs = _agrees(x.st, slices)
    assert sum(r is not None for r in refs) == 32 * (x.S.P + 1)
    over = [sl for sl in slices if sl.steps > x.S.P][:4]
    assert [g.verdict for g in V3.evaluate(x.st, over)] == [0x6] * len(over)


def _window(rng: random.Random, AS: int) -> CS.Slice:
    """Three arrivals of 4 to 16 tokens, most sharing a first block, and five random decisions."""
    base = [rng.randrange(1, 15) for _ in range(4)]
    arrivals, tokens = [], []
    for i in range(3):
        P, O = rng.randrange(4, 13), rng.randrange(1, 5)
        pre = base if rng.random() < 0.6 else [rng.randrange(1, 15) for _ in range(4)]
        tokens.append(tuple((pre + [rng.randrange(1, 15) for _ in range(P + O)])[:P + O]))
        arrivals.append((i, P, O, rng.randrange(2)) if AS else (i, P, O))
    return CS.Slice(tuple(arrivals), tuple(rng.randrange(2) for _ in range(5)), 5, tuple(tokens))


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_random_windows_under_eviction(request, which):
    """CompileSlice_v2's 40 random windows of the demo's class (26 synchronous and 13 asynchronous evict): the circuit
    compiles every one to the reference's outputs."""
    x = request.getfixturevalue(which)
    rng = random.Random(18 + x.S.AS)
    slices = [_window(rng, x.S.AS) for _ in range(40)]
    refs = _agrees(x.st, slices)
    assert all(refs)
    S2 = SimpleNamespace(**V3._v2(x.st))
    assert sum(any(any(w[3]) for w in V2.scan(S2, sl).wit) for sl in slices) == (13 if x.S.AS else 26)


def test_refusals(synth):
    """Advice or arrivals the class can't compile are refused, and the reference refuses them too: a prompt past the
    class's CU·B tokens (by the steps and the pool, 0x1), and advice of the demo's first two decisions (DA = 2), which
    its steps read past (by the steps alone, 0x5). Its first three compile."""
    x = synth
    big = replace(x.sl, arrivals=((0, 30, 4),) + x.sl.arrivals[1:])
    assert V3.reference(x.S, big) is None and V3.evaluate(x.st, [big])[0].verdict == 0x1
    for da, verdict in ((2, 0x5), (3, V3.ACCEPT)):
        st = {**x.st, "DA": da}
        short = replace(x.sl, decisions=x.sl.decisions[:da])
        assert V3.size(**st)["free_bits"] == da + 3
        assert _agrees(st, [short]) and V3.evaluate(st, [short])[0].verdict == verdict


def test_size(synth, asynth):
    """What the output form costs: CompileSlice_v2's 83,092 and 91,785 ANDs at the demos' classes (`test_network_
    certifier_compile_slice_v2.test_size`), less its witness checks, plus the waiting loop's computed lookups and the
    threshold searches."""
    assert V3.size(**synth.st) == {"ands": 76011, "adv_bits": 8, "ing_bits": 27, "wired_bits": 207, "free_bits": 8}
    assert V3.size(**asynth.st) == {"ands": 82715, "adv_bits": 8, "ing_bits": 27, "wired_bits": 210, "free_bits": 8}
