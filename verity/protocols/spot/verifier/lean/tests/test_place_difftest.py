"""A class's place in its window's session, its family and its run (`Spot.Place`: `place`, `family`, `run`) held to
their Python: the ``place`` piece of the difftest's executable answers a window's place, its family and its run, or
refuses a window that names no place, and the reference below computes the same from the window's fields, as the
two-stage driver computes a place and a family from a class's layout (its ``place`` and ``family_of``, copied below
rather than imported: nothing outside ``benchmarks/`` imports its code). Named cases are the three families of a class
of prompt 4 and outputs 4 (its spans', a stop class's spans' and a stop class's stop blocks'), the two refusals (a size
of 2^32, and the stop blocks of a class that is no stop class) and the fields' other readings; generated ones are
windows whose sizes sit on both sides of 2^32 and whose other fields take every kind of value, and must reach each
family's kind and both refusals.
"""

from __future__ import annotations

import shutil

import pytest
from hypothesis import strategies as st

from test_wires_difftest import PACKAGE, PIECES

import verity_difftest as D

lean = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

#: `Spot.Place.SIZES`: a class's two sizes are each below this.
SIZES = 1 << 32
#: `Spot.Place.FAMILY`.
FAMILY = "draw"


def _size(x) -> int | None:
    """`Spot.Place.size?`: a natural number below `SIZES`."""
    return x if type(x) is int and 0 <= x < SIZES else None


def reference(w, seen: dict | None = None):
    """`Spot.Place.piece`: the window's place, its family and its run; `seen` counts each kind of answer."""
    get = w.get if isinstance(w, dict) else lambda _k: None
    prompt, outputs = _size(get("prompt")), _size(get("outputs"))
    stopped = get("stopped") is True
    blocks = isinstance(get("population"), str) and get("population") == "stop"
    kind = ("size" if prompt is None or outputs is None else "no stop class" if blocks and not stopped
            else "blocks" if blocks else "stopped" if stopped else "spans")
    if seen is not None:
        seen[kind] = seen.get(kind, 0) + 1
    if kind in ("size", "no stop class"):
        raise D.Reject("the window names no place in its session")
    n = ((prompt * SIZES + outputs) * 2 + int(stopped)) * 2 + int(blocks)
    run = get("run")
    return {"place": n, "family": f"{FAMILY}/{n}", "run": run if isinstance(run, str) else None}


# --- named cases -----------------------------------------------------------------------------------------------------

def _window(prompt=4, outputs=4, **more) -> dict:
    return {"prompt": prompt, "outputs": outputs, **more}


def _placed(n: int, run: str | None = None) -> dict:
    return {"place": n, "family": f"draw/{n}", "run": run}


TOP = SIZES - 1

CASES = [
    ("a class of prompt 4 and outputs 4", _window(), _placed(68719476752)),
    ("a stop class", _window(stopped=True), _placed(68719476754)),
    ("a stop class's stop blocks", _window(stopped=True, population="stop"), _placed(68719476755)),
    ("stopped false is no stop class", _window(stopped=False), _placed(68719476752)),
    ("stopped that is no boolean is no stop class", _window(stopped="true"), _placed(68719476752)),
    ("a stop class's other population", _window(stopped=True, population="served"), _placed(68719476754)),
    ("a population that is no string", _window(stopped=True, population=["stop"]), _placed(68719476754)),
    ("its run", _window(run="ab"), _placed(68719476752, "ab")),
    ("a run that is no string", _window(run=7), _placed(68719476752)),
    ("the sizes the other way round", _window(2, 4), _placed(34359738384)),
    ("the largest sizes", _window(TOP, TOP), _placed((TOP * SIZES + TOP) * 4)),
    ("no sizes at all", _window(0, 0), _placed(0)),
]

REFUSED = [
    ("a prompt of 2^32", _window(prompt=SIZES)),
    ("outputs of 2^32", _window(outputs=SIZES)),
    ("the stop blocks of a class that is no stop class", _window(population="stop")),
    ("the stop blocks of a class whose stopped is false", _window(stopped=False, population="stop")),
    ("the stop blocks of a class whose stopped is no boolean", _window(stopped="true", population="stop")),
    ("a size below zero", _window(prompt=-4)),
    ("a size that is a string", _window(outputs="4")),
    ("a size that is a boolean", _window(prompt=True)),
    ("no outputs", {"prompt": 4}),
    ("no window", None),
    ("a list", [4, 4]),
]


@pytest.mark.parametrize("name, w, want", CASES, ids=lambda x: x if isinstance(x, str) else "")
def test_the_reference_on_the_named_cases(name, w, want):
    assert reference(w) == want


@pytest.mark.parametrize("name, w", REFUSED, ids=lambda x: x if isinstance(x, str) else "")
def test_the_reference_refuses(name, w):
    with pytest.raises(D.Reject):
        reference(w)


def test_two_classes_have_two_places_and_a_stop_class_two():
    """Every window of a few sizes, the bounds among them, that names a place names one no other names."""
    sizes = [0, 1, 4, TOP]
    windows = [(p, o, s, b) for p in sizes for o in sizes for s in (False, True) for b in (False, True) if s or not b]
    places = {reference(_window(p, o, stopped=s, **({"population": "stop"} if b else {})))["place"]
              for p, o, s, b in windows}
    assert len(places) == len(windows) == 48


# --- the Lean -------------------------------------------------------------------------------------------------------

JSON = st.recursive(st.none() | st.booleans() | st.integers(-3, 3) | st.text("ab/1", max_size=3),
                    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text("abk", max_size=2), inner,
                                                                               max_size=3),
                    max_leaves=6)
SIZE = st.sampled_from([0, 1, 2, 4, TOP - 1, TOP, TOP, SIZES, SIZES + 1, -1]) | st.integers(0, 6)


@st.composite
def windows(pick):
    """A window: its two sizes mostly natural numbers on both sides of the bound, each other field absent, of its own
    kind, or arbitrary JSON."""
    w: dict = {}
    for k in ("prompt", "outputs"):
        if pick(st.integers(0, 15)):
            w[k] = pick(SIZE) if pick(st.integers(0, 11)) else pick(JSON)
    for k, values in (("stopped", [True, True, False, "true", 1, None]),
                      ("population", ["stop", "stop", "served", "Stop", "", None, True, ["stop"]]),
                      ("run", ["ab", "", 7, None])):
        if pick(st.booleans()):
            w[k] = pick(st.sampled_from(values)) if pick(st.integers(0, 7)) else pick(JSON)
    if pick(st.integers(0, 7)) == 0:
        w[pick(st.text("abk", max_size=2))] = pick(JSON)
    return w


INPUTS = st.one_of(windows(), windows(), windows(), JSON)
PLACE = D.Port(PIECES, "place", reference, INPUTS, package=PACKAGE, of_record="lean")

pieces = D.server_fixture(PLACE)


@lean
@pytest.mark.parametrize("name, w, want", CASES, ids=lambda x: x if isinstance(x, str) else "")
def test_named_cases(pieces, name, w, want):
    assert pieces.call("place", w) == ("output", D.canonical(want))


@lean
@pytest.mark.parametrize("name, w", REFUSED, ids=lambda x: x if isinstance(x, str) else "")
def test_refused_by_both(pieces, name, w):
    kind, _why = pieces.call("place", w)
    assert kind == "reject"


@lean
def test_the_two_agree_and_reach_each_family_and_each_refusal(pieces):
    seen: dict = {}
    port = D.Port(PIECES, "place", lambda w: reference(w, seen), INPUTS, package=PACKAGE, of_record="lean")
    stats = D.run(port, examples=1500, server=pieces)
    assert stats.outputs > 300 and stats.rejects > 300, stats
    few = {k: seen.get(k, 0) for k in ("spans", "stopped", "blocks", "size", "no stop class") if seen.get(k, 0) < 40}
    assert not few, f"kinds of window the generated ones reach fewer than 40 times: {few}"
