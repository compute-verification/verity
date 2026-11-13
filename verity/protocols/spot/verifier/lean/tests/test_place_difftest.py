"""A sample's place in its window's session, its family and its run (`Spot.Place`: `place`, `family`, `run`) held to
their Python: the ``place`` piece of the difftest's executable answers a window's place, its family and its run, or
refuses a window that names no place, and the reference below computes the same from the window's four items as the
two-stage driver's one definition of a place does (its ``of``, copied below rather than imported: nothing outside
``benchmarks/`` imports its code). `PLACES` are that definition's rows, word for word: 11 windows with a place, the
three families of a class of prompt 4 and outputs 4 among them (its spans', a stop class's spans' and a stop class's
stop blocks'), and 17 with none (a size that is no natural number below 2^32, a ``stopped`` that is neither true nor
false, a ``population`` that is not "stop", and the stop blocks of a class that is no stop class). One row, a size
written 4.0, is outside the harness's canonical JSON, which holds integers only: the reference refuses it here, and
the Lean refuses it as its reader reads that text, by a `#guard` of `Spot.Place`. Generated windows put their sizes
on both sides of 2^32 and each other item absent, of its own kind or arbitrary JSON, and must reach each family's
kind and each refusal.
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


def of(window, seen: dict | None = None) -> int:
    """The place of the sample a record registers, from the record's window (the driver's ``of``, with `seen` counting
    each kind of answer): ValueError for a size that is no natural number below 2^32, for a ``stopped`` the window
    names that is neither true nor false, for a ``population`` it names that is not "stop", and for the stop blocks of
    a class that is no stop class."""
    def kind(k: str) -> None:
        if seen is not None:
            seen[k] = seen.get(k, 0) + 1

    prompt, outputs = window.get("prompt"), window.get("outputs")
    if not all(type(v) is int and 0 <= v < SIZES for v in (prompt, outputs)):
        kind("size")
        raise ValueError("a place names two sizes, natural numbers below 2^32")
    stopped, blocks = window.get("stopped", False), "population" in window
    if type(stopped) is not bool or (blocks and window["population"] != "stop"):
        kind("word")
        raise ValueError("where its window names them, stopped is true or false and population is \"stop\"")
    if blocks and not stopped:
        kind("no stop class")
        raise ValueError("the class is no stop class: it has no stop blocks")
    kind("blocks" if blocks else "stopped" if stopped else "spans")
    return ((prompt * SIZES + outputs) * 2 + int(stopped)) * 2 + int(blocks)


def reference(w, seen: dict | None = None):
    """`Spot.Place.piece`: the window's place, its family and its run, refused where it names no place; a value that is
    no object names none."""
    if not isinstance(w, dict):
        raise D.Reject("the window names no place in its session")
    try:
        n = of(w, seen)
    except ValueError as e:
        raise D.Reject(f"the window names no place in its session: {e}") from None
    run = w.get("run")
    return {"place": n, "family": f"{FAMILY}/{n}", "run": run if isinstance(run, str) else None}


# --- the definition's rows ------------------------------------------------------------------------------------------

#: the driver's rows of its one definition, a record's window and its place, None for a window that has none
PLACES = [
    ({"prompt": 4, "outputs": 4}, 68719476752),
    ({"prompt": 4, "outputs": 4, "stopped": False}, 68719476752),
    ({"prompt": 4, "outputs": 4, "stopped": True}, 68719476754),
    ({"prompt": 4, "outputs": 4, "stopped": True, "population": "stop"}, 68719476755),
    ({"prompt": 4, "outputs": 2}, 68719476744),
    ({"prompt": 2, "outputs": 8}, 34359738400),
    ({"prompt": 8, "outputs": 2}, 137438953480),
    ({"prompt": 0, "outputs": 0}, 0),
    ({"prompt": 0, "outputs": 1}, 4),
    ({"prompt": 1, "outputs": 0}, 17179869184),
    ({"prompt": 2**32 - 1, "outputs": 2**32 - 1, "stopped": True, "population": "stop"}, 2**66 - 1),
    ({"prompt": 2**32, "outputs": 4}, None),
    ({"prompt": 4, "outputs": 2**32}, None),
    ({"prompt": 4, "outputs": -1}, None),
    ({"prompt": "4", "outputs": 4}, None),
    ({"prompt": 4.0, "outputs": 4}, None),
    ({"prompt": True, "outputs": 4}, None),
    ({"prompt": None, "outputs": 4}, None),
    ({"outputs": 4}, None),
    ({"prompt": 4}, None),
    ({"prompt": 4, "outputs": 4, "stopped": 1}, None),
    ({"prompt": 4, "outputs": 4, "stopped": "true"}, None),
    ({"prompt": 4, "outputs": 4, "stopped": None}, None),
    ({"prompt": 4, "outputs": 4, "population": "stop"}, None),
    ({"prompt": 4, "outputs": 4, "stopped": False, "population": "stop"}, None),
    ({"prompt": 4, "outputs": 4, "stopped": True, "population": "spans"}, None),
    ({"prompt": 4, "outputs": 4, "stopped": True, "population": None}, None),
    ({"prompt": 4, "outputs": 4, "stopped": True, "population": 1}, None),
]

#: what else a record's window holds, which moves no place
MORE = {"run": "ab", "row": 3, "check": "c", "verdicts": True}


def _canonical(w) -> bool:
    """Whether the harness carries `w`: canonical JSON holds integers only, so a number that is not whole is outside it."""
    try:
        D.canonical(w)
    except TypeError:
        return False
    return True


def _id(x) -> str:
    return D.canonical(x) if _canonical(x) else repr(x)


#: the rows the harness carries, each as given and beside a window's other items
CARRIED = [(w, at) for window, at in PLACES if _canonical(window) for w in (window, {**window, **MORE})]


def test_the_rows_are_the_definitions():
    assert len(PLACES) == 28 and sum(at is not None for _w, at in PLACES) == 11
    assert [w for w, _at in PLACES if not _canonical(w)] == [{"prompt": 4.0, "outputs": 4}]
    assert [at for _w, at in PLACES[:4]] == [68719476752, 68719476752, 68719476754, 68719476755]


@pytest.mark.parametrize("window, at", PLACES, ids=_id)
def test_the_reference_on_the_rows(window, at):
    for w in (window, {**window, **MORE}):
        if at is None:
            with pytest.raises(D.Reject):
                reference(w)
        else:
            assert reference(w) == {"place": at, "family": f"draw/{at}", "run": w.get("run")}


def test_two_classes_have_two_places_and_a_stop_class_two():
    """Every window of a few sizes, the bounds among them, that names a place names one no other names."""
    sizes = [0, 1, 4, SIZES - 1]
    windows = [(p, o, s, b) for p in sizes for o in sizes for s in (False, True) for b in (False, True) if s or not b]
    places = {of({"prompt": p, "outputs": o, "stopped": s, **({"population": "stop"} if b else {})})
              for p, o, s, b in windows}
    assert len(places) == len(windows) == 48


RUNS = [({"prompt": 4, "outputs": 4, "run": "ab"}, "ab"), ({"prompt": 4, "outputs": 4, "run": ""}, ""),
        ({"prompt": 4, "outputs": 4, "run": 7}, None), ({"prompt": 4, "outputs": 4, "run": None}, None),
        ({"prompt": 4, "outputs": 4}, None)]
NO_WINDOW = [None, [4, 4], "window", 4, True]


# --- the Lean -------------------------------------------------------------------------------------------------------

JSON = st.recursive(st.none() | st.booleans() | st.integers(-3, 3) | st.text("ab/1", max_size=3),
                    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text("abk", max_size=2), inner,
                                                                               max_size=3),
                    max_leaves=6)
TOP = SIZES - 1
SIZE = st.sampled_from([0, 1, 2, 4, TOP - 1, TOP, TOP, SIZES, SIZES + 1, -1]) | st.integers(0, 6)
#: words a window's `stopped` and `population` may hold that give it no place
WORDS = (("stopped", ["true", 1, 0, None, [True]]), ("population", ["spans", "Stop", "", None, True, 1, ["stop"]]))


@st.composite
def windows(pick):
    """A window: its two sizes mostly natural numbers on both sides of the bound; its `stopped` and `population` those
    of a class's spans, a stop class's spans, a stop class's stop blocks or the stop blocks of a class that is no stop
    class, or one of them another word, or each absent, of its own kind or arbitrary JSON; its run a string, absent
    or anything else."""
    w: dict = {}
    for k in ("prompt", "outputs"):
        if pick(st.integers(0, 15)):
            w[k] = pick(SIZE) if pick(st.integers(0, 11)) else pick(JSON)
    kind = pick(st.sampled_from(["spans", "spans", "stopped", "stopped", "blocks", "blocks", "no stop class", "word",
                                 "word", "any"]))
    if kind == "spans" and pick(st.booleans()):
        w["stopped"] = False
    if kind in ("stopped", "blocks") or (kind == "word" and pick(st.booleans())):
        w["stopped"] = True
    if kind in ("blocks", "no stop class"):
        w["population"] = "stop"
    if kind == "no stop class" and pick(st.booleans()):
        w["stopped"] = False
    if kind == "word":
        k, values = pick(st.sampled_from(WORDS))
        w[k] = pick(st.sampled_from(values))
    if kind == "any":
        for k in ("stopped", "population"):
            if pick(st.booleans()):
                w[k] = pick(JSON)
    if pick(st.booleans()):
        w["run"] = pick(st.sampled_from(["ab", "", 7, None])) if pick(st.integers(0, 7)) else pick(JSON)
    if pick(st.integers(0, 7)) == 0:
        w[pick(st.text("abk", max_size=2))] = pick(JSON)
    return w


INPUTS = st.one_of(windows(), windows(), windows(), JSON)
PLACE = D.Port(PIECES, "place", reference, INPUTS, package=PACKAGE, of_record="lean")
#: the kinds of window the generated ones must reach: the three kinds of family, and each refusal
KINDS = ("spans", "stopped", "blocks", "size", "word", "no stop class")

pieces = D.server_fixture(PLACE)


@lean
@pytest.mark.parametrize("w, at", CARRIED, ids=_id)
def test_the_lean_on_the_rows(pieces, w, at):
    if at is None:
        assert pieces.call("place", w)[0] == "reject"
    else:
        want = {"place": at, "family": f"draw/{at}", "run": w.get("run")}
        assert pieces.call("place", w) == ("output", D.canonical(want))


@lean
@pytest.mark.parametrize("w, run", RUNS, ids=_id)
def test_a_windows_run_is_its_string(pieces, w, run):
    want = {"place": 68719476752, "family": "draw/68719476752", "run": run}
    assert reference(w) == want
    assert pieces.call("place", w) == ("output", D.canonical(want))


@lean
@pytest.mark.parametrize("w", NO_WINDOW, ids=_id)
def test_a_value_that_is_no_object_names_no_place(pieces, w):
    with pytest.raises(D.Reject):
        reference(w)
    assert pieces.call("place", w)[0] == "reject"


@lean
def test_the_two_agree_and_reach_each_family_and_each_refusal(pieces):
    seen: dict = {}
    port = D.Port(PIECES, "place", lambda w: reference(w, seen), INPUTS, package=PACKAGE, of_record="lean")
    stats = D.run(port, examples=1500, server=pieces)
    assert stats.outputs > 300 and stats.rejects > 300, stats
    few = {k: seen.get(k, 0) for k in KINDS if seen.get(k, 0) < 40}
    assert not few, f"kinds of window the generated ones reach fewer than 40 times: {few}"
