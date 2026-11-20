"""A two-stage window's values sized per sampled span (`Stream(span_values=)`): the interior program over a replay sample
of Q takes each at its leaves times Q, typed by its leaf type, so a table read's supplied root, one `Digest` a sampled
span, is a parameter of `Digest`s, whose rows R7 lays out in Digest order (`registered.digest_leaves`). An interior
record registering it so passes both verifiers; one registering it in the flat form, a `Digest` port's 32 u16 words a
row, is refused. Per-span values are a two-stage window's alone."""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

from verity.circuits import population as PN
from verity.circuits.types import Array, Value
from verity.protocols.proofs.spot.verifier import service as S
from verity.protocols.proofs.spot.verifier.one_stage import partition as P

from test_service import PAIR as ONE_TEMPLATE, _spec as one_template_spec
from test_two_stage import (DELTA, MIXED, Q, WINDOW, Clock, _window_spec)

import fake_challenger as FC

#: a table read's supplied root: one `Digest` a sampled span
HIST = ("hist", 1, (8, 64, 1))
DIGEST = Array(8, Array(64, Value(1)))
#: each sampled span's table root, as the reads' supplied value takes them
ROOTS = [hashlib.sha512(f"span {s}".encode()).digest() for s in range(Q)]
SUPPLIED = {"hist": [[{"bytes": r.hex()}] for r in ROOTS]}
#: the challenger's copy of the registered value's row width: a `Digest`'s 512 one-bit leaves a row
REGISTERED = {"hist": 512}


def _spec() -> S.Spec:
    w = _window_spec()
    return dataclasses.replace(
        w, values=(*w.values, S.Value("hist", None, "interior", registered=True)),
        public=(*w.public, *(S.Public(f"{x}:hist", "structure", "the supplied root's layout")
                             for x in ("leaves", "words", "domain"))))


def _stream(span_values=(HIST,), spec=None) -> S.Stream:
    return FC.Stream(spec or _spec(), WINDOW, replay=MIXED, budget=2, delta=DELTA, challenger="challenger-test",
                     clock=Clock(), span_values=span_values)


def test_the_interior_program_takes_a_per_span_value_at_q_typed_by_its_leaf_type():
    stream = _stream()
    assert stream.span_values == (HIST,) and stream.values == ()
    for q in (1, Q, 3):
        assert stream.interior_program(q).fn.signature()[0][0] == ("hist", Array(q, DIGEST))
    plain = FC.Stream(_spec(), WINDOW, replay=MIXED, budget=2, delta=DELTA, challenger="challenger-test", clock=Clock())
    assert stream.digest != plain.digest and stream.digest != _stream(span_values=(("hist", 2, (8, 64, 1)),)).digest
    assert P.program_digest(stream.interior_program(Q)) != P.program_digest(plain.interior_program(Q))


def test_the_interior_program_takes_the_streams_values_then_its_per_span_values():
    """The interior program's parameters are the stream's `values`, then its `span_values` at Q: the program, so the
    digest, both verifiers check an interior record against and an integration registers its values in. The other order
    is another program."""
    x = ("x", 2, 16)
    stream = FC.Stream(_spec(), WINDOW, replay=MIXED, budget=2, delta=DELTA, challenger="challenger-test", clock=Clock(),
                       values=(x,), span_values=(HIST,))
    for q in (1, Q):
        assert stream.interior_program(q).fn.signature()[0][:2] == (("x", Array(2, Value(16))), ("hist", Array(q, DIGEST)))
    cells = [(w, n * Q) for w, n in zip(WINDOW, stream.cells)]
    span = ("hist", Q, (8, 64, 1))
    got = P.program_digest(stream.interior_program(Q))
    assert got == P.program_digest(PN.mixed_population_program(cells, values=(x, span)))
    assert got != P.program_digest(PN.mixed_population_program(cells, values=(span, x)))


def test_per_span_values_are_a_two_stage_windows():
    with pytest.raises(ValueError, match="per-span values are a two-stage window's"):
        FC.Stream(one_template_spec(), ONE_TEMPLATE, budget=2, delta=DELTA, challenger="c", span_values=(HIST,))
    with pytest.raises(ValueError, match="per-span values are a two-stage window's"):
        FC.Stream(_spec(), WINDOW, budget=2, delta=DELTA, challenger="c", span_values=(HIST,))
    for bad in ((("hist", 0, (8, 64, 1)),), (("hist", 1, (512,)),), (("0.x", 1, 16),)):
        with pytest.raises(ValueError, match="each \\(name, leaves, width\\)"):
            _stream(span_values=bad)


def test_a_window_without_per_span_values_keeps_its_interior_program():
    stream = FC.Stream(_spec(), WINDOW, replay=MIXED, budget=2, delta=DELTA, challenger="challenger-test", clock=Clock())
    assert stream.span_values == ()
    assert stream.interior_program(Q).root.id == PN.mixed_population_program([(w, n * Q) for w, n in
                                                                            zip(WINDOW, stream.cells)]).root.id
