"""A two-stage window's values sized per sampled span (`Stream(span_values=)`): the interior program over a replay sample
of Q takes each at its leaves times Q, typed by its leaf type, so a table read's supplied root, one `Digest` a sampled
span, is a parameter of `Digest`s, whose rows R7 lays out in Digest order (`registered.digest_leaves`). An interior
record registering it so passes both verifiers; one registering it in the flat form, a `Digest` port's 32 u16 words a
row, is refused. Per-span values are a two-stage window's alone."""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

from verity.ml.circuits.types import Array, Value
from verity.core.protocols.spot import service as S
from verity.core.protocols.spot.one_stage import partition as P, registered as G, registration as R

from test_service import PAIR as ONE_TEMPLATE, _spec as one_template_spec
from test_two_stage import (DELTA, MIXED, Q, WINDOW, WINDOW_QUERY, Clock, _refused, _window_boundary,
                            _window_interior, _window_spec)

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


def _digest_words(d: bytes) -> list[int]:
    """A digest as a `Digest` port's 32 u16 words: each big-endian 64-bit word's little-endian 16-bit pieces."""
    return [(int.from_bytes(d[i:i + 8], "big") >> (16 * j)) & 0xFFFF for i in range(0, 64, 8) for j in range(4)]


def _interior(stream: S.Stream, previous: str, first, flat: bool = False) -> tuple[dict, dict]:
    """The interior registration of `first`'s replay sample, a v2 record registering ``hist``: as the verifier fills it
    from `SUPPLIED` (`registered.supplied_value`), or, `flat`, committed as the flat form's rows, 32 u16 words a span."""
    program = stream.interior_program(len(first.units))
    bare, ports = _window_interior(program, previous, first)
    if flat:
        prog = P.program_digest(program)
        rows = [_digest_words(r) for r in ROOTS]
        value = G.Registered.commit(prog, "hist", rows, G.supplied_salts(prog, "hist", len(rows)), 16)
    else:
        value = G.supplied_value(program, "hist", G.sources(SUPPLIED)["hist"], bare)
    n = P.units(program, WINDOW_QUERY)
    return R.record(program, WINDOW_QUERY, {"law": "subset", "k": n}, S.SCHEME, bare["roots"], bare["window"],
                    previous=previous, registered=[value.entry()]), ports


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
    assert got == P.program_digest(P.mixed_population_program(cells, values=(x, span)))
    assert got != P.program_digest(P.mixed_population_program(cells, values=(span, x)))


def test_a_per_span_value_of_digests_is_registered_in_digest_order_and_its_flat_form_refused():
    """The interior registers the supplied root R7 fills in Digest order, both verifiers agreeing; the same roots
    committed in the flat form, a `Digest` port's 32 u16 words a row, are refused (R7: another row width, so another
    domain, and not the root R7 fills)."""
    stream = _stream()
    reg = _window_boundary(stream)
    first = stream.select(reg)
    flat = _interior(stream, reg.digest, first, flat=True)
    assert set(_refused(lambda: stream.register_interiors(first, *flat, registered=REGISTERED, supplied=SUPPLIED))) == \
        {"R7-words", "R7-domain", "R7-supplied"}
    ireg = stream.register_interiors(first, *_interior(stream, reg.digest, first), registered=REGISTERED,
                                     supplied=SUPPLIED)
    assert ireg.record["registered"][0]["value"] == "hist" and ireg.record["registered"][0]["words"] == 512


def test_r7_reads_the_flat_form_by_its_own_type():
    """A stream that types the value flat, 32 sixteen-bit leaves a span, has R7 read each root's bytes as big-endian
    32-bit halves (`registered.bytes_leaves`), which are not a `Digest` port's words: the flat form's rows are refused
    there too."""
    rows = [_digest_words(r) for r in ROOTS]
    assert all(G.bytes_leaves(r, 16) != w for r, w in zip(ROOTS, rows))
    assert all(G.digest_leaves(r) == [(w >> b) & 1 for w in ws for b in range(16)] for r, ws in zip(ROOTS, rows))
    stream = _stream(span_values=(("hist", 32, 16),))
    reg = _window_boundary(stream)
    first = stream.select(reg)
    flat = _interior(stream, reg.digest, first, flat=True)
    assert _refused(lambda: stream.register_interiors(first, *flat, registered={"hist": 32}, supplied=SUPPLIED)) == \
        ("R7-supplied",)


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
    assert stream.interior_program(Q).root.id == P.mixed_population_program([(w, n * Q) for w, n in
                                                                            zip(WINDOW, stream.cells)]).root.id
