"""The small test model's epoch at class 1,1,1 as the audit's stream meets it: a declared circuit of two layers whose
circuit templates each have many ranges, interleaved, which `service.Stream` takes as the window's program."""

from __future__ import annotations

import pytest
from verity.circuits.program import Program
from verity.commitments.periods import interface as IF
from verity.protocols.proof.spot.verifier import service as S
from verity.protocols.proof.spot.verifier.one_stage import partition as P

from tests.program.predict import test_steps as ST
from tests.program.predict.test_window_circuit import GRAIN
from verity_vllm.program.predict.window_circuit import WindowClass, class_circuit


@pytest.mark.slow
def test_the_class_1_1_1_epoch_is_a_cover_the_stream_takes_at_each_templates_ranges_summed():
    """At class 1,1,1 the epoch's circuit templates have several ranges each, so the stream's former test (one range a
    template, in the window's order) refused it (`stream`). Its units, numbered by the ranges in the declared circuit's
    order, are each unit's circuit template by index as the strata give them, a disjoint cover of the population but
    not in slot order (`partition.covers`, not `partition.in_order`), which the Lean's run takes under a subset design
    only. The stream builds on it, its population program at each template's ranges summed, and refuses counts that
    leave a range out. About two minutes, the class circuit's build: the one test that holds the stream to a served
    epoch's declared circuit, the shape the next rehearsal meets, which the spot suite's toy epochs only stand in for."""
    circuit = class_circuit(ST._registered(), WindowClass(1, 1, 1), None, GRAIN)
    w = IF.Window(circuit, "ab" * 64, {"law": "subset", "k": 2}, 64).window
    order = w.query["params"]["templates"]
    ranges = P.ranges(w.program, w.query)
    n = sum(m for _t, _lo, m in ranges)
    assert len(ranges) > len(order) and [t for t, _lo, _m in ranges] != order
    assert (len(ranges), len(order), n) == (36, 14, 139)
    by_ranges = [t for t, _lo, m in ranges for _ in range(m)]
    by_strata = [None] * n
    for t, rs in P.strata(w.program, w.query):
        for lo, hi in rs:
            by_strata[lo:hi] = [t] * (hi - lo)
    assert by_ranges == by_strata
    assert P.covers(w.program, w.query, n) and not P.in_order(w.program, w.query, n)
    defs = {P.template_id(fn): fn for fn, _ in circuit.templates}
    units = [defs[t] for t in order]
    spec = S.Spec((S.Value("x", "hm96-sha512/row/v1", "serve"),), S.All(), S.Check(tuple(P.program_digest(Program(u)) for u in units), w.query, proof_deadline=60.0),
                  (), b"window-stream-test")
    stream = S.Stream(spec, units, budget=1, delta=0.1, challenger="challenger-test", program=w.program)
    counts = {t: 0 for t in order}
    for t, _lo, m in ranges:
        counts[t] += m
    assert stream._program(n, counts) is w.program
    t, _lo, m = next(r for r in ranges if counts[r[0]] > r[2])
    assert stream._program(n - m, {**counts, t: counts[t] - m}) is None
