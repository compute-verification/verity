"""The post-seed verdict (`audit.post_seed`): the verifier runs property/v1 on the writers' Program W itself and binds it
to the session by equalities between public things, each refused by name: W's digest and census (1), the memory-map
units' value rows (2), the session's appends (3) and property/v1's check.  `ledger` holds the verdict's windows and
lifetime cap to the record's (4), which waits on their conversion."""

from __future__ import annotations

import dataclasses
import hashlib
from fractions import Fraction
from types import SimpleNamespace

import pytest

from test_appends import P512, Link, honest_session
from verity.core.protocols.remote_bandwidth import audit
from verity.ml.circuits import stacking
from verity.ml.circuits.census import census
from verity.ml.circuits.codec import program_digest
from verity.ml.circuits.defs import CompositeDefinition
from verity.ml.circuits.ops import fp32, prims
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import array_of
from verity.ml.circuits.types import Array, Value

S8, S32 = Value(8), Value(32)
K = 64
FREE = {"roundings": 0, "compare_select": 0, "gathers": 0, "folds": 0}
LIFETIME = 4
R_IN = 10**6
COMMON = dict(capacity_bytes_per_s=4096 * 10**9 / 10, chip_bytes=0, rebuild=audit.Rebuild(P512, 0.0))


def comp(name: str, params, ret, body) -> CompositeDefinition:
    return CompositeDefinition(name, 1, (), lambda _S: (tuple(params), ret), body, register=False)


STEP = comp("PostSeedStep", (("acc", S32), ("x", Array(K, S8)), ("w", Array(K, S8))), S32,
            lambda B, _S, acc, x, w: B.call(prims.BlackwellS8ImmaDot32, B.call(prims.BlackwellS8ImmaDot32, acc, x[:32],
                                                                               w[:32]), x[32:], w[32:]))
EPILOGUE = comp("PostSeedEpilogue", (("acc", S32),), S32,
                lambda B, _S, acc: B.call(fp32.F32Add, B.call(fp32.F32Fma, acc, acc, acc), acc))


def writers(steps: int = 5, epilogue: bool = False) -> tuple[Program, list[int]]:
    """`test_stacking.py`'s exact int8 Program: step s reads its run input x[s] and the registered w and carries step
    s − 1's accumulator, from zero, in two chained s8 core steps over 64 lanes; with `epilogue`, each step also outputs
    an FP32 FMA and an add of it, 2 roundings.  Returns the Program and its step ends."""
    ends: list[int] = []

    def body(B, _S, x, w):
        ends.clear()
        acc, outs = B.call(prims.ZERO32), []
        for s in range(steps):
            acc = B.call(STEP, acc, x[s], w)
            outs.append(B.call(EPILOGUE, acc) if epilogue else acc)
            ends.append(len(B.nodes))
        return array_of(outs)

    fn = comp(f"PostSeedWriters{steps}{'Epilogue' if epilogue else ''}",
              (("x", Array(steps, Array(K, S8))), ("w", Array(K, S8))), Array(steps, S32), body)
    return Program(fn, registered=("w",)), ends


#: the one-stage value commitments' leaves (opaque commit strings here), and two memory-map units' public rows and rows
VALUES = [hashlib.sha512(b"value leaf %d" % i).digest() for i in range(4)]
MAP = SimpleNamespace(C=2, K=2, V=4)
MAP_PUBLIC = [{"first": 8, "count": 2, "present": 0b11, "pos": [0, 1]}, {"first": 10, "count": 1, "present": 0b01,
                                                                          "pos": [3, 0]}]
MAP_ROWS = [{"val0": VALUES[0], "val1": VALUES[1], "pay0": b"page 8", "pay1": b"page 9"},
            {"val0": VALUES[3], "pay0": b"page 10"}]


@pytest.fixture(scope="module")
def record() -> audit.Record:
    r = honest_session()
    assert all(v.accepted for v in r.finish())
    return r.s.record()


def windows_of(rec: audit.Record) -> list[stacking.Window]:
    """The windows these tests judge at: the record's passes' steps, its certified reads and its spans.  Not the record's
    own reading, which is binding 4's open question."""
    assert [w.steps for w in rec.windows] == [(0, 1, 2), (3, 4)]
    return [stacking.Window(range(0, 3), rec.windows[0].receipts * 4096, Fraction(rec.windows[0].span_ns, 10**9)),
            stacking.Window(range(3, 5), rec.windows[1].receipts * 4096, Fraction(rec.windows[1].span_ns, 10**9))]


def verdict(rec: audit.Record, program=None, ends=None, **kw) -> audit.PostSeed:
    if program is None:
        program, ends = writers()
    d = program_digest(program)
    args = dict(census=census(program, ends, d), audited=d, values=VALUES, map_statics=MAP, map_public=MAP_PUBLIC,
                map_rows=MAP_ROWS, link=Link(True, 8 * 4096, {}), c=FREE, windows=windows_of(rec), lifetime=LIFETIME,
                R_in=R_IN)
    return audit.post_seed(rec, program, **{**args, **kw})


def test_the_verifier_s_verdict_on_an_exact_int8_writer_holds_and_carries_what_it_was_judged_at(record) -> None:
    """The exact int8 Program passes property/v1 with every unit free, at 64 run-input bytes a step, and every binding
    holds: the verdict names W's digest, its L and its windows, and asked the link about the record's own chain."""
    program, ends = writers()
    link = Link(True, 8 * 4096, {})
    v = verdict(record, link=link)
    assert v.holds and v.refused == ()
    assert v.program == program_digest(program) and v.lifetime == LIFETIME and v.windows == tuple(windows_of(record))
    assert v.checked.ok and v.checked.numbers["max_lifetime"] == 1 and v.checked.numbers["macs"] == 10 * 32
    assert [w["input_bytes"] for w in v.checked.numbers["windows"]] == ["192", "128"]
    assert link.asked == [record.appends] and record.appends is not None
    assert set(v.refused) <= set(audit.POST_SEED_BINDINGS)


def test_a_verdict_on_another_program_is_refused(record) -> None:
    """Binding 1: W's digest must be the one-stage audit's `program`.  The digest of a Program one step shorter, or of
    the same steps with an epilogue, is another Program's."""
    for other, _ in (writers(steps=4), writers(epilogue=True)):
        v = verdict(record, audited=program_digest(other))
        assert not v.holds and v.refused == ("program",) and v.checked.ok


def test_a_census_that_isn_t_w_s_is_refused_and_not_judged(record) -> None:
    """Binding 1: the census must be W's, at its own steps, naming W's digest.  One naming no Program, one of another
    Program under W's digest, and one of W whose step reads were edited are refused, and property/v1 never reads
    them."""
    program, ends = writers()
    d = program_digest(program)
    other, other_ends = writers(epilogue=True)
    honest = census(program, ends, d)
    edited = {**honest, "values": {**honest["values"], "reads": [[] for _ in honest["values"]["reads"]]}}
    for cen in (census(program, ends), census(other, other_ends, d), edited):
        v = verdict(record, census=cen)
        assert not v.holds and v.refused == ("census",) and v.checked is None


def test_a_map_unit_s_value_row_that_isn_t_w_s_leaf_is_refused(record) -> None:
    """Binding 2: each memory-map unit's `val{k}` row is W's value leaf at `pos[k]`.  Another leaf in its place is
    refused, as is a `pos` past the value commitments."""
    for unit, row, leaf in ((1, "val0", 2), (0, "val1", 0)):
        rows = [dict(r) for r in MAP_ROWS]
        rows[unit][row] = VALUES[leaf]
        v = verdict(record, map_rows=rows)
        assert not v.holds and v.refused == ("values",)
    v = verdict(record, values=VALUES[:3])
    assert not v.holds and v.refused == ("values",)


def test_a_verdict_whose_map_units_don_t_make_the_session_s_appends_is_refused(record) -> None:
    """Binding 3: the session's appends are the map units' payloads, `RequirementLink.mapped` of the record's chain."""
    link = Link(True, 8 * 4096, {}, maps=False)
    v = verdict(record, link=link)
    assert not v.holds and v.refused == ("mapped",) and link.asked == [record.appends]


def test_a_writer_over_its_budget_by_one_unit_in_one_window_is_refused(record) -> None:
    """property/v1's check: with the epilogue's 2 roundings a step, window 0 (3 steps, 30 receipts) spends 6 units
    and window 1 (2 steps, 24 receipts) 4.  At 6 units per window 0's certified bytes both fit; at 5, window 0 is one
    unit over and window 1 exactly at its budget."""
    program, ends = writers(epilogue=True)
    reads = record.windows[0].receipts * 4096
    v = verdict(record, program, ends, c={**FREE, "roundings": Fraction(6, reads)})
    assert v.holds and [w["units"]["roundings"] for w in v.checked.numbers["windows"]] == [6, 4]
    v = verdict(record, program, ends, c={**FREE, "roundings": Fraction(5, reads)})
    assert not v.holds and v.refused == ("property",)
    assert [(r["code"], r["steps"]) for r in v.checked.reasons] == [("roundings", [0, 3])]


def test_a_verdict_that_fails_credits_nothing_and_its_l_is_a_count_of_steps(record) -> None:
    v = verdict(record, link=Link(True, 8 * 4096, {}, maps=False))
    with pytest.raises(ValueError, match="fails"):
        audit.ledger(record, **COMMON, credit_post_seed=v)
    for bad in (None, -1, 1.0, True):
        with pytest.raises(ValueError, match="census steps"):
            verdict(record, lifetime=bad)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason=audit.OPEN_READING)
def test_the_ledger_holds_a_verdict_to_the_record_s_windows_and_lifetime_cap(record) -> None:
    """Binding 4: `ledger` credits a verifier's verdict judged at the record's own windows and lifetime cap
    (`property_reading`), and refuses one at another record's reads or another L, though property/v1 holds at both."""
    windows, L = audit.property_reading(record)
    audit.ledger(record, **COMMON, credit_post_seed=verdict(record, windows=windows, lifetime=L))
    more = [dataclasses.replace(windows[0], reads=windows[0].reads + 4096), *windows[1:]]
    for ws, lifetime in ((more, L), (windows, L + 1)):
        v = verdict(record, windows=ws, lifetime=lifetime)
        assert v.holds
        with pytest.raises(ValueError, match="binding 4"):
            audit.ledger(record, **COMMON, credit_post_seed=v)
