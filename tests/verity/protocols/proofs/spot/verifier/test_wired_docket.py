"""R2 with a wired Program (`docket.check`'s `wired`, `wired_record`): the record names the wired Program, the window's
circuit is its record's text, and the program is its cut under that record, each refused `R2-program`."""

from __future__ import annotations

import hashlib
import json

from verity.circuits import codec
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.ops.boolean import cast as BC, gemm as BG
from verity.circuits.population import cut_wires
from verity.commitments.periods import interface as I
from verity.protocols.proofs.spot.prover import docket as DK
from verity.protocols.proofs.spot.verifier.one_stage import docket as R

SAMPLING_DESIGN = {"law": "subset", "k": 2}
SCHEME = "frame-v3-sha512/hm96-sha512"
BASE = ("R3-roots",)


def rounded(n: int, at: int = 0) -> I.Circuit:
    """``n`` f32 words rounded to bf16 and widened back, widening ``at + i``'s ``a`` wired from rounding ``i``."""
    first, then = bind(BG.F2fpBf16), bind(BC.Bf16ToF32)
    wire = I.Wire(descriptor_id(then), "a", descriptor_id(first), "o0", at=at, radix=(n - at,), base=0, step=(1,))
    return I.Circuit("bits", ((first, n), (then, n)), wires=(wire,))


def _case(c: I.Circuit):
    wired, _ = c.program()
    record = {**c.description(), "external": list(c.external)}
    cut, query, _ = cut_wires(wired, record)
    circuit = hashlib.sha512(json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n").hexdigest()
    rec = DK.record(cut, query, SAMPLING_DESIGN, SCHEME, [], {"kind": "served", "row": 1, "run": "r-test",
                                                              "circuit": circuit}, leaf_layer="ee" * 64)
    return wired, record, cut, query, {**rec, "program": codec.program_digest(wired)}


def _codes(rec, cut, query, **kw):
    return R.check(rec, cut, query, sampling_design=SAMPLING_DESIGN, scheme=SCHEME, ports={}, leaf_layer="ee" * 64,
                   **kw).codes


def test_the_honest_wired_pair_is_accepted():
    wired, record, cut, query, rec = _case(rounded(4))
    assert codec.program_digest(wired) != codec.program_digest(cut)
    assert _codes(rec, cut, query, wired=wired, wired_record=record) == BASE


def test_a_record_naming_the_unwired_digest_is_refused():
    wired, record, cut, query, rec = _case(rounded(4))
    unwired = {**rec, "program": codec.program_digest(cut)}
    assert _codes(unwired, cut, query, wired=wired, wired_record=record) == ("R2-program", *BASE)


def test_a_wired_program_of_another_record_of_the_same_cut_is_refused():
    """Two wirings of the same templates cut to the same program; the record names the first, the verifier holds the
    second, and the window's circuit is the second's text."""
    wired, record, cut, query, rec = _case(rounded(4))
    other, other_record, other_cut, _, other_rec = _case(rounded(4, at=1))
    assert codec.program_digest(other_cut) == codec.program_digest(cut)
    assert codec.program_digest(other) != codec.program_digest(wired)
    named_first = {**other_rec, "program": rec["program"]}
    assert _codes(named_first, cut, query, wired=other, wired_record=other_record) == ("R2-program", *BASE)


def test_a_program_that_is_not_the_cut_is_refused():
    wired, record, _, _, rec = _case(rounded(4))
    _, _, other_cut, other_query, _ = _case(rounded(6))
    assert _codes(rec, other_cut, other_query, wired=wired, wired_record=record)[:1] == ("R2-program",)


def test_a_record_that_is_not_the_windows_circuit_is_refused():
    """The cut holds under the record given, but the window registered another circuit's text."""
    wired, record, cut, query, rec = _case(rounded(4))
    moved = {**rec, "window": {**rec["window"], "circuit": "00" * 64}}
    assert _codes(moved, cut, query, wired=wired, wired_record=record) == ("R2-program", *BASE)


def test_without_a_wired_program_the_record_names_the_program():
    _, _, cut, query, rec = _case(rounded(4))
    assert _codes({**rec, "program": codec.program_digest(cut)}, cut, query) == BASE
    assert _codes(rec, cut, query) == ("R2-program", *BASE)
