"""A member's rows in another form than its circuit's are refused by name before any file is staged
(`flock_inputs.ports_form`): output rows padded to whole SHA-512 blocks where the circuit's are exactly their words
(`ROWS_PADDED`), and a port committed as one whole row where the circuit cuts it (`ROWS_CUT`)."""
from __future__ import annotations

import pytest

from verity.commitments.periods import flock_inputs as FI

CUT = {"ports": [["x#0", 16, 3], ["x#1", 16, 3], ["x#2", 16, 3], ["x#3", 16, 3], ["w", 1024, 16], ["o0", 16, 3]],
       "out_ports": [["o0", 16]], "input_ports": 5}


def member(ports, input_ports):
    return {"ports": ports, "out_ports": [["o0", 16]], "input_ports": input_ports}


def stage(meta, mem):
    return FI.flock_input_files("/nonexistent", 0, FI.FlockCircuit(meta, "", ""), "/nonexistent",
                                files=({"members": [mem]}, {}))


def test_the_circuits_own_rows_pass_the_form():
    assert FI.ports_form(CUT, member([p[:2] for p in CUT["ports"]], 5)) is None


def test_padded_output_rows_are_refused_by_name():
    mem = member([*(p[:2] for p in CUT["ports"][:5]), ["o0", 64]], 5)
    assert FI.ports_form(CUT, mem).startswith(FI.ROWS_PADDED)
    with pytest.raises(ValueError, match=f"member 0: {FI.ROWS_PADDED}"):
        stage(CUT, mem)


def test_a_whole_row_where_the_circuit_cuts_is_refused_by_name():
    mem = member([["x", 64], ["w", 1024], ["o0", 16]], 2)
    assert FI.ports_form(CUT, mem).startswith(FI.ROWS_CUT)
    with pytest.raises(ValueError, match=f"member 0: {FI.ROWS_CUT}"):
        stage(CUT, mem)


def test_rows_of_other_words_are_no_other_form():
    assert FI.ports_form(CUT, member([["x", 32], ["w", 1024], ["o0", 16]], 2)) is None
