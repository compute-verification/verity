"""The single lowering adapter (IR §F): legacy backend names appear only in `verity_sp1.proofs.lowering.lower_subcircuit` output."""

from verity.ml.circuits import query

from verity_sp1.proofs.lowering import lower_subcircuit
from tests.program.test_running_example import make_program


def test_lowering_adapter():
    C = make_program().circuit
    # lowering of one Row call: boundary-in values become the backend's ports (ascending producing gate),
    # the subcircuit's gates its straight-line program in canonical order
    row = query(C, "//Row")[0]
    low = lower_subcircuit(row)
    kp = low["KindProgram"]
    assert [p["gate"] for p in kp["ports"]] == [0, 1, 4, 5]
    assert kp["gates"][0]["operands"] == [{"port": 0}, {"port": 2}]
    assert kp["gates"][2]["operands"] == [{"local": 0}, {"local": 1}]
    assert kp["outputs"] == [3]
    # an Input gate inside a subcircuit lowers to an input equation, not a trusted source
    low_in = lower_subcircuit(C.gate_selection([0, 14]))
    assert low_in["KindProgram"]["input_equations"][0]["gate"] == 0
