"""Scratch, never for landing: each wire of the small served phase's circuit the vllm suite builds
(tests/program/predict/test_window_circuit.py) under the record form's leaf-widths rule at this checkout, one at a
time: how many are admitted, and each one refused with its two widths. Run from integrations/vllm:
uv run python served_wires.py"""
import sys

sys.path[:0] = [".", "../../verity/core/protocols/network/nci/lean/scripts"]
import nci_record_reference as ref
from tests.program.predict import test_window_circuit as T
from verity.ml.windows import interface as IF

c = T._circuit(T.WINDOW)
ts = [ref.Template(IF.descriptor_id(fn), fn, n) for fn, n in c.templates]


def refused(w):
    try:
        ref.leaf_widths(ts, [{**w.doc(), "c0": w.cut[0], "C": w.cut[1]}], c.cuts)
    except ref.Malformed as e:
        return e.message.split(". ")[0]


bad = [m for w in c.wires if (m := refused(w))]
print(f"{len(c.wires)} wires, {len(c.wires) - len(bad)} admitted, {len(bad)} refused:", *bad, sep="\n")
