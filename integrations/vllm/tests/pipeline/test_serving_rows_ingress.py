"""A served window's window ingress (`pipeline.ingress_rows.ingress`): a unit's input port is window ingress at its engine step
when every slice it reads is a committed member of a row before the window, so the circuit record declares it apart from the
run's ingress, and the leak count holds it apart too (`interface.input_count`'s ``window_ingress``, its ``free`` still the
run's); a port reading a row of the window, registered weights or a mix stays free.  The record states only the Programs'
public structure, and the same window ingress whether the template is chunked or not."""
from __future__ import annotations

import json

import pytest
from verity_vllm.pipeline import ingress_rows as IR, serving_rows as SRW

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import b1 as B1
from verity.ml.windows import interface as IF

FN = bind(B1.ResidualAdd, N=8)
T = descriptor_id(FN)
#: rows 10 .. 14 of each request: their two argument groups (``n``: a committed member of row j, ``p``: the weights) and steps;
#: rows 0 .. 9 are the layers before, outside the window
ARGS = {10: ([["n", 3, 0, 8]], [["n", 4, 0, 8]]),
        11: ([["n", 10, 0, 8]], [["n", 5, 0, 8]]),
        12: ([["n", 6, 0, 8]], [["p", 0, 0, 8]]),
        13: ([["n", 7, 0, 8]], [["n", 8, 0, 8]]),
        14: ([["n", 7, 0, 4], ["n", 12, 4, 8]], [["n", 9, 0, 8]])}
STEPS = {10: 0, 11: 0, 12: 0, 13: 1, 14: 1}
ARRIVALS = {"r0": 0, "r1": 3}
PARTITION = {"templates": [T], "law": "subset:1"}


@pytest.fixture
def served(monkeypatch):
    by_i = {i: {"i": i, "spec": T, "args": [list(g) for g in a]} for i, a in ARGS.items()}
    ds = {i: {"statics": {"N": 8}, "op_path": "model.layers.1.post_attention_layernorm", "row": i - 10, "step": s,
              "members": [("out", 8, "bf16")]} for i, s in STEPS.items()}

    class Index:
        def __init__(self, *_a, **_k):
            self.by_i, self.pnames = by_i, ["weights"]

        def describe(self, r):
            return ds[r["i"]]

    monkeypatch.setattr(SRW, "index_of", lambda inst, _pool, **kw: Index(inst, **kw))

    def window(partition):
        w, rows = SRW.population({"r0": {}, "r1": {}}, partition, "ab" * 64, arrivals=ARRIVALS)
        return w, rows, json.loads(IF.circuit_bytes(SRW.circuit_record(w, rows, ARRIVALS)))
    return window


def test_a_port_is_window_ingress_where_every_slice_it_reads_is_a_row_before_the_window(served):
    """Each request's rows 10 .. 14 at its arrival plus their steps: ``in0`` of rows 11 (row 10's output, inside the window)
    and 14 (half of it row 12's) and ``in1`` of row 12 (the weights) are free; every other port is window ingress, in runs
    of one port at one step over consecutive units.  None is the run's ingress, so the count's ``free`` is every input."""
    window, rows, doc = served(PARTITION)
    got = [(g["port"], g["at"], g["units"], g["step"]) for g in doc[IF.WINDOW_INGRESS]]
    assert got == [("in0", 0, 1, 0), ("in1", 0, 2, 0), ("in0", 2, 1, 0), ("in0", 3, 1, 1), ("in1", 3, 2, 1),
                   ("in0", 5, 1, 3), ("in1", 5, 2, 3), ("in0", 7, 1, 3), ("in0", 8, 1, 4), ("in1", 8, 2, 4)]
    assert {g["reader"] for g in doc[IF.WINDOW_INGRESS]} == {T} and not {"cap", "ingress"} & set(doc)
    row = 16 * 8
    assert IF.input_count(doc) == {"units": 10, "inputs": 20 * row, "wired": 0, "registered": 0, "ingress": 0,
                                   IF.WINDOW_INGRESS: 14 * row, "free": 20 * row}
    assert IF.input_count({k: v for k, v in doc.items() if k != IF.WINDOW_INGRESS})[IF.WINDOW_INGRESS] == 0
    assert IF.window_of_record(doc, window.law, {T: FN}.__getitem__).members == window.members
    assert [g.doc() for g in IR.ingress(window, rows, SRW.row_units, ARRIVALS)] == doc[IF.WINDOW_INGRESS]


def test_a_chunked_templates_window_ingress_is_its_units_and_counts_the_same(served):
    """Chunked (the partition's ``chunk_cap``), the template's units are members of at most 3 on its bit rows, and the window
    ingress the record declares covers the same units and bits across the chunks' bounds."""
    whole = served(PARTITION)[2]
    window, _rows, doc = served({**PARTITION, "chunk_cap": {T: 3}})
    assert [m.n for m in window.members] == [3, 3, 3, 1]
    assert doc[IF.WINDOW_INGRESS] == whole[IF.WINDOW_INGRESS] and IF.input_count(doc) == IF.input_count(whole)
