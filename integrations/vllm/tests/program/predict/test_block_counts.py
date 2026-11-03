"""verity_vllm.program.predict.block_counts: a day's (c, k)-inference blocks from its served windows' circuit records, as a
public report. On a small circuit (two catalog templates over two slots) its count, a block's gates and its gate table agree
with `NciCheck.Count`'s Python reference (`nci_reference.check_count`, which `nci-count-difftest` holds the Lean to) on the
session Program, m activations of the block Definition; the report holds only its record's fields, every count padded; and
on `test_window_circuit`'s TINY window a block's fresh bits are B's k (`inference_blocks.budgets`).
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest
from verity_vllm.program.predict import block_counts as BC
from verity_vllm.records import block_counts as RBC
from verity_vllm.records.fields import RecordError, UnknownField

from verity.ml.circuits import codec, partition_v2 as PV2
from verity.ml.circuits.population import population_program
from verity.ml.windows import interface as IF

REFERENCE = Path(__file__).resolve().parents[5] / "verity/core/protocols/network/nci/lean/scripts/nci_reference.py"
GEMM = 'Gemm_v2{K=64,N=8,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}'
ROPE = "RoPEHead_v1{D=16}"
DAY = "2026-11-01"


def _reference():
    spec = importlib.util.spec_from_file_location("nci_reference", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def small() -> dict:
    """A window circuit record of two slots and three steps: four `GEMM` units, a slot's two in one entry, and six `ROPE`
    units, a slot's three across the steps."""
    entries = (IF.InferenceEntry(GEMM, 0, (2, 2), ("slot", "-"), 0), IF.InferenceEntry(ROPE, 0, (3, 2), ("step", "slot"), 0))
    circuit = IF.Circuit("test/blocks", ((BC.resolve(GEMM), 4), (BC.resolve(ROPE), 6)),
                         blocks=IF.Blocks(slots=2, steps=3, inference=entries))
    return IF.descriptor_record(circuit, 1)


def test_the_count_is_ncicheck_counts_on_the_session_program(small):
    """The session Definition activated m times is the block Definition `NciCheck.Count` counts: its row's gates, by the
    report's own table, are the report's gates a block, and its blocks the report's."""
    rep = BC.count(DAY, [(small, None)])
    (cls,) = rep.classes
    fn, m, steps = BC.session(small)
    assert (m, steps, cls.slots, cls.steps, cls.blocks, rep.total) == (2, 3, 2, 3, 2, 2)
    assert cls.definition == PV2.definition_digest(fn) and not cls.unpriced
    text = codec.canonical_json(codec.encode_program(population_program(fn, m))).decode()
    key = codec.descriptor_id(fn)
    got = _reference().check_count({"program": text, "theta": [], "blocks": [key], "table": dict(rep.table),
                                    "c": cls.gates, "k": 10 ** 12, "n": rep.total})
    assert got["blocks"][0][:2] == [key, cls.gates] and got["blocks"][0][3] == m
    assert (got["total"], got["complies"]) == (rep.total, True)
    by = {t: BC.gates(BC.resolve(t), rep.table)[0] for t in (GEMM, ROPE)}
    assert cls.gates == 2 * by[GEMM] + 3 * by[ROPE] > 0
    assert cls.fresh == IF.input_count(small)["free"] // 2


def test_the_report_holds_its_fields_alone_its_counts_padded(small):
    """Seventeen windows of two slots are 34 blocks, published as the grid's 36; the report is its record's fields, which
    `of` reads back, and refuses one more (a window list, a partition row); it names no window, run or root."""
    rep = BC.count(DAY, [(small, {"weights/x@4": "ab" * 64})] * 17)
    doc = rep.dump()
    assert RBC.BlockCounts.of(json.loads(json.dumps(doc)), "report") == rep
    assert set(doc) == set(RBC.FIELDS) and all(set(c) == set(RBC.CLASS) for c in doc["classes"])
    assert (doc["classes"][0]["blocks"], doc["total"]) == (PV2.grid(34), 36)
    for extra in ({"windows": 17}, {"entries": small["blocks"]["inference"]["entries"]}):
        with pytest.raises(UnknownField):
            RBC.BlockCounts.of({**doc, **extra}, "report")
    with pytest.raises(UnknownField):
        RBC.BlockCounts.of({**doc, "classes": [{**doc["classes"][0], "at": 0}]}, "report")
    with pytest.raises(RecordError, match="padded"):
        RBC.BlockCounts.of({**doc, "classes": [{**doc["classes"][0], "blocks": 34}], "total": 34}, "report")
    text = json.dumps(doc)
    assert set(re.findall(r"[0-9a-f]{128}", text)) == {doc["classes"][0]["definition"]}
    assert "ab" * 64 not in text and "test/blocks" not in text


def test_a_primitive_with_no_boolean_version_is_unpriced():
    """`TokenFeed_v1` reads `BitAtx32_v1`, which no Boolean Definition names as its word: the table leaves it out and a
    block lists its activations apart, its gates not guessed."""
    feed = BC.resolve("TokenFeed_v1{NP=4,FIRST=false}")
    table, unpriced = BC.gate_table(BC.primitives([feed]))
    assert unpriced == ["BitAtx32_v1"] and "BitAtx32_v1" not in table
    g, miss = BC.gates(feed, table)
    assert dict(miss) == {"BitAtx32_v1": 1} and g == sum(table[k] for k in BC.primitives([feed]) if k in table) > 0


@pytest.mark.slow
def test_on_the_tiny_window_a_blocks_fresh_bits_are_bs_k():
    """TINY's window at its class (2 slots): two blocks a window, each of the session's gates, and fresh bits B's k with the
    weights held (its slot's lane of the scan state)."""
    from tests.program.predict.test_window_circuit import WINDOW, _circuit
    from verity_vllm.program.predict import inference_blocks as IB
    from verity_vllm.program.predict.window_builder import WEIGHTS

    circuit = _circuit(WINDOW)
    doc = {**circuit.description(), "name": circuit.name, "partition": "",
           "blocks": circuit.blocks.doc(sum(circuit.units.values()), "")}
    roots = {p: "ab" * 64 for p in circuit.registered.params if p.split("/", 1)[0] == WEIGHTS}
    rep = BC.count(DAY, [(doc, roots)], pins=roots)
    (cls,) = rep.classes
    assert (cls.slots, cls.blocks, cls.fresh) == (2, 2, IB.budgets(circuit)[1]) and cls.fresh > 0
    assert set(cls.unpriced) == {"BitAtx32_v1"} and cls.gates > IB.budgets(circuit)[0]
