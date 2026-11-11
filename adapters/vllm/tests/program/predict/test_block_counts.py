"""verity_vllm.program.predict.block_counts: a day's (c, k)-inference blocks from its served windows' circuit records,
judged against its declared c, k and n. On a small circuit (two catalog templates over two slots) its count, a block's
gates, its gate table and its verdict agree with `NciCheck.Count`'s Python reference (`nci_reference.check_count_detail`
and `check_count`, which `nci-count-difftest` holds the Lean to) on the session Program, m activations of the block
Definition; the public judgment holds only its record's fields and no count, so days of different counts under n publish
the same bytes, and a refusal past n names n; the exact counts are in the private file alone (0600); a block reaching an
unpriced primitive leaves the gates undecided; and on `test_window_circuit`'s TINY window a block's fresh bits are B's k
(`inference_blocks.budgets`)."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest
from verity_vllm.program.predict import block_counts as BC
from verity_vllm.records import block_counts as RBC
from verity_vllm.records.fields import RecordError, UnknownField

from verity.circuits import codec, partition_v2 as PV2
from verity.circuits.defs import REGISTRY
from verity.circuits.population import population_program
from verity.definitions.boolean import gate_table
from verity.commitments.windows import interface as IF

REFERENCE = Path(__file__).resolve().parents[5] / "verity/protocols/inference_only/verifier/nci/lean/scripts/nci_reference.py"
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
    judgment's own table, are the judgment's gates a block, its blocks the private judgment's, and its verdict against the
    same c, k and n the judgment's."""
    pub, private = BC.judgment(DAY, [(small, None)], {"c": 10 ** 12, "k": 10 ** 12, "n": 2})
    (cls,) = pub.classes
    fn, m, steps = BC.session(small)
    assert (m, steps, cls.slots, cls.steps, private["total"], private["windows"]) == (2, 3, 2, 3, 2, 1)
    assert cls.definition == PV2.id_digest(fn) and not cls.unpriced and private["blocks"] == {cls.definition: 2}
    text = codec.canonical_json(codec.encode_program(population_program(fn, m))).decode()
    key = codec.descriptor_id(fn)
    for n, verdict in ((2, "accept"), (1, "refuse")):
        inst = {"program": text, "theta": [], "blocks": [key], "table": dict(pub.table), "c": cls.gates, "k": 10 ** 12,
                "n": n}
        got = _reference().check_count_detail(inst)
        assert got["blocks"][0][:2] == [key, cls.gates] and got["blocks"][0][3] == m == got["total"]
        assert _reference().check_count(inst) == {"complies": got["complies"]}
        mine, _ = BC.judgment(DAY, [(small, None)], {"c": cls.gates, "k": 10 ** 12, "n": n})
        assert mine.verdict == verdict == ("accept" if got["complies"] else "refuse")
    by = {t: BC.gates(BC.resolve(t), pub.table)[0] for t in (GEMM, ROPE)}
    assert cls.gates == 2 * by[GEMM] + 3 * by[ROPE] > 0
    assert cls.fresh == IF.input_count(small)["free"] // 2


def _run(tmp_path: Path, record: dict, windows: int, decl: dict, tag: str) -> tuple[bytes, Path]:
    """`main` on ``windows`` served windows of ``record``, each with a registration: its public file's bytes and its
    private file."""
    (tmp_path / "circuit.json").write_text(json.dumps(record))
    (tmp_path / "registration.json").write_text("{}")
    (tmp_path / "declaration.json").write_text(json.dumps(decl))
    out, private = tmp_path / f"public-{tag}.json", tmp_path / f"private-{tag}.json"
    a = BC.Options(day=DAY, declaration=tmp_path / "declaration.json", private=private, out=out,
                   window=[[str(tmp_path / "circuit.json"), "-"]] * windows)
    assert BC.main(a) == 0
    return out.read_bytes(), private


def test_days_of_different_counts_under_n_publish_the_same_bytes(small, tmp_path, capsys):
    """One, three and seventeen windows of two slots (2, 6 and 34 blocks) under n = 40 publish byte-identical public
    judgments, which name no count and which the record reads back; only the private file (0600) holds the exact counts."""
    decl = {"c": 10 ** 12, "k": 10 ** 12, "n": 40, "note": "anything else the declaration holds"}
    runs = {w: _run(tmp_path, small, w, decl, str(w)) for w in (1, 3, 17)}
    assert len({pub for pub, _ in runs.values()}) == 1
    doc = json.loads(runs[1][0])
    assert RBC.BlockJudgment.of(doc, "public").dump() == doc and set(doc) == set(RBC.FIELDS)
    assert (doc["verdict"], doc["reason"], doc["checks"]) == ("accept", None, dict.fromkeys(RBC.RULES, "accept"))
    assert all(set(c) == set(RBC.CLASS) for c in doc["classes"])
    for w, (_, private) in runs.items():
        held = json.loads(private.read_text())
        assert (held["total"], held["windows"], private.stat().st_mode & 0o777) == (2 * w, w, 0o600)
    assert "blocks" not in capsys.readouterr().err


def test_a_refusal_past_n_names_n_and_not_the_count(small, tmp_path, capsys):
    """Seventeen windows (34 blocks) against n = 5 refuse on the count rule alone, the reason naming n and not the count,
    padded or exact; the stderr line says the same."""
    pub, private = _run(tmp_path, small, 17, {"c": 10 ** 12, "k": 10 ** 12, "n": 5}, "17")
    doc = json.loads(pub)
    assert doc["checks"] == {"gates": "accept", "fresh": "accept", "count": "refuse"} and doc["verdict"] == "refuse"
    assert doc["reason"] == "the blocks' exact count is more than n = 5"
    err = capsys.readouterr().err
    for count in ("34", str(PV2.grid(34)), "17"):
        assert count not in doc["reason"] and count not in err
    assert json.loads(private.read_text())["total"] == 34


def test_the_judgment_holds_its_fields_alone(small):
    """The public record refuses a count (a class's blocks, the day's total or windows), a window list, and checks or a
    verdict that aren't its classes'; it names no window, run or root."""
    pub, private = BC.judgment(DAY, [(small, {"weights/x@4": "ab" * 64})] * 3, {"c": 10 ** 12, "k": 10 ** 12, "n": 6})
    doc = pub.dump()
    for extra in ({"total": 6}, {"windows": 3}, {"blocks": private["blocks"]}):
        with pytest.raises(UnknownField):
            RBC.BlockJudgment.of({**doc, **extra}, "judgment")
    with pytest.raises(UnknownField):
        RBC.BlockJudgment.of({**doc, "classes": [{**doc["classes"][0], "blocks": 6}]}, "judgment")
    with pytest.raises(RecordError, match="checks.gates"):
        RBC.BlockJudgment.of({**doc, "c": 0}, "judgment")
    with pytest.raises(RecordError, match="verdict"):
        RBC.BlockJudgment.of({**doc, "checks": {**doc["checks"], "count": "refuse"}}, "judgment")
    text = json.dumps(doc)
    assert set(re.findall(r"[0-9a-f]{128}", text)) == {doc["classes"][0]["definition"], doc["declaration"]}
    assert "ab" * 64 not in text and "test/blocks" not in text


def test_the_gate_table_prices_bit_at_by_its_boolean_version():
    """`TokenFeed_v1` reads `BitAtx32_v1`, which the one gate table (`definitions.boolean.gate_table`) prices by
    `BitAtx32_v2`: 147 gates, every primitive priced, so a feed's gates are its primitives' sum."""
    feed = BC.resolve("TokenFeed_v1{NP=4,FIRST=false}")
    table, unpriced = gate_table([feed])
    assert unpriced == [] and table["BitAtx32_v1"] == 147
    g, miss = BC.gates(feed, table)
    assert not miss and g == sum(table.values()) > 147


def test_a_primitive_with_no_boolean_version_is_unpriced():
    """`F32TanhRn_v1` has no Boolean version: the table leaves it out and a block lists its activations apart, its gates not
    guessed."""
    tanh = REGISTRY["F32TanhRn_v1"]
    assert gate_table([tanh]) == ({}, ["F32TanhRn_v1"])
    assert BC.gates(tanh, {}) == (0, {"F32TanhRn_v1": 1})


def test_a_block_with_an_unpriced_primitive_leaves_the_gates_undecided():
    """Its gates are a lower bound: under c the gates rule is undecided and the day refused, naming the primitive and c; a
    lower bound already over c refuses on the gates."""
    cls = RBC.BlockClass(definition="ab" * 64, slots=2, steps=3, gates=100, unpriced={"BitAtx32_v1": 1}, fresh=7)
    checks, reason = BC.judge([cls], 2, 10 ** 6, 10, 2)
    assert checks == {"gates": "undecided", "fresh": "accept", "count": "accept"}
    assert reason == ("undecided: a block reaches BitAtx32_v1, which the gate table can't price, so its gates are a lower "
                      "bound against c = 1000000")
    assert BC.judge([cls], 2, 99, 10, 2) == ({"gates": "refuse", "fresh": "accept", "count": "accept"},
                                             "a block of 100 gates, more than c = 99")


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
    pub, private = BC.judgment(DAY, [(doc, roots)], {"c": 0, "k": 0, "n": 0}, pins=roots)
    (cls,) = pub.classes
    assert (cls.slots, private["total"], cls.fresh) == (2, 2, IB.budgets(circuit)[1]) and cls.fresh > 0
    assert not cls.unpriced and cls.gates > IB.budgets(circuit)[0]
