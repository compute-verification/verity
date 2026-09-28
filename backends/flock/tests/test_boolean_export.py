"""The Boolean export's sizings of what is not one listed circuit: a constant-table read (M0's gate circuit), a wide gather as a
multiplexer, each against the circuit it counts; and its modules, expanded, against the circuits they rebuild."""
import gzip
import json
import random
import tempfile
from pathlib import Path

import pytest

from verity_flock import boolean_export as BX
from verity_flock import gf2
from verity_flock import ir_lower as IL
from verity_flock import tail_pieces as TP

pytest.importorskip("verity_vllm.program.registry.prims")


def _analysed(build):
    reg = BX.Registry(Path(tempfile.mkdtemp()))
    C, root = BX.trace(build)
    return reg.nodes[reg.analyse(C, root, "root", "test")], C


def test_a_table_read_is_m0s_gate_circuit():
    """A MUFU table read is M0's table circuit, counted over the pinned table: 41,308 ANDs for the 2^23 tables, 49,576 for 2^24,
    and in a traced piece it is a subcircuit of kind ``table`` with those counts."""
    assert [BX.table_gates(t)["and"] for t in ("ex2", "rcp", "rsq", "sqrt")] == [41308, 41308, 49576, 49576]

    def build():
        C = gf2.Circuit(cse=True)
        C.output("out0", IL.PIECES["MufuEx2Ftz_v1"](C, C.input("in0", 32)))
        return C

    reg = BX.Registry(Path(tempfile.mkdtemp()))
    C, root = BX.trace(build)
    reg.analyse(C, root, "ex2", "piece")
    (read,) = [n for n in reg.nodes.values() if n["function"] == "table_read"]
    assert (read["kind"], read["and"], read["table"], read["children"]) == ("table", 41308, "ex2", [])


def test_a_wide_gather_is_its_muxes_counted(monkeypatch):
    """A gather wider than GATHER_INLINE is a composed Definition with exactly the counts of the one circuit."""
    V = 5000
    reg = BX.Registry(Path(tempfile.mkdtemp()))
    walker = BX.Walker(reg, None, {})
    composed = walker.defs[walker.gather(V)["def"]]
    monkeypatch.setattr(BX, "GATHER_INLINE", V)
    whole = reg.nodes[BX.Walker(reg, None, {}).gather(V)["ref"]]
    assert (composed["and"], composed["xor"], composed["not"]) == (whole["and"], whole["xor"], whole["not"])
    assert composed["and"] == 16 * (V - 1) + reg.nodes[composed["children"][1]["ref"]]["and"]


def _piece(pid, widths):
    def build():
        C = gf2.Circuit(cse=True)
        C.output("out0", IL.PIECES[pid](C, *[C.input(f"in{i}", w) for i, w in enumerate(widths)]))
        return C

    reg = BX.Registry(Path(tempfile.mkdtemp()))
    C, root = BX.trace(build)
    return reg, reg.analyse(C, root, pid, "piece", list_root=True)


@pytest.mark.parametrize("pid,widths", [("F32Add_v1", [32, 32]), ("F32Mul_v1", [32, 32]), ("F32Fma_v1", [32, 32, 32])])
def test_modules_expand_to_the_subcircuits_gates(pid, widths):
    """Every instance's module checks one level against the circuit and equals its type's; expanded from the modules alone, a
    piece is its circuit (checked in ``analyse``) with the tree's counts, the shared ANDs read through taps."""
    reg, rid = _piece(pid, widths)
    rec, gl = reg.nodes[rid], BX.expand_module(reg, rid)
    assert reg.module_instances > len(reg.modules) and not reg.module_differs and reg.expanded_verified
    assert (gl["counts"]["and"], gl["counts"]["xor"], gl["counts"]["not"]) == (rec["and"], rec["xor"], rec["not"])
    assert BX.module_count_differs(reg) == []
    assert BX.DEAD not in gl["outputs"]


def test_a_table_reads_module_counts_m0s_circuit():
    """A table read is one module item; expanded, the piece's ANDs and NOTs are the tree's with the table read counting M0's
    circuit, and its XORs are more by the table index's (the tree leaves those out)."""
    reg, rid = _piece("MufuEx2Ftz_v1", [32])
    rec, gl = reg.nodes[rid], BX.expand_module(reg, rid)
    (t,) = gl["tables"]
    slot = BX.table_gates(t["table"])
    assert t["table"] == "ex2" and t["width"] == 31
    assert gl["counts"]["and"] + slot["and"] == rec["and"] and gl["counts"]["not"] + slot["not"] == rec["not"]
    assert 0 < gl["counts"]["xor"] + slot["xor"] - rec["xor"] <= len(t["index"])


def test_the_modules_file_round_trips():
    reg, rid = _piece("F32Add_v1", [32, 32])
    path = Path(tempfile.mkdtemp()) / "modules.json.gz"
    BX.write_modules(reg, path)
    doc = json.loads(gzip.decompress(path.read_bytes()))
    assert doc["schema"].endswith("/modules") and set(doc["modules"]) == set(reg.modules)
    assert doc["modules"][rid]["items"] == reg.modules[rid]["items"].tolist()


def test_the_gather_matches_the_ir_primitive():
    from verity_vllm.program.registry import prims as RP

    rng = random.Random(2)
    for V in (1, 2, 3, 64, 128, 1000):
        vecs = [[rng.choice([rng.randrange(V), V, V + 3, 0xFFFFFFFF, 0x80000000, rng.getrandbits(32)])]
                + [rng.getrandbits(16) for _ in range(V)] for _ in range(300)]
        C = gf2.Circuit(cse=True)
        ins = [C.input(f"i{k}", w) for k, w in enumerate([32] + [16] * V)]
        C.output("o", TP.gather_bf16(C, *ins))
        outs, _ = C.evaluate({f"i{k}": IL.slice_words([v[k] for v in vecs], w) for k, w in enumerate([32] + [16] * V)}, len(vecs))
        assert [int(x) for x in IL.unslice_words(outs["o"], len(vecs))] == [RP.gather_bf16(V).evaluate(*v) for v in vecs]
