"""The Boolean export: its circuit types (`circuit_types`) against the circuits they are, a table read at the prover's inline
read, a wide gather as its multiplexer."""
import gzip
import json
import random
import tempfile
from pathlib import Path

import pytest

from verity_flock import boolean_export as BX
from verity_flock import circuit_types as CT
from verity_flock import gf2
from verity_flock import ir_lower as IL
from verity_flock import tail_pieces as TP

pytest.importorskip("verity.ml.definitions.registry.prims")


def _types(build, name="root", label=None):
    reg = BX.Registry()
    return reg, reg.type_of(build, name, "test", label)


def _piece_build(pid, widths):
    def build():
        C = gf2.Circuit(cse=True)
        C.output("out0", IL.PIECES[pid](C, *[C.input(f"in{i}", w) for i, w in enumerate(widths)]))
        return C

    return build


#: rows per inline read of the prover (constant design step 1, ir_lower._Read): its ANDs plus its live value bits' copy rows
READ_ROWS = {"rcp": 27073, "ex2": 27193, "rsq": 36958, "sqrt": 37183}


def test_a_table_read_is_the_provers_inline_read():
    """A read costs the prover's inline read's gates: its ANDs and its copy rows are the prover's rows; in a traced piece it is
    one read item naming its table by SHA-512 (the library's pin)."""
    from verity.ml.circuits import library

    for t, rows in READ_ROWS.items():
        g = BX.table_gates(t)
        assert g["and"] + g["live_value_bits"] == rows and g["label"] == library.LIBRARY, t
    reg, rid = _types(_piece_build("MufuEx2Ftz_v1", [32]))
    (read,) = [n for n in reg.nodes.values() if n.get("reads")]
    (r,) = read["reads"]
    assert (r["sha512"], r["index_bits"], r["value_bits"]) == (library.table("ex2").sha512, 23, 31)
    assert reg.nodes[rid]["and"] >= BX.table_gates("ex2")["and"]


def test_a_wide_gather_is_its_muxes_counted(monkeypatch):
    """A gather wider than GATHER_INLINE is a composed Definition with exactly the counts of the one circuit."""
    V = 5000
    reg = BX.Registry()
    walker = BX.Walker(reg, None, {})
    composed = walker.defs[walker.gather(V)["def"]]
    monkeypatch.setattr(BX, "GATHER_INLINE", V)
    whole = reg.nodes[BX.Walker(reg, None, {}).gather(V)["ref"]]
    assert (composed["and"], composed["xor"], composed["not"]) == (whole["and"], whole["xor"], whole["not"])
    assert composed["and"] == 16 * (V - 1) + reg.nodes[composed["children"][1]["ref"]]["and"]


@pytest.mark.parametrize("pid,widths", [("F32Add_v1", [32, 32]), ("F32Mul_v1", [32, 32]), ("F32Fma_v1", [32, 32, 32])])
def test_a_pieces_types_are_its_circuit(pid, widths):
    """Every type validates; the root, evaluated, is the flat circuit (``type_of`` checks it on random vectors); expanded to
    one type it counts the same gates; ids are full SHA-512 digests."""
    reg, rid = _types(_piece_build(pid, widths))
    flat = CT.expand(reg.types, rid)
    assert not flat.calls and sum(1 for it in flat.items if it[0] == CT.AND) == reg.nodes[rid]["and"]
    assert all(d == t.digest() and len(d) == 128 for d, t in reg.types.items())
    assert reg.evaluated == 1


def test_two_calls_of_a_piece_share_no_gates():
    """Gates are shared only inside a type (constant-api §2.5): two calls of one piece are one type, counted twice."""
    def build():
        C = gf2.Circuit(cse=True)
        a, b = C.input("in0", 32), C.input("in1", 32)
        C.output("out0", IL.PIECES["F32Add_v1"](C, a, b) + IL.PIECES["F32Add_v1"](C, a, b))
        return C

    reg, rid = _types(build)
    (kid,) = reg.nodes[rid]["children"]
    assert kid["count"] == 2 and reg.nodes[rid]["and"] == 2 * reg.nodes[kid["ref"]]["and"] > 0


def test_a_helper_is_one_type_wherever_it_is_called():
    """A type is its circuit, not its caller's: F32Add's and F32Mul's operand unpacking (and leading-zero counts, muxes) are
    the same types."""
    reg = BX.Registry()
    add = reg.type_of(_piece_build("F32Add_v1", [32, 32]), "add", "test")
    mul = reg.type_of(_piece_build("F32Mul_v1", [32, 32]), "mul", "test")

    def below(d, out):
        for c in reg.types[d].calls:
            if c not in out:
                out.add(c)
                below(c, out)
        return out

    shared = below(add, set()) & below(mul, set())
    assert {"unpack", "lzc", "mux"} <= {reg.nodes[d]["function"] for d in shared}


def test_names_are_metadata_and_labels_follow_the_library():
    from verity.ml.circuits import library

    reg, a = _types(_piece_build("F32Add_v1", [32, 32]), "one name")
    _, b = _types(_piece_build("F32Add_v1", [32, 32]), "another")
    assert a == b and reg.nodes[a]["label"] == library.PROGRAM
    step = BX.Walker(reg, None, {}).tc_unit("HopperBF16WgmmaDot16_v1")
    assert reg.nodes[step]["label"] == library.LIBRARY
    assert all(reg.types[c["ref"]].label == library.LIBRARY for c in reg.nodes[step]["children"])


class _Store:
    def __init__(self, defs):
        self.defs, self.ix = defs, {k: {"form": v["form"]} for k, v in defs.items()}

    def get(self, did):
        return self.defs.get(did)


#: ``ArgmaxStep_v1`` as the program graphs store it: carry (best value, best index, lane), step x
ARGMAX_STEP = json.loads(r'''{"form": "composite", "id": "ArgmaxStep_v1", "bindings": {}, "body": {"params": [["carry", ["Tuple", [["Value", 16], ["Value", 32], ["Value", 32]]]], ["x", ["Value", 16]]], "ret_type": ["Tuple", [["Tuple", [["Value", 16], ["Value", 32], ["Value", 32]]], ["Value", 32]]], "nodes": [{"form": "primitive", "fn": "Bf16GtStrict_v1", "args": [{"type": ["Value", 16], "refs": {"affine": ["p", 1, 0, 1]}}, {"type": ["Value", 16], "refs": {"affine": ["p", 0, 0, 1]}}], "out": ["Value", 1]}, {"form": "primitive", "fn": "SelectBf16_v1", "args": [{"type": ["Value", 1], "refs": {"affine": ["n", 0, 0, 1]}}, {"type": ["Value", 16], "refs": {"affine": ["p", 1, 0, 1]}}, {"type": ["Value", 16], "refs": {"affine": ["p", 0, 0, 1]}}], "out": ["Value", 16]}, {"form": "primitive", "fn": "SelectI32_v1", "args": [{"type": ["Value", 1], "refs": {"affine": ["n", 0, 0, 1]}}, {"type": ["Value", 32], "refs": {"affine": ["p", 0, 2, 1]}}, {"type": ["Value", 32], "refs": {"affine": ["p", 0, 1, 1]}}], "out": ["Value", 32]}, {"form": "primitive", "fn": "Const32[0x00000001]_v1", "args": [], "out": ["Value", 32]}, {"form": "primitive", "fn": "I32Add_v1", "args": [{"type": ["Value", 32], "refs": {"affine": ["p", 0, 2, 1]}}, {"type": ["Value", 32], "refs": {"affine": ["n", 3, 0, 1]}}], "out": ["Value", 32]}], "ret": {"concat": [{"concat": [{"affine": ["n", 1, 0, 1]}, {"affine": ["n", 2, 0, 1]}, {"affine": ["n", 4, 0, 1]}]}, {"affine": ["n", 2, 0, 1]}]}}, "fold": []}''')


def _token_select(V):
    carry = ["Tuple", [["Value", 16], ["Value", 32], ["Value", 32]]]
    scan = {"form": "scan", "fn": "ArgmaxStep_v1", "n": V - 1, "n_xs": 1,
            "args": [{"type": carry, "refs": {"concat": [{"affine": ["p", 0, 0, 1]}, {"affine": ["n", 0, 0, 1]}, {"affine": ["n", 1, 0, 1]}]}},
                     {"type": ["Array", V - 1, ["Value", 16]], "refs": {"affine": ["p", 0, 1, V - 1]}}],
            "out": ["Tuple", [carry, ["Array", V - 1, ["Value", 32]]]]}
    consts = [{"form": "primitive", "fn": f"Const32[0x0000000{k}]_v1", "args": [], "out": ["Value", 32]} for k in (0, 1)]
    return {"form": "composite", "id": "TokenSelect_v1", "body": {"params": [["logits", ["Array", V, ["Value", 16]]]], "ret_type": ["Value", 32],
                                                                   "nodes": consts + [scan], "ret": {"affine": ["n", 2, 1, 1]}}}


def test_the_walker_folds_a_scans_lane_counter():
    """``TokenSelect``'s lane index is a constant at every step (the units fold it): the step walked with it a lane counter has
    no I32Add, the generic step does."""
    reg = BX.Registry(Path(tempfile.mkdtemp()))
    W = BX.Walker(reg, _Store({"ArgmaxStep_v1": ARGMAX_STEP, "TokenSelect_v1{V=5}": _token_select(5)}), {})
    (step,) = W.defs[W.walk("TokenSelect_v1{V=5}")]["children"]

    def pieces(did):
        return {reg.nodes[c["ref"]]["name"].split(" ")[0] for c in W.defs[did]["children"] if "ref" in c}

    assert step["count"] == 4 and "lane counter" in W.defs[step["def"]]["name"]
    assert pieces(step["def"]) == {"Bf16GtStrict_v1", "SelectBf16_v1", "SelectI32_v1"}
    assert "I32Add_v1" in pieces(W.walk("ArgmaxStep_v1"))


def test_a_constant_sub_circuit_is_folded():
    """Gemma's ``x + 1.0``: the constant reaches F32Add through a Bf16ToF32, which is itself constant, so F32Add reads it."""
    from types import SimpleNamespace as G

    import verity.ml.circuits.ops.prims as MP

    cvt = next(v for v in vars(MP).values() if getattr(v, "id", None) == "Bf16ToF32_v1")
    gates = [G(index=0, prim=G(family="Input"), operands=()), G(index=1, prim=MP.const(16, 0x3F80), operands=()),
             G(index=2, prim=cvt, operands=(1,)), G(index=3, prim=BX.prims()["F32Add_v1"], operands=(0, 2))]
    assert BX._constants(gates) == {1: 0x3F80, 2: 0x3F800000}


def test_the_modules_file_round_trips():
    reg, rid = _types(_piece_build("F32Add_v1", [32, 32]))
    path = Path(tempfile.mkdtemp()) / "modules.json.gz"
    BX.write_modules(reg, path)
    doc = json.loads(gzip.decompress(path.read_bytes()))
    assert doc["format"] == CT.FORMAT and set(doc["types"]) == set(reg.types)
    assert all(CT.from_content(c).digest() == d for d, c in doc["types"].items())


def test_the_gather_matches_the_ir_primitive():
    from verity.ml.definitions.registry import prims as RP

    rng = random.Random(2)
    for V in (1, 2, 3, 64, 128, 1000):
        vecs = [[rng.choice([rng.randrange(V), V, V + 3, 0xFFFFFFFF, 0x80000000, rng.getrandbits(32)])]
                + [rng.getrandbits(16) for _ in range(V)] for _ in range(300)]
        C = gf2.Circuit(cse=True)
        ins = [C.input(f"i{k}", w) for k, w in enumerate([32] + [16] * V)]
        C.output("o", TP.gather_bf16(C, *ins))
        outs, _ = C.evaluate({f"i{k}": IL.slice_words([v[k] for v in vecs], w) for k, w in enumerate([32] + [16] * V)}, len(vecs))
        assert [int(x) for x in IL.unslice_words(outs["o"], len(vecs))] == [RP.gather_bf16(V).evaluate(*v) for v in vecs]
