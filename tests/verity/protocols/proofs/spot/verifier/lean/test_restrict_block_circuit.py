"""Lean's C_S (`Spot.OneStageZk.restrict`, `spot-one-stage-zk restrict`) is the door's Python circuit file
(`one_circuit.lower` of `one_circuit.restrict`) byte for byte, on every block of the one-circuit chain."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path

import pytest

from verity.circuits import partition_object as PO
from verity.circuits.codec import descriptor_id
from verity.commitments.periods import flock_inputs as FI
from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as OC
from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF

HERE = Path(__file__).resolve()
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@cache
def chain_module():
    path = HERE.parents[1] / "test_one_circuit_chain.py"
    spec = importlib.util.spec_from_file_location("one_circuit_chain_for_block_circuit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def chain():
    return chain_module().registered()


@cache
def sliced():
    """`test_interface.sliced`: six ``Gemm_v2{K=16, N=8}`` slices, then three ``Gemm_v2{K=16, N=16}`` each reading its
    ``x`` cut in two, a row from each slice: C, the partition, and the window's members."""
    TI, I = chain_module().TI, chain_module().I
    win = I.Window(TI.sliced(), TI.RUN, TI.SAMPLING_DESIGN, 2)
    program, _query = TI.sliced().program()
    return b"".join(chain_module().descriptor_chunks(program)), win.window.partition, win.window.members


def lean(c: bytes, partition: dict, first: int) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as d:
        cp, pp, out = Path(d, "C.json"), Path(d, "partition.json"), Path(d, "circuit.txt")
        cp.write_bytes(c)
        pp.write_bytes(PO.canonical(partition))
        p = subprocess.run([str(PF.v2_executable()), "restrict", str(cp), str(pp), str(first), str(out)],
                           capture_output=True, text=True)
        p.stdout = out.read_bytes() if p.returncode == 0 else p.stderr
        return p


@pytest.mark.parametrize("k", range(4))
def test_lean_c_s_is_the_doors(k):
    r = chain()
    c, part = r["C"], r["partition"]
    blocks = PO.blocks(part)
    if k >= len(blocks):
        pytest.skip(f"the chain's partition has {len(blocks)} blocks")
    units = OC.block(c, part, k)
    door = OC.lower(OC.restrict(c, part, units), part, PF.lower_executable, cut=OC.cuts(c, part))
    got = lean(c, part, units[0][0])
    assert got.returncode == 0, got.stdout
    assert got.stdout == door


def test_a_unit_that_starts_no_block_is_refused():
    r = chain()
    c, part = r["C"], r["partition"]
    lo, hi = PO.blocks(part)[0]
    if hi - lo < 2:
        pytest.skip("the chain's first block is one unit")
    got = lean(c, part, lo + 1)
    assert got.returncode == 1 and f"no block of the partition starts at unit {lo + 1}" in got.stdout


def _e(n: int) -> dict:
    """`Spot.OneStageZk.E3` / `E2047`: E(x : Array n (Value 1), y : Value 1) = (x, Not y), one instance."""
    v1, a = ["Value", 1], ["Array", n, ["Value", 1]]
    r = ["Tuple", [a, v1]]
    return {"annotations": {}, "boolean": {}, "callees": ["Input1_v1", "Not_v1", "E_v1"], "definitions": {
        "E_v1": {"bindings": {}, "form": "composite", "id": "E_v1", "body": {
            "nodes": [{"args": [{"refs": {"affine": ["p", 1, 0, 1]}, "type": 0}], "fn": 1, "form": "primitive", "out": 0}],
            "params": [["x", a], ["y", v1]], "ret": {"concat": [{"affine": ["p", 0, 0, n]}, {"affine": ["n", 0, 0, 1]}]},
            "ret_type": r}},
        "Input1_v1": {"form": "primitive", "id": "Input1_v1", "params": [], "ret": v1},
        "Not_v1": {"form": "primitive", "id": "Not_v1", "params": [["a", v1]], "ret": v1},
        "Root[EProg_v1]_v1": {"bindings": {}, "form": "composite", "id": "Root[EProg_v1]_v1", "body": {
            "nodes": [{"args": [], "axes": [], "fn": 0, "form": "batch", "n": n, "out": 1},
                      {"args": [], "fn": 0, "form": "primitive", "out": 0},
                      {"args": [{"refs": {"affine": ["n", 0, 0, n]}, "type": 1}, {"refs": {"affine": ["n", 1, 0, 1]}, "type": 0}],
                       "fn": 2, "form": "call", "out": 2}],
            "params": [], "ret": {"affine": ["n", 2, 0, n + 1]}, "ret_type": r}}},
        "program_fn": "EProg_v1", "root": "Root[EProg_v1]_v1", "schema": "verity-ir/descriptor/v2", "types": [v1, a, r]}


@pytest.mark.parametrize("n", [3, 2047])
def test_a_row_not_whole_words_is_refused_by_both(n):
    """A parameter not whole u16 words: the one-stage path lays each row from a word while the lowering lays the leaves
    one after another, so `y` would be read from x's padding. Lean's C_S and the door's refuse it with one text."""
    desc = _e(n)
    c = json.dumps(desc, sort_keys=True, separators=(",", ":")).encode()
    part = PO.build(desc, PO.template_instance_query("E_v1"), [[(0, 1)]])
    why = f"lower: C_S's template E_v1: parameter x of {n} bits {OC.NOT_WHOLE}"
    with pytest.raises(OC.Refused, match=re.escape(why)):
        OC.lower(OC.restrict(c, part, OC.block(c, part, 0)), part, PF.lower_executable, cut=OC.cuts(c, part))
    got = lean(c, part, 0)
    assert got.returncode == 1 and why in got.stdout, got.stdout


def _desc(key: str, n: int, nodes: list, ret: dict, ret_type: list, types: list, prims: dict) -> dict:
    """`Spot.OneStageZk.X16` / `F48`: template ``key`` on x : Array n (Value 1), one instance over n inputs."""
    v1, a = ["Value", 1], ["Array", n, ["Value", 1]]
    callees = ["Input1_v1", *prims, key]
    root = f"Root[{key[:-3]}Prog_v1]_v1"
    return {"annotations": {}, "boolean": {}, "callees": callees, "definitions": {
        key: {"bindings": {}, "form": "composite", "id": key, "body": {
            "nodes": nodes, "params": [["x", a]], "ret": ret, "ret_type": ret_type}},
        "Input1_v1": {"form": "primitive", "id": "Input1_v1", "params": [], "ret": v1},
        **{p: {"form": "primitive", "id": p, "params": ps, "ret": v1} for p, ps in prims.items()},
        root: {"bindings": {}, "form": "composite", "id": root, "body": {
            "nodes": [{"args": [], "axes": [], "fn": 0, "form": "batch", "n": n, "out": 1},
                      {"args": [{"refs": {"affine": ["n", 0, 0, n]}, "type": 1}], "fn": len(callees) - 1, "form": "call",
                       "out": types.index(ret_type)}],
            "params": [], "ret": {"affine": ["n", 1, 0, 1 if ret_type == v1 else n]}, "ret_type": ret_type}}},
        "program_fn": f"{key[:-3]}Prog_v1", "root": root, "schema": "verity-ir/descriptor/v2", "types": types}


def _x16() -> dict:
    """X(x : Array 16 (Value 1)) = Xor2(x[0], x[0]): a 1-bit return field."""
    v1, a2 = ["Value", 1], ["Array", 2, ["Value", 1]]
    x0 = {"affine": ["p", 0, 0, 1]}
    node = {"args": [{"refs": {"concat": [x0, x0]}, "type": 2}], "fn": 1, "form": "primitive", "out": 0}
    return _desc("X_v1", 16, [node], {"affine": ["n", 0, 0, 1]}, v1, [v1, ["Array", 16, v1], a2], {"Xor2_v1": [["x", a2]]})


def _f48() -> dict:
    """F(x : Array 48 (Value 1)) = x[0:47] ++ [Not x[47]]: rows of 24 bits when x is cut in two."""
    v1, a = ["Value", 1], ["Array", 48, ["Value", 1]]
    node = {"args": [{"refs": {"affine": ["p", 0, 47, 1]}, "type": 0}], "fn": 1, "form": "primitive", "out": 0}
    ret = {"concat": [{"affine": ["p", 0, 0, 47]}, {"affine": ["n", 0, 0, 1]}]}
    return _desc("F_v1", 48, [node], ret, a, [v1, a], {"Not_v1": [["a", v1]]})


def test_the_lean_guards_read_these():
    text = HERE.parents[7].joinpath("verity/protocols/proofs/spot/verifier/lean/Spot/OneStageZkStatement.lean").read_text()
    for key, desc in (("X16", _x16()), ("F48", _f48())):
        assert f'def {key} : String := r#"{json.dumps(desc, sort_keys=True, separators=(",", ":"))}"#' in text


def test_a_one_bit_return_is_refused_by_both():
    """Xor(x, x) with a 1-bit output: its word's other 15 bits would be columns no row reads."""
    desc = _x16()
    c = json.dumps(desc, sort_keys=True, separators=(",", ":")).encode()
    part = PO.build(desc, PO.template_instance_query("X_v1"), [[(0, 1)]])
    why = f"lower: C_S's template X_v1: return field o0 of 1 bits {OC.NOT_WHOLE}"
    with pytest.raises(OC.Refused, match=re.escape(why)):
        OC.lower(OC.restrict(c, part, OC.block(c, part, 0)), part, PF.lower_executable, cut=OC.cuts(c, part))
    got = lean(c, part, 0)
    assert got.returncode == 1 and why in got.stdout, got.stdout


def test_a_cut_row_not_whole_words_is_refused():
    """x of 48 bits, whole, cut in two: rows of 24 bits (Lean's twin: `Spot.OneStageZk.F48`'s guard)."""
    desc = _f48()
    c = json.dumps(desc, sort_keys=True, separators=(",", ":")).encode()
    part = PO.build(desc, PO.template_instance_query("F_v1"), [[(0, 1)]])
    why = f"lower: C_S's template F_v1: input row x#0 of 24 bits {OC.NOT_WHOLE}"
    with pytest.raises(OC.Refused, match=re.escape(why)):
        OC.lower(OC.restrict(c, part, OC.block(c, part, 0)), part, PF.lower_executable, cut={"F_v1": {"x": 2}})


def _ports(text: bytes) -> list[list]:
    meta = json.loads(text.split(b"\n", 2)[1][len(b"META "):])
    return [list(p[:2]) for p in meta["ports"]], meta


@pytest.mark.parametrize("k", range(5))
def test_a_cut_blocks_rows_are_the_periods_rows(k):
    """On `sliced`, whose ``Gemm_v2{K=16, N=16}`` reads ``x`` cut in two: Lean's C_S is the door's byte for byte, and its
    ports are the period member's rows exactly, ``x#0, x#1`` cut by in(S) (`ioCuts`) and each output row exactly its words
    (`Flock.Compose.outPort`); the door's file with ``x`` one whole row, or a member whose output rows are padded to
    whole SHA-512 blocks, is refused by name (`flock_inputs.ports_form`)."""
    c, part, members = sliced()
    blocks = PO.blocks(part)
    units = OC.block(c, part, k)
    member = next(m for m in members if m.base == units[0][0])
    cut = OC.cuts(c, part)
    door = OC.lower(OC.restrict(c, part, units), part, PF.lower_executable, cut=cut)
    got = lean(c, part, units[0][0])
    assert got.returncode == 0, got.stdout
    assert got.stdout == door
    ports, meta = _ports(door)
    mem = {"ports": [list(p) for p in member.all_rows], "input_ports": len(member.rows)}
    assert ports == mem["ports"] and FI.ports_form(meta, mem) is None
    gemm = descriptor_id(chain_module().TI.GEMM)
    assert cut == {gemm: {"x": 2}} and len(blocks) == 5
    if member.template != gemm:
        return
    assert [p for p, _ in member.rows][:2] == ["x#0", "x#1"]
    whole, whole_meta = _ports(OC.lower(OC.restrict(c, part, units), part, PF.lower_executable))
    assert whole[0][0] == "x" and FI.ports_form(whole_meta, mem).startswith(FI.ROWS_CUT)
    padded = {**mem, "ports": mem["ports"][:mem["input_ports"]] + [[p, -(-w // 64) * 64] for p, w in member.out_rows]}
    assert FI.ports_form(meta, padded).startswith(FI.ROWS_PADDED)
