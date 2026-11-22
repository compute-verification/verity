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
from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC
from verity.protocols.proof.spot.verifier.one_stage import proofs as PF

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
    door = OC.lower(OC.restrict(c, part, units), part, PF.lower_executable, cut=OC.cuts(c, part),
                    formed=OC.formed_in(c, part, units))
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


@cache
def door_form():
    path = HERE.parents[1] / "test_door_form.py"
    spec = importlib.util.spec_from_file_location("door_form_for_block_circuit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _through_c():
    """`Spot/Admitted.lean`'s refusals of `admitted` on a C whose root calls the query's one template, at no
    cut: the cases C alone gives, through Lean's executable and the door alike."""
    df = door_form()
    return [g for g in df.guards() if not g[3] and g[2] in json.loads(g[1])["definitions"]]


@pytest.mark.parametrize("case", _through_c(), ids=lambda g: f"{g[4]}:{g[5][:48]}")
def test_a_block_not_admitted_is_refused_by_both_with_one_text(case):
    """Lean's two faults among them: a row not whole u16 words (`E3`, `E2047`), whose leaves after it the lowering would
    read from its padding, and a word parameter the lift leaves as it is (`C32`)."""
    _how, d, t, _cs, clause, detail = case
    c, desc = d.encode(), json.loads(d)
    part = PO.build(desc, PO.template_instance_query(t), [[(0, 1)]])
    why = f"{OC.NOT_ADMITTED}, {clause}: {detail}"
    with pytest.raises(OC.Refused, match=re.escape(why)):
        OC.lower(OC.restrict(c, part, OC.block(c, part, 0)), part, PF.lower_executable, cut=OC.cuts(c, part),
                 formed=OC.formed_in(c, part, OC.block(c, part, 0)))
    got = lean(c, part, 0)
    assert got.returncode == 1 and why in got.stdout, got.stdout


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
    door = OC.lower(OC.restrict(c, part, units), part, PF.lower_executable, cut=cut, formed=OC.formed_in(c, part, units))
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
    whole, whole_meta = _ports(OC.lower(OC.restrict(c, part, units), part, PF.lower_executable,
                                        formed=OC.formed_in(c, part, units)))
    assert whole[0][0] == "x" and FI.ports_form(whole_meta, mem).startswith(FI.ROWS_CUT)
    padded = {**mem, "ports": mem["ports"][:mem["input_ports"]] + [[p, -(-w // 64) * 64] for p, w in member.out_rows]}
    assert FI.ports_form(meta, padded).startswith(FI.ROWS_PADDED)
