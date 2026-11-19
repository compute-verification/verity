"""Lean's C_S (`Spot.OneStageZk.restrict`, `spot-one-stage-zk restrict`) is the door's Python circuit file
(`one_circuit.lower` of `one_circuit.restrict`) byte for byte, on every block of the one-circuit chain."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path

import pytest

from verity.circuits import partition_object as PO
from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as OC
from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF

HERE = Path(__file__).resolve()
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@cache
def chain():
    path = HERE.parents[1] / "test_one_circuit_chain.py"
    spec = importlib.util.spec_from_file_location("one_circuit_chain_for_block_circuit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.registered()


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
    door = OC.lower(OC.restrict(c, part, units), part, PF.lower_executable)
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
