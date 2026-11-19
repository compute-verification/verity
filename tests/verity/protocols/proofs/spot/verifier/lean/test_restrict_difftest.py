"""Lean's `restrict` (`Spot.Restrict`, the ``restrict`` piece), the block's Program as a value, held to
`Flock.Program.decode` of the bytes `verity.circuits.restrict` writes, value against value, every reference one by one: on
the one-circuit chain's first block, an interior block and a unit block."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from functools import cache
from pathlib import Path

import pytest

import verity_difftest as D
from verity.circuits import restrict as R
from verity.circuits.codec import descriptor_chunks

HERE = Path(__file__).resolve()
PACKAGE = HERE.parents[7] / "verity" / "protocols" / "proofs" / "spot" / "verifier" / "lean"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@cache
def chain():
    """The one-circuit chain's descriptor text and its template query."""
    path = HERE.parents[1] / "test_one_circuit_chain.py"
    spec = importlib.util.spec_from_file_location("one_circuit_chain_for_restrict", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    program, _query = mod.circuit().program()
    query = {"name": "Q_template_instances", "version": 0, "params": {"templates": [mod.TI.GID]}}
    return b"".join(descriptor_chunks(program)).decode(), query


@cache
def python(units: tuple[int, ...]) -> str:
    """Python's `restrict` bytes for the block of ``units``, consecutive units of the chain."""
    text, query = chain()
    c = text.encode()
    return R.restrict(c, {"query": query}, [(units[0], units[-1] + 1)], R.Bits(R.program(c))).descriptor.decode()


def pieces_port():
    return D.Port("spot-difftest", "restrict", lambda _j: None, None, package=PACKAGE, of_record="lean")


pieces = D.server_fixture(pieces_port())

#: the chain's first block, an interior block (each the window's member of two units) and a unit block
BLOCKS = {"first": (0, 1), "interior": (4, 5), "unit": (6,)}


@pytest.mark.parametrize("name", BLOCKS)
def test_lean_restrict_is_pythons(pieces, name):
    text, _ = chain()
    units = BLOCKS[name]
    kind, body = pieces.call("restrict", {"descriptor": text, "units": list(units), "python": python(units)})
    assert kind == "output", body
    out = json.loads(body)
    assert out["agree"] is True and out["at"] is None, out
    assert out["root_nodes"] >= 2 and out["definitions"] > 1, out


def test_another_blocks_program_differs(pieces):
    """The chain's member blocks share one Program, so a member is held against a unit block's."""
    text, _ = chain()
    kind, body = pieces.call("restrict", {"descriptor": text, "units": [0, 1], "python": python((6,))})
    assert kind == "output", body
    out = json.loads(body)
    assert out["agree"] is False and out["at"], out
