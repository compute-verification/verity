"""Lean's `restrict` (`Spot.Restrict`, the ``restrict`` piece), the block's Program as a value, held to
`Flock.Program.decode` of the bytes `verity.circuits.restrict` writes, value against value, every reference one by one: on
the one-circuit chain's first block, an interior block and a unit block, and `test_restrict`'s toys, a root batch among
them taken whole, in part and one member."""

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
PACKAGE = HERE.parents[7] / "verity" / "protocols" / "proof" / "spot" / "verifier" / "lean"
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
def toys():
    """`test_restrict`'s toys by name: each descriptor's text and its blocks' units as Python names them."""
    path = HERE.parents[5] / "circuits" / "test_restrict.py"
    spec = importlib.util.spec_from_file_location("restrict_toys_for_lean", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def toy_cases() -> list[tuple[str, int, list[tuple[int, int]]]]:
    return [(name, i, units) for name, (_f, blocks) in toys().TOYS.items() for i, units in enumerate(blocks)]


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


@pytest.mark.parametrize("name, i, ranges", toy_cases(), ids=lambda x: str(x))
def test_lean_restrict_is_pythons_on_the_toys(pieces, name, i, ranges):
    T = toys()
    _p, c, bits = T.toy(name)
    py = R.restrict(c, T.TOY_PARTITION, ranges, bits).descriptor.decode()
    units = [u for lo, hi in ranges for u in range(lo, hi)]
    kind, body = pieces.call("restrict", {"descriptor": c.decode(), "units": units, "python": py})
    assert kind == "output", body
    out = json.loads(body)
    assert out["agree"] is True and out["at"] is None, out


@pytest.mark.parametrize("at", [0, 1])
def test_lean_restrict_is_pythons_on_the_products_blocks(pieces, at):
    from verity.circuits.ops import product as PR
    p = PR.product(k=16, s=2, units=4, block=2, k_log=22, g=2)
    lo, hi = p.blocks[at]
    py = R.restrict(p.description, p.partition, [(lo, hi)]).descriptor.decode()
    kind, body = pieces.call("restrict", {"descriptor": p.description.decode(), "units": list(range(lo, hi)), "python": py})
    assert kind == "output", body
    out = json.loads(body)
    assert out["agree"] is True and out["at"] is None, out


def test_a_repeated_unit_is_refused(pieces):
    text, _ = chain()
    kind, body = pieces.call("restrict", {"descriptor": text, "units": [6, 6], "python": python((6,))})
    assert kind == "reject" and "a block names each of its units once" in body, body


def test_the_blocks_program_carries_no_digest(pieces):
    """`Prog.digest` refuses an empty digest, so nothing compares the block's Program with a record by one."""
    text, _ = chain()
    out = json.loads(pieces.call("restrict", {"descriptor": text, "units": [0, 1], "python": python((0, 1))})[1])
    assert out["digest_bytes"] == 0


def test_what_restrict_reads_is_the_blocks_not_cs(pieces):
    """What `front`, `members` and `restrict` read (the piece's ``cost``) on the chain's member blocks (a batch held whole)
    and unit blocks (a batch held in part), printed for the count beside C's description and gate count: equal for
    blocks of one shape wherever they sit in C."""
    text, _ = chain()
    cost = {}
    for units in ((0, 1), (4, 5), (6,), (1,)):
        cost[units] = json.loads(pieces.call("restrict", {"descriptor": text, "units": list(units), "python": python(units)})[1])["cost"]
        print("restrict cost", units, json.dumps(cost[units], sort_keys=True))
    assert cost[(0, 1)] == cost[(4, 5)] and cost[(6,)] == cost[(1,)], cost


def test_another_blocks_program_differs(pieces):
    """The chain's member blocks share one Program, so a member is held against a unit block's."""
    text, _ = chain()
    kind, body = pieces.call("restrict", {"descriptor": text, "units": [0, 1], "python": python((6,))})
    assert kind == "output", body
    out = json.loads(body)
    assert out["agree"] is False and out["at"], out


@pytest.mark.parametrize("units", list(BLOCKS.values()), ids=list(BLOCKS))
def test_the_block_lowers(pieces, units):
    """`Flock.Lower.lower` takes the block's Program with its input nodes as parameters (`Spot.Restrict.ported`)."""
    text, _ = chain()
    out = json.loads(pieces.call("restrict", {"descriptor": text, "units": list(units), "python": python(units)})[1])
    assert out["lowers"] is None, out["lowers"]
