"""`one_stage.members`' verdict trees (`OneStageZk.verdictTreesHeld`) on a toy opening: an `EmbeddingRead` v3 whose
verdict is its output field 1, over a toy `FrameV3Read` v1. The record's root of ``<template>/o1`` is accepted where
every row is the word 1 under the public salts, and refused by name where a row is 0, where the rows are salted from the
operating system, and where C holds an opening on no entry of the list; the writer's root is the verifier's."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest

from verity.circuits import population as PN
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.refs import tuple_of
from verity.circuits.types import Tuple, Value
from verity.commitments import registered_value as RV
from verity.commitments.periods import rows as R
from verity.commitments.merkle import _LEAF, _uint, frame_hash, root_from_subtrees
from verity.protocols.proof.spot.verifier.one_stage import members as M, partition as P

READ = PrimitiveDefinition("FrameV3Read", 1, (("a", Value(16)),), Value(16), lambda a: a, register=False)
EMBED = bind(CompositeDefinition("EmbeddingRead", 3, (), lambda S: ((("t", Value(16)),), Tuple(Value(16), Value(16))),
                                 lambda B, S, t: tuple_of(B.call(READ, t), B.call(READ, t)), register=False))
OPENS = bind(CompositeDefinition("VerdictTreesOpens", 1, (), lambda S: ((("t", Value(16)),), Value(16)),
                                 lambda B, S, t: B.call(READ, t), register=False))
DOMAIN = bytes(range(64))
MEMBERS = {"e/c0": {"units": [[0, 3]]}}
PINNED = ("25cabd2e7f8d2d0a8ffd61d69d0fa58ac19d876606f65b72832f5a6fcc23cf3d"
          "51c7234eb8f93c1f41f3bc77650fc0a0622895ca1fb36bb5c5ef6cfe60b58a84")
LEAN = (Path(__file__).resolve().parents[6] / "verity/protocols/proof/spot/verifier/lean/Spot/OneStageZk.lean")


def window(*defs):
    program = PN.mixed_population_program([(d, n) for d, n in defs])
    return program, P.template_instances_query([d.id for d, _ in defs])


def record(root: str) -> dict:
    return {"roots": [{"port": "e/o1", "root": root, "leaves": 3, "domain": DOMAIN.hex()}]}


def os_salted(domain_id: bytes, rows) -> bytes:
    bcs = RV.commit_strings(rows, [os.urandom(192) for _ in rows])
    return root_from_subtrees(domain_id, len(bcs), {
        (0, i): frame_hash(_LEAF, domain_id, _uint(i), _uint(i), RV.SCHEMA.encode(),
                           RV._INST.tree_leaf(RV._INST.default_key, bc)) for i, bc in enumerate(bcs)})


def test_an_honest_verdict_tree_is_held():
    program, query = window((EMBED, 3))
    assert M._verdict_trees(program, query, MEMBERS) == [("e/o1", 3)]
    root = RV.public_root(P.program_sha512(program), "e/o1", DOMAIN, [[1]] * 3).hex()
    M._verdict_trees_held(program, query, MEMBERS, record(root))


@pytest.mark.parametrize("root", [
    lambda prog: RV.public_root(prog, "e/o1", DOMAIN, [[1], [0], [1]]),
    lambda prog: os_salted(DOMAIN, [[1]] * 3),
], ids=["a-zero-row", "os-salts"])
def test_a_verdict_tree_not_the_verifiers_is_refused(root):
    program, query = window((EMBED, 3))
    with pytest.raises(M._Refused, match=re.escape(M.verdict_tree_why("e/o1"))):
        M._verdict_trees_held(program, query, MEMBERS, record(root(P.program_sha512(program)).hex()))


def test_an_unlisted_opening_is_refused():
    program, query = window((EMBED, 3), (OPENS, 1))
    with pytest.raises(M._Refused, match=re.escape(M.unlisted_opening_why(OPENS.id))):
        M._verdict_trees(program, query, MEMBERS)


def test_the_pinned_root_is_leans():
    """The root `Spot.Test.VerdictTrees`' `#guard` pins for `OneStageZk.verdictRoot`."""
    assert RV.public_root("ab" * 64, "EmbeddingRead/o1", bytes(64), [[1]] * 3).hex() == PINNED


def test_the_lists_are_leans():
    text = LEAN.read_text()
    verdicts = re.search(r"^def VERDICTS : List \(String × Nat × Nat\) := \[(.*)\]$", text, re.M).group(1)
    openers = re.search(r"^def OPENERS : List String := \[(.*)\]$", text, re.M).group(1)
    assert [(n, int(v), int(j)) for n, v, j in re.findall(r'\("(\w+)", (\d+), (\d+)\)', verdicts)] == list(M.VERDICTS)
    assert re.findall(r'"(\w+)"', openers) == list(M.OPENERS)


@pytest.mark.parametrize("into", [False, True], ids=["in-memory", "into-a-directory"])
def test_the_writers_verdict_tree_is_the_verifiers(into, tmp_path):
    """`rows.commit` salts a verdict tree's rows with `Binding.verdict_salts`, so an honest window's record holds."""
    program, query = window((EMBED, 3))
    b = R.Binding(program, query, {"law": "stratified", "k": 2}, "ab" * 64,
                  (R.Member(EMBED.id, (("t", 64),), (("o0", 1), ("o1", 1)), 3),))
    ones = np.ones((3, 1), dtype=np.uint16)
    c = R.commit(b, [{"rows": {"t": np.zeros((3, 64), dtype=np.uint16)}, "outs": {"o0": ones, "o1": ones}}],
                 into=tmp_path / "w" if into else None)
    tree = f"{EMBED.id}/o1"
    rec = {"roots": [{"port": tree, "root": c.roots[tree].hex(), "leaves": 3, "domain": c.domains[tree].domain_id.hex()}]}
    M._verdict_trees_held(program, query, {f"{EMBED.id}/c0": {"units": [[0, 3]]}}, rec)
