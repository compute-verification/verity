"""The checks the Boolean Definitions' test modules share: a specialization against its pinned counts and digest, a
Definition's traced ANDs against its builder's live ANDs, and a specialization against its word primitive."""
from __future__ import annotations

import json

import numpy as np

from verity.circuits import boolean as BL
from verity.circuits import codec
from verity.circuits import evaluation as E
from verity.circuits.boolean import trace as TR
from verity.circuits.boolean.forms import Circuit
from verity.circuits.defs import REGISTRY
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation.reference import standalone_call


def assert_pinned(target, pins: dict) -> None:
    """``target`` is Boolean, its AND, XOR and NOT counts and its standalone Program's digest are ``pins[target.id]``, and the
    Program's canonical encoding decodes to the same gates and digest."""
    ands, xors, nots, digest = pins[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    desc = codec.encode_program(prog)
    assert codec.program_digest(desc) == digest
    D = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
    assert D.gates == prog.gates and codec.program_digest(codec.encode_program(D)) == digest


def assert_live_ands(d, pins: dict) -> None:
    """``d``'s traced AND count is the live ANDs of its builder on a CSE circuit, and its pin's."""
    T = d.traced()
    params = [t for _, t in d.signature(None)[0]]
    C = Circuit(cse=True)
    ins = TR._elements_in(C, params)
    C.output("out", TR._flat_out(d.build(C, *ins), d.signature(None)[1]))
    assert T.counts["and"] == C.counts()["ands"] == pins[f"{d.name}_v{d.version}"][0]


def assert_agrees(target, inputs) -> None:
    """``target``'s Boolean evaluation of ``inputs`` is its word primitive's, instance by instance."""
    word = TR.word_of(target)
    got = BS.evaluate_words(target, *inputs).astype(np.uint64)
    ref = E.evaluate_batch(word, *inputs).astype(np.uint64)
    assert got.shape == ref.shape
    bad = np.flatnonzero(got.reshape(len(got), -1) != ref.reshape(len(ref), -1))
    assert not bad.size, f"{target.id}: {bad.size} instances differ from {word.id}, first {bad[:5]}"
