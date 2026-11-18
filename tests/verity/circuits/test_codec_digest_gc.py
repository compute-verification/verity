"""`program_digest` computes a program's definition digests with the cyclic collector off, and leaves the collector as it found
it: on after a digest that returns or raises, off after one its caller ran with it off."""
from __future__ import annotations

import gc

import pytest

from verity.circuits import codec as C

DESC = {"schema": C.SCHEMA, "program_fn": "F_v1", "inputs": [], "root": "R", "boolean": {}, "definitions": {}}


def test_a_digest_composes_with_the_collector_off_and_leaves_it_as_it_was(collector, monkeypatch) -> None:
    seen = []

    def digests(program):
        seen.append(gc.isenabled())
        return {"R": "00"}

    monkeypatch.setattr(C, "definition_digests", digests)
    want = C._sha512_json({"schema": C.SCHEMA, "program_fn": "F_v1", "inputs": [], "root": "00", "functions": {}, "boolean": {}})
    assert (C.program_digest(DESC), seen, gc.isenabled()) == (want, [False], True)
    gc.disable()
    assert (C.program_digest(DESC), seen, gc.isenabled()) == (want, [False, False], False)


def test_a_digest_that_raises_turns_the_collector_back_on(collector, monkeypatch) -> None:
    def digests(program):
        raise ValueError("outside the canonical JSON domain")

    monkeypatch.setattr(C, "definition_digests", digests)
    with pytest.raises(ValueError, match="canonical JSON domain"):
        C.program_digest(DESC)
    assert gc.isenabled()
