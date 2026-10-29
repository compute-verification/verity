"""`program_digest` streams a program's descriptor with the cyclic collector off, and leaves the collector as it found it: on
after a digest that returns or raises, off after one its caller ran with it off."""
from __future__ import annotations

import gc
import hashlib

import pytest

from verity.ml.circuits import codec as C


@pytest.fixture
def collector():
    was = gc.isenabled()
    gc.enable()
    yield
    (gc.enable if was else gc.disable)()


def test_a_digest_streams_with_the_collector_off_and_leaves_it_as_it_was(collector, monkeypatch) -> None:
    seen = []

    def chunks(program):
        seen.append(gc.isenabled())
        yield b'{"definitions":{}'
        yield b"}"

    monkeypatch.setattr(C, "descriptor_chunks", chunks)
    want = hashlib.sha512(b'{"definitions":{}}').hexdigest()
    assert (C.program_digest(object()), seen, gc.isenabled()) == (want, [False], True)
    gc.disable()
    assert (C.program_digest(object()), seen, gc.isenabled()) == (want, [False, False], False)


def test_a_digest_that_raises_turns_the_collector_back_on(collector, monkeypatch) -> None:
    def chunks(program):
        yield b"{"
        raise ValueError("outside the canonical JSON domain")

    monkeypatch.setattr(C, "descriptor_chunks", chunks)
    with pytest.raises(ValueError, match="canonical JSON domain"):
        C.program_digest(object())
    assert gc.isenabled()
