"""A stand-in for the `tokenizers` library, which a CPU node does not install: a `Tokenizer` of one id a character that records what
each call was asked for.  `install` puts it where `from tokenizers import Tokenizer` finds it, for one test."""
from __future__ import annotations

import sys
import types

import pytest


class Tokenizer:
    """An id a character; `asked` holds each call's name with what it was told of special tokens."""

    asked: set[tuple[str, bool]] = set()

    def __init__(self, path: str) -> None:
        self.path = path

    @classmethod
    def from_file(cls, path: str) -> Tokenizer:
        return cls(path)

    def encode(self, text: str, add_special_tokens: bool) -> types.SimpleNamespace:
        self.asked.add(("encode", add_special_tokens))
        return types.SimpleNamespace(ids=[ord(c) for c in text])

    def decode(self, ids: list[int], skip_special_tokens: bool) -> str:
        self.asked.add(("decode", skip_special_tokens))
        return "".join(map(chr, ids))


def install(monkeypatch: pytest.MonkeyPatch) -> type[Tokenizer]:
    """The stand-in as the module `tokenizers` until the test ends, with nothing asked yet."""
    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=Tokenizer))
    monkeypatch.setattr(Tokenizer, "asked", set())
    return Tokenizer
