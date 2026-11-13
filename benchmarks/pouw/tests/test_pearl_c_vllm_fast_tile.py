"""Which BLAKE3 `fast_tile.install` puts in place (``benchmarks/pouw/pearl_c_vllm/fast_tile.py``): the `blake3` package's
digests are the tree's, the tree's stays where the package is absent, and a package that is there and does not import
raises."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys

import fast_tile as FT
import pytest

from verity.commitments.crypto import blake3 as pure
from verity.protocols.pouw.prover.schemes import pearl_c as C
from verity.protocols.pouw.prover.schemes.pearl_c_device import DEVICES

KEY = bytes(range(32))
#: the empty message, one byte, one block, one chunk and a byte more, and five chunks
MESSAGES = (b"", b"a", bytes(range(64)), bytes(range(256)) * 4 + b"a", bytes(range(256)) * 20)


def test_the_package_gives_the_trees_digests():
    """The two functions `install` swaps in hash and key as `verity.commitments.crypto.blake3` does."""
    pytest.importorskip("blake3")
    plain, keyed = FT._native_blake3()
    for message in MESSAGES:
        assert plain(message) == pure.blake3(message) and keyed(KEY, message) == pure.keyed(KEY, message)


def test_without_the_package_install_leaves_the_trees_blake3(monkeypatch):
    """`None` in ``sys.modules`` is Python's own mark for a module that is not to be found: `install` then swaps no
    BLAKE3 and still puts the batch forms in place."""
    monkeypatch.setitem(sys.modules, "blake3", None)
    assert FT._native_blake3() is None
    before = (pure.blake3, pure.keyed, C.blake3)
    undo = FT.install(C.PearlC(forming="v1", device=DEVICES["h100"], hashing="h3"))
    try:
        assert (pure.blake3, pure.keyed, C.blake3) == before and C.form_v1 is FT.form_v1
    finally:
        undo()
    assert (pure.blake3, pure.keyed, C.blake3) == before and C.form_v1 is not FT.form_v1


def test_a_package_that_does_not_import_raises(monkeypatch):
    """A package that is found and fails to import is an error, not an absent package: the slower BLAKE3 never takes its
    place in silence."""
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "blake3", None)
        patch.setattr(importlib.util, "find_spec", lambda name, package=None: importlib.machinery.ModuleSpec(name, None))
        with pytest.raises(ImportError):
            FT._native_blake3()
