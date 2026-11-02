"""`circuit.h1tile` registers `eps1-sm120-h1fx1` only ``if SG is not None`` (``SG`` is `circuit.segment`), so on a
tree without segment the scheme is quietly absent.  That guard must not fail silently: once `circuit.segment` imports, the
one scheme has to be in `compile.SCHEMES`.

This test keys its skip on segment's own importability, not on `h1tile.SG`, so the silent-miss mode the guard could hit --
segment present, yet the register call skipped (`SG` left None, or the register not reached) -- is caught here, not masked by
the same guard that would hide it."""
from __future__ import annotations

import importlib

import pytest

from verity.core.protocols.pouw.circuit import compile as C
from verity.core.protocols.pouw.circuit import h1tile as H


def _segment_present() -> bool:
    try:
        importlib.import_module("verity.core.protocols.pouw.circuit.segment")
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _segment_present(), reason="circuit.segment is not on this tree, so the scheme cannot register")
def test_the_scheme_registers_once_segment_is_present():
    """With segment on the tree, h1tile's guard must have registered the one `eps1-sm120-h1fx1` scheme; a silent miss fails
    here rather than leaving the scheme quietly absent."""
    assert H.SG is not None, "circuit.segment imports but h1tile.SG is None: the registration guard would skip silently"
    assert H.SCHEME in C.SCHEMES, (
        f"{H.SCHEME} is not in compile.SCHEMES though circuit.segment is present: the registration guard failed silently")
    s = C.SCHEMES[H.SCHEME]
    assert s.tile == (32, 32) and set(s.units) == set(H.KINDS)
