"""Leaf names -> scheme objects, and the relation-string convention ``<relation>+<leaf tag>``.

``poseidon2`` (aliases ``hash``, ``poseidon2-babybear-w24``) is today's built-in sponge of ``hashchain.py`` and is
represented by ``None`` here: a hashed relation / v5 statement WITHOUT a suffix is the Poseidon2 one, byte for byte as
before.  ``ajtai-n<N>`` (default ``ajtai`` = ``ajtai-n256``) resolves to :class:`leaf.ajtai.AjtaiLeaf`.
"""
from __future__ import annotations

import re

__all__ = ["POSEIDON2_ALIASES", "leaf_for", "leaf_tag", "split_relation", "join_relation"]

POSEIDON2_ALIASES = ("poseidon2", "hash", "poseidon2-babybear-w24", "")
_AJTAI = re.compile(r"^ajtai(?:-n(\d+))?$")


def leaf_for(name: str | None, m_col: int = 256):
    """The scheme object of ``name`` (``None`` for Poseidon2, the built-in)."""
    if name is None or name in POSEIDON2_ALIASES:
        return None
    m = _AJTAI.match(name)
    if m:
        from .ajtai import DEFAULT_N, AjtaiLeaf
        return AjtaiLeaf(int(m.group(1)) if m.group(1) else DEFAULT_N, m_col=m_col)
    raise ValueError(f"unknown leaf scheme {name!r} (known: poseidon2 | hash, ajtai[-n<N>])")


def leaf_tag(leaf) -> str:
    """The suffix a relation string carries for ``leaf`` ('' for Poseidon2)."""
    return "" if leaf is None else leaf.tag


def split_relation(relation: str) -> tuple[str, str | None]:
    """``'fp8-ada+ajtai-n256' -> ('fp8-ada', 'ajtai-n256')``; ``'fp8-ada' -> ('fp8-ada', None)`` (Poseidon2 / bare)."""
    base, sep, suffix = relation.partition("+")
    return base, (suffix if sep else None)


def join_relation(base: str, leaf) -> str:
    tag = leaf_tag(leaf)
    return f"{base}+{tag}" if tag else base
