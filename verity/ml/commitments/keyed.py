"""The keyed leaf: a frame-v3-sha512 subtree's public digest d, keyed by a window's secret ks_w and its live salt salt_w.

    S_w = f(IV, ks_w || salt_w)              one SHA-512 compression: ks_w (64 secret bytes) and salt_w (64) are one block
    d   = SHA-512(ks_w || salt_w || M_top)   SHA-512 resumed from S_w over M_top, its length counting the 128 key bytes

M_top is the frame-v3-sha512 message whose SHA-512 is the subtree's root in :func:`~verity.ml.commitments.merkle.tree_levels`:
``node(D - 1, 0, L, R)`` for a subtree of depth D >= 1, or for a one-leaf subtree the leaf's own message.  Below d the subtree
is hashed from the IV as any frame-v3-sha512 tree; d takes its root's place and is not a frame-v3 message.  M_top's first block
holds the subtree's domain id, so distinct domains give a window prefix-free keyed messages.  d hides its subtree only while a
window keys one d at each position and its positions are injective, and only while ks_w was committed (its hm96 leaf) before
salt_w was drawn: the committer that keys digests refuses everything else.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

from verity.errors import ProtocolError

from .merkle import _NODE, CommitmentDomain, _message, _uint, merkle_depth

KEY_BYTES = 64


def top_from(dom: CommitmentDomain, schema: str, count: int, top: bytes) -> bytes:
    """M_top of a subtree of ``count`` leaves under ``dom`` from its top as a hasher hands it: the root's two children (128
    bytes) for ``count`` >= 2, or the one leaf's value."""
    if dom.hash != "sha512" or type(count) is not int or count < 1:
        raise ProtocolError("a keyed digest is over a nonempty frame-v3-sha512 subtree")
    depth = merkle_depth(count)
    if depth == 0:
        return dom.leaf_prefix(0, 0, schema, len(top)) + bytes(top)
    if len(top) != 128:
        raise ProtocolError(f"a subtree's top is its root's two 64-byte children, not {len(top)} bytes")
    return _message(_NODE, dom.domain_id, _uint(depth - 1), _uint(0), bytes(top[:64]), bytes(top[64:]))


def top_message(dom: CommitmentDomain, schema: str, values: Sequence[bytes], levels: Sequence[Sequence[bytes]]) -> bytes:
    """M_top of ``levels`` (``tree_levels`` under ``dom`` over the leaves of ``values``), refused unless its SHA-512 is their
    root."""
    if dom.hash != "sha512" or not values or len(levels) != merkle_depth(len(values)) + 1:
        raise ProtocolError("a keyed digest is over a nonempty frame-v3-sha512 subtree")
    depth = len(levels) - 1
    message = top_from(dom, schema, len(values), values[0] if depth == 0 else b"".join(levels[depth - 1]))
    if hashlib.sha512(message).digest() != levels[-1][0]:
        raise ProtocolError("M_top is not the subtree root's message")
    return message


def digest(ks_w: bytes, salt_w: bytes, m_top: bytes) -> bytes:
    """d over M_top."""
    if len(ks_w) != KEY_BYTES or len(salt_w) != KEY_BYTES:
        raise ProtocolError(f"ks_w and salt_w are {KEY_BYTES} bytes each")
    return hashlib.sha512(ks_w + salt_w + m_top).digest()
