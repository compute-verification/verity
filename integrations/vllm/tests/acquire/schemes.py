"""Adapters that present every commitment construction under test through ONE interface, so the security tests in
`test_security.py` run unchanged against each of them (and against another engine once an adapter is added here).

    scheme.commit(values: list[bytes], ctx: bytes) -> Committed          # values in tree order; ctx = context the root is bound to
    scheme.open(c: Committed, i: int) -> Opening
    scheme.verify(root: bytes, n: int, ctx: bytes, opening) -> bool     # verifier side: only (root, n, ctx) + the opening

Adapters:
  (the Δ_commit committer `native_host` exposes `verify(RunCommitment, Opening)` — the verifier's held commitment — and is
   tested directly in `test_native_host_security.py` through its offline core `commit_offline`)
  * `C0PrimitiveScheme`  — the bare production primitive both native_host and fa2_commit build on: `pos_leaf` + `scheme.fold_levels`
                           + `path_in_levels` + `commit.fold_path`/`path_shape` (verity_vllm, the production opening verifier core)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from verity_vllm.commit import scheme


@dataclass
class Committed:
    root: bytes
    n: int
    ctx: bytes
    state: Any


class C0PrimitiveScheme:
    """pos_leaf leaves (one per value), scheme fold, path_in_levels; verifier = scheme.path_shape + fold_path + step_root(ctx)."""

    name = "c0_primitive"

    def _bind(self, ctx: bytes, n: int, merkle: bytes) -> bytes:
        z = hashlib.sha256(ctx).digest()
        return scheme.step_root(z, z, z, z, n, merkle)

    def commit(self, values: list[bytes], ctx: bytes) -> Committed:
        leaves = [scheme.pos_leaf(v) for v in values]
        levels = scheme.fold_levels(list(leaves))
        return Committed(self._bind(ctx, len(values), levels[-1][0]), len(values), ctx, (levels, list(values)))

    def open(self, c: Committed, i: int):
        levels, values = c.state
        return (i, values[i], scheme.path_in_levels(levels, i))

    def verify(self, root: bytes, n: int, ctx: bytes, opening) -> bool:
        i, value, path = opening
        if not 0 <= i < n:
            return False
        try:
            shape = scheme.path_shape(n, i)
        except Exception:  # noqa: BLE001
            return False
        if len(path) != len(shape) or any((p is None) == has for p, has in zip(path, shape, strict=True)):
            return False
        return self._bind(ctx, n, scheme.fold_path(scheme.pos_leaf(value), i, list(path))) == root

    @staticmethod
    def with_value(opening, value: bytes):
        return (opening[0], value, opening[2])

    @staticmethod
    def with_index(opening, index: int, identity=None):
        return (index, opening[1], opening[2])

    @staticmethod
    def with_path(opening, path):
        return (opening[0], opening[1], tuple(path))


SCHEMES = [C0PrimitiveScheme()]
