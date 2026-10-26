"""C0 — semantic commitment layout: position carries identity (commitment engine, 2026-09-17).

A commitment position ``i`` of an RU tree is derived by arithmetic from STATIC context only:

    Template   (from the Program / record layout, published once per profile):
        per_position : ordered rows (layer, op, port, dtype, shape) emitted for every token position
        per_ru       : ordered rows emitted once per RU at its last position (lm_head, sampler)
    RUContext  (from the record's request table): profile id, request id, prompt length P, step s
        positions(s) = [0, P)          for the prefill step (s = 0)
                     = [P + s - 1]     for decode step s

    N = len(positions) * len(per_position) + len(per_ru)
    unrank(i) = (positions[i // R], per_position[i % R])          for i <  len(positions) * R
              = (positions[-1],     per_ru[i - len(positions) * R]) otherwise
    rank(canonical id) = the inverse (dictionary on (layer, op, port) + arithmetic on t)

so the runtime hashes ONLY packed values: leaf_i = SHA-256("verity/pos-leaf/v0" || u64 nbytes || value bytes) — no id text,
no dtype/shape text (those are in the template digest, which is part of the static context).  The tree over the leaves uses
the production pairing/lift rule (``scheme.fold`` / ``StreamingMerkle``) and the published root is

    R = SHA-256("verity/semantic-root/v0" || program_digest || query_id || template_digest || ctx_digest || epoch || u64 N || merkle_root)

An opening of position ``i`` is ``(i, value, path)``; with the static context it authenticates the same value bytes that the
production scheme authenticates via its id-carrying leaf, and ``unrank(i)`` names the value (dtype/shape from the template).

Honest limits (what is NOT arithmetic): the prompt length P and the request id are runtime data of the record — they must be
published (the record's request table); dynamic batch composition, ragged rows, speculative/rejected tokens and MoE routing
would make ``positions``/``per_position`` runtime data too and would have to be committed as leaves or published as metadata.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from verity_vllm.program import dtypes
from verity_vllm.commit import identity, scheme


@dataclass(frozen=True, slots=True)
class RowSpec:
    layer: str
    op: str
    port: int
    dtype: str
    shape: tuple[int, ...]

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.layer, self.op, self.port)


@dataclass(frozen=True, slots=True)
class Template:
    """Static per-profile layout: rows per token position and rows per RU (emitted once, at the last position)."""

    per_position: tuple[RowSpec, ...]
    per_ru: tuple[RowSpec, ...]

    def digest(self) -> bytes:
        doc = {"per_position": [[r.layer, r.op, r.port, r.dtype, list(r.shape)] for r in self.per_position],
               "per_ru": [[r.layer, r.op, r.port, r.dtype, list(r.shape)] for r in self.per_ru]}
        return hashlib.sha256(json.dumps(doc, separators=(",", ":")).encode()).digest()

    @classmethod
    def from_decode_index(cls, entries: Iterable[Any], per_ru_layers: Sequence[str] = ("lm_head", "sampler")) -> Template:
        """Derive the template from ONE decode RU's index (emission order).  Stands in for the Program layout tonight;
        the same object would be produced from the Program/profile once the correspondence is wired."""
        pos_rows: list[RowSpec] = []
        ru_rows: list[RowSpec] = []
        for e in entries:
            cid = identity.parse(e.canonical_id)
            spec = RowSpec(str(cid.layer), str(cid.op), int(cid.port), e.dtype, tuple(e.shape))
            (ru_rows if cid.layer in per_ru_layers else pos_rows).append(spec)
        return cls(tuple(pos_rows), tuple(ru_rows))


@dataclass(frozen=True, slots=True)
class RUContext:
    profile: str
    request: str
    prompt_len: int
    step: int

    def positions(self) -> range:
        return range(self.prompt_len) if self.step == 0 else range(self.prompt_len + self.step - 1, self.prompt_len + self.step)

    @property
    def phase(self) -> str:
        return "prefill" if self.step == 0 else "decode"

    def digest(self) -> bytes:
        return hashlib.sha256(f"{self.profile}/{self.request}/{self.prompt_len}/{self.step}".encode()).digest()


class Layout:
    """rank / unrank between commitment positions and canonical ids, from static context only."""

    __slots__ = ("template", "ctx", "_positions", "_R", "_npos", "_row_rank", "_ru_rank", "count")

    def __init__(self, template: Template, ctx: RUContext) -> None:
        self.template, self.ctx = template, ctx
        self._positions = ctx.positions()
        self._R = len(template.per_position)
        self._npos = len(self._positions)
        self._row_rank = {r.key: k for k, r in enumerate(template.per_position)}
        self._ru_rank = {r.key: k for k, r in enumerate(template.per_ru)}
        self.count = self._npos * self._R + len(template.per_ru)

    def unrank(self, i: int) -> tuple[int, RowSpec]:
        if not 0 <= i < self.count:
            raise IndexError(i)
        q, r = divmod(i, self._R)
        if q < self._npos:
            return self._positions[q], self.template.per_position[r]
        return self._positions[-1], self.template.per_ru[i - self._npos * self._R]

    def canonical_id(self, i: int) -> str:
        t, row = self.unrank(i)
        return f"p={self.ctx.profile}/r={self.ctx.request}/t={t}/ph={self.ctx.phase}/l={row.layer}/op={row.op}/port={row.port}"

    def rank(self, canonical_id: str) -> int:
        cid = identity.parse(canonical_id)
        key = (str(cid.layer), str(cid.op), int(cid.port))
        t = int(cid.t)
        k = self._ru_rank.get(key)
        if k is not None:
            return self._npos * self._R + k
        return (t - self._positions[0]) * self._R + self._row_rank[key]

    def rank_fast(self, t: int, key: tuple[str, str, int]) -> int:
        """rank without parsing an id string (the caller already knows (t, layer, op, port))."""
        k = self._ru_rank.get(key)
        if k is not None:
            return self._npos * self._R + k
        return (t - self._positions[0]) * self._R + self._row_rank[key]


@dataclass(frozen=True, slots=True)
class StaticContext:
    program_digest: bytes
    query_id: bytes
    template: Template
    ctx: RUContext
    epoch: int
    n: int

    def root(self, merkle_root: bytes) -> bytes:
        return scheme.semantic_root(self.program_digest, self.query_id, self.template.digest(), self.ctx.digest(), self.epoch, self.n, merkle_root)


def verify_position_opening(R: bytes, sc: StaticContext, index: int, value: bytes, path: Sequence[bytes | None]) -> tuple[bool, str | None]:
    """Authenticate ``value`` at ``index`` against the published ``R`` given the static context; returns (ok, canonical id)."""
    if not 0 <= index < sc.n:
        return False, None
    try:
        expected = scheme.path_shape(sc.n, index)
    except Exception:  # noqa: BLE001
        return False, None
    if len(path) != len(expected) or any((p is None) == has for p, has in zip(path, expected, strict=True)):
        return False, None
    layout = Layout(sc.template, sc.ctx)
    if layout.count != sc.n:
        return False, None
    _t, row = layout.unrank(index)
    if len(value) != dtypes.expected_nbytes(row.dtype, row.shape):
        return False, None
    merkle = scheme.fold_path(scheme.pos_leaf(value), index, path)
    return sc.root(merkle) == R, layout.canonical_id(index)
