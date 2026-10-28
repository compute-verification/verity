"""Range openings of the native host committer's step trees.

`NativeRangeOpenings` is a mixin of `commit.committer.native_host.NativeHostCommitter`: a run of leaves of a step is opened from the retained copy
with the boundary siblings of its tree (`open_range`) and verified against the run commitment (`verify_range`); `leaf_span` maps a tensor's
bytes to the leaves that hold them.  `fault_retained_flip` is the negative test of that path.  `open_levels` is one leaf's path from level
tensors; `level_sizes` is the number of nodes on each level.

The step trees are rows-sha512's (`step_rows`): their leaves are hiding leaves salted from each step's 64-byte key (``row_keys``), their
trees complete frame-v3-sha512 trees and their roots 64 bytes; a range opening carries its leaves' salts.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

from verity.ml.commitments import Commitment, Opening as FrameOpening, VerificationLimits, verify_opening
from verity_vllm.commit import step_rows


@dataclass
class RangeOpening:
    step: int
    lo: int                     # leaves [lo, hi) of the step tree
    hi: int
    value: bytes                # the leaves' bytes, concatenated in leaf order
    path: tuple                 # per level (left sibling | None, right sibling | None), `step_rows.range_shape`'s shape
    salts: bytes = b""          # the leaves' salts, concatenated


def open_levels(levels: list, index: int) -> tuple:
    """Sibling path from level tensors (uint8 [n, 64], leaves first), device or host."""
    path: list = []
    cursor = index
    for lv in levels[:-1]:
        n = int(lv.shape[0])
        if cursor == n - 1 and n % 2 == 1:
            path.append(None)
        else:
            path.append(lv[cursor ^ 1].cpu().numpy().tobytes())
        cursor //= 2
    return tuple(path)


def open_range_levels(levels: list, lo: int, hi: int) -> tuple:
    """Per level, the (left, right) siblings of the leaf run [lo, hi) (`step_rows.range_shape`'s pattern), read from host levels (lists of
    digests) or level tensors (uint8 [n, 64], gathered in one copy), leaves first."""
    need: list[tuple[int | None, int | None]] = []
    for lv in levels[:-1]:
        width = int(lv.shape[0]) if hasattr(lv, "shape") else len(lv)
        need.append((lo - 1 if lo % 2 else None, hi if (hi - 1) % 2 == 0 and hi < width else None))
        lo, hi = lo // 2, (hi + 1) // 2
    if not levels or not hasattr(levels[0], "shape"):
        return tuple((None if l is None else levels[d][l], None if r is None else levels[d][r]) for d, (l, r) in enumerate(need))
    import torch

    idx = [(d, i) for d, pair in enumerate(need) for i in pair if i is not None]
    parts = [levels[d][i] for d, i in idx]
    if len({p.device for p in parts}) > 1:
        parts = [p.cpu() for p in parts]
    rows = torch.stack(parts).cpu().numpy() if parts else None
    got = {di: rows[k].tobytes() for k, di in enumerate(idx)}
    return tuple((None if l is None else got[(d, l)], None if r is None else got[(d, r)]) for d, (l, r) in enumerate(need))


def level_sizes(leaf_count: int, levels: int) -> list[int]:
    """Nodes per level of a step tree over `leaf_count` leaves, leaves first: ceil(n / 2^L) for L = 0..min(levels, height) (an odd
    trailing node is lifted).  A windowed step keeps these levels on host below its window level (`levels` = the window's)."""
    height, m = 0, leaf_count
    while m > 1:
        m = (m + 1) // 2
        height += 1
    return [-(-leaf_count // (1 << L)) for L in range(min(levels, height) + 1)]


class NativeRangeOpenings:
    """Range openings over the committer's retained copies (`_gpu_blocks`) and levels (`_gpu_levels`)."""

    commit_scheme = step_rows.NAME
    rows = True
    replay_row_keys: dict[int, bytes] | None = None

    def _fold_run_root(self, roots: list[bytes]) -> bytes | None:
        """The run root over the step roots; None over a placeholder step root (b""): equal to no run root, the empty one included."""
        return step_rows.run_root(self.program_digest, self._geo_digest, roots) if all(len(r) == step_rows.ROOT_BYTES for r in roots) else None

    def _row_key(self, step: int) -> bytes:
        """rows-sha512: step `step`'s salt key, fresh from the OS generator, or the replayed run's (`replay_row_keys`)."""
        if self.replay_row_keys is None:
            ks = os.urandom(step_rows.KEY_BYTES)
        elif step in self.replay_row_keys:
            ks = self.replay_row_keys[step]
        else:
            raise RuntimeError(f"{step_rows.NAME} replay: the replayed run has no salt key for step {step}")
        self.row_keys[step] = ks
        return ks

    def _row_domain(self, step: int, n: int) -> Any:
        return step_rows.domain(self.program_digest, self._ctx_digest(step), self._geo_digest, n)

    def _gpu_salt(self, step: int, index: int) -> bytes:
        """The salt a step's leaf carries, from the step's key."""
        return step_rows.salt(self.row_keys[step], index)

    def _range_salts(self, step: int, lo: int, hi: int) -> bytes:
        return b"".join(self._gpu_salt(step, i) for i in range(lo, hi))

    def _run_root(self, run: Any) -> bytes | None:
        return self._fold_run_root([s.root for s in run.steps])

    def leaf_span(self, step: int, m: Any, lo: int, hi: int) -> tuple[int, int, int]:
        """The leaves [a, b) of `step` that hold bytes [lo, hi) of tensor `m` (rows of the packed step stream), and the offset of byte `lo`
        in their concatenated bytes."""
        s0, s1 = m.stream_off + lo, m.stream_off + hi
        return s0 // self.chunk, -(-s1 // self.chunk), s0 % self.chunk

    def fault_retained_flip(self, spec: str) -> dict:
        """NEGATIVE TEST ONLY: `VERITY_FAULT=retained_flip[:<name substring>]` (default `qkv_proj`).  After the commit and before the openings
        and value checks, flip the low bit of byte 0 of the earliest committed member whose name contains the substring, in THIS committer's
        retained copy (the copy every opening and value check reads).  The run root binds the unflipped bytes, so the opened value check must
        FAIL naming the member.  Raises when no member matches or its retained copy does not hold that byte."""
        sub = spec.partition(":")[2] or "qkv_proj"
        step, m = next(((s, m) for s in sorted(self._layouts) for m in self._layouts[s] if sub in m.name), (None, None))
        if m is None:
            raise RuntimeError(f"{spec}: no committed member name contains {sub!r}")
        a, _b, _o = self.leaf_span(step, m, 0, 1)
        blk, pos = self._gpu_blocks[step][0], m.stream_off
        if blk is None or not self._block_holds(blk, pos, pos + 1):
            raise RuntimeError(f"{spec}: step {step} does not retain byte {pos} of its stream (retain={self.retain})")
        # windowed / ranged host copy: a one-byte slice is a view into the window or range holding the byte
        cell, i = (blk, pos) if getattr(blk, "holds", None) is None else (blk[pos:pos + 1], 0)
        old = int(cell[i])
        torch = sys.modules.get("torch")                # a retained tensor exists only if torch is loaded
        if torch is not None and isinstance(cell, torch.Tensor):
            with torch.inference_mode():             # copies retained under inference mode are inference tensors: writable only inside it
                cell[i] = old ^ 1
        else:
            cell[i] = old ^ 1
        return {"fault": spec, "step": step, "name": m.name, "ordinal": m.ordinal, "byte": 0, "leaf": a, "old": old, "new": old ^ 1}

    def open_range(self, step: int, lo: int, hi: int) -> RangeOpening | None:
        """A range opening of leaves [lo, hi) of `step`: their bytes from this committer's retained copy and the run's boundary siblings
        from its levels.  None when the retained copy does not hold them (they open by replay, `open(values_from=)`)."""
        C = self.chunk
        blk = self._gpu_blocks[step][0]
        if not self._block_holds(blk, lo * C, hi * C):
            return None
        if step not in self._gpu_levels:
            raise RuntimeError(f"open_range({step}, {lo}, {hi}): this run kept no tree levels (retain_levels=False)")
        seg = blk[lo * C:hi * C]
        value = bytes((seg.cpu() if getattr(seg, "is_cuda", False) else seg).numpy().tobytes())
        return RangeOpening(step, lo, hi, value, open_range_levels(self._gpu_levels[step], lo, hi), self._range_salts(step, lo, hi))

    def verify_range(self, run: Any, ro: RangeOpening, roots: dict[int, bytes] | None = None) -> bool:
        """`verify` for a range opening: each row and its salt give its hiding leaf, the run and its boundary siblings fold to the tree root
        (`step_rows.fold_range`), which must bind to the step's ORIGINAL root, and the step roots must fold to the 64-byte run root.
        `roots` (step -> tree root already verified against this `run`) lets a caller pay the binding once per step."""
        if not 0 <= ro.step < len(run.steps):
            return False
        sc = run.steps[ro.step]
        if not 0 <= ro.lo < ro.hi <= sc.n_leaves:
            return False
        n, C, y = ro.hi - ro.lo, step_rows.ROW_BYTES, step_rows.SALT_BYTES
        if ro.step not in self._gpu_blocks or len(ro.value) != n * C or len(ro.salts) != n * y:
            return False
        hs = [step_rows.hiding_leaf(ro.value[k * C:(k + 1) * C], ro.salts[k * y:(k + 1) * y], 8 * C)[1] for k in range(n)]
        merkle = step_rows.fold_range(self._row_domain(ro.step, sc.n_leaves), ro.lo, hs, ro.path)
        if merkle is None:
            return False
        if roots is not None and ro.step in roots:
            return merkle == roots[ro.step]
        root = step_rows.step_root(self.program_digest, self._ctx_digest(ro.step), self._geo_digest,
                                   self.layout_digest(self._layouts[ro.step]), sc.n_leaves, merkle)
        if root != sc.root or len(run.run_root) != step_rows.ROOT_BYTES or self._run_root(run) != run.run_root:
            return False
        if roots is not None:
            roots[ro.step] = merkle
        return True

    def _verify_rows(self, run: Any, opening: Any) -> bool:
        """rows-sha512's `verify`: the row and its salt give the hiding leaf, which the frame path opens under the step's tree root (core's
        `verify_opening`), which binds into the step root, which is in the run root."""
        try:
            sc = run.steps[opening.step]
            ident = self.identity_of(opening.step, opening.index)
        except (IndexError, KeyError):
            return False
        if opening.step not in self._gpu_blocks or ident != opening.identity or len(opening.value) != step_rows.ROW_BYTES:
            return False
        if len(opening.salt) != step_rows.SALT_BYTES or any(not isinstance(p, bytes) for p in opening.path):
            return False
        h = step_rows.hiding_leaf(opening.value, opening.salt, 8 * step_rows.ROW_BYTES)[1]
        try:
            if not verify_opening(self._row_domain(opening.step, sc.n_leaves), Commitment(sc.merkle_root, sc.n_leaves),
                                  FrameOpening(opening.index, h, tuple(opening.path)), step_rows.SCHEMA, VerificationLimits()):
                return False
        except ValueError:
            return False
        root = step_rows.step_root(self.program_digest, self._ctx_digest(opening.step), self._geo_digest,
                                   self.layout_digest(self._layouts[opening.step]), sc.n_leaves, sc.merkle_root)
        return root == sc.root and len(run.run_root) == step_rows.ROOT_BYTES and self._run_root(run) == run.run_root
