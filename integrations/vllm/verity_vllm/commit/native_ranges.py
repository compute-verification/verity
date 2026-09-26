"""Range openings of the native host committer's step trees.

`NativeRangeOpenings` is a mixin of `commit.committer.native_host.NativeHostCommitter`: a run of leaves of a step is opened from the retained copy
with the boundary siblings of its tree (`open_range`) and verified against the run commitment (`verify_range`); `leaf_span` maps a tensor's
bytes to the leaves that hold them.  `fault_retained_flip` is the negative test of that path.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from verity_vllm.commit import scheme
from verity_vllm.commit.merkle import fold_range, range_path_shape


@dataclass
class RangeOpening:
    step: int
    lo: int                     # leaves [lo, hi) of the step tree
    hi: int
    value: bytes                # the leaves' bytes, concatenated in leaf order
    path: tuple                 # per level (left sibling | None, right sibling | None) — `merkle.range_path_shape` shape


def open_range_levels(levels: list, lo: int, hi: int) -> tuple:
    """Per level, the (left, right) siblings of the leaf run [lo, hi) (`merkle.range_path_shape`'s pattern), read from host levels (lists of
    digests) or level tensors (int32 [n, 8], gathered in one copy), leaves first."""
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
    rows = torch.stack(parts).cpu().numpy().astype("uint32") if parts else None
    got = {di: rows[k].tobytes() for k, di in enumerate(idx)}
    return tuple((None if l is None else got[(d, l)], None if r is None else got[(d, r)]) for d, (l, r) in enumerate(need))


class NativeRangeOpenings:
    """Range openings over the committer's retained copies (`_gpu_blocks`, `_streams`, per-tensor `host`, padding streams) and levels."""

    def _run_root(self, run: Any) -> bytes | None:
        roots = [s.root for s in run.steps]                     # None over a placeholder step root (b""): equal to no run root, the empty one included
        return scheme.run_root(self.program_digest, self._geo_digest, roots) if all(len(r) == 32 for r in roots) else None

    def leaf_span(self, step: int, m: Any, lo: int, hi: int) -> tuple[int, int, int]:
        """The leaves [a, b) of `step` that hold bytes [lo, hi) of tensor `m`, and the offset of byte `lo` in their concatenated bytes."""
        so = m.stream_off
        if so is not None and so >= 0:                            # packed (GPU block, host stream, padding): chunks of the step stream
            s0, s1 = so + lo, so + hi
            return s0 // self.chunk, -(-s1 // self.chunk), s0 % self.chunk
        return m.first_leaf + lo // m.chunk, m.first_leaf + -(-hi // m.chunk), lo % m.chunk

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
        if step in self._gpu_blocks:
            blk, pos = self._gpu_blocks[step][0], m.stream_off
            if blk is None or not self._block_holds(blk, pos, pos + 1):
                raise RuntimeError(f"{spec}: step {step} does not retain byte {pos} of its stream (retain={self.retain})")
            # windowed / ranged host copy: a one-byte slice is a view into the window or range holding the byte
            cell, i = (blk, pos) if getattr(blk, "holds", None) is None else (blk[pos:pos + 1], 0)
        elif step in self._streams:
            cell, i = self._streams[step], m.stream_off
        else:
            if m.host is None:
                raise RuntimeError(f"{spec}: {m.name} at step {step} has no retained host copy (retain={self.retain})")
            cell, i = m.host, 0
        old = int(cell[i])
        torch = sys.modules.get("torch")                # a retained tensor exists only if torch is loaded
        if torch is not None and isinstance(cell, torch.Tensor):
            with torch.inference_mode():             # copies retained under inference mode are inference tensors: writable only inside it
                cell[i] = old ^ 1
        else:
            cell[i] = old ^ 1
        return {"fault": spec, "step": step, "name": m.name, "ordinal": m.ordinal, "byte": 0, "leaf": a, "old": old, "new": old ^ 1}

    def _tensor_of_leaves(self, step: int, lo: int, hi: int) -> Any:
        return next((m for m in self._layouts.get(step, []) if m.first_leaf <= lo and hi <= m.first_leaf + m.n_leaves), None)

    def open_range(self, step: int, lo: int, hi: int) -> RangeOpening | None:
        """A range opening of leaves [lo, hi) of `step` (all executed or all padding): their bytes from this committer's retained copy and the
        run's boundary siblings from its levels.  None when the retained copy does not hold them (they open by replay, `open(values_from=)`)."""
        C = self.chunk
        ps = self._padding_leaf(step, lo)
        if ps is not None:
            value = bytes(ps.stream[(lo - ps.leaf_base) * C:(hi - ps.leaf_base) * C])
            levels = self._levels[step]
        elif step in self._gpu_blocks:
            blk = self._gpu_blocks[step][0]
            if not self._block_holds(blk, lo * C, hi * C):
                return None
            seg = blk[lo * C:hi * C]
            value = bytes((seg.cpu() if getattr(seg, "is_cuda", False) else seg).numpy().tobytes())
            levels = self._gpu_levels[step]
        elif step in self._streams:
            value = bytes(self._streams[step][lo * C:hi * C])
            levels = self._levels[step]
        else:
            m = self._tensor_of_leaves(step, lo, hi)
            if m is None or m.host is None:
                return None
            value = bytes(m.host[(lo - m.first_leaf) * m.chunk:(hi - m.first_leaf) * m.chunk])
            levels = self._levels[step]
        return RangeOpening(step, lo, hi, value, open_range_levels(levels, lo, hi))

    def _range_leaves(self, step: int, lo: int, hi: int, value: bytes) -> list[bytes] | None:
        """The leaf digests of leaves [lo, hi) of `step` recomputed from their concatenated bytes under `verify`'s leaf rules; None when the
        bytes do not tile the leaves or the run mixes executed and padding leaves."""
        C, mv = self.chunk, memoryview(value)
        ps = self._padding_leaf(step, lo)
        if ps is not None or self._padding_leaf(step, hi - 1) is not None:
            if ps is None:
                return None
            from verity_vllm.commit import padding_steps as PSm

            end = ps.stream_base + ps.n_bytes - lo * C
            if len(value) != min((hi - lo) * C, end):
                return None
            return [PSm.leaf_digest_of(ps, mv[k * C:min((k + 1) * C, end)], lo + k - ps.leaf_base) for k in range(hi - lo)]
        if step in self._gpu_blocks:
            if len(value) != (hi - lo) * C:
                return None
            padded = -(-self._gpu_blocks[step][1] // C) * C
            layout = self._gpu_layout(step, padded // 4)
            return [scheme.chunk_leaf(scheme.chunk_header(layout, step, lo + k, C // 4, self.GPU_TREE_SRC), mv[k * C:(k + 1) * C])
                    for k in range(hi - lo)]
        if step in self._streams:
            end = len(self._streams[step]) - lo * C
        else:
            m = self._tensor_of_leaves(step, lo, hi)
            if m is None:
                return None
            C = m.chunk
            end = m.nbytes - (lo - m.first_leaf) * C
        if len(value) != min((hi - lo) * C, end):
            return None
        return [scheme.pos_leaf(mv[k * C:min((k + 1) * C, end)]) for k in range(hi - lo)]

    def verify_range(self, run: Any, ro: RangeOpening, roots: dict[int, bytes] | None = None) -> bool:
        """`verify` for a range opening: every leaf digest recomputed from the opening's bytes, the run folded with its boundary siblings to
        the bare root, which must bind to the step's ORIGINAL root, and the step roots must fold to the run root.  `roots` (step -> bare root
        already verified against this `run`) lets a caller pay the binding once per step."""
        if not 0 <= ro.step < len(run.steps):
            return False
        sc = run.steps[ro.step]
        if not 0 <= ro.lo < ro.hi <= sc.n_leaves:
            return False
        try:
            shape = range_path_shape(sc.n_leaves, ro.lo, ro.hi)
            if len(ro.path) != len(shape) or any((left is None) == need_l or (right is None) == need_r
                                                 for (left, right), (need_l, need_r) in zip(ro.path, shape)):
                return False
            leaves = self._range_leaves(ro.step, ro.lo, ro.hi, ro.value)
            if leaves is None:
                return False
            merkle = fold_range(leaves, sc.n_leaves, ro.lo, ro.path)
        except (ValueError, TypeError):
            return False
        if roots is not None and ro.step in roots:
            return merkle == roots[ro.step]
        root = scheme.step_root(self.program_digest, self._ctx_digest(ro.step), self._geo_digest,
                             self.layout_digest(self._layouts[ro.step]), sc.n_leaves, merkle)
        if root != sc.root or len(run.run_root) != 32 or self._run_root(run) != run.run_root:
            return False
        if roots is not None:
            roots[ro.step] = merkle
        return True
