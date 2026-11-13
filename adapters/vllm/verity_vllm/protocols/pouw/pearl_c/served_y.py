"""A call's served y, committed: its real rows (the y vLLM gets, ``y[:m]``) as A's rows are (each BF16 word widened exactly
to FP32, followed by its hm96 salt under the prover's key), in a ``frame-v3-sha512`` tree bound to the epoch's salt and the
call's index, and retained beside A's (``y/<index>.rows.bf16``, ``y/<index>.tree``).  `verify_run` opens a drawn tile's
real rows of it and holds each word to the unscale of the tile's checked U (`pearl_c_u.y`); the root is what a consumer of
y (the sampled proofs' blocks after the linear) is tied to.  A served call's tree is built on its device by the kernel's
own tree launches (`DeviceTrees`), `commit_y`'s bytes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from verity.protocols.pouw.verifier.audit import HM96, ROW_SCHEMA, hm96_salt, rows_domain
from verity.protocols.pouw.verifier.protocol import tagged
from verity.commitments import CommitmentDomain
from verity.commitments.merkle import tree_levels

from verity_vllm.protocols.pouw.pearl_c.host import _bytes, _put, _stream, _write, _zeros

__all__ = ["DeviceTrees", "HM96_Y", "Y_ROWS_TAG", "commit_y", "f32_rows", "retain_y", "y_binding", "y_domain"]

Y_ROWS_TAG = "verity/pouw/pearl-plus/y-rows/v1"
#: the hm96 salt stream's nonce of a call's y tree: this plus the call's index (A's trees take the index, below 2^61)
HM96_Y = 1 << 61
#: the bytes of FP32 words a device tree's row digests read at a time, widened from y's BF16 rows a chunk at a time
Y_CHUNK_BYTES = 64 << 20
#: the rows of the fixed y each width's first device tree is held to `commit_y` over: not a power of two, so the pads are
#: written, and past s512_tree_small's widest level (128 nodes), so s512_nodes runs too
Y_CHECK_ROWS = 129


def y_binding(salt: bytes, index: int) -> bytes:
    return tagged(Y_ROWS_TAG, salt, index)


def f32_rows(y_bf16: bytes, n: int) -> list[bytes]:
    """The rows of a row-major BF16 (m, n) y as the committed FP32 words."""
    words = np.frombuffer(y_bf16, "<u2").reshape(-1, n).astype("<u4") << 16
    return [r.tobytes() for r in words]


def y_domain(salt: bytes, index: int, m: int) -> CommitmentDomain:
    return rows_domain(y_binding(salt, index), index, m, HM96)


def commit_y(y_bf16: bytes, n: int, salt: bytes, index: int, salt_key: bytes) -> tuple[bytes, bytes]:
    """(root, the tree in the device's layout: level 0 padded to a power of two, then each level above it)."""
    rows = f32_rows(y_bf16, n)
    d = y_domain(salt, index, len(rows))
    levels = tree_levels(d, [d.leaf(i, i, ROW_SCHEMA, r + hm96_salt(salt_key, HM96_Y + index, i)) for i, r in enumerate(rows)])
    return levels[-1][0], b"".join(b"".join(level) for level in levels)


def retain_y(d: Path, index: int, y_bf16: bytes, n: int, salt: bytes, salt_key: bytes) -> dict:
    """Write the call's y rows and tree under ``d/y`` and return the matmul's ``y`` entry for its manifest."""
    (d / "y").mkdir(parents=True, exist_ok=True)
    root, tree = commit_y(y_bf16, n, salt, index, salt_key)
    return {"root": root.hex(), "rows": _write(d / "y" / f"{index}.rows.bf16", y_bf16, digest=False),
            "tree": _write(d / "y" / f"{index}.tree", tree)}


def y_tree(run: Any, y: Any, index: int) -> Any:
    """Call `index`'s tree over its real rows `y` (BF16, m x n) as uint8 bytes in `commit_y`'s layout, its root the last
    node: on a device by the run's `DeviceTrees`, on the CPU by `commit_y`."""
    import torch
    if not y.is_cuda:
        tree = commit_y(_bytes(y.contiguous().view(torch.uint8)), y.shape[1], run.salt, index, run.salt_key)[1]
        return torch.frombuffer(bytearray(tree), dtype=torch.uint8)
    if run._y_trees is None:
        run._y_trees = DeviceTrees(run)
    return run._y_trees(y, index)


class DeviceTrees:
    """A run's y trees on its device, by its kernel's own tree launches, the ones B's weight trees run: the row digests
    (`run.Pipeline.row_digests`) over y's rows widened to FP32 a chunk at a time (exact, as `f32_rows`), the hm96 leaves at
    nonce `HM96_Y` + index and the frame leaves under the call's `y_domain` (`hm_level0`), its pads and its levels, on the
    current stream, into a tree that starts as the pass's poison byte where the pass has one.  One pipeline a width n, y's
    rows its rows of k = n words, so n is a k the kernels take; its first tree is held to `commit_y` over a fixed
    `Y_CHECK_ROWS`-row y."""

    def __init__(self, run: Any) -> None:
        self.run, self.pipes = run, {}
        self.dom = _zeros(64, run.device)

    def __call__(self, y: Any, index: int) -> Any:
        n = y.shape[1]
        if n not in self.pipes:
            p = self._pipe(n)
            self._check(p, n, index)
            self.pipes[n] = p
        return self._tree(self.pipes[n], y.contiguous(), index)

    def _pipe(self, n: int) -> Any:
        run, R = self.run, self.run.kernel.run
        try:
            return R.Pipeline(run.kernel.dev, R.BM, R.BM, n, run.variant, {"dom_b": self.dom.data_ptr(), **run.const_keys}, 0, 0,
                              bufs={"tree_b": 0, "segdig_b": 0})
        except ValueError as e:
            raise ValueError(f"y's device tree hashes y's rows as the kernels' rows of k = {n} words: {e}") from None

    def _tree(self, p: Any, y: Any, index: int) -> Any:
        import torch
        R, dev, (m, n) = self.run.kernel.run, self.run.kernel.dev, y.shape
        poison = getattr(self.run.cur, "poison", None)
        size = R.NB * (2 * R.pow2(m) - 1)
        tree = (torch.empty(size, dtype=torch.uint8, device=y.device) if poison is None
                else torch.full((size,), poison, dtype=torch.uint8, device=y.device))
        rows = max(1, min(m, Y_CHUNK_BYTES // (4 * n)))
        dig = torch.empty(64 * m, dtype=torch.uint8, device=y.device)
        seg = torch.empty(64 * (n // 256) * rows, dtype=torch.uint8, device=y.device)
        _put(self.dom, y_domain(self.run.salt, index, m).domain_id)
        p.b["tree_b"] = tree.data_ptr()
        before, dev.stream = dev.stream, _stream(y.device)
        try:
            for _, f in p.tree_setup("b", m, "dom_b"):
                f()
            for r in range(0, m, rows):
                words = y[r:r + rows].float()
                p.row_digests(words.data_ptr(), dig.data_ptr() + 64 * r, seg.data_ptr(), words.shape[0])
            R.Pipeline.hm_level0(p, "b", dig.data_ptr(), m, "dom_b", nonce=R.nonce_args(HM96_Y + index))
            p.levels("b", m, "dom_b")
        finally:
            dev.stream = before
        return tree

    def _check(self, p: Any, n: int, index: int) -> None:
        import torch
        y = torch.linspace(-4, 4, Y_CHECK_ROWS * n).reshape(Y_CHECK_ROWS, n).to(torch.bfloat16)
        got = _bytes(self._tree(p, y.to(self.run.device), index))
        want = commit_y(_bytes(y.view(torch.uint8)), n, self.run.salt, index, self.run.salt_key)[1]
        if got != want:
            at = next(i for i in range(0, len(want), 64) if got[i:i + 64] != want[i:i + 64]) // 64
            raise ValueError(f"y's device tree at width {n} differs from served_y.commit_y over {Y_CHECK_ROWS} rows, first "
                             f"at node {at} of {len(want) // 64}")
