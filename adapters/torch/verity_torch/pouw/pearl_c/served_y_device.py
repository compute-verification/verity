"""A served call's y tree on its device (`DeviceTrees`, `y_tree`), over torch tensors: `served_y.commit_y`'s bytes by the
kernel's own tree launches (`served_y.tree_launches`)."""
from __future__ import annotations

from typing import Any

from verity_torch.pouw.pearl_c.host import _bytes, _put, _stream, _zeros
from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c.served_y import (
    Y_CHECK_ROWS,
    chunk_rows,
    commit_y,
    tree_launches,
    y_domain,
)


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

    pipes: dict[int, Any]

    def __init__(self, run: Any) -> None:
        self.run, self.pipes = run, {}
        self.dom = _zeros(64, run.device)

    def __call__(self, y: Any, index: int) -> Any:
        return self._tree(self.pipe(y.shape[1], index), y.contiguous(), index)

    def pipe(self, n: int, index: int) -> Any:
        """Width `n`'s pipeline, made and held to `commit_y` at its first use."""
        if n not in self.pipes:
            p = self._pipe(n)
            self._check(p, n, index)
            self.pipes[n] = p
        return self.pipes[n]

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
        rows = chunk_rows(m, n)
        dig = torch.empty(64 * m, dtype=torch.uint8, device=y.device)
        seg = torch.empty(64 * (n // 256) * rows, dtype=torch.uint8, device=y.device)
        _put(self.dom, y_domain(self.run.salt, index, m).domain_id)
        held: list[Any] = []

        def words(r: int, count: int) -> int:
            held[:] = [y[r:r + count].float()]
            return held[0].data_ptr()
        before, dev.stream = dev.stream, _stream(y.device)
        try:
            tree_launches(R, p, tree.data_ptr(), dig.data_ptr(), seg.data_ptr(), m, rows, index, words)
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


__all__ = ["DeviceTrees", "y_tree"]
