"""A call's served y, committed: its real rows (the y the workload gets, ``y[:m]``) as A's rows are (each BF16 word widened exactly
to FP32, followed by its hm96 salt under the prover's key), in a ``frame-v3-sha512`` tree bound to the epoch's salt and the
call's index, and retained beside A's (``y/<index>.rows.bf16``, ``y/<index>.tree``).  `verify_run` opens a sampled tile's
real rows of it and holds each word to the unscale of the tile's checked U (`pearl_c_u.y`); the root is what a consumer of
y (SPOT's proof blocks after the linear) is tied to.  A served call's tree is built on its device by the kernel's
own tree launches (`tree_launches`), `commit_y`'s bytes."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from verity.commitments import CommitmentDomain
from verity.commitments.merkle import tree_levels
from verity.protocols.resource_accounting.compute.pouw.verifier.audit import (
    HM96,
    ROW_SCHEMA,
    hm96_salt,
    rows_domain,
)
from verity.protocols.resource_accounting.compute.pouw.verifier.protocol import tagged

from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c.manifest import _write

__all__ = ["HM96_Y", "Y_ROWS_TAG", "chunk_rows", "commit_y", "f32_rows", "retain_y", "tree_launches", "y_binding", "y_domain"]

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


def chunk_rows(m: int, n: int) -> int:
    """The rows of y a device tree widens and digests at a time: `Y_CHUNK_BYTES` of FP32 words, at least one row."""
    return max(1, min(m, Y_CHUNK_BYTES // (4 * n)))


def tree_launches(R: Any, p: Any, tree: int, dig: int, seg: int, m: int, rows: int, index: int,
                  words: Callable[[int, int], int]) -> None:
    """Call `index`'s y tree over its `m` rows into `tree` (`commit_y`'s layout), launches only, on the device's stream:
    the pads, the row digests `rows` rows at a time into `dig` through the chunk's scratch `seg`, each chunk's FP32 words at
    the address `words(r, count)` returns once it has widened rows r to r + count - 1, then the hm96 leaves at nonce
    `HM96_Y` + index and the frame leaves under p's ``dom_b`` (y's `y_domain`), and the levels.  `R` is p's run module."""
    p.b["tree_b"] = tree
    for _, f in p.tree_setup("b", m, "dom_b"):
        f()
    for r in range(0, m, rows):
        count = min(rows, m - r)
        p.row_digests(words(r, count), dig + 64 * r, seg, count)
    R.Pipeline.hm_level0(p, "b", dig, m, "dom_b", nonce=R.nonce_args(HM96_Y + index))
    p.levels("b", m, "dom_b")
