"""A call's served y, committed: its rows as A's are (each BF16 word widened exactly to FP32, followed by its hm96 salt under
the prover's key), in a ``frame-v3-sha512`` tree bound to the epoch's salt and the call's index, and retained beside A's
(``y/<index>.rows.bf16``, ``y/<index>.tree``).  `verify_run` opens a drawn tile's rows of it and holds each word to the
unscale of the tile's checked U (`pearl_c_u.y`); the root is what a consumer of y (the sampled proofs' blocks after the
linear) is tied to."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from verity.core.protocols.pouw.audit import HM96, ROW_SCHEMA, hm96_salt, rows_domain
from verity.core.protocols.pouw.protocol import tagged
from verity.ml.commitments import CommitmentDomain
from verity.ml.commitments.merkle import tree_levels

from verity_vllm.protocols.pouw.pearl_c.host import _write

Y_ROWS = "verity/pouw/pearl-plus/y-rows/v1"
#: the hm96 salt stream's nonce of a call's y tree: this plus the call's index (A's trees take the index, below 2^61)
HM96_Y = 1 << 61


def y_binding(salt: bytes, index: int) -> bytes:
    return tagged(Y_ROWS, salt, index)


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
