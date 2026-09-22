"""EXPERIMENTAL (R9 lane CMT-1, 2026-09-18): one-stage commitment of naturally-materialized interface values.

Nothing here is on a gate / verdict path.  The engine REUSES the production framing and verifier core
(`verity_vllm/{hashing,commit}.py`: SHA-256 `H(tag, *parts)`, node / lift rule, `path_shape` / `fold_path`) and the C0
position-identified leaf (`verity_vllm.commit_bench.semantic.pos_leaf`).  Small stable API for CMT-2 / CMT-3 (see
`out/gen/r9/cmt-ref/API.md`):

    from verity_capture.experimental.commit import commit_stream, open, verify, Committer, Context, StepLayout

    root, positions, c = commit_stream(program_digest, ctx, steps, backend="torch")   # steps: [(StepLayout, stream), ...]
    op = open(c, step, class_id, instance, element, stream=steps[step][1])            # Opening (json-able)
    ok, why = verify(root, positions, op)                                               # production fold_path / path_shape

Streaming form (one engine step at a time, GPU-resident when the stream is a CUDA uint8 tensor):

    c = Committer(program_digest, ctx, backend="torch", host_levels_below=256)
    for layout, stream in steps: c.commit_step(layout, stream)
    root, positions = c.finish()
"""
from __future__ import annotations

from .engine import (  # noqa: F401
    LEAF_BYTES,
    ROOT_TAG,
    Committer,
    Context,
    Opening,
    Verifier,
    bind_root,
    leaf_digests_host,
    open_value,
    verify_opening,
)
from .positions import Segment, StepLayout, positions_digest  # noqa: F401


def commit_stream(program_digest: bytes, ctx: Context, steps, *, backend: str = "host", compiled: bool = False,
                  host_levels_below: int = 0, keep_levels: bool = True) -> tuple[bytes, dict, Committer]:
    """Commit a whole request at once: `steps` = iterable of (StepLayout, stream) in step order.  Returns (root, positions, committer);
    keep the committer (its retained tree levels) to generate openings with `open(...)`."""
    c = Committer(program_digest, ctx, backend=backend, compiled=compiled, host_levels_below=host_levels_below, keep_levels=keep_levels)
    for layout, stream in steps:
        c.commit_step(layout, stream)
    root, positions = c.finish()
    return root, positions, c


def gpu_sha256(padded):
    """ONE Triton launch: FIPS-padded uint8 [N, L] (L % 64 == 0, any block count) -> digests uint8 [N, 32] on the same CUDA device.
    Checked against hashlib (`test_triton_sha256.py`).  Use with `torch_sha256.pad_messages` / `pos_leaf_digests(..., compiled="triton")`
    / `fold_levels(..., compiled="triton")`, or simply `Committer(..., backend="torch", compiled="triton")`."""
    from .triton_sha256 import sha256_padded_triton
    return sha256_padded_triton(padded)


def gpu_leaf_digests(values, *, leaf_bytes: int | None = None):
    """uint8 [N, V] (or a flat uint8 stream + leaf_bytes) -> pos_leaf digests uint8 [N, 32] with the Triton kernel (one launch)."""
    from .torch_sha256 import pos_leaf_digests
    if leaf_bytes is not None:
        n = -(-values.numel() // leaf_bytes)
        if n * leaf_bytes != values.numel():
            import torch
            values = torch.cat([values.reshape(-1), values.new_zeros(n * leaf_bytes - values.numel())])
        values = values.view(n, leaf_bytes)
    return pos_leaf_digests(values, compiled="triton")


def gpu_fold(leaf_digests, *, stop_at: int = 0):
    """leaf digests uint8 [N, 32] -> list of levels (production pairing/lift frames), one Triton launch per level; levels[-1][0] is the root
    (when stop_at == 0)."""
    from .torch_sha256 import fold_levels
    return fold_levels(leaf_digests, compiled="triton", stop_at=stop_at)


open = open_value          # noqa: A001  -- the stable short names promised to CMT-2 / CMT-3
verify = verify_opening

__all__ = ["LEAF_BYTES", "ROOT_TAG", "Committer", "Context", "Opening", "Segment", "StepLayout", "bind_root", "commit_stream",
           "Verifier", "gpu_fold", "gpu_leaf_digests", "gpu_sha256",
           "leaf_digests_host", "open", "open_value", "positions_digest", "verify", "verify_opening"]
