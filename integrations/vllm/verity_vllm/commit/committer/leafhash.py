"""Device-side weight registration -- the SAME weights root as the host path, computed where the parameters live.

`native_host.register_weights` hashes every parameter/buffer as CHUNK-byte position-identified leaves (`scheme.pos_leaf`) folded per
tensor (`scheme.fold`), and folds the per-tensor roots into the weights root.  With the GPU tree the chunk is 256 bytes: a 2.5 GB model
is 10 M Python `pos_leaf` calls (~0.9 us each) plus a Python fold -- 17-24 s measured on cp-sys, ~120 s for a 7B model -- and a Commit
does it twice (before the workload; again at the C2 replay for the O5 'committed weights' check).  `native_leafhash.cu` computes the same
leaves and the same tree on the device (SHA-256 core and tree kernel verbatim from native_tree.cu, one new pos_leaf kernel).

Modes (`VERITY_WEIGHTS_HASH` / commit_delta `--weights-hash`): `host` (default, unchanged), `device` (per-tensor subtree roots from the
GPU; the tensors' bytes never leave the device), `device-checked` (both; every tensor root compared, a mismatch raises -- validation runs).
Equality is not assumed: `tests/test_leafhash_device.py` asserts device == host over sizes with and without a shorter tail leaf, 1 leaf,
odd leaf counts; the checked mode asserts it on the real model at runtime.
"""
from __future__ import annotations

import os
import time
from typing import Any

from verity_vllm import settings

_EXT: Any = None
HERE = os.path.dirname(os.path.abspath(__file__))
MODES = ("host", "device", "device-checked")


def mode_from_env(default: str = "host") -> str:
    m = settings.VERITY_WEIGHTS_HASH if settings.is_set("VERITY_WEIGHTS_HASH") else default
    if m not in MODES:
        raise ValueError(f"VERITY_WEIGHTS_HASH must be one of {MODES}, got {m!r}")
    return m


def ext(build_dir: str | None = None, verbose: bool = False):
    """JIT-build the leaf-hash extension once per process into `<VERITY_LEAFHASH_BUILD or torch's default>/<source digest12>`, serialised
    with other jobs on the host (`native_jit.jit_load`)."""
    global _EXT
    if _EXT is None:
        from verity_vllm.commit.committer import native_jit

        t0 = time.perf_counter()
        _EXT, _bd, _digest = native_jit.jit_load("verity_leafhash", [os.path.join(HERE, "native_leafhash.cu")],
                                                 parent=build_dir or settings.VERITY_LEAFHASH_BUILD or None, cuda_cflags=("-O3",),
                                                 verbose=verbose)
        _EXT._verity_load_s = time.perf_counter() - t0
    return _EXT


def flat_u8(p) -> Any:
    """The tensor's bytes as a contiguous 1-D uint8 view ON ITS DEVICE (the host path's `p.detach().contiguous().reshape(-1).view(uint8)`
    before the `.cpu()`)."""
    import torch

    src = p.detach().contiguous().reshape(-1)
    return src.view(torch.uint8) if src.dtype != torch.uint8 else src


def tensor_root_device(p, chunk: int) -> bytes:
    """Subtree root of one tensor: pos_leaf over CHUNK-byte leaves of its bytes (last leaf shorter when the size is not a multiple), folded.
    Bytes identical to `scheme.fold([scheme.pos_leaf(mv[o:o+chunk]) ...])` on the host copy."""
    e = ext()
    flat = flat_u8(p)
    if not flat.is_cuda:
        flat = flat.cuda()
    leaves = e.pos_leaves(flat, int(chunk))
    root = e.tree_root(leaves)
    return bytes(root.cpu().numpy().tobytes())


def tensor_root_host(p, chunk: int) -> bytes:
    """The host path, verbatim (for the checked mode and the tests)."""
    from verity_vllm.commit import scheme

    flat = flat_u8(p)
    host = flat.cpu()
    mv = memoryview(host.numpy()).cast("B")
    return scheme.fold([scheme.pos_leaf(mv[o:o + chunk]) for o in range(0, len(mv), chunk)])
