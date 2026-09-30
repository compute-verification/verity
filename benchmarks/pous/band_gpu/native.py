"""The JIT build of `pous_native.cu` (the batched SHAKE256 kernel and the audit responder), keyed by what it compiles.

`load(build_dir)` builds (or adopts) the extension under `<build_dir>/<digest12>`, the digest covering the source bytes, the flags,
the torch/CUDA versions and the device capability, so a `.so` built from another source set is never adopted (the rule of
`commit.committer.native_jit`, which this layer may not import); `ext()` is the loaded extension.
"""
from __future__ import annotations

import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "pous_native.cu")
CUDA_CFLAGS = ("-O3", "-lineinfo")
_LOADED: list = []


def source_digest(torch_version: str, cuda_version: str, arch: str) -> str:
    h = hashlib.sha256()
    with open(SOURCE, "rb") as f:
        h.update(f.read())
    h.update(("|".join(CUDA_CFLAGS) + f"|torch={torch_version}|cuda={cuda_version}|sm={arch}").encode())
    return h.hexdigest()


def load(build_dir: str):
    if _LOADED:
        return _LOADED[0]
    import torch
    from torch.utils.cpp_extension import load as jit_load

    major, minor = torch.cuda.get_device_capability()
    build = os.path.join(build_dir, source_digest(torch.__version__, str(torch.version.cuda), f"{major}.{minor}")[:12])
    os.makedirs(build, exist_ok=True)
    _LOADED.append(jit_load(name="verity_pous_native", sources=[SOURCE], build_directory=build, extra_cflags=["-O3"],
                            extra_cuda_cflags=list(CUDA_CFLAGS), verbose=False))
    return _LOADED[0]


def ext():
    if not _LOADED:
        raise RuntimeError("pous_native is not loaded: call native.load(build_dir) first")
    return _LOADED[0]
