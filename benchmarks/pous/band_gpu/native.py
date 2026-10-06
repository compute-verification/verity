"""The JIT build of `pous_native.cu` (the batched SHAKE256 kernel and the audit responder), keyed by what it compiles.

`load(build_dir)` builds (or adopts) the extension under `<build_dir>/<digest12>`, the digest covering the source bytes, the flags,
the torch/CUDA versions, the release of the nvcc that compiles it (`nvcc_release`) and the arch list (`arch_list`: always the
device's capability, 12.0 on an sm_120), so a `.so` built from another source set or by another toolkit is never adopted (the
rule of `commit.committer.native_jit`, which this layer may not import), compiling a copy of the source staged in that directory
(`staged`) so that a tree at another commit adopts the same build; `ext()` is the loaded extension.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "pous_native.cu")
CUDA_CFLAGS = ("-O3", "-lineinfo")
_LOADED: list = []


def source_digest(torch_version: str, cuda_version: str, nvcc: str, arch: str) -> str:
    h = hashlib.sha256()
    with open(SOURCE, "rb") as f:
        h.update(f.read())
    h.update(("|".join(CUDA_CFLAGS) + f"|torch={torch_version}|cuda={cuda_version}|nvcc={nvcc}|sm={arch}").encode())
    return h.hexdigest()


def nvcc_release(cuda_home: str | None) -> str:
    """The release of the nvcc torch's JIT build runs (`<cuda_home>/bin/nvcc`), `"none"` when there is none."""
    nvcc = os.path.join(cuda_home, "bin", "nvcc") if cuda_home else "nvcc"
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "none"
    return next((line.split("release", 1)[1].split(",")[0].strip() for line in out.splitlines() if "release" in line), "none")


def arch_list(device_arch: str, given: str | None) -> str:
    """The `TORCH_CUDA_ARCH_LIST` the extension is built for: `given` when it names this device's capability, else the device's
    alone (torch's own default clamps it to the newest arch its wheel was built for, so a device newer than the wheel, such
    as an sm_120 under a cu126 wheel, would get only older code)."""
    named = {a.removesuffix("+PTX") for a in (given or "").replace(";", " ").split()}
    return given if given and device_arch in named else device_arch


def load(build_dir: str):
    if _LOADED:
        return _LOADED[0]
    import torch
    from torch.utils import cpp_extension
    from torch.utils.cpp_extension import load as jit_load

    major, minor = torch.cuda.get_device_capability()
    arches = arch_list(f"{major}.{minor}", os.environ.get("TORCH_CUDA_ARCH_LIST"))
    os.environ["TORCH_CUDA_ARCH_LIST"] = arches
    nvcc = nvcc_release(cpp_extension.CUDA_HOME)
    build = os.path.join(build_dir, source_digest(torch.__version__, str(torch.version.cuda), nvcc, arches)[:12])
    _LOADED.append(jit_load(name="verity_pous_native", sources=[staged(build)], build_directory=build, extra_cflags=["-O3"],
                            extra_cuda_cflags=list(CUDA_CFLAGS), verbose=False))
    return _LOADED[0]


def staged(build: str) -> str:
    """`<build>/src/pous_native.cu`, copied once: torch's build.ninja names its sources' paths, and a research run's tree lives
    under its commit (`src/<sha>/`), so compiling from the tree would rebuild for every commit; the digest keying `build`
    covers the source's bytes, so a copy already there is this one."""
    path = os.path.join(build, "src", os.path.basename(SOURCE))
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}"
        shutil.copyfile(SOURCE, tmp)
        os.replace(tmp, path)
    return path


def ext():
    if not _LOADED:
        raise RuntimeError("pous_native is not loaded: call native.load(build_dir) first")
    return _LOADED[0]
