"""How the kernels tests build a test source: by the host's C++ compiler (`host_lib`), and by nvcc for the first GPU's compute
capability (`device_lib`), the device build cached by the source, the headers it includes, the flags and nvcc's version under
``~/.cache/verity/kernels``, so a GPU job builds before it leases a GPU and the leased test only runs what it built."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from functools import cache
from pathlib import Path

CACHE = Path.home() / ".cache" / "verity" / "kernels"
HOST_FLAGS = ("-x", "c++", "-O1", "-std=c++17", "-fPIC", "-shared", "-Wall", "-Werror")
CUDA_FLAGS = ("-x", "cu", "-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-shared")


def build(tool: str, flags: tuple[str, ...], source: Path, out: Path) -> None:
    r = subprocess.run([tool, *flags, "-o", str(out), str(source)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, f"{tool} failed:\n{r.stderr[-4000:]}"


def host_lib(source: Path, flags: tuple[str, ...], out: Path) -> Path:
    """source built into out by the host's C++ compiler with HOST_FLAGS and flags."""
    cxx = next((c for c in ("g++", "c++", "clang++") if shutil.which(c)), None)
    assert cxx is not None, "no C++ compiler on the path"
    build(cxx, (*HOST_FLAGS, *flags), source, out)
    return out


@cache
def gpu_arch() -> str | None:
    """The first GPU's compute capability as nvcc's sm number (``"90"``), or None without nvcc or a GPU."""
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None:
        return None
    r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True,
                       check=False)
    caps = r.stdout.split() if r.returncode == 0 else []
    return caps[0].replace(".", "") if caps else None


def device_lib(source: Path, headers: tuple[Path, ...], flags: tuple[str, ...] = ()) -> Path:
    """source built by nvcc with CUDA_FLAGS and flags for the first GPU, from the cache where it is there."""
    arch = gpu_arch()
    assert arch is not None, "no nvcc or no GPU"
    full = (*CUDA_FLAGS, f"-gencode=arch=compute_{arch},code=sm_{arch}", *flags)
    version = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True).stdout
    key = hashlib.sha256(b"\0".join([source.read_bytes(), *(h.read_bytes() for h in headers), " ".join(full).encode(),
                                     version.encode()]))
    lib = CACHE / key.hexdigest()[:16] / f"{source.stem}_cuda.so"
    if not lib.exists():
        lib.parent.mkdir(parents=True, exist_ok=True)
        tmp = lib.with_suffix(f".{os.getpid()}.tmp")
        build("nvcc", full, source, tmp)
        tmp.replace(lib)
    return lib
