"""How the kernels tests build a test source: by the host's C++ compiler (`host_lib`), and by nvcc for the target of the first GPU's
card file (`device_lib`), the device build cached by the source, the headers it includes, the flags and nvcc's version under
``~/.cache/verity/kernels``, so a GPU job builds before it leases a GPU and the leased test only runs what it built."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from functools import cache
from pathlib import Path

from verity.device import compute_capability, gpu_card_reporting

#: the GPU card files (`verity.device`), whose ``arch`` is the target the device library is built for
CARD_FILES = Path(__file__).resolve().parents[4] / "calibration/hardware/cards"
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
def gpu_name() -> str | None:
    """The name the first GPU reports to nvidia-smi (the card files' ``driver`` values), or None without nvcc or a GPU."""
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None:
        return None
    r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                       check=False)
    names = r.stdout.splitlines() if r.returncode == 0 else []
    return names[0].strip() if names else None


def arch(name: str) -> str:
    """The target a library is built for on the device reporting ``name``: its card file's ``arch``
    (`verity.device.gpu_card_reporting`) without ``sm_``, such as ``90a``; a device no card file names is refused."""
    card = gpu_card_reporting(name, directory=CARD_FILES)
    compute_capability(card)
    return card["arch"].removeprefix("sm_")


def device_lib(source: Path, headers: tuple[Path, ...], flags: tuple[str, ...] = ()) -> Path:
    """source built by nvcc with CUDA_FLAGS and flags for the first GPU, from the cache where it is there."""
    name = gpu_name()
    assert name is not None, "no nvcc or no GPU"
    target = arch(name)
    full = (*CUDA_FLAGS, f"-gencode=arch=compute_{target},code=sm_{target}", *flags)
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
