"""``sha512.cuh`` against hashlib's SHA-512: every compression form it holds digests whole messages bit for bit, the FIPS
180-4 examples and every length to 300 bytes (one, two and three chunks, and both sides of each padding boundary), under
each setting of its issue knobs. ``sha512_forms.cu`` runs them, built by the host's C++ compiler and, where nvcc and a GPU
are present, on the device.

The device builds are cached by their source, header, flags and nvcc's version under ``~/.cache/verity/kernels``, so a GPU
job builds them before it leases a GPU (``cuda_lib`` for each of ``KNOBS``) and the leased test only runs them."""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
from functools import cache
from pathlib import Path

import pytest

from verity.ml.kernels import SHA512

SOURCE = Path(__file__).resolve().with_name("sha512_forms.cu")
CACHE = Path.home() / ".cache" / "verity" / "kernels"

FIPS = {
    b"": "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce"
         "47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
    b"abc": "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
            "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
    b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu":
        "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018"
        "501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909",
}
MESSAGES = [*FIPS, *(bytes((131 * i + n) % 256 for i in range(n)) for n in range(301))]
#: the defaults; every add as 32-bit pairs, every sigma on the FMA pipe, straight-line; sha::compress as one of sx's forms
KNOBS = {"defaults": (), "pairs-fma-straight": ("-DSHA_ADD=1", "-DSHA_ROT=15", "-DSHA_ROLL=0"),
         "sx": ("-DSHA_SX", f"-DSX_V={0xe4}", "-DSX_ROLL=2")}
HOST_FLAGS = ("-x", "c++", "-O1", "-std=c++17", "-fPIC", "-shared", "-Wall", "-Werror")
CUDA_FLAGS = ("-x", "cu", "-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-shared")

vp, u64 = ctypes.c_void_p, ctypes.c_uint64


def _compile(tool: str, flags: tuple[str, ...], out: Path) -> None:
    r = subprocess.run([tool, *flags, "-o", str(out), str(SOURCE)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, f"{tool} failed:\n{r.stderr[-4000:]}"


def _load(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.sha512_form_name.argtypes, lib.sha512_form_name.restype = [ctypes.c_int], ctypes.c_char_p
    lib.sha512_digest.argtypes, lib.sha512_digest.restype = [ctypes.c_int, ctypes.c_char_p, u64, vp], None
    return lib


def _gpu_arch() -> str | None:
    """The first GPU's compute capability as nvcc's sm number (``"90"``), or None without nvcc or a GPU."""
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None:
        return None
    r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True,
                       check=False)
    caps = r.stdout.split() if r.returncode == 0 else []
    return caps[0].replace(".", "") if caps else None


@cache
def cuda_lib(knobs: str) -> ctypes.CDLL:
    arch = _gpu_arch()
    assert arch is not None, "no nvcc or no GPU"
    flags = (*CUDA_FLAGS, f"-gencode=arch=compute_{arch},code=sm_{arch}", *KNOBS[knobs])
    version = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True).stdout
    key = hashlib.sha256(b"\0".join([SOURCE.read_bytes(), SHA512.read_bytes(), " ".join(flags).encode(), version.encode()]))
    lib = CACHE / key.hexdigest()[:16] / "sha512_forms_cuda.so"
    if not lib.exists():
        lib.parent.mkdir(parents=True, exist_ok=True)
        tmp = lib.with_suffix(f".{os.getpid()}.tmp")
        _compile("nvcc", flags, tmp)
        tmp.replace(lib)
    dll = _load(lib)
    dll.sha512_digest_cuda.argtypes = [ctypes.c_int, ctypes.c_char_p, u64, vp, vp, ctypes.c_int, vp]
    dll.sha512_digest_cuda.restype = ctypes.c_int
    return dll


@pytest.fixture(scope="module", params=list(KNOBS))
def host(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    cxx = next((c for c in ("g++", "c++", "clang++") if shutil.which(c)), None)
    assert cxx is not None, "no C++ compiler on the path"
    lib = tmp_path_factory.mktemp("host") / "sha512_forms.so"
    _compile(cxx, (*HOST_FLAGS, *KNOBS[request.param]), lib)
    return _load(lib)


def _digest(lib: ctypes.CDLL, form: int, m: bytes) -> bytes:
    out = ctypes.create_string_buffer(64)
    lib.sha512_digest(form, m, len(m), out)
    return out.raw


def test_the_package_names_the_header_the_forms_include() -> None:
    assert SHA512.resolve() == SOURCE.parents[1] / "sha512.cuh"
    assert '#include "../sha512.cuh"' in SOURCE.read_text()


def test_hashlib_gives_the_fips_examples() -> None:
    assert {m: hashlib.sha512(m).hexdigest() for m in FIPS} == FIPS


def test_the_tables_are_the_schedule_and_the_iv(host: ctypes.CDLL) -> None:
    assert host.sha512_tables() == 0


def test_every_form_on_the_host_is_sha512(host: ctypes.CDLL) -> None:
    want = [hashlib.sha512(m).digest() for m in MESSAGES]
    bad = {host.sha512_form_name(f).decode() if f >= 0 else "sha::digest": n for f in range(-1, host.sha512_forms())
           if (n := sum(_digest(host, f, m) != w for m, w in zip(MESSAGES, want)))}
    assert not bad, bad


@pytest.mark.slow
@pytest.mark.skipif(_gpu_arch() is None, reason="no nvcc or no CUDA device")
@pytest.mark.parametrize("knobs", list(KNOBS))
def test_every_form_on_the_device_is_sha512(knobs: str) -> None:
    """Slow for its nvcc build of every form, where the cache has none: the device's pipe choices, which the host lacks."""
    lib = cuda_lib(knobs)
    blob, n = b"".join(MESSAGES), len(MESSAGES)
    lens = (u64 * n)(*map(len, MESSAGES))
    offs = (u64 * n)(*(sum(map(len, MESSAGES[:i])) for i in range(n)))
    want = [hashlib.sha512(m).digest() for m in MESSAGES]
    bad = {}
    for f in range(lib.sha512_forms()):
        out = ctypes.create_string_buffer(64 * n)
        assert lib.sha512_digest_cuda(f, blob, len(blob), offs, lens, n, out) == 0
        if n_bad := sum(out.raw[64 * i:64 * i + 64] != w for i, w in enumerate(want)):
            bad[lib.sha512_form_name(f).decode()] = n_bad
    assert not bad, bad
