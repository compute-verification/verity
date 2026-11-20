"""The engine facts a sealed window records (`engine_facts_of_record`): the evaluators' configuration of record and the cuBLAS
libraries the process mapped."""
from __future__ import annotations

import ctypes
from collections.abc import Callable
from pathlib import Path
from typing import Any

from verity_vllm import settings

__all__ = ["engine_facts_of_record"]


ENGINE_FACTS_RULE = ("the evaluators' semantics are the engine's configuration of record (on cc 9.0 the Gemm_v2 Hopper chain "
                     "equals cuBLASLt only under VLLM_BATCH_INVARIANT=1, which disables split-K by CUBLAS_WORKSPACE_CONFIG=:16:8 / "
                     "CUBLASLT_WORKSPACE_SIZE=1 / allow_bf16_reduced_precision_reduction False; on cc 8.x it installs the Triton persistent "
                     "matmul; without batch invariance on cc 9.0 the cuBLAS GEMMs are the route pin's, which holds only where `cublas` is "
                     "bf16_gemm_check.ROUTE_CONDITIONS) -- MEASURED in the process the replay ran in (the Commit stage runs in the engine's "
                     "process), never declared; a fact this process cannot see is 'unavailable: <why>' by name (a laptop re-read of a stored "
                     "record has no engine)")
#: the workspace variables cuBLAS and cuBLASLt size their workspaces by, which move their routes
CUBLAS_ENV = ("CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE")


def _unavailable(e: BaseException) -> str:
    return f"unavailable: {type(e).__name__}: {e}"[:200]


def _mapped_library(prefix: str, maps: str) -> str:
    """The one file named `prefix`* that `maps` (a /proc/<pid>/maps) lists: the library torch loaded, never the linker's search order,
    which can name another CUDA install's library than the one the GEMMs run."""
    files = {f[-1] for f in (ln.split() for ln in Path(maps).read_text().splitlines()) if len(f) >= 6}
    got = sorted(f for f in files if f.rsplit("/", 1)[-1].startswith(prefix))
    if len(got) != 1:
        raise LookupError(f"{len(got)} {prefix}* mapped {got}")
    return got[0]


def cublas_versions(maps: str = "/proc/self/maps") -> dict[str, Any]:
    """The cuBLAS and cuBLASLt this process maps, by their public API (`cublasGetProperty`'s major, minor and patch level, the
    number `cublasGetVersion` returns; `cublasLtGetVersion`), without a handle or a device."""
    out: dict[str, Any] = {}
    try:
        path = _mapped_library("libcublas.so", maps)
        lib = ctypes.CDLL(path)
        lib.cublasGetProperty.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        parts = [ctypes.c_int() for _ in range(3)]                                  # MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
        if any(lib.cublasGetProperty(i, ctypes.byref(p)) for i, p in enumerate(parts)):
            raise RuntimeError("cublasGetProperty returned an error status")
        out["cublas_library"], out["cublas_version"] = path, parts[0].value * 10000 + parts[1].value * 100 + parts[2].value
    except Exception as e:                                                          # noqa: BLE001 -- named, never guessed
        out["cublas_library"] = out["cublas_version"] = _unavailable(e)
    try:
        lt = ctypes.CDLL(_mapped_library("libcublasLt.so", maps))
        lt.cublasLtGetVersion.restype = ctypes.c_size_t
        out["cublaslt_version"] = int(lt.cublasLtGetVersion())
    except Exception as e:                                                          # noqa: BLE001
        out["cublaslt_version"] = _unavailable(e)
    return out


def _cublas_facts(env: dict[str, Any]) -> dict[str, Any]:
    """What decides this process's cuBLAS routes beside the launch's shape, each read as the route pin's scan read it (the keys of
    `bf16_gemm_check.ROUTE_CONDITIONS`), with the cuBLASLt version and the library's path beside them."""
    out: dict[str, Any] = {}
    try:
        import torch  # type: ignore
    except Exception as e:                                                          # noqa: BLE001
        return {"torch": _unavailable(e)} | cublas_versions() | {"env": env}
    m = torch.backends.cuda.matmul
    reads: dict[str, Callable[[], Any]] = {
        "device": torch.cuda.get_device_name, "compute_capability": lambda: list(torch.cuda.get_device_capability()),
        "multiprocessors": lambda: int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count),
        "torch": lambda: str(torch.__version__),
        "allow_fp16_reduced_precision_reduction": lambda: repr(m.allow_fp16_reduced_precision_reduction),
        "allow_bf16_reduced_precision_reduction": lambda: repr(m.allow_bf16_reduced_precision_reduction),
        "preferred_blas_library": lambda: str(torch.backends.cuda.preferred_blas_library())}
    for k, read in reads.items():
        try:
            out[k] = read()
        except Exception as e:                                                      # noqa: BLE001
            out[k] = _unavailable(e)
    return out | cublas_versions() | {"env": env}


def engine_facts_of_record() -> dict[str, Any]:
    """The batch-invariant configuration beside the evaluator strings: what the replay's pins were conditioned on;
    `cublas`, what the route pin's routes were conditioned on (`_cublas_facts`)."""
    facts: dict[str, Any] = {k: settings.raw(k) for k in ("VLLM_BATCH_INVARIANT", *CUBLAS_ENV)}
    try:
        from vllm.model_executor.determinism import batch_invariant as bi  # type: ignore
        facts["vllm_batch_invariant_mode"] = bool(bi.vllm_is_batch_invariant()) if hasattr(bi, "vllm_is_batch_invariant") else "unavailable: no vllm_is_batch_invariant()"
    except Exception as e:                                                          # noqa: BLE001 -- named, never guessed
        facts["vllm_batch_invariant_mode"] = f"unavailable: {type(e).__name__}"
    try:
        import torch  # type: ignore
        bf16 = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction   # (allow, allow for split-K) once set as a pair
        facts["torch_allow_bf16_reduced_precision_reduction"] = bool(bf16[0] if isinstance(bf16, tuple) else bf16)
        try:
            facts["torch_preferred_blas_library"] = str(torch.backends.cuda.preferred_blas_library()).split(".")[-1]
        except Exception as e:                                                      # noqa: BLE001
            facts["torch_preferred_blas_library"] = f"unavailable: {type(e).__name__}"
        facts["compute_capability"] = list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else "unavailable: no CUDA device"
    except Exception as e:                                                          # noqa: BLE001
        facts["torch_allow_bf16_reduced_precision_reduction"] = facts["torch_preferred_blas_library"] = facts["compute_capability"] = f"unavailable: {type(e).__name__}"
    facts["cublas"] = _cublas_facts({k: facts[k] for k in CUBLAS_ENV})
    facts["rule"] = ENGINE_FACTS_RULE
    return facts


