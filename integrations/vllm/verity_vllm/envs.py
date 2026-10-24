"""The environment variables verity_vllm reads, each named once with its type, its default and what it does.

Read one as ``envs.NAME`` after ``from verity_vllm import envs`` (vLLM's ``vllm/envs.py`` pattern). Each access reads the
process environment at that moment, so a value set after import (a test's ``monkeypatch.setenv``, a parent before it
starts a child) is seen; ``from verity_vllm.envs import NAME`` would freeze it at import and is refused
(``tests/lint/test_p14_settings.py``). A variable is still an undeclared input (P7): prefer an argument, and read here
only what must stay a variable. Writes (a child's environment, vLLM's pins in ``engine/env.py``) are not settings.

This module imports nothing from the package, so every layer may read it.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    CUDA_VISIBLE_DEVICES: str | None = None
    CXX: str = "g++"
    PY: str | None = None
    RESEARCH_RUN_ID: str | None = None
    TORCHINDUCTOR_CACHE_DIR: str | None = None
    TRITON_CACHE_DIR: str | None = None
    VERITY_CENSUS_GPU_UTIL: str | None = None
    VERITY_DENSE_THREADS: str | None = None
    VERITY_FA2_COMPACT: bool = True
    VERITY_FA2_TAP_CAP_MB: int | None = None
    VERITY_FA2_ZERO: bool = True
    VERITY_LEAFHASH_BUILD: str | None = None
    VERITY_NUMERICS_BUILD_DIR: str | None = None
    VERITY_PINNED_EXACT: bool = True
    VERITY_REPLAY_DUMP_DIR: str | None = None
    VERITY_REPLAY_DUMP_FAMILIES: tuple[str, ...] = ()
    VERITY_REPLAY_DUMP_MAX: int = 1024
    VERITY_TIMELINE: str | None = None
    VERITY_WEIGHTS_HASH: str | None = None
    XDG_CACHE_HOME: str | None = None


def _on_unless_off(name: str) -> bool:
    """On unless the variable is ``0``, ``false`` or ``no`` (after stripping); unset is on."""
    return os.environ.get(name, "1").strip() not in ("0", "false", "no")


environment_variables: dict[str, Callable[[], Any]] = {
    # The GPUs this process may use (CUDA's own variable); read to name the device a target-family check probed.
    "CUDA_VISIBLE_DEVICES": lambda: os.environ.get("CUDA_VISIBLE_DEVICES"),
    # The C++ compiler the numerics twins build with (program/kernels/_jit.py), part of their build cache's key.
    "CXX": lambda: os.environ.get("CXX", "g++"),
    # The pod's Python interpreter whose torch and vLLM versions a research Tool's derivation records; unset: the default.
    "PY": lambda: os.environ.get("PY"),
    # The research run this process runs in (set by `research run`), recorded as a result's run_id when none is given.
    "RESEARCH_RUN_ID": lambda: os.environ.get("RESEARCH_RUN_ID"),
    # Inductor's cache directory (torch's own variable); unset: torch's default. Where the compiled wrappers are read.
    "TORCHINDUCTOR_CACHE_DIR": lambda: os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
    # Triton's cache directory (Triton's own variable); unset: ~/.triton/cache. Where the kernel binaries are read.
    "TRITON_CACHE_DIR": lambda: os.environ.get("TRITON_CACHE_DIR"),
    # A census run's gpu_memory_utilization, when it shares the GPU with another engine; unset: the engine's default.
    "VERITY_CENSUS_GPU_UTIL": lambda: os.environ.get("VERITY_CENSUS_GPU_UTIL"),
    # How many threads the dense row kernels use (program/kernels/dense_rows.py, read at import); unset: all.
    "VERITY_DENSE_THREADS": lambda: os.environ.get("VERITY_DENSE_THREADS"),
    # FA2's hidden tap stages only the rows a multi-sequence launch wrote; off with 0, false or no.
    "VERITY_FA2_COMPACT": lambda: _on_unless_off("VERITY_FA2_COMPACT"),
    # The FA2 hidden tap's device buffer cap in MiB (empty: 0); unset: the tap's DEFAULT_TAP_CAP_MB.
    "VERITY_FA2_TAP_CAP_MB": lambda: (None if (v := os.environ.get("VERITY_FA2_TAP_CAP_MB")) is None
                                      else int(v.strip() or 0)),
    # FA2's hidden tap zeroes its buffer after each launch; off with 0, false or no.
    "VERITY_FA2_ZERO": lambda: _on_unless_off("VERITY_FA2_ZERO"),
    # The parent directory of the leaf-hash CUDA extension's JIT build; unset: the JIT's default.
    "VERITY_LEAFHASH_BUILD": lambda: os.environ.get("VERITY_LEAFHASH_BUILD"),
    # Where the numerics twins' JIT-built libraries live; unset: ${XDG_CACHE_HOME:-~/.cache}/verity/numerics/<digest>/.
    "VERITY_NUMERICS_BUILD_DIR": lambda: os.environ.get("VERITY_NUMERICS_BUILD_DIR"),
    # The pinned host pool hands out buffers of exactly the size asked; off only with 0 (read at import).
    "VERITY_PINNED_EXACT": lambda: os.environ.get("VERITY_PINNED_EXACT", "1") != "0",
    # Diagnostic: the directory a C2 replay writes each evaluated row's committed and evaluated words to; unset: off.
    "VERITY_REPLAY_DUMP_DIR": lambda: os.environ.get("VERITY_REPLAY_DUMP_DIR"),
    # The Definition families the replay dump writes, comma-separated; unset or empty: every family.
    "VERITY_REPLAY_DUMP_FAMILIES": lambda: tuple(f for f in os.environ.get("VERITY_REPLAY_DUMP_FAMILIES", "").split(",") if f),
    # How many files the replay dump writes at most.
    "VERITY_REPLAY_DUMP_MAX": lambda: int(os.environ.get("VERITY_REPLAY_DUMP_MAX", "1024")),
    # The file a stage appends its timeline spans to; unset or empty: no timeline.
    "VERITY_TIMELINE": lambda: os.environ.get("VERITY_TIMELINE"),
    # Where the weights' leaf hashes are computed: host, device or device-checked; unset: the caller's default.
    "VERITY_WEIGHTS_HASH": lambda: os.environ.get("VERITY_WEIGHTS_HASH"),
    # The XDG cache root (the freedesktop variable); unset or empty: ~/.cache.
    "XDG_CACHE_HOME": lambda: os.environ.get("XDG_CACHE_HOME"),
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}: declare the variable in verity_vllm/envs.py")


def __dir__() -> list[str]:
    return sorted(environment_variables)
