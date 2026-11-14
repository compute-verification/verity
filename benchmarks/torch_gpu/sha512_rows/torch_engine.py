"""The SHA-512 rows engine (`verity/commitments/cuda/sha512_rows/engine.py`) with the benchmarks' settings: its build
directory and default variant, read from `benchmarks/torch_gpu/settings.py` and passed in, since the engine reads none.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from verity_config.settings import of

from verity.commitments.cuda.sha512_rows import engine as E

settings = of(__file__)
BUILD = Path(settings.SHA512_ROWS_BUILD or Path.home() / ".cache" / "verity" / "sha512_rows")
VARIANT: str = settings.SHA512_ROWS_V


class Engine(E.Engine):
    """`engine.Engine` building under ``BUILD``, its kernels but the row kernel on ``VARIANT`` unless given."""

    def __init__(self, device: Any, arch: str | None = None, key: bytes = E.KEY, default: str = VARIANT, *,
                 build: Path = BUILD) -> None:
        super().__init__(device, arch, key, default, build=build)


def host_lib() -> Any:
    return E.host_lib(BUILD)


def device_lib(arch: str, default: str = VARIANT) -> Any:
    return E.device_lib(BUILD, arch, default)
