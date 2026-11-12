"""circuit-check's bindings of `verity.definitions.registry.fp8_ds_mla_upconvert`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The row read back at GLM-5.3's statics (512 NoPE values under 4 scales, 64 RoPE words: the 656-byte row,
    `FP8_DS_MLA_GLM53`) and a small row (one scale, 8 RoPE words)."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import fp8_ds_mla_upconvert as U

    return [
        ("Fp8DsMlaUpconvertRow_v1:1", lambda: bind(U.Fp8DsMlaUpconvertRow, **U.FP8_DS_MLA_GLM53)),
        ("Fp8DsMlaUpconvertRow_v1:2", lambda: bind(U.Fp8DsMlaUpconvertRow, DN=U.QT, DR=8)),
    ]
