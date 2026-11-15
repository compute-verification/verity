"""circuit-check's bindings of `verity.definitions.registry.boolean_dsa_ds_mla`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`DsMlaFp8Tile_v2` at one warp's tile (32) and GLM-5.3's (128); `DsaLatentRowDsMla_v2` at the small row (256 in four tiles of
    64, 8 rotary values) and at GLM-5.3's (512 in four tiles of 128, 64 rotary values)."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_dsa_ds_mla

    return [
        ("DsMlaFp8Tile_v2/warp", lambda: bind(boolean_dsa_ds_mla.DsMlaFp8TileV2, T=32)),
        ("DsMlaFp8Tile_v2/glm53", lambda: bind(boolean_dsa_ds_mla.DsMlaFp8TileV2, T=128)),
        ("DsaLatentRowDsMla_v2/small", lambda: bind(boolean_dsa_ds_mla.DsaLatentRowDsMlaV2, KVL=256, R=8, TILE=64, EPS=1e-5)),
        ("DsaLatentRowDsMla_v2/glm53", lambda: bind(boolean_dsa_ds_mla.DsaLatentRowDsMlaV2, KVL=512, R=64, TILE=128, EPS=1e-5)),
    ]
