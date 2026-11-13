"""circuit-check's bindings of `verity.definitions.registry.dsa_ds_mla`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`DsaLatentRowDsMla_v1` at GLM-5.3's row (kv_c 512 in four tiles of 128, k_pe 64, eps 1e-5) and at a small one (256 in four
    tiles of 64, 8 rotary values), and `DsMlaFp8Tile_v1` at each tile width."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import dsa_ds_mla

    return [
        ("DsMlaFp8Tile_v1/glm53", lambda: bind(dsa_ds_mla.DsMlaFp8Tile, T=128)),
        ("DsMlaFp8Tile_v1/small", lambda: bind(dsa_ds_mla.DsMlaFp8Tile, T=64)),
        ("DsaLatentRowDsMla_v1/glm53", lambda: bind(dsa_ds_mla.DsaLatentRowDsMla, KVL=512, R=64, TILE=128, EPS=1e-5)),
        ("DsaLatentRowDsMla_v1/small", lambda: bind(dsa_ds_mla.DsaLatentRowDsMla, KVL=256, R=8, TILE=64, EPS=1e-5)),
    ]
