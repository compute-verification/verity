"""circuit-check's bindings of `verity.definitions.registry.pearl_c_tile`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """TileRead at a 2 x 2 tile of K = 160 on the H100's atom (two promotion windows, the last short) and a 1 x 2 tile of one
    whole window on sm_120's, each reaching its cell and chain; the unscale at 2 x 2 with and without the bias; each word
    Definition and its σ."""
    from verity.circuits.defs import bind
    from verity.definitions.ops.boolean import gemm as BG
    from verity.definitions.ops.prims import BlackwellE4m3QmmaDot32, HopperE4m3QgmmaDot32
    from verity.definitions.registry import pearl_c_tile as PT

    return [
        ("PearlCTileRead_v1", lambda: bind(PT.PearlCTileRead, K=160, TM=2, TN=2, DOT=HopperE4m3QgmmaDot32, G=4)),
        ("PearlCTileRead_v1/blackwell", lambda: bind(PT.PearlCTileRead, K=128, TM=1, TN=2, DOT=BlackwellE4m3QmmaDot32, G=4)),
        ("PearlCUnscale_v1", lambda: bind(PT.PearlCUnscale, TM=2, TN=2, BIAS=False)),
        ("PearlCUnscale_v1/bias", lambda: bind(PT.PearlCUnscale, TM=2, TN=2, BIAS=True)),
        ("PearlCTileRead_v2", lambda: bind(PT.PearlCTileReadV2, K=160, TM=2, TN=2, DOT=BG.HopperE4m3QgmmaDot32, G=4)),
        ("PearlCTileRead_v2/blackwell", lambda: bind(PT.PearlCTileReadV2, K=128, TM=1, TN=2, DOT=BG.BlackwellE4m3QmmaDot32, G=4)),
        ("PearlCUnscale_v2", lambda: bind(PT.PearlCUnscaleV2, TM=2, TN=2, BIAS=False)),
        ("PearlCUnscale_v2/bias", lambda: bind(PT.PearlCUnscaleV2, TM=2, TN=2, BIAS=True)),
    ]
