"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_dsv4_compressor`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions at small statics: the ratio-4 column over the 4 rows at position 3, the ratio-128 column over 2
    warps of 2, each kernel's RMSNorm and row at D = 16 in groups of 8, a group of 8, a rotary group of 4, and both compressors'
    rows at D = 16.  Every column and norm is over `targets.MAX_GATES` (each ex2 and rsqrt is a MUFU piece), so `select` keeps a
    family's smallest; the served widths are the word Definitions' (`dsv4_compressor.circuit-check.py`)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_dsv4_compressor as BC

    row = {"D": 16, "QB": 8, "EPS": 1e-6}
    return [
        ("Dsv4MlaC4Column_v2", lambda: bind(BC.Dsv4MlaC4ColumnV2, ROWS=4)),
        ("Dsv4MlaC128Column_v2", lambda: bind(BC.Dsv4MlaC128ColumnV2, W=2, R=2)),
        ("Dsv4MlaRmsNorm_v2:1", lambda: bind(BC.Dsv4MlaRmsNormV2, RATIO=4, **row)),
        ("Dsv4MlaRmsNorm_v2:2", lambda: bind(BC.Dsv4MlaRmsNormV2, RATIO=128, **row)),
        ("Dsv4MlaQuantGroup_v2", lambda: bind(BC.Dsv4MlaQuantGroupV2, QB=8)),
        ("Dsv4MlaRope_v2", lambda: bind(BC.Dsv4MlaRopeV2, ROPE=4)),
        ("Dsv4MlaNormRopeQuant_v2:1", lambda: bind(BC.Dsv4MlaNormRopeQuantV2, RATIO=4, **row)),
        ("Dsv4MlaNormRopeQuant_v2:2", lambda: bind(BC.Dsv4MlaNormRopeQuantV2, RATIO=128, **row)),
        ("Dsv4MlaCompressC4_v2", lambda: bind(BC.Dsv4MlaCompressC4V2, ROWS=4, **row)),
        ("Dsv4MlaCompressC128_v2", lambda: bind(BC.Dsv4MlaCompressC128V2, W=2, R=2, **row)),
    ]
