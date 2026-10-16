"""circuit-check's bindings of `verity_vllm.program.registry.dsv4_compressor`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions at DeepSeek-V4-Pro's statics (`dsv4_compressor.DSV4_COMPRESSOR`): the ratio-4 column over 8 rows
    and over the 4 at position 3, the ratio-128 column over 8 warps of 16, each kernel's RMSNorm and row at head 512, one group
    of 64, the rotary group, and both compressors' rows."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import dsv4_compressor as C

    S = C.DSV4_COMPRESSOR
    row = {"D": S["D"], "QB": S["QB"], "EPS": S["EPS"]}
    return [
        ("Dsv4MlaC4Column_v1:1", lambda: bind(C.Dsv4MlaC4Column, ROWS=S["ROWS"])),
        ("Dsv4MlaC4Column_v1:2", lambda: bind(C.Dsv4MlaC4Column, ROWS=4)),
        ("Dsv4MlaC128Column_v1", lambda: bind(C.Dsv4MlaC128Column, W=S["W"], R=S["R"])),
        ("Dsv4MlaRmsNorm_v1:1", lambda: bind(C.Dsv4MlaRmsNorm, RATIO=4, **row)),
        ("Dsv4MlaRmsNorm_v1:2", lambda: bind(C.Dsv4MlaRmsNorm, RATIO=128, **row)),
        ("Dsv4MlaQuantGroup_v1", lambda: bind(C.Dsv4MlaQuantGroup, QB=S["QB"])),
        ("Dsv4MlaRope_v1", lambda: bind(C.Dsv4MlaRope, ROPE=S["QB"])),
        ("Dsv4MlaNormRopeQuant_v1:1", lambda: bind(C.Dsv4MlaNormRopeQuant, RATIO=4, **row)),
        ("Dsv4MlaNormRopeQuant_v1:2", lambda: bind(C.Dsv4MlaNormRopeQuant, RATIO=128, **row)),
        ("Dsv4MlaCompressC4_v1:1", lambda: bind(C.Dsv4MlaCompressC4, ROWS=S["ROWS"], **row)),
        ("Dsv4MlaCompressC4_v1:2", lambda: bind(C.Dsv4MlaCompressC4, ROWS=4, **row)),
        ("Dsv4MlaCompressC128_v1", lambda: bind(C.Dsv4MlaCompressC128, W=S["W"], R=S["R"], **row)),
    ]
