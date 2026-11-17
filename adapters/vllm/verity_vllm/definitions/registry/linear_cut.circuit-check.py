"""circuit-check's bindings of `verity_vllm.definitions.registry.linear_cut`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """Every shape Llama-3.2-1B's cut linears take (`window_cut.linears`): the A row at qkv and gate_up (K = 2,048 from one
    RMSNorm row), at o_proj (K = 2,048 read as its 32 heads' rows) and at down_proj (K = 8,192 from one SiluMul row), and
    at one row of a small K; y at qkv (N = 3,072 in rows of 64), at o_proj's and down_proj's (N = 2,048 in rows of 64) and
    at gate_up's, the widest (N = 16,384 in rows of 256); each word Definition and its σ."""
    from verity.circuits.defs import bind
    from verity_vllm.definitions.registry import linear_cut as LC

    return [
        ("LinearARow_v3", lambda: bind(LC.LinearARow, K=2048, R=32)),
        ("LinearARow_v3/qkv", lambda: bind(LC.LinearARow, K=2048, R=1)),
        ("LinearARow_v3/down", lambda: bind(LC.LinearARow, K=8192, R=1)),
        ("LinearARow_v3/row", lambda: bind(LC.LinearARow, K=64, R=1)),
        ("LinearOut_v3", lambda: bind(LC.LinearOut, N=2048, W=64)),
        ("LinearOut_v3/qkv", lambda: bind(LC.LinearOut, N=3072, W=64)),
        ("LinearOut_v3/wide", lambda: bind(LC.LinearOut, N=16384, W=256)),
        ("LinearARow_v4", lambda: bind(LC.LinearARowV4, K=2048, R=32)),
        ("LinearARow_v4/qkv", lambda: bind(LC.LinearARowV4, K=2048, R=1)),
        ("LinearARow_v4/down", lambda: bind(LC.LinearARowV4, K=8192, R=1)),
        ("LinearARow_v4/row", lambda: bind(LC.LinearARowV4, K=64, R=1)),
        ("LinearOut_v4", lambda: bind(LC.LinearOutV4, N=2048, W=64)),
        ("LinearOut_v4/qkv", lambda: bind(LC.LinearOutV4, N=3072, W=64)),
        ("LinearOut_v4/wide", lambda: bind(LC.LinearOutV4, N=16384, W=256)),
    ]
