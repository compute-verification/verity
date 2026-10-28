"""The sm_120 FP8 / FP4 GEMM coordinates as the program a class sweep stages: one row of N coordinates of
`GemmCoordinate{E4m3,Nvf4,Mxf4}_v1{K, DOT}` over one activation row, as `Gemm_v2` batches `GemmCoordinate_v2` for BF16, so the
statement commits the activation row (and its scale bytes) once for every coordinate, and each weight row (and its scales) once.

    python -m verity_flock.class_statement --program-module verity/ml/flock/pod/gemm_fp.py:e4m3_K2048_N2048 ...

The row is an unregistered composite (``GemmRow[<coordinate id>]{N}``): core has no Gemm-level FP8 / FP4 Definition, and the
proof block, the class the statement proves, is the registered coordinate.
"""
from __future__ import annotations

import re

#: dtype -> (coordinate composite, binding record) in verity.ml.circuits.ops.gemm
DTYPES = {"e4m3": ("GemmCoordinateE4m3", "GEMM_SM120_E4M3"), "nvf4": ("GemmCoordinateNvf4", "GEMM_SM120_NVF4"),
          "mxf4": ("GemmCoordinateMxf4", "GEMM_SM120_MXF4")}


def coordinate(dtype: str, K: int):
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import gemm as G

    comp, rec = DTYPES[dtype]
    return bind(getattr(G, comp), K=K, **getattr(G, rec))


def program(dtype: str, K: int, N: int):
    """A Program whose one Call is N coordinates over one activation row: the activation (and its scales) broadcast, the weight rows
    (and theirs) one per coordinate."""
    from verity.ml.circuits.defs import CompositeDefinition
    from verity.ml.circuits.types import Array
    from verity_flock.partition_units import one_call

    fn = coordinate(dtype, K)
    params, ret = fn.signature()
    axes = tuple(None if name in ("x", "sx") else 0 for name, _ in params)
    row_params = tuple((name, t if ax is None else Array(N, t)) for (name, t), ax in zip(params, axes))
    row = CompositeDefinition(f"GemmRow[{fn.id}]{{N={N}}}", 1, (), lambda S: (row_params, Array(N, ret)),
                              lambda B, S, *xs: B.batch(fn, *xs, axes=axes), register=False)
    from verity.ml.circuits.defs import bind
    return one_call(bind(row))


def __getattr__(name: str):
    m = re.fullmatch(r"(e4m3|nvf4|mxf4)_K(\d+)_N(\d+)", name)
    if not m:
        raise AttributeError(name)
    return lambda: program(m[1], int(m[2]), int(m[3]))
