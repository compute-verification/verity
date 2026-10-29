"""sha512_rows' settings: the variables engine.py reads, declared once (`verity_config.settings`).

It owns ``SHA512_ROWS_``: a variable with it that nothing here declares is refused when the registry loads.  gpu.sh's
own variables (LEASE_MIN, PY, BENCH_ARGS) are the shell's.
"""
from __future__ import annotations

from verity_config.settings import VALUE, WHERE, Registry, Setting as S

REGISTRY = Registry(
    __file__, ("SHA512_ROWS_",),
    S("SHA512_ROWS_BUILD", str, None, WHERE, "the build cache (default ~/.cache/verity/sha512_rows): builds are keyed by a "
                                             "hash of their sources, flags and compiler, so where they live changes nothing"),
    S("SHA512_ROWS_V", str, "0000", VALUE, "the compression variant of every kernel but the row kernel (a name, e.g. 2211)"))
