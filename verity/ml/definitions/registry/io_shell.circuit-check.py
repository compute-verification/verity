"""circuit-check's bindings of `verity.ml.definitions.registry.io_shell`'s `TokenFeed_v1`, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`TokenFeed_v1` at a small class (4 prompt ids) and at the class of record's (1,024), each the first position's and
    every other's."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import io_shell as X

    return [(f"TokenFeed_v1:{NP}:{int(first)}", lambda NP=NP, first=first: bind(X.TokenFeed, NP=NP, FIRST=first))
            for NP in (4, 1024) for first in (True, False)]
