"""circuit-check's bindings of `verity.ml.definitions.registry.io_shell`'s `TokenFeed_v1`, `SlotRow_v1` and `SlotMask_v1`, in
`circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`TokenFeed_v1` at a small class (4 prompt ids) and at the class of record's (1,024), each the first position's and
    every other's; `SlotRow_v1` at both; and `SlotMask_v1`."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import io_shell as X

    return [("TokenFeed_v1:4:1", lambda: bind(X.TokenFeed, NP=4, FIRST=True)),
            ("TokenFeed_v1:4:0", lambda: bind(X.TokenFeed, NP=4, FIRST=False)),
            ("TokenFeed_v1:1024:1", lambda: bind(X.TokenFeed, NP=1024, FIRST=True)),
            ("TokenFeed_v1:1024:0", lambda: bind(X.TokenFeed, NP=1024, FIRST=False)),
            ("SlotRow_v1:4", lambda: bind(X.SlotRow, NP=4)),
            ("SlotRow_v1:1024", lambda: bind(X.SlotRow, NP=1024)),
            ("SlotMask_v1", lambda: bind(X.SlotMask))]
