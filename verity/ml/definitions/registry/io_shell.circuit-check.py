"""circuit-check's bindings of `verity.ml.definitions.registry.io_shell`'s `TokenFeed_v1`, `OutCollect_v1` and `SlotEntry_v1`,
in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`TokenFeed_v1` at a small class (4 prompt ids) and at the class of record's (1,024), each the first position's and
    every other's; `OutCollect_v1` at KO 2 and 64, the first position's and every other's; and
    `SlotEntry_v1` at the small class (KO 2, 4 prompt ids) with no stop id (TINY's) and at the class of record's (KO 64,
    1,024) with Llama-3.2-1B's (128001)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import io_shell as X

    return [("TokenFeed_v1:4:1", lambda: bind(X.TokenFeed, NP=4, FIRST=True)),
            ("TokenFeed_v1:4:0", lambda: bind(X.TokenFeed, NP=4, FIRST=False)),
            ("TokenFeed_v1:1024:1", lambda: bind(X.TokenFeed, NP=1024, FIRST=True)),
            ("TokenFeed_v1:1024:0", lambda: bind(X.TokenFeed, NP=1024, FIRST=False)),
            ("OutCollect_v1:2:1", lambda: bind(X.OutCollect, KO=2, FIRST=True)),
            ("OutCollect_v1:2:0", lambda: bind(X.OutCollect, KO=2, FIRST=False)),
            ("OutCollect_v1:64:1", lambda: bind(X.OutCollect, KO=64, FIRST=True)),
            ("OutCollect_v1:64:0", lambda: bind(X.OutCollect, KO=64, FIRST=False)),
            ("SlotEntry_v1:2", lambda: bind(X.SlotEntry, KO=2, IDB=17, NP=4, STOP=())),
            ("SlotEntry_v1:64", lambda: bind(X.SlotEntry, KO=64, IDB=17, NP=1024, STOP=(128001,)))]
