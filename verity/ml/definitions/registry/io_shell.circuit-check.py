"""circuit-check's bindings of `verity.ml.definitions.registry.io_shell`'s `TokenFeed_v1`, `OutCollect_v1`, `SlotEntry_v1`,
`SelectCollect_v1` and `SelectEntry_v1`, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`TokenFeed_v1` at a small class (4 prompt ids) and at the class of record's (1,024), each the first position's and
    every other's; `OutCollect_v1` at KO 2 and 64, the first position's and every other's; and
    `SlotEntry_v1` at the small class (KO 2, 4 prompt ids) with no stop id (TINY's), at the network certifier's KO 4 class
    (4 outputs, 32 prompt ids) and at the class of record's (KO 64, 1,024), these two with Llama-3.2-1B's stop id
    (128001); `SelectCollect_v1` and `SelectEntry_v1` at TINY's select (V 256, groups of 16) and small class, and at
    Llama-3.2-1B's (V 128,256) at 4 outputs and 32 prompt ids with its stop id, `SelectEntry_v1` also at the class of
    record's (64 outputs, 1,024 prompt ids)."""
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
            ("SlotEntry_v1:4", lambda: bind(X.SlotEntry, KO=4, IDB=17, NP=32, STOP=(128001,))),
            ("SlotEntry_v1:64", lambda: bind(X.SlotEntry, KO=64, IDB=17, NP=1024, STOP=(128001,))),
            ("SelectCollect_v1:2:1", lambda: bind(X.SelectCollect, V=256, C=16, KO=2, FIRST=True)),
            ("SelectCollect_v1:2:0", lambda: bind(X.SelectCollect, V=256, C=16, KO=2, FIRST=False)),
            ("SelectCollect_v1:4:0", lambda: bind(X.SelectCollect, V=128256, C=16, KO=4, FIRST=False)),
            ("SelectEntry_v1:2", lambda: bind(X.SelectEntry, V=256, C=16, KO=2, FIRST=False, IDB=17, NP=4, STOP=())),
            ("SelectEntry_v1:4", lambda: bind(X.SelectEntry, V=128256, C=16, KO=4, FIRST=False, IDB=17, NP=32,
                                              STOP=(128001,))),
            ("SelectEntry_v1:64", lambda: bind(X.SelectEntry, V=128256, C=16, KO=64, FIRST=False, IDB=17, NP=1024,
                                               STOP=(128001,)))]
