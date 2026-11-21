"""circuit-check's bindings of `verity_vllm.definitions.registry.io_shell`'s `TokenFeed_v1`, `OutCollect_v1`, `SlotEntry_v1`,
`SelectCollect_v1`, `SelectCollect_v2`, `SelectEntry_v1`, `OutCollect_v2`, `SlotEntry_v2` and `TokenFeed_v2`, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`TokenFeed_v1` at a small class (4 prompt ids) and at the class of record's (1,024), each the first position's and
    every other's; `OutCollect_v1` at KO 2 and 64, the first position's and every other's; and
    `SlotEntry_v1` at the small class (KO 2, 4 prompt ids) with no stop id (TINY's), at the network attester's KO 4 class
    (4 outputs, 32 prompt ids) and at the class of record's (KO 64, 1,024), these two with Llama-3.2-1B's stop id
    (128001), and a served request's (one prompt length) at KO 2, with no stop id and with Llama-3.2-1B's
    (``SlotEntry_v1:2:np1:stop``, class 4,27,2's), and at 64; `SelectCollect_v1` and
    `SelectEntry_v1` at TINY's select (V 256, groups of 16) and small class, and at Llama-3.2-1B's (V 128,256) at 4 outputs
    and 32 prompt ids with its stop id, both also at the class of record's 64 outputs (`SelectEntry_v1` at its 1,024 prompt
    ids, and at class 4,27,2's 2 outputs and 27 prompt ids, ``SelectEntry_v1:2:27``), the collector the first
    position's and every other's; `SelectCollect_v2` at TINY's select and at Llama-3.2-1B's 4 outputs, at a later round."""
    from verity.circuits.defs import bind

    from verity_vllm.definitions.registry import io_shell as X

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
            ("SlotEntry_v1:2:np1", lambda: bind(X.SlotEntry, KO=2, IDB=17, NP=1, STOP=())),
            ("SlotEntry_v1:2:np1:stop", lambda: bind(X.SlotEntry, KO=2, IDB=17, NP=1, STOP=(128001,))),
            ("SlotEntry_v1:64:np1", lambda: bind(X.SlotEntry, KO=64, IDB=17, NP=1, STOP=(128001,))),
            ("OutCollect_v2:2:1", lambda: bind(X.OutCollectV2, KO=2, FIRST=True)),
            ("OutCollect_v2:2:0", lambda: bind(X.OutCollectV2, KO=2, FIRST=False)),
            ("SlotEntry_v2:2:np1:stop", lambda: bind(X.SlotEntryV2, KO=2, IDB=17, NP=1, STOP=(128001,))),
            ("TokenFeed_v2:2:1", lambda: bind(X.TokenFeedV2, NP=2, FIRST=True)),
            ("TokenFeed_v2:27:1", lambda: bind(X.TokenFeedV2, NP=27, FIRST=True)),
            ("TokenFeed_v2:27:0", lambda: bind(X.TokenFeedV2, NP=27, FIRST=False)),
            ("SelectCollect_v1:2:1", lambda: bind(X.SelectCollect, V=256, C=16, KO=2, FIRST=True)),
            ("SelectCollect_v1:2:0", lambda: bind(X.SelectCollect, V=256, C=16, KO=2, FIRST=False)),
            ("SelectCollect_v1:4:0", lambda: bind(X.SelectCollect, V=128256, C=16, KO=4, FIRST=False)),
            ("SelectCollect_v1:64:1", lambda: bind(X.SelectCollect, V=128256, C=16, KO=64, FIRST=True)),
            ("SelectCollect_v1:64:0", lambda: bind(X.SelectCollect, V=128256, C=16, KO=64, FIRST=False)),
            ("SelectCollect_v2:2:0:1", lambda: bind(X.SelectCollectV2, V=256, C=16, KO=2, FIRST=False, ROUND=1)),
            ("SelectCollect_v2:4:0:3", lambda: bind(X.SelectCollectV2, V=128256, C=16, KO=4, FIRST=False, ROUND=3)),
            ("SelectEntry_v1:2", lambda: bind(X.SelectEntry, V=256, C=16, KO=2, FIRST=False, IDB=17, NP=4, STOP=())),
            ("SelectEntry_v1:4", lambda: bind(X.SelectEntry, V=128256, C=16, KO=4, FIRST=False, IDB=17, NP=32,
                                              STOP=(128001,))),
            ("SelectEntry_v1:64", lambda: bind(X.SelectEntry, V=128256, C=16, KO=64, FIRST=False, IDB=17, NP=1024,
                                               STOP=(128001,))),
            ("SelectEntry_v1:2:27", lambda: bind(X.SelectEntry, V=128256, C=16, KO=2, FIRST=False, IDB=17, NP=27,
                                                 STOP=(128001,)))]
