"""circuit-check's bindings of `verity.definitions.registry.greedy_stop`'s `GreedyStopCheck_v1`, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`GreedyStopCheck_v1` on the combine of `TokenSelect_v1{V=320}`'s cut (20 chunks of 16 lanes) over two outputs and one
    stop id, and at GLM-5.3's served statics (154,880 lanes in 20 chunks, its three EOS ids) over two outputs at max_tokens."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import greedy_stop as X

    return [("GreedyStopCheck_v1:320:2:stop", lambda: bind(X.GreedyStopCheck, V=320, N=16, T=2, STOP=(3,), MAX_TOKENS=8,
                                                           PROMPT_LEN=8, MAX_MODEL_LEN=64, FINISH=1)),
            ("GreedyStopCheck_v1-served", lambda: bind(X.GreedyStopCheck, V=154880, N=7744, T=2, STOP=(154820, 154827, 154829),
                                                       MAX_TOKENS=2, PROMPT_LEN=1024, MAX_MODEL_LEN=131072, FINISH=2))]
