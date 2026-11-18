"""circuit-check's bindings of `verity_vllm.definitions.registry.greedy_stop`'s `GreedyStopCheck_v1`, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`GreedyStopCheck_v1` on the combine of `TokenSelect_v1{V=320}`'s cut (20 chunks of 16 lanes) over two outputs and one
    stop id, and at GLM-5.3's served statics (154,880 lanes in 20 chunks, its three EOS ids) over two outputs at max_tokens.
    `GreedyStopCheckClass_v1`, a class's check, on that small combine at a class of 4 served ids with two stop ids, at a prompt
    of 8 ids and of 43 (a class is keyed by its prompt's exact length, a power of two or not), and at GLM-5.3's statics at
    (P, T) = (64, 128) and (43, 128), each under `MAX_GATES`, so kept in ``--all`` with no pin."""
    from verity.circuits.defs import bind

    from verity_vllm.definitions.registry import greedy_stop as X

    return [("GreedyStopCheck_v1:320:2:stop", lambda: bind(X.GreedyStopCheck, V=320, N=16, T=2, STOP=(3,), MAX_TOKENS=8,
                                                           PROMPT_LEN=8, MAX_MODEL_LEN=64, FINISH=1)),
            ("GreedyStopCheck_v1-served", lambda: bind(X.GreedyStopCheck, V=154880, N=7744, T=2, STOP=(154820, 154827, 154829),
                                                       MAX_TOKENS=2, PROMPT_LEN=1024, MAX_MODEL_LEN=131072, FINISH=2)),
            ("GreedyStopCheckClass_v1:320:8:4", lambda: bind(X.GreedyStopCheckClass, V=320, N=16, P=8, T=4, STOP=(37, 301),
                                                        MAX_MODEL_LEN=64)),
            ("GreedyStopCheckClass_v1:320:43:4", lambda: bind(X.GreedyStopCheckClass, V=320, N=16, P=43, T=4, STOP=(37, 301),
                                                         MAX_MODEL_LEN=64)),
            ("GreedyStopCheckClass_v1-served", lambda: bind(X.GreedyStopCheckClass, V=154880, N=7744, P=64, T=128,
                                                       STOP=(154820, 154827, 154829), MAX_MODEL_LEN=131072)),
            ("GreedyStopCheckClass_v1-served-43", lambda: bind(X.GreedyStopCheckClass, V=154880, N=7744, P=43, T=128,
                                                          STOP=(154820, 154827, 154829), MAX_MODEL_LEN=131072))]
