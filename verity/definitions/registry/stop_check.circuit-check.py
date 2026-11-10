"""circuit-check's bindings of `verity.definitions.registry.stop_check`'s `StopCheck_v1`, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`StopCheck_v1` at small classes on each finish (4 outputs and one stop id on a stop, 5 and two at max_tokens, 4 and none
    running on, 3 and one at the model's length), and at GLM-5.3's served class (32 outputs, its three EOS ids) at max_tokens."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import stop_check as X

    return [("StopCheck_v1:4:1:stop", lambda: bind(X.StopCheck, T=4, STOP=(3,), MAX_TOKENS=8, PROMPT_LEN=8, MAX_MODEL_LEN=64,
                                                    FINISH=1)),
            ("StopCheck_v1:5:2:length", lambda: bind(X.StopCheck, T=5, STOP=(3, 7), MAX_TOKENS=5, PROMPT_LEN=8, MAX_MODEL_LEN=64,
                                                     FINISH=2)),
            ("StopCheck_v1:4:0:runs-on", lambda: bind(X.StopCheck, T=4, STOP=(), MAX_TOKENS=8, PROMPT_LEN=8, MAX_MODEL_LEN=64,
                                                      FINISH=0)),
            ("StopCheck_v1:3:1:model-length", lambda: bind(X.StopCheck, T=3, STOP=(3,), MAX_TOKENS=8, PROMPT_LEN=61,
                                                           MAX_MODEL_LEN=64, FINISH=3)),
            ("StopCheck_v1:32:3:length", lambda: bind(X.StopCheck, T=32, STOP=(154820, 154827, 154829), MAX_TOKENS=32,
                                                      PROMPT_LEN=1024, MAX_MODEL_LEN=131072, FINISH=2))]
