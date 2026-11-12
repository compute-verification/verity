"""circuit-check's bindings of `verity_vllm.program.predict.draft_rows`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`KeptCheck_v1` at a pass that keeps none, some and every draft, as a pass before the last (equality) and as the last
    (at least); `DraftLayerStandIn_v1` at 1, 3 and 5 drafts over a table of two tokens."""
    from verity.circuits.defs import bind
    from verity_vllm.program.predict import draft_rows as DR

    def layer(k: int) -> Callable[[], Any]:
        return lambda: bind(DR.DraftLayerStandIn, K=k, TABLE=((3, tuple(range(7, 7 + k))), (200, (0,) * k)))

    return [
        *((f"KeptCheck_v1:{kept}:{last}", lambda kept=kept, last=last: bind(DR.KeptCheck, KEPT=kept, LAST=last))
          for kept, last in ((0, 0), (3, 0), (5, 0), (0, 1), (5, 1))),
        *((f"DraftLayerStandIn_v1:{k}", layer(k)) for k in (1, 3, 5)),
    ]
