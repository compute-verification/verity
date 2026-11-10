"""What `engine.served.read_rank` answers on the ranks of a stand-in engine built exactly as its workload declares: each switch the
declared value, vLLM's default elsewhere (a stand-in LLM's `collective_rpc` returns `answer` for `read_rank`)."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from verity_vllm.engine import served as SV


def switches(wl: dict[str, Any], target: Any = None) -> dict[str, Any]:
    decl = SV.declared(wl, target)
    return {k: decl[k]["value"] if k in decl else (None if v == SV.RESOLVED else v) for k, v in SV.DEFAULTS.items()}


def answer(method: Any, wl: dict[str, Any], ranks: Iterable[int] = range(2), target: Any = None) -> list[dict[str, Any]] | None:
    """`read_rank`'s answer from every rank of ``ranks``; None for any other method."""
    return [{"rank": r, "switches": switches(wl, target)} for r in ranks] if method is SV.read_rank else None
