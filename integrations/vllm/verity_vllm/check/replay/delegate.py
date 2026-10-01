"""Row families whose evaluation a protocol's replay supplies instead of a row kernel.

A delegate evaluates a row from its resolved operands like a row kernel (`evaluate._evaluate_row`), and also receives the
row's place (`Place`), so it can answer from a protocol's own commitment. PoUW's `ncp-v2` installs one for
`NcpLinearRow_v1` for the length of a replay (`check.replay.pouw_circuit`). Outside that, the family is not evaluable,
and a Program with such rows fails closed.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Place:
    """Where the row is: its request, its engine step, its module, its row within the request's block, its committed member
    and the store."""

    request_id: str
    step: int
    op_path: str
    row: int
    member: str
    store: Any


Delegate = Callable[[dict[str, Any], list[list[np.ndarray]], list[np.ndarray | None], Place], dict[str, np.ndarray]]

DELEGATES: dict[str, Delegate] = {}


@contextmanager
def delegated(family: str, fn: Delegate) -> Iterator[None]:
    if family in DELEGATES:
        raise ValueError(f"{family}: a delegate is already installed")
    DELEGATES[family] = fn
    try:
        yield
    finally:
        DELEGATES.pop(family, None)


__all__ = ["DELEGATES", "Delegate", "Place", "delegated"]
