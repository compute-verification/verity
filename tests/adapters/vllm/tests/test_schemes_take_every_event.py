"""A handler on the exchange takes only the events it names; every scheme taking the serve's whole order is
`test_serve_order_smoke.py`."""

from __future__ import annotations

import pytest
import torch

from verity import hooks as H
from verity.channel import Refused
from verity_vllm import messages as M


def test_a_handler_takes_only_the_events_it_names():
    ex, got = M.Exchange(), []

    def registered(event: H.RegisteredValues) -> None:
        got.append(event.values)

    ex.subscribe("s", "memory", registered)
    ex.subscribe("s", "memory", lambda e: got.append(type(e).__name__))
    w, kv = torch.zeros(64, 256, dtype=torch.int8), torch.zeros(2, 3, 16, 256, dtype=torch.uint8)
    ex.emit(H.RegisteredValues((("w", w),), operands={"w": 0}))
    ex.emit(H.RangesDeclared(3, 16, 256, 2, (kv,)))
    assert [type(g).__name__ for g in got] == ["tuple", "str", "str"] and got[1:] == ["RegisteredValues", "RangesDeclared"]

    def stepped(event: H.StepBegan) -> None: ...

    with pytest.raises(Refused, match="StepBegan"):
        ex.subscribe("s", "memory", stepped)
