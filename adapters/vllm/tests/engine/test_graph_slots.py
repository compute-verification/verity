"""The static slots (`engine.graph_slots`) under a stand-in of vLLM's graph capture: no compiled piece, values keyed by name.

`Manager.capture` runs each size as vLLM's `CudaGraphManager.capture` does: an uncaptured warm-up forward, then the
captured one, each made by ``create_forward_fn``. A forward taps two values a "layer": in the warm-up it only stages them
(the slot is allocated there), in the capture it records the copy in the size's plan. A replay issues the recorded copies
again, which the stand-in does by staging the step's values into the same slots. CPU only, made-up values.
"""
from __future__ import annotations

from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from verity_vllm.engine.graph_slots import GRAPH_MODES, Entry, Slots

LAYERS, WIDTH = 2, 3


def desc(mode: str, pad: int) -> NS:
    return NS(cg_mode=NS(name=mode), num_tokens=pad)


class Manager:
    def __init__(self, descs: list[NS], profiling: bool = False) -> None:
        self._capture_descs = {"all": descs}
        self._capture_mem_samples: list[int] | None = [] if profiling else None

    def capture(self, create_forward_fn: Any) -> None:
        for d in self._capture_descs["all"]:
            create_forward_fn(d, warmup=True)("NONE")
            create_forward_fn(d, warmup=False)(d.cg_mode.name)


def values(pad: int, seed: int) -> dict[tuple[int, str], torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {(layer, plane): torch.randint(0, 99, shape(pad), generator=g, dtype=dtype)
            for layer in range(LAYERS)
            for plane, shape, dtype in (("rows", lambda p: (p, WIDTH), torch.int32), ("fixed", lambda p: (WIDTH,), torch.int16))}


def tap(slots: Slots, vals: dict[tuple[int, str], torch.Tensor], pad: int, captured: bool) -> list[Entry]:
    staged = [slots.stage((layer, i), f"{layer}/{plane}", v, pad) for i, ((layer, plane), v) in enumerate(vals.items())]
    if captured and slots.capturing is not None:
        slots.plans[slots.capturing].extend(staged)
    return staged


def capture(slots: Slots, sizes: tuple[int, ...], mine: Any = None, profiling: bool = False) -> Manager:
    manager = Manager([desc(mode, pad) for mode in GRAPH_MODES for pad in sizes], profiling)

    def create_forward_fn(d: NS, warmup: bool) -> Any:
        return lambda mode: tap(slots, values(d.num_tokens, 100 + d.num_tokens), d.num_tokens, captured=mode != "NONE")

    slots.capture_wrapper(mine)(Manager.capture)(manager, create_forward_fn)
    return manager


def test_a_replayed_step_reads_its_first_rows_from_one_slot_a_value() -> None:
    slots = Slots()
    capture(slots, (4, 8))
    assert slots.max_rows == 8 and slots.capturing is None and slots.errors == []
    assert sorted(slots.plans) == [(mode, pad) for mode in sorted(GRAPH_MODES) for pad in (4, 8)]
    # a token-row value has one slot of the largest size's rows for every size; a fixed-shape one has its own
    assert sorted(tuple(s.shape) for s in slots.slots.values()) == [(WIDTH,)] * LAYERS + [(8, WIDTH)] * LAYERS
    for pad, n in ((8, 5), (4, 4), (8, 8)):
        step = values(pad, seed=n)
        tap(slots, step, pad, captured=False)                          # the replay issues the plan's copies again
        got = slots.entries("PIECEWISE", pad)
        assert got is not None and [name for _o, name, _s, _r in got] == ["0/rows", "0/fixed", "1/rows", "1/fixed"]
        for entry, (key, want) in zip(got, step.items(), strict=True):
            cut = slots.value(entry, n, pad)
            assert torch.equal(cut, want[:n] if key[1] == "rows" else want), (pad, n, key)


def test_a_steps_live_copies_come_in_order_with_its_plan() -> None:
    slots = Slots()
    capture(slots, (4,))
    live = [slots.stage((0, 0.5), "0/live", torch.ones(4, 1), 4)]
    got = slots.entries("FULL", 4, live)
    assert got is not None and [name for _o, name, _s, _r in got] == ["0/rows", "0/live", "0/fixed", "1/rows", "1/fixed"]
    assert slots.entries("FULL", 16) is None and slots.entries("NONE", 4) is None


def test_a_manager_that_is_not_the_followed_one_leaves_the_slots_as_they_are() -> None:
    slots = Slots()
    followed: list[Manager] = []

    def mine(manager: Manager) -> bool:
        return not followed or manager is followed[0]

    followed.append(capture(slots, (4,), mine))
    plans, held = {k: list(v) for k, v in slots.plans.items()}, dict(slots.slots)
    other = Manager([desc("FULL", 16)])
    slots.capture_wrapper(mine)(Manager.capture)(other, lambda d, warmup: lambda mode: None)
    assert slots.plans == plans and slots.slots == held and slots.max_rows == 4


def test_a_profiling_capture_and_a_larger_one_drop_the_slots() -> None:
    dropped: list[int] = []
    slots = Slots(dropped=lambda: dropped.append(1))
    capture(slots, (4,), profiling=True)
    assert slots.slots == {} and slots.plans == {} and slots.max_rows == 0 and slots.stats["profiling_captures"] == 1
    capture(slots, (4,))
    was = slots.stats["slot_bytes"]
    capture(slots, (8,))
    assert slots.max_rows == 8 and sorted(slots.plans) == [("FULL", 8), ("PIECEWISE", 8)]
    assert slots.stats["slot_resets"] == 1 and slots.stats["slot_bytes"] > was > 0 and len(dropped) == 4


def test_a_slot_reused_for_another_dtype_and_a_copy_never_flushed_are_errors() -> None:
    slots = Slots()
    slots.begin_capture(4)
    slots.stage((0,), "a", torch.zeros(4, 2), 4)
    slots.stage((0,), "a", torch.zeros(4, 2, dtype=torch.int8), 4)
    assert slots.errors == ["a: slot (4, 2)/torch.float32 reused for (4, 2)/torch.int8"]
    held = torch.arange(8.0).reshape(4, 2)
    _o, _n, slot, _r = slots.stage((1,), "b", held, 4, defer=True)
    slots.end_capture(("FULL", 4))
    assert slots.pending == [] and slots.errors[-1] == "capture FULL:4: 1 captured copies never flushed"
    slots.stage((1,), "b", held, 4, defer=True)
    slots.flush()
    assert torch.equal(slot, held) and slots.stats["capture_flushes"] == 1
