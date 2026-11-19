"""The served run's events and requests (`verity.hooks`)."""

from __future__ import annotations

import pytest

from verity import hooks as H


def test_hook_types_hold_no_tensor_type() -> None:
    store = object()
    assert {e.hook for e in H.EVENTS} == set(H.HOOKS)
    assert H.RangesDeclared(4, 16, 128, 2, (store,)).stores == (store,)
    assert H.MemoryChanged("freed", (7,), 3).ranges == (7,)
    assert H.StepBegan(3, ("in-1",), 16).inputs == ("in-1",)

    def kernel(due: H.BlockDue, original):
        return original(None)
    due = H.BlockDue("layers.0.mlp", 0, {"n": 8, "k": 4}, "int8", {"x": store, "weight": store})
    assert kernel(due, lambda inputs: ("served", inputs)) == ("served", None)


def test_a_memory_change_and_an_admission_say_one_thing_each() -> None:
    assert H.AdmitInput("hold").input is None and H.AdmitInput("refuse", input="i", why="w").why == "w"
    for bad in (lambda: H.AdmitInput("start"), lambda: H.AdmitInput("hold", input="i"), lambda: H.AdmitInput("refuse"),
                lambda: H.MemoryChanged("moved", (1,)), lambda: H.MemoryChanged("registered", (1,)),
                lambda: H.MemoryChanged("written", (1, 2)), lambda: H.MemoryChanged("freed")):
        with pytest.raises(ValueError):
            bad()


def test_unmet_names_each_request_the_properties_do_not_meet() -> None:
    eager, one = H.RequireProperty("eager", True, "e"), H.RequireProperty("world", 1, "w")
    bf16, kernels = H.RequireProperty("dtype", "bf16", "d"), H.RequireProperty("kernels", "profile", "k", fixes=True)
    asked = [("pouw", eager), ("pouw", one), ("pous", bf16), ("spot", kernels)]
    assert H.unmet(asked, {"eager": True, "world": 1, "dtype": "bf16", "kernels": "profile"}) == []
    assert H.unmet(asked, {"eager": False, "world": 1, "dtype": "fp8"}) == [("pouw", eager), ("pous", bf16), ("spot", kernels)]
