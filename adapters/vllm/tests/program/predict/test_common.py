from __future__ import annotations

import gc

import pytest
from verity_vllm.program.predict.common import cyclic_gc_off


def test_cyclic_gc_off_holds_the_collector_off_inside_and_restores_its_state():
    @cyclic_gc_off
    def inside(fail: bool = False) -> bool:
        if fail:
            raise ValueError("boom")
        return gc.isenabled()

    was = gc.isenabled()
    try:
        gc.enable()
        assert inside() is False and gc.isenabled()
        with pytest.raises(ValueError):
            inside(fail=True)
        assert gc.isenabled()
        gc.disable()
        assert inside() is False and not gc.isenabled()
    finally:
        (gc.enable if was else gc.disable)()
