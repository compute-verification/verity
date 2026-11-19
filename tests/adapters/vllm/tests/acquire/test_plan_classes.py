"""The capture classes the acquisition's sources read are the capture plan's (`program.predict.capture_plan`): the plan is
a layer above the acquisition, which takes it as data, so the sources name the classes they read once more, held equal here."""
from __future__ import annotations

from verity_vllm.acquire.sources import call_boundary_plan as CBP
from verity_vllm.program.predict import capture_plan as CP


def test_the_sources_capture_classes_are_the_plans():
    assert (CBP.MODULE_OUTPUT, CBP.CALL_BOUNDARY, CBP.SCALE_PRODUCTS) == (CP.MODULE_OUTPUT, CP.CALL_BOUNDARY, CP.SCALE_PRODUCTS)
    assert {CBP.MODULE_OUTPUT, CBP.CALL_BOUNDARY, CBP.SCALE_PRODUCTS} <= set(CP.CLASSES)
