"""`check.commit_rules.hidden_classes_required`: a Commit attaches the attention hidden tap only when the required-value manifest names
its stream, so a Program whose attention is its own Calls (the exact-int8 profile's) is not held to a tap that never fires."""
from __future__ import annotations

from verity_vllm.check.commit_rules import hidden_classes_required


def _manifest(**by_family: int) -> dict:
    return {"populations": {"by_family": dict(by_family)}}


def test_the_hidden_tap_is_required_only_when_the_manifest_names_its_stream():
    assert hidden_classes_required(_manifest(instance_outputs=400, fa2_hidden_m1_stream=630)) is True
    assert hidden_classes_required(_manifest(instance_outputs=400, fa3_hidden_m1_stream=64)) is True
    assert hidden_classes_required(_manifest(instance_outputs=16480, call_boundaries=8192, norm_scales=2080)) is False
    assert hidden_classes_required(_manifest(instance_outputs=400, fa2_hidden_m1_stream=0)) is False
    assert hidden_classes_required(None) is None
