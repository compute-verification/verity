"""`check.commit_rules.hidden_stream_refusal`: no Commit acquires an attention interior, so a required-value manifest that still names
one is refused with the command that rebuilds it, and any other manifest passes."""
from __future__ import annotations

from verity_vllm.check.commit_rules import hidden_stream_refusal


def _manifest(**by_family: int) -> dict:
    return {"populations": {"by_family": dict(by_family)}}


def test_a_manifest_naming_an_attention_interior_is_refused_with_its_rebuild():
    why = hidden_stream_refusal(_manifest(instance_outputs=400, fa2_hidden_m1_stream=630, fa3_hidden_m1_stream=64))
    assert why.startswith("the required-value manifest names fa2_hidden_m1_stream, fa3_hidden_m1_stream rows")
    assert "verity-vllm manifest build-global" in why


def test_any_other_manifest_passes():
    assert hidden_stream_refusal(_manifest(instance_outputs=16480, call_boundaries=8192, norm_scales=2080)) is None
    assert hidden_stream_refusal(_manifest(instance_outputs=400, fa2_hidden_m1_stream=0)) is None
    assert hidden_stream_refusal(None) is None
