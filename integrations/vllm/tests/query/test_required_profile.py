"""The attention hidden family a request manifest requires follows the FA version its attention layers ran; a resolution that did not
record it falls back to the declared capability's FA version (`TargetProfile.effective_flash_attn_version`): FA3 only on cc 9.x."""
from __future__ import annotations

import pytest

from verity_vllm.query.required import _profile


def _res(cc=None, fa=None):
    res = {"attention_impls_observed": {"model.layers.0.self_attn.attn": {} if fa is None else {"vllm_flash_attn_version": fa}}}
    if cc is not None:
        res["target"] = {"declared": {"compute_capability": list(cc)}}
    return res


@pytest.mark.parametrize("cc, want", [(None, "ampere"), ((8, 9), "ampere"), ((8, 0), "ampere"), ((9, 0), "hopper"), ((12, 0), "ampere")])
def test_unobserved_version_falls_back_to_the_capabilitys_fa_version(cc, want):
    assert _profile(_res(cc), None)[0] == want


def test_the_observed_version_decides():
    assert _profile(_res((9, 0), fa=2), None)[0] == "ampere"
    assert _profile(_res((12, 0), fa=3), None)[0] == "hopper"
    assert _profile(_res((12, 0)), "hopper")[0] == "hopper"
