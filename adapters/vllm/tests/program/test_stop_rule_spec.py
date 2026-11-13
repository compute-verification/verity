"""`verity_vllm.program.frontend.stop_rule.spec`: the length rule's config under its keys, and a ValueError for a config
`StopCheck_v1` has no check for or whose outputs are no positive integer."""
from __future__ import annotations

import pytest

from verity_vllm.program.frontend import stop_rule as SR

REST = {"max_tokens": 8, "prompt_len": 4, "max_model_len": 64, "outputs": 3, "finish": 1}


def test_spec_is_the_config_under_its_keys():
    s = SR.spec((7, 9), **REST)
    assert tuple(s) == SR.KEYS and s == {"stops": [7, 9], **REST}


@pytest.mark.parametrize("outputs", [0, -1, True, 1.0, "3", None])
def test_outputs_that_are_no_positive_integer_are_a_value_error(outputs):
    with pytest.raises(ValueError, match="outputs must be a positive integer"):
        SR.spec((7, 9), **{**REST, "outputs": outputs})


@pytest.mark.parametrize("stops,over", [((9, 7), {}), ((), {}), ((7, 9), {"max_tokens": -1}), ((7, 9), {"finish": 4})])
def test_a_config_the_check_refuses_is_a_value_error_with_its_reason(stops, over):
    with pytest.raises(ValueError, match="STOP increasing u32 ids"):
        SR.spec(stops, **{**REST, **over})
