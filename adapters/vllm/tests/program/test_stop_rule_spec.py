"""`verity_vllm.program.frontend.stop_rule.spec`: the length rule's config under its keys, and a ValueError for a config
`StopCheck_v1` has no check for or whose outputs are no positive integer; and the rule bound from a class's statics
(`class_definition`) against the Call a request's config binds (`definition`)."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits.evaluation import evaluate_batch
from verity.definitions.registry import stop_check as SC
from verity_vllm.program.frontend import stop_rule as SR

REST = {"max_tokens": 8, "prompt_len": 4, "max_model_len": 64, "outputs": 3, "finish": 1}
M32 = 0xFFFFFFFF


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


def _rows(fn, *inputs: list) -> np.ndarray:
    arrays = [np.array(x, dtype=np.uint32) for x in inputs]
    return np.asarray(evaluate_batch(fn, *arrays, kernel="reference")).reshape(len(inputs[0]), -1)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("stops", [(3, 7), ()])
@pytest.mark.parametrize("max_tokens, prompt_len, outputs", [(2, 8, 3), (4, 8, 3), (9, 61, 3), (0, 8, 3), (5, 8, 4), (2, 60, 6)])
def test_the_rule_bound_from_a_class_s_statics_gives_what_a_request_s_config_checks(n, stops, max_tokens, prompt_len, outputs):
    """On every stream of n outputs over the stop ids, an other id and `NOT_AN_ID`, at every finish of each config (a Program
    whose outputs follow earlier ones among them, max_tokens and the model's length 64 each reached within n or not): today's
    `StopCheck_v1` from the request's config is 1 exactly where the class's `StopClass_v1`, bound from its stop ids and model
    length alone, gives (n, the config's finish) over the request's max_tokens and prompt length. A config whose max_tokens
    the outputs before the Program's first already reach binds no check today, and is refused."""
    words = [3, 7, 5, SC.NOT_AN_ID]
    ids = [[words[k // 4 ** i % 4] for i in range(n)] for k in range(4 ** n)]
    cls = SR.class_definition(n, stops, 64)
    for finish in [f for f in (0, 1, 2, 3) if f != 1 or stops]:
        config = SR.spec(stops, max_tokens=max_tokens, prompt_len=prompt_len, max_model_len=64, outputs=outputs, finish=finish)
        _, mt, pl = SR.request_values(n, config)
        if mt < 0:
            with pytest.raises(ValueError, match="StopCheck_v1 refuses this binding"):
                SR.definition(n, config)
            continue
        today = _rows(SR.definition(n, config)[0], ids)[:, 0].tolist()
        got = _rows(cls, ids, [mt & M32] * len(ids), [pl & M32] * len(ids))
        assert today == [int(tuple(g) == (n, finish)) for g in got.tolist()]


def test_requests_of_one_class_bind_one_call():
    """Requests of one class that differ in max_tokens, prompt length and finish bind one Call; another stop id or model length
    is another Call."""
    configs = [SR.spec((3, 7), max_tokens=m, prompt_len=p, max_model_len=64, outputs=3, finish=f)
               for m, p, f in ((8, 4, 1), (2, 40, 2), (9, 61, 3), (99, 4, 0))]
    ids = {SR.class_definition(3, *(c[k] for k in SR.CLASS_KEYS)).id for c in configs}
    assert len(ids) == 1 and ids.isdisjoint({SR.class_definition(3, (3,), 64).id, SR.class_definition(3, (3, 7), 65).id})
