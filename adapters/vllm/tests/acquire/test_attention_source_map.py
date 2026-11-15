"""`verity_vllm.acquire.sources.attention_source` refuses, when it is built and by name, a capture map that gives one of its
taps a form it cannot bind: a tap captured at one named layer of the model, where the source binds at each attention layer's
call, and a cache tap or the top-k whose row width reads a per-step size, where the source cuts their rows by a number."""
from __future__ import annotations

import copy
from typing import Any

import pytest

pytest.importorskip("torch")
from tests.acquire.glm53_model import CMAP, glm_model
from verity_vllm.acquire.sources import attention_source as AS
from verity_vllm.acquire.sources import route_source as RS


def _build(tap_id: str, **fields: Any) -> AS.AttentionSource:
    """The source over the GLM map with `fields` set on one tap (a field set to None is removed)."""
    cmap = copy.deepcopy(CMAP)
    tap = next(t for t in cmap["taps"] if t["id"] == tap_id)
    for k, v in fields.items():
        if v is None:
            del tap[k]
        else:
            tap[k] = v
    return AS.AttentionSource(None, RS.RouteSource(None, glm_model(), cmap))


def test_the_map_as_registered_is_bound_with_a_number_for_each_width_the_source_cuts_rows_by():
    src = _build(AS.MLA_ROWS)
    assert {AS.MLA_ROWS, AS.INDEX_ROWS, AS.TOPK} <= set(src.width) and all(isinstance(w, int) for w in src.width.values())
    assert src.names and set(src.names) <= set(src.taps)


def test_a_model_level_tap_of_the_source_is_refused_by_name():
    tap_id, model_level = AS.ARGUMENTS["q_c"], next(t["layer"] for t in CMAP["taps"] if "layer" in t)
    with pytest.raises(ValueError, match=rf"binds per-layer taps \(`layers`\); {tap_id} is a model-level one"):
        _build(tap_id, layers=None, layer=model_level)


@pytest.mark.parametrize("tap_id", [AS.MLA_ROWS, AS.INDEX_ROWS, AS.TOPK])
def test_a_tap_whose_row_width_is_a_per_step_size_is_refused_by_name(tap_id: str):
    free = next(iter(CMAP["free_symbols"]))
    with pytest.raises(ValueError, match=rf"the row width of \['{tap_id}'\] reads a per-step size"):
        _build(tap_id, shape=[free, free])
