"""`verity_vllm.acquire.sources.attention_source` is built over a capture map without a check of each tap's form.  Over the
map as registered it holds a number for the width of every tap.  A tap of its own captured at one named layer of the model
(`layer`, where the source binds at each attention layer's call) stops the build with the `TypeError` of reading layers that
are not there.  A tap whose last dimension reads a per-step size is built, and its width is held as the map writes it.  The
route source it is built on (`verity_vllm.acquire.sources.route_source`) reads the model's modules before the model's config."""
from __future__ import annotations

import copy
import types
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


def test_the_map_as_registered_is_bound_with_a_number_for_the_width_of_every_tap():
    src = _build(AS.MLA_ROWS)
    assert {AS.MLA_ROWS, AS.INDEX_ROWS, AS.TOPK} <= set(src.width) and all(isinstance(w, int) for w in src.width.values())
    assert set(src.width) == set(src.taps)
    assert src.names and set(src.names) <= set(src.taps)


def test_a_model_level_tap_of_the_source_stops_the_build_where_its_layers_are_read():
    tap_id, model_level = AS.ARGUMENTS["q_c"], next(t["layer"] for t in CMAP["taps"] if "layer" in t)
    with pytest.raises(TypeError, match=r"^'NoneType' object is not iterable$"):
        _build(tap_id, layers=None, layer=model_level)


@pytest.mark.parametrize("tap_id", [AS.MLA_ROWS, AS.INDEX_ROWS, AS.TOPK, AS.ARGUMENTS["q_c"]])
@pytest.mark.parametrize("written", ["{free}", "={free}*2"])
def test_a_tap_whose_last_dimension_reads_a_per_step_size_is_built_with_the_width_as_the_map_writes_it(tap_id: str, written: str):
    free = next(iter(CMAP["free_symbols"]))
    width = written.format(free=free)
    src = _build(tap_id, shape=[free, width])
    assert src.width[tap_id] == width and set(src.width) == set(src.taps)
    assert all(isinstance(w, int) for tid, w in src.width.items() if tid != tap_id)


def test_the_route_source_reads_the_models_modules_before_its_config():
    """A model that is no module is refused where its modules are read, whatever reading its config would raise."""
    class Config:
        def to_dict(self) -> dict[str, Any]:
            raise TypeError("the config was read first")

    with pytest.raises(AttributeError, match="has no attribute 'modules'"):
        RS.RouteSource(None, types.SimpleNamespace(config=Config()), copy.deepcopy(CMAP))
