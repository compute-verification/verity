"""The registry refuses a retired model's Definition ids by name."""
from __future__ import annotations

import pytest
from verity.circuits.codec import lookup_primitive
from verity.circuits.defs import REGISTRY

import verity_vllm.definitions.registry  # noqa: F401  -- registers the lazy families


@pytest.mark.parametrize("fid", ["Dsv4Retired_v1", "MhcRetired_v1{HC=4}", "MarlinRetired_v1", "MoeAlignRetired_v1",
                                 "Lifted[Dsv4Retired_v1]_v2"])
def test_a_retired_definition_id_is_refused_in_one_sentence(fid: str) -> None:
    with pytest.raises(KeyError, match="is refused: its model is not served through the one circuit"):
        lookup_primitive(REGISTRY, fid)


def test_a_served_id_with_a_retired_prefix_s_neighbour_still_resolves() -> None:
    assert lookup_primitive(REGISTRY, "GatherBf16x8_v1").id == "GatherBf16x8_v1"
    with pytest.raises(KeyError, match="is not in the registry"):
        lookup_primitive(REGISTRY, "MoeRouterRetired_v1")
