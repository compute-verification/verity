"""The slice of the attention split's snapshot this file's number names (`attention_split.py`) keeps its ids and digests."""
from __future__ import annotations

import pytest
from attention_split import changed_ids_and_digests


@pytest.mark.slow
def test_every_family_definition_keeps_its_id_and_digest():
    """Each Definition of the slice, made from its id, has the snapshot's id and canonical digest: what catches a change to a family
    Definition's circuit, which circuit-check's pins (gate counts) don't hold."""
    assert changed_ids_and_digests(__file__) == {}
