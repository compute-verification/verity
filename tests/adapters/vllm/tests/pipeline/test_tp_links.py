"""`verity_vllm.pipeline.tp.links.read_tp_links` on a document that lists no link, and what `verity_vllm.pipeline.tp.fold_match`
makes of one: the reader returns the document as it is, and the fold Match has links for the rank only where they are a list."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_vllm.pipeline.tp import fold_match as FM
from verity_vllm.pipeline.tp.links import read_tp_links

RANK = {"schema": "cb-a/tp_links/v2", "world": 2, "rank": 1}


def _write(tmp_path: Path, doc: dict) -> Path:
    (tmp_path / "tp_links.json").write_text(json.dumps(doc))
    return tmp_path / "tp_links.json"


@pytest.mark.parametrize("links", [{}, {"links": None}, {"links": {}}, {"links": 0}, {"links": ""}], ids=["absent", "null", "{}", "0", "''"])
def test_a_document_that_lists_no_link_is_returned_as_it_is_and_is_no_links_for_the_rank(tmp_path, links):
    doc = read_tp_links(str(_write(tmp_path, {**RANK, **links})))
    assert doc == {**RANK, **links, "peer_ranks": [0]}
    assert FM.links_for_rank(1, tmp_path, {}, []) is None


def test_a_rank_with_no_collective_has_an_empty_list_of_links(tmp_path):
    doc = read_tp_links(str(_write(tmp_path, {**RANK, "links": []})))
    assert doc["links"] == [] and doc["peer_ranks"] == [0]
    assert FM.links_for_rank(1, tmp_path, {}, []) == []


def test_a_build_without_the_file_has_no_links_for_the_rank(tmp_path):
    assert FM.links_for_rank(1, tmp_path, {}, []) is None
