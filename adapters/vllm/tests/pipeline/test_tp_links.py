"""`verity_vllm.pipeline.tp.links.read_tp_links` on a document that is not a rank's links, and what its two readers make of one:
the check arm of `verity_vllm.pipeline.tp.match` refuses it by name, and `verity_vllm.pipeline.tp.fold_match` has no links for
the rank."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_vllm.errors import Refused
from verity_vllm.pipeline.tp import fold_match as FM
from verity_vllm.pipeline.tp.links import read_tp_links

RANK = {"schema": "cb-a/tp_links/v2", "world": 2, "rank": 1}


def _write(tmp_path: Path, doc: dict) -> Path:
    (tmp_path / "tp_links.json").write_text(json.dumps(doc))
    return tmp_path / "tp_links.json"


@pytest.mark.parametrize("doc, said", [(RANK, "its `links` is absent, not a list"),
                                       ({**RANK, "links": None}, "its `links` is absent, not a list"),
                                       ({**RANK, "links": {"k": 0}}, "its `links` is a dict, not a list"),
                                       ({**RANK, "links": 0}, "its `links` is a int, not a list")])
def test_a_document_whose_links_are_not_a_list_is_refused_by_that_name(tmp_path, doc, said):
    path = _write(tmp_path, doc)
    with pytest.raises(Refused) as e:
        read_tp_links(str(path))
    assert str(e.value) == f"{path}: not a rank's tp_links.json: {said}"
    assert FM.links_for_rank(1, tmp_path, {}, []) is None


def test_a_rank_with_no_collective_has_an_empty_list_of_links(tmp_path):
    doc = read_tp_links(str(_write(tmp_path, {**RANK, "links": []})))
    assert doc["links"] == [] and doc["peer_ranks"] == [0]
    assert FM.links_for_rank(1, tmp_path, {}, []) == []


def test_a_build_without_the_file_has_no_links_for_the_rank(tmp_path):
    assert FM.links_for_rank(1, tmp_path, {}, []) is None
