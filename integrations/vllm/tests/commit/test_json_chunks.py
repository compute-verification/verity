"""`commit.json_chunks`: a list or a dict's last list encoded a chunk of items at a time, to the bytes one `json.dumps` writes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from verity_vllm.commit import json_chunks as JC

ENTRIES = [{"identity": {"request_id": "alice", "step": 3, "op_path": "model.layers.0.mlp", "element_range": [0, 12]},
            "kind": "leaf", "encoding": {"dtype": "bfloat16", "shape": [4, 3]}, "zeta": 1.5e-7, "name": "naïve\n\"q\""},
           {"kind": "registered-input", "digest": Path("/w/x"), "b": None, "a": [True, False, {}]},
           {}, [], "s", 7]
COMPACT = {"separators": (",", ":"), "default": str}


@pytest.mark.parametrize("n", [0, 1, 2, 5, len(ENTRIES)])
@pytest.mark.parametrize("chunk", [1, 2, 4096])
@pytest.mark.parametrize("kw", [COMPACT | {"sort_keys": True}, COMPACT, {"default": str}])
def test_a_list_in_pieces_is_the_bytes_of_one_dumps(n, chunk, kw):
    items = ENTRIES[:n]
    assert "".join(JC.list_pieces(items, chunk=chunk, **kw)) == json.dumps(items, **kw)
    assert JC.sha256_hex(JC.list_pieces(items, chunk=chunk, **kw)) == hashlib.sha256(json.dumps(items, **kw).encode()).hexdigest()


@pytest.mark.parametrize("head", [{}, {"schema": "binding-map/1", "entries_n": 6, "digest": "ab"}])
@pytest.mark.parametrize("chunk", [1, 3])
def test_a_dict_in_pieces_is_the_bytes_of_one_dumps_with_its_list_last(head, chunk):
    doc = {"entries": ENTRIES, **head}
    want = json.dumps({k: v for k, v in doc.items() if k != "entries"} | {"entries": doc["entries"]}, **COMPACT)
    assert "".join(JC.dict_pieces(doc, "entries", chunk=chunk, **COMPACT)) == want


def test_what_pieces_cannot_write_is_refused():
    with pytest.raises(ValueError, match="indent"):
        list(JC.list_pieces(ENTRIES, indent=1))
    with pytest.raises(ValueError, match="sort_keys"):
        list(JC.dict_pieces({"entries": []}, "entries", sort_keys=True))
