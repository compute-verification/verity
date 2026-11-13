"""`verity_vllm.engine.vllm_adapter.load_workload` on a workload file that leaves out an optional block (`sampling`,
`variants`, `sweep`), holds it as null or holds it empty: the greedy run outside a sweep, and a variant the file does not
hold refused by name."""
from __future__ import annotations

import json
from typing import Any

import pytest

BASE: dict[str, Any] = {"schema": "verity-vllm/workload/v2", "requests": [{"request_id": "a", "prompt_token_ids": [1, 2], "max_tokens": 4}]}
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "logprobs": None, "seed": None, "n": 1, "variant": "greedy", "greedy": True}
ROW = "m__bf16__h200__tp1__b1__i8__o4__mixed__greedy__bi-eager"


def _load(tmp_path, doc):
    va = pytest.importorskip("verity_vllm.engine.vllm_adapter")
    p = tmp_path / "wl.json"
    p.write_text(json.dumps(doc))
    return va.load_workload(p)


@pytest.mark.parametrize("blocks", [
    {},
    {"sampling": None, "variants": None, "sweep": None},
    {"sampling": {}, "variants": {}, "sweep": {}},
    {"sampling": [], "variants": [], "sweep": []},
    {"sampling": "", "variants": 0, "sweep": False},
], ids=["absent", "null", "empty", "empty lists", "other falsy values"])
def test_a_workload_without_its_optional_blocks_is_the_greedy_run_outside_a_sweep(tmp_path, blocks):
    wl = _load(tmp_path, {**BASE, **blocks})
    assert wl["sampling"] == GREEDY
    assert wl["row_id"] is None and wl["execution"] is None
    assert wl["requests"][0]["seed"] is None and wl["requests"][0]["n"] == 1


@pytest.mark.parametrize("sweep", [{}, {"sweep": None}, {"sweep": "not a block"}], ids=["absent", "null", "not a block"])
def test_without_a_sweep_block_the_row_id_is_the_top_level_one(tmp_path, sweep):
    wl = _load(tmp_path, {**BASE, **sweep, "row_id": ROW})
    assert (wl["row_id"], wl["execution"]) == (ROW, "bi-eager")


def test_a_sweep_block_owns_the_row_id_even_when_it_holds_none(tmp_path):
    wl = _load(tmp_path, {**BASE, "sweep": {"row_id": ROW}, "row_id": "another__bi-compiled"})
    assert (wl["row_id"], wl["execution"]) == (ROW, "bi-eager")
    wl = _load(tmp_path, {**BASE, "sweep": {}, "row_id": ROW})
    assert wl["row_id"] is None            # an empty block is still the block: the top-level id is not read for the record
    assert wl["execution"] == "bi-eager"    # `execution_of_workload` falls back to it


@pytest.mark.parametrize("variants", [{}, {"variants": None}, {"variants": {}}, {"variants": {"greedy": {"temperature": 0.0}}}],
                         ids=["absent", "null", "empty", "without it"])
def test_a_variant_the_file_does_not_hold_is_refused_by_name(tmp_path, variants):
    with pytest.raises(KeyError, match="sampling variant 'sampled' is not in `variants`"):
        _load(tmp_path, {**BASE, **variants, "sampling": {"variant": "sampled"}})
    mixed = {**BASE, **variants, "requests": [{**BASE["requests"][0], "variant": "sampled"}]}
    with pytest.raises(KeyError, match="request a variant 'sampled' is not in `variants`"):
        _load(tmp_path, mixed)


def test_a_variant_the_file_holds_gives_the_run_and_a_request_its_parameters(tmp_path):
    doc = {**BASE, "sampling": {"variant": "sampled", "seed": 7, "n": 2},
           "variants": {"sampled": {"temperature": 0.8, "top_p": 0.9, "top_k": 40}, "greedy": {"temperature": 0.0}},
           "requests": [{**BASE["requests"][0], "variant": "greedy"}]}
    wl = _load(tmp_path, doc)
    assert wl["sampling"] == {"temperature": 0.8, "top_p": 0.9, "top_k": 40, "logprobs": None, "seed": 7, "n": 2, "variant": "sampled",
                              "greedy": False,
                              "per_request": [{"request_id": "a", "variant": "greedy", "temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 7}]}
    assert (wl["requests"][0]["variant"], wl["requests"][0]["temperature"], wl["requests"][0]["n"]) == ("greedy", 0.0, 2)
