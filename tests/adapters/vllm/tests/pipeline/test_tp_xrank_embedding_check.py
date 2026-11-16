"""`verity_vllm.pipeline.tp.xrank_dump.embedding_check`: a rank's own check of a pick at its embedding site, which the
cross-rank pass carries as the pick's ``emb`` (`check.match.xrank_collectives.check_picks` reads ``ok``, ``n_diff`` and
``why``).  The rank's committed partial rows are the masked gather of the committed ids from its vocabulary shard: a row
whose id is in the rank's range is the shard's row, and any other row is zero.  A check that cannot be evaluated says why
and answers neither way.  CPU-only: the module is a shard of a few rows and ``tensor_of`` reads a table."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from verity_vllm.pipeline.tp import xrank_dump

SITE = "model.embed_tokens"
STEP = 3
#: the rank's shard: ids 2 to 5 of the vocabulary, four rows of three words
LO, HI = 2, 6
SHARD = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3).to(torch.bfloat16)
#: the step's committed ids: two are outside the rank's range (1 below it, 6 at its end)
IDS = torch.tensor([1, 2, 5, 6, 3])
#: what the rank commits for them
PARTIAL = torch.stack([torch.zeros(3), SHARD[0].float(), SHARD[3].float(), torch.zeros(3), SHARD[1].float()]).to(torch.bfloat16)
MISMATCH = "committed partial != masked gather from this rank's shard"


def _module(weight: torch.Tensor = SHARD, **shard: int) -> SimpleNamespace:
    return SimpleNamespace(weight=weight, shard_indices=SimpleNamespace(**{"org_vocab_start_index": LO, "org_vocab_end_index": HI, **shard}))


def _check(part: Any = PARTIAL, *, rows: list[int] | None = None, ids: Any = IDS, module: Any = None, rows_total: int = 5,
           unverified: dict[tuple[int, str], str] | None = None, member: bool = True) -> dict[str, Any]:
    """The check of a pick of ``rows`` (every row unless given) of a partial of ``rows_total`` rows; ``ids`` is what the opened
    ``<site>/in`` member reads as (None: its bytes could not be read), and ``member`` whether the step's layout has it."""
    rows = list(range(rows_total)) if rows is None else rows
    meta = SimpleNamespace(name=f"{SITE}/in")
    asked: list[tuple[int, Any]] = []

    def tensor_of(step: int, m: Any) -> Any:
        asked.append((step, m))
        return ids

    out = xrank_dump.embedding_check({"site": SITE, "rows": rows, "rows_total": rows_total}, {meta.name: meta} if member else {}, STEP,
                                     torch.tensor(rows, dtype=torch.long), part, {SITE: _module() if module is None else module},
                                     tensor_of, unverified or {})
    assert asked == ([(STEP, meta)] if member else [])            # the ids are read once, at the pick's step, and nothing else is
    return out


def test_a_partial_that_is_the_masked_gather_from_the_ranks_shard_passes_and_says_what_it_read():
    assert _check() == {"ok": True, "n_diff": 0, "vocab_range": [LO, HI], "masked_rows": 2, "rows": 5, "why": None}


def test_a_pick_of_some_rows_is_held_to_those_rows_ids():
    """The partial is the picked rows and the ids member the step's every row: the ids are cut to the pick."""
    assert _check(PARTIAL[[1, 4]], rows=[1, 4]) == {"ok": True, "n_diff": 0, "vocab_range": [LO, HI], "masked_rows": 0, "rows": 2, "why": None}
    assert _check(PARTIAL[[0, 2]], rows=[0, 2])["masked_rows"] == 1
    assert _check(PARTIAL[[1, 4]], rows=[1, 2])["ok"] is False      # rows 1 and 4 are not rows 1 and 2


def test_ids_that_are_not_one_a_row_of_the_partial_are_read_whole():
    """A partial that was not cut to the pick (its row count is not the step's) is held to every id."""
    assert _check(PARTIAL, rows=[1, 4]) == {"ok": True, "n_diff": 0, "vocab_range": [LO, HI], "masked_rows": 2, "rows": 5, "why": None}
    assert _check(PARTIAL[:4], rows=[0, 1, 2, 3], ids=IDS[:4], rows_total=5)["rows"] == 4
    # ids that are the pick's already (two of the step's five) are not cut again
    assert _check(PARTIAL[[1, 4]], rows=[1, 4], ids=IDS[[1, 4]]) == {"ok": True, "n_diff": 0, "vocab_range": [LO, HI], "masked_rows": 0, "rows": 2, "why": None}


def test_ids_of_any_shape_are_read_in_row_order():
    assert _check(ids=IDS.reshape(5, 1))["ok"] is True
    assert _check(ids=IDS.flip(0))["ok"] is False


@pytest.mark.parametrize("row, word, n_diff", [(1, 0, 1), (4, 2, 1), (0, 1, 1), (3, 0, 1)],
                         ids=["a word of a row in range", "its last word", "a row below the range that is not zero",
                              "a row at the range's end that is not zero"])
def test_a_partial_with_one_word_changed_fails_and_counts_it(row, word, n_diff):
    part = PARTIAL.clone()
    part[row, word] = 99.0
    assert _check(part) == {"ok": False, "n_diff": n_diff, "vocab_range": [LO, HI], "masked_rows": 2, "rows": 5, "why": MISMATCH}


def test_the_range_is_the_modules_own():
    """Another rank's range over the same shard: the first row is in it and the third is not."""
    moved = _module(org_vocab_start_index=1, org_vocab_end_index=5)
    got = _check(module=moved)
    assert (got["ok"], got["vocab_range"], got["masked_rows"]) == (False, [1, 5], 2)
    part = torch.stack([SHARD[0].float(), SHARD[1].float(), torch.zeros(3), torch.zeros(3), SHARD[2].float()]).to(torch.bfloat16)
    assert _check(part, module=moved)["ok"] is True


def test_the_words_differing_are_counted_for_bf16_only():
    """A shard of another element type is compared whole and no count is given."""
    wide, part = _module(SHARD.float()), PARTIAL.float()
    assert _check(part, module=wide) == {"ok": True, "n_diff": None, "vocab_range": [LO, HI], "masked_rows": 2, "rows": 5, "why": None}
    part[1, 0] = 99.0
    assert _check(part, module=wide) == {"ok": False, "n_diff": None, "vocab_range": [LO, HI], "masked_rows": 2, "rows": 5, "why": MISMATCH}


def test_a_partial_of_another_element_type_does_not_pass_though_its_values_are_the_shards():
    wide = _module(SHARD.float())
    assert _check(PARTIAL, module=wide) == {"ok": False, "n_diff": None, "vocab_range": [LO, HI], "masked_rows": 2, "rows": 5, "why": MISMATCH}
    assert _check(PARTIAL.float())["ok"] is not True


def test_a_partial_of_another_shape_fails_with_no_count():
    assert _check(PARTIAL[:, :2]) == {"ok": False, "n_diff": None, "vocab_range": [LO, HI], "masked_rows": 2, "rows": 5, "why": MISMATCH}


@pytest.mark.parametrize("given, why", [
    ({"member": False}, "no committed ids member <site>/in"),
    ({"ids": None}, "no committed ids member <site>/in"),
    ({"ids": None, "unverified": {(STEP, f"{SITE}/in"): "opened value not verified: a leaf"}}, "opened value not verified: a leaf"),
    ({"ids": None, "unverified": {(STEP + 1, f"{SITE}/in"): "another step's"}}, "no committed ids member <site>/in"),
    ({"module": SimpleNamespace(weight=SHARD)}, "no vocabulary shard_indices on the module"),
    ({"part": None}, "partial not committed"),
    ({"ids": None, "part": None, "module": SimpleNamespace(weight=SHARD)}, "no committed ids member <site>/in"),
    ({"part": None, "module": SimpleNamespace(weight=SHARD)}, "no vocabulary shard_indices on the module"),
], ids=["no ids member in the layout", "ids not retained", "ids opened and not verified", "another step's ids not verified",
        "a module that is no shard", "no partial", "nothing: the ids are named first", "no shard and no partial: the shard is named"])
def test_a_check_that_cannot_be_evaluated_answers_neither_way_and_says_why(given, why):
    assert _check(**given) == {"ok": None, "why": why}


def test_a_site_with_no_module_is_no_shard():
    meta = SimpleNamespace(name=f"{SITE}/in")
    got = xrank_dump.embedding_check({"site": SITE, "rows": [0], "rows_total": 1}, {meta.name: meta}, STEP, torch.tensor([0]), PARTIAL[:1], {},
                                     lambda step, m: IDS[:1], {})
    assert got == {"ok": None, "why": "no vocabulary shard_indices on the module"}


def test_an_error_inside_the_check_is_its_answer_and_not_the_ranks():
    """A shard that does not say where it ends: the check is not evaluable, by the error's own words, cut to 200 characters."""
    got = _check(module=SimpleNamespace(weight=SHARD, shard_indices=SimpleNamespace(org_vocab_start_index=LO)))
    assert got["ok"] is None and got["why"].startswith("embedding-shard check error: AttributeError(") and "org_vocab_end_index" in got["why"]
    long = _check(module=SimpleNamespace(weight=SHARD, shard_indices=SimpleNamespace(org_vocab_start_index="x" * 400, org_vocab_end_index=HI)))
    assert long["ok"] is None and len(long["why"]) == 200 and long["why"].startswith("embedding-shard check error: ValueError(")
