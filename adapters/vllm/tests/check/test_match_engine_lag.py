"""The rows a request runs after its EOS sample, as the record attests them: `verity_vllm.check.match.declaration.eos_lag_rows`
over `verity_vllm.check.match.declaration.record_engine_facts`' result.  The attestation is `async_scheduling`, written as a boolean
or as its word; the count is read from it and from nothing else, and the answer names every location that states it."""
from __future__ import annotations

import pytest

from verity_vllm.check.match import declaration as GD


@pytest.mark.parametrize("stated, rows", [(True, 1), (False, 0), ("true", 1), (" False ", 0)])
def test_the_lag_is_the_attested_flag_as_a_boolean_or_its_word(stated, rows):
    facts = GD.record_engine_facts({"engine_kwargs": {"async_scheduling": stated}})
    assert GD.eos_lag_rows(facts) == (rows, f"async_scheduling={stated!r} (RunHeader.engine_kwargs)")


def test_the_answer_names_every_location_that_states_the_flag():
    facts = GD.record_engine_facts({"engine_kwargs": {"async_scheduling": "true"}, "profile": {"engine_kwargs": {"async_scheduling": "true"}}})
    assert facts["sources"]["async_scheduling"] == ["RunHeader.profile.engine_kwargs", "RunHeader.engine_kwargs"]
    assert GD.eos_lag_rows(facts) == (1, "async_scheduling='true' (RunHeader.profile.engine_kwargs, RunHeader.engine_kwargs)")


def test_a_record_that_does_not_state_the_flag_attests_no_lag():
    facts = GD.record_engine_facts({"engine_kwargs": {"enforce_eager": True}})
    assert "async_scheduling" not in facts["values"] and GD.eos_lag_rows(facts) == (None, "not attested")


@pytest.mark.parametrize("stated", ["yes", 1])
def test_a_flag_that_is_neither_a_boolean_nor_its_word_gives_no_lag(stated):
    assert GD.eos_lag_rows(GD.record_engine_facts({"engine_kwargs": {"async_scheduling": stated}})) == (None, f"async_scheduling={stated!r} is not a boolean")


EMPTY: list[object] = [None, {}, [], 0, "", False]


@pytest.mark.parametrize("block", EMPTY)
def test_facts_whose_values_are_left_out_null_or_empty_attest_no_lag(block):
    assert GD.eos_lag_rows({"sources": {}}) == GD.eos_lag_rows({"values": block, "sources": {}}) == (None, "not attested")


@pytest.mark.parametrize("block", EMPTY)
def test_a_flag_whose_sources_are_left_out_null_or_empty_is_read_and_names_no_location(block):
    values = {"async_scheduling": [True]}
    assert GD.eos_lag_rows({"values": values}) == GD.eos_lag_rows({"values": values, "sources": block}) == (1, "async_scheduling=True ()")
    assert GD.eos_lag_rows({"values": {"async_scheduling": ["false"]}, "sources": block}) == (0, "async_scheduling='false' ()")
