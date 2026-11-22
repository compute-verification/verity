"""SPOT's levels (`levels`): the auditor's count read off the parameters' shape, the prover's off its record, and each
refusal by its name."""
from __future__ import annotations

import json

import pytest

from verity.protocols.proof.spot.verifier import levels as LV

ONE = {"proofs": {}, "partition": {}, "sampling_design": "rate:1/2"}
TWO = {"boundary": {}, "interior": {}, "rule": {}, "delta": "1/2"}


def refused(reason: str, fn, *args):
    with pytest.raises(LV.Refused) as e:
        fn(*args)
    assert e.value.reason == reason, e.value
    return str(e.value)


@pytest.mark.parametrize("value,want", [(1, 1), (2, 2)])
def test_one_and_two_levels_parse(value, want):
    assert LV.parse(value, "x") == want


def test_three_levels_and_more_are_refused_by_name():
    for n in (3, 7):
        assert "only 1 or 2 levels are built" in refused("levels-not-built", LV.parse, n, "x")


@pytest.mark.parametrize("value", [True, 1.0, "1", " 2 ", "two", "", None, [2], {"n": 2}])
def test_a_non_integer_is_refused(value):
    refused("levels-not-integer", LV.parse, value, "x")


def test_zero_is_refused():
    refused("levels-not-positive", LV.parse, 0, "x")


def test_the_parameters_shape_decides_and_no_levels_reads_as_one_level():
    assert LV.of_params(ONE) == 1
    assert LV.of_params({**ONE, "levels": 1}) == 1
    assert LV.of_params({**TWO, "levels": 2}) == 2
    assert "name 1 level(s), and their shape is 2" in refused("levels-params-mismatch", LV.of_params, TWO)
    refused("levels-params-mismatch", LV.of_params, {**ONE, "levels": 2})
    refused("levels-shape", LV.of_params, {**ONE, "boundary": {}})
    refused("levels-not-built", LV.of_params, {**ONE, "levels": 3})


def test_the_parameters_levels_as_a_string_is_refused():
    refused("levels-not-integer", LV.of_params, {**ONE, "levels": "1"})
    refused("levels-not-integer", LV.of_params, {**TWO, "levels": "2"})


def test_a_records_levels_as_a_string_is_refused():
    refused("levels-not-integer", LV.served, {"levels": "1"})
    refused("levels-not-integer", LV.served, {"format": LV.BOUNDARY_RECORD, "levels": "2"})


def test_a_boundary_record_is_two_levels_by_its_format_and_refuses_another_count():
    assert LV.served({"format": LV.BOUNDARY_RECORD, "levels": 2}) == 2
    why = refused("levels-format-mismatch", LV.served, {"format": LV.BOUNDARY_RECORD, "levels": 1})
    assert "levels is 1" in why and LV.BOUNDARY_RECORD in why
    refused("levels-not-built", LV.served, {"format": LV.BOUNDARY_RECORD, "levels": 3})


def test_the_record_is_the_provers_and_a_difference_is_refused():
    assert LV.served({}) == 1
    assert LV.served({"format": LV.BOUNDARY_RECORD}) == 2
    assert LV.served({"levels": 2}) == 2
    assert LV.check(ONE, {}) == 1
    two = {**TWO, "levels": 2}
    assert LV.check(two, {"format": LV.BOUNDARY_RECORD}) == 2
    assert "served 1 level(s), and the parameters check 2" in refused("levels-record-mismatch", LV.check, two, {})
    refused("levels-record-mismatch", LV.check, ONE, {"levels": 2})
    refused("levels-not-built", LV.served, {"levels": 3})


def test_a_command_refuses_a_count_it_cannot_run_rather_than_running_another():
    assert LV.command(None, (1,), 1) == 1
    assert LV.command("", (2,), 2) == 2
    assert LV.command(" 2 ", (2,), 1) == 2
    assert "SPOT_LEVELS=2: this command runs 1 level(s)" in refused("levels-no-path", LV.command, "2", (1,), 1)
    refused("levels-no-path", LV.command, "1", (2,), 2)
    refused("levels-not-built", LV.command, "3", (1, 2), 1)
    refused("levels-not-integer", LV.command, "x", (1,), 1)


def canon(o: dict) -> bytes:
    return json.dumps(o, sort_keys=True, separators=(",", ":")).encode()


def test_with_levels_names_the_levels_and_refuses_a_case_naming_them():
    case = {"params": ONE, "evidence": canon({"proofs": []}).hex(), "commitment": "00"}
    got = LV.with_levels(case, 1, canon)
    assert got["params"]["levels"] == 1 and json.loads(bytes.fromhex(got["evidence"]))["levels"] == 1
    refused("levels-named", LV.with_levels, got, 1, canon)
    refused("levels-params-mismatch", LV.with_levels, case, 2, canon)
    refused("levels-not-built", LV.with_levels, case, 3, canon)
