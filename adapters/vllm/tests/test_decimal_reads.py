"""A digit `int` does not read is no number: one case for each kind of place in `verity_vllm` that asks whether a string
is one (`tests/lint/test_isdecimal.py` holds the rule for every place). `²` is a digit to `str.isdigit` and none to
`str.isdecimal`, and `int("²")` raises a bare `ValueError`. A place that refuses "no number" by name refuses `²` by that
name, one that answers None answers None, and one that sorts by the number sorts it where it sorts a name with none.
A place that reads bytes asks their text: `bytes` has no `isdecimal`, and its `²` is a byte no number is written with."""
from __future__ import annotations

import asyncio

import pytest
from verity.errors import InvalidArtifact

from verity_vllm.check.replay import driver
from verity_vllm.commitments.leaves import parse_shape_text
from verity_vllm.pipeline import ids_front

NO_DECIMAL = "²"


def test_the_digit_is_one_int_does_not_read() -> None:
    assert NO_DECIMAL.isdigit() and not NO_DECIMAL.isdecimal()
    with pytest.raises(ValueError, match="invalid literal for int"):
        int(NO_DECIMAL)


@pytest.mark.parametrize("text", [f"[{NO_DECIMAL}]", f"[4,{NO_DECIMAL}]", f"[{NO_DECIMAL}8]"])
def test_a_place_that_refuses_no_number_by_name_refuses_it_by_that_name(text: str) -> None:
    with pytest.raises(InvalidArtifact, match="malformed shape text"):
        parse_shape_text(text)


@pytest.mark.parametrize(("limit", "current"), [(NO_DECIMAL, "400"), ("1000", NO_DECIMAL), (f"1{NO_DECIMAL}", "400")])
def test_a_place_that_answers_none_for_no_number_answers_none(tmp_path, monkeypatch, limit: str, current: str) -> None:
    lim, cur = tmp_path / "memory.max", tmp_path / "memory.current"
    lim.write_text(limit + "\n")                    # written as the place reads it, in the default encoding
    cur.write_text(current + "\n")
    monkeypatch.setattr(driver, "CGROUP_MEMORY", (str(lim), str(cur)))
    assert driver._cgroup_memory() is None
    lim.write_text("1000\n")
    cur.write_text("400\n")
    assert driver._cgroup_memory() == (1000, 400)


def test_a_place_that_sorts_by_the_number_sorts_it_with_the_names_that_have_none() -> None:
    pytest.importorskip("torch")                     # the module of the place imports it; the two places above need none
    from verity_vllm.pipeline import commit

    rows = {0: [("r2", 0, 1)], 1: [(f"r{NO_DECIMAL}", 0, 1)], 2: [("r1", 0, 1)], 3: [("rx", 0, 1)]}
    schedule = commit.engine_schedule(rows, {}, {})
    assert list(schedule["observed_arrive_steps"]) == [f"r{NO_DECIMAL}", "rx", "r1", "r2"]
    assert schedule["observed_arrive_steps"] == {"r2": 0, f"r{NO_DECIMAL}": 1, "r1": 2, "rx": 3}


@pytest.mark.parametrize("length", [NO_DECIMAL.encode("latin-1"), NO_DECIMAL.encode(), b"4" + NO_DECIMAL.encode("latin-1"), b"x", b""])
def test_a_place_that_reads_bytes_takes_no_length_from_a_byte_that_is_no_ascii_digit(length: bytes) -> None:
    body = b"abcd"

    async def first(stated: bytes) -> bytes:
        reader = asyncio.StreamReader()
        reader.feed_data(ids_front.COMPLETIONS + b"\r\ncontent-length: " + stated + ids_front.BLANK + body)
        reader.feed_eof()
        return await ids_front._first(reader)

    assert asyncio.run(first(b"4")).endswith(ids_front.BLANK + body)             # a length: the head and the body
    assert asyncio.run(first(length)).endswith(b"content-length: " + length + ids_front.BLANK)   # none: the head alone
