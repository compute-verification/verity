"""A digit `int` does not read is no number: one case for each kind of place in `verity_vllm` that asks whether a string
is one (`tests/lint/test_isdecimal.py` holds the rule for every place). `²` is a digit to `str.isdigit` and none to
`str.isdecimal`, and `int("²")` raises a bare `ValueError`.
A place that reads bytes asks their text: `bytes` has no `isdecimal`, and its `²` is a byte no number is written with."""
from __future__ import annotations

import asyncio

import pytest

from verity.protocols.accounting.network.prover import id_relay

NO_DECIMAL = "²"


def test_the_digit_is_one_int_does_not_read() -> None:
    assert NO_DECIMAL.isdigit() and not NO_DECIMAL.isdecimal()
    with pytest.raises(ValueError, match="invalid literal for int"):
        int(NO_DECIMAL)


@pytest.mark.parametrize("length", [NO_DECIMAL.encode("latin-1"), NO_DECIMAL.encode(), b"4" + NO_DECIMAL.encode("latin-1"), b"x", b""])
def test_a_place_that_reads_bytes_takes_no_length_from_a_byte_that_is_no_ascii_digit(length: bytes) -> None:
    body = b"abcd"

    async def first(stated: bytes) -> bytes:
        reader = asyncio.StreamReader()
        reader.feed_data(id_relay.COMPLETIONS + b"\r\ncontent-length: " + stated + id_relay.BLANK + body)
        reader.feed_eof()
        return await id_relay._first(reader)

    assert asyncio.run(first(b"4")).endswith(id_relay.BLANK + body)             # a length: the head and the body
    assert asyncio.run(first(length)).endswith(b"content-length: " + length + id_relay.BLANK)   # none: the head alone
