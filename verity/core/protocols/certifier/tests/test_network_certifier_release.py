"""The release order's headers on the server's copy of a request (`release`): inserted after the request line with the
head and body otherwise as the client sent them, a client's own release headers dropped, bytes that aren't an HTTP/1
request with its whole head unchanged, and read back by a shim as the first of each."""

from __future__ import annotations

import pytest

from verity.core.protocols.certifier.release import (RELEASE_HEADER, RUN_HEADER, release_head, release_of,
                                                     with_release)

BODY = b'{"prompt":"a\\r\\n\\r\\nb","max_tokens":4}'
REQUEST = (b"POST /v1/completions HTTP/1.1\r\nhost: server\r\ncontent-type: application/json\r\n"
           b"content-length: %d\r\n\r\n" % len(BODY) + BODY)
RUN = bytes(range(32))


def test_the_headers_go_right_after_the_request_line_and_nothing_else_moves():
    got = with_release(REQUEST, 7, RUN)
    line, rest = REQUEST.split(b"\r\n", 1)
    assert got == line + b"\r\nx-verity-release: 7\r\nx-verity-release-run: " + RUN.hex().encode() + b"\r\n" + rest
    assert with_release(REQUEST, 0) == line + b"\r\nx-verity-release: 0\r\n" + rest
    assert got.endswith(b"\r\n\r\n" + BODY) and len(got) == len(REQUEST) + len(release_head(7, RUN))
    assert release_of(got.partition(b"\r\n\r\n")[0] + b"\r\n\r\n") == (7, RUN)


def test_a_clients_own_release_headers_are_dropped_whatever_their_case():
    forged = REQUEST.replace(b"host: server\r\n", b"X-Verity-Release: 0\r\nhost: server\r\nx-verity-release-RUN :"
                             b" " + bytes(32).hex().encode() + b"\r\nx-verity-released: 1\r\n")
    got = with_release(forged, 3, RUN)
    assert got == with_release(REQUEST.replace(b"host: server\r\n", b"host: server\r\nx-verity-released: 1\r\n"), 3,
                               RUN)
    assert release_of(got) == (3, RUN)


@pytest.mark.parametrize("data", [b"", b"x" * 40 + b"\n", REQUEST.partition(b"\r\n\r\n")[0],
                                  b"\r\n\r\n" + BODY, b"POST /v1/completions HTTP/2\r\nhost: s\r\n\r\n",
                                  b"POST  /v1 HTTP/1.1\r\n\r\n", b"PO ST /v1 HTTP/1.1\r\n\r\n"])
def test_bytes_that_are_not_an_http1_request_with_its_whole_head_pass_unchanged(data):
    assert with_release(data, 5, RUN) == data
    assert release_of(data) is None


def test_release_head_takes_a_non_negative_integer_and_a_non_empty_run():
    assert release_head(12) == RELEASE_HEADER + b": 12\r\n"
    assert release_head(0, b"\xab") == RELEASE_HEADER + b": 0\r\n" + RUN_HEADER + b": ab\r\n"
    for n in (-1, 1.0, True, "3", None):
        with pytest.raises(ValueError):
            release_head(n)
    for run in (b"", "ab", bytearray(b"ab")):
        with pytest.raises(ValueError):
            release_head(0, run)


def test_release_of_reads_the_first_of_each_and_refuses_what_is_not_a_release():
    def head(*fields: bytes) -> bytes:
        return b"POST /v1/completions HTTP/1.1\r\n" + b"".join(f + b"\r\n" for f in fields) + b"\r\n"

    assert release_of(head(b"x-verity-release: 4", b"X-Verity-Release: 9")) == (4, None)
    assert release_of(head(b"X-VERITY-RELEASE:  41 ", b"x-verity-release-run: 00ff")) == (41, b"\x00\xff")
    assert release_of(head(b"x-verity-release: 4", b"x-verity-release-run: zz")) == (4, None)
    assert release_of(head(b"x-verity-release: 4", b"x-verity-release-run: ")) == (4, None)
    for bad in (b"x-verity-release: -1", b"x-verity-release: 1.0", b"x-verity-release: ", b"x-verity-release: \xd9\xa1",
                b"x-verity-release-run: 00"):
        assert release_of(head(bad)) is None
    assert release_of(head()) is None
