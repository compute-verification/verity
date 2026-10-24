"""REDTEAM phase 17, Target 4 (#3855, WARDEN): x-verity-release on the server's copy of a request (`release`). The PR's
own tests drop a client's release header placed among the first headers and read the first of each; these push on whether
a client header can survive by sitting elsewhere (last, many duplicates) or by smuggling a release line into the body
(where a real CRLFCRLF ends the head), and confirm the server's N is always the one read back.

The warden-level cases the lead named — N dense and in the signed END-cell order even when ENDs release out of ordinal
order, a connection that fails before its END taking no number, and the cells/records unchanged by the forward-whole-at-END
— are already exercised by the PR's own warden tests (`test_the_server_gets_each_request_whole_with_its_place_among_the
_signed_ends`, `test_a_connection_that_fails_before_its_end_is_never_numbered_and_its_server_gets_nothing`), so this file
does not repeat them."""
from __future__ import annotations

import pytest

from verity.core.protocols.certifier.release import RELEASE_HEADER, RUN_HEADER, release_head, release_of, with_release

RUN = bytes(range(32))


def http(body: bytes, extra: bytes = b"") -> bytes:
    return b"POST /v1/completions HTTP/1.1\r\nhost: s\r\n%scontent-length: %d\r\n\r\n%s" % (extra, len(body), body)


def test_a_client_release_header_is_stripped_wherever_it_sits_and_however_many():
    """A client release header first, last, or in several copies at once is dropped; the server's N is inserted once,
    right after the request line, and read back as the first of each."""
    forged = (b"x-verity-release: 11\r\n"
              b"X-VERITY-RELEASE: 22\r\n"
              b"x-verity-release-run: " + bytes(32).hex().encode() + b"\r\n")
    tail = b"x-verity-release:\t999\r\n"              # a last-position client header, tab after the colon
    req = http(b"{}", extra=forged) + b""            # forged sit among the first headers
    req = req.replace(b"content-length:", tail + b"content-length:")  # and one just before content-length
    got = with_release(req, 5, RUN)
    assert got.count(RELEASE_HEADER + b":") == 1 and got.count(RELEASE_HEADER.upper() + b":") == 0
    assert got.count(RUN_HEADER + b":") == 1
    assert release_of(got) == (5, RUN)
    for line in (b"x-verity-release: 11", b"X-VERITY-RELEASE: 22", b"x-verity-release:\t999",
                 b"x-verity-release-run: " + bytes(32).hex().encode()):
        assert line not in got                       # every client release line, whatever its case or position, is gone
    assert b"x-verity-release: 5" in got and b"x-verity-release-run: " + RUN.hex().encode() in got


def test_a_release_line_smuggled_into_the_body_stays_in_the_body(tmp_path=None):
    """The body carries a real CRLFCRLF and a line that looks like a release header after it. `_split` ends the head at
    the first blank line, so the smuggled line is body: it is left untouched, the server's N sits in the head, and the
    read-back N is the server's, not 999."""
    body = b'{"prompt":"x"}\r\n\r\nx-verity-release: 999\r\n\r\ntrailing'
    req = http(body)
    got = with_release(req, 7, RUN)
    head, _, tail = got.partition(b"\r\n\r\n")
    assert release_of(got) == (7, RUN)
    assert tail == body                                    # the body, blank lines and smuggled line, byte for byte
    assert head.count(RELEASE_HEADER + b":") == 1          # exactly the server's, in the head
    assert b"999" in tail and b"999" not in head


def test_a_client_run_header_without_a_release_header_is_dropped_too():
    """A client that sends only `x-verity-release-run` (no release) has it dropped; `with_release` adds the server's pair,
    and without a run the server's is release-only."""
    req = http(b"{}", extra=b"x-verity-release-run: " + bytes(16).hex().encode() + b"\r\n")
    got = with_release(req, 3)
    assert got.count(RUN_HEADER + b":") == 0 and release_of(got) == (3, None)
    assert with_release(req, 3, RUN).count(RUN_HEADER + b":") == 1
    assert release_of(with_release(req, 3, RUN)) == (3, RUN)


def test_the_request_is_unchanged_but_for_the_inserted_headers_and_the_clients_dropped():
    """`with_release` only inserts the server's header lines after the request line and removes the client's release
    headers; everything else — the other headers' order and bytes, and the body — is preserved."""
    req = http(b'{"a":1}', extra=b"x-verity-release: 42\r\nuser-agent: ua\r\n")
    got = with_release(req, 8, RUN)
    line, rest = req.split(b"\r\n", 1)
    want = line + b"\r\n" + release_head(8, RUN) + rest.replace(b"x-verity-release: 42\r\n", b"")
    assert got == want
    assert got.endswith(b"\r\n\r\n" + b'{"a":1}') and b"user-agent: ua" in got
