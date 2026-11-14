"""The release order's headers on the server's copy of a request (`release`): inserted after the request line with the
head and body otherwise as the client sent them, a client's own release headers dropped, bytes that aren't an HTTP/1
request with its whole head unchanged, and read back by a shim as the first of each; and the watermark stream, one
chunked POST whose lines only advance."""

from __future__ import annotations

import pytest

from verity.protocols.accounting.certifier.verifier.release import (BUCKET_HEADER, RELEASE_HEADER, RUN_HEADER, WATERMARK_END,
                                                     WATERMARK_PATH, one_request, read_watermark, release_head,
                                                     release_of, to_watermark, watermark_chunk, watermark_head,
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
    assert release_of(got.partition(b"\r\n\r\n")[0] + b"\r\n\r\n") == (7, RUN, None)


def test_the_bucket_follows_the_run_and_is_read_back_with_it():
    got = with_release(REQUEST, 7, RUN, 41)
    line, rest = REQUEST.split(b"\r\n", 1)
    assert got == line + b"\r\nx-verity-release: 7\r\nx-verity-release-run: " + RUN.hex().encode() + \
        b"\r\nx-verity-release-bucket: 41\r\n" + rest
    assert release_of(got) == (7, RUN, 41) and with_release(REQUEST, *release_of(got)) == got
    assert with_release(REQUEST, 0, None, 0) == line + b"\r\nx-verity-release: 0\r\nx-verity-release-bucket: 0\r\n" + rest
    assert release_of(with_release(REQUEST, 0, None, 0)) == (0, None, 0)


def test_a_clients_own_release_headers_are_dropped_whatever_their_case():
    forged = REQUEST.replace(b"host: server\r\n", b"X-Verity-Release: 0\r\nhost: server\r\nx-verity-release-RUN :"
                             b" " + bytes(32).hex().encode() + b"\r\nx-verity-released: 1\r\n"
                             b"X-Verity-Release-Bucket: 0\r\n")
    got = with_release(forged, 3, RUN, 9)
    assert got == with_release(REQUEST.replace(b"host: server\r\n", b"host: server\r\nx-verity-released: 1\r\n"), 3,
                               RUN, 9)
    assert release_of(got) == (3, RUN, 9)
    assert release_of(with_release(forged, 3, RUN)) == (3, RUN, None)


@pytest.mark.parametrize("data", [b"", b"x" * 40 + b"\n", REQUEST.partition(b"\r\n\r\n")[0],
                                  b"\r\n\r\n" + BODY, b"POST /v1/completions HTTP/2\r\nhost: s\r\n\r\n",
                                  b"POST  /v1 HTTP/1.1\r\n\r\n", b"PO ST /v1 HTTP/1.1\r\n\r\n"])
def test_bytes_that_are_not_an_http1_request_with_its_whole_head_pass_unchanged(data):
    assert with_release(data, 5, RUN, 2) == data
    assert release_of(data) is None


def test_release_head_takes_non_negative_integers_and_a_non_empty_run():
    assert release_head(12) == RELEASE_HEADER + b": 12\r\n"
    assert release_head(0, b"\xab") == RELEASE_HEADER + b": 0\r\n" + RUN_HEADER + b": ab\r\n"
    assert release_head(3, None, 5) == RELEASE_HEADER + b": 3\r\n" + BUCKET_HEADER + b": 5\r\n"
    for n in (-1, 1.0, True, "3", None):
        with pytest.raises(ValueError, match="a release is a non-negative integer"):
            release_head(n)
        if n is not None:
            with pytest.raises(ValueError, match="a bucket is a non-negative integer"):
                release_head(0, RUN, n)
    for run in (b"", "ab", bytearray(b"ab")):
        with pytest.raises(ValueError, match="a run id is non-empty bytes"):
            release_head(0, run)


def test_release_of_reads_the_first_of_each_and_refuses_what_is_not_a_release():
    def head(*fields: bytes) -> bytes:
        return b"POST /v1/completions HTTP/1.1\r\n" + b"".join(f + b"\r\n" for f in fields) + b"\r\n"

    assert release_of(head(b"x-verity-release: 4", b"X-Verity-Release: 9")) == (4, None, None)
    assert release_of(head(b"X-VERITY-RELEASE:  41 ", b"x-verity-release-run: 00ff")) == (41, b"\x00\xff", None)
    assert release_of(head(b"x-verity-release: 4", b"x-verity-release-run: zz")) == (4, None, None)
    assert release_of(head(b"x-verity-release: 4", b"x-verity-release-run: ")) == (4, None, None)
    assert release_of(head(b"x-verity-release: 4", b"x-verity-release-bucket: 6",
                           b"x-verity-release-bucket: 7")) == (4, None, 6)
    for bad in (b"-1", b"1.0", b"", b"\xd9\xa1", b"x"):
        assert release_of(head(b"x-verity-release: 4", b"x-verity-release-bucket: " + bad)) == (4, None, None)
    for bad in (b"x-verity-release: -1", b"x-verity-release: 1.0", b"x-verity-release: ", b"x-verity-release: \xd9\xa1",
                b"x-verity-release-run: 00", b"x-verity-release-bucket: 3"):
        assert release_of(head(bad)) is None
    assert release_of(head()) is None


def test_the_watermark_stream_is_one_chunked_post_of_one_line_a_tick():
    """The head names the run and a chunked body at `WATERMARK_PATH`; each tick's chunk is its line `t n`, read back
    by an HTTP/1.1 chunked decoder as the lines in order, and the stream ends with the terminating chunk."""
    head = watermark_head(RUN)
    assert head.startswith(b"POST " + WATERMARK_PATH + b" HTTP/1.1\r\n") and head.endswith(b"\r\n\r\n")
    fields = dict(f.split(b": ", 1) for f in head[:-4].split(b"\r\n")[1:])
    assert fields[RUN_HEADER] == RUN.hex().encode() and fields[b"transfer-encoding"] == b"chunked"
    assert watermark_chunk(0, 0) == b"4\r\n0 0\n\r\n" and watermark_chunk(599, 12) == b"7\r\n599 12\n\r\n"
    assert watermark_chunk(3, 10, [2, 9]) == b"9\r\n3 10 2 9\n\r\n"
    stream = b"".join(watermark_chunk(t, n, lost) for t, n, lost in [(0, 0, ()), (1, 2, ()), (2, 2, ()),
                                                                      (3, 10, (2, 9))]) + WATERMARK_END
    lines, rest = [], stream
    while True:
        size, _, rest = rest.partition(b"\r\n")
        if not int(size, 16):
            assert rest == b"\r\n"
            break
        lines.append(rest[:int(size, 16)])
        assert rest[int(size, 16):int(size, 16) + 2] == b"\r\n"
        rest = rest[int(size, 16) + 2:]
    assert lines == [b"0 0\n", b"1 2\n", b"2 2\n", b"3 10 2 9\n"]
    with pytest.raises(ValueError, match="a run id is non-empty bytes"):
        watermark_head(b"")
    for t, n in ((-1, 0), (0, -1), (True, 0), (0, 1.0)):
        with pytest.raises(ValueError, match="is a non-negative integer"):
            watermark_chunk(t, n)
    for lost in ((10,), (3, 3), (5, 2), (-1,)):
        with pytest.raises(ValueError, match="rise below n|is a non-negative integer"):
            watermark_chunk(3, 10, lost)


#: request lines a server routes to the watermark's path
TO_WATERMARK = {
    "the stream's own": b"POST /verity/watermark HTTP/1.1",
    "a fourth word": b"POST /verity/watermark HTTP/1.1 x",
    "a query string": b"POST /verity/watermark?x=1 HTTP/1.1",
    "a fragment": b"POST /verity/watermark#x HTTP/1.1",
    "another method": b"GET /verity/watermark HTTP/1.0",
    "an escaped letter": b"POST /verity/%77atermark HTTP/1.1",
    "an escaped slash": b"POST /verity%2Fwatermark HTTP/1.1",
    "a scheme and authority": b"POST http://server/verity/watermark HTTP/1.1",
    "a scheme, userinfo, port and query": b"POST HTTP://u@h:80/verity/watermark?x HTTP/1.1",
    "line breaks first": b"\r\n\n\rPOST /verity/watermark HTTP/1.1",
    "two spaces before the path": b"POST  /verity/watermark HTTP/1.1",
    "no version": b"POST /verity/watermark",
}
#: request lines to another path, and bytes that open no request
ELSEWHERE = {
    "completions": b"POST /v1/completions HTTP/1.1",
    "a longer path": b"POST /verity/watermarks HTTP/1.1",
    "a trailing slash": b"POST /verity/watermark/ HTTP/1.1",
    "two leading slashes": b"POST //verity/watermark HTTP/1.1",
    "a dot segment": b"POST /verity/./watermark HTTP/1.1",
    "upper case": b"POST /VERITY/WATERMARK HTTP/1.1",
    "an escaped question mark": b"POST /verity/watermark%3Fx HTTP/1.1",
    "an escape of an escape": b"POST /verity/%2577atermark HTTP/1.1",
    "a path parameter": b"POST /verity/watermark;x HTTP/1.1",
    "the path in a query string": b"POST /v1/completions?next=/verity/watermark HTTP/1.1",
    "the path after another": b"POST /x/http://h/verity/watermark HTTP/1.1",
    "the path as an authority": b"POST http://verity/watermark HTTP/1.1",
    "the path in a header": b"POST /v1/completions HTTP/1.1\r\nx: /verity/watermark",
    "the path on a later line": b"x" * 40 + b"\nPOST /verity/watermark HTTP/1.1",
    "one word": b"/verity/watermark",
    "nothing": b"",
}


@pytest.mark.parametrize("name", sorted(TO_WATERMARK))
def test_a_request_to_the_watermarks_path_is_read_as_a_server_reads_its_line(name):
    """The path is the second word of the first line, whatever the method and whatever follows it, less a scheme and
    authority, a query string and a fragment, its escapes decoded, past any line breaks before the line: every form
    one of the servers routes to the watermark, with the rest of the head or cut right after the path."""
    line = TO_WATERMARK[name]
    assert to_watermark(line) and to_watermark(line + b"\r\nhost: s\r\n\r\n0 1 0\n")
    assert not to_watermark(line.replace(b"verity", b"verify"))


@pytest.mark.parametrize("name", sorted(ELSEWHERE))
def test_a_request_to_another_path_is_not_read_as_one_to_the_watermarks(name):
    line = ELSEWHERE[name]
    assert not to_watermark(line) and not to_watermark(line + b"\r\nhost: s\r\n\r\n")


def test_the_watermarks_path_is_read_on_a_line_the_release_headers_pass_unchanged_and_only_in_the_bytes_given():
    """A line with a fourth word is no request line to `with_release`, which passes it with the client's own release
    headers, and one a server still routes by its second word; `to_watermark` reads it. It reads the bytes it is given
    and no further: the path cut a byte short isn't the watermark's, and a second request after the first's body is
    `one_request`'s to read."""
    forged = TO_WATERMARK["a fourth word"] + b"\r\nx-verity-release: 0\r\nx-verity-release-run: 00\r\n\r\n0 1 0\n"
    assert with_release(forged, 5, RUN, 2) == forged and release_of(forged) is None and to_watermark(forged)
    assert to_watermark(watermark_head(RUN)) and to_watermark(with_release(watermark_head(RUN), 5, RUN, 2))
    own = TO_WATERMARK["the stream's own"]
    assert own.endswith(WATERMARK_PATH + b" HTTP/1.1")
    assert to_watermark(own[:-9]) and not to_watermark(own[:-10])
    assert not to_watermark(REQUEST + own + b"\r\n\r\n") and not one_request(REQUEST + own + b"\r\n\r\n")


#: a second request as its sender writes it: to the watermark's path, with release headers of its own
SECOND = b"POST /verity/watermark HTTP/1.1\r\nx-verity-release: 0\r\nx-verity-release-run: 00\r\n\r\n"


def framed(head: bytes, body: bytes = b"{}") -> bytes:
    """A request to the completions path with the fields `head` after its first, then `body`."""
    return b"POST /v1/completions HTTP/1.1\r\nhost: s\r\n" + head + b"\r\n" + body


#: bytes that are at most one request to a server
ONE_REQUEST = {
    "a request and its body": REQUEST,
    "a body cut short": REQUEST[:-1],
    "no body and no length": framed(b"", b""),
    "a length of zero": framed(b"content-length: 0\r\n", b""),
    "zeros before the length": framed(b"content-length: 0002\r\n"),
    "blanks around the length": framed(b"Content-Length:\t 2 \r\n"),
    "a length past any request": framed(b"content-length: " + b"9" * 5000 + b"\r\n"),
    "a second request in the body": framed(b"content-length: %d\r\n" % (4 + len(SECOND)), b"{}\r\n" + SECOND),
    "nothing": b"",
    "one word, then a request": b"x\r\n\r\n" + SECOND,
    "two words and no head's end": b"POST /v1/completions\r\nhost: s\r\n" + SECOND[:-2],
    "bytes of every value": bytes(range(256)) * 2,
}
#: bytes a server reads a second request in, or whose head servers end differently
NOT_ONE_REQUEST = {
    "a second request behind the body": REQUEST + SECOND,
    "a second request behind a head with no length": framed(b"", SECOND),
    "one byte past the body": REQUEST + b"x",
    "a line break past the body": REQUEST + b"\r\n",
    "line feeds for line breaks": framed(b"content-length: 2\r\n").replace(b"\r\n", b"\n"),
    "a line feed in a field": framed(b"x: y\ncontent-length: 2\r\n"),
    "a carriage return in a field": framed(b"x: y\rcontent-length: 2\r\n"),
    "line breaks before the request line": b"\r\n" + REQUEST,
    "a folded field": framed(b"x: y\r\n content-length: 2\r\n"),
    "a space before the colon": framed(b"content-length : 2\r\n"),
    "a field with no colon": framed(b"content-length 2\r\n"),
    "a chunked body": framed(b"transfer-encoding: chunked\r\n", b"2\r\n{}\r\n0\r\n\r\n"),
    "a length and a transfer coding": framed(b"content-length: 2\r\nTransfer-Encoding: chunked\r\n"),
    "two lengths": framed(b"content-length: 2\r\ncontent-length: 2\r\n"),
    "a length with a sign": framed(b"content-length: +2\r\n"),
    "a length with no digits": framed(b"content-length:\r\n", b""),
    "a list of lengths": framed(b"content-length: 2, 2\r\n"),
    "a fourth word on the request line": REQUEST.replace(b"HTTP/1.1", b"HTTP/1.1 x", 1),
    "two blanks on the request line": REQUEST.replace(b"POST ", b"POST  ", 1),
    "another version": b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n",
    "no version": REQUEST.replace(b" HTTP/1.1", b"", 1),
}


@pytest.mark.parametrize("name", sorted(ONE_REQUEST))
def test_bytes_that_are_at_most_one_request_to_a_server_are_read_as_one(name):
    """A request whose head `with_release` reads, with no byte past the body its one length gives it: the length is
    counted, so a second request inside the body is the body's, and one longer than any request is never converted.
    And bytes that open no request: a first line of one word, or no head's end."""
    assert one_request(ONE_REQUEST[name])


@pytest.mark.parametrize("name", sorted(NOT_ONE_REQUEST))
def test_bytes_a_server_may_read_a_second_request_in_are_not_one_request(name):
    """A byte past the first request's body, with a length or without, and every head `with_release` doesn't read or
    servers end differently: line breaks that aren't CRLF or come first, a field folded or with no colon right after
    its name, a transfer coding, more than one length or one that is no decimal count, a request line of another
    form. Each is read on bytes that open a request: a first line of two words or more and a head's end."""
    assert not one_request(NOT_ONE_REQUEST[name])


def test_a_request_read_as_one_gets_the_release_headers_and_none_of_its_senders():
    """Each request `one_request` takes is one `with_release` puts the headers in, dropping the client's own: none
    reaches a server with a release header its sender wrote in its head."""
    requests = [name for name, data in ONE_REQUEST.items() if data.startswith(b"POST /v1/completions HTTP/1.1\r\n")]
    assert len(requests) == 8
    for name in requests:
        forged = ONE_REQUEST[name].replace(b"\r\nhost:", b"\r\nX-Verity-Release: 9\r\nhost:", 1)
        assert one_request(forged) and release_of(with_release(forged, 5, RUN, 2)) == (5, RUN, 2), name


def test_a_watermark_line_is_read_only_when_it_advances():
    """Each line is t one past the last line's, from 0, with n at least the last's, and the lost releases it names
    rising from the last line's n to below its own: one that skips or repeats a bucket, lowers n, or names a release
    of another bucket, out of order or twice, is refused, as is anything but decimal counts, single spaces and a
    newline."""
    last = None
    for t, n, lost in [(0, 0, ()), (1, 0, ()), (2, 3, (0, 2)), (3, 3, ()), (4, 9, (3,)), (5, 9, ())]:
        line = b"%d %d" % (t, n) + b"".join(b" %d" % k for k in lost) + b"\n"
        last = read_watermark(line, last)
        assert last == (t, n, lost)
    for line in (b"7 9\n", b"5 9\n", b"6 8\n", b"4 9\n"):
        with pytest.raises(ValueError, match="t steps by 1 from 0 and n never decreases"):
            read_watermark(line, last)
    for line in (b"6 12 8\n", b"6 12 12\n", b"6 12 10 9\n", b"6 12 10 10\n"):
        with pytest.raises(ValueError, match="rise from 9 below n"):
            read_watermark(line, last)
    assert read_watermark(b"6 12 9 11\n", last) == (6, 12, (9, 11))
    with pytest.raises(ValueError, match="t steps by 1 from 0"):
        read_watermark(b"1 0\n")
    for line in (b"6 9", b"6 9\r\n", b"06 9\n", b"6  9\n", b"+6 9\n", b"6 -9\n", b" 6 9\n", b"6 9\n\n", b"\xd9\xa1 9\n",
                 b"6 12 09\n", b"6 12  9\n", b"6 12 9 \n"):
        with pytest.raises(ValueError, match="a watermark line is"):
            read_watermark(line, last)
