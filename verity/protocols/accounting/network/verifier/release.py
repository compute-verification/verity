"""The network attester's refusals of what it would release to the server: the release order's formats are
`verity.release`'s.

The watermark's path is the network attester's alone: it releases no client's request that opens as one to that path
(`to_watermark`, read on the request's first cell and on the whole request at its `END`), which would reach the server
under the run's id as the stream does.

A connection's bytes are one request, the one the headers go on: the network attester releases none that a server
would read a second request in, or read the first of without the headers (`one_request`, read on the whole request at
its `END`). A second request would reach the server behind the first with no release headers but its sender's own, to
any path, the watermark's included."""

from __future__ import annotations

import re
from urllib.parse import unquote_to_bytes

from verity.release import WATERMARK_PATH, _name, _split

#: a scheme and authority before a request target's path (`http://server`), which a server reads past
_ORIGIN = re.compile(rb"[A-Za-z][A-Za-z0-9+.-]*://[^/?#]*")
#: a header field every server reads alike: its name, the colon right after it, and a value that stays on its line
_FIELD = re.compile(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+:[^\r\n]*")


def to_watermark(opening: bytes) -> bool:
    """Whether a request that opens with `opening` is one to the watermark's path as a server reads it: the second word
    of its first line, past any line breaks before it, is `WATERMARK_PATH` once a scheme and authority before the path
    and a query string or fragment after it are off and its percent-escapes are decoded. A server routes on that word
    whatever the method before it and whatever follows it on the line, so this reads no more of the line, and reads
    one `with_release` doesn't take for a request line. It reads only `opening`: a path that ends past it isn't
    there."""
    words = opening.lstrip(b"\r\n").replace(b"\r", b"\n").split(b"\n", 1)[0].split()
    if len(words) < 2:
        return False
    origin = _ORIGIN.match(words[1])
    path = re.split(rb"[?#]", words[1][origin.end() if origin else 0:], maxsplit=1)[0]
    return unquote_to_bytes(path) == WATERMARK_PATH


def one_request(request: bytes) -> bool:
    """Whether `request`, a connection's whole request, is at most one request to a server, and one `with_release`
    reads. Bytes open a request where their first line, past any line breaks before it, has a second word, as
    `to_watermark` reads it, and a head ends where two line breaks meet; bytes that don't are no request to any
    server. Those that do are one request where their head is the one `with_release` puts the headers in, each
    field a name, the colon right after it and a value on its own line; where it names no transfer coding and at most
    one `content-length`, a decimal count; and where no byte follows the body that length gives it, none without one.
    Any other head is one servers end differently: where one ends the request, another reads a second in the same
    bytes, with no release headers but its sender's own. This reads the head and counts the rest: a line in the body
    that looks like a request line is the body's."""
    lines = request.lstrip(b"\r\n").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if len(lines.split(b"\n", 1)[0].split()) < 2 or b"\n\n" not in lines:
        return True
    parts = _split(request)
    if parts is None or not all(_FIELD.fullmatch(f) for f in parts[1]):
        return False
    names = [_name(f) for f in parts[1]]
    lengths = [f.partition(b":")[2].strip(b" \t") for f in parts[1] if _name(f) == b"content-length"]
    if b"transfer-encoding" in names or len(lengths) > 1 or not all(n.isdigit() for n in lengths):
        return False
    rest, digits = len(parts[2]), lengths[0].lstrip(b"0") if lengths else b""
    return len(digits) > len(b"%d" % rest) or rest <= int(digits or b"0")
