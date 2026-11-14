"""A request's head as serve-openai reads it: its first line and its header fields' names, each taken only as the
network certifier takes them (`verity.protocols.accounting.certifier.verifier.release.with_release`). The network certifier passes
on as they are the bytes it doesn't take for a request, without its release, and keeps the fields it doesn't name a
release header. Read more loosely here, either would be served as a request it released, or taken for its watermark.
"""
from __future__ import annotations

from collections.abc import Sequence

from verity.protocols.accounting.certifier.verifier import release as REL

NO_REQUEST_LINE = "the request's first line isn't `METHOD target HTTP/1.0` or `HTTP/1.1`, one space between and nothing after"


def request_line(line: bytes) -> tuple[str, str] | None:
    """(method, target) of a request's first line, or None unless it is one the network certifier takes for a request's
    (`verity.protocols.accounting.certifier.verifier.release.with_release`): a token, a target without whitespace, and ``HTTP/1.0``
    or ``HTTP/1.1``, one space between and nothing after. The network certifier passes any other bytes on as they are:
    without its release, and with whatever release headers the client wrote. Read more loosely here, they would be served
    as a request it released, or taken for its watermark."""
    probe = line + b"\r\n\r\n"
    if REL.with_release(probe, 0) == probe:
        return None
    method, target, _ = line.decode("latin-1").split(" ")
    return method, target


def header_fields(fields: Sequence[bytes]) -> dict[str, str]:
    """A request's header fields by name, the last of a name. A name is the field's bytes before its colon, lower-cased,
    less the ASCII whitespace around it: what the network certifier names a field when it drops a client's own release
    headers (`verity.protocols.accounting.certifier.verifier.release.with_release`). Named more loosely (`str.strip` takes ``\\xa0``
    too, for one), a field the certifier kept would be read here as its release, after its own."""
    return {k.strip().lower().decode("latin-1"): v.strip().decode("latin-1")
            for k, _, v in (f.partition(b":") for f in fields if f)}
