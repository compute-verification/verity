"""The release order on the server's side: each request a certifier releases reaches the server with
`x-verity-release: N`, N its 0-based index in the certifier run's release order (dense: each released request once,
a request a fault drops before its release unnumbered), and `x-verity-release-run: ID`, the run's id in hex, so that
`serve-openai --admit release-order` adds requests to its engine in the order the certifier released them, whatever
order it reads their bodies in, and the advice needn't carry the permutation between the two.

The headers are the certifier's, put on the server's copy of a request only. The link's frames, the warden's cells
and every record hold the client's bytes, so what is certified and tied (cell counts, prompt lengths, the I/O tie's
request frames) is the same with them or without, and N is read back from the signed ingress records: the warden's
`END` cells in record order (`warden.Warden`), and on the active certifier's link request s in `active.Requests`'
cut, which ends before s + 1 starts, so it is release s − 1 (`benchmarks/network_traces/active_replay.ReleaseProxy`).
A client's own release headers are dropped; bytes that aren't an HTTP/1 request with its whole head pass unchanged.
A live shim that rebuilds the POST copies them onto it (`release_of`, `release_head`)."""

from __future__ import annotations

import re

RELEASE_HEADER, RUN_HEADER = b"x-verity-release", b"x-verity-release-run"
_REQUEST_LINE = re.compile(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+ [^\s]+ HTTP/1\.[01]")


def release_head(n: int, run: bytes | None = None) -> bytes:
    """The header lines of release `n`, each ending in CRLF: `x-verity-release: n`, then, with a run id,
    `x-verity-release-run:` its hex."""
    if type(n) is not int or n < 0:
        raise ValueError(f"a release is a non-negative integer, not {n!r}")
    if run is not None and not (isinstance(run, bytes) and run):
        raise ValueError("a run id is non-empty bytes")
    head = RELEASE_HEADER + b": %d\r\n" % n
    return head if run is None else head + RUN_HEADER + b": " + run.hex().encode() + b"\r\n"


def _split(request: bytes) -> tuple[bytes, list[bytes], bytes] | None:
    head, blank, body = bytes(request).partition(b"\r\n\r\n")
    if not blank:
        return None
    line, *fields = head.split(b"\r\n")
    return (line, fields, body) if _REQUEST_LINE.fullmatch(line) else None


def _name(field: bytes) -> bytes:
    return field.partition(b":")[0].strip().lower()


def with_release(request: bytes, n: int, run: bytes | None = None) -> bytes:
    """`request` as the server gets it: `release_head(n, run)` right after its request line, and none of the client's
    own release headers; unchanged if it isn't an HTTP/1 request with its whole head."""
    lines = release_head(n, run)
    parts = _split(request)
    if parts is None:
        return request
    line, fields, body = parts
    kept = b"".join(f + b"\r\n" for f in fields if _name(f) not in (RELEASE_HEADER, RUN_HEADER))
    return line + b"\r\n" + lines + kept + b"\r\n" + body


def release_of(head: bytes) -> tuple[int, bytes | None] | None:
    """(N, run id) of a request's head (through its blank line), each its first header, or None without a well-formed
    N; a run that isn't hex is None."""
    parts = _split(head)
    if parts is None:
        return None
    values: dict[bytes, bytes] = {}
    for f in parts[1]:
        values.setdefault(_name(f), f.partition(b":")[2].strip())
    raw = values.get(RELEASE_HEADER)
    if raw is None or not raw.isdigit():
        return None
    run = None
    if RUN_HEADER in values:
        try:
            run = bytes.fromhex(values[RUN_HEADER].decode("ascii")) or None
        except ValueError:
            pass
    return int(raw), run
