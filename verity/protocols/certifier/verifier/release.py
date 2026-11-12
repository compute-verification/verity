"""The release order on the server's side. Each request a certifier releases reaches the server with three headers:
`x-verity-release: N`, N its 0-based index in the certifier run's release order (dense: each released request once, a
request a fault drops before its release unnumbered); `x-verity-release-run: ID`, the run's id in hex; and, from the
network certifier, `x-verity-release-bucket: A`, the bucket of the release, counted from 0 at the run's origin. So
`serve-openai --admit release-order` adds requests to its engine in the order the certifier released them, whatever
order it reads their bodies in, and the advice needn't carry the permutation between the two.

**The watermark** is a stream on which the server learns that a bucket has closed. The network certifier opens one
connection to the server for it and sends one HTTP/1.1 request, `POST /verity/watermark`, whose head (`watermark_head`)
names the run. Its chunked body has one chunk a tick (`watermark_chunk`), the ASCII line `t n`, then any lost releases
` l …`: bucket t has closed, the releases of buckets ≤ t are exactly those numbered N < n, and each l, rising, is a
release of bucket t that its server connection lost, closing, failed or ended before the request was written whole
and its transport's buffer emptied. t steps by 1 from 0, n never decreases, and the run's end sends the terminating
chunk (`WATERMARK_END`). Line t is sent only once each release of bucket t has been written or lost, so a server that
has read line t has been sent every N < n that no line through t names. Those come on their own connections, which it
may read after line t, so a server that admits at bucket t waits for each of them, and skips a lost one should its
bytes still arrive (`read_watermark` gives t, n and the lost). The network certifier never reads the server's side of
that connection and never waits on it.

The headers and the watermark are the certifier's, sent to the server only. The link's frames, the network certifier's
cells and every record hold the client's bytes, so what is certified and tied (cell counts, prompt lengths, the I/O
tie's request frames) is the same with them or without. A lost release is the server connection's state, which the
server sees itself. N, A and n are read back from the signed ingress records: the
network certifier's `END` cells in record order and the bucket of each (`warden.Warden`), and on the active certifier's
link request s in `active.Requests`' cut, which ends before s + 1 starts, so it is release s − 1
(`benchmarks/network_traces/active_replay.ReleaseProxy`, which sends no bucket: it forwards a request's head before its
last frame). A client's own release headers are dropped; bytes that aren't an HTTP/1 request with its whole head pass
unchanged. A live shim that rebuilds the POST copies the headers onto it (`release_of`, `release_head`)."""

from __future__ import annotations

import re
from typing import Sequence

RELEASE_HEADER, RUN_HEADER, BUCKET_HEADER = b"x-verity-release", b"x-verity-release-run", b"x-verity-release-bucket"
WATERMARK_PATH = b"/verity/watermark"
WATERMARK_END = b"0\r\n\r\n"
_REQUEST_LINE = re.compile(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+ [^\s]+ HTTP/1\.[01]")
_LINE = re.compile(rb"(0|[1-9][0-9]*) (0|[1-9][0-9]*)((?: (?:0|[1-9][0-9]*))*)\n")


def _count(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} is a non-negative integer, not {value!r}")
    return value


def release_head(n: int, run: bytes | None = None, bucket: int | None = None) -> bytes:
    """The header lines of release `n`, each ending in CRLF: `x-verity-release: n`, then, with a run id,
    `x-verity-release-run:` its hex, then, with a bucket, `x-verity-release-bucket: bucket`."""
    _count("a release", n)
    if run is not None and not (isinstance(run, bytes) and run):
        raise ValueError("a run id is non-empty bytes")
    head = RELEASE_HEADER + b": %d\r\n" % n
    if run is not None:
        head += RUN_HEADER + b": " + run.hex().encode() + b"\r\n"
    return head if bucket is None else head + BUCKET_HEADER + b": %d\r\n" % _count("a bucket", bucket)


def _split(request: bytes) -> tuple[bytes, list[bytes], bytes] | None:
    head, blank, body = bytes(request).partition(b"\r\n\r\n")
    if not blank:
        return None
    line, *fields = head.split(b"\r\n")
    return (line, fields, body) if _REQUEST_LINE.fullmatch(line) else None


def _name(field: bytes) -> bytes:
    return field.partition(b":")[0].strip().lower()


def with_release(request: bytes, n: int, run: bytes | None = None, bucket: int | None = None) -> bytes:
    """`request` as the server gets it: `release_head(n, run, bucket)` right after its request line, and none of the
    client's own release headers; unchanged if it isn't an HTTP/1 request with its whole head."""
    lines = release_head(n, run, bucket)
    parts = _split(request)
    if parts is None:
        return request
    line, fields, body = parts
    ours = (RELEASE_HEADER, RUN_HEADER, BUCKET_HEADER)
    kept = b"".join(f + b"\r\n" for f in fields if _name(f) not in ours)
    return line + b"\r\n" + lines + kept + b"\r\n" + body


def release_of(head: bytes) -> tuple[int, bytes | None, int | None] | None:
    """(N, run id, bucket) of a request's head (through its blank line), each its first header, or None without a
    well-formed N; a run that isn't hex, or a bucket that isn't a decimal count, is None."""
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
    bucket = values.get(BUCKET_HEADER)
    return int(raw), run, int(bucket) if bucket is not None and bucket.isdigit() else None


def watermark_head(run: bytes) -> bytes:
    """The head of the watermark stream's one request: `POST /verity/watermark` for run `run`, its body chunked."""
    if not (isinstance(run, bytes) and run):
        raise ValueError("a run id is non-empty bytes")
    return (b"POST " + WATERMARK_PATH + b" HTTP/1.1\r\nhost: network-certifier\r\n" + RUN_HEADER + b": " +
            run.hex().encode() + b"\r\ncontent-type: text/plain\r\ntransfer-encoding: chunked\r\n\r\n")


def _rising(lost: Sequence[int], least: int, n: int) -> bool:
    return all(least <= k < n for k in lost) and all(a < b for a, b in zip(lost, lost[1:]))


def watermark_chunk(t: int, n: int, lost: Sequence[int] = ()) -> bytes:
    """Tick t's chunk of the watermark stream: the line `t n`, bucket t closed with n releases through it, then each of
    `lost`, the releases below n that their server connections lost, rising."""
    line = b"%d %d" % (_count("a bucket", t), _count("a release count", n))
    if not _rising([_count("a lost release", k) for k in lost], 0, n):
        raise ValueError(f"the lost releases rise below n = {n}, not {list(lost)!r}")
    line += b"".join(b" %d" % k for k in lost) + b"\n"
    return b"%x\r\n%s\r\n" % (len(line), line)


def read_watermark(line: bytes, last: tuple[int, ...] | None = None) -> tuple[int, int, tuple[int, ...]]:
    """(t, n, lost) of a watermark line, `t n`, any lost releases and its newline, or ValueError unless it follows
    `last`, the line before it (its (t, n, …)): t one past last's (0 first), n at least last's, and the lost rising
    from last's n to below n, so the watermark only advances and names each lost release once."""
    m = _LINE.fullmatch(line)
    if m is None:
        raise ValueError(f"a watermark line is 't n', any lost releases and a newline, not {line[:40]!r}")
    t, n, lost = int(m[1]), int(m[2]), tuple(int(k) for k in m[3].split())
    want, least = (0, 0) if last is None else (last[0] + 1, last[1])
    if t != want or n < least:
        raise ValueError(f"watermark ({t}, {n}) after {last}: t steps by 1 from 0 and n never decreases")
    if not _rising(lost, least, n):
        raise ValueError(f"watermark ({t}, {n}) after {last}: its lost releases {list(lost)} rise from {least} below n")
    return t, n, lost
