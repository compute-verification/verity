"""The formats of a network attester's release of a request: its headers and its watermark.
It says what a release head and a watermark stream are, and nothing about whether a release is charged, held or
acceptable to the attester, which is a verifier's.

Each request a network attester releases reaches the server with three headers: `x-verity-release: N`, N its 0-based index in the attester
run's release order (dense: each released request once, a request a fault drops before its release unnumbered);
`x-verity-release-run: ID`, the run's id in hex; and, from the network attester, `x-verity-release-bucket: A`, the
bucket of the release, counted from 0 at the run's origin. So a server under the network attester's order
`release-order` adds requests to its engine in the order the attester released them, whatever order it reads their
bodies in, and the advice needn't carry the permutation between the two.

**The watermark** is a stream on which the server learns that a bucket has closed. The network attester opens one
connection to the server for it and sends one HTTP/1.1 request, `POST /verity/watermark`, whose head (`watermark_head`)
names the run. Its chunked body has one chunk a tick (`watermark_chunk`), the ASCII line `t n`, then any lost releases
` l …`: bucket t has closed, the releases of buckets ≤ t are exactly those numbered N < n, and each l, rising, is a
release of bucket t that its server connection lost, closing, failed or ended before the request was written whole
and its transport's buffer emptied. t steps by 1 from 0, n never decreases, and the run's end sends the terminating
chunk (`WATERMARK_END`). Line t is sent only once each release of bucket t has been written or lost, so a server that
has read line t has been sent every N < n that no line through t names. Those come on their own connections, which it
may read after line t, so a server that admits at bucket t waits for each of them, and skips a lost one should its
bytes still arrive (`read_watermark` gives t, n and the lost). The network attester never reads the server's side of
that connection and never waits on it.

The headers and the watermark are the attester's, sent to the server only. The link's frames, the network attester's
cells and every record hold the client's bytes, so what is attested and tied (cell counts, prompt lengths, the I/O
tie's request frames) is the same with them or without. A lost release is the server connection's state, which the
server sees itself. N, A and n are read back from the signed ingress records: the network attester's `END` cells in
record order and the bucket of each (`network_attester.NetworkAttester`). A client's own release headers are dropped;
bytes that aren't an HTTP/1 request with its whole head pass unchanged. A live shim that rebuilds the POST copies the
headers onto it (`release_of`, `release_head`).

**The id variant's bytes**, which the server writes and the network attester's cells carry unchanged: the link frame's
header (`FRAME_HEADER`), the canonical id request (`id_request`, read back by `id_request_fields`), a response's id events (`id_event`) and its last
bytes (`TRAILER`)."""

from __future__ import annotations

import json
import re
import struct
from typing import Any, Mapping, Sequence

RELEASE_HEADER, RUN_HEADER, BUCKET_HEADER = b"x-verity-release", b"x-verity-release-run", b"x-verity-release-bucket"
#: the bytes of a network attester's run id, which it samples at its construction
RUN_ID_BYTES = 32
WATERMARK_PATH = b"/verity/watermark"
WATERMARK_END = b"0\r\n\r\n"
_REQUEST_LINE = re.compile(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+ [^\s]+ HTTP/1\.[01]")
_HEX = re.compile(r"(?:[0-9A-Fa-f]{2})+")
_DECIMAL = re.compile(r"[0-9]+")
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


def release_fields(fields: Mapping[str, str]) -> tuple[int, str | None, int | None] | None:
    """(N, the run id's hex, bucket) of a request's header fields by lower-cased name, each value less the spaces and
    tabs around it, the run and bucket None without theirs; None without `x-verity-release`. `ValueError`, its message
    starting with the header's name, where an N that isn't a decimal count, a run that isn't hex or a bucket that isn't
    a decimal count comes with an `x-verity-release`."""
    raw = fields.get(RELEASE_HEADER.decode())
    if raw is None:
        return None
    values = {h: None if v is None else v.strip(" \t") for h in (RELEASE_HEADER, RUN_HEADER, BUCKET_HEADER)
              for v in [fields.get(h.decode())]}
    n, run, bucket = values[RELEASE_HEADER], values[RUN_HEADER], values[BUCKET_HEADER]
    if not _DECIMAL.fullmatch(n):
        raise ValueError(f"{RELEASE_HEADER.decode()}: a decimal count, not {n[:40]!r}")
    if run is not None and not _HEX.fullmatch(run):
        raise ValueError(f"{RUN_HEADER.decode()}: a run id in hex, not {run[:40]!r}")
    if bucket is not None and not _DECIMAL.fullmatch(bucket):
        raise ValueError(f"{BUCKET_HEADER.decode()}: a decimal count, not {bucket[:40]!r}")
    return int(n), run, None if bucket is None else int(bucket)


def release_of(head: bytes) -> tuple[int, bytes | None, int | None] | None:
    """(N, run id, bucket) of a request's head (through its blank line), each its first header (`release_fields`);
    None for bytes that aren't an HTTP/1 request's whole head."""
    parts = _split(head)
    if parts is None:
        return None
    values: dict[str, str] = {}
    for f in parts[1]:
        values.setdefault(_name(f).decode("latin-1"), f.partition(b":")[2].strip().decode("latin-1"))
    rel = release_fields(values)
    if rel is None:
        return None
    n, run, bucket = rel
    return n, None if run is None else bytes.fromhex(run), bucket


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
    """(t, n, lost) of a watermark line, `t n`, any lost releases and its newline, where it follows `last`, the line
    before it (its (t, n, …)), so the watermark only advances and names each lost release once. `ValueError`, its
    message starting with the rule's name: `watermark-form` for a line of another form, `watermark-tick` unless t is
    one past last's (0 first), `watermark-count` where n is below last's, `watermark-lost` unless the lost rise from
    last's n to below n."""
    m = _LINE.fullmatch(line)
    if m is None:
        raise ValueError(f"watermark-form: a watermark line is 't n', any lost releases and a newline, not {line[:40]!r}")
    t, n, lost = int(m[1]), int(m[2]), tuple(int(k) for k in m[3].split())
    want, least = (0, 0) if last is None else (last[0] + 1, last[1])
    if t != want:
        raise ValueError(f"watermark-tick: watermark ({t}, {n}) after {last}: t steps by 1 from 0")
    if n < least:
        raise ValueError(f"watermark-count: watermark ({t}, {n}) after {last}: n never decreases")
    if not _rising(lost, least, n):
        raise ValueError(f"watermark-lost: watermark ({t}, {n}) after {last}: its lost releases {list(lost)} rise from "
                         f"{least} below n")
    return t, n, lost


#: a link frame's header: its kind, its stream and its payload's length; a frame is B bytes, the header, the payload,
#: then zero fill
FRAME_HEADER = struct.Struct(">BII")
#: a request's length (u32be), which opens it in the link's DATA payloads
REQUEST_LENGTH = struct.Struct(">I")
ID_DIGITS = 10
EVENT_END = b"\n\n"
EVENT_HEAD = (b'data: {"object":"chat.completion.chunk","created":0,"model":"verity","choices":[{"index":0,'
              b'"delta":{"base64":"')
ID_EVENT_HEAD = EVENT_HEAD[:EVENT_HEAD.index(b'"delta"')] + b'"token_id":'
ID_EVENT_TAIL = b',"finish_reason":null}]}' + EVENT_END
ID_EVENT_BYTES = len(ID_EVENT_HEAD) + ID_DIGITS + len(ID_EVENT_TAIL)
#: a response's last bytes, after its id events
TRAILER = b"data: [DONE]" + EVENT_END
#: an id request's HTTP head, up to its length (`id_request`)
ID_REQUEST_HEAD = (b"POST /v1/completions HTTP/1.1\r\nhost: server\r\ncontent-type: application/json\r\n"
                   b"content-length:")
#: the room of an id request's `model` and `user`, each its JSON string padded with spaces
ID_NAME_BYTES = 64


def id_event(token_id: int, size: int = ID_EVENT_BYTES) -> bytes:
    """The canonical event of one token id in `size` bytes: `ID_EVENT_HEAD`, the id right-aligned in 10 decimal
    characters, then `ID_EVENT_TAIL` with spaces (JSON whitespace) ahead of its blank line to fill `size`."""
    if type(token_id) is not int or not 0 <= token_id < 1 << 32:
        raise ValueError(f"a token id is a u32, not {token_id!r}")
    if type(size) is not int or size < ID_EVENT_BYTES:
        raise ValueError(f"an id event is at least {ID_EVENT_BYTES} bytes, not {size!r}")
    tail = ID_EVENT_TAIL[:-len(EVENT_END)] + b" " * (size - ID_EVENT_BYTES) + EVENT_END
    return ID_EVENT_HEAD + b"%10d" % token_id + tail


def _u32_field(x: int, what: str) -> bytes:
    if type(x) is not int or not 0 <= x < 1 << 32:
        raise ValueError(f"{what} is a u32, not {x!r}")
    return b"%*d" % (ID_DIGITS, x)


def _ratio_field(x: float, what: str) -> bytes:
    s = b"%8.6f" % x if isinstance(x, (int, float)) and not isinstance(x, bool) else b""
    if not (s and 0 <= x < 10 and float(s) == x):
        raise ValueError(f"{what} is a number in [0, 10) of at most six decimals, not {x!r}")
    return s


def _name_field(s: str, what: str) -> bytes:
    raw = json.dumps(s).encode() if isinstance(s, str) else b""
    if not raw or len(raw) > ID_NAME_BYTES:
        raise ValueError(f"{what} is a string of at most {ID_NAME_BYTES} bytes as JSON, not {s!r}")
    return raw.ljust(ID_NAME_BYTES)


def id_request(*, model: str, user: str, prompt: Sequence[int], max_tokens: int, temperature: float = 0.0,
               top_p: float = 1.0, seed: int = 0, frame_bytes: int, offset: int = REQUEST_LENGTH.size) -> bytes:
    """A canonical `/v1/completions` request of a token-id prompt, whose every field is at a place fixed by the
    prompt's length and the frame size alone, in `frame_bytes`-byte link frames: `ID_REQUEST_HEAD`, its body's length
    in `ID_DIGITS` characters (HTTP's optional whitespace before it), in the first frame, then the body, streamed,
    with no stop strings: `max_tokens` and `seed` in `ID_DIGITS` characters, `temperature` and `top_p` in eight (six
    decimals), `model` and `user` each in `ID_NAME_BYTES`, and each prompt id after `[` or `,`. No field straddles a
    frame: spaces (JSON's whitespace) before one that would move it, with its key, to the next. `offset` is the
    request's first byte's place in its first frame's payload: `REQUEST_LENGTH`'s 4 bytes on the active network
    attester's link, 0 in the network attester's cells, which carry the bytes alone."""
    if offset not in (0, REQUEST_LENGTH.size):
        raise ValueError(f"an id request starts its first frame's payload at 0 or {REQUEST_LENGTH.size}, not {offset!r}")
    if not prompt or isinstance(prompt, (str, bytes)):
        raise ValueError("an id request's prompt is at least one token id")
    pieces = [b'{"max_tokens":' + _u32_field(max_tokens, "max_tokens"),
              b',"temperature":' + _ratio_field(temperature, "temperature"),
              b',"top_p":' + _ratio_field(top_p, "top_p"), b',"seed":' + _u32_field(seed, "a seed"),
              b',"stop":[],"stream":true', b',"model":' + _name_field(model, "a model"),
              b',"user":' + _name_field(user, "a user"), b',"prompt":',
              *((b"," if k else b"[") + _u32_field(t, "a prompt id") for k, t in enumerate(prompt)), b"]}"]
    room = frame_bytes - FRAME_HEADER.size
    at = offset + len(ID_REQUEST_HEAD) + ID_DIGITS + 4
    if at > room or max(map(len, pieces)) > room:
        raise ValueError(f"a {frame_bytes}-byte frame has no room for an id request's head or a field")
    parts = []
    for p in pieces:
        gap = -at % room if at // room != (at + len(p) - 1) // room else 0
        parts.append(b" " * gap + p)
        at += gap + len(p)
    raw = b"".join(parts)
    return ID_REQUEST_HEAD + b"%*d" % (ID_DIGITS, len(raw)) + b"\r\n\r\n" + raw


#: the scan's cell payload, P = B − 9 at its B = 512-byte cells
ID_PAYLOAD = 503


def id_request_fields(request: bytes, payload: int) -> dict[str, Any] | str:
    """The fields (`id_request`'s keywords) of `request`, its head and body as the server read them, where it
    is an id request as the network attester forwards it: without the release headers it puts on the
    server's copy (`with_release`), exactly `id_request` of its own fields in the scan's B = P + 9-byte
    cells (P = ``payload``). For anything else, why it is not one."""
    head, _, body = request.partition(b"\r\n\r\n")
    line, *fields = head.split(b"\r\n")
    kept = b"".join(f + b"\r\n" for f in fields if _name(f) not in (RELEASE_HEADER, RUN_HEADER, BUCKET_HEADER))
    try:
        b = json.loads(body)
        got = {k: b[k] for k in ("model", "user", "prompt", "max_tokens", "temperature", "top_p", "seed")}
        canonical = id_request(**got, frame_bytes=payload + FRAME_HEADER.size, offset=0)
    except (ValueError, KeyError, TypeError) as e:
        return f"no id request's fields: {e!r}"
    return got if canonical == line + b"\r\n" + kept + b"\r\n" + body else "not the id request of its fields"
