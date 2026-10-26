"""`S3Remote`: an S3-compatible bucket (Cloudflare R2, AWS) over stdlib `http.client` with hand-rolled SigV4 (README §7).

Path-style addressing (`<endpoint>/<bucket>/<key>`), signed payloads (`x-amz-content-sha256` is the real sha256 of the body,
never UNSIGNED-PAYLOAD), one PUT for objects <= `multipart_min` (64 MiB) and a MULTIPART upload above that (`put_multipart`):
`POST ?uploads`, one `PUT ?partNumber=N&uploadId=` per `part_size` bytes (`choose_part_size_for`: ~16 parts of 16..256 MiB for a
mid-size object, 256 MiB above 4 GiB, grown so that there are never more than `MAX_PARTS` parts; every part but the last has
the same size, as R2 requires), the parts uploaded `part_jobs` at a time (bounded by `PART_MEM_BUDGET` of in-flight bytes),
`POST ?uploadId=` to complete, `DELETE ?uploadId=` to abort on any failure.  A multipart ETag is not an MD5 of the object but it
IS a function of the content: `md5(md5(part_1) .. md5(part_n))-n` (`multipart_etag`), so the client computes it from the local
file and the service's answer must match, and a later HEAD of the key is the same check (`etag-multipart`).  A sidecar
`<key>.parts.json` (`parts_sidecar`) records the part size and the per-part sha256/md5 so the parts can be re-checked without
the local file.  Objects above `SINGLE_PUT_MAX` (5 GiB, the service's PutObject limit) can only go multipart.

`put(key, path)` is a conditional create for an immutable key.  Small objects (<= `put_first_max`, 64 KiB) go straight up as
`PUT If-None-Match: *` -- one round trip when the key is new, which is the common case on a first push; a 412 means the key
exists, and a HEAD then decides `skipped` (same size and, for a single-part ETag, the same MD5) or RemoteConflict.  Whether a
small object goes PUT-first or HEAD-first is ADAPTIVE: the last `ADAPT_WINDOW` small objects' outcomes are kept, and once more
than half of them turned out to exist already (a re-push of a preserved tree) the client HEADs first (one round trip for a present
key) until the window says otherwise; a first push settles on PUT-first, a re-push on HEAD-first, each after a few objects.  Larger
objects HEAD first (present -> `skipped` / RemoteConflict without moving the body), else conditional PUT; 412 there means
another writer got in between (`exists`).  400/501 for `If-None-Match` means the service does not implement the header, so
conditional PUT is switched off for the rest of this session (`conditional_put`, the reason in `conditional_put_error`) and the
object goes up with a plain PUT after a HEAD.  A service that silently IGNORES the header would let PUT-first overwrite (and hide a
RemoteConflict), so the session's first conditional create is probed once: the same bytes are PUT again with `If-None-Match: *`,
and anything but a 412 switches conditional PUT off the same way (`conditional_put_probed`).  The result dict says what happened
(`action`, the local `md5`/`sha256`, the ETag seen, `fallback`, and for a multipart upload `multipart` = {part_bytes, parts, etag}).

CONNECTIONS.  Every request runs on a pooled keep-alive TLS connection (`_Pool`: idle connections are kept per instance, at most
`max_pool_connections` = 64 of them; a request takes one, uses it exclusively, and gives it back when the response was consumed
whole; a connection the service closed, or that failed mid-request, is dropped and the request retried on a fresh one).  Before
this (r20) each request opened its own TLS connection: 259,030 blobs / 46 GB took 1.9 h at ~27 blobs/s.  `jobs` (default 32) is
the bounded thread pool `put_many` runs on; `LocalStore.push` brings its own pool, of
the same size, over `put`.

MEASURED (2026-09-23, runpod pod vyv-v2cpu3 -> R2, 10k x 4 KiB blobs pushed through `LocalStore.push`, then re-pushed, then fetched,
plus one 512 MiB blob; `tools/research/bench` figures in the r20 report):

    client              first push        re-push (all skipped)   get           512 MiB put / get
    old (TLS/request)   21.1 blobs/s       85 blobs/s              55 blobs/s     25 MB/s / 43 MB/s
    pooled, jobs=32     47.6 blobs/s      145 blobs/s              72 blobs/s     47 MB/s / 22 MB/s  (multipart 16 x 32 MiB)
    pooled, jobs=128   151.3 blobs/s      568 blobs/s             292 blobs/s     51 MB/s / 42 MB/s
    pooled, jobs=256   320.6 blobs/s      292 blobs/s             502 blobs/s     45 MB/s / 39 MB/s

The first-push rate scales almost linearly with `jobs` because the cost is per-request latency to R2 from that pod (~0.8 s per
4 KiB PUT, 1.01 requests per blob with conditional PUT + the ETag from the PUT response), not connection setup any more; the
pool opened 32 / 155 / 398 connections for the three runs.  The pooled client is 2.3x (32) .. 7x (128) the old one.  Re-push at
256 came in below 128 because the adaptive PUT-first window flips to HEAD-first once more than half of the last `ADAPT_WINDOW`
small objects existed already, and HEADs are no cheaper than PUTs at this latency (the run-to-run spread on the pod is also large).

Re-run on the same pod 18:17-18:42Z (load 130-170 on 32 vCPUs; `put` driven directly by a 32-thread pool, no store in the loop,
two interleaved rounds): old 25.1 / 22.2 blobs/s, pooled 45.9 / 51.0 blobs/s (32 connections for 10,033 requests); 512 MiB old
single PUT 24.2 MB/s, pooled multipart (16 x 32 MiB, 8 in flight) 8.5 / 9.0 MB/s -- slower than one PUT in that window, against
47 MB/s in the table above; not diagnosed (the pod was terminated mid-run).  R2 answered 429 to 1 (old) and 4 (pooled) of 10k
PUTs at jobs=32; a 429 is retried like a 5xx since (`retry_delay`: Retry-After, capped).

PACK FORMAT.  The r20 plan's threshold (`PACK_THRESHOLD_BLOBS_PER_S` = 300) is not met at the default `jobs` = 32 (46-51 blobs/s
in both runs); it is reached only at `jobs` = 256 on a first push (320.6; re-push 292), the concurrency at which R2 throttles.  By
that plan a pack format for many-small-file trees is warranted; it is not implemented, and one object per blob stays the storage
unit until it is.  Raising `store.toml [remote] jobs` (with `max_pool_connections` >= jobs) is the interim lever.

The signing functions are pure and tested against the AWS documented vectors (`sign_headers`, `presign`, `canonical_request`,
`string_to_sign`, `signature`).  One instance may be used from several threads (`LocalStore.push` does): the pool is locked, a
connection is never shared by two requests at once, and the only other shared mutable is the conditional-PUT flag.
"""
from __future__ import annotations

import hashlib
import hmac
import http.client
import io
import os
import socket
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .remote import RemoteConflict

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SINGLE_PUT_MAX = 5 << 30        # PutObject carries at most 5 GiB (S3 and R2 alike); above that only multipart works
MULTIPART_MIN = 64 << 20        # objects above this go multipart (parallel parts) even though one PUT would carry them
PUT_FIRST_MAX = 64 << 10        # objects up to this size skip the HEAD and are created with one conditional PUT ...
ADAPT_WINDOW = 16               # ... unless more than half of the last this many small objects existed already (then HEAD first)
PART_SIZE = 256 << 20           # one UploadPart per 256 MiB (>= the 5 MiB minimum; 21 parts for a 5 GiB object)
PART_SIZE_MIN = 16 << 20        # the smallest part `choose_part_size_for` picks for a mid-size object (R2's floor is 5 MiB)
TARGET_PARTS = 16               # a mid-size object is cut into about this many parts so they can go up in parallel
MAX_PARTS = 10_000              # the service's limit on parts per upload: `choose_part_size` grows the part above this
PART_JOBS = 8                   # parts in flight at once (each is `part_size` bytes in memory) ...
PART_MEM_BUDGET = 1 << 30       # ... but never more than this many bytes of parts in flight
PARTS_SIDECAR_SCHEMA = "research/parts/v1"
REQUEST_DEADLINE_S = 180.0       # a request's hard wall-clock deadline to its headers; a GET body up to GET_STALL_MIN_BYTES then gets
                                 # max(this, bytes / MIN_BODY_RATE), a larger one a STALL_WINDOW_S progress window
MIN_BODY_RATE = 64 * 1024        # bytes/s a GET body up to GET_STALL_MIN_BYTES may not fall below on average before its deadline cuts it off
MIN_UPLOAD_RATE = 8 * 1024       # the same for a request body (a PUT from the laptop's 25-220 KB/s uplink must still finish)
STALL_WINDOW_S = 120.0           # a part PUT is cut off when a window this long passes with under STALL_MIN_BYTES of its body sent,
STALL_MIN_BYTES = 1 << 20        # or (body sent) with no response: its whole-body deadline above is hours for a 256 MiB part
GET_STALL_MIN_BYTES = 32 << 20   # a GET body is cut off (and the GET retried) when a STALL_WINDOW_S window brings under this much (~280
                                 # KB/s): a stalled stream that still trickles kept a 16 GiB part going at 13-170 KB/s for an hour while a
                                 # fresh connection on the same node got 45-70 MB/s (INF-214); its whole-body deadline was 73 h
HEAD_DEADLINE_S = 30.0           # a HEAD carries no body: a much shorter deadline, and HEAD_RETRIES tries after the first
HEAD_RETRIES = 1
RETRIES = 3                     # transient 5xx / 429 / connection errors: 4 tries, 1 s, 2 s, 4 s backoff (or the service's Retry-After)
RETRY_BACKOFF_S = 1.0
RETRY_AFTER_MAX_S = 30.0        # a Retry-After above this is capped (the try budget stays bounded)
JOBS = 32                       # the bounded thread pool behind put_many
MAX_POOL_CONNECTIONS = 64       # idle keep-alive connections kept per instance
PACK_THRESHOLD_BLOBS_PER_S = 300   # below this, the r20 plan called for a pack format for many-small-file trees (not met at jobs=32: docstring)
ALGORITHM = "AWS4-HMAC-SHA256"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


# ---- SigV4 (pure) -----------------------------------------------------------------------------------------------------------------------

def uri_encode(s: str, *, keep_slash: bool = False) -> str:
    """RFC 3986 encoding as S3 wants it: unreserved `A-Za-z0-9-_.~` untouched, space -> %20, '/' kept only in the path."""
    return urllib.parse.quote(s, safe="-_.~/" if keep_slash else "-_.~")


def canonical_query(query: dict[str, str] | list[tuple[str, str]]) -> str:
    """Names and values URI-encoded individually, then sorted by (encoded) name."""
    items = list(query.items()) if isinstance(query, dict) else list(query)
    return "&".join(f"{k}={v}" for k, v in sorted((uri_encode(k), uri_encode(v)) for k, v in items))


def canonical_request(method: str, path: str, query: dict[str, str] | list[tuple[str, str]], headers: dict[str, str], payload_hash: str) -> str:
    """`headers` are the ones being signed (name -> value); names are lower-cased and sorted, values whitespace-trimmed."""
    canon_headers = sorted((k.lower().strip(), " ".join(v.strip().split())) for k, v in headers.items())
    signed = ";".join(k for k, _ in canon_headers)
    return "\n".join([method.upper(), uri_encode(path, keep_slash=True) or "/", canonical_query(query),
                      "".join(f"{k}:{v}\n" for k, v in canon_headers), signed, payload_hash])


def string_to_sign(amz_date: str, scope: str, creq: str) -> str:
    return "\n".join([ALGORITHM, amz_date, scope, hashlib.sha256(creq.encode("utf-8")).hexdigest()])


def signing_key(secret: str, date: str, region: str, service: str = "s3") -> bytes:
    k = ("AWS4" + secret).encode("utf-8")
    for part in (date, region, service, "aws4_request"):
        k = hmac.new(k, part.encode("utf-8"), hashlib.sha256).digest()
    return k


def signature(secret: str, date: str, region: str, sts: str, service: str = "s3") -> str:
    return hmac.new(signing_key(secret, date, region, service), sts.encode("utf-8"), hashlib.sha256).hexdigest()


def sign_headers(method: str, host: str, path: str, query: dict[str, str] | list[tuple[str, str]], headers: dict[str, str], payload_hash: str,
                 access_key: str, secret_key: str, region: str, amz_date: str, service: str = "s3") -> dict[str, str]:
    """Return `headers` + host, x-amz-date, x-amz-content-sha256, Authorization.  `amz_date` is `YYYYMMDDTHHMMSSZ`."""
    h = {**headers, "host": host, "x-amz-date": amz_date, "x-amz-content-sha256": payload_hash}
    creq = canonical_request(method, path, query, h, payload_hash)
    scope = f"{amz_date[:8]}/{region}/{service}/aws4_request"
    sig = signature(secret_key, amz_date[:8], region, string_to_sign(amz_date, scope, creq), service)
    signed = ";".join(sorted(k.lower() for k in h))
    h["Authorization"] = f"{ALGORITHM} Credential={access_key}/{scope},SignedHeaders={signed},Signature={sig}"
    return h


def presign(method: str, scheme: str, host: str, path: str, access_key: str, secret_key: str, region: str, amz_date: str,
            expires_s: int, session_token: str | None = None, service: str = "s3") -> str:
    """A query-signed URL (SigV4 `X-Amz-*` parameters, only `host` signed, `UNSIGNED-PAYLOAD`): whoever holds it may send this one
    request for `expires_s` seconds (at most 7 days) from `amz_date`, and nothing else."""
    scope = f"{amz_date[:8]}/{region}/{service}/aws4_request"
    query = {"X-Amz-Algorithm": ALGORITHM, "X-Amz-Credential": f"{access_key}/{scope}", "X-Amz-Date": amz_date,
             "X-Amz-Expires": str(int(expires_s)), "X-Amz-SignedHeaders": "host"}
    if session_token:
        query["X-Amz-Security-Token"] = session_token
    creq = canonical_request(method, path, query, {"host": host}, "UNSIGNED-PAYLOAD")
    sig = signature(secret_key, amz_date[:8], region, string_to_sign(amz_date, scope, creq), service)
    return f"{scheme}://{host}{uri_encode(path, keep_slash=True)}?{canonical_query(query)}&X-Amz-Signature={sig}"


def sha256_file(path: Path) -> tuple[str, int]:
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def hash_file(path: Path) -> tuple[str, str, int]:
    """(sha256 hex, md5 hex, size) in one pass: the sha256 signs the payload, the md5 is what a single-part ETag must equal."""
    s, m, n = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            s.update(chunk)
            m.update(chunk)
            n += len(chunk)
    return s.hexdigest(), m.hexdigest(), n


def etag_value(raw: str | None) -> str | None:
    """`"abc"` -> `abc`, `W/"abc"` -> `abc`, None stays None.  A `-` inside marks a multipart ETag (which is not an MD5)."""
    if raw is None:
        return None
    v = raw.strip()
    if v.startswith("W/"):
        v = v[2:]
    return v.strip('"').lower() or None


def is_md5_etag(etag: str | None) -> bool:
    return bool(etag) and "-" not in etag and len(etag) == 32


def error_code(e: urllib.error.HTTPError) -> str:
    """`<Code>` of an S3 error body when there is one, else the HTTP status."""
    try:
        root = ET.fromstring(e.read())
        return root.findtext("Code") or root.findtext(f"{S3_NS}Code") or f"HTTP {e.code}"
    except Exception:  # noqa: BLE001  (no body, or not XML: the status line is all there is)
        return f"HTTP {e.code}"


# ---- multipart (pure) -------------------------------------------------------------------------------------------------------------------

def retry_delay(retry_after: str | None, attempt: int) -> float:
    """Seconds before try `attempt + 1`: the service's Retry-After (delta-seconds, capped at RETRY_AFTER_MAX_S), else 1 s, 2 s, 4 s."""
    try:
        if retry_after is not None:
            return max(0.0, min(float(retry_after), RETRY_AFTER_MAX_S))
    except ValueError:                      # an HTTP-date Retry-After: fall back to the exponential backoff
        pass
    return RETRY_BACKOFF_S * (2**attempt)


def choose_part_size(size: int, part_size: int = PART_SIZE, max_parts: int = MAX_PARTS) -> int:
    """`part_size`, grown (in whole MiB) until `size` fits in `max_parts` parts."""
    need = -(-size // max_parts)                                        # ceil
    if need <= part_size:
        return part_size
    return -(-need // (1 << 20)) * (1 << 20)


def choose_part_size_for(size: int, part_size: int = PART_SIZE, target_parts: int = TARGET_PARTS, part_min: int = PART_SIZE_MIN) -> int:
    """The part size for an object of `size` bytes going multipart: about `target_parts` equal parts, in whole MiB, never below
    `part_min` and never above `part_size` (which is what an object of >= `part_size * target_parts` = 4 GiB gets), then
    `choose_part_size` so that `MAX_PARTS` is respected.  A 2 GiB object -> 128 MiB x 16; 100 MiB -> 16 MiB x 7; 5 GiB -> 256 MiB x 20."""
    want = -(-size // max(1, target_parts))
    want = -(-want // (1 << 20)) * (1 << 20)
    return choose_part_size(size, min(part_size, max(part_min, want)))


def plan_parts(size: int, part_size: int) -> list[tuple[int, int, int]]:
    """[(part_number, offset, length)]: every part `part_size` bytes but the last, numbered from 1.  An empty file is one empty part."""
    if size <= 0:
        return [(1, 0, 0)]
    return [(i + 1, off, min(part_size, size - off)) for i, off in enumerate(range(0, size, part_size))]


def multipart_etag(part_md5s: list[str]) -> str:
    """What S3 / R2 report for a completed multipart upload: md5 of the concatenated raw part digests, then `-<parts>`."""
    return hashlib.md5(b"".join(bytes.fromhex(m) for m in part_md5s), usedforsecurity=False).hexdigest() + f"-{len(part_md5s)}"


def parts_of(etag: str | None) -> int | None:
    """`abc-21` -> 21; a single-part ETag (or none) -> None."""
    if etag and "-" in etag:
        n = etag.rsplit("-", 1)[1]
        return int(n) if n.isdigit() else None
    return None


def hash_parts(path: Path, part_size: int) -> list[dict]:
    """[{"n", "offset", "bytes", "sha256", "md5"}] for `plan_parts(size, part_size)`, one pass over the file."""
    out = []
    with open(path, "rb") as f:
        for n, off, length in plan_parts(path.stat().st_size, part_size):
            f.seek(off)
            s, m, left = hashlib.sha256(), hashlib.md5(usedforsecurity=False), length
            while left:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    raise OSError(f"{path}: short read at {off + length - left}")
                s.update(chunk)
                m.update(chunk)
                left -= len(chunk)
            out.append({"n": n, "offset": off, "bytes": length, "sha256": s.hexdigest(), "md5": m.hexdigest()})
    return out


def parts_sidecar(key: str, size: int, sha256: str, part_size: int, parts: list[dict]) -> dict:
    return {"schema": PARTS_SIDECAR_SCHEMA, "key": key, "bytes": size, "sha256": sha256, "part_bytes": part_size,
            "etag": multipart_etag([p["md5"] for p in parts]), "parts": [{k: p[k] for k in ("n", "bytes", "sha256", "md5")} for p in parts]}


def sidecar_key(key: str) -> str:
    return f"{key}.parts.json"


class MultipartError(Exception):
    """A multipart upload that did not complete as the local bytes say it should (aborted on the service when it could be)."""


# ---- connections ------------------------------------------------------------------------------------------------------------------------

_RETRIABLE = (http.client.HTTPException, ConnectionError, TimeoutError, socket.timeout, ssl.SSLError, OSError)


class _Pool:
    """Idle keep-alive connections to one host, taken and returned under a lock.  `acquire` hands out an idle one or opens a new
    one; `release(conn)` keeps it for the next request (up to `limit` idle), `discard(conn)` closes it.  `https_proxy` /
    `HTTPS_PROXY` in the environment (the only proxy source the old opener consulted) is honoured as a CONNECT tunnel."""

    def __init__(self, scheme: str, host: str, timeout: float, limit: int = MAX_POOL_CONNECTIONS):
        self.scheme, self.timeout, self.limit = scheme, timeout, limit
        self.host, _, port = host.partition(":")
        self.port = int(port) if port else (443 if scheme == "https" else 80)
        self._idle: list[http.client.HTTPConnection] = []
        self._lock = threading.Lock()
        self._ctx = ssl.create_default_context() if scheme == "https" else None
        proxies = urllib.request.getproxies_environment()
        self._proxy = urllib.parse.urlsplit(proxies[scheme]) if proxies.get(scheme) and not urllib.request.proxy_bypass_environment(self.host, proxies) else None
        self.opened = 0                  # connections opened over the pool's lifetime (a bench / test statistic)

    def _open(self) -> http.client.HTTPConnection:
        if self._proxy is not None:
            phost, pport = self._proxy.hostname, self._proxy.port or (443 if self._proxy.scheme == "https" else 80)
            conn = (http.client.HTTPSConnection(phost, pport, timeout=self.timeout, context=self._ctx) if self._proxy.scheme == "https"
                    else http.client.HTTPConnection(phost, pport, timeout=self.timeout))
            conn.set_tunnel(self.host, self.port)
        elif self.scheme == "https":
            conn = http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout, context=self._ctx)
        else:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        with self._lock:
            self.opened += 1
        return conn

    def acquire(self, fresh: bool = False) -> http.client.HTTPConnection:
        """An idle connection, else a new one; `fresh` always opens a new one (the retry after a connection error: when R2 has closed
        idle keep-alives, every connection in the pool is as dead as the one that just failed)."""
        if not fresh:
            with self._lock:
                if self._idle:
                    return self._idle.pop()
        return self._open()

    def release(self, conn: http.client.HTTPConnection) -> None:
        with self._lock:
            if len(self._idle) < self.limit:
                self._idle.append(conn)
                return
        conn.close()

    @staticmethod
    def discard(conn: http.client.HTTPConnection) -> None:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        with self._lock:
            idle, self._idle = self._idle, []
        for c in idle:
            self.discard(c)

    @property
    def idle(self) -> int:
        with self._lock:
            return len(self._idle)


class _Watchdog:
    """A hard deadline on one request: when it passes, the connection's socket is shut down, so a read blocked in the kernel (a
    stalled TLS stream that trickles, a peer that never answers) returns at once and the request fails as retriable.  The socket
    timeout alone bounds each recv, not the request: a laptop parity run sat 65 min in such reads."""

    timer = threading.Timer          # arms the deadline; a test replaces it to decide when a deadline passes

    def __init__(self, conn: http.client.HTTPConnection, seconds: float, progress: Callable[[], int] | None = None, min_bytes: int = 0):
        """With `progress` (bytes of the body sent so far) the deadline is a window: each one that saw `min_bytes` more re-arms it."""
        self.progress, self.min_bytes, self.seconds, self._seen = progress, min_bytes, seconds, 0
        self.conn, self.fired, self.sock = conn, False, None     # `sock` is taken once the request is sent: http.client hands a
                                                                  # closing connection's socket to the response and clears conn.sock
        self._t: threading.Timer | None = None
        self.extend(seconds)

    def _fire(self) -> None:
        if self.progress is not None:
            now = self.progress()
            if now - self._seen >= self.min_bytes and now > self._seen:
                self._seen = now
                self.extend(self.seconds)
                return
        self.fired = True
        sock = self.sock or getattr(self.conn, "sock", None)
        try:
            if sock is not None:
                sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def window(self, seconds: float, progress: Callable[[], int], min_bytes: int) -> None:
        """From now on, a window of `seconds` that sees under `min_bytes` more of `progress` fires the deadline."""
        self.progress, self.min_bytes, self.seconds, self._seen = progress, min_bytes, seconds, progress()
        self.extend(seconds)

    def extend(self, seconds: float) -> None:
        self.cancel()
        self._t = self.timer(seconds, self._fire)
        self._t.daemon = True
        self._t.start()

    def cancel(self) -> None:
        if self._t is not None:
            self._t.cancel()


class _Counted:
    """A bytes body http.client reads in blocks, counting what it has taken to send (a :class:`_Watchdog`'s progress)."""

    def __init__(self, data: bytes):
        self._f, self.n = io.BytesIO(data), 0

    def read(self, size: int = -1) -> bytes:
        b = self._f.read(size)
        self.n += len(b)
        return b


class _Response:
    """What `_request` returns: the `http.client.HTTPResponse` (`.status`, `.headers`, `.read()`) bound to its pooled connection.
    Leaving the `with` block gives the connection back to the pool when the body was consumed whole (or is drained here when it
    is small), else closes it -- a half-read body would poison the next request on that connection."""

    DRAIN_MAX = 1 << 20

    def __init__(self, pool: _Pool, conn: http.client.HTTPConnection, resp: http.client.HTTPResponse, watchdog: _Watchdog | None = None):
        self._pool, self._conn, self.resp, self.watchdog = pool, conn, resp, watchdog
        self.status, self.headers = resp.status, resp.headers

    def read(self, n: int = -1) -> bytes:
        return self.resp.read() if n is None or n < 0 else self.resp.read(n)

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is None:
            return
        if self.watchdog is not None:
            self.watchdog.cancel()
            if self.watchdog.fired:
                self._pool.discard(conn)
                return
        try:
            if not self.resp.isclosed():
                left = self.resp.length
                if left is not None and left <= self.DRAIN_MAX:
                    self.resp.read()
                else:
                    self.resp.close()
                    self._pool.discard(conn)
                    return
            if self.resp.will_close:
                self._pool.discard(conn)
            else:
                self._pool.release(conn)
        except Exception:  # noqa: BLE001
            self._pool.discard(conn)

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False


# ---- client -----------------------------------------------------------------------------------------------------------------------------

class S3Remote:
    """`session_token` (a temporary credential's third part, `AWS_SESSION_TOKEN`) travels as `x-amz-security-token`, inside the
    signed headers, on every request.  `jobs` bounds the thread pool of the `*_many` helpers; `max_pool_connections` the idle
    keep-alive connections kept (module docstring)."""

    single_put_max = SINGLE_PUT_MAX
    multipart_min = MULTIPART_MIN
    put_first_max = PUT_FIRST_MAX
    part_size = PART_SIZE
    part_jobs = PART_JOBS
    part_mem_budget = PART_MEM_BUDGET

    def __init__(self, endpoint: str, bucket: str, access_key: str, secret_key: str, region: str = "auto", timeout: float = 120.0,
                 session_token: str | None = None, jobs: int = JOBS, max_pool_connections: int = MAX_POOL_CONNECTIONS):
        u = urllib.parse.urlsplit(endpoint)
        if u.scheme not in ("http", "https") or not u.netloc:
            raise ValueError(f"endpoint must be an http(s) URL, got {endpoint!r}")
        self.scheme, self.host, self.base_path = u.scheme, u.netloc, u.path.rstrip("/")
        self.bucket, self.access_key, self.secret_key, self.region, self.timeout = bucket, access_key, secret_key, region, timeout
        self.session_token = session_token or None
        self.jobs = max(1, int(jobs))
        self.conditional_put = True                 # `If-None-Match: *` on PUT until the service says it does not implement it
        self.conditional_put_probed = False         # the session's first conditional create re-PUTs once to see a 412 (module docstring)
        self.conditional_put_error: str | None = None
        self._probe_lock = threading.Lock()
        self._existed: deque[bool] = deque(maxlen=ADAPT_WINDOW)   # per small object: did its key exist already? (module docstring, ADAPTIVE)
        self.pool = _Pool(self.scheme, self.host, timeout, limit=max(1, int(max_pool_connections)))
        self.requests = 0                           # requests sent (a bench / test statistic; the pool counts connections opened)
        self.deadline_hits: list[dict] = []         # {method, key, try, of, deadline_s} per request cut off by its deadline (logged too)
        self.exhausted: list[dict] = []             # {method, key, error} per request that failed on every try
        self._stats_lock = threading.Lock()

    def describe(self) -> str:
        return f"s3://{self.bucket} @ {self.scheme}://{self.host}"

    def close(self) -> None:
        """Drop the idle connections (an instance is usable again after this: the pool reopens on demand)."""
        self.pool.close()

    def stats(self) -> dict:
        return {"requests": self.requests, "connections_opened": self.pool.opened, "idle": self.pool.idle, "jobs": self.jobs,
                "conditional_put": self.conditional_put, "conditional_put_probed": self.conditional_put_probed, "conditional_put_error": self.conditional_put_error,
                "put_first": self._prefer_put_first()}

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _path(self, key: str) -> str:
        return f"{self.base_path}/{self.bucket}/{key}" if key else f"{self.base_path}/{self.bucket}"

    def _request(self, method: str, key: str, query: dict[str, str] | None = None, body: bytes | Path | None = None, headers: dict[str, str] | None = None,
                 payload_hash: str | None = None, stall_window: float | None = None) -> _Response:
        """One signed request on a pooled connection, with retries on 5xx / 429 / connection errors (each try re-signed with a fresh
        timestamp, a file body re-read from the start, a failed connection discarded).  `payload_hash` (sha256 hex of a Path body)
        skips re-hashing when the caller already has it.  `stall_window` (a bytes body) replaces the whole-request deadline by
        :class:`_Watchdog`'s progress window.  A status >= 400 raises `urllib.error.HTTPError` (the body read, so
        `error_code` can parse it); the returned `_Response` is a context manager that recycles the connection."""
        query = query or {}
        path = self._path(key)
        if isinstance(body, Path):
            payload_hash, size = (payload_hash, body.stat().st_size) if payload_hash else sha256_file(body)
            data = open(body, "rb")
        elif body is not None:
            payload_hash, size, data = hashlib.sha256(body).hexdigest(), len(body), body
        else:
            payload_hash, size, data = EMPTY_SHA256, 0, None
        extra = dict(headers or {})
        if self.session_token:
            extra["x-amz-security-token"] = self.session_token
        if method in ("PUT", "POST"):
            extra["content-length"], extra["content-type"] = str(size), "application/octet-stream"
        target = uri_encode(path, keep_slash=True) + (("?" + canonical_query(query)) if query else "")
        url = urllib.parse.urlunsplit((self.scheme, self.host, uri_encode(path, keep_slash=True), canonical_query(query), ""))
        retries = HEAD_RETRIES if method == "HEAD" else RETRIES
        deadline = HEAD_DEADLINE_S if method == "HEAD" else stall_window or REQUEST_DEADLINE_S + size / MIN_UPLOAD_RATE
        windowed = stall_window is not None and isinstance(data, bytes)
        conn_failed = False
        try:
            for attempt in range(retries + 1):
                if isinstance(body, Path):
                    data.seek(0)
                signed = sign_headers(method, self.host, path, query, extra, payload_hash, self.access_key, self.secret_key, self.region, self._now())
                conn = self.pool.acquire(fresh=True) if conn_failed else self.pool.acquire()
                sent = _Counted(data) if windowed else None
                dog = _Watchdog(conn, deadline, progress=(lambda s=sent: s.n) if sent else None, min_bytes=STALL_MIN_BYTES)
                try:
                    with self._stats_lock:
                        self.requests += 1
                    conn.request(method, target, body=sent if sent is not None else data, headers=signed)     # a file body streams in blocks (content-length is set above)
                    dog.sock = getattr(conn, "sock", None)
                    resp = conn.getresponse()
                except _RETRIABLE as e:
                    dog.cancel()
                    self.pool.discard(conn)
                    conn_failed = True
                    if dog.fired:
                        self._deadline_hit(method, key, attempt + 1, retries + 1, deadline)
                    if attempt == retries:
                        err = (f"{method} {key}: no response within the {deadline:.0f}s deadline on all {retries + 1} tries" if dog.fired
                               else f"{method} {key}: {type(e).__name__}: {e}")
                        with self._stats_lock:
                            self.exhausted.append({"method": method, "key": key, "error": err})
                        if dog.fired:
                            raise TimeoutError(err) from e
                        raise
                    time.sleep(RETRY_BACKOFF_S * (2**attempt))
                    continue
                out = _Response(self.pool, conn, resp, dog)
                if resp.status < 400:
                    return out
                with out:
                    err_body = out.read()
                if (resp.status >= 500 or resp.status == 429) and attempt < retries:
                    time.sleep(retry_delay(resp.headers.get("Retry-After"), attempt))
                    continue
                raise urllib.error.HTTPError(url, resp.status, resp.reason, resp.headers, io.BytesIO(err_body))   # other 4xx are the caller's (404 = absent, 403 = credentials)
        finally:
            if isinstance(body, Path):
                data.close()
        raise AssertionError("unreachable")

    def _deadline_hit(self, method: str, key: str, n: int, of: int, deadline: float) -> None:
        with self._stats_lock:
            self.deadline_hits.append({"method": method, "key": key, "try": n, "of": of, "deadline_s": deadline})
        print(f"research: {method} {key}: no response within {deadline:.0f}s (try {n} of {of})" + ("; retrying" if n < of else ""),
              file=sys.stderr, flush=True)

    def head(self, key: str) -> dict | None:
        """`{"bytes": n, "etag": <hex without quotes, or None>}`, or None when absent."""
        try:
            with self._request("HEAD", key) as r:
                return {"bytes": int(r.headers.get("Content-Length", "0")), "etag": etag_value(r.headers.get("ETag"))}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def _note_existed(self, existed: bool) -> None:
        with self._stats_lock:
            self._existed.append(existed)

    def _prefer_put_first(self) -> bool:
        """PUT-first for small objects unless more than half of the recent ones existed already (module docstring)."""
        with self._stats_lock:
            return sum(self._existed) * 2 <= len(self._existed)

    def _same_object(self, key: str, existing: dict, size: int, md5: str) -> None:
        """RemoteConflict unless what the remote holds under `key` can be this file (same size; same MD5 when the ETag is one)."""
        if existing["bytes"] != size:
            raise RemoteConflict(f"{key}: remote holds {existing['bytes']} bytes, local is {size}")
        if is_md5_etag(existing.get("etag")) and existing["etag"] != md5:
            raise RemoteConflict(f"{key}: remote ETag {existing['etag']} != local MD5 {md5} (same size, different bytes)")

    def put(self, key: str, path: Path) -> dict:
        """Conditional create (module docstring).  Returns `{"action": "skipped"|"created"|"exists", "bytes", "etag", "md5", "sha256",
        "conditional", "fallback"}`: `bytes`/`etag` are what HEAD (skipped) or the PUT response (created) reported, `md5`/`sha256` are
        the local file's, `conditional` says whether this PUT carried `If-None-Match: *`, `fallback` is set on the call that switched
        conditional PUT off.  Objects above `multipart_min` go through `put_multipart` (same contract, `multipart` added)."""
        path = Path(path)
        sha, md5, size = hash_file(path)
        if size > min(self.multipart_min, self.single_put_max):        # parallel parts above `multipart_min`; the only way above `single_put_max`
            return self.put_multipart(key, path, size=size, sha256=sha, md5=md5)
        res: dict = {"action": "created", "bytes": None, "etag": None, "md5": md5, "sha256": sha, "conditional": None, "fallback": None}
        small = size <= self.put_first_max
        put_first = small and self.conditional_put and self.conditional_put_probed and self._prefer_put_first()   # HEAD first until probed
        if not put_first:
            existing = self.head(key)
            if small:
                self._note_existed(existing is not None)
            if existing is not None:
                self._same_object(key, existing, size, md5)
                res.update(action="skipped", bytes=existing["bytes"], etag=existing.get("etag"))
                return res
        if self.conditional_put:
            res["conditional"] = True
            try:
                with self._request("PUT", key, body=path, headers={"if-none-match": "*"}, payload_hash=sha) as r:
                    res["etag"] = etag_value(r.headers.get("ETag"))
                res["bytes"] = size
                if put_first:
                    self._note_existed(False)
                if not self.conditional_put_probed:
                    self._probe_conditional_put(key, path, sha, res)
                return res
            except urllib.error.HTTPError as e:
                if e.code == 412:
                    if put_first:                                       # the key was there all along: HEAD says whether it is this object
                        self._note_existed(True)
                        existing = self.head(key)
                        if existing is None:                            # deleted between the PUT and the HEAD: not an immutable store's behaviour
                            raise RemoteConflict(f"{key}: PUT If-None-Match answered 412 but HEAD finds nothing") from None
                        self._same_object(key, existing, size, md5)
                        res.update(action="skipped", bytes=existing["bytes"], etag=existing.get("etag"))
                        return res
                    res["action"] = "exists"                            # someone else created it between our HEAD and PUT: immutable keys, same bytes
                    return res
                if e.code not in (400, 501):
                    raise
                self.conditional_put_error = f"{error_code(e)} ({e.code}) for `If-None-Match: *`; plain PUT for the rest of this session"
                self.conditional_put = False
                res["fallback"] = self.conditional_put_error
                if put_first:                                           # the HEAD that was skipped: never overwrite blindly
                    existing = self.head(key)
                    if existing is not None:
                        self._same_object(key, existing, size, md5)
                        res.update(action="skipped", bytes=existing["bytes"], etag=existing.get("etag"))
                        return res
        res["conditional"] = False
        with self._request("PUT", key, body=path, payload_hash=sha) as r:
            res["etag"] = etag_value(r.headers.get("ETag"))
        res["bytes"] = size
        return res

    def _probe_conditional_put(self, key: str, path: Path, sha: str, res: dict) -> None:
        """Once per session, right after the first conditional create: PUT the same bytes again with `If-None-Match: *`.  412 = the
        service honours the header (PUT-first is safe from here on); anything else = it ignores the header, and conditional PUT is
        switched off (module docstring).  Re-PUTting the bytes just stored changes nothing either way."""
        with self._probe_lock:
            if self.conditional_put_probed or not self.conditional_put:
                return
            try:
                with self._request("PUT", key, body=path, headers={"if-none-match": "*"}, payload_hash=sha):
                    pass
                honoured = False
            except urllib.error.HTTPError as e:
                if e.code != 412:
                    raise
                honoured = True
            self.conditional_put_probed = True
            if not honoured:
                self.conditional_put_error = "`If-None-Match: *` ignored (a second PUT of a fresh key answered 2xx, not 412); HEAD-then-plain-PUT for the rest of this session"
                self.conditional_put = False
                res["fallback"] = self.conditional_put_error

    def presign(self, key: str, expires_s: int, method: str = "GET") -> str:
        """A URL for one `method` request on `key`, valid `expires_s` seconds (`presign`): read access to one object for a machine
        that holds no key."""
        return presign(method, self.scheme, self.host, self._path(key), self.access_key, self.secret_key, self.region, self._now(),
                       expires_s, self.session_token)

    def delete(self, key: str) -> bool:
        """Remove `key` (True) or find it absent already (False).  Not part of the `Remote` protocol: only the store's GC (README §7.4),
        the bench (`bench/<ts>/...`) and an operator cleaning a throwaway prefix delete."""
        try:
            with self._request("DELETE", key):
                pass
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            raise

    # ---- bounded parallelism ------------------------------------------------------------------------------------------------------

    def _map(self, fn: Callable[[Any], Any], items: Iterable[Any], jobs: int | None = None) -> list[Any]:
        items = list(items)
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=max(1, min(jobs or self.jobs, len(items)))) as ex:
            return list(ex.map(fn, items))

    def put_many(self, items: Iterable[tuple[str, Path]], *, jobs: int | None = None) -> list[dict]:
        """`put` over (key, path) pairs, `jobs` at a time, results in input order.  A failure is the item's result as
        `{"key", "error": "<Type>: <msg>"}` (never raised: one bad object must not stop the rest)."""
        def one(kp: tuple[str, Path]) -> dict:
            k, p = kp
            try:
                return {"key": k, **self.put(k, Path(p))}
            except Exception as e:  # noqa: BLE001
                return {"key": k, "error": f"{type(e).__name__}: {e}"}
        return self._map(one, items, jobs)

    # ---- multipart --------------------------------------------------------------------------------------------------------------------

    def _abort_multipart(self, key: str, upload_id: str) -> None:
        try:
            with self._request("DELETE", key, query={"uploadId": upload_id}):
                pass
        except Exception:  # noqa: BLE001  best effort: an orphaned upload costs storage until the bucket's lifecycle rule clears it
            pass

    def _upload_part(self, key: str, upload_id: str, path: Path, part: dict) -> str:
        with open(path, "rb") as f:
            f.seek(part["offset"])
            data = f.read(part["bytes"])
        if len(data) != part["bytes"] or hashlib.sha256(data).hexdigest() != part["sha256"]:
            raise MultipartError(f"{key} part {part['n']}: the local file changed while uploading")
        with self._request("PUT", key, query={"partNumber": str(part["n"]), "uploadId": upload_id}, body=data, payload_hash=part["sha256"],
                           stall_window=STALL_WINDOW_S) as r:
            etag = etag_value(r.headers.get("ETag"))
        if is_md5_etag(etag) and etag != part["md5"]:
            raise MultipartError(f"{key} part {part['n']}: service ETag {etag} != local MD5 {part['md5']}")
        return etag or part["md5"]

    def put_multipart(self, key: str, path: Path, *, size: int | None = None, sha256: str | None = None, md5: str | None = None,
                      part_size: int | None = None, jobs: int | None = None) -> dict:
        """Module docstring.  Same contract as `put` (immutable key: HEAD first; `skipped` when the service already holds an object of
        this size whose multipart ETag is the one this file yields, RemoteConflict when size or ETag differ), then initiate / parts /
        complete, aborting the upload on any failure.  The sidecar `<key>.parts.json` is written after the object.  Result adds
        `multipart` = {"part_bytes", "parts", "etag", "upload_id", "sidecar"}.  Without an explicit `part_size` the parts are sized by
        `choose_part_size_for` (about 16 parallel parts for a mid-size object); `jobs` parts are in flight at once, capped so that
        `part_mem_budget` bytes of parts are in memory at most."""
        path = Path(path)
        if size is None or sha256 is None or md5 is None:
            sha256, md5, size = hash_file(path)
        part_size = choose_part_size(size, part_size) if part_size else choose_part_size_for(size, self.part_size)
        jobs = max(1, min(jobs or self.part_jobs, max(1, self.part_mem_budget // max(1, part_size))))
        parts = hash_parts(path, part_size)
        expected = multipart_etag([p["md5"] for p in parts])
        side = parts_sidecar(key, size, sha256, part_size, parts)
        res: dict = {"action": "created", "bytes": None, "etag": None, "md5": md5, "sha256": sha256, "conditional": False, "fallback": None,
                     "multipart": {"part_bytes": part_size, "parts": len(parts), "etag": expected, "upload_id": None, "sidecar": sidecar_key(key)}}
        existing = self.head(key)
        if existing is not None:
            if existing["bytes"] != size:
                raise RemoteConflict(f"{key}: remote holds {existing['bytes']} bytes, local is {size}")
            et = existing.get("etag")
            if et and et != expected:
                n = parts_of(et)
                if n is not None and n != len(parts):                    # uploaded with another part size: recompute for that count
                    alt = -(-size // n)
                    alt_parts = hash_parts(path, alt) if plan_parts(size, alt) and len(plan_parts(size, alt)) == n else None
                    if alt_parts is None or multipart_etag([p["md5"] for p in alt_parts]) != et:
                        raise RemoteConflict(f"{key}: remote multipart ETag {et} ({n} parts) is not what this file yields")
                    expected = et
                elif is_md5_etag(et) and et != md5:
                    raise RemoteConflict(f"{key}: remote ETag {et} != local MD5 {md5} (same size, different bytes)")
                elif not is_md5_etag(et):
                    raise RemoteConflict(f"{key}: remote multipart ETag {et} != {expected} (same size, different bytes or part size)")
            res.update(action="skipped", bytes=existing["bytes"], etag=et)
            res["multipart"]["etag"] = expected
            return res
        with self._request("POST", key, query={"uploads": ""}) as r:
            root = ET.fromstring(r.read())
        upload_id = root.findtext(f"{S3_NS}UploadId") or root.findtext("UploadId")
        if not upload_id:
            raise MultipartError(f"{key}: initiate returned no UploadId")
        res["multipart"]["upload_id"] = upload_id
        etags: dict[int, str] = {}
        try:
            with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(parts)))) as ex:
                for p, et in zip(parts, ex.map(lambda p: self._upload_part(key, upload_id, path, p), parts)):
                    etags[p["n"]] = et
            body = ("<CompleteMultipartUpload>" + "".join(f"<Part><PartNumber>{n}</PartNumber><ETag>\"{etags[n]}\"</ETag></Part>" for n in sorted(etags))
                    + "</CompleteMultipartUpload>").encode()
            with self._request("POST", key, query={"uploadId": upload_id}, body=body) as r:
                doc = ET.fromstring(r.read())
            got = etag_value(doc.findtext(f"{S3_NS}ETag") or doc.findtext("ETag") or r.headers.get("ETag"))
        except BaseException:
            self._abort_multipart(key, upload_id)
            raise
        if got != expected:
            head = self.head(key)
            if head is None or head.get("etag") != expected:
                raise MultipartError(f"{key}: completed multipart ETag {got} != {expected} computed from the local parts")
            got = expected
        res["etag"], res["bytes"] = got, size
        import json as _json
        import tempfile

        with tempfile.NamedTemporaryFile("wb", suffix=".parts.json", delete=False) as tf:
            tf.write(_json.dumps(side, sort_keys=True, separators=(",", ":")).encode())
            tmp = Path(tf.name)
        try:
            self.put(sidecar_key(key), tmp)
        finally:
            tmp.unlink(missing_ok=True)
        return res

    def get(self, key: str, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".tmp-{path.name}-{os.getpid()}-{threading.get_ident()}"
        for attempt in range(RETRIES + 1):                   # the whole GET is retried: a body cut off by its deadline or the network
            try:
                with self._request("GET", key) as r, open(tmp, "wb") as f:
                    want = r.headers.get("Content-Length")
                    got, cut = [0], REQUEST_DEADLINE_S
                    if r.watchdog is not None and int(want or 0) > GET_STALL_MIN_BYTES:
                        cut = STALL_WINDOW_S
                        r.watchdog.window(cut, lambda: got[0], GET_STALL_MIN_BYTES)
                    elif r.watchdog is not None:
                        r.watchdog.extend(max(REQUEST_DEADLINE_S, int(want or 0) / MIN_BODY_RATE))
                    n = 0
                    for chunk in iter(lambda: r.read(1 << 20), b""):
                        f.write(chunk)
                        n += len(chunk)
                        got[0] = n
                    if r.watchdog is not None and r.watchdog.fired:
                        self._deadline_hit("GET", key, attempt + 1, RETRIES + 1, cut)
                        raise TimeoutError(f"GET {key}: body not complete within its deadline ({n} bytes)")
                    if want is not None and n != int(want):
                        raise http.client.IncompleteRead(b"", int(want) - n)
                os.replace(tmp, path)
                return
            except urllib.error.HTTPError as e:
                tmp.unlink(missing_ok=True)
                if e.code == 404:
                    raise KeyError(key) from None
                raise
            except _RETRIABLE as e:
                tmp.unlink(missing_ok=True)
                if attempt == RETRIES:
                    with self._stats_lock:
                        self.exhausted.append({"method": "GET", "key": key, "error": f"{type(e).__name__}: {e}"})
                    raise
                time.sleep(RETRY_BACKOFF_S * (2**attempt))
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise

    def list_meta(self, prefix: str, progress: Callable[[int], None] | None = None, keep: Callable[[str], bool] | None = None) -> dict[str, tuple]:
        """`{key: (bytes, etag)}` for every key under `prefix` that `keep` accepts, from the listing itself (1,000 keys per request, each request under
        the hard deadline): one pass over a prefix instead of a HEAD per key.  `progress(n)` after each page."""
        out: dict[str, dict] = {}
        token = None
        while True:
            q = {"list-type": "2", "prefix": prefix}
            if token:
                q["continuation-token"] = token
            with self._request("GET", "", query=q) as r:
                root = ET.parse(io.BytesIO(r.read())).getroot()
            page = 0
            for c in root.iter(f"{S3_NS}Contents"):
                k = c.findtext(f"{S3_NS}Key")
                if k and (keep is None or keep(k)):
                    out[k] = (int(c.findtext(f"{S3_NS}Size") or 0), etag_value(c.findtext(f"{S3_NS}ETag")))
                page += 1
            if progress:
                progress(page)
            if root.findtext(f"{S3_NS}IsTruncated") != "true":
                return out
            token = root.findtext(f"{S3_NS}NextContinuationToken")

    def list_times(self, prefix: str, start_after: str | None = None) -> list[tuple[str, str]]:
        """`[(key, LastModified)]` under `prefix`, sorted; with `start_after`, only the keys after it (ListObjectsV2 `start-after`,
        so a listing of new run ids costs one request rather than one per 1,000 runs)."""
        out, token = [], None
        while True:
            q = {"list-type": "2", "prefix": prefix}
            if start_after:
                q["start-after"] = start_after
            if token:
                q["continuation-token"] = token
            with self._request("GET", "", query=q) as r:
                root = ET.parse(io.BytesIO(r.read())).getroot()
            out += [(c.findtext(f"{S3_NS}Key"), c.findtext(f"{S3_NS}LastModified")) for c in root.iter(f"{S3_NS}Contents")]
            if root.findtext(f"{S3_NS}IsTruncated") != "true":
                return sorted((k, m) for k, m in out if k)
            token = root.findtext(f"{S3_NS}NextContinuationToken")

    def list(self, prefix: str) -> list[str]:
        keys, token = [], None
        while True:
            q = {"list-type": "2", "prefix": prefix}
            if token:
                q["continuation-token"] = token
            with self._request("GET", "", query=q) as r:
                root = ET.parse(io.BytesIO(r.read())).getroot()
            keys += [c.findtext(f"{S3_NS}Key") for c in root.iter(f"{S3_NS}Contents")]
            if root.findtext(f"{S3_NS}IsTruncated") != "true":
                return sorted(k for k in keys if k)
            token = root.findtext(f"{S3_NS}NextContinuationToken")
