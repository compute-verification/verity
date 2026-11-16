"""`S3Remote`: an S3-compatible bucket over stdlib `http.client` with hand-rolled SigV4."""
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from .remote import RemoteConflict

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SINGLE_PUT_MAX = 5 << 30        # PutObject carries at most 5 GiB (S3 and R2 alike); above that only multipart works
MULTIPART_MIN = 64 << 20
PUT_FIRST_MAX = 64 << 10
ADAPT_WINDOW = 16
PART_SIZE = 256 << 20
PART_SIZE_MIN = 16 << 20
TARGET_PARTS = 16
MAX_PARTS = 10_000
PART_JOBS = 8
PART_MEM_BUDGET = 1 << 30       # ... but never more than this many bytes of parts in flight
PARTS_SIDECAR_SCHEMA = "research/parts/v1"
REQUEST_DEADLINE_S = 180.0
MIN_BODY_RATE = 64 * 1024
MIN_UPLOAD_RATE = 8 * 1024       # the same for a request body (a PUT from the laptop's 25-220 KB/s uplink must still finish)
STALL_WINDOW_S = 120.0
STALL_MIN_BYTES = 1 << 20
GET_STALL_MIN_BYTES = 32 << 20
HEAD_DEADLINE_S = 30.0
HEAD_RETRIES = 1
RETRIES = 3
RETRY_BACKOFF_S = 1.0
RETRY_AFTER_MAX_S = 30.0
JOBS = 32
MAX_POOL_CONNECTIONS = 64
PACK_THRESHOLD_BLOBS_PER_S = 300
ALGORITHM = "AWS4-HMAC-SHA256"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def uri_encode(s: str, *, keep_slash: bool = False) -> str:
    return urllib.parse.quote(s, safe="-_.~/" if keep_slash else "-_.~")


def canonical_query(query: dict[str, str] | list[tuple[str, str]]) -> str:
    items = list(query.items()) if isinstance(query, dict) else list(query)
    return "&".join(f"{k}={v}" for k, v in sorted((uri_encode(k), uri_encode(v)) for k, v in items))


def canonical_request(method: str, path: str, query: dict[str, str] | list[tuple[str, str]], headers: dict[str, str], payload_hash: str) -> str:
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
    h = {**headers, "host": host, "x-amz-date": amz_date, "x-amz-content-sha256": payload_hash}
    creq = canonical_request(method, path, query, h, payload_hash)
    scope = f"{amz_date[:8]}/{region}/{service}/aws4_request"
    sig = signature(secret_key, amz_date[:8], region, string_to_sign(amz_date, scope, creq), service)
    signed = ";".join(sorted(k.lower() for k in h))
    h["Authorization"] = f"{ALGORITHM} Credential={access_key}/{scope},SignedHeaders={signed},Signature={sig}"
    return h


def presign(method: str, scheme: str, host: str, path: str, access_key: str, secret_key: str, region: str, amz_date: str,
            expires_s: int, session_token: str | None = None, service: str = "s3") -> str:
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
    s, m, n = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            s.update(chunk)
            m.update(chunk)
            n += len(chunk)
    return s.hexdigest(), m.hexdigest(), n


def etag_value(raw: str | None) -> str | None:
    if raw is None:
        return None
    v = raw.strip()
    if v.startswith("W/"):
        v = v[2:]
    return v.strip('"').lower() or None


def is_md5_etag(etag: str | None) -> bool:
    return bool(etag) and "-" not in etag and len(etag) == 32


def error_code(e: urllib.error.HTTPError) -> str:
    try:
        root = ET.fromstring(e.read())
        return root.findtext("Code") or root.findtext(f"{S3_NS}Code") or f"HTTP {e.code}"
    except Exception:  # noqa: BLE001  (no body, or not XML: the status line is all there is)
        return f"HTTP {e.code}"


def retry_delay(retry_after: str | None, attempt: int) -> float:
    try:
        if retry_after is not None:
            return max(0.0, min(float(retry_after), RETRY_AFTER_MAX_S))
    except ValueError:
        pass
    return RETRY_BACKOFF_S * (2**attempt)


def choose_part_size(size: int, part_size: int = PART_SIZE, max_parts: int = MAX_PARTS) -> int:
    need = -(-size // max_parts)
    if need <= part_size:
        return part_size
    return -(-need // (1 << 20)) * (1 << 20)


def choose_part_size_for(size: int, part_size: int = PART_SIZE, target_parts: int = TARGET_PARTS, part_min: int = PART_SIZE_MIN) -> int:
    want = -(-size // max(1, target_parts))
    want = -(-want // (1 << 20)) * (1 << 20)
    return choose_part_size(size, min(part_size, max(part_min, want)))


def plan_parts(size: int, part_size: int) -> list[tuple[int, int, int]]:
    if size <= 0:
        return [(1, 0, 0)]
    return [(i + 1, off, min(part_size, size - off)) for i, off in enumerate(range(0, size, part_size))]


def multipart_etag(part_md5s: list[str]) -> str:
    return hashlib.md5(b"".join(bytes.fromhex(m) for m in part_md5s), usedforsecurity=False).hexdigest() + f"-{len(part_md5s)}"


def parts_of(etag: str | None) -> int | None:
    if etag and "-" in etag:
        n = etag.rsplit("-", 1)[1]
        return int(n) if n.isdigit() else None
    return None


def hash_parts(path: Path, part_size: int) -> list[dict]:
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


_RETRIABLE = (http.client.HTTPException, ConnectionError, TimeoutError, socket.timeout, ssl.SSLError, OSError)


class _Pool:
    def __init__(self, scheme: str, host: str, timeout: float, limit: int = MAX_POOL_CONNECTIONS):
        self.scheme, self.timeout, self.limit = scheme, timeout, limit
        self.host, _, port = host.partition(":")
        self.port = int(port) if port else (443 if scheme == "https" else 80)
        self._idle: list[http.client.HTTPConnection] = []
        self._lock = threading.Lock()
        self._ctx = ssl.create_default_context() if scheme == "https" else None
        proxies = urllib.request.getproxies_environment()
        self._proxy = urllib.parse.urlsplit(proxies[scheme]) if proxies.get(scheme) and not urllib.request.proxy_bypass_environment(self.host, proxies) else None
        self.opened = 0

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

    timer = threading.Timer

    def __init__(self, conn: http.client.HTTPConnection, seconds: float, progress: Callable[[], int] | None = None, min_bytes: int = 0):
        self.progress, self.min_bytes, self.seconds, self._seen = progress, min_bytes, seconds, 0
        self.conn, self.fired, self.sock = conn, False, None
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
    def __init__(self, data: bytes):
        self._f, self.n = io.BytesIO(data), 0

    def read(self, size: int = -1) -> bytes:
        b = self._f.read(size)
        self.n += len(b)
        return b


class _Response:
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


class S3Remote:
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
        self.conditional_put = True
        self.conditional_put_probed = False
        self.conditional_put_error: str | None = None
        self._probe_lock = threading.Lock()
        self._existed: deque[bool] = deque(maxlen=ADAPT_WINDOW)
        self.pool = _Pool(self.scheme, self.host, timeout, limit=max(1, int(max_pool_connections)))
        self.requests = 0
        self.deadline_hits: list[dict] = []
        self.exhausted: list[dict] = []
        self._stats_lock = threading.Lock()

    def describe(self) -> str:
        return f"s3://{self.bucket} @ {self.scheme}://{self.host}"

    def close(self) -> None:
        self.pool.close()

    def stats(self) -> dict:
        return {"requests": self.requests, "connections_opened": self.pool.opened, "idle": self.pool.idle, "jobs": self.jobs,
                "conditional_put": self.conditional_put, "conditional_put_probed": self.conditional_put_probed, "conditional_put_error": self.conditional_put_error,
                "put_first": self._prefer_put_first()}

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    def _path(self, key: str) -> str:
        return f"{self.base_path}/{self.bucket}/{key}" if key else f"{self.base_path}/{self.bucket}"

    def _request(self, method: str, key: str, query: dict[str, str] | None = None, body: bytes | Path | None = None, headers: dict[str, str] | None = None,
                 payload_hash: str | None = None, stall_window: float | None = None) -> _Response:
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
                    conn.request(method, target, body=sent if sent is not None else data, headers=signed)
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
        with self._stats_lock:
            return sum(self._existed) * 2 <= len(self._existed)

    def _same_object(self, key: str, existing: dict, size: int, md5: str) -> None:
        if existing["bytes"] != size:
            raise RemoteConflict(f"{key}: remote holds {existing['bytes']} bytes, local is {size}")
        if is_md5_etag(existing.get("etag")) and existing["etag"] != md5:
            raise RemoteConflict(f"{key}: remote ETag {existing['etag']} != local MD5 {md5} (same size, different bytes)")

    def put(self, key: str, path: Path) -> dict:
        path = Path(path)
        sha, md5, size = hash_file(path)
        if size > min(self.multipart_min, self.single_put_max):
            return self.put_multipart(key, path, size=size, sha256=sha, md5=md5)
        res: dict = {"action": "created", "bytes": None, "etag": None, "md5": md5, "sha256": sha, "conditional": None, "fallback": None}
        small = size <= self.put_first_max
        put_first = small and self.conditional_put and self.conditional_put_probed and self._prefer_put_first()
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
                    if put_first:
                        self._note_existed(True)
                        existing = self.head(key)
                        if existing is None:
                            raise RemoteConflict(f"{key}: PUT If-None-Match answered 412 but HEAD finds nothing") from None
                        self._same_object(key, existing, size, md5)
                        res.update(action="skipped", bytes=existing["bytes"], etag=existing.get("etag"))
                        return res
                    res["action"] = "exists"
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

    def create(self, key: str, path: Path) -> dict:
        path = Path(path)
        sha, md5, size = hash_file(path)
        if size > self.single_put_max:
            raise ValueError(f"{key}: {size} bytes is above the {self.single_put_max}-byte single PUT, and a multipart create "
                             "can't be conditional without reading the bucket")
        res: dict = {"action": "created", "bytes": size, "etag": None, "md5": md5, "sha256": sha, "conditional": True, "fallback": None}
        try:
            with self._request("PUT", key, body=path, headers={"if-none-match": "*"}, payload_hash=sha) as r:
                res["etag"] = etag_value(r.headers.get("ETag"))
        except urllib.error.HTTPError as e:
            if e.code != 412:
                raise
            res["action"] = "exists"
        return res

    def replace(self, key: str, path: Path) -> None:
        path = Path(path)
        sha, size = sha256_file(path)
        if size > self.single_put_max:
            raise ValueError(f"{key}: {size} bytes is over one PUT's {self.single_put_max}")
        with self._request("PUT", key, body=path, payload_hash=sha):
            pass

    def presign(self, key: str, expires_s: int, method: str = "GET") -> str:
        return presign(method, self.scheme, self.host, self._path(key), self.access_key, self.secret_key, self.region, self._now(),
                       expires_s, self.session_token)

    def delete(self, key: str) -> bool:
        try:
            with self._request("DELETE", key):
                pass
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            raise

    def _map(self, fn: Callable[[Any], Any], items: Iterable[Any], jobs: int | None = None) -> list[Any]:
        items = list(items)
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=max(1, min(jobs or self.jobs, len(items)))) as ex:
            return list(ex.map(fn, items))

    def put_many(self, items: Iterable[tuple[str, Path]], *, jobs: int | None = None) -> list[dict]:
        """Never raises: a failure is the item's result as `{"key", "error"}`."""
        def one(kp: tuple[str, Path]) -> dict:
            k, p = kp
            try:
                return {"key": k, **self.put(k, Path(p))}
            except Exception as e:  # noqa: BLE001
                return {"key": k, "error": f"{type(e).__name__}: {e}"}
        return self._map(one, items, jobs)

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
                if n is not None and n != len(parts):
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
        for attempt in range(RETRIES + 1):
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
