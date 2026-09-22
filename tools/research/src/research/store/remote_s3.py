"""`S3Remote`: an S3-compatible bucket (Cloudflare R2, AWS) over stdlib `urllib` with hand-rolled SigV4 (README §7).

Path-style addressing (`<endpoint>/<bucket>/<key>`), signed payloads (`x-amz-content-sha256` is the real sha256 of the body,
never UNSIGNED-PAYLOAD), one PUT for objects <= 5 GiB (`SINGLE_PUT_MAX`, the service's PutObject limit).  Multipart above
that is not implemented: the design is one `UploadPart` per 256 MiB with the per-part sha256 recorded in a sidecar key
`<key>.parts.json` so that PRESERVED can be checked part by part (multipart ETags are not content hashes).

`put(key, path)` is a conditional create for an immutable key: HEAD first (present with the same size and, for a single-part
ETag, the same MD5 -> `skipped`; a different size or MD5 -> RemoteConflict), else `PUT` with `If-None-Match: *`.  412 from the
service means another writer got there first (`exists`); 400/501 means the service does not implement the header, so conditional
PUT is switched off for the rest of this session (`conditional_put`, the reason in `conditional_put_error`) and the object goes
up with a plain PUT.  The result dict says what happened (`action`, the local `md5`/`sha256`, the ETag seen, `fallback`).

The signing functions are pure and tested against the AWS documented vectors (`sign_headers`, `canonical_request`,
`string_to_sign`, `signature`).  One instance may be used from several threads (`LocalStore.push` does): the opener holds no
connection state and the only shared mutable is the conditional-PUT flag.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from .remote import RemoteConflict

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SINGLE_PUT_MAX = 5 << 30        # PutObject carries at most 5 GiB (S3 and R2 alike); above that only multipart works
RETRIES = 3                     # transient 5xx / connection errors: 4 tries, 1 s, 2 s, 4 s backoff
RETRY_BACKOFF_S = 1.0
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


# ---- client -----------------------------------------------------------------------------------------------------------------------------

class S3Remote:
    """`session_token` (a temporary credential's third part, `AWS_SESSION_TOKEN`) travels as `x-amz-security-token`, inside the
    signed headers, on every request."""

    single_put_max = SINGLE_PUT_MAX

    def __init__(self, endpoint: str, bucket: str, access_key: str, secret_key: str, region: str = "auto", timeout: float = 120.0,
                 session_token: str | None = None):
        u = urllib.parse.urlsplit(endpoint)
        if u.scheme not in ("http", "https") or not u.netloc:
            raise ValueError(f"endpoint must be an http(s) URL, got {endpoint!r}")
        self.scheme, self.host, self.base_path = u.scheme, u.netloc, u.path.rstrip("/")
        self.bucket, self.access_key, self.secret_key, self.region, self.timeout = bucket, access_key, secret_key, region, timeout
        self.session_token = session_token or None
        self.conditional_put = True                 # `If-None-Match: *` on PUT until the service says it does not implement it
        self.conditional_put_error: str | None = None
        # env-only proxy lookup: the default opener consults macOS SystemConfiguration (_scproxy), which leaves a system thread behind
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler(urllib.request.getproxies_environment()))

    def describe(self) -> str:
        return f"s3://{self.bucket} @ {self.scheme}://{self.host}"

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _path(self, key: str) -> str:
        return f"{self.base_path}/{self.bucket}/{key}" if key else f"{self.base_path}/{self.bucket}"

    def _request(self, method: str, key: str, query: dict[str, str] | None = None, body: bytes | Path | None = None, headers: dict[str, str] | None = None,
                 payload_hash: str | None = None):
        """One signed request with retries on 5xx / connection errors.  `payload_hash` (sha256 hex of a Path body) skips re-hashing
        when the caller already has it."""
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
        url = urllib.parse.urlunsplit((self.scheme, self.host, uri_encode(path, keep_slash=True), canonical_query(query), ""))
        try:
            for attempt in range(RETRIES + 1):
                # each try is re-signed (fresh timestamp) and, for a file body, re-read from the start
                if isinstance(body, Path):
                    data.seek(0)
                signed = sign_headers(method, self.host, path, query, extra, payload_hash, self.access_key, self.secret_key, self.region, self._now())
                req = urllib.request.Request(url, data=data, method=method, headers=signed)
                try:
                    return self._opener.open(req, timeout=self.timeout)  # noqa: S310  (scheme checked in __init__)
                except urllib.error.HTTPError as e:
                    if e.code < 500 or attempt == RETRIES:            # 4xx are the caller's (404 = absent, 403 = credentials); 5xx are retried
                        raise
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                    if attempt == RETRIES:
                        raise
                time.sleep(RETRY_BACKOFF_S * (2**attempt))
        finally:
            if isinstance(body, Path):
                data.close()
        raise AssertionError("unreachable")

    def head(self, key: str) -> dict | None:
        """`{"bytes": n, "etag": <hex without quotes, or None>}`, or None when absent."""
        try:
            with self._request("HEAD", key) as r:
                return {"bytes": int(r.headers.get("Content-Length", "0")), "etag": etag_value(r.headers.get("ETag"))}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def put(self, key: str, path: Path) -> dict:
        """Conditional create (module docstring).  Returns `{"action": "skipped"|"created"|"exists", "bytes", "etag", "md5", "sha256",
        "conditional", "fallback"}`: `bytes`/`etag` are what HEAD (skipped) or the PUT response (created) reported, `md5`/`sha256` are
        the local file's, `conditional` says whether this PUT carried `If-None-Match: *`, `fallback` is set on the call that switched
        conditional PUT off."""
        path = Path(path)
        sha, md5, size = hash_file(path)
        if size > self.single_put_max:
            raise NotImplementedError(f"{key}: {size} bytes > {self.single_put_max} (one PutObject); multipart upload with a {key}.parts.json sidecar is not implemented")
        res: dict = {"action": "created", "bytes": None, "etag": None, "md5": md5, "sha256": sha, "conditional": None, "fallback": None}
        existing = self.head(key)
        if existing is not None:
            if existing["bytes"] != size:
                raise RemoteConflict(f"{key}: remote holds {existing['bytes']} bytes, local is {size}")
            if is_md5_etag(existing.get("etag")) and existing["etag"] != md5:
                raise RemoteConflict(f"{key}: remote ETag {existing['etag']} != local MD5 {md5} (same size, different bytes)")
            res.update(action="skipped", bytes=existing["bytes"], etag=existing.get("etag"))
            return res
        if self.conditional_put:
            res["conditional"] = True
            try:
                with self._request("PUT", key, body=path, headers={"if-none-match": "*"}, payload_hash=sha) as r:
                    res["etag"] = etag_value(r.headers.get("ETag"))
                return res
            except urllib.error.HTTPError as e:
                if e.code == 412:                                       # someone else created it between our HEAD and PUT: immutable keys, same bytes
                    res["action"] = "exists"
                    return res
                if e.code not in (400, 501):
                    raise
                self.conditional_put_error = f"{error_code(e)} ({e.code}) for `If-None-Match: *`; plain PUT for the rest of this session"
                self.conditional_put = False
                res["fallback"] = self.conditional_put_error
        res["conditional"] = False
        with self._request("PUT", key, body=path, payload_hash=sha) as r:
            res["etag"] = etag_value(r.headers.get("ETag"))
        return res

    def get(self, key: str, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".tmp-{path.name}-{os.getpid()}"
        try:
            with self._request("GET", key) as r, open(tmp, "wb") as f:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    f.write(chunk)
            os.replace(tmp, path)
        except urllib.error.HTTPError as e:
            tmp.unlink(missing_ok=True)
            if e.code == 404:
                raise KeyError(key) from None
            raise
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

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
