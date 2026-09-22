"""`S3Remote`: an S3-compatible bucket (Cloudflare R2, AWS) over stdlib `urllib` with hand-rolled SigV4 (README §7).

Path-style addressing (`<endpoint>/<bucket>/<key>`), signed payloads (`x-amz-content-sha256` is the real sha256 of the body,
never UNSIGNED-PAYLOAD), one PUT for objects <= 4 GiB.  Multipart above that is not implemented yet: the design is one
`UploadPart` per 256 MiB with the per-part sha256 recorded in a sidecar key `<key>.parts.json` so that PRESERVED can be checked
part by part (ETags are not content hashes).  `put` refuses to overwrite a key whose HEAD size differs (RemoteConflict).

The signing functions are pure and tested against the AWS documented vectors (`sign_headers`, `canonical_request`,
`string_to_sign`, `signature`).
"""
from __future__ import annotations

import hashlib
import hmac
import io
import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from .remote import RemoteConflict

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SINGLE_PUT_MAX = 4 << 30
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


# ---- client -----------------------------------------------------------------------------------------------------------------------------

class S3Remote:
    def __init__(self, endpoint: str, bucket: str, access_key: str, secret_key: str, region: str = "auto", timeout: float = 120.0):
        u = urllib.parse.urlsplit(endpoint)
        if u.scheme not in ("http", "https") or not u.netloc:
            raise ValueError(f"endpoint must be an http(s) URL, got {endpoint!r}")
        self.scheme, self.host, self.base_path = u.scheme, u.netloc, u.path.rstrip("/")
        self.bucket, self.access_key, self.secret_key, self.region, self.timeout = bucket, access_key, secret_key, region, timeout

    def describe(self) -> str:
        return f"s3://{self.bucket} @ {self.scheme}://{self.host}"

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _path(self, key: str) -> str:
        return f"{self.base_path}/{self.bucket}/{key}" if key else f"{self.base_path}/{self.bucket}"

    def _request(self, method: str, key: str, query: dict[str, str] | None = None, body: bytes | Path | None = None, headers: dict[str, str] | None = None):
        query = query or {}
        path = self._path(key)
        if isinstance(body, Path):
            payload_hash, size = sha256_file(body)
            data = open(body, "rb")
        elif body is not None:
            payload_hash, size, data = hashlib.sha256(body).hexdigest(), len(body), body
        else:
            payload_hash, size, data = EMPTY_SHA256, 0, None
        extra = dict(headers or {})
        if method in ("PUT", "POST"):
            extra["content-length"] = str(size)
        signed = sign_headers(method, self.host, path, query, extra, payload_hash, self.access_key, self.secret_key, self.region, self._now())
        url = urllib.parse.urlunsplit((self.scheme, self.host, uri_encode(path, keep_slash=True), canonical_query(query), ""))
        req = urllib.request.Request(url, data=data, method=method, headers=signed)
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)  # noqa: S310  (scheme checked in __init__)
        finally:
            if isinstance(body, Path):
                data.close()

    def head(self, key: str) -> dict | None:
        try:
            with self._request("HEAD", key) as r:
                return {"bytes": int(r.headers.get("Content-Length", "0")), "etag": r.headers.get("ETag")}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def put(self, key: str, path: Path) -> None:
        path = Path(path)
        size = path.stat().st_size
        existing = self.head(key)
        if existing is not None:
            if existing["bytes"] != size:
                raise RemoteConflict(f"{key}: remote holds {existing['bytes']} bytes, local is {size}")
            return
        if size > SINGLE_PUT_MAX:
            raise NotImplementedError(f"{key}: {size} bytes > 4 GiB needs multipart upload with a {key}.parts.json sidecar (not implemented)")
        with self._request("PUT", key, body=path):
            pass

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
