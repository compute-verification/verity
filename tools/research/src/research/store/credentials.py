"""Short-lived, bucket-scoped R2 credentials for pods (README §7): `research data mint-credential`.

Cloudflare R2 derives temporary S3 credentials from a parent R2 API token; they sign with SigV4 like any key pair plus a session
token (`X-Amz-Security-Token`, which `S3Remote(session_token=...)` sends).  Two ways to mint, both stdlib:

  local   `mint_local`: sign a JWT (HS256) with the PARENT SECRET ACCESS KEY -- no Cloudflare API call, works offline, and the only
          way to scope by S3 action (`actions`).  Ported claim for claim from Cloudflare's reference implementation
          (developers.cloudflare.com/r2/examples/authenticate-r2-temp-credentials/):
              header  {"alg": "HS256", "typ": "JWT"}
              claims  {bucket, scope | actions, paths?: {prefixPaths, objectPaths}, sub: <account id>, iss: <parent access key id>,
                       aud: <endpoint host>, iat, exp: iat + ttl}      (scope and actions together -> 400 from R2; one or the other)
              temporary access key id  = the parent access key id
              temporary secret         = sha256 hex of the signed JWT (the compact `h.p.s` string)
              session token            = standard base64 of "jwt/" + JWT
  api     `mint_api`: POST /client/v4/accounts/{account}/r2/temp-access-credentials with a Cloudflare API token (`CLOUDFLARE_API_TOKEN`)
          -> {accessKeyId, secretAccessKey, sessionToken}.  Cannot scope by action (the API has `permission`, `prefixes`, `objects`).

Scopes (`permission`): object-read-only | object-read-write | admin-read-only | admin-read-write.  TTL <= 7 days (604800 s).  A
temporary credential never exceeds its parent; revoking the parent kills every child.  Nothing here writes a secret to disk: the CLI
prints JSON, or `--env` export lines for a pod's bootstrap.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

SCOPES = ("object-read-only", "object-read-write", "admin-read-only", "admin-read-write")
ACTIONS = frozenset({"HeadObject", "GetObject", "GetBucketLocation", "ListObjectsV1", "ListObjectsV2", "ListMultipartUploads", "ListParts",
                     "PutObject", "DeleteObject", "DeleteObjects", "CopyObject",
                     "CreateMultipartUpload", "UploadPart", "UploadPartCopy", "AbortMultipartUpload", "CompleteMultipartUpload"})
TTL_MAX_S = 604800
CF_API = "https://api.cloudflare.com/client/v4"
PARENT_ENV_FILE = "~/.config/verity/r2.env"      # where the control machine keeps the parent R2 key (store README); never on a pod
_TTL_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


class MintError(ValueError):
    """Cannot mint: missing inputs, a bad request, or the API said no."""


def parse_ttl(text: str | int) -> int:
    """`900`, `15m`, `6h`, `7d` -> seconds; 1 <= ttl <= 604800."""
    s = str(text).strip().lower()
    m = re.fullmatch(r"(\d+)([smhd]?)", s)
    if not m:
        raise MintError(f"--ttl wants <n>[s|m|h|d], got {text!r}")
    ttl = int(m.group(1)) * _TTL_UNITS[m.group(2) or "s"]
    if not 1 <= ttl <= TTL_MAX_S:
        raise MintError(f"--ttl must be between 1 s and {TTL_MAX_S} s (7 days), got {ttl}")
    return ttl


def account_id_from_endpoint(endpoint: str) -> str | None:
    """`https://<account>.r2.cloudflarestorage.com` -> `<account>`; None for any other host."""
    m = re.fullmatch(r"([0-9a-f]{32})\.r2\.cloudflarestorage\.com", urllib.parse.urlsplit(endpoint).hostname or "")
    return m.group(1) if m else None


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _json_compact(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")   # JSON.stringify: insertion order, no spaces


def sign_jwt_hs256(claims: dict, secret: str) -> str:
    """Compact JWS, HS256, header `{"alg":"HS256","typ":"JWT"}`; claims serialised in the given order."""
    head = _b64url(_json_compact({"alg": "HS256", "typ": "JWT"}))
    body = _b64url(_json_compact(claims))
    sig = hmac.new(secret.encode("utf-8"), f"{head}.{body}".encode("ascii"), hashlib.sha256).digest()
    return f"{head}.{body}.{_b64url(sig)}"


def decode_jwt(token: str) -> tuple[dict, dict]:
    """(header, claims) of a compact JWT, unverified (for tests and `--json` inspection)."""
    h, p, _ = token.split(".")
    pad = lambda s: s + "=" * (-len(s) % 4)  # noqa: E731
    return json.loads(base64.urlsafe_b64decode(pad(h))), json.loads(base64.urlsafe_b64decode(pad(p)))


def mint_local(endpoint: str, account_id: str, parent_access_key_id: str, parent_secret_access_key: str, bucket: str, scope: str, *,
               actions: list[str] | None = None, ttl_s: int = 3600, prefixes: list[str] | None = None, objects: list[str] | None = None,
               now: int | None = None) -> dict:
    """Cloudflare's `createTempCredentials`, claim for claim.  Returns {accessKeyId, secretAccessKey, sessionToken, expires_utc, jwt}."""
    if scope not in SCOPES:
        raise MintError(f"scope must be one of {', '.join(SCOPES)}, got {scope!r}")
    bad = sorted(set(actions or []) - ACTIONS)
    if bad:
        raise MintError(f"unknown S3 action(s) {bad}; valid: {', '.join(sorted(ACTIONS))}")
    if not 1 <= ttl_s <= TTL_MAX_S:
        raise MintError(f"ttl must be between 1 and {TTL_MAX_S} s")
    iat = int(time.time()) if now is None else int(now)
    # `scope` OR `actions`, not both: R2 answers 400 to a JWT carrying both (checked live 2026-09-22 against verity-dev), although the
    # reference TypeScript sets both; the prose ("specify permitted operations using scope or actions") is what the service implements.
    claims: dict[str, Any] = {"bucket": bucket, "actions": list(actions)} if actions else {"bucket": bucket, "scope": scope}
    if prefixes is not None or objects is not None:
        claims["paths"] = {"prefixPaths": list(prefixes or []), "objectPaths": list(objects or [])}
    claims.update(sub=account_id, iss=parent_access_key_id, aud=urllib.parse.urlsplit(endpoint).netloc, iat=iat, exp=iat + ttl_s)
    jwt = sign_jwt_hs256(claims, parent_secret_access_key)
    return {"accessKeyId": parent_access_key_id, "secretAccessKey": hashlib.sha256(jwt.encode("utf-8")).hexdigest(),
            "sessionToken": base64.b64encode(f"jwt/{jwt}".encode("utf-8")).decode("ascii"),
            "expires_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(iat + ttl_s)), "jwt": jwt}


def mint_api(account_id: str, api_token: str, bucket: str, parent_access_key_id: str, permission: str, *, ttl_s: int = 3600,
             prefixes: list[str] | None = None, objects: list[str] | None = None, opener: Any = None) -> dict:
    """POST temp-access-credentials.  Returns {accessKeyId, secretAccessKey, sessionToken, expires_utc}."""
    if permission not in SCOPES:
        raise MintError(f"permission must be one of {', '.join(SCOPES)}, got {permission!r}")
    if not 1 <= ttl_s <= TTL_MAX_S:
        raise MintError(f"ttl must be between 1 and {TTL_MAX_S} s")
    body: dict[str, Any] = {"bucket": bucket, "parentAccessKeyId": parent_access_key_id, "permission": permission, "ttlSeconds": ttl_s}
    if prefixes:
        body["prefixes"] = list(prefixes)
    if objects:
        body["objects"] = list(objects)
    req = urllib.request.Request(f"{CF_API}/accounts/{account_id}/r2/temp-access-credentials", data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"})
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler(urllib.request.getproxies_environment()))
    t0 = int(time.time())
    try:
        with opener.open(req, timeout=60) as r:  # noqa: S310
            doc = json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            errs = json.loads(e.read()).get("errors")
        except Exception:  # noqa: BLE001
            errs = None
        raise MintError(f"Cloudflare API: HTTP {e.code} {e.reason}" + (f": {errs}" if errs else "")) from None
    if not doc.get("success") or not isinstance(doc.get("result"), dict):
        raise MintError(f"Cloudflare API refused: {doc.get('errors') or doc}")
    res = doc["result"]
    for k in ("accessKeyId", "secretAccessKey", "sessionToken"):
        if not res.get(k):
            raise MintError(f"Cloudflare API result lacks {k}")
    return {k: res[k] for k in ("accessKeyId", "secretAccessKey", "sessionToken")} | {"expires_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 + ttl_s))}


def remote_config(config_path: str | os.PathLike[str] | None = None) -> dict:
    """The `[remote]` table of store.toml when it is an s3 one, else {} (the CLI falls back to R2_ENDPOINT / R2_BUCKET / AWS_* env)."""
    from .remote import config_path as _cp  # noqa: PLC0415

    p = _cp(config_path)
    if not p.is_file():
        return {}
    try:
        r = tomllib.loads(p.read_text(encoding="utf-8")).get("remote") or {}
    except tomllib.TOMLDecodeError:
        return {}
    return r if r.get("type") == "s3" else {}


def env_lines(cred: dict) -> str:
    """`export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_SESSION_TOKEN=...` for a pod's bootstrap (stdout only, never a file)."""
    return "\n".join(f"export {k}={v}" for k, v in (("AWS_ACCESS_KEY_ID", cred["accessKeyId"]), ("AWS_SECRET_ACCESS_KEY", cred["secretAccessKey"]),
                                                    ("AWS_SESSION_TOKEN", cred["sessionToken"])))


def mint(*, via: str = "auto", scope: str = "object-read-write", ttl_s: int = 3600, prefixes: list[str] | None = None, objects: list[str] | None = None,
         actions: list[str] | None = None, endpoint: str | None = None, bucket: str | None = None, account_id: str | None = None,
         config_path: str | os.PathLike[str] | None = None, environ: dict[str, str] | None = None, opener: Any = None) -> dict:
    """Resolve inputs (flags > store.toml `[remote]` > env) and mint.  `via`: `local` (parent key in the env), `api` (CLOUDFLARE_API_TOKEN +
    account id), `auto` = local when the parent key is present, else api.  Returns the credential + {via, bucket, scope, endpoint, expires_utc}."""
    env = os.environ if environ is None else environ
    cfg = remote_config(config_path)
    endpoint = endpoint or cfg.get("endpoint") or env.get("R2_ENDPOINT")
    bucket = bucket or cfg.get("bucket") or env.get("R2_BUCKET")
    if not endpoint or not bucket:
        raise MintError("need the R2 endpoint and bucket: store.toml [remote] type s3, or --endpoint/--bucket, or R2_ENDPOINT/R2_BUCKET")
    account_id = account_id or env.get("R2_ACCOUNT_ID") or env.get("CLOUDFLARE_ACCOUNT_ID") or account_id_from_endpoint(endpoint)
    ak_env, sk_env = cfg.get("access_key_id_env") or "AWS_ACCESS_KEY_ID", cfg.get("secret_access_key_env") or "AWS_SECRET_ACCESS_KEY"
    ak, sk = env.get(ak_env), env.get(sk_env)
    if env.get(cfg.get("session_token_env") or "AWS_SESSION_TOKEN") and via != "api":
        raise MintError("the environment holds a session token: a temporary credential cannot mint another; source the parent key first")
    api_token = env.get("CLOUDFLARE_API_TOKEN")
    parent = f"the parent R2 key pair in {ak_env} ({'set' if ak else 'unset'}) / {sk_env} ({'set' if sk else 'unset'})"
    load = f"load it first, e.g. `(set -a; . {PARENT_ENV_FILE}; set +a; research data mint-credential --via local ...)` on the control machine"
    if via == "auto":
        if not (ak and sk) and not api_token:
            raise MintError(f"--via auto found no parent credential in this environment: local signing needs {parent}, the Cloudflare API needs "
                            f"CLOUDFLARE_API_TOKEN (unset).  Sign locally with the parent key -- {load} -- or set CLOUDFLARE_API_TOKEN for --via api")
        via = "local" if ak and sk else "api"
    if not account_id:
        raise MintError("need the Cloudflare account id: --account-id, R2_ACCOUNT_ID / CLOUDFLARE_ACCOUNT_ID, or an <account>.r2.cloudflarestorage.com endpoint")
    if via == "local":
        if not ak or not sk:
            raise MintError(f"--via local needs {parent}: {load}" + (" (or --via api: CLOUDFLARE_API_TOKEN is set)" if api_token else ""))
        cred = mint_local(endpoint, account_id, ak, sk, bucket, scope, actions=actions, ttl_s=ttl_s, prefixes=prefixes, objects=objects)
    elif via == "api":
        if actions:
            raise MintError("--actions is supported by local signing only (the Temporary Credentials API takes --permission / --prefix / --object)")
        if not api_token:
            raise MintError("--via api needs CLOUDFLARE_API_TOKEN (an API token with R2 write permission on the account), which is unset; "
                            + ("the parent key is loaded, so --via local works" if ak and sk else f"or sign locally (--via local) with {parent}: {load}"))
        if not ak:
            raise MintError(f"the API needs the parent access key id in {ak_env} (parentAccessKeyId), which is unset: {load.replace('--via local', '--via api')}")
        cred = mint_api(account_id, api_token, bucket, ak, scope, ttl_s=ttl_s, prefixes=prefixes, objects=objects, opener=opener)
    else:
        raise MintError(f"--via must be auto, local or api, got {via!r}")
    return {**cred, "via": via, "bucket": bucket, "scope": None if actions else scope, "endpoint": endpoint, "account_id": account_id, "ttl_s": ttl_s,
            "prefixes": list(prefixes or []), "objects": list(objects or []), "actions": list(actions or [])}
