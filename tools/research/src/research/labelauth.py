"""Authenticated ready marks and land verdicts (drive.toml's `signed_verdicts`)."""
from __future__ import annotations

import base64
import binascii
import calendar
import fnmatch
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable

from .store import canon, labels_sync, vocab
from .store.model import Label, utc_now

ISSUER = "https://api.cursor.com"
JWKS_URL = "https://api.cursor.com/keys"
AUDIENCE = "urn:verity:label-auth"
LABEL_TAG = b"verity/label-auth/v1\0"
DELEGATION_TAG = b"verity/label-delegation/v1\0"
DELEGATION_TARGET_TAG = b"verity/label-delegation-target/v1\0"
DELEGATION_SCHEMA = "verity/label-delegation/v1"
SKEW_S = 300
DELEGATION_MAX_DAYS = 14
LANDING_VALUES = {vocab.READY: "true", vocab.REVIEWED: "land"}
_SHA256_INFO = bytes.fromhex("3031300d060960864801650304020105000420")
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class AuthError(Exception):
    """Why a label isn't authenticated, or can't be signed."""


def _who(name: str) -> str:
    return str(name).strip().lstrip("@").lower()


def _nonce(tag: bytes, doc: dict) -> str:
    return hashlib.sha512(tag + canon.dumps(doc)).hexdigest()


def label_nonce(label: Label) -> str:
    return _nonce(LABEL_TAG, label.to_doc())


def landing(label: Label) -> bool:
    want = LANDING_VALUES.get(label.key)
    return want is not None and str(label.value).lower() == want


def delegation_target(agent: str) -> str:
    return "task:" + hashlib.sha512(DELEGATION_TARGET_TAG + agent.encode()).hexdigest()


def epoch(ts: str) -> int:
    try:
        return calendar.timegm(time.strptime(ts, TS_FORMAT))
    except (TypeError, ValueError):
        raise AuthError(f"{ts!r} isn't a time as YYYY-MM-DDTHH:MM:SSZ") from None


def _b64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def unverified_claims(token: str) -> dict:
    try:
        body = json.loads(_b64(token.split(".")[1]))
    except (IndexError, ValueError, binascii.Error) as e:
        raise AuthError(f"the token isn't a JWT ({e})") from None
    if not isinstance(body, dict):
        raise AuthError("the token's claims aren't an object")
    return body


def jwks(doc: object) -> dict[str, tuple[int, int]]:
    keys = doc.get("keys") if isinstance(doc, dict) else None
    out = {}
    for k in keys if isinstance(keys, list) else ():
        if isinstance(k, dict) and k.get("kty") == "RSA" and isinstance(k.get("kid"), str) and isinstance(k.get("n"), str) \
                and isinstance(k.get("e"), str) and k.get("use", "sig") == "sig":
            try:
                out[k["kid"]] = (int.from_bytes(_b64(k["n"]), "big"), int.from_bytes(_b64(k["e"]), "big"))
            except (ValueError, binascii.Error) as e:
                raise AuthError(f"the issuer's key {k['kid']!r} isn't base64url ({e})") from None
    return out


def claims(token: object, keys: dict[str, tuple[int, int]]) -> dict:
    if not isinstance(token, str) or token.count(".") != 2:
        raise AuthError("no token, or not a JWT")
    h64, p64, s64 = token.split(".")
    try:
        head, body, sig = json.loads(_b64(h64)), json.loads(_b64(p64)), _b64(s64)
    except (ValueError, binascii.Error) as e:
        raise AuthError(f"the token isn't a JWT ({e})") from None
    if not (isinstance(head, dict) and isinstance(body, dict)):
        raise AuthError("the token's header or claims aren't an object")
    if head.get("alg") != "RS256":
        raise AuthError(f"the token is signed with {head.get('alg')!r}, not RS256")
    if (key := keys.get(head.get("kid"))) is None:
        raise AuthError(f"the token is signed with key {head.get('kid')!r}, not one of the issuer's ({', '.join(sorted(keys)) or 'none'})")
    n, e = key
    size = (n.bit_length() + 7) // 8
    digest = hashlib.sha256(f"{h64}.{p64}".encode()).digest()
    block = b"\x00\x01" + b"\xff" * (size - 3 - len(_SHA256_INFO) - len(digest)) + b"\x00" + _SHA256_INFO + digest
    s = int.from_bytes(sig, "big")
    if len(sig) != size or s >= n or pow(s, e, n).to_bytes(size, "big") != block:
        raise AuthError("the token's signature doesn't verify under the issuer's key")
    if body.get("iss") != ISSUER:
        raise AuthError(f"the token's issuer is {body.get('iss')!r}, not {ISSUER}")
    aud = body.get("aud")
    if aud != AUDIENCE and not (isinstance(aud, list) and AUDIENCE in aud):
        raise AuthError(f"the token's audience is {aud!r}, not {AUDIENCE}")
    if type(body.get("iat")) is not int:
        raise AuthError("the token has no `iat`")
    return body


def delegation_statement(lane: str, agent: str, until: str) -> dict:
    return {"schema": DELEGATION_SCHEMA, "lane": lane, "agent": agent, "until": until}


def delegation(lane: str, agent: str, until: str, mint: Callable[[str], str]) -> dict:
    st = delegation_statement(lane, agent, until)
    return {**st, "token": mint(_nonce(DELEGATION_TAG, st))}


def speaks_for(handle: str | None, name: str, staffs: dict[str, tuple[str, ...]]) -> bool:
    return bool(handle) and (handle == name or any(fnmatch.fnmatchcase(name, p) for p in staffs.get(handle, ())))


class Verifier:
    def __init__(self, keys: dict[str, tuple[int, int]], holders: dict[str, str],
                 staffs: dict[str, tuple[str, ...]] | None = None):
        self.keys, self.holders = keys, {a: _who(h) for a, h in holders.items()}
        self.staffs = {_who(h): tuple(globs) for h, globs in (staffs or {}).items()}
        self._seen: dict[tuple, str | None] = {}

    def why(self, label: Label, beside: Iterable[Label]) -> str | None:
        """None when a `label_auth` label of ``beside`` (the target's) authenticates ``label``, else why none does."""
        try:
            want = label_nonce(label)
        except ValueError as e:
            return f"its document isn't canonical JSON ({e})"
        tokens = [b.value for b in beside if b.key == vocab.LABEL_AUTH and b.target == label.target
                  and isinstance(b.value, dict) and b.value.get("for") == want]
        if not tokens:
            return "unsigned"
        key = (want, tuple(json.dumps(v, sort_keys=True) for v in tokens))
        if key not in self._seen:
            whys: list[str] | None = []
            for value in tokens:
                try:
                    self._check(label, want, value)
                except AuthError as e:
                    whys.append(str(e))
                else:
                    whys = None
                    break
            self._seen[key] = None if whys is None else "; ".join(whys)
        return self._seen[key]

    def _check(self, label: Label, want: str, value: dict) -> None:
        cl = claims(value.get("token"), self.keys)
        if cl.get("nonce") != want:
            raise AuthError("its token's nonce isn't this label's")
        t = epoch(label.ts)
        if abs(cl["iat"] - t) > SKEW_S:
            raise AuthError(f"its token was minted {cl['iat'] - t:+d} s from the label's time, more than {SKEW_S} s")
        agent, lane = cl.get("cloud_agent_id"), _who(label.by)
        if isinstance(agent, str) and self.holders.get(agent) == lane:
            return
        if not isinstance(d := value.get("delegation"), dict):
            raise AuthError(f"its token is {agent!r}'s, which doesn't hold {lane} in main's registry, and it carries no delegation")
        self._delegated(d, lane, agent, t)

    def _delegated(self, d: dict, lane: str, agent: object, t: int) -> None:
        if d.get("schema") != DELEGATION_SCHEMA or not all(isinstance(d.get(k), str) for k in ("lane", "agent", "until")):
            raise AuthError(f"its delegation isn't a {DELEGATION_SCHEMA} statement")
        st = delegation_statement(d["lane"], d["agent"], d["until"])
        if _who(st["lane"]) != lane or st["agent"] != agent:
            raise AuthError(f"its delegation is of {st['lane']!r} to {st['agent']!r}, not of {lane} to its token's {agent!r}")
        cl = claims(d.get("token"), self.keys)
        if cl.get("nonce") != _nonce(DELEGATION_TAG, st):
            raise AuthError("its delegation's token isn't over the delegation")
        if not isinstance(by := cl.get("cloud_agent_id"), str) or not speaks_for(self.holders.get(by), lane, self.staffs):
            raise AuthError(f"its delegation was minted by {by!r}, which doesn't hold {lane} in main's registry, nor a handle "
                            "that staffs it")
        if by == st["agent"]:
            raise AuthError(f"its delegation is to the agent that minted it, {by!r}, and an agent's delegation to itself never "
                            "counts")
        until = epoch(st["until"])
        if until - cl["iat"] > DELEGATION_MAX_DAYS * 86400 + SKEW_S:
            raise AuthError(f"its delegation runs past {DELEGATION_MAX_DAYS} days")
        if not cl["iat"] - SKEW_S <= t <= until:
            raise AuthError(f"the label's time is outside its delegation (minted at {cl['iat']}, until {st['until']})")


class Signer:
    def __init__(self, store, mint: Callable[[str], str], required: bool = False, holders: dict[str, str] | None = None,
                 now: Callable[[], str] = utc_now, log: Callable[[str], None] = print):
        self.store, self.mint, self.required, self.now, self.log = store, mint, required, now, log
        self.holders = None if holders is None else {a: _who(h) for a, h in holders.items()}

    def __getattr__(self, name: str):
        return getattr(self.store, name)

    def add_label(self, target: str, key: str, value: object, by: str, ref: str | None = None, ts: str | None = None) -> Path:
        label = Label(target=target, key=key, value=value, by=by, ts=ts or self.now(), ref=ref)
        auth = None
        if landing(label):
            try:
                auth = self.sign(label)
            except AuthError as e:
                if self.required:
                    raise AuthError(f"drive.toml's `signed_verdicts` is on at `main`, and it can't be signed: {e}") from None
                self.log(f"{key} = {value} on {target} is written unsigned ({e}); drive counts it until drive.toml's "
                         "`signed_verdicts` is on")
        path = self.store.add_label(target, key, value, by, ref=ref, ts=label.ts)
        if auth is not None:
            self.store.add_label(target, vocab.LABEL_AUTH, auth, by, ts=label.ts)
        return path

    def sign(self, label: Label) -> dict:
        want = label_nonce(label)
        token = self.mint(want)
        agent = unverified_claims(token).get("cloud_agent_id")
        auth = {"for": want, "token": token}
        if isinstance(agent, str) and (self.holders or {}).get(agent) != _who(label.by):
            if d := self.delegation(agent, label.by, label.ts):
                auth["delegation"] = d
            elif self.holders is not None:
                raise AuthError(f"this agent ({agent}) doesn't hold {label.by} in main's registry and nothing delegates "
                                f"{label.by} to it: the holder of {label.by}, or of the handle that staffs it, runs "
                                f"`research queue delegate {agent} --by {label.by}`")
        return auth

    def delegation(self, agent: str, lane: str, ts: str) -> dict | None:
        target = delegation_target(agent)
        labels_sync.read_through(self.store, [target], self.log)
        for lab in reversed(self.store.labels(target, key=vocab.LABEL_DELEGATION)):
            v = lab.value
            if isinstance(v, dict) and _who(v.get("lane", "")) == _who(lane) and v.get("agent") == agent \
                    and isinstance(v.get("until"), str) and v["until"] >= ts:
                return v
        return None


def cursor_mint(nonce: str) -> str:
    from .slack import SlackError, oidc_token

    try:
        return oidc_token(AUDIENCE, nonce=nonce)
    except SlackError as e:
        raise AuthError(str(e)) from None


def issuer_keys(cache: Path, get: Callable[[str], bytes | None] | None = None,
                log: Callable[[str], None] = print) -> dict[str, tuple[int, int]]:
    from .slack import SlackError, http_get

    try:
        body = (get or http_get)(JWKS_URL)
        if not body:
            raise AuthError(f"{JWKS_URL} has no keys")
        doc = json.loads(body)
        if not (keys := jwks(doc)):
            raise AuthError(f"{JWKS_URL} lists no RSA signing key")
    except (AuthError, SlackError, ValueError) as e:
        try:
            kept = jwks(json.loads(cache.read_text())) if cache.is_file() else {}
        except (OSError, ValueError) as bad:
            raise AuthError(f"the issuer's keys can't be read ({e}), and those kept at {cache} can't either ({bad})") from None
        if not kept:
            raise AuthError(f"the issuer's keys can't be read ({e}), and none are kept at {cache}") from None
        log(f"the issuer's keys can't be read ({e}); using the {len(kept)} kept at {cache}")
        return kept
    cache.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{cache.name}.", dir=cache.parent)
    with os.fdopen(fd, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, cache)
    return keys
