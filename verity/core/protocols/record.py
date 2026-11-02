"""`Core.Record` in Python: a record entry, one message of a run's transcript, numbered, chained to the entry before it
and signed by the party that sent it (`verity/core/protocols/interface`, `CoreProtocol/Record.lean`).

    entry   = {body, kind, prev, schema, seq, session, sig}
    message = SHA-512([role] || I || [d])              d: the schema's tagged SHA-512 of the entry without its sig

each `[x]` a u64 big-endian length then `x`, and `I` the signer's 16-byte LMS identifier.  `body` is the message, an
object of the schema's form; `kind` its kind under the schema; `schema` the protocol's entry schema and version, which
names the tag `d` is under; `session` the session it belongs to, None for an entry of none.  Entry `i` of a record has
`seq = i` and `prev` the SHA-512 of entry `i - 1`'s line (`ZERO` for entry 0, `prev_of`), the line being the entry's
`line`: sorted keys, no whitespace, every character past ASCII escaped.

* `of_json`: `Record.ofJson`, the form: the seven keys, each of its type.  Everything a protocol accepts beyond the
  form (its schemas, kinds and bodies, which key signs, its sessions) is its own check; the auditor's ledger
  (`sampled_proofs.challenger`) is the first.
* `line`: an entry's line, as the auditor's ledger writes it (Lean `SampledProofs.PyJson.canonical`).
* `unsigned`, `message`, `prev_of`: `Record.unsigned`, `Record.message`, `Record.prevOf`.

Stdlib only: a verifier imports no `verity.ml` code.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

FIELDS = ("body", "kind", "prev", "schema", "seq", "session", "sig")
"""A record entry's keys, sorted."""
ZERO = "0" * 128
"""The `prev` of a record's first entry."""
FORM = ("a record entry is {body, kind, prev, schema, seq, session, sig}: an object, a string, a SHA-512 in lowercase "
        "hex, a string, a natural number, a string or null, and lowercase hex")
_PREV = re.compile(r"[0-9a-f]{128}")
_HEX = re.compile(r"(?:[0-9a-f]{2})*")


def of_json(e: Any) -> dict:
    """`e`, a parsed JSON value, when it is a record entry: its seven keys, each of its type; ValueError(`FORM`)
    otherwise."""
    if not (isinstance(e, dict) and sorted(e) == list(FIELDS) and isinstance(e["body"], dict)
            and type(e["kind"]) is str and type(e["prev"]) is str and _PREV.fullmatch(e["prev"]) is not None
            and type(e["schema"]) is str and type(e["seq"]) is int and e["seq"] >= 0
            and (e["session"] is None or type(e["session"]) is str)
            and type(e["sig"]) is str and _HEX.fullmatch(e["sig"]) is not None):
        raise ValueError(FORM)
    return e


def line(entry: dict) -> str:
    """An entry's line: sorted keys, no whitespace, every character past ASCII escaped."""
    return json.dumps(entry, sort_keys=True, separators=(",", ":"))


def unsigned(entry: dict) -> dict:
    """An entry without its `sig`: what its signature's digest is over."""
    return {k: v for k, v in entry.items() if k != "sig"}


def _field(b: bytes) -> bytes:
    return len(b).to_bytes(8, "big") + b


def message(role: str, I: bytes, d: bytes) -> bytes:
    """What an entry's `sig` signs under the protocol's `role`, the signer's identifier `I` and `d`, the digest of the
    entry without its sig under the tag its schema names."""
    return hashlib.sha512(_field(role.encode()) + I + _field(d)).digest()


def prev_of(line: str | bytes) -> str:
    """The `prev` of the entry after the one whose line is `line`."""
    return hashlib.sha512(line.encode() if isinstance(line, str) else line).hexdigest()


__all__ = ["FIELDS", "FORM", "ZERO", "line", "message", "of_json", "prev_of", "unsigned"]
