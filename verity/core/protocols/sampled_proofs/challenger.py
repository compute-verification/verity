"""The proof service's challenger: its live coins, the one live-coin source of every protocol (Lean of record:
`SampledProofs.Challenger`, `verity/core/service/sampled_proofs`, which `tests/test_difftest.py` there holds this to).

Coins are indexed by label and index, and named by no protocol.  A layout is a scheme, an epoch (the client's session the
coins are for) and its families ``(label, count)`` in order: coin ``(label, i)`` sits at its family's offset plus i, at
most `MAX_COINS` coins.  Before the epoch the challenger draws every coin and its hm96 salt from its own generator and
publishes the coin commitment:

    x_p     = SHA-512(scheme || 0x00 || coin_p)                                     coin_p: 64 fresh bytes (`COIN_BYTES`)
    value_p = hm96-sha512 tree_leaf(key, commit_string(key, x_p, salt_p))           salt_p: 192 fresh bytes, key pinned
    root    = frame-v3-sha512 tree over value_0 … value_(K−1) at positions 0 … K−1, leaf schema the scheme, domain
              binding identity_digest_sha512("verity/challenger/coins/v1", {epoch, families, scheme}), owner -1

The hm96 leaves keep every unopened coin statistically hidden; binding rests on SHA-512's collision resistance.  The
challenger opens coin ``(label, i)`` once, after the verifier recorded the root the coin follows (`LiveOpening.after`),
and never twice: a retry never redraws.  There is no beacon and no Fiat–Shamir: a coin is the challenger's own bytes, and
a client derives what it draws from the opened coin (`verity.ml.randomness.derive`).  The coins are live and the verifier
designated: an outcome convinces the auditor, who trusts its own challenger, and no third party.

Timing is a trust assumption, not a check.  `after` binds an opening to the root the verifier recorded, but can't show
the coin was released after it: that rests on the challenger releasing coin i only once it has received root i.  The
challenger is auditor-trusted, the auditor's device on a machine the prover doesn't control, under the named assumption
`trusted-nebius-host`.  The record keeps the release order with its own time (`LiveRecord.log`).

`live_reason` is the verifier's check of one opening and `order_reason` of a family's openings against the roots it
recorded, in order: coin i opened after root i, each once, none before its root is recorded, and every recorded root's
coin opened.  `LiveRecord` is the verifier's side of one family: it records a root and only then asks for that root's
coin, and logs both on its clock.

**The challenger's record** is the verifier's trusted coin commitment.  Before any coin opens, the challenger signs its
commitment with its LMS key over SHA-512 (`LiveChallenger.sign_record`) and writes the record on its own host
(`write_record`):

    record  = {commitment, host, signature}                       host: the auditor machine the challenger runs on
    message = SHA-512([RECORD_ROLE] || I || [[host]] || [identity_digest_sha512(COMMITMENT_TAG, commitment)])

each [x] a u64 big-endian length then x, and I the key's identifier (its public key's first 16 bytes): the LMS role
digest's form (`verity.ml.crypto.lms`).  The verifier reads the record from the challenger's file, never from the prover's
bundle, and checks it under its own challenger's key (`ChallengerKey`: the LMS parameters and public key):
`trusted_commitment` returns the commitment, or raises naming why the record is missing, malformed or unsigned, so a client
that trusts no other commitment fails closed.  `record_reason` is that check and `trusted_reason` adds that a commitment the
prover presents is the record's.  This module verifies no signature itself (a verifier imports no `verity.ml` code beyond
its frozen edges): the caller hands in `verify`, LMS at the key's parameters, and the Lean of record verifies on its own
(`Lms.verify`).

**The session's close** is the only evidence that each coin followed its root: an opening's `after` is the client's to
write, and a coin opened before its root still opens against the commitment.  The challenger logs each root it receives
and each release after it, a line each in canonical JSON (`remote`'s `releases.jsonl`, its head the commitment, its times
integer nanoseconds on the challenger's monotonic clock), and when its session ends,
however it ends, it signs the log at its key's next leaf and writes the close record (`remote`'s `challenger.json`):

    close   = {chain, ended, lines, session, signature}       ended: null when the client closed the session, else why not
    chain   = c_n,  c_0 = SHA-512([LOG_TAG]),  c_(i+1) = SHA-512(c_i || [line_i])        line_i: the log's i-th line, UTF-8
    message = SHA-512([CLOSE_ROLE] || I || [record message] || [session] || [u64be(n)] || [chain] || [end])

`end` being 0x00 when the client closed the session and 0x01 || ended otherwise, and the record message the record's
own (`record_message`), which binds the close to the commitment and host.  `session_of` is the verifier's read of a
session: the record under its key, the close signed under it, the log the one the close signs, line for line, and a
session its client closed; then the roots it received (`released_roots_of`).  A session that ends unclosed is no window's.

**The auditor's ledger** names the one challenger session each window's coins come from (`audited_session`).  The
auditor (infra's launcher on the challenger's host) appends a signed entry, under the challenger's key, when it opens a
session for a window, before its first coin, and another when the session ends; a window gets one session:

    entry   = {body, kind, prev, schema, seq, session, sig}        kind: open or end; seq from 0; prev: SHA-512 of the
                                                                   previous line (128 zeros first), schema LEDGER_SCHEMA
    open    = body {window: {commit, period}, host, pinned, key_sha512, ...}    host: the one the session's record signs
    end     = body {closed, files, ...}                            files: each file of the session's SHA-512
    message = SHA-512([LEDGER_ROLE] || I || [identity_digest_sha512(LEDGER_TAG, entry without sig)])

each line canonical JSON.  The window is `{commit, period}`: the served Commit's root and the period its lane names.  The
verifier pins the challenger key from outside the ledger (the window's deployment), refuses any open naming another
key's SHA-512 before it checks a signature, so a ledger signed under its own key can't pin itself, then checks each
entry's chain and signature, one open a window and session and one end a session after its open.  The window's session
must have its end, closed, and its record must name the open's host, its signed close the ledger's session id and its
release log the one the end sealed (`trusted_session_of`).  A ledger older than the window's end is refused, as is a
window it opens no session for.  So two sessions under one key can't both pass for one window, and one session's coins
pass for one window only.  `ledger_line` writes an entry as the auditor does.

The first client is PoUW's live window (`verity.core.protocols.pouw.window`, "Live windows"): its salt, its draw and
its round coins, round r's opened after round r's root.  The module reads only the standard library and
`verity.ml.commitments`, so any client calls it as it is.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from verity.ml.commitments import (Commitment, CommitmentDomain, MerkleTree, Opening, RangeIndexedDomain,
                                   VerificationLimits, verify_opening)
from verity.ml.commitments import hm96
from verity.ml.commitments.identity import canonical_json_text, identity_digest_sha512

BINDING = "verity/challenger/coins/v1"
COIN_BYTES = 64
HM96 = hm96.SHA512
KEY = HM96.default_key
#: the most coins one commitment holds
MAX_COINS = 1 << 20
_NAME = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-/")
_HEX = re.compile(r"(?:[0-9a-f]{2})*")
COMMITMENT_FORM = ("a live coin commitment is {epoch, families, root, scheme}: families a list of [label, count], root "
                   "lowercase hex")
OPENING_FORM = ("a live coin opening is {after, coin, epoch, index, label, path, salt}: integers, a string, lowercase hex "
                "and a path of SHA-512 digests")
RECORD_ROLE = "verity/challenger-record/v1"
COMMITMENT_TAG = "verity/challenger/commitment/v1"
#: an LMS public key over SHA-512: I (16 bytes) || T[1] (64)
PUBLIC_BYTES = 80
_HOST = frozenset(string.ascii_letters + string.digits + "-._")
RECORD_FORM = ("a challenger record is {commitment, host, signature}: a live coin commitment, a host of ASCII letters, "
               "digits, '-', '.' and '_', and lowercase hex")
KEY_FORM = ("a challenger key is {H, W, public}: LMS over SHA-512 (N = 64) with W in 1, 2, 4, 8 and 1 <= H <= 30, and its "
            "80-byte public key in lowercase hex")
NO_RECORD = "no challenger record: the verifier holds no coin commitment its challenger recorded and signed"
CLOSE_ROLE = "verity/challenger-close/v1"
LOG_TAG = "verity/challenger/release-log/v1"
MAX_SESSION_ID = 128
#: a release log's most lines: its head, then a root and its release per coin
MAX_LOG_LINES = 2 * MAX_COINS + 1
#: a release log line's most characters
MAX_LOG_LINE = 4096
CLOSE_FORM = ("a challenger close record is {chain, ended, lines, session, signature}: a 64-byte chain and a signature in "
              "lowercase hex, how the session ended (null when its client closed it), the release log's line count, "
              f"1 to {MAX_LOG_LINES}, and a session id of at most {MAX_SESSION_ID} ASCII letters, digits, '-', '.' and '_'")
NO_CLOSE = ("no challenger close record: a session the challenger didn't close and sign shows nothing of the order it "
            "released its coins in")
LOG_FORM = "the challenger's release log is a list of its lines"
LOG_MALFORMED = "the challenger's release log is malformed"
SESSION_FORM = "a challenger session is {close, log, record}: its close record, its release log's lines and its record"
LOG_FILE = "releases.jsonl"
LEDGER_SCHEMA = "verity/challenger-auditor/v1"
LEDGER_ROLE = "verity/challenger-auditor-record/v1"
LEDGER_TAG = "verity/challenger-auditor/entry/v1"
#: an auditor's ledger line's most characters: its signature at W = 1, H = 30 is 70928 hex digits
MAX_LEDGER_LINE = 1 << 18
_ZERO = "0" * 128
_PERIOD = frozenset(string.ascii_letters + string.digits + "._:-")
WINDOW_FORM = ("a window's identity is {commit, period}: the served Commit's root, 64 bytes in lowercase hex, and the "
               "period, 1 to 64 ASCII letters, digits, '.', '_', ':' and '-', the first a letter or digit")
LEDGER_FORM = ("an auditor's ledger line is canonical JSON {body, kind, prev, schema, seq, session, sig}: kind open, its "
               "body naming the window, the challenger's host, the pinned source commit (40 lowercase hex) and the SHA-512 "
               "of the challenger key's public key (key_sha512), or kind end, its body whether its client closed the "
               "session (closed) and the SHA-512 of each file the session left (files)")
NO_AUDIT = "no auditor's ledger: the verifier trusts a challenger session only as the auditor's signed ledger names it"

class CoinError(ValueError):
    """A layout, a coin commitment or an opening the challenger refuses, or a coin its commitment doesn't hold."""


def _name(s: Any) -> bool:
    """A scheme or a label: nonempty, of lowercase ASCII letters, digits, '-' and '/'."""
    return type(s) is str and s != "" and all(c in _NAME for c in s)


def layout_problem(scheme: Any, epoch: Any, families: Any) -> str | None:
    """Why `scheme`, `epoch` and `families` are not a live coin layout; None when they are."""
    if not _name(scheme):
        return "a live coin scheme is a nonempty name of lowercase letters, digits, '-' and '/'"
    if type(epoch) is not int or epoch < 0:
        return "an epoch is a nonnegative integer"
    if type(families) is not tuple or not families or not all(
            type(f) is tuple and len(f) == 2 and _name(f[0]) and type(f[1]) is int and f[1] >= 1 for f in families):
        return "a live coin layout is a nonempty sequence of (label, count), a name and a positive integer"
    if len({label for label, _ in families}) != len(families):
        return "a live coin layout's labels are distinct"
    if sum(n for _, n in families) > MAX_COINS:
        return f"a live coin layout holds at most {MAX_COINS} coins"
    return None


def live_domain(scheme: str, epoch: int, families: Sequence[tuple[str, int]]) -> CommitmentDomain:
    """The layout's coin tree: positions 0 … K−1 under the layout's binding, owner -1, frame-v3-sha512."""
    binding = identity_digest_sha512(BINDING, {"epoch": epoch, "families": [[label, n] for label, n in families],
                                               "scheme": scheme})
    return CommitmentDomain(binding, -1, RangeIndexedDomain(sum(n for _, n in families)), hash="sha512")


def live_leaf(scheme: str, coin: bytes, salt: bytes) -> bytes:
    """The committed value of one coin: the hm96-sha512 leaf over SHA-512(scheme ‖ 0x00 ‖ coin) under `salt`."""
    return HM96.tree_leaf(KEY, HM96.commit_string(KEY, hashlib.sha512(scheme.encode() + b"\0" + coin).digest(), salt))


def _hex(s: Any) -> bytes | None:
    return bytes.fromhex(s) if type(s) is str and _HEX.fullmatch(s) else None


@dataclass(frozen=True, slots=True)
class LiveCommitment:
    """What the challenger publishes before epoch `epoch`: its coins' layout under `scheme`, families `(label, count)` in
    order, and the 64-byte root over them."""

    scheme: str
    epoch: int
    families: tuple[tuple[str, int], ...]
    root: bytes

    def __post_init__(self) -> None:
        why = layout_problem(self.scheme, self.epoch, self.families)
        if why is not None:
            raise CoinError(why)
        if type(self.root) is not bytes or len(self.root) != 64:
            raise CoinError("a coin commitment's root is 64 bytes")

    @property
    def count(self) -> int:
        return sum(n for _, n in self.families)

    def family(self, label: str) -> int | None:
        """How many coins family `label` holds, or None when the layout has no such family."""
        return next((n for name, n in self.families if name == label), None)

    def position(self, label: str, index: int) -> int | None:
        """Coin (`label`, `index`)'s position in the tree, or None when the layout has no such coin."""
        offset = 0
        for name, n in self.families:
            if name == label:
                return offset + index if type(index) is int and 0 <= index < n else None
            offset += n
        return None

    def to_json(self) -> dict:
        return {"epoch": self.epoch, "families": [[label, n] for label, n in self.families], "root": self.root.hex(),
                "scheme": self.scheme}

    @classmethod
    def from_json(cls, doc: Any) -> LiveCommitment:
        if not isinstance(doc, dict) or sorted(doc) != ["epoch", "families", "root", "scheme"]:
            raise CoinError(COMMITMENT_FORM)
        families, root = doc["families"], _hex(doc["root"])
        if type(families) is not list or not all(type(f) is list for f in families) or root is None:
            raise CoinError(COMMITMENT_FORM)
        return cls(doc["scheme"], doc["epoch"], tuple(tuple(f) for f in families), root)


@dataclass(frozen=True, slots=True)
class LiveOpening:
    """Coin (`label`, `index`) of epoch `epoch`, opened by the challenger after the verifier recorded `after`: the coin, its
    hm96 salt and its path."""

    epoch: int
    label: str
    index: int
    after: bytes
    coin: bytes
    salt: bytes
    path: tuple[bytes, ...]

    def to_json(self) -> dict:
        return {"after": self.after.hex(), "coin": self.coin.hex(), "epoch": self.epoch, "index": self.index,
                "label": self.label, "path": [p.hex() for p in self.path], "salt": self.salt.hex()}

    @classmethod
    def from_json(cls, doc: Any) -> LiveOpening:
        if not isinstance(doc, dict) or sorted(doc) != ["after", "coin", "epoch", "index", "label", "path", "salt"]:
            raise CoinError(OPENING_FORM)
        hexes = [_hex(doc[k]) for k in ("after", "coin", "salt")]
        path = [_hex(p) for p in doc["path"]] if type(doc["path"]) is list else None
        if (type(doc["epoch"]) is not int or type(doc["index"]) is not int or type(doc["label"]) is not str
                or None in hexes or path is None or any(p is None or len(p) != 64 for p in path)):
            raise CoinError(OPENING_FORM)
        return cls(doc["epoch"], doc["label"], doc["index"], *hexes, tuple(path))


def live_reason(commitment: LiveCommitment, opening: LiveOpening, label: str, index: int, after: bytes) -> str | None:
    """Why `opening` is not coin (`label`, `index`) of `commitment`, opened after `after`; None when it is."""
    c, o = commitment, opening
    position = c.position(label, index)
    if position is None:
        raise CoinError(f"the epoch's coins have no {label} coin {index}")
    if o.epoch != c.epoch:
        return f"coin is epoch {o.epoch}'s, not epoch {c.epoch}'s"
    if (o.label, o.index) != (label, index):
        return f"coin is the epoch's {o.label} coin {o.index}, not its {label} coin {index}"
    if len(o.coin) != COIN_BYTES or len(o.salt) != HM96.salt_bytes:
        return "coin opening is malformed"
    if not verify_opening(live_domain(c.scheme, c.epoch, c.families), Commitment(c.root, c.count),
                          Opening(position, live_leaf(c.scheme, o.coin, o.salt), o.path), c.scheme, VerificationLimits()):
        return "coin does not open against the epoch's coin commitment"
    if o.after != after:
        return "coin was not opened after what the verifier recorded"
    return None


def order_reason(commitment: LiveCommitment, label: str, roots: Sequence[bytes], openings: Sequence[LiveOpening]
                 ) -> str | None:
    """Why `openings` are not family `label`'s coins in order, coin i opened once after `roots[i]`, the i-th root the
    verifier recorded, with every recorded root's coin opened; None when they are."""
    count = commitment.family(label)
    if count is None:
        raise CoinError(f"the epoch's coins have no {label} family")
    if len(roots) > count:
        return f"{len(roots)} roots recorded, more than the epoch's {count} {label} coins"
    seen: set[tuple[str, int]] = set()
    for i, o in enumerate(openings):
        if (o.label, o.index) in seen:
            return f"the epoch's {o.label} coin {o.index} is opened twice: a retry never redraws"
        seen.add((o.label, o.index))
        if i >= len(roots):
            return f"{label} coin {i} was opened before its root was recorded"
        why = live_reason(commitment, o, label, i, roots[i])
        if why is not None:
            return f"{label} coin {i}: {why}"
    if len(openings) < len(roots):
        return f"{label} coin {len(openings)} was never opened"
    return None


def _host(s: Any) -> bool:
    return type(s) is str and s != "" and all(c in _HOST for c in s)


def _field(b: bytes) -> bytes:
    return len(b).to_bytes(8, "big") + b


def record_message(I: bytes, host: str, commitment: LiveCommitment) -> bytes:
    """What a challenger's record signs (module docstring), under the key whose identifier is `I`."""
    if len(I) != 16:
        raise CoinError(f"an LMS identifier is 16 bytes, not {len(I)}")
    digest = identity_digest_sha512(COMMITMENT_TAG, commitment.to_json())
    return hashlib.sha512(_field(RECORD_ROLE.encode()) + I + _field(_field(host.encode())) + _field(digest)).digest()


@dataclass(frozen=True, slots=True)
class ChallengerKey:
    """The verifier's key for its own challenger: LMS over SHA-512 at (64, `W`, `H`), and the public key `public`."""

    W: int
    H: int
    public: bytes

    def __post_init__(self) -> None:
        if (type(self.W) is not int or self.W not in (1, 2, 4, 8) or type(self.H) is not int or not 1 <= self.H <= 30
                or type(self.public) is not bytes or len(self.public) != PUBLIC_BYTES):
            raise CoinError(KEY_FORM)

    def to_json(self) -> dict:
        return {"H": self.H, "W": self.W, "public": self.public.hex()}

    @classmethod
    def from_json(cls, doc: Any) -> ChallengerKey:
        if not isinstance(doc, dict) or sorted(doc) != ["H", "W", "public"] or _hex(doc["public"]) is None:
            raise CoinError(KEY_FORM)
        return cls(doc["W"], doc["H"], _hex(doc["public"]))


@dataclass(frozen=True, slots=True)
class ChallengerRecord:
    """The challenger's record of an epoch's coin commitment, signed on host `host` before any coin opened."""

    commitment: LiveCommitment
    host: str
    signature: bytes

    def to_json(self) -> dict:
        return {"commitment": self.commitment.to_json(), "host": self.host, "signature": self.signature.hex()}

    @classmethod
    def from_json(cls, doc: Any) -> ChallengerRecord:
        if not isinstance(doc, dict) or sorted(doc) != ["commitment", "host", "signature"]:
            raise CoinError(RECORD_FORM)
        signature = _hex(doc["signature"])
        if not _host(doc["host"]) or signature is None:
            raise CoinError(RECORD_FORM)
        return cls(LiveCommitment.from_json(doc["commitment"]), doc["host"], signature)


#: `verify(key, message, signature)`: whether `signature` signs `message` under LMS at `key`'s parameters
#: (`verity.ml.crypto.lms.verify` over `Params(64, key.W, key.H)`), false for a signature that doesn't decode.
Verify = Callable[[ChallengerKey, bytes, bytes], bool]


def record_reason(key: ChallengerKey, record: Any, verify: Verify) -> str | None:
    """Why `record` (its JSON, None when the verifier has none) is not a challenger record signed under `key`; None when
    it is."""
    if record is None:
        return NO_RECORD
    try:
        r = ChallengerRecord.from_json(record)
    except CoinError as e:
        return f"the challenger record is malformed: {e}"
    try:
        ok = verify(key, record_message(key.public[:16], r.host, r.commitment), r.signature) is True
    except ValueError:
        ok = False
    if not ok:
        return "the challenger record's signature does not verify under the verifier's challenger key"
    return None


def trusted_reason(key: ChallengerKey, record: Any, commitment: LiveCommitment, verify: Verify) -> str | None:
    """Why `commitment`, a coin commitment the prover presents, is not the one in the challenger's record (`record_reason`
    under `key`); None when it is."""
    why = record_reason(key, record, verify)
    if why is not None:
        return why
    trusted = ChallengerRecord.from_json(record).commitment
    if commitment != trusted:
        return (f"coin commitment {commitment.root.hex()[:16]} is not the one the verifier's challenger recorded "
                f"({trusted.root.hex()[:16]})")
    return None


def write_record(path: str | os.PathLike, record: ChallengerRecord) -> None:
    """Write the challenger's record to `path` on its own host, once: a record that exists is refused."""
    with open(path, "x", encoding="utf-8") as f:
        f.write(json.dumps(record.to_json(), sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def trusted_commitment(path: str | os.PathLike, key: ChallengerKey, verify: Verify) -> LiveCommitment:
    """The verifier's trusted coin commitment: the record at `path`, the file the challenger wrote on its own host (never
    the prover's bundle), checked under the verifier's challenger key (`record_reason`).  A missing, malformed or unsigned
    record raises `CoinError` naming why."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        doc = None
    except (OSError, ValueError) as e:
        raise CoinError(f"the challenger record at {path} is unreadable: {e}") from e
    why = record_reason(key, doc, verify)
    if why is not None:
        raise CoinError(why if doc is not None else f"{why} (none at {path})")
    return ChallengerRecord.from_json(doc).commitment


def _session_id(s: Any) -> bool:
    return type(s) is str and 0 < len(s) <= MAX_SESSION_ID and all(c in _HOST for c in s)


def log_chain(lines: Sequence[str]) -> bytes:
    """The hash chain over the release log's lines (module docstring, "The session's close")."""
    h = hashlib.sha512(_field(LOG_TAG.encode())).digest()
    for line in lines:
        h = hashlib.sha512(h + _field(line.encode())).digest()
    return h


def close_message(I: bytes, host: str, commitment: LiveCommitment, session: str, lines: int, chain: bytes,
                  ended: str | None) -> bytes:
    """What the challenger's close record signs (module docstring, "The session's close"), under the key whose
    identifier is `I`."""
    end = b"\0" if ended is None else b"\1" + ended.encode()
    return hashlib.sha512(_field(CLOSE_ROLE.encode()) + I + _field(record_message(I, host, commitment))
                          + _field(session.encode()) + _field(lines.to_bytes(8, "big")) + _field(chain)
                          + _field(end)).digest()


@dataclass(frozen=True, slots=True)
class ChallengerClose:
    """The challenger's signed close of session `session`: its release log's `lines` lines and their chain `chain`, and
    how the session ended, None when its client closed it."""

    session: str
    lines: int
    chain: bytes
    ended: str | None
    signature: bytes

    def to_json(self) -> dict:
        return {"chain": self.chain.hex(), "ended": self.ended, "lines": self.lines, "session": self.session,
                "signature": self.signature.hex()}

    @classmethod
    def from_json(cls, doc: Any) -> ChallengerClose:
        if not isinstance(doc, dict) or sorted(doc) != ["chain", "ended", "lines", "session", "signature"]:
            raise CoinError(CLOSE_FORM)
        chain, signature, lines, ended = _hex(doc["chain"]), _hex(doc["signature"]), doc["lines"], doc["ended"]
        if (chain is None or len(chain) != 64 or signature is None or type(lines) is not int
                or not 1 <= lines <= MAX_LOG_LINES or not (ended is None or type(ended) is str)
                or not _session_id(doc["session"])):
            raise CoinError(CLOSE_FORM)
        return cls(doc["session"], lines, chain, ended, signature)


_ROOT_EVENT = ["event", "index", "label", "received_at", "root"]
_RELEASE_EVENT = ["after", "event", "index", "label", "released_at"]


def _event(line: str) -> Any:
    """A release log line's event: the line is canonical JSON (`canonical_json_text`, integers only) of at most
    `MAX_LOG_LINE` characters, so no two readers of it can read one line two ways."""
    try:
        doc = json.loads(line) if len(line) <= MAX_LOG_LINE else None
        if doc is not None and canonical_json_text(doc) == line:
            return doc
    except (ValueError, TypeError, RecursionError):
        pass
    raise CoinError(LOG_MALFORMED)


def _time(t: Any) -> bool:
    return type(t) is int and t >= 0


def released_roots_of(commitment: LiveCommitment, log: Sequence[str]) -> dict[str, tuple[bytes, ...]]:
    """The roots the challenger received, by family in order: its release log's lines `log`, whose head names
    `commitment`, each root followed by its coin's release.  A malformed log, or another commitment's, raises `CoinError`."""
    if not log or log[0] != canonical_json_text({"event": "commitment", "root": commitment.root.hex()}):
        raise CoinError(f"the challenger's release log is not coin commitment {commitment.root.hex()[:16]}'s")
    roots: dict[str, list[bytes]] = {label: [] for label, _ in commitment.families}
    if len(log) % 2 == 0:
        raise CoinError("the challenger's release log ends on a root without its release")
    for i in range(1, len(log), 2):
        root, release = _event(log[i]), _event(log[i + 1])
        if (not isinstance(root, dict) or not isinstance(release, dict) or sorted(root) != _ROOT_EVENT
                or sorted(release) != _RELEASE_EVENT or root["event"] != "root" or release["event"] != "release"
                or type(root["label"]) is not str or root["label"] not in roots or type(root["index"]) is not int
                or root["index"] != len(roots[root["label"]]) or type(release["index"]) is not int
                or release["label"] != root["label"] or release["index"] != root["index"]
                or release["after"] != root["root"] or type(root["root"]) is not str
                or len(root["root"]) != 128 or _hex(root["root"]) is None or not _time(root["received_at"])
                or not _time(release["released_at"])):
            raise CoinError(LOG_MALFORMED)
        roots[root["label"]].append(bytes.fromhex(root["root"]))
    return {label: tuple(r) for label, r in roots.items()}


def session_of(key: ChallengerKey, record: Any, log: Any, close: Any, verify: Verify
               ) -> tuple[LiveCommitment, dict[str, tuple[bytes, ...]], str]:
    """The coin commitment the challenger signed under `key`, the roots its release log shows it received, by family in
    order, and its session id: its record (`record_reason`), its release log's lines `log` and its signed close `close`
    (each JSON, None when the verifier has none).  Raises `CoinError` naming why when the record or the close is missing,
    malformed or unsigned, the log isn't the one the close signs, or the client didn't close the session."""
    why = record_reason(key, record, verify)
    if why is not None:
        raise CoinError(why)
    r = ChallengerRecord.from_json(record)
    if close is None:
        raise CoinError(NO_CLOSE)
    try:
        c = ChallengerClose.from_json(close)
    except CoinError as e:
        raise CoinError(f"the challenger's close record is malformed: {e}") from None
    try:
        message = close_message(key.public[:16], r.host, r.commitment, c.session, c.lines, c.chain, c.ended)
        ok = verify(key, message, c.signature) is True
    except ValueError:
        ok = False
    if not ok:
        raise CoinError("the challenger's close record's signature does not verify under the verifier's challenger key")
    if type(log) is not list or not all(type(line) is str for line in log):
        raise CoinError(LOG_FORM)
    if len(log) != c.lines:
        raise CoinError(f"the challenger's release log has {len(log)} lines, not the {c.lines} its close record signs")
    if log_chain(log) != c.chain:
        raise CoinError("the challenger's release log is not the one its close record signs")
    if c.ended is not None:
        raise CoinError(f"the challenger's session ended without its client closing it: {c.ended}")
    return r.commitment, released_roots_of(r.commitment, log), c.session


def _digest_hex(s: Any, n: int = 128) -> bool:
    return type(s) is str and len(s) == n and _HEX.fullmatch(s) is not None


def _period(s: Any) -> bool:
    return type(s) is str and 1 <= len(s) <= 64 and s[0] not in "._:-" and all(c in _PERIOD for c in s)


def _window(doc: Any) -> bool:
    return (isinstance(doc, dict) and sorted(doc) == ["commit", "period"] and _digest_hex(doc["commit"])
            and _period(doc["period"]))


def ledger_message(I: bytes, entry: dict) -> bytes:
    """What an auditor's ledger entry's `sig` signs (module docstring), under the key whose identifier is `I`."""
    if len(I) != 16:
        raise CoinError(f"an LMS identifier is 16 bytes, not {len(I)}")
    digest = identity_digest_sha512(LEDGER_TAG, {k: v for k, v in entry.items() if k != "sig"})
    return hashlib.sha512(_field(LEDGER_ROLE.encode()) + I + _field(digest)).digest()


def ledger_line(lines: Sequence[str], kind: str, session: str, body: dict, I: bytes, sign: Callable[[bytes], bytes]) -> str:
    """The auditor's next ledger line after `lines`: entry `kind` for session `session` with body `body`, signed with
    `sign` under the key whose identifier is `I`."""
    entry = {"body": body, "kind": kind, "prev": hashlib.sha512(lines[-1].encode()).hexdigest() if lines else _ZERO,
             "schema": LEDGER_SCHEMA, "seq": len(lines), "session": session}
    return canonical_json_text({**entry, "sig": bytes(sign(ledger_message(I, entry))).hex()})


def _ledger_entry(line: Any) -> dict | None:
    """A ledger line's entry when it is canonical JSON of the ledger's form, else None."""
    try:
        e = json.loads(line) if type(line) is str and len(line) <= MAX_LEDGER_LINE else None
        if e is None or canonical_json_text(e) != line:
            return None
    except (ValueError, TypeError, RecursionError):
        return None
    if not (isinstance(e, dict) and sorted(e) == ["body", "kind", "prev", "schema", "seq", "session", "sig"]
            and type(e["schema"]) is str and type(e["seq"]) is int and e["seq"] >= 0 and _digest_hex(e["prev"])
            and _session_id(e["session"]) and type(e["sig"]) is str and _HEX.fullmatch(e["sig"]) is not None
            and isinstance(e["body"], dict)):
        return None
    b = e["body"]
    if e["kind"] == "open":
        ok = (_window(b.get("window")) and _host(b.get("host")) and _digest_hex(b.get("pinned"), 40)
              and _digest_hex(b.get("key_sha512")))
    elif e["kind"] == "end":
        files = b.get("files")
        ok = (type(b.get("closed")) is bool and isinstance(files, dict)
              and all(_digest_hex(v) for v in files.values()))
    else:
        ok = False
    return e if ok else None


class Audited(NamedTuple):
    """The window's session as the auditor's ledger names it: its id, the host its open names, and the SHA-512 of each
    file its end sealed."""
    session: str
    host: str
    files: dict[str, str]


def audited_session(ledger: Any, window: Any, key: ChallengerKey, verify: Verify) -> Audited:
    """The one challenger session the auditor's ledger `ledger` (its lines, None when the verifier has none) opens for
    window `window` (``{commit, period}``), under the challenger key the window's deployment pins, `key`, and ends closed.
    Raises `CoinError` naming why when there is no ledger, it is malformed, an open names another key, an entry isn't
    chained or signed under `key`, a window or a session is opened twice or a session ended twice or unopened, or the
    window's session is missing, unended or ended unclosed."""
    if ledger is None:
        raise CoinError(NO_AUDIT)
    if type(ledger) is not list:
        raise CoinError(f"the auditor's ledger is malformed: {LEDGER_FORM}")
    if not _window(window):
        raise CoinError(f"the window's identity is malformed: {WINDOW_FORM}")
    pinned = hashlib.sha512(key.public).hexdigest()
    entries = []
    for i, line in enumerate(ledger):
        e = _ledger_entry(line)
        if e is None:
            raise CoinError(f"the auditor's ledger entry {i} is malformed: {LEDGER_FORM}")
        if e["kind"] == "open" and e["body"]["key_sha512"] != pinned:
            raise CoinError(f"the auditor's ledger entry {i} opens session {e['session']} under challenger key "
                            f"{e['body']['key_sha512'][:16]}..., not the window's pinned key {pinned[:16]}...")
        entries.append(e)
    opens: dict[str, dict] = {}
    ends: dict[str, dict] = {}
    prev = _ZERO
    for i, (line, e) in enumerate(zip(ledger, entries)):
        if e["schema"] != LEDGER_SCHEMA or e["seq"] != i or e["prev"] != prev:
            raise CoinError(f"the auditor's ledger entry {i} is not entry {i} of a {LEDGER_SCHEMA} ledger chained to its "
                            "predecessor")
        try:
            ok = verify(key, ledger_message(key.public[:16], e), bytes.fromhex(e["sig"])) is True
        except ValueError:
            ok = False
        if not ok:
            raise CoinError(f"the auditor's ledger entry {i}'s signature does not verify under the verifier's challenger "
                            "key")
        s = e["session"]
        if e["kind"] == "open":
            if s in opens or any(o["body"]["window"] == e["body"]["window"] for o in opens.values()):
                raise CoinError(f"the auditor's ledger entry {i} opens a second session for its window, or session {s} "
                                "again: a window has one session")
            opens[s] = e
        else:
            if s not in opens or s in ends:
                raise CoinError(f"the auditor's ledger entry {i} ends session {s}, which it has not opened or has ended")
            ends[s] = e
        prev = hashlib.sha512(line.encode()).hexdigest()
    named = [o for o in opens.values() if o["body"]["window"] == window]
    if not named:
        raise CoinError("the auditor's ledger opens no challenger session for this window")
    s = named[0]["session"]
    if s not in ends:
        raise CoinError(f"the auditor's ledger has no end for session {s}: the session never ended, or the ledger is "
                        "older than its end")
    if ends[s]["body"]["closed"] is not True:
        raise CoinError(f"the auditor's ledger ends session {s} unclosed: its client didn't close it")
    return Audited(s, named[0]["body"]["host"], dict(ends[s]["body"]["files"]))


def log_digest(lines: Sequence[str]) -> str:
    """The SHA-512 of the release log file whose lines are `lines` (each ended by a newline), in lowercase hex."""
    return hashlib.sha512("".join(line + "\n" for line in lines).encode()).hexdigest()


def session_named(audited: Audited, key: ChallengerKey, session: Any, verify: Verify
                  ) -> tuple[LiveCommitment, dict[str, tuple[bytes, ...]]]:
    """What the verifier takes from session `audited`'s files `session`, ``{"close", "log", "record"}`` (`session_of`):
    the coin commitment the challenger signed and the roots its release log shows it received, when its close names the
    ledger's session, its record the open's host and its release log is the one the ledger's end sealed.  Raises
    `CoinError` naming why."""
    if not isinstance(session, dict) or sorted(session) != ["close", "log", "record"]:
        raise CoinError(SESSION_FORM)
    commitment, roots, sid = session_of(key, session["record"], session["log"], session["close"], verify)
    if sid != audited.session:
        raise CoinError(f"the challenger's session is {sid}, not {audited.session}, the one the auditor's ledger opens "
                        "for this window")
    host = ChallengerRecord.from_json(session["record"]).host
    if host != audited.host:
        raise CoinError(f"the challenger's record names host {host}, not {audited.host}, the one the auditor's ledger "
                        "opened its session on")
    if audited.files.get(LOG_FILE) != log_digest(session["log"]):
        raise CoinError("the challenger's release log is not the one the auditor's ledger sealed at its session's end")
    return commitment, roots


def trusted_session_of(ledger: Any, window: Any, key: ChallengerKey, session: Any, verify: Verify
                       ) -> tuple[LiveCommitment, dict[str, tuple[bytes, ...]]]:
    """What the verifier takes from the challenger's session for window `window`: the one session the auditor's ledger
    opens for it under `key` (`audited_session`), and that session's files `session` (`session_named`): the coin
    commitment the challenger signed and the roots its release log shows it received.  Raises `CoinError` naming why, so
    a window whose coins aren't that session's fails closed."""
    return session_named(audited_session(ledger, window, key, verify), key, session, verify)


class LiveChallenger:
    """The challenger's side of one epoch's coins.  It draws every coin and hm96 salt of the layout from `entropy` and
    commits to them at construction, before the epoch; `open` releases coin (`label`, `index`) once, after the verifier
    recorded `after`.  `entropy` has no default, so nothing here reads the operating system: a served challenger's is the
    Lean verifier of record's read of it (`IO.getRandomBytes`, A3)."""

    def __init__(self, scheme: str, epoch: int, families: Sequence[tuple[str, int]],
                 entropy: Callable[[int], bytes]) -> None:
        families = tuple(tuple(f) for f in families)
        why = layout_problem(scheme, epoch, families)
        if why is not None:
            raise CoinError(why)

        def draw(n: int) -> bytes:
            out = entropy(n)
            if type(out) is not bytes or len(out) != n:
                raise ValueError(f"entropy({n}) returned other than {n} bytes")
            return out

        self._coins = tuple((draw(COIN_BYTES), draw(HM96.salt_bytes)) for _ in range(sum(n for _, n in families)))
        self._tree = MerkleTree(live_domain(scheme, epoch, families),
                                {p: live_leaf(scheme, *c) for p, c in enumerate(self._coins)}, lambda _: scheme)
        self.commitment = LiveCommitment(scheme, epoch, families, self._tree.commitment.root)
        self._opened: set[tuple[str, int]] = set()
        self.record: ChallengerRecord | None = None

    def sign_record(self, host: str, public: bytes, sign: Callable[[bytes], bytes]) -> ChallengerRecord:
        """The challenger's record of its commitment, signed once and before any coin opens: `sign(message)` signs with
        the LMS key whose public key is `public`, on host `host`."""
        if self._opened:
            raise CoinError("a challenger signs its record before any coin opens, and a coin is open")
        if self.record is not None:
            raise CoinError("the challenger's record is signed once")
        if not _host(host) or type(public) is not bytes or len(public) != PUBLIC_BYTES:
            raise CoinError(RECORD_FORM)
        self.record = ChallengerRecord(self.commitment, host, bytes(sign(record_message(public[:16], host, self.commitment))))
        return self.record

    def open(self, label: str, index: int, after: bytes) -> LiveOpening:
        position = self.commitment.position(label, index)
        if position is None:
            raise CoinError(f"the epoch's coins have no {label} coin {index}")
        if (label, index) in self._opened:
            raise CoinError(f"epoch {self.commitment.epoch}'s {label} coin {index} is already open: a retry never redraws")
        self._opened.add((label, index))
        coin, salt = self._coins[position]
        return LiveOpening(self.commitment.epoch, label, index, bytes(after), coin, salt, self._tree.open(position).path)


class LiveRecord:
    """The verifier's record of one family of an epoch's coins, in order: `record(root)` records the family's next root and
    only then asks the challenger for its coin, after that root; `reason` is `order_reason` over the record.  `clock` is
    the record's clock in seconds, and `log` the release order on it."""

    def __init__(self, challenger: LiveChallenger, label: str, clock: Callable[[], float] = time.monotonic) -> None:
        if challenger.commitment.family(label) is None:
            raise CoinError(f"the epoch's coins have no {label} family")
        self.challenger, self.label, self._clock = challenger, label, clock
        self.roots: list[bytes] = []
        self.openings: list[LiveOpening] = []
        self._log: list[dict[str, Any]] = []

    @property
    def log(self) -> tuple[dict[str, Any], ...]:
        """The release order with the record's time, two events a coin: root i's receipt, `{"event": "root", "label",
        "index", "root": <hex>, "received_at"}`, then coin i's release after it, `{"event": "release", "label", "index",
        "after": <hex>, "released_at"}`.  It records the order; that the challenger kept it is `trusted-nebius-host`'s."""
        return tuple(self._log)

    def record(self, root: bytes) -> LiveOpening:
        if len(self.roots) == self.challenger.commitment.family(self.label):
            raise CoinError(f"every {self.label} coin of the epoch is open")
        i = len(self.roots)
        self.roots.append(bytes(root))
        self._log.append({"event": "root", "label": self.label, "index": i, "root": bytes(root).hex(),
                          "received_at": self._clock()})
        opening = self.challenger.open(self.label, i, root)
        self.openings.append(opening)
        self._log.append({"event": "release", "label": self.label, "index": i, "after": opening.after.hex(),
                          "released_at": self._clock()})
        return opening

    def reason(self) -> str | None:
        return order_reason(self.challenger.commitment, self.label, self.roots, self.openings)


__all__ = ["BINDING", "CLOSE_FORM", "CLOSE_ROLE", "COIN_BYTES", "COMMITMENT_FORM", "COMMITMENT_TAG", "KEY_FORM", "LEDGER_FORM",
           "LEDGER_ROLE", "LEDGER_SCHEMA", "LEDGER_TAG", "LOG_FILE", "LOG_FORM", "LOG_MALFORMED", "LOG_TAG", "MAX_COINS",
           "MAX_LEDGER_LINE", "MAX_LOG_LINE", "MAX_LOG_LINES", "MAX_SESSION_ID", "NO_AUDIT", "NO_CLOSE", "NO_RECORD",
           "OPENING_FORM", "PUBLIC_BYTES", "RECORD_FORM", "RECORD_ROLE", "SESSION_FORM", "WINDOW_FORM", "Audited",
           "ChallengerClose", "ChallengerKey", "ChallengerRecord", "CoinError", "LiveChallenger", "LiveCommitment",
           "LiveOpening", "LiveRecord", "Verify", "audited_session", "close_message", "layout_problem", "ledger_line",
           "ledger_message", "live_domain", "live_leaf", "live_reason", "log_chain", "log_digest", "order_reason",
           "record_message", "record_reason", "released_roots_of", "session_named", "session_of", "trusted_commitment",
           "trusted_reason", "trusted_session_of", "write_record"]
