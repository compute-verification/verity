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

**Every coin source is named by a claim instance**: its ``claim``, a ``uniform/<instance>`` assumption of
`verity.core.claims` (`coin_claim`), which the challenger keeps as its `LiveChallenger.claim`.  The Lean verifier of
record's `coins` (`IO.getRandomBytes`) is ``uniform/io-getrandombytes`` (A3); `named` names another source.  A served
challenger refuses entropy no claim instance names (`served_claim`, `NO_CLAIM`), so its coins never come from
`os.urandom` or any other source no assumption covers; a test's seeded stream stays unnamed.

Timing is a trust assumption, not a check.  `after` binds an opening to the root the verifier recorded, but can't show
the coin was released after it: that rests on the challenger releasing coin i only once it has received root i.  The
challenger is auditor-trusted, the auditor's device on a machine the prover doesn't control, under the named assumption
`trusted-host`.  The record keeps the release order with its own time (`LiveRecord.log`).

`live_reason` is the verifier's check of one opening and `order_reason` of a family's openings against the roots it
recorded, in order: coin i opened after root i, each once, none before its root is recorded, and every recorded root's
coin opened.  `LiveRecord` is the verifier's side of one family: it records a root and only then asks for that root's
coin, and logs both on its clock.  `record_many` does that for several families' next coins at once, asked in one
`open_many` (one round trip to a remote challenger).

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
auditor (`benchmarks/live_challenger/auditor.py` on the challenger's host, run by the host's operator) appends a signed
entry, under the
challenger's key, when a lane reserves a window (its request, start and expiry), when it opens the reserved session for
the window's root, before its first coin, and when the session ends; a window gets one session.  A lane's period is two
more entries, its `period-open` and `period-close`:

    entry   = {body, kind, prev, schema, seq, session, sig}        seq from 0; prev: SHA-512 of the previous line (128
                                                                   zeros first); schema v1 or v2, no v1 after a v2
    reserve = body {by, [for], peer, scheme, epoch, families, minutes, started_utc, expires_utc}
    open    = body {window: {root}, host, pinned, key_sha512, by, [for], reserve, opened_by, started_utc, ...}
                                                                   host: the one the session's record signs
    end     = body {closed, files, record_leaf, commitment, refused_hellos, complete, ...}    files: each file's SHA-512
    period  = body {period, lane, ...} (session null); a close lists its windows as {root, session, complete}, root
              null for a reserve never opened
    message = SHA-512([LEDGER_ROLE] || I || [identity_digest_sha512(LEDGER_TAGS[schema], entry without sig)])

each entry a record entry (`verity.core.protocols.record`, `Core.Record`), each line as the auditor writes it
(`ledger_text`, every character past ASCII escaped), and each signature over the entry's canonical JSON in UTF-8
(`ledger_message`).  The window is `{root}`, the served Commit's root, its prover the
open's `for`, else its `by` (`lane_of`).  The verifier pins the challenger key from outside the ledger (the window's
deployment; `key_of_record` reads the operator's published pin), and refuses any open or period-open naming another
key's
SHA-512 before it checks a signature, so a ledger signed under its own key can't pin itself.  Then the ledger is
the operator's reference check, `verify_ledger` and `verify_periods` (one `_verify`, which the auditor's host runs
too): each
entry chained and signed at strictly increasing leaves of the key (a ledger restored with its key's state forks at a
reused leaf), a reserve first in its session and expiring its minutes after its start, one open a root and a session,
binding its session's reserve and carrying its request after the first reserve, one end a session after its open, each
record's leaf between its session's open and end and no entry's, and every period's close listing exactly its lane's
windows, counted at their reserves.  The window's session must be
opened under v2, ended and closed, its end naming its record's leaf and commitment and how many hellos it refused, and
sealing the session's record, close, release log and refused hellos; then its record names the open's host and is
signed at that leaf over that commitment, its signed close names the ledger's session id, and its release log and
refused hellos are the ones the end sealed (`session_named`).  A ledger older than the window's end is refused, as is a
window it opens no session for, so one session's coins pass for one window only.  Infra publishes the ledger's heads;
`ledger_head` checks each, refuses any two that don't extend one another, and takes the longest.  Every entry is under
the one pinned key, so a ledger whose key changes is refused: no entry, signed under the old key, hands the ledger to a
new one.  `ledger_line` writes an entry as the auditor does.

**A window counts toward its period** only when it proves the circuit the period's open committed (`credited_session`):
the period's `circuit` is the declared circuit's SHA-512 (`circuit_of`), and the verifier holds the descriptor of the
circuit the window's sessions prove, its canonical JSON without annotations (the bytes `program_digest` hashes), so the
check is one SHA-512 over the descriptor, never a walk over the circuit it unrolls to (`circuit_reason`).  A window in
no period, in a period that committed no circuit, or whose descriptor hashes to another circuit counts toward none.

**A session runs main's code** only when the host's pins log records its pin (`pinned_session`, RT4-INT-40).  The open's
`pinned` is the commit the host's tree was pinned to.  The host keeps an append-only pins log, a line per pin as the
auditor writes it (`pin_line`):

    pin     = {commit, main, pinned_utc}        commit: the pin; main: the origin/main commit whose history held it
                                                 when vy-challenger-pin pinned it (each 40 lowercase hex)
    open    = body {..., pinned, pins: {lines, sha512}}        the pins log as it stood when the gate admitted the open

and each open seals the log as it stood (`pins_seal`), under the open's signature.  The session counts only when the
verifier's copy of the log (the operator's published one, any extension of it) begins with the sealed lines and one of
them
pins the open's commit (`pins_reason`), and that commit is on main's first-parent line in the verifier's own copy of it
(`first_parent_reason`; the auditor's ``git rev-list --first-parent origin/main``, an argument, never the host's or the
prover's word).  Ancestry is not enough: main lands each change as a merge, so a branch commit merged into main is in
main's history, but main's check ran on the merge's tree, never on the branch commit's.  That the host ran the commit
its open names is the auditor-trusted host's word (`trusted-host`).  Every path that credits a session reads it
through `pinned_session`: `credited_session` (the pin first, then the circuit) and `trusted_session_of`.
`audited_session` checks no pin and credits nothing.

The first client is PoUW's live window (`verity.core.protocols.pouw.window`, "Live windows"): its salt, its draw and
its round coins, round r's opened after round r's root.  The module reads only the standard library,
`verity.ml.commitments`, `verity.core.claims` and `verity.core.protocols.record`, so any client calls it as it is.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import string
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NamedTuple

from verity.core import claims
from verity.core.protocols import record as core_record
from verity.ml.commitments import (Commitment, CommitmentDomain, MerkleTree, Opening, RangeIndexedDomain,
                                   VerificationLimits, verify_opening)
from verity.ml.commitments import hm96
from verity.ml.commitments.identity import canonical_json_text, identity_digest_sha512, tagged_sha512

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
KEY_RECORD_SCHEMA = "challenger-key/v1"
KEY_RECORD_FORM = (f"a challenger key record is {KEY_RECORD_SCHEMA} {{H, W, pubkey, pubkey_sha512, schema, ...}}: a "
                   "challenger key's parameters, its public key in lowercase hex and that public key's SHA-512")
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
SESSION_FORM = ("a challenger session is {close, hellos, log, record}: its close record, its refused hellos' lines, its "
                "release log's lines and its record")
LOG_FILE = "releases.jsonl"
LEDGER_SCHEMA_V1 = "verity/challenger-auditor/v1"
LEDGER_SCHEMA = "verity/challenger-auditor/v2"
LEDGER_ROLE = "verity/challenger-auditor-record/v1"
LEDGER_TAGS = {LEDGER_SCHEMA_V1: "verity/challenger-auditor/entry/v1", LEDGER_SCHEMA: "verity/challenger-auditor/entry/v2"}
PERIOD_KINDS = ("period-open", "period-close")
SESSION_KINDS = ("reserve", "open", "end")
#: the request a reserve records and its session's open carries
RESERVED = ("by", "for", "peer", "scheme", "epoch", "families", "minutes")
#: a reserve's most minutes
MINUTES = (1, 720)
RECORD_FILE, CLOSE_FILE, HELLOS_FILE = "record.json", "challenger.json", "hellos.jsonl"
#: the files of a session the verifier reads, each of which the ledger's end must seal
SEALED_FILES = (RECORD_FILE, CLOSE_FILE, LOG_FILE, HELLOS_FILE)
#: an auditor's ledger line's most characters: its signature at W = 1, H = 30 is 70928 hex digits
MAX_LEDGER_LINE = 1 << 18
_ZERO = core_record.ZERO
_LEDGER_ROOT = re.compile(r"[0-9a-f]{128}")
_LEDGER_HEX = re.compile(r"(?:[0-9a-f]{2})+")
_LEDGER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
_RESERVED_SESSION = re.compile(r"s[0-9]{8}T[0-9]{6}Z-[0-9a-f]{4}")
_UTC = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})Z")
CAP = (1, 2 ** 24)
CIRCUIT = re.compile(r"[0-9a-fA-F]{128}")
WINDOW_FORM = "a window's identity is {root}: the served Commit's root, 64 bytes in lowercase hex"
LEDGER_FORM = ("an auditor's ledger line is an entry {body, kind, prev, schema, seq, session, sig} as the auditor writes it "
               "(sorted keys, no whitespace, ASCII, no number but an integer): kind open, its body naming the window, the "
               "challenger's host, the pinned source commit (40 lowercase hex), the SHA-512 of the challenger key's public "
               "key (key_sha512) and in v2 its opener (by, and for when opened for a lane); kind reserve, its body the "
               "window's request; kind end, its body whether its client closed the session (closed), the SHA-512 of each "
               "file the session left (files) and in v2 the leaf its record was signed at, if any; kind period-open, its "
               "period, lane and key_sha512; or kind period-close, its period, lane, windows ({complete, root, session} "
               "each, root null for a reserve never opened) and whether all are complete")
END_FORM = ("a window's end names the leaf its record was signed at (record_leaf), its commitment's root (commitment), how "
            "many hellos it refused (refused_hellos) and seals " + ", ".join(SEALED_FILES))
HEADS_FORM = "the auditor's published heads are a list of ledgers, each its lines"
DESCRIPTOR_FORM = ("the descriptor of the circuit a window's sessions prove is bytes: its canonical JSON without "
                   "annotations, the bytes program_digest hashes")
HELLOS_FORM = "the challenger's refused hellos are a list of their lines"
PINS_FORM = ("the host's pins log is a list of its lines, each {commit, main, pinned_utc} as the auditor writes it: "
             "the commit the host was pinned to and the origin/main commit whose history held it (40 lowercase hex "
             "each), and when, YYYY-MM-DDTHH:MM:SSZ")
NO_PINS = ("no pins log: the verifier trusts a challenger session only from a tree the host's pins log records as a "
           "commit of main's history")
MAIN_FORM = ("a list of main's first-parent commits, each 40 lowercase hex (`git rev-list --first-parent origin/main` in "
             "the auditor's own checkout)")
NO_MAIN = ("the verifier holds no copy of main's history, so it can't tell whether the host's pinned tree is one main's "
           "check passed: give it the auditor's own `git rev-list --first-parent origin/main`")
NO_AUDIT = "no auditor's ledger: the verifier trusts a challenger session only as the auditor's signed ledger names it"
#: the property every coin source's claim instance is assumed of: independent uniform output
COIN_PROPERTY = "uniform"
NO_CLAIM = ("coins-without-claim: the challenger's entropy is named by no claim instance (its `claim`, a uniform/<instance> "
            "assumption of verity.core.claims), so no assumption covers its coins: a served challenger draws from a named "
            "source, the Lean verifier of record's `coins` (uniform/io-getrandombytes)")


class CoinError(ValueError):
    """A layout, a coin commitment or an opening the challenger refuses, or a coin its commitment doesn't hold."""


def coin_claim(entropy: Any) -> str | None:
    """The claim instance that names `entropy`'s bytes: its ``claim``, a ``uniform/<instance>`` assumption of
    `verity.core.claims`, or None when it carries none (a test's seeded stream).  A ``claim`` that is no such id raises
    `CoinError`: a misnamed source is refused, never taken for an unnamed one."""
    claim = getattr(entropy, "claim", None)
    if claim is None:
        return None
    prop, sep, instance = claim.partition("/") if type(claim) is str else ("", "", "")
    if not (sep and prop == COIN_PROPERTY and instance in claims.INSTANCES):
        raise CoinError(f"a coin source's claim is a {COIN_PROPERTY}/<instance> assumption of verity.core.claims, "
                        f"not {claim!r}")
    return claim


def served_claim(entropy: Any) -> str:
    """`coin_claim` of a served challenger's entropy, which must have one (`NO_CLAIM`)."""
    claim = coin_claim(entropy)
    if claim is None:
        raise CoinError(NO_CLAIM)
    return claim


def named(claim: str, entropy: Callable[[int], bytes]) -> Callable[[int], bytes]:
    """`entropy` named by the claim instance `claim` (`coin_claim`): for a source whose bytes that assumption covers."""
    def draw(n: int) -> bytes:
        return entropy(n)
    draw.claim = claim  # type: ignore[attr-defined]
    coin_claim(draw)
    return draw


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


def key_of_record(doc: Any) -> ChallengerKey:
    """The challenger key a published key record pins (`KEY_RECORD_SCHEMA`, the store artifact the auditor's operator
    labels ``challenger-key current``; the caller reads only one the operator published): its `W`, `H` and public key,
    refused
    (`CoinError`) unless the record is of that form and the public key's SHA-512 is the one it names."""
    if not (isinstance(doc, dict) and doc.get("schema") == KEY_RECORD_SCHEMA and _digest_hex(doc.get("pubkey_sha512"))
            and _hex(doc.get("pubkey")) is not None):
        raise CoinError(KEY_RECORD_FORM)
    key = ChallengerKey(doc.get("W"), doc.get("H"), _hex(doc["pubkey"]))
    if hashlib.sha512(key.public).hexdigest() != doc["pubkey_sha512"]:
        raise CoinError("the challenger key record's public key is not the one whose SHA-512 it names")
    return key


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
    return type(s) is str and _LEDGER_NAME.fullmatch(s) is not None


def _v1_window(doc: Any) -> bool:
    return (isinstance(doc, dict) and sorted(doc) == ["commit", "period"] and _digest_hex(doc["commit"])
            and _period(doc["period"]))


def _window(doc: Any) -> bool:
    return isinstance(doc, dict) and sorted(doc) == ["root"] and _digest_hex(doc["root"])


def _hexed(v: Any) -> Any:
    """`v` with every string, keys included, as its UTF-8 in hex: what `canonical_json_text` takes to check `v`'s numbers
    and keys alone, since a string past ASCII is canonical as the ledger writes it, escaped."""
    if isinstance(v, str):
        return v.encode().hex()
    if isinstance(v, dict):
        return {_hexed(k): _hexed(x) for k, x in v.items()}
    return [_hexed(x) for x in v] if isinstance(v, (list, tuple)) else v


def ledger_message(I: bytes, entry: dict) -> bytes:
    """What an entry's `sig` signs: SHA-512([ROLE] || I || [identity_digest_sha512(TAG, entry without sig)]), each [x] x
    prefixed with its length (u64), as `challenger.record_message` frames the challenger's record under its own role; TAG is
    the entry's schema's (`TAGS`). The digest is over the entry's canonical JSON in UTF-8, a string past ASCII as its own
    characters, as the challenger signs it, not over its line, where `ledger_text` escapes them."""
    if len(I) != 16:
        raise ValueError(f"an LMS identifier is 16 bytes, not {len(I)}")
    tag = LEDGER_TAGS.get(entry.get("schema"))
    if tag is None:
        raise ValueError(f"schema {entry.get('schema')!r} is none of {sorted(LEDGER_TAGS)}")
    unsigned = core_record.unsigned(entry)
    canonical_json_text(_hexed(unsigned))
    text = json.dumps(unsigned, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return core_record.message(LEDGER_ROLE, I, tagged_sha512(tag, text.encode()))


def leaf_of(sig: Any) -> int:
    """The LMS leaf a hex signature was made at (its first 4 bytes, `lms.Signature.encode`)."""
    if type(sig) is not str or len(sig) < 8 or not _LEDGER_HEX.fullmatch(sig):
        raise ValueError("a signature is lowercase hex")
    return int(sig[:8], 16)


def lane_of(e: dict) -> str:
    """A reserve's or an open's prover: the lane vy-admin reserved or opened it for (`for`), else its requester (`by`)."""
    return e["body"].get("for") or e["body"]["by"]


def root_of(e: dict) -> str:
    """An open entry's window root: v2's `window.root`, v1's `window.commit` (v1's period label names nothing)."""
    w = e["body"]["window"]
    return w["root"] if e["schema"] == LEDGER_SCHEMA else w["commit"]


def circuit_of(s: Any) -> str:
    """A period's circuit as its open records it: the declared circuit's SHA-512 (128 hex digits, kept in lowercase),
    never a Program's id; ValueError otherwise."""
    if type(s) is not str or not CIRCUIT.fullmatch(s):
        raise ValueError(f"--circuit {s!r}: the declared circuit's SHA-512 (128 hex digits), not a Program's id")
    return s.lower()


def later(t: Any, minutes: int) -> str | None:
    """`t` (``YYYY-MM-DDTHH:MM:SSZ``, UTC) plus `minutes` in that form: a reserve's expiry; None when `t` isn't a time of
    that form."""
    m = _UTC.fullmatch(t) if type(t) is str else None
    if m is None:
        return None
    try:
        d = datetime(*map(int, m.groups()), tzinfo=timezone.utc) + timedelta(minutes=minutes)
    except (ValueError, OverflowError):
        return None
    return f"{d.year:04d}-{d.month:02d}-{d.day:02d}T{d.hour:02d}:{d.minute:02d}:{d.second:02d}Z"


def cap_of(v: Any) -> int:
    """A period's cap: the most windows it may hold, a positive integer; ValueError otherwise."""
    if type(v) is not int or not CAP[0] <= v <= CAP[1]:
        raise ValueError(f"--cap {v!r}: the most windows the period may hold, {CAP[0]} to {CAP[1]}")
    return v


def extends(published: list[bytes], ledger: list[bytes]) -> bool:
    """Whether `ledger` is `published` with entries appended: a head that isn't (a fork, a cut, an edit) is refused."""
    return ledger[:len(published)] == published


def verify_ledger(lines: list[bytes], I: bytes, verify: Callable[[bytes, bytes], bool]) -> dict[str, dict]:
    """The reference check of a published ledger under the challenger key: every entry v1 or v2 (no v1 after a v2) and signed,
    `seq` from 0, `prev` chained, its leaves strictly increasing; a `reserve` first in its session, its expiry its start
    plus its minutes; one `open` a root, whatever its period label, in canonical lowercase, and after the first reserve
    each open binds its session's reserve (`reserve`, its seq), carries its request, is by its prover and falls inside its
    minutes and before its period's close; one `end` a session, after its `open`, and a v2 end's `record_leaf` between its
    session's open and end and no entry's or other record's; periods as `verify_periods` checks them. Returns root ->
    {"reserve" (None for a one-step open), "open", "end" or None}: S is the open's `session`, R its root; raises
    ValueError at the first entry that fails. It reads one head: a verifier takes the operator's newest published head,
    and refuses
    two that don't `extend` one another."""
    return _verify(lines, I, verify)[0]


def verify_periods(lines: list[bytes], I: bytes, verify: Callable[[bytes, bytes], bool]) -> dict[str, dict]:
    """The ledger's periods once `verify_ledger` passes: period -> {"lane", "open", "close" (None while open), "windows":
    [session, ...], "cap", "circuit"}. A period's name is used once; a lane has one period open at a time; its windows are
    exactly the reserves (and one-step v2 opens) whose prover is its lane (`lane_of`) between its period-open and its
    period-close, counted at the reserve, so a reserve that expired unopened is one, at most its `cap` (a positive
    integer, with `circuit` committed beside it before window 0's first coin; both None only for a period opened before
    they were required, which a reader of the cap refuses), and the close lists each of them, in order, with its root
    (null when never opened) and outcome (complete when its end precedes the close and is complete) and is complete only
    when it has a window and every one is. An end after its period's close (the gate's lock leaves none) changes
    nothing: the close recorded that window incomplete."""
    return _verify(lines, I, verify)[1]


def _same(a: Any, b: Any) -> bool:
    """JSON equality that tells `true` from 1, as no Python `==` does."""
    try:
        return canonical_json_text(a) == canonical_json_text(b)
    except (TypeError, ValueError):
        return False


def _verify(lines: list[bytes], I: bytes, verify: Callable[[bytes, bytes], bool]) -> tuple[dict[str, dict], dict[str, dict]]:
    windows: dict[str, dict] = {}
    by_session: dict[str, str] = {}
    leaves: set[int] = set()
    records: dict[int, int] = {}
    periods: dict[str, dict] = {}
    current: dict[str, str] = {}
    reserves: dict[str, dict] = {}
    prev, last, v2, two_step = _ZERO, -1, False, False

    def join(i: int, lane: str | None, session: str) -> str | None:
        if (period := current.get(lane) if lane is not None else None) is not None:
            p = periods[period]
            if p["cap"] is not None and len(p["windows"]) >= p["cap"]:
                raise ValueError(f"entry {i}: period {period}'s window {len(p['windows'])} is past its cap {p['cap']}")
            p["windows"].append(session)
        return period

    for i, ln in enumerate(lines):
        try:
            e = json.loads(ln)
        except ValueError:
            raise ValueError(f"entry {i}: not JSON") from None
        if (type(e) is not dict or type(e.get("schema")) is not str or e["schema"] not in LEDGER_TAGS
                or (v2 and e["schema"] == LEDGER_SCHEMA_V1) or e.get("seq") != i or e.get("prev") != prev):
            raise ValueError(f"entry {i}: not entry {i} of a {LEDGER_SCHEMA} ledger chained to its predecessor")
        v2 = v2 or e["schema"] == LEDGER_SCHEMA
        kind = e.get("kind")
        period_kind = kind in PERIOD_KINDS
        if (e.get("session") is not None if period_kind else type(e.get("session")) is not str) or type(e.get("body")) is not dict:
            raise ValueError(f"entry {i}: a session's entry has a session, a period's none, and each a body")
        q = leaf_of(e.get("sig"))
        if not verify(ledger_message(I, e), bytes.fromhex(e["sig"])):
            raise ValueError(f"entry {i}: its signature doesn't verify under the challenger key")
        if q <= last:
            raise ValueError(f"entry {i}: signed at leaf {q}, not after the last entry's {last}")
        if kind == "reserve":
            b, session = e["body"], e["session"]
            if (e["schema"] != LEDGER_SCHEMA or not _RESERVED_SESSION.fullmatch(session) or session in reserves
                    or session in by_session):
                raise ValueError(f"entry {i}: a reserve is v2 and the first entry of its session {session}")
            if (not _period(b.get("by")) or ("for" in b and not _period(b["for"]))
                    or type(b.get("minutes")) is not int or not MINUTES[0] <= b["minutes"] <= MINUTES[1]
                    or later(b.get("started_utc"), b["minutes"]) is None
                    or b.get("expires_utc") != later(b["started_utc"], b["minutes"])):
                raise ValueError(f"entry {i}: a reserve names its prover, its minutes, its start and its expiry (start + "
                                 "minutes)")
            two_step = True
            reserves[session] = {"entry": e, "period": join(i, lane_of(e), session), "opened": False}
        elif kind == "open":
            try:
                root = root_of(e)
                lane = lane_of(e) if e["schema"] == LEDGER_SCHEMA else None
            except (KeyError, TypeError):
                raise ValueError(f"entry {i}: an open names its window and its prover") from None
            if type(root) is not str or not _LEDGER_ROOT.fullmatch(root):
                raise ValueError(f"entry {i}: a window root is 128 lowercase hex digits")
            if root in windows or e["session"] in by_session:
                raise ValueError(f"entry {i}: a second open for root {root[:16]}… or session {e['session']}")
            b = e["body"]
            if e["schema"] == LEDGER_SCHEMA and "reserve" in b:
                r = reserves.get(e["session"])
                if r is None or type(b["reserve"]) is not int or r["entry"]["seq"] != b["reserve"]:
                    raise ValueError(f"entry {i}: an open for session {e['session']} binds no reserve of it")
                rb = r["entry"]["body"]
                if not all(_same(b.get(k), rb.get(k)) for k in RESERVED):
                    raise ValueError(f"entry {i}: session {e['session']}'s open doesn't carry its reserve's request")
                who = b.get("opened_by")
                if type(who) is not str or who not in (rb["by"], rb.get("for")):
                    raise ValueError(f"entry {i}: session {e['session']} was opened by "
                                     f"{repr(who) if type(who) is str else 'no lane'}, not its reserve's prover")
                if (type(b.get("started_utc")) is not str or later(b["started_utc"], 0) != b["started_utc"]
                        or not rb["started_utc"] <= b["started_utc"] <= rb["expires_utc"]):
                    raise ValueError(f"entry {i}: session {e['session']} was opened outside its reserve's minutes")
                if r["period"] is not None and periods[r["period"]]["close"] is not None:
                    raise ValueError(f"entry {i}: session {e['session']} was opened after its period {r['period']} closed")
                r["opened"] = True
            elif e["schema"] == LEDGER_SCHEMA and two_step:
                raise ValueError(f"entry {i}: a v2 open without a reserve after the ledger's first reserve")
            else:
                join(i, lane, e["session"])
            windows[root], by_session[e["session"]] = {"open": e, "end": None, "q": q}, root
        elif kind == "end":
            root = by_session.get(e["session"])
            if root is None or windows[root]["end"] is not None:
                raise ValueError(f"entry {i}: an end for session {e['session']}, which has no open or has ended")
            r = e["body"].get("record_leaf") if e["schema"] == LEDGER_SCHEMA else None
            if r is not None:
                if type(r) is not int or not windows[root]["q"] < r < q or r in records:
                    raise ValueError(f"entry {i}: record leaf {r!r} isn't between its open's and its end's, once")
                records[r] = i
            windows[root]["end"] = e
        elif period_kind:
            name, lane = e["body"].get("period"), e["body"].get("lane")
            if (e["schema"] != LEDGER_SCHEMA or type(name) is not str or not _LEDGER_NAME.fullmatch(name)
                    or type(lane) is not str or not _LEDGER_NAME.fullmatch(lane)):
                raise ValueError(f"entry {i}: a {kind} is v2 and names its period and its lane")
            if kind == "period-open":
                if name in periods or lane in current:
                    raise ValueError(f"entry {i}: period {name} is named twice, or lane {lane} has a period open")
                cap, circuit = e["body"].get("cap"), e["body"].get("circuit")
                if cap is not None or circuit is not None:
                    try:
                        if cap_of(cap) != cap or circuit_of(circuit) != circuit:
                            raise ValueError("not canonical")
                    except ValueError:
                        raise ValueError(f"entry {i}: period {name}'s cap is a positive integer and its circuit a "
                                         "SHA-512 in lowercase") from None
                periods[name], current[lane] = {"lane": lane, "open": e, "close": None, "windows": [], "cap": cap,
                                                "circuit": circuit}, name
            else:
                p = periods.get(name)
                if p is None or p["close"] is not None or p["lane"] != lane:
                    raise ValueError(f"entry {i}: a period-close for {name}, which isn't lane {lane}'s open period")
                outcomes = [{"root": by_session.get(s), "session": s,
                             "complete": s in by_session and windows[by_session[s]]["end"] is not None
                             and windows[by_session[s]]["end"]["body"].get("complete") is True} for s in p["windows"]]
                if (e["body"].get("windows") != outcomes
                        or e["body"].get("complete") is not (bool(outcomes) and all(w["complete"] for w in outcomes))):
                    raise ValueError(f"entry {i}: period {name}'s close doesn't list exactly its windows and their outcomes")
                p["close"] = e
                del current[lane]
        else:
            raise ValueError(f"entry {i}: kind {kind!r}")
        leaves.add(q)
        prev, last = core_record.prev_of(ln), q
    if reused := sorted(leaves & set(records)):
        raise ValueError(f"entry {records[reused[0]]}: record leaf {reused[0]} is also an entry's")
    return ({root: {"reserve": (reserves.get(w["open"]["session"]) or {}).get("entry"), "open": w["open"], "end": w["end"]}
             for root, w in windows.items()},
            {name: {k: p[k] for k in ("lane", "open", "close", "windows", "cap", "circuit")} for name, p in periods.items()})


def ledger_text(entry: dict) -> str:
    """An entry's line as the auditor writes it (`core_record.line`): sorted keys, no whitespace, every character past ASCII
    escaped."""
    return core_record.line(entry)


def ledger_line(lines: Sequence[str], kind: str, session: str | None, body: dict, I: bytes,
                sign: Callable[[bytes], bytes], schema: str = LEDGER_SCHEMA) -> str:
    """The auditor's next ledger line after `lines`: entry `kind` for session `session` (None for a period's) with body
    `body`, signed with `sign` under the key whose identifier is `I`."""
    entry = {"body": body, "kind": kind, "prev": core_record.prev_of(lines[-1]) if lines else _ZERO,
             "schema": schema, "seq": len(lines), "session": session}
    return ledger_text({**entry, "sig": bytes(sign(ledger_message(I, entry))).hex()})


def _nat(v: Any) -> bool:
    return type(v) is int and v >= 0


def _outcome(w: Any) -> bool:
    return (isinstance(w, dict) and sorted(w) == ["complete", "root", "session"]
            and (w["root"] is None or _digest_hex(w["root"]))
            and _session_id(w["session"]) and type(w["complete"]) is bool)


def _ledger_entry(line: Any) -> dict | None:
    """A ledger line's entry when it is its own `ledger_text`, a record entry (`core_record.of_json`) and of the
    ledger's form, else None."""
    try:
        e = json.loads(line) if type(line) is str and len(line) <= MAX_LEDGER_LINE else None
        if e is None or ledger_text(e) != line or any(f"\\ud{c}" in line for c in "89abcdef"):
            return None
        canonical_json_text(_hexed(e))
        core_record.of_json(e)
    except (ValueError, TypeError, RecursionError):
        return None
    if e["schema"] not in LEDGER_TAGS or len(e["sig"]) < 8:
        return None
    b, kind, v2 = e["body"], e["kind"], e["schema"] == LEDGER_SCHEMA
    if kind in SESSION_KINDS:
        if not _session_id(e["session"]):
            return None
    elif kind in PERIOD_KINDS:
        if e["session"] is not None:
            return None
    else:
        return None
    if kind == "open":
        ok = ((_window(b.get("window")) if v2 else _v1_window(b.get("window"))) and _host(b.get("host"))
              and _digest_hex(b.get("pinned"), 40) and _digest_hex(b.get("key_sha512"))
              and (not v2 or type(b.get("by")) is str and type(b.get("for", "")) is str))
    elif kind == "reserve":
        ok = True
    elif kind == "end":
        files = b.get("files")
        ok = (type(b.get("closed")) is bool and isinstance(files, dict) and all(_digest_hex(v) for v in files.values())
              and (not v2 or b.get("record_leaf") is None or _nat(b["record_leaf"])))
    elif kind == "period-open":
        ok = _period(b.get("period")) and _period(b.get("lane")) and _digest_hex(b.get("key_sha512"))
    else:
        windows = b.get("windows")
        ok = (_period(b.get("period")) and _period(b.get("lane")) and type(windows) is list
              and all(_outcome(w) for w in windows) and type(b.get("complete")) is bool)
    return e if ok else None


def _checked(ledger: Any, key: ChallengerKey, verify: Verify) -> tuple[dict[str, dict], dict[str, dict]]:
    """`ledger` (its lines) as `verify_ledger` and `verify_periods` read it, under the pinned key `key`, each line first
    refused unless it is its own `ledger_text` of the ledger's form and names no other key."""
    if ledger is None:
        raise CoinError(NO_AUDIT)
    if type(ledger) is not list:
        raise CoinError(f"the auditor's ledger is malformed: {LEDGER_FORM}")
    pinned = hashlib.sha512(key.public).hexdigest()
    for i, line in enumerate(ledger):
        e = _ledger_entry(line)
        if e is None:
            raise CoinError(f"the auditor's ledger entry {i} is malformed: {LEDGER_FORM}")
        if e["kind"] in ("open", "period-open") and e["body"]["key_sha512"] != pinned:
            raise CoinError(f"the auditor's ledger entry {i} is signed under challenger key "
                            f"{e['body']['key_sha512'][:16]}..., not the window's pinned key {pinned[:16]}...")

    def verifies(message: bytes, signature: bytes) -> bool:
        try:
            return verify(key, message, signature) is True
        except ValueError:
            return False

    try:
        return _verify([line.encode() for line in ledger], key.public[:16], verifies)
    except ValueError as e:
        raise CoinError(f"the auditor's ledger is refused: {e}") from None


class Audited(NamedTuple):
    """The window's session as the auditor's ledger names it: its id, the host its open names, its prover, the SHA-512 of
    each file its end sealed, the leaf its record was signed at, its commitment's root, how many hellos it refused, the
    commit its open pinned the host's tree to (`pinned_session` holds it to the pins log and main), and the period
    holding the window with the cap and circuit its open committed (all None when no period holds it)."""
    session: str
    host: str
    prover: str
    files: dict[str, str]
    record_leaf: int
    commitment: str
    refused_hellos: int
    pinned: str = ""
    period: str | None = None
    cap: int | None = None
    circuit: str | None = None


def audited_session(ledger: Any, window: Any, key: ChallengerKey, verify: Verify) -> Audited:
    """The one challenger session the auditor's ledger `ledger` (its lines, None when the verifier has none) opens for
    window `window` (``{root}``), under the challenger key the window's deployment pins, `key`, and ends closed.
    Raises `CoinError` naming why when there is no ledger, a line isn't of the ledger's form or names another key, the
    ledger fails `verify_ledger`, or the window's session is missing, opened under v1, unended, ended unclosed or its end
    names no record leaf, commitment, refused hellos or sealed file the verifier reads, or the period holding the window
    committed no cap or circuit.  It checks no pin, so nothing credits a session read here alone: a credit path reads
    it through `pinned_session`."""
    return _audited(ledger, window, key, verify)[0]


def _audited(ledger: Any, window: Any, key: ChallengerKey, verify: Verify) -> tuple[Audited, dict]:
    """`audited_session`, and the body of the window's open."""
    if not _window(window):
        raise CoinError(f"the window's identity is malformed: {WINDOW_FORM}")
    windows, periods = _checked(ledger, key, verify)
    w = windows.get(window["root"])
    if w is None:
        raise CoinError("the auditor's ledger opens no challenger session for this window")
    o, end = w["open"], w["end"]
    s = o["session"]
    if o["schema"] != LEDGER_SCHEMA:
        raise CoinError(f"the auditor's ledger opens this window's session {s} under {o['schema']}, which names no prover "
                        "and seals no refused hellos")
    if end is None:
        raise CoinError(f"the auditor's ledger has no end for session {s}: the session never ended, or the ledger is "
                        "older than its end")
    b = end["body"]
    if b["closed"] is not True:
        raise CoinError(f"the auditor's ledger ends session {s} unclosed: its client didn't close it")
    if not (_nat(b.get("record_leaf")) and _digest_hex(b.get("commitment")) and _nat(b.get("refused_hellos"))
            and all(f in b["files"] for f in SEALED_FILES)):
        raise CoinError(f"the auditor's ledger's end of session {s} is malformed: {END_FORM}")
    name = next((n for n, p in periods.items() if s in p["windows"]), None)
    p = periods.get(name, {"cap": None, "circuit": None})
    if name is not None and (p["cap"] is None or p["circuit"] is None):
        raise CoinError(f"the auditor's ledger's period {name}, which holds session {s}, commits no cap or circuit")
    return Audited(s, o["body"]["host"], lane_of(o), dict(b["files"]), b["record_leaf"], b["commitment"],
                   b["refused_hellos"], o["body"]["pinned"], name, p["cap"], p["circuit"]), o["body"]


def ledger_head(heads: Any, key: ChallengerKey, verify: Verify) -> list[str]:
    """The auditor's ledger as the auditor host's operator published it: every head of `heads` (each its lines; None
    when the verifier has none)
    checked under the pinned key `key` (`audited_session`'s check of a ledger), any two extending one another, and of them
    the longest.  Raises `CoinError` naming why (and which head, when there are several), so a ledger forked, cut or
    rewritten between two publications is refused whichever head a verifier is shown."""
    if heads is None or heads == []:
        raise CoinError(NO_AUDIT)
    if type(heads) is not list:
        raise CoinError(HEADS_FORM)
    for i, head in enumerate(heads):
        try:
            _checked(head, key, verify)
        except CoinError as e:
            raise (e if len(heads) == 1 else CoinError(f"the auditor's published head {i} is refused: {e}")) from None
    for i, j in itertools.combinations(range(len(heads)), 2):
        if not (extends(heads[i], heads[j]) or extends(heads[j], heads[i])):
            raise CoinError(f"the auditor's published heads {i} and {j} don't extend one another: its ledger was forked "
                            "or rewritten")
    return max(heads, key=len)


def log_digest(lines: Sequence[str]) -> str:
    """The SHA-512 of the file whose lines are `lines` (each ended by a newline), in lowercase hex."""
    return hashlib.sha512("".join(line + "\n" for line in lines).encode()).hexdigest()


def session_named(audited: Audited, key: ChallengerKey, session: Any, verify: Verify
                  ) -> tuple[LiveCommitment, dict[str, tuple[bytes, ...]]]:
    """What the verifier takes from session `audited`'s files `session`, ``{"close", "hellos", "log", "record"}``
    (`session_of`; ``hellos`` the lines of its refused hellos): the coin commitment the challenger signed and the roots
    its release log shows it received, when its close names the ledger's session, its record the open's host, the record
    was signed at the leaf the ledger's end names over the commitment it names, and its release log and refused hellos
    are the ones the end sealed, as many hellos as the end counts.  Raises `CoinError` naming why."""
    if not isinstance(session, dict) or sorted(session) != ["close", "hellos", "log", "record"]:
        raise CoinError(SESSION_FORM)
    commitment, roots, sid = session_of(key, session["record"], session["log"], session["close"], verify)
    if sid != audited.session:
        raise CoinError(f"the challenger's session is {sid}, not {audited.session}, the one the auditor's ledger opens "
                        "for this window")
    record = ChallengerRecord.from_json(session["record"])
    if record.host != audited.host:
        raise CoinError(f"the challenger's record names host {record.host}, not {audited.host}, the one the auditor's "
                        "ledger opened its session on")
    q = int.from_bytes(record.signature[:4], "big")
    if q != audited.record_leaf or commitment.root.hex() != audited.commitment:
        raise CoinError(f"the challenger's record is signed at leaf {q} over commitment {commitment.root.hex()[:16]}..., "
                        f"not at leaf {audited.record_leaf} over {audited.commitment[:16]}..., the ones the auditor's "
                        "ledger's end names")
    if audited.files.get(LOG_FILE) != log_digest(session["log"]):
        raise CoinError("the challenger's release log is not the one the auditor's ledger sealed at its session's end")
    hellos = session["hellos"]
    if type(hellos) is not list or not all(type(line) is str for line in hellos):
        raise CoinError(HELLOS_FORM)
    if audited.files.get(HELLOS_FILE) != log_digest(hellos) or len(hellos) != audited.refused_hellos:
        raise CoinError(f"the challenger's refused hellos are not the {audited.refused_hellos} the auditor's ledger sealed "
                        "at its session's end")
    return commitment, roots


def trusted_session_of(ledger: Any, window: Any, key: ChallengerKey, pins: Any, main: Any, session: Any, verify: Verify
                       ) -> tuple[LiveCommitment, dict[str, tuple[bytes, ...]]]:
    """What the verifier takes from the challenger's session for window `window`: the one session the auditor's ledger
    opens for it under `key`, run from a commit the host's pins log `pins` records and the verifier's copy of main's
    first-parent history `main` holds (`pinned_session`), and that session's files `session` (`session_named`): the coin
    commitment the challenger signed and the roots its release log shows it received.  Raises `CoinError` naming why, so
    a window whose coins aren't that session's, or whose session ran any other tree, fails closed."""
    return session_named(pinned_session(ledger, window, key, pins, main, verify), key, session, verify)


def circuit_reason(period: str | None, circuit: Any, descriptor: Any) -> str | None:
    """Why a window doesn't count toward `period` (None: no period holds it), whose open committed `circuit`, when
    `descriptor` is the descriptor of the circuit the window's sessions prove (`DESCRIPTOR_FORM`); None when its SHA-512
    is that circuit.  One hash over the descriptor's bytes, never a walk over the circuit it unrolls to."""
    if type(descriptor) is not bytes:
        return DESCRIPTOR_FORM
    if period is None:
        return ("the auditor's ledger holds the window's session in no period, so it counts toward none: open the lane's "
                "period (period-open --cap N --circuit SHA512) before reserving the window")
    if not _digest_hex(circuit):
        return (f"period {period} commits no circuit, so no window counts toward it: open a period whose --circuit is "
                "the declared circuit's SHA-512")
    got = hashlib.sha512(descriptor).hexdigest()
    if got != circuit:
        return (f"the circuit the window's sessions prove (descriptor SHA-512 {got[:16]}...) is not {circuit[:16]}..., "
                f"the one period {period} committed: prove the period's declared circuit, or open a period whose "
                "--circuit is this descriptor's SHA-512")
    return None


def credited_session(ledger: Any, window: Any, key: ChallengerKey, pins: Any, main: Any, descriptor: Any,
                     verify: Verify) -> Audited:
    """`pinned_session`, counted toward its period: the window's session as the auditor's ledger names it, run from a
    commit the host's pins log `pins` records on main's first-parent line `main`, when `descriptor`, the descriptor of
    the circuit the window's sessions prove, hashes to the circuit its period committed (`circuit_reason`).  The pin
    first, then the circuit; raises `CoinError` naming why, so a window whose session ran another tree, or that proves
    another circuit, counts toward no period.  This is PoUW's per-window credit gate; the period-level pin hygiene over
    the whole period (every window no older than the period's open pin, and none past the freshness floor) is
    `pouw.credit`'s, which reads the pin this returns (`Audited.pinned`)."""
    audited = pinned_session(ledger, window, key, pins, main, verify)
    why = circuit_reason(audited.period, audited.circuit, descriptor)
    if why is not None:
        raise CoinError(why)
    return audited


def pin_line(commit: str, main: str, pinned_utc: str) -> str:
    """The host's pins log line for a pin of `commit`, which origin/main's commit `main` holds in its history, at
    `pinned_utc` (``YYYY-MM-DDTHH:MM:SSZ``), as the auditor writes it (`ledger_text`); ValueError when it isn't of that
    form."""
    line = ledger_text({"commit": commit, "main": main, "pinned_utc": pinned_utc})
    if _pin(line) is None:
        raise ValueError(PINS_FORM)
    return line


def pins_seal(pins: Sequence[str]) -> dict:
    """What an open records of the host's pins log `pins` (its lines) as it stands: ``{lines, sha512}``."""
    return {"lines": len(pins), "sha512": log_digest(pins)}


def _pin(line: Any) -> str | None:
    """A pins log line's commit when the line is its own `ledger_text` of the pins log's form, else None."""
    try:
        e = json.loads(line) if type(line) is str else None
        if e is None or ledger_text(e) != line:
            return None
    except (ValueError, TypeError, RecursionError):
        return None
    if not (isinstance(e, dict) and sorted(e) == ["commit", "main", "pinned_utc"] and _digest_hex(e["commit"], 40)
            and _digest_hex(e["main"], 40) and later(e["pinned_utc"], 0) == e["pinned_utc"]):
        return None
    return e["commit"]


def pins_reason(session: str, opened: dict, pins: Any) -> str | None:
    """Why session `session`, whose open's body is `opened`, didn't run main's code by the host's pins log `pins` (its
    lines, None when the verifier has none): the log malformed, the open sealing none of it, the log not beginning with
    the lines the open sealed, or none of those lines pinning the open's `pinned`; None when one does.  An open the
    ledger's check passed names its pin (`audited_session`); one that names none is refused as unrecorded."""
    if pins is None:
        return NO_PINS
    if type(pins) is not list:
        return f"the host's pins log is malformed: {PINS_FORM}"
    commits = []
    for i, line in enumerate(pins):
        commit = _pin(line)
        if commit is None:
            return f"the host's pins log line {i} is malformed: {PINS_FORM}"
        commits.append(commit)
    seal = opened.get("pins")
    if not (isinstance(seal, dict) and sorted(seal) == ["lines", "sha512"] and _nat(seal["lines"])
            and seal["lines"] >= 1 and _digest_hex(seal["sha512"])):
        return (f"the auditor's ledger's open of session {session} seals no pins log (pins: {{lines, sha512}}): open "
                "the window on a host whose gate seals its pins log in each open")
    n = seal["lines"]
    if len(pins) < n:
        return (f"the host's pins log has {len(pins)} lines, fewer than the {n} the open of session {session} sealed: "
                "take infra's newest published pins log")
    if log_digest(pins[:n]) != seal["sha512"]:
        return (f"the host's pins log is not the one the open of session {session} sealed: its first {n} lines were "
                "edited or rewritten")
    pinned = opened.get("pinned")
    if pinned not in commits[:n]:
        return (f"session {session} ran commit {pinned}, which the host's pins log doesn't record as a commit of main's "
                "history: repin the challenger host to a commit origin/main contains (vy-challenger-pin) and rerun the "
                "window there")
    return None


def first_parent_reason(session: str, pinned: Any, main: Any) -> str | None:
    """Why session `session`, run from commit `pinned`, didn't run a tree main's check passed by the verifier's own copy
    of main's first-parent history `main` (its commits, the auditor's ``git rev-list --first-parent origin/main``; None
    when the verifier has none): the copy missing or malformed, or `pinned` not on it; None when it is.  Ancestry is not
    enough: a branch commit merged into main is in main's history, but main's check ran on the merge, never on it."""
    if main is None:
        return NO_MAIN
    if type(main) is not list or not all(_digest_hex(c, 40) for c in main):
        return f"the verifier's copy of main's history is malformed: {MAIN_FORM}"
    if pinned not in main:
        return (f"session {session} ran commit {pinned}, which the host's pins log records but which is not on main's "
                "first-parent line in the verifier's copy (`git rev-list --first-parent origin/main`): a branch commit "
                "merged into main is in main's history but no tree main's check ran on; repin the challenger host to a "
                "commit on main's first-parent line (vy-challenger-pin) and rerun the window there")
    return None


def pinned_session(ledger: Any, window: Any, key: ChallengerKey, pins: Any, main: Any, verify: Verify) -> Audited:
    """`audited_session`, from main's code: the window's session as the auditor's ledger names it, when the host's pins
    log `pins` (its lines; the auditor host's operator's published one) records the open's pin (`pins_reason`) and the
    verifier's own copy of
    main's first-parent history `main` holds it (`first_parent_reason`).  Raises `CoinError` naming why, so a session
    the host ran from any other tree fails closed.  For PoUW this is what makes a credited window's coins come from a
    session the auditor opened on a commit main's check passed; `pouw.credit` layers on it that the pin is recent and no
    older than its period's open (the freshness floor and the period pin), which this function doesn't check.  Every path
    that credits a session reads it here: PoUW's `credited_session` and `trusted_session_of`, and memory-accounting's
    remote bandwidth audit (`remote_bandwidth.coins`).  `audited_session` alone checks no pin and credits
    nothing.  Only the open's pin is held: a repin of the host after the open neither helps nor harms the session, which
    is credited by the commit its open sealed, the pins log only growing past it."""
    audited, opened = _audited(ledger, window, key, verify)
    why = (pins_reason(audited.session, opened, pins)
           or first_parent_reason(audited.session, opened.get("pinned"), main))
    if why is not None:
        raise CoinError(why)
    return audited


class LiveChallenger:
    """The challenger's side of one epoch's coins.  It draws every coin and hm96 salt of the layout from `entropy` and
    commits to them at construction, before the epoch; `open` releases coin (`label`, `index`) once, after the verifier
    recorded `after`.  `entropy` has no default, so nothing here reads the operating system: a served challenger's is the
    Lean verifier of record's read of it (`IO.getRandomBytes`, A3).  `claim` is the claim instance naming `entropy`
    (`coin_claim`), None for an unnamed source, which a served challenger refuses (`served_claim`)."""

    def __init__(self, scheme: str, epoch: int, families: Sequence[tuple[str, int]],
                 entropy: Callable[[int], bytes]) -> None:
        families = tuple(tuple(f) for f in families)
        why = layout_problem(scheme, epoch, families)
        if why is not None:
            raise CoinError(why)
        self.claim = coin_claim(entropy)

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

    def open_many(self, requests: Sequence[tuple[str, int, bytes]]) -> list[LiveOpening]:
        """`open` of each (label, index, after) of `requests`, in order: the in-process twin of
        `remote.RemoteChallenger.open_many`, so a caller asks either challenger for several coins at once."""
        return [self.open(label, index, after) for label, index, after in requests]


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
        "after": <hex>, "released_at"}`.  It records the order; that the challenger kept it is `trusted-host`'s."""
        return tuple(self._log)

    def record(self, root: bytes) -> LiveOpening:
        request = self._root(root)
        opening = self.challenger.open(*request)
        self._release(request[1], opening)
        return opening

    def _root(self, root: bytes) -> tuple[str, int, bytes]:
        if len(self.roots) == self.challenger.commitment.family(self.label):
            raise CoinError(f"every {self.label} coin of the epoch is open")
        i = len(self.roots)
        self.roots.append(bytes(root))
        self._log.append({"event": "root", "label": self.label, "index": i, "root": bytes(root).hex(),
                          "received_at": self._clock()})
        return self.label, i, bytes(root)

    def _release(self, i: int, opening: LiveOpening) -> None:
        self.openings.append(opening)
        self._log.append({"event": "release", "label": self.label, "index": i, "after": opening.after.hex(),
                          "released_at": self._clock()})

    def reason(self) -> str | None:
        return order_reason(self.challenger.commitment, self.label, self.roots, self.openings)


def record_many(asks: Sequence[tuple[LiveRecord, bytes]]) -> list[LiveOpening]:
    """`LiveRecord.record` of each (record, root) of `asks`, the records one challenger's, their coins asked for in one
    `open_many` (one round trip to a remote challenger): every root recorded in turn, then the coins, then each release.
    A batch past a family's last coin records nothing; as with `record`, a coin the challenger doesn't release leaves
    its root recorded without it, and nothing is returned."""
    if not asks:
        return []
    challenger = asks[0][0].challenger
    if any(r.challenger is not challenger for r, _ in asks):
        raise CoinError("a batch's coins are one challenger's")
    for r, _ in asks:
        if len(r.roots) + sum(q is r for q, _ in asks) > challenger.commitment.family(r.label):
            raise CoinError(f"every {r.label} coin of the epoch is open")
    requests = [r._root(root) for r, root in asks]
    openings = challenger.open_many(requests)
    for (r, _), (_, i, _), opening in zip(asks, requests, openings):
        r._release(i, opening)
    return openings


__all__ = ["Audited", "BINDING", "CAP", "CIRCUIT", "CLOSE_FILE", "CLOSE_FORM", "CLOSE_ROLE", "COIN_BYTES",
           "COMMITMENT_FORM", "COMMITMENT_TAG", "ChallengerClose", "ChallengerKey", "ChallengerRecord", "CoinError",
           "DESCRIPTOR_FORM", "END_FORM", "HEADS_FORM", "HELLOS_FILE", "HELLOS_FORM", "KEY_FORM", "KEY_RECORD_FORM", "KEY_RECORD_SCHEMA",
           "LEDGER_FORM", "LEDGER_ROLE", "LEDGER_SCHEMA", "LEDGER_SCHEMA_V1", "LEDGER_TAGS", "LOG_FILE", "LOG_FORM",
           "LOG_MALFORMED", "LOG_TAG", "LiveChallenger", "LiveCommitment", "LiveOpening", "LiveRecord", "MAX_COINS",
           "MAIN_FORM", "MAX_LEDGER_LINE", "MAX_LOG_LINE", "MAX_LOG_LINES", "MAX_SESSION_ID", "MINUTES", "NO_AUDIT", "NO_CLOSE",
           "NO_MAIN", "NO_PINS", "NO_RECORD", "OPENING_FORM", "PERIOD_KINDS", "PINS_FORM", "PUBLIC_BYTES", "RECORD_FILE",
           "RECORD_FORM", "RECORD_ROLE",
           "RESERVED", "SEALED_FILES", "SESSION_FORM", "SESSION_KINDS", "Verify", "WINDOW_FORM", "audited_session",
           "cap_of", "circuit_of", "circuit_reason", "close_message", "credited_session", "extends", "first_parent_reason", "key_of_record", "lane_of", "later", "layout_problem",
           "leaf_of", "ledger_head", "ledger_line", "pin_line", "pinned_session", "pins_reason", "pins_seal",
           "ledger_message", "ledger_text", "live_domain", "live_leaf", "live_reason", "log_chain", "log_digest",
           "order_reason", "record_many", "record_message", "record_reason", "released_roots_of", "root_of",
           "session_named", "session_of", "trusted_commitment", "trusted_reason", "trusted_session_of", "verify_ledger",
           "verify_periods", "write_record"]
