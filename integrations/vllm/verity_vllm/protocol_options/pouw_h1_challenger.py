"""H1's coins from the proof service's challenger on its auditor host (`verity.core.protocols.sampled_proofs.remote`), and the
verifier's check that a served window's coins are that challenger's.

* **Serving** (`pouw_h1_window`): a window takes its salt, round and draw coins from one session of the auditor host's
  challenger, H1's opened on the window's open root and ncp-v2's after the weights' root.
* **Checking** (`trusted_rounds_reason`): the verifier reads the challenger's directory from the challenger's own run
  (``research fetch RUN --all``, its ``out/``), never from the prover's bundle, under its own challenger key (`pinned_key`),
  and the auditor's signed ledger, which opens and ends the one session of the window (`remote.trusted_session`): the coin
  commitment the challenger signed and the roots its signed release log shows it received. The window's coins must be
  that commitment's, its salt coin must have followed its own open root, the root the ledger opened its session on
  (PoUW's salt rule of record, `core_check.salt`, Lean `PouwCheck.saltCheck`: that root is `salt.window_open_root` of the
  window's weights' root, its round-0 request table's root, round 0's ``requests`` in its log, and the ledger's session,
  so a salt opened after another window's root or after the bare weights' root is refused), and its rounds pass PoUW's
  round rule of record
  (`core_check.rounds`, Lean `PouwCheck.roundsCheck`): each coin opened after its root, the roots the challenger's log
  shows in order. A window whose coins came from a challenger the prover holds (the in-process ``coins`` and ``rounds``
  of tests, or a host of the prover's), whose log no one signed, or whose ledger is signed under another key, is cut
  before the session's end or opens a second session for the window, is refused. The window's draw coin must have been
  opened once, after `pouw_h1_requests.draw_root` over its manifest, last receipts and every round's root and coin: a
  draw opened early, from another family, or never (a session closed at `finish` with no draw) is refused.
* **Sessions** (RT4-INT-17, the pouw setting ``session``): a lane reserves a session on infra's challenger gate
  (``reserve --peer NAME --scheme S --families JSON``, `benchmarks/live_challenger/auditor.py` on vy-challenger-1, under
  the lane's own gate key), and the window opens it on its root (``open --session S --root R``, H1's R its window-open
  root, opened from the serving process: `pouw_h1_window.Window`), and carries the gate's open answer (``commits["session"]``: its
  ``session``, ``seq``, ``root`` and coin ``commitment``,
  without its ``secret``; `session_of`, `carried`). The serving side's hello carries the answer's secret, and it refuses a
  challenger whose coin commitment isn't the answer's before its first coin (`pouw_h1_window.connect`). The gate appends the window's
  ``open`` entry (v2: ``window.root``) to its ledger, signed under the challenger key, before the session's first coin,
  and refuses a second one for the root, ever; the session's ``end`` entry seals the SHA-512 of each file of its
  ``out/``, the coins each family released, whether that was all of them (``complete``), the leaf its record was signed
  at and its commitment's root. `remote.trusted_session` reads that v2 ledger with core's check of record
  (`challenger.ledger_head`, which infra's ``auditor.py`` imports) and a signed close; `session_reason` reads it here too:
  `Ledger.verify` is infra's ``verify_ledger`` and ``verify_periods`` over one published head under the verifier's own
  key (`pinned_key`, its pin from config): chained, no v1 entry after a v2, its LMS leaves strictly increasing, each
  record's leaf between its session's open and end and no entry's, one session a root and one root a session, each
  two-step open bound to its session's reserve and inside its minutes, and each period's close listing exactly its
  windows and their outcomes; `Ledger.newest` reads every published head and refuses two that don't extend one another.
  `session_reason` holds a window to the one session the ledger opened for its Commit's root, ended, closed and complete
  with the window's commitment, reading that session's tree from the gate's ``fetch`` (never the prover's bundle), each
  file the one its end entry sealed, its key the verifier's, its record signed for the open entry's host and layout at
  the end's record leaf, and its release log the end's counts; its coins pass the checks above, the rounds by the rule
  `core_check.rounds` applies (`window.rounds_reason` and the log's roots in order).
* **Periods**: a lane's period is its windows between its ``period-open`` and ``period-close`` entries (session null),
  each window counted at its reserve (or a one-step v2 open) whose prover (``for``, else ``by``) is the lane.
  `period_reason` refuses a period the ledger doesn't close, one whose close isn't complete (an empty one, or a window
  never opened, never ended, unclosed or short of its coins), an accepted window that isn't one of its windows, and a
  window of it that has no accepted window (abandoned after its coins).
* **Credit** (`credit_reason`) is core's rules of record (`verity.core.protocols.pouw.credit`, PRF-132): the calls whose
  manifest rows' credit bits the draw coin followed (`credited`), only when every A value is the tree the round roots
  commit (`a_link`) and every drawn unit's proof was verified and accepted (`outcome`). The serving process's gate-passed
  count (`pouw_h1.H1Run.credit`) is never credit.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs import remote
from verity.core.protocols.sampled_proofs.challenger import (ChallengerKey, ChallengerRecord, CoinError, LiveCommitment,
                                                             LiveOpening, ledger_head, live_reason, trusted_commitment)
from verity.ml.commitments.identity import identity_digest_sha512

from verity_vllm.protocol_options import pouw_h1_requests as RQ

SCHEME = Wd.LIVE_SCHEME
#: infra's challenger ledger (`benchmarks/live_challenger/auditor.py`): its entries' schema (v2, after the host's v1
#: entries in one chain, which are `challenger.audited_session`'s), and the role and tag by schema of what an entry's
#: ``sig`` signs (`entry_message`)
LEDGER_V1 = Ch.LEDGER_SCHEMA_V1
LEDGER_SCHEMA = Ch.LEDGER_SCHEMA
LEDGER_ROLE = Ch.LEDGER_ROLE
ENTRY_TAGS = dict(Ch.LEDGER_TAGS)
#: the files of a session's ``out/`` the verifier reads, each of which its end entry must seal
SEALED = (remote.RECORD, remote.KEY, remote.RELEASES)
#: the pouw setting ``session``: the gate's open answer (``commitment`` from ``status SESSION`` when the answer's is
#: null), and what else the answer may carry; the window carries it without its ``secret`` (`carried`)
SESSION_FIELDS = ("session", "seq", "root", "commitment")
REPLY_FIELDS = ("host", "port", "peer", "by", "for", "reserve", "opened_by", "secret")
#: the ledger's period entries (session null), and what a two-step open carries unchanged from its reserve
PERIOD_KINDS = ("period-open", "period-close")
RESERVED = ("by", "for", "peer", "scheme", "epoch", "families", "minutes")
_ROOT = re.compile(r"^[0-9a-f]{128}$")
_SECRET = re.compile(r"^[0-9a-f]{64}$")
_HEX = re.compile(r"^(?:[0-9a-f]{2})+$")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
_SESSION = re.compile(r"s\d{8}T\d{6}Z-[0-9a-f]{4}")
_UTC = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")
_MINUTES = (1, 720)
_CAP = (1, 2 ** 24)
_ZERO = "0" * 128


def families(steps: int) -> str:
    """The challenger's ``FAMILIES`` for a window of `steps` rounds: a live epoch's salt, draw and round coins."""
    return json.dumps([list(f) for f in Wd.live_families(steps)])


def address(raw: Any) -> str | None:
    """The pouw setting ``challenger``: None, or ``HOST:PORT``; ValueError otherwise."""
    if raw is None:
        return None
    host, sep, port = raw.rpartition(":") if isinstance(raw, str) else ("", "", "")
    if not (sep and host and port.isdigit()):
        raise ValueError(f"pouw setting 'challenger' is the auditor host's challenger as HOST:PORT, not {raw!r}")
    return raw


def session_of(raw: Any) -> dict[str, Any] | None:
    """The pouw setting ``session``: None, or the gate's answer to the window's ``open`` (its ``session``, ledger ``seq``,
    window ``root`` (the served Commit's root) and coin ``commitment``'s root, each root 128 lowercase hex, and optionally
    the challenger's ``host``, ``port`` and ``peer``, the reserve's ``by``, ``for`` and ``seq`` (``reserve``), the
    ``opened_by`` and the hello's ``secret``, 64 lowercase hex); ValueError otherwise."""
    if raw is None:
        return None
    ok = (isinstance(raw, Mapping) and set(SESSION_FIELDS) <= set(raw) <= {*SESSION_FIELDS, *REPLY_FIELDS}
          and isinstance(raw["session"], str) and bool(raw["session"]) and type(raw["seq"]) is int and raw["seq"] >= 0
          and all(isinstance(raw[k], str) and bool(_ROOT.match(raw[k])) for k in ("root", "commitment"))
          and (raw.get("secret") is None or (isinstance(raw["secret"], str) and bool(_SECRET.match(raw["secret"])))))
    if not ok:
        raise ValueError("pouw setting 'session' is the challenger gate's open answer for the window, {'session', 'seq', "
                         "'root', 'commitment'} (the served Commit's root and the session's coin commitment root, 128 "
                         "lowercase hex; the commitment from 'status SESSION' when the answer's is null) and optionally "
                         f"{', '.join(repr(k) for k in REPLY_FIELDS[:-1])} and 'secret' (64 lowercase hex), not {raw!r}")
    return dict(raw)


def carried(given: Mapping[str, Any]) -> dict[str, Any]:
    """The session a window or a Commit carries: the gate's answer `given` without its hello ``secret``."""
    return {k: v for k, v in given.items() if k != "secret"}



def lms_verify(key: ChallengerKey, message: bytes, signature: bytes) -> bool:
    """`challenger.Verify`: LMS over SHA-512 at the verifier's challenger key's (64, W, H)."""
    from verity.ml.crypto import lms

    p = lms.Params(64, key.W, key.H)
    try:
        return lms.verify(p, key.public, message, lms.Signature.decode(p, signature))
    except ValueError:
        return False


def pinned_key(path: str | Path, public_sha512: str) -> ChallengerKey:
    """The verifier's challenger key: the key at `path` (a challenger's ``key.json``), refused (`CoinError`) unless the
    SHA-512 of its public key is `public_sha512`, the auditor's pin, so a key the prover wrote is never the verifier's."""
    try:
        key = ChallengerKey.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError) as e:
        raise CoinError(f"no challenger key at {path} ({e})") from e
    got = hashlib.sha512(key.public).hexdigest()
    if got != public_sha512.lower():
        raise CoinError(f"the challenger key at {path} is {got[:16]}..., not the auditor's pinned {public_sha512[:16]}...")
    return key


def _window_reason(commits: Mapping[str, Any], trusted: LiveCommitment, received: Mapping[str, Sequence[bytes]],
                   rounds: Callable[[list], str | None], session: str, root: str) -> str | None:
    """Why the window's coins (its H1 commitment `commits`) aren't those of the session `session`, which the auditor's
    ledger opened on `root` (hex), whose signed coin commitment is `trusted` and whose log shows it received `received`,
    its salt coin opened after the window's own open root (`core_check.salt`) and its round log held to `rounds`; None
    when they are."""
    for name in ("round_coins", "coins"):
        try:
            c = LiveCommitment.from_json(commits.get(name))
        except CoinError as e:
            return f"the window's {name} are not a live coin commitment ({e}), so not the verifier's challenger's"
        if c != trusted:
            return (f"the window's {name} {c.root.hex()[:16]} are not the coin commitment the verifier's challenger signed "
                    f"({trusted.root.hex()[:16]}): coins from a challenger the verifier doesn't hold")
    given = commits.get("session")
    named = given.get("session") if isinstance(given, Mapping) else None
    if named != session:
        return f"the window records session {named!r}, not {session}, the one the auditor's ledger opened on its root"
    refused = CC.salt(trusted.to_json(), session, root, commits.get("weights_root"), commits.get("requests_root"),
                      [r.hex() for r in received.get(Wd.SALT, ())], commits.get("salt_opening"))
    if refused is not None:
        return f"the window's salt: {refused[1]}"
    log = list(commits.get("rounds") or ())
    if len(log) != commits.get("steps"):
        return f"the window's round log has {len(log)} rounds, not its {commits.get('steps')} steps"
    if [x.get("round") for x in log] != list(range(len(log))):
        return "the round log is not rounds 0, 1, ... in order"
    if (log[0].get("requests") if log else None) != commits.get("requests_root"):
        return "round 0's request table is not the one the window's open root is over (its requests_root)"
    why = rounds(log)
    if why is not None:
        return f"the window's rounds: {why}"
    return draw_reason(commits, trusted, received.get(Wd.DRAW, ()))


def trusted_rounds_reason(commits: Mapping[str, Any], directory: str | Path, heads: Sequence[str | Path],
                          window: Mapping[str, Any], key: ChallengerKey, verify: Any = lms_verify) -> str | None:
    """Why the window's coins (its H1 commitment `commits`: ``coins``, ``round_coins``, ``salt_opening``, ``weights_root``,
    ``requests_root``, ``session``, ``steps``, the round log ``rounds`` and the ``draw``) aren't the verifier's own
    challenger's, read from `directory`, the session the auditor's signed ledger opens and ends for `window` (``{root}``,
    the window's open root), under `key`, the challenger key the deployment pins, the ledger the longest of the heads
    infra published at `heads` (each a ``windows.jsonl``), any two extending one another; None when they are. Fails
    closed: a missing, malformed or unsigned record, log, hello log or ledger, a ledger signed under another key, forked
    or older than the session's end, or a window it opens no one session for, trusts nothing (`remote.trusted_session`)."""
    try:
        trusted, received = remote.trusted_session(directory, heads, window, key, verify)
        lines = ledger_head([remote.ledger_lines(p) for p in heads], key, verify)
        files = remote.session_files(directory)
        sid = Ch.audited_session(lines, window, key, verify).session
    except CoinError as e:
        return f"the challenger's session is not trusted: {e}"

    def rounds(log: list) -> str | None:
        refused = CC.rounds(key.to_json(), lines, window, files, [x.get("root") for x in log],
                            [x.get("coin") for x in log], verify)
        return None if refused is None else refused[1]

    return _window_reason(commits, trusted, received, rounds, sid, window["root"])


def draw_reason(commits: Mapping[str, Any], trusted: LiveCommitment, received: Sequence[bytes]) -> str | None:
    """Why the window's draw coin isn't its one draw coin 0, opened after its `pouw_h1_requests.draw_root` (over the
    manifest's ``calls_root``, the last round's receipts and every round's root and coin), given the draw roots the
    challenger's log shows it received; None when it is."""
    if not received:
        return "the challenger's session closed with no draw coin opened, so the window has no draw of the challenger's"
    draw = commits.get("draw") or {}
    try:
        root = RQ.draw_root(bytes.fromhex(commits["calls_root"]), bytes.fromhex(draw["receipts"]),
                            list(commits.get("rounds") or ()))
    except (KeyError, TypeError, ValueError) as e:
        return f"the window's draw root is malformed ({e})"
    if draw.get("root") != root.hex():
        return "the window's draw root is not over its manifest, its last round's receipts and every round's root and coin"
    if tuple(received) != (root,):
        return ("the challenger's log does not show the window's draw root as the one root its draw coin followed: a draw "
                "opened before the window's rounds, or after another root")
    try:
        opening = LiveOpening.from_json(draw.get("coin"))
    except CoinError as e:
        return f"the window's draw coin is malformed ({e})"
    why = live_reason(trusted, opening, Wd.DRAW, 0, root)
    return None if why is None else f"the window's draw coin: {why}"


def _field(b: bytes) -> bytes:
    return len(b).to_bytes(8, "big") + b


def entry_message(I: bytes, entry: Mapping[str, Any]) -> bytes:
    """What a ledger entry's ``sig`` signs, infra's ``entry_message``: SHA-512([LEDGER_ROLE] || I || [identity_digest_sha512(
    ENTRY_TAGS[schema], entry without sig)]), each [x] prefixed with its length (u64), under the key whose identifier is
    `I`; a v1 entry's is `challenger.ledger_message`."""
    if entry.get("schema") == LEDGER_V1:
        return Ch.ledger_message(I, dict(entry))
    if len(I) != 16:
        raise CoinError(f"an LMS identifier is 16 bytes, not {len(I)}")
    tag = ENTRY_TAGS.get(entry.get("schema"))
    if tag is None:
        raise CoinError(f"a ledger entry's schema is one of {sorted(ENTRY_TAGS)}, not {entry.get('schema')!r}")
    body = {k: v for k, v in entry.items() if k != "sig"}
    return hashlib.sha512(_field(LEDGER_ROLE.encode()) + I + _field(identity_digest_sha512(tag, body))).digest()


def leaf_of(sig: Any) -> int:
    """The LMS leaf an entry's or a record's signature (lowercase hex) was signed at: its first 4 bytes; ValueError for
    anything else."""
    if not (isinstance(sig, str) and len(sig) >= 8 and _HEX.match(sig)):
        raise ValueError("an LMS signature is lowercase hex, at least 4 bytes")
    return int(sig[:8], 16)


def root_of(entry: Mapping[str, Any]) -> str:
    """An open entry's window: its root (v2's ``window.root``, v1's ``window.commit``), 128 lowercase hex; ValueError
    otherwise."""
    try:
        w = entry["body"]["window"]
        root = w["root"] if entry["schema"] == LEDGER_SCHEMA else w["commit"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"an open names its window ({e!r})") from e
    if not (isinstance(root, str) and _ROOT.match(root)):
        raise ValueError(f"a window root is 128 lowercase hex digits, not {root!r}")
    return root


def lines_of(path: str | Path) -> list[bytes]:
    """A published ledger head's lines (its ``windows.jsonl``), `CoinError` when it can't be read."""
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        raise CoinError(f"no challenger ledger at {path} ({e})") from e
    return [line for line in data.split(b"\n") if line]


def _later(t: str, minutes: int) -> str:
    """`t` (``%Y-%m-%dT%H:%M:%SZ``) plus `minutes`: a reserve's expiry, as infra's ``later`` computes it."""
    return (datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ") + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _named(v: Any) -> bool:
    return type(v) is str and bool(_NAME.fullmatch(v))


def _reserve_ok(b: Mapping[str, Any]) -> bool:
    return (_named(b.get("by")) and ("for" not in b or _named(b["for"])) and type(b.get("minutes")) is int
            and _MINUTES[0] <= b["minutes"] <= _MINUTES[1] and type(b.get("started_utc")) is str
            and bool(_UTC.fullmatch(b["started_utc"])) and b.get("expires_utc") == _later(b["started_utc"], b["minutes"]))


def _period(i: int, kind: str, schema: Any, e: Mapping[str, Any], periods: dict[str, dict[str, Any]],
            current: dict[str, str], windows: Mapping[str, Mapping[str, Any]], by_session: Mapping[str, str]) -> None:
    """`Ledger.verify`'s period-open or period-close entry `e` (entry `i`), recorded in `periods` and `current` (each
    lane's open period), its close listing exactly its windows and their outcomes; `CoinError` otherwise."""
    body = e["body"]
    name, lane = body.get("period"), body.get("lane")
    if schema != LEDGER_SCHEMA or not (_named(name) and _named(lane)):
        raise CoinError(f"ledger entry {i} is a {kind} that doesn't name its period and its lane (v2)")
    if kind == "period-open":
        if name in periods:
            raise CoinError(f"ledger entry {i} opens period {name} a second time: a period's name is used once")
        if lane in current:
            raise CoinError(f"ledger entry {i} opens period {name} while lane {lane}'s period {current[lane]} is open")
        cap, circuit = body.get("cap"), body.get("circuit")
        if (cap is not None or circuit is not None) and not (
                type(cap) is int and _CAP[0] <= cap <= _CAP[1] and type(circuit) is str and _ROOT.fullmatch(circuit)):
            raise CoinError(f"ledger entry {i}: period {name}'s cap is a positive integer and its circuit a "
                            "SHA-512 in lowercase")
        periods[name] = {"lane": lane, "open": e, "close": None, "windows": [], "cap": cap, "circuit": circuit}
        current[lane] = name
    else:
        p = periods.get(name)
        if p is None or p["close"] is not None or p["lane"] != lane:
            raise CoinError(f"ledger entry {i} closes period {name}, which isn't lane {lane}'s open period")
        outcomes = [{"root": by_session.get(s), "session": s,
                     "complete": s in by_session and windows[by_session[s]]["end"] is not None
                     and windows[by_session[s]]["end"]["body"].get("complete") is True} for s in p["windows"]]
        if (body.get("windows") != outcomes
                or body.get("complete") is not (bool(outcomes) and all(w["complete"] for w in outcomes))):
            raise CoinError(f"ledger entry {i} closes period {name} without listing exactly its windows and "
                            "their outcomes")
        p["close"] = e
        del current[lane]


# TODO(#2953, VLLM-61): with core's `ledger_head` and `audited_session` on main, this ledger mirror, `evidence` and
# `released_sealed` go; the module keeps vllm's checks (a session a window, the commitment first, the slot tables).
@dataclass(frozen=True)
class Ledger:
    """Infra's challenger ledger (`LEDGER_SCHEMA`, after the host's `LEDGER_V1` entries), verified under the verifier's
    challenger `key`: each window root to its one session's ``reserve`` entry (None for a one-step open), ``open`` entry
    and ``end`` entry (None while it hasn't ended), and each period's name to its ``lane``, ``open`` and ``close`` entries
    (``close`` None while it is open), its ``windows`` (session ids, in order) and its ``cap`` and ``circuit``."""

    key: ChallengerKey
    windows: Mapping[str, Mapping[str, Any]]
    periods: Mapping[str, Mapping[str, Any]]

    @classmethod
    def verify(cls, lines: Sequence[bytes], key: ChallengerKey, verify: Any = lms_verify) -> Ledger:
        """The ledger of `lines` (one head's raw lines, as published), infra's ``verify_ledger`` and ``verify_periods``
        under `key`, refused (`CoinError`) at its first entry that isn't entry i of a v1-then-v2 ledger chained to its
        predecessor and signed under `key` at a leaf after the last entry's: a reserve, an open or a period-open under
        another key's pin; a reserve that isn't v2 and first in its session or whose expiry isn't its start plus its
        minutes; an open of no root, a second open for a root or a session, a two-step open that doesn't bind its
        session's reserve, carry its request, name its prover as opener and fall inside its minutes and before its
        period's close, or a one-step v2 open after the first reserve; an end with no open or a second end, or a v2 end
        whose record leaf isn't between its open's and its end's (or is another end's or an entry's); a period name used
        twice, a lane's second open period, a cap or circuit off form, a window past its period's cap, a close of a
        period that isn't its lane's open one or that doesn't list exactly its windows and their outcomes; or another
        kind. Fails closed: one bad entry trusts nothing."""
        pin = hashlib.sha512(key.public).hexdigest()
        windows: dict[str, dict[str, Any]] = {}
        opened_at: dict[str, int] = {}
        by_session: dict[str, str] = {}
        reserves: dict[str, dict[str, Any]] = {}
        periods: dict[str, dict[str, Any]] = {}
        current: dict[str, str] = {}
        leaves: set[int] = set()
        records: dict[int, int] = {}
        prev, last, v2, two_step = _ZERO, -1, False, False

        def join(i: int, lane: str | None, sid: str) -> str | None:
            period = current.get(lane) if lane is not None else None
            if period is not None:
                p = periods[period]
                if p["cap"] is not None and len(p["windows"]) >= p["cap"]:
                    raise CoinError(f"ledger entry {i} puts window {len(p['windows'])} in period {period}, past its cap "
                                    f"{p['cap']}")
                p["windows"].append(sid)
            return period

        for i, line in enumerate(lines):
            try:
                e = json.loads(line)
                if type(e) is not dict:
                    raise TypeError(f"an entry is an object, not {e!r}")
                kind, sid, body, q = e["kind"], e["session"], e["body"], leaf_of(e["sig"])
                if not ((sid is None if kind in PERIOD_KINDS else isinstance(sid, str)) and isinstance(body, dict)):
                    raise TypeError("a session's entry has a session (a string), a period's none, and each a body (an "
                                    "object)")
            except (KeyError, TypeError, ValueError) as err:
                raise CoinError(f"ledger entry {i} is malformed ({err})") from err
            schema = e.get("schema")
            if (schema not in ENTRY_TAGS or (v2 and schema == LEDGER_V1) or type(e.get("seq")) is not int
                    or e["seq"] != i or e.get("prev") != prev):
                raise CoinError(f"ledger entry {i} is not entry {i} of a {LEDGER_SCHEMA} ledger chained to its predecessor"
                                f" (v1 entries, then v2)")
            v2 = v2 or schema == LEDGER_SCHEMA
            if not verify(key, entry_message(key.public[:16], e), bytes.fromhex(e["sig"])):
                raise CoinError(f"ledger entry {i}'s signature doesn't verify under the verifier's challenger key")
            if q <= last:
                raise CoinError(f"ledger entry {i} is signed at LMS leaf {q}, not after the last entry's {last}: a key "
                                "state restored or forked")
            if kind in ("reserve", "open", "period-open") and body.get("key_sha512") != pin:
                raise CoinError(f"ledger entry {i} ({kind} of {sid or body.get('period')}) is under the key pin "
                                f"{body.get('key_sha512')}, not the verifier's {pin[:16]}...")
            if kind == "reserve":
                if schema != LEDGER_SCHEMA or not _SESSION.fullmatch(sid) or sid in reserves or sid in by_session:
                    raise CoinError(f"ledger entry {i} reserves session {sid}, which is not a v2 entry first in its "
                                    "session")
                if not _reserve_ok(body):
                    raise CoinError(f"ledger entry {i} reserves session {sid} without its prover, its minutes, its start "
                                    "and its expiry (its start plus its minutes)")
                two_step = True
                reserves[sid] = {"entry": e, "period": join(i, body.get("for") or body["by"], sid)}
            elif kind == "open":
                try:
                    root = root_of(e)
                except ValueError as err:
                    raise CoinError(f"ledger entry {i} opens session {sid} for no window ({err})") from err
                if root in windows:
                    raise CoinError(f"ledger entry {i} opens a second session ({sid}, after "
                                    f"{windows[root]['open']['session']}) for root {root}: one window, one session")
                if sid in by_session:
                    raise CoinError(f"ledger entry {i} opens session {sid} for a second window")
                reserved = None
                if schema == LEDGER_SCHEMA and "reserve" in body:
                    r = reserves.get(sid)
                    if r is None or type(body["reserve"]) is not int or r["entry"]["seq"] != body["reserve"]:
                        raise CoinError(f"ledger entry {i} opens session {sid} binding no reserve of it")
                    reserved, rb = r["entry"], r["entry"]["body"]
                    if any(body.get(k) != rb.get(k) for k in RESERVED):
                        raise CoinError(f"ledger entry {i} opens session {sid} without its reserve's request")
                    if body.get("opened_by") not in {rb["by"], rb.get("for")} - {None}:
                        raise CoinError(f"ledger entry {i} opens session {sid} by {body.get('opened_by')!r}, not its "
                                        "reserve's prover")
                    t = body.get("started_utc")
                    if not (type(t) is str and _UTC.fullmatch(t) and rb["started_utc"] <= t <= rb["expires_utc"]):
                        raise CoinError(f"ledger entry {i} opens session {sid} outside its reserve's minutes")
                    if r["period"] is not None and periods[r["period"]]["close"] is not None:
                        raise CoinError(f"ledger entry {i} opens session {sid} after its period {r['period']} closed")
                elif schema == LEDGER_SCHEMA and two_step:
                    raise CoinError(f"ledger entry {i} opens session {sid} without a reserve, after the ledger's first "
                                    "reserve")
                elif schema == LEDGER_SCHEMA:
                    lane = body.get("for") or body.get("by")
                    if not _named(lane):
                        raise CoinError(f"ledger entry {i} opens session {sid} for no prover")
                    join(i, lane, sid)
                windows[root] = {"reserve": reserved, "open": e, "end": None}
                opened_at[root], by_session[sid] = q, root
            elif kind in PERIOD_KINDS:
                _period(i, kind, schema, e, periods, current, windows, by_session)
            elif kind == "end":
                root = by_session.get(sid)
                if root is None or windows[root]["end"] is not None:
                    raise CoinError(f"ledger entry {i} ends session {sid}, which has no open entry or has ended")
                r = body.get("record_leaf") if schema == LEDGER_SCHEMA else None
                if r is not None:
                    if type(r) is not int or not opened_at[root] < r < q or r in records:
                        raise CoinError(f"ledger entry {i} ends session {sid} with its record signed at leaf {r!r}, not "
                                        "between its open's and its end's leaves, once")
                    records[r] = i
                windows[root]["end"] = e
            else:
                raise CoinError(f"ledger entry {i} is of kind {kind!r}, not reserve, open, end, period-open or "
                                "period-close")
            leaves.add(q)
            prev, last = hashlib.sha512(line).hexdigest(), q
        if reused := sorted(leaves & set(records)):
            raise CoinError(f"ledger entry {records[reused[0]]}'s record leaf {reused[0]} is also an entry's leaf")
        return cls(key, windows, periods)

    @classmethod
    def load(cls, path: str | Path, key: ChallengerKey, verify: Any = lms_verify) -> Ledger:
        return cls.verify(lines_of(path), key, verify)

    @classmethod
    def newest(cls, heads: Sequence[Sequence[bytes]], key: ChallengerKey, verify: Any = lms_verify) -> Ledger:
        """The ledger of the longest of `heads` (every published ``challenger-ledger/v2`` head, each one's lines), refused
        (`CoinError`) unless each verifies under `key` and every two extend one another (infra's ``extends``): a head cut,
        edited or forked from another is refused, so the newest label alone is never trusted."""
        if not heads:
            raise CoinError("no published challenger ledger head")
        ordered = sorted(((list(h), cls.verify(h, key, verify)) for h in heads), key=lambda h: len(h[0]))
        for (shorter, _s), (longer, _l) in zip(ordered, ordered[1:]):
            if longer[:len(shorter)] != shorter:
                raise CoinError(f"the published ledger heads of {len(shorter)} and {len(longer)} entries don't extend one "
                                "another: a forked ledger")
        return ordered[-1][1]

    def window(self, root: str) -> Mapping[str, Any] | None:
        return self.windows.get(root)


#: ``fetch(session)``: the session's tree on the auditor's side (the gate's ``fetch SESSION``, unpacked: its
#: ``request.json`` and ``out/``), never the prover's bundle
Fetch = Callable[[str], "str | Path | None"]


def _sealed(path: Path) -> str:
    return hashlib.sha512(path.read_bytes()).hexdigest()


_ROOT_EVENT = ["event", "index", "label", "received_at", "root"]
_RELEASE_EVENT = ["after", "event", "index", "label", "released_at"]


def released_sealed(path: str | Path, commitment: LiveCommitment) -> dict[str, tuple[bytes, ...]]:
    """The roots the release log at `path` of an infra v2 session (JSON lines, its times floats, no signed close: the
    sealing end entry stands in for it) shows the challenger received, by family in order, its head naming `commitment`,
    each root followed by its coin's release; `CoinError` for a missing, unreadable or malformed log, or another
    commitment's."""
    try:
        events = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    except FileNotFoundError:
        raise CoinError(f"no challenger release log at {path}") from None
    except (OSError, ValueError) as e:
        raise CoinError(f"the challenger release log at {path} is unreadable ({e})") from e
    if not events or events[0] != {"event": "commitment", "root": commitment.root.hex()}:
        raise CoinError(f"the challenger release log at {path} is not coin commitment {commitment.root.hex()[:16]}'s")
    roots: dict[str, list[bytes]] = {label: [] for label, _ in commitment.families}
    pairs = events[1:]
    if len(pairs) % 2:
        raise CoinError(f"the challenger release log at {path} ends on a root without its release")
    for root, release in zip(pairs[::2], pairs[1::2]):
        if (not isinstance(root, dict) or not isinstance(release, dict) or sorted(root) != _ROOT_EVENT
                or sorted(release) != _RELEASE_EVENT or root["event"] != "root" or release["event"] != "release"
                or type(root["label"]) is not str or root["label"] not in roots or type(root["index"]) is not int
                or root["index"] != len(roots[root["label"]]) or type(release["index"]) is not int
                or (release["label"], release["index"]) != (root["label"], root["index"])
                or release["after"] != root["root"] or type(root["root"]) is not str or not _ROOT.match(root["root"])):
            raise CoinError(f"the challenger release log at {path} is malformed")
        roots[root["label"]].append(bytes.fromhex(root["root"]))
    return {label: tuple(r) for label, r in roots.items()}


def evidence(window: Mapping[str, Any], fetch: Fetch, key: ChallengerKey, verify: Any = lms_verify) -> Path:
    """The ``out/`` of `window`'s session as `fetch` gives it, refused (`CoinError`) unless the session ended, each file its
    end entry seals is the one it sealed (`SEALED` among them), its key is `key` (the one the ledger's open pins) and its
    record is signed for the open entry's host and layout (scheme, epoch, families); for a v2 end, the record's
    commitment root and LMS leaf are the end's ``commitment`` and ``record_leaf``, and the coins its release log shows each
    family released (`released_sealed`, under the commitment its record signs under `key`) are the end's ``released``."""
    opened, end = window["open"], window["end"]
    sid, body = opened["session"], opened["body"]
    if end is None:
        raise CoinError(f"session {sid} never ended: the ledger has its open entry and no end, so nothing seals its out/")
    tree = fetch(sid)
    if tree is None:
        raise CoinError(f"session {sid} has no tree on the auditor's side (the gate's fetch)")
    out, files = Path(tree) / "out", end["body"].get("files")
    if not isinstance(files, Mapping) or not set(SEALED) <= set(files):
        raise CoinError(f"session {sid}'s end entry doesn't seal its {', '.join(SEALED)}")
    for name, digest in sorted(files.items()):
        try:
            got = _sealed(out / name)
        except OSError as e:
            raise CoinError(f"session {sid}'s {name} is unreadable ({e})") from e
        if got != digest:
            raise CoinError(f"session {sid}'s {name} is not the one its end entry sealed")
    try:
        theirs = ChallengerKey.from_json(json.loads((out / remote.KEY).read_text(encoding="utf-8")))
        doc = json.loads((out / remote.RECORD).read_text(encoding="utf-8"))
        record, leaf = ChallengerRecord.from_json(doc), leaf_of(doc["signature"])
    except (OSError, KeyError, TypeError, ValueError) as e:
        raise CoinError(f"session {sid}'s key or record is malformed ({e})") from e
    if theirs != key:
        raise CoinError(f"session {sid}'s key is not the verifier's challenger key, the one its open entry pins")
    if record.host != body.get("host"):
        raise CoinError(f"session {sid}'s challenger record is signed for host {record.host}, not {body.get('host')}, the "
                        "one its open entry names")
    layout = json.loads(json.dumps({k: record.commitment.to_json()[k] for k in ("scheme", "epoch", "families")}))
    if layout != {k: body.get(k) for k in layout}:
        raise CoinError(f"session {sid}'s coin commitment is {layout}, not the layout its open entry names")
    if end["schema"] == LEDGER_SCHEMA:
        sealed = end["body"]
        if record.commitment.root.hex() != sealed.get("commitment"):
            raise CoinError(f"session {sid}'s record commits to coins {record.commitment.root.hex()[:16]}..., not "
                            f"{str(sealed.get('commitment'))[:16]}..., the commitment its end entry records")
        if leaf != sealed.get("record_leaf"):
            raise CoinError(f"session {sid}'s record is signed at LMS leaf {leaf}, not {sealed.get('record_leaf')}, the "
                            "leaf its end entry records")
        received = released_sealed(out / remote.RELEASES, trusted_commitment(out / remote.RECORD, key, verify))
        released = {label: len(r) for label, r in received.items()}
        if released != sealed.get("released"):
            raise CoinError(f"session {sid}'s release log shows {released} coins released, not {sealed.get('released')}, "
                            "the count its end entry records")
    return out


def held(given: Any, ledger: Ledger, commit: str | None = None) -> tuple[str | None, Mapping[str, Any] | None]:
    """The window of `ledger` that the session `given` (the gate's answer a window or a Commit carries, `session_of`) is,
    ended with its client closing it after every coin of its families was released, under the answer's coin commitment,
    and None; or why it isn't and None. `commit` is the Commit root the verifier holds for it, which the answer's must
    be."""
    try:
        given = session_of(given)
    except ValueError as e:
        return f"the window's challenger session is malformed ({e})", None
    if given is None:
        return "the window carries no challenger session (the pouw setting 'session'), so no ledger entry binds its coins", None
    sid, root = given["session"], given["root"]
    if commit is not None and root != commit:
        return f"the window's session is for Commit {root}, not the window's Commit {commit}", None
    window = ledger.window(root)
    if window is None:
        return f"the auditor's ledger has no session for Commit {root}, so the window's coins are no session's of record", None
    opened, end = window["open"], window["end"]
    if opened["session"] != sid:
        return (f"the window's session {sid} is not {opened['session']}, the one the auditor's ledger opened for Commit "
                f"{root}"), None
    if opened["seq"] != given["seq"]:
        return f"the window's session {sid} is ledger entry {given['seq']} in its reply, but entry {opened['seq']} opened it", None
    if end is None:
        return f"session {sid} never ended (the auditor's ledger has its open entry and no end): a refused window", None
    sealed = end["body"]
    if sealed.get("closed") is not True:
        return f"session {sid} ended without its client closing it ({sealed.get('ended')}): a refused window", None
    if sealed.get("complete") is not True:
        return (f"session {sid} ended without releasing every coin of its families (released {sealed.get('released')}): "
                "a refused window"), None
    if sealed.get("commitment") != given["commitment"]:
        return (f"session {sid}'s end entry records coin commitment {str(sealed.get('commitment'))[:16]}..., not "
                f"{given['commitment'][:16]}..., the window's (the gate's answer): a refused window"), None
    return None, window


def session_reason(commits: Mapping[str, Any], ledger: Ledger, fetch: Fetch, registered: bytes | None = None,
                   commit: str | None = None, verify: Any = lms_verify) -> str | None:
    """Why the window (its H1 commitment `commits`) isn't held to the one session the auditor's `ledger` opened for its
    Commit's root, ended closed and complete under the window's commitment (`held`), with that session's coins
    (its ``out/`` as `fetch` gives it, `evidence`, checked as `trusted_rounds_reason` checks a session's, over its
    unsigned log, `released_sealed`); None when it is. `registered`
    is the weights' root the period registered and `commit` the window's Commit root as the verifier holds it, which the
    window's must be. Fails closed."""
    if commits.get("coin_source") != "challenger":
        return ("the window's coins were held in the serving process, not drawn by the auditor host's challenger (the pouw "
                "setting 'challenger'), so no session binds them")
    why, window = held(commits.get("session"), ledger, commit)
    if why is not None:
        return why
    if registered is not None and commits.get("weights_root") != bytes(registered).hex():
        return "the window's weights' root is not the one the period registered"
    sid = window["open"]["session"]
    try:
        out = evidence(window, fetch, ledger.key, verify)
    except CoinError as e:
        return f"session {sid}'s evidence is not trusted: {e}"
    try:
        trusted = trusted_commitment(out / remote.RECORD, ledger.key, verify)
        received = released_sealed(out / remote.RELEASES, trusted)
    except CoinError as e:
        return f"the challenger's session is not trusted: {e}"

    def rounds(log: list) -> str | None:
        why = RQ.rounds_reason(log, trusted)
        if why is not None:
            return why
        try:
            roots = tuple(bytes.fromhex(x["root"]) for x in log)
        except (KeyError, TypeError, ValueError):
            return "the round roots are not a list of SHA-512 digests"
        if received.get(Wd.ROUND, ()) != roots:
            return "the round roots are not the roots the challenger's log shows it received, in order"
        return None

    return _window_reason(commits, trusted, received, rounds, sid, root_of(window["open"]))


def _incomplete(ledger: Ledger, outcome: Mapping[str, Any]) -> str:
    if outcome["root"] is None:
        return "reserved and never opened"
    end = ledger.windows[outcome["root"]]["end"]
    if end is None:
        return f"Commit {outcome['root']}, never ended"
    return f"Commit {outcome['root']}, ended {end['body'].get('ended')!r}, released {end['body'].get('released')}"


def period_reason(ledger: Ledger, period: str, accepted: Sequence[str]) -> list[str]:
    """Why the auditor's `ledger` doesn't credit the lane's period `period`, given `accepted` (the session ids of its
    windows that `session_reason` accepted), one reason each; [] when it does. Refused: a period the ledger has no
    period-open for, or no period-close (its windows aren't all known); an accepted window that isn't one of the
    period's (its lane's reserves and one-step opens between its period-open and its period-close); a close that isn't
    complete (no window, or a window never opened, never ended, unclosed or short of its coins: every window counts); and
    a window of it with no accepted window, which ended complete, so released every coin: a window abandoned after its
    coins, which a prover keeps only when it liked them."""
    p = ledger.periods.get(period)
    if p is None:
        return [f"the auditor's ledger has no period {period}"]
    out = [f"session {s}, an accepted window, is not one of period {period}'s windows ({p['windows']}, lane {p['lane']}'s "
           "between its period-open and its period-close)" for s in accepted if s not in p["windows"]]
    if p["close"] is None:
        return out + [f"period {period} has no period-close in the auditor's ledger, so its windows aren't all known: an "
                      "open period credits nothing"]
    listed = p["close"]["body"]["windows"]
    if not listed:
        out.append(f"period {period} closed with no window")
    for w in listed:
        if not w["complete"]:
            out.append(f"period {period}'s window {w['session']} is incomplete ({_incomplete(ledger, w)}): a refused "
                       "window refuses its period")
        elif w["session"] not in accepted:
            out.append(f"period {period}'s window {w['session']} (Commit {w['root']}) ended complete and has no accepted "
                       "window: a window abandoned after its coins")
    return out


def manifest_reason(record: Mapping[str, Any]) -> str | None:
    """Why a window's record (`pouw_h1.CALLS_FILE`) isn't one manifest under its coins: the Program salt from its salt
    coin over its weights' root, each round's root over its parts, each call's A table under its own round's root (step
    s's calls digest over the record's rows of step s, their tables' roots among them: `pouw_h1_requests.tables_reason`),
    and ``calls_root`` over every row and, with ``served``, the served trees' entries
    (`pouw_h1_served.digest_of`); None when it is. The coins themselves are `session_reason`'s and
    `trusted_rounds_reason`'s to check; their credit is `credit_reason`'s."""
    from verity_vllm.protocol_options import pouw_h1 as PH
    from verity_vllm.protocol_options import pouw_h1_served as SV

    c, rows = record.get("commitment") or {}, list(record.get("calls") or ())
    try:
        salt = program_salt_of(c)
    except (CoinError, KeyError, TypeError, ValueError) as e:
        return f"the window's salt can't be derived from its record ({e})"
    for row in c.get("rounds") or ():
        why = RQ.round_reason(salt, row) if isinstance(row, Mapping) else "a round of the log is not an object"
        if why is not None:
            return why
    why = RQ.tables_reason(salt, c.get("rounds") or (), rows)
    if why is not None:
        return why
    served = c.get("served")
    if served is not None:
        try:
            served = SV.digest_of(served["entries"], served["refused"])
        except (KeyError, TypeError, ValueError) as e:
            return f"the window's served trees are malformed ({e})"
    if PH.calls_root(salt, rows, served).hex() != c.get("calls_root"):
        return "the window's calls_root is not over its record's calls" + (" and its served trees" if served else "")
    return None


def program_salt_of(c: Mapping[str, Any]) -> bytes:
    """The window's Program salt, from its H1 commitment `c`'s salt coin over its weights' root."""
    from verity.core.protocols.pouw.audit import salt_of

    from verity_vllm.protocol_options import pouw_h1 as PH
    return PH.program_salt(salt_of(LiveCommitment.from_json(c.get("coins")), LiveOpening.from_json(c.get("salt_opening")).coin,
                                   bytes.fromhex(c["weights_root"])))


def credit_reason(record: Mapping[str, Any], directory: str | Path, heads: Sequence[str | Path], window: Mapping[str, Any],
                  key: ChallengerKey, drawn: Any = None, proofs: Any = None, verify: Any = lms_verify) -> dict[str, Any]:
    """The window's credit by core's rules (`credit`, PRF-132), on its record, its session `directory` and ledger `heads`
    for `window` (``{root}``) under `key` (as `trusted_rounds_reason` reads them), the draw's units `drawn` and their
    proofs' outcomes `proofs` (``{unit, verdict}``): the refusal (or None) of ``credited`` (each credit bit fixed before
    the draw coin), ``a_link``, ``outcome`` and ``manifest``; ``calls``, the indices `credited` returns; and ``credit``,
    those only when all four accept, else None (no partial credit)."""
    from verity.core.protocols.pouw import credit as CR

    from verity_vllm.protocol_options import pouw_h1_served as SV
    c, rows = record.get("commitment") or {}, list(record.get("calls") or ())
    out: dict[str, Any] = {"manifest": manifest_reason(record), "calls": None}
    try:
        lines = ledger_head([remote.ledger_lines(p) for p in heads], key, verify)
        served = c.get("served")
        digest = None if served is None else SV.digest_of(served["entries"], served["refused"]).hex()
        out["calls"], out["credited"] = CR.credited(key.to_json(), lines, window, remote.session_files(directory),
                                                    program_salt_of(c).hex(), rows, digest, c.get("rounds"), c.get("draw"),
                                                    verify)
    except (CoinError, KeyError, TypeError, ValueError) as e:
        out["credited"] = ("coin", f"the window's coins or salt can't be read ({e})")
    out["a_link"] = CR.a_link(c.get("rounds"), (c.get("served") or {}).get("entries"))
    out["outcome"] = CR.outcome(drawn, proofs)
    ok = all(out[k] is None for k in ("manifest", "credited", "a_link", "outcome"))
    out["credit"] = out["calls"] if ok else None
    return out


__all__ = ["ENTRY_TAGS", "Fetch", "LEDGER_ROLE", "LEDGER_SCHEMA", "LEDGER_V1", "Ledger", "PERIOD_KINDS",
           "REPLY_FIELDS", "RESERVED", "SCHEME", "SEALED", "SESSION_FIELDS", "address", "carried", "credit_reason",
           "draw_reason", "entry_message", "evidence", "families", "held", "leaf_of", "lines_of", "lms_verify",
           "manifest_reason", "pinned_key", "period_reason", "program_salt_of", "released_sealed", "root_of", "session_of",
           "session_reason", "trusted_rounds_reason"]
