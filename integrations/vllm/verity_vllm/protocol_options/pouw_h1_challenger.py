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
  window's weights' root, its round-0 request table's root, round 0's ``requests`` in its log, and the session the
  ledger's signed reserve entry names, each read from the ledger and the session's own files, never the prover's; a salt
  opened after another window's root or after the bare weights' root is refused, and so is a session opened without a
  reserve), and its rounds pass PoUW's
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
  at and its commitment's root. The ledger is read only with core's check of record, the one infra's ``auditor.py``
  imports: `Ledger.of` is `challenger.ledger_head` over every head infra published, under the verifier's own key
  (`pinned_key`, its pin from config), and that check's tables of windows and periods. `session_reason` holds a window to
  the one session the ledger opened for its Commit's root, ended, closed and complete with the window's commitment
  (`held`), and reads that session's tree from the gate's ``fetch`` (never the prover's bundle) as core reads a session
  (`challenger.audited_session` and `challenger.session_named`: its record, signed close, release log and refused
  hellos the ones the end sealed); its coins pass the checks above, the salt by `core_check.salt` and the rounds by
  `core_check.rounds`.
* **Periods**: a lane's period is its windows between its ``period-open`` and ``period-close`` entries (session null),
  each window counted at its reserve (or a one-step v2 open) whose prover (``for``, else ``by``) is the lane, as core's
  check lists them (`Ledger.periods`). `period_reason`, a reading of that table, refuses a period the ledger doesn't
  close, one whose close isn't complete (an empty one, or a window never opened, never ended, unclosed or short of its
  coins), an accepted window that isn't one of its windows, and a window of it that has no accepted window (abandoned
  after its coins).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs import remote
from verity.core.protocols.sampled_proofs.challenger import (ChallengerKey, CoinError, LiveCommitment, LiveOpening,
                                                             audited_session, ledger_head, live_reason, session_named)

from verity_vllm.protocol_options import pouw_h1_requests as RQ

SCHEME = Wd.LIVE_SCHEME
#: the pouw setting ``session``: the gate's open answer (``commitment`` from ``status SESSION`` when the answer's is
#: null), and what else the answer may carry; the window carries it without its ``secret`` (`carried`)
SESSION_FIELDS = ("session", "seq", "root", "commitment")
REPLY_FIELDS = ("host", "port", "peer", "by", "for", "reserve", "opened_by", "secret")
_ROOT = re.compile(r"^[0-9a-f]{128}$")
_SECRET = re.compile(r"^[0-9a-f]{64}$")


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


def _window_reason(commits: Mapping[str, Any], key: ChallengerKey, lines: list[str], window: Mapping[str, Any],
                   files: Mapping[str, Any], trusted: LiveCommitment, received: Mapping[str, Sequence[bytes]],
                   session: str, verify: Any) -> str | None:
    """Why the window's coins (its H1 commitment `commits`) aren't those of the session `session`, the one the auditor's
    ledger `lines` (core's check of it under `key`) opened for `window` (``{root}``), whose files are `files`, whose
    signed coin commitment is `trusted` and whose signed log shows it received `received`: its salt coin opened after the
    window's own open root (`core_check.salt`, S from the ledger's reserve and R from its open) and its round log held to
    `core_check.rounds`; None when they are."""
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
    refused = CC.salt(key.to_json(), lines, window, files, commits.get("weights_root"), commits.get("requests_root"),
                      commits.get("salt_opening"), verify)
    if refused is not None:
        return f"the window's salt: {refused[1]}"
    log = list(commits.get("rounds") or ())
    if len(log) != commits.get("steps"):
        return f"the window's round log has {len(log)} rounds, not its {commits.get('steps')} steps"
    if [x.get("round") for x in log] != list(range(len(log))):
        return "the round log is not rounds 0, 1, ... in order"
    if (log[0].get("requests") if log else None) != commits.get("requests_root"):
        return "round 0's request table is not the one the window's open root is over (its requests_root)"
    refused = CC.rounds(key.to_json(), lines, window, files, [x.get("root") for x in log], [x.get("coin") for x in log],
                        verify)
    if refused is not None:
        return f"the window's rounds: {refused[1]}"
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
        sid = audited_session(lines, window, key, verify).session
    except CoinError as e:
        return f"the challenger's session is not trusted: {e}"
    return _window_reason(commits, key, lines, window, files, trusted, received, sid, verify)


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


@dataclass(frozen=True)
class Ledger:
    """The auditor's ledger as core's check of record reads it under the verifier's challenger `key`: `lines`, the
    longest of every head infra published, each checked and any two extending one another (`challenger.ledger_head`),
    and that check's tables (`challenger.verify_ledger`, `challenger.verify_periods`): each window root to its one
    session's ``reserve`` (None for a one-step open), ``open`` and ``end`` (None while it hasn't ended) entries, and each
    period's name to its ``lane``, ``open`` and ``close`` (None while it is open) entries, its ``windows`` (session ids,
    in order), its ``cap`` and its ``circuit``."""

    key: ChallengerKey
    lines: tuple[str, ...]
    windows: Mapping[str, Mapping[str, Any]]
    periods: Mapping[str, Mapping[str, Any]]

    @classmethod
    def of(cls, heads: Sequence[Sequence[str]], key: ChallengerKey, verify: Any = lms_verify) -> Ledger:
        """The ledger of `heads` (every published head, each its lines), refused (`CoinError`) as core refuses it."""
        lines = ledger_head([list(h) for h in heads], key, verify)
        raw, I = [line.encode() for line in lines], key.public[:16]

        def verifies(message: bytes, signature: bytes) -> bool:
            try:
                return verify(key, message, signature) is True
            except ValueError:
                return False

        return cls(key, tuple(lines), Ch.verify_ledger(raw, I, verifies), Ch.verify_periods(raw, I, verifies))

    @classmethod
    def load(cls, heads: Sequence[str | Path], key: ChallengerKey, verify: Any = lms_verify) -> Ledger:
        """`of` the heads at `heads` (each a ``windows.jsonl``)."""
        return cls.of([remote.ledger_lines(p) for p in heads], key, verify)

    def window(self, root: str) -> Mapping[str, Any] | None:
        return self.windows.get(root)


#: ``fetch(session)``: the session's tree on the auditor's side (the gate's ``fetch SESSION``, unpacked: its
#: ``request.json`` and ``out/``), never the prover's bundle
Fetch = Callable[[str], "str | Path | None"]


def evidence(ledger: Ledger, root: str, fetch: Fetch, verify: Any = lms_verify
             ) -> tuple[dict[str, Any], LiveCommitment, dict[str, tuple[bytes, ...]]]:
    """The files of the session `ledger` opens for window root `root`, from its ``out/`` as `fetch` gives it
    (`remote.session_files`), the coin commitment its challenger signed and the roots its signed release log shows it
    received, as core reads a session (`challenger.audited_session`, `challenger.session_named`); `CoinError` naming
    why."""
    window = {"root": root}
    # TODO(#3015): `Ch.credited_session(lines, window, key, descriptor, verify)` here once it is on main.
    audited = audited_session(list(ledger.lines), window, ledger.key, verify)
    tree = fetch(audited.session)
    if tree is None:
        raise CoinError(f"session {audited.session} has no tree on the auditor's side (the gate's fetch)")
    files = remote.session_files(Path(tree) / "out")
    trusted, received = session_named(audited, ledger.key, files, verify)
    return files, trusted, received


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
    Commit's root, ended closed and complete under the window's commitment (`held`), with that session's coins (its
    ``out/`` as `fetch` gives it, read as core reads a session, `evidence`, and checked as `trusted_rounds_reason`
    checks a session's: its salt by `core_check.salt`, S the ledger's reserve's session and R its open's root, and its
    rounds by `core_check.rounds`); None when it is. `registered` is the weights' root the period registered and
    `commit` the window's Commit root as the verifier holds it, which the window's must be. Fails closed."""
    if commits.get("coin_source") != "challenger":
        return ("the window's coins were held in the serving process, not drawn by the auditor host's challenger (the pouw "
                "setting 'challenger'), so no session binds them")
    why, window = held(commits.get("session"), ledger, commit)
    if why is not None:
        return why
    if registered is not None and commits.get("weights_root") != bytes(registered).hex():
        return "the window's weights' root is not the one the period registered"
    sid, root = window["open"]["session"], Ch.root_of(window["open"])
    try:
        files, trusted, received = evidence(ledger, root, fetch, verify)
    except CoinError as e:
        return f"session {sid} is not trusted: {e}"
    return _window_reason(commits, ledger.key, list(ledger.lines), {"root": root}, files, trusted, received, sid, verify)


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
    `trusted_rounds_reason`'s to check."""
    from verity.core.protocols.pouw.audit import salt_of

    from verity_vllm.protocol_options import pouw_h1 as PH
    from verity_vllm.protocol_options import pouw_h1_served as SV

    c, rows = record.get("commitment") or {}, list(record.get("calls") or ())
    try:
        opening = LiveOpening.from_json(c.get("salt_opening"))
        salt = PH.program_salt(salt_of(LiveCommitment.from_json(c.get("coins")), opening.coin,
                                       bytes.fromhex(c["weights_root"])))
    except (CoinError, KeyError, TypeError, ValueError) as e:
        return f"the window's salt can't be derived from its record ({e})"
    for row in c.get("rounds") or ():
        try:
            r = row["round"]
            parts = [bytes.fromhex(row[k]) for k in ("calls", "requests", "slots", "receipts")]
        except (KeyError, TypeError, ValueError) as e:
            return f"a round of the log is malformed ({e})"
        if RQ.round_root(salt, r, *parts).hex() != row.get("root"):
            return f"round {r}'s root is not over its calls, its request table, its slot table and the receipts before it"
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



__all__ = ["Fetch", "Ledger", "REPLY_FIELDS", "SCHEME", "SESSION_FIELDS", "address", "carried", "draw_reason",
           "evidence", "families", "held", "lms_verify", "manifest_reason", "pinned_key", "period_reason", "session_of",
           "session_reason", "trusted_rounds_reason"]
