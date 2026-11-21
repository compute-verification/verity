"""A served window's coins from the proof system's challenger on its auditor device
(`verity.protocols.proof.spot.verifier.challenger.remote`), and the verifier's check that they are that challenger's.

* **Serving**: a window takes its salt, round and sampling coins from one session of the auditor device's challenger,
  opened on the window's open root (ncp-v2's after the weights' root). H1's window is deleted, so no kept serving path
  opens an H1 session.
* **Checking** (`trusted_rounds_reason`, and `session_reason` without a trusted ledger): the verifier's program, run by
  the developer's side (no entry of the auditor's runs it yet), reads the challenger's directory from the challenger's own run
  (its ``out/``, fetched whole), never from the prover's bundle, under its own challenger key (`pinned_key`),
  and the auditor's signed ledger, which opens and ends the one session of the window (`remote.trusted_session`): the coin
  commitment the challenger signed and the roots its signed release log shows it received. The window's coins must be
  that commitment's, its salt coin must have followed its own open root, the root the ledger opened its session on
  (PoUW's salt rule of record, Lean `PouwVerifier.saltCheck`, `core_check.salt_of_record`, `core_check.salt_twin` its Python
  twin and a disagreement a refusal: that root is `salt.window_open_root` of the
  window's weights' root, its round-0 request table's root, round 0's ``requests`` in its log, and the session the
  ledger's signed reserve entry names, each read from the ledger and the session's own files, never the prover's; a salt
  opened after another window's root or after the bare weights' root is refused, and so is a session opened without a
  reserve), and its rounds pass PoUW's
  round rule of record
  (`core_check.rounds`, Lean `PouwVerifier.roundsCheck`): each coin opened after its root, the roots the challenger's log
  shows in order. A window whose coins came from a challenger the prover holds (the in-process ``coins`` and ``rounds``
  of tests, or a host of the prover's), whose log no one signed, or whose ledger is signed under another key, is cut
  before the session's end or opens a second session for the window, is refused. The window's sampling coin comes from a
  sample session of its own (`window.DRAW_FAMILIES`), the one the ledger opens on
  the prover's draw root over its manifest, last receipts and every round's root and coin, opened withheld
  (`challenger.withheld_on`): its challenger released its coin root only after that root, the prover's commitment.
  Its coin must have been opened once, after that root: a sample opened early, from another session or family,
  from a session the ledger didn't open withheld, or never, is refused.
* **Sessions** (the pouw setting ``session``): a lane reserves a session on infra's challenger gate
  (``reserve --peer NAME --scheme S --families JSON``, `verity/protocols/run/verifier/live_challenger/auditor.py` on vy-challenger-1, under
  the lane's own gate key), and the window opens it on its root (``open --session S --root R``, R its window-open
  root, opened from the serving process), and carries the gate's open answer (``commits["session"]``: its
  ``session``, ``seq``, ``root`` and coin ``commitment``,
  without its ``secret``; `session_of`, `carried`). The serving side's hello carries the answer's secret, and it refuses a
  challenger whose coin commitment isn't the answer's before its first coin. The gate appends the window's
  ``open`` entry (v2: ``window.root``) to its ledger, signed under the challenger key, before the session's first coin,
  and refuses a second one for the root, ever; the session's ``end`` entry seals the SHA-512 of each file of its
  ``out/``, the coins each family released, whether that was all of them (``complete``), the leaf its record was signed
  at and its commitment's root. The ledger is read only with core's check of record, the one infra's ``auditor.py``
  imports: `Ledger.of` is `challenger.ledger_head` over every head infra published, under the verifier's own key
  (`pinned_key`, its pin from config), and that check's tables of windows and periods. `session_reason` holds a window to
  the one session the ledger opened for its Commit's root, ended, closed and complete with the window's commitment
  (`held`), and reads that session's tree from the gate's ``fetch`` (never the prover's bundle) as core reads a session
  (`challenger.pinned_session` and `challenger.session_named`: its record, signed close, release log and refused
  hellos the ones the end sealed); its coins pass the checks above, the salt by `core_check.salt_of_record` and the
  rounds by `core_check.rounds`.
* **Pearl-C passes** (`salt_session_reason`): a Pearl-C device run's session is reserved with its salt coin alone
  (`window.SALT_FAMILIES`) and opened on its salt epoch's root (`salt.epoch_open_root` of the weights' root and the
  pass's ``pass_index``, recomputed, never read from the pass), so it counts only ended closed and complete, its one coin
  released; the pass's salt is that coin's over the weights' root, read from the session, never from the pass.
* **Pins**: every session read is held to the host's pins log (`challenger.pinned_session`): the open's signed body
  seals the log's lines (``pins``), and the log, as infra published it (the gate's ``pins``, put by ``publish.sh`` as
  kind ``challenger-pins/v1``, labelled ``challenger-pins head``), must begin with them and record the open's
  ``pinned`` commit as main's, and that commit must be on the verifier's own first-parent history of main (``main``,
  ``git rev-list --first-parent origin/main``, `remote.main_lines`; `Ledger.main`), not only in main's history. Its
  heads are read as core reads a ledger's (`pins_head`, `pins_of`: the longest, any two extending one another). No pins
  log, a cut one, a tree it doesn't record, no copy of main, or a pin off main's first-parent line, credits nothing.
* **Monitoring periods**: a lane's monitoring period is its windows between its ``period-open`` and ``period-close`` entries (session null),
  each window counted at its reserve (or a one-step v2 open) whose prover (``for``, else ``by``) is the lane, as core's
  check lists them (`Ledger.periods`). `period_reason`, a reading of that table, refuses a period the ledger doesn't
  close, one whose close isn't complete (an empty one, or a window never opened, never ended, unclosed or short of its
  coins), an accepted window that isn't one of its windows, and a window of it that has no accepted window (abandoned
  after its coins).
* **Credit** is core's rules of record (`verity.protocols.accounting.compute.pouw.verifier.credit`), called through its one composed
  entry, never copied or run step by step: `credit.window` on a window's evidence and `credit.period_credit` on a
  period, both the Lean's (`PouwVerifier.Credit.executionPhase`, `Credit.periodCredit`), core's Python their twin and a
  disagreement a refusal (`verifier.of_record`). This module builds no credit evidence: H1's builder of it is deleted
  with H1's serving side.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity.commitments.crypto import lms
from verity.protocols.accounting.compute.pouw.verifier import core_check as CC
from verity.protocols.accounting.compute.pouw.verifier import credit as CR
from verity.protocols.accounting.compute.pouw.verifier import salt as PS
from verity.protocols.accounting.compute.pouw.verifier import window as Wd
from verity.protocols.accounting.compute.pouw.verifier.audit import salt_of
from verity.protocols.accounting.compute.pouw.verifier.protocol import DomainError
from verity.protocols.proof.spot.verifier.challenger import challenger as Ch
from verity.protocols.proof.spot.verifier.challenger import remote
from verity.protocols.proof.spot.verifier.challenger.challenger import (
    ChallengerKey,
    CoinError,
    LiveCommitment,
    LiveOpening,
    extends,
    ledger_head,
    live_reason,
    pinned_session,
    session_named,
)

from verity.protocols.accounting.compute.pouw.verifier.kernel_schemes import names as kernel_names
from verity.protocols.accounting.compute.pouw.prover.serving.coins.sample import (
    NO_DRAW_SESSION,
    draw_files,
    draw_reason,
    draw_session_reason,
)

SCHEME = Wd.LIVE_SCHEME
#: the pouw setting ``session``: the gate's open answer (``commitment`` from ``status SESSION`` when the answer's is
#: null), and what else the answer may carry; the window carries it without its ``secret`` (`carried`)
SESSION_FIELDS = ("session", "seq", "root", "commitment")
REPLY_FIELDS = ("host", "port", "peer", "by", "for", "reserve", "opened_by", "secret")
_ROOT = re.compile(r"^[0-9a-f]{128}$")
_SECRET = re.compile(r"^[0-9a-f]{64}$")
#: the ε the proof system sets its committed sample size K for (`draw_k`), at core's δ = 2^-40 (`window.DELTA`), and the
#: share of a window's work it expects credited; the verifier reports the ε a window's sample established, not this one
DRAW_EPS = Fraction(1, 1000)
EXPECTED_CREDIT = Fraction(8, 9)


def families(steps: int) -> str:
    """The challenger's ``FAMILIES`` for a window of `steps` rounds: a live epoch's salt and round coins. Its sampling coin
    is a session of its own (`draw_families`)."""
    return json.dumps([list(f) for f in Wd.live_families(steps)])


def draw_families() -> str:
    """The challenger's ``FAMILIES`` for a window's sample session: its sampling coin alone (`window.DRAW_FAMILIES`)."""
    return json.dumps([list(f) for f in Wd.DRAW_FAMILIES])


def address(raw: Any) -> str | None:
    """The pouw setting ``challenger``: None, or ``HOST:PORT``; ValueError otherwise."""
    if raw is None:
        return None
    host, sep, port = raw.rpartition(":") if isinstance(raw, str) else ("", "", "")
    if not (sep and host and port.isdecimal()):
        raise ValueError(f"pouw setting 'challenger' is the auditor device's challenger as HOST:PORT, not {raw!r}")
    return raw


def session_of(raw: Any) -> dict[str, Any] | None:
    """The pouw setting ``session``: None, or the gate's answer to the window's ``open`` (its ``session``, ledger ``seq``,
    window ``root`` (the served Commit's root) and coin ``commitment``'s root, each root 128 lowercase hex, and optionally
    the challenger's ``host``, ``port`` and ``peer``, the reserve's ``by``, ``for`` and ``seq`` (``reserve``), the
    ``opened_by`` and the hello's ``secret``, 64 lowercase hex); ValueError otherwise, which quotes the answer without
    its secret."""
    if raw is None:
        return None
    ok = (isinstance(raw, Mapping) and set(SESSION_FIELDS) <= set(raw) <= {*SESSION_FIELDS, *REPLY_FIELDS}
          and isinstance(raw["session"], str) and bool(raw["session"]) and type(raw["seq"]) is int and raw["seq"] >= 0
          and all(isinstance(raw[k], str) and bool(_ROOT.match(raw[k])) for k in ("root", "commitment"))
          and (raw.get("secret") is None or (isinstance(raw["secret"], str) and bool(_SECRET.match(raw["secret"])))))
    if not ok:
        held = isinstance(raw, Mapping) and "secret" in raw
        raise ValueError("pouw setting 'session' is the challenger gate's open answer for the window, {'session', 'seq', "
                         "'root', 'commitment'} (the served Commit's root and the session's coin commitment root, 128 "
                         "lowercase hex; the commitment from 'status SESSION' when the answer's is null) and optionally "
                         f"{', '.join(repr(k) for k in REPLY_FIELDS[:-1])} and 'secret' (64 lowercase hex), not "
                         f"{carried(raw) if held else raw!r}{' with a secret (not quoted)' if held else ''}")
    return dict(raw)


def carried(given: Mapping[str, Any]) -> dict[str, Any]:
    """The session a window or a Commit carries: the gate's answer `given` without its hello ``secret``."""
    return {k: v for k, v in given.items() if k != "secret"}



def lms_verify(key: ChallengerKey, message: bytes, signature: bytes) -> bool:
    """`challenger.Verify`: LMS over SHA-512 at the verifier's challenger key's (64, W, H)."""
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


def _window_reason(commits: Mapping[str, Any], key: ChallengerKey, lines: list[str], pins: Any, main: Any,
                   window: Mapping[str, Any], files: Mapping[str, Any], trusted: LiveCommitment,
                   session: str, verify: Any, draw_files: Any) -> str | None:
    """Why the window's coins (its H1 commitment `commits`) aren't those of the session `session`, the one the auditor's
    ledger `lines` (core's check of it under `key`) opened for `window` (``{root}``), whose files are `files` and whose
    signed coin commitment is `trusted`: its salt coin opened after the window's own open root
    (`core_check.salt_of_record`, S from the ledger's reserve and R from its open), its round log held to
    `core_check.rounds`, each reading the session only
    when the host's pins log `pins` records its open's pin and the verifier's first-parent history of main `main` holds
    it (`challenger.pinned_session`), and its sampling coin its sample session's, whose files are `draw_files`
    (`draw_session_reason`); None when they are."""
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
    refused = CC.salt_of_record(key.to_json(), lines, pins, main, window, files, commits.get("weights_root"),
                                commits.get("requests_root"), commits.get("salt_opening"), verify)
    if refused is not None:
        return f"the window's salt: {refused[1]}"
    log = list(commits.get("rounds") or ())
    if len(log) != commits.get("steps"):
        return f"the window's round log has {len(log)} rounds, not its {commits.get('steps')} steps"
    if [x.get("round") for x in log] != list(range(len(log))):
        return "the round log is not rounds 0, 1, ... in order"
    if (log[0].get("requests") if log else None) != commits.get("requests_root"):
        return "round 0's request table is not the one the window's open root is over (its requests_root)"
    refused = CC.rounds(key.to_json(), lines, pins, main, window, files, [x.get("root") for x in log],
                        [x.get("coin") for x in log], verify)
    if refused is not None:
        return f"the window's rounds: {refused[1]}"
    return draw_session_reason(commits, key, lines, pins, main, draw_files, verify)


def pins_of(heads: Any) -> list[str] | None:
    """The host's pins log as infra published it: of `heads` (every head, each its lines), the longest, any two extending
    one another, as `challenger.ledger_head` takes the ledger's; None when there is none, which `pinned_session` refuses
    (`challenger.NO_PINS`). `CoinError` when two heads don't extend one another: the log was cut or rewritten."""
    if not heads:
        return None
    heads = [list(h) for h in heads]
    for i, j in itertools.combinations(range(len(heads)), 2):
        if not (extends(heads[i], heads[j]) or extends(heads[j], heads[i])):
            raise CoinError(f"the host's published pins logs {i} and {j} don't extend one another: its pins log was cut "
                            "or rewritten")
    return max(heads, key=len)


def pins_head(paths: Sequence[str | Path] | None) -> list[str] | None:
    """`pins_of` the heads at `paths`: each a published head of the host's pins log (infra's ``pins.jsonl``, the gate's
    ``pins`` as ``publish.sh`` puts it, kind ``challenger-pins/v1`` labelled ``challenger-pins head``), read line by line
    as core reads a ledger head (`remote.ledger_lines`). A head missing or unreadable raises `CoinError`."""
    heads = []
    for p in paths or ():
        try:
            heads.append([line for line in Path(p).read_text(encoding="utf-8").split("\n") if line])
        except (OSError, UnicodeDecodeError) as e:
            raise CoinError(f"{Ch.NO_PINS} (none readable at {p}: {e})") from e
    return pins_of(heads)


def trusted_rounds_reason(commits: Mapping[str, Any], directory: str | Path, heads: Sequence[str | Path],
                          window: Mapping[str, Any], key: ChallengerKey, verify: Any = lms_verify, *,
                          pins: Sequence[str | Path] | None, main: Sequence[str] | None,
                          draw_directory: str | Path | None = None) -> str | None:
    """Why the window's coins (its H1 commitment `commits`: ``coins``, ``round_coins``, ``salt_opening``, ``weights_root``,
    ``requests_root``, ``session``, ``steps``, the round log ``rounds`` and the ``draw``) aren't the verifier's own
    challenger's, read from `directory` (its sampling coin from `draw_directory`, its sample session's, `draw_session_reason`;
    None names none, and the sample is refused), the session the auditor's signed ledger opens and ends for `window` (``{root}``,
    the window's open root), under `key`, the challenger key the deployment pins, the ledger the longest of the heads
    infra published at `heads` (each a ``windows.jsonl``), any two extending one another, and run from a commit the
    host's pins log at `pins` (`pins_head`) records on the verifier's own first-parent history of main `main`
    (``git rev-list --first-parent origin/main``, `remote.main_lines`; `challenger.pinned_session`); None when they
    are. Fails closed: a missing, malformed or unsigned record, log, hello log or ledger, a ledger signed under another
    key, forked or older than the session's end, a window it opens no one session for, or a session whose pin the log
    or main doesn't hold, trusts nothing (`remote.trusted_session`, `pinned_session`, `core_check.salt_of_record`,
    `core_check.rounds`)."""
    try:
        trusted, _log = remote.trusted_session(directory, heads, window, key, verify)
        lines = ledger_head([remote.ledger_lines(p) for p in heads], key, verify)
        files = remote.session_files(directory)
        draw_files = None if draw_directory is None else remote.session_files(draw_directory)
        log, main = pins_head(pins), _listed(main)
        sid = pinned_session(lines, window, key, log, main, verify).session
    except CoinError as e:
        return f"the challenger's session is not trusted: {e}"
    return _window_reason(commits, key, lines, log, main, window, files, trusted, sid, verify, draw_files)


@dataclass(frozen=True)
class Ledger:
    """The auditor's ledger as core's check of record reads it under the verifier's challenger `key`: `lines`, the
    longest of every head infra published, each checked and any two extending one another (`challenger.ledger_head`),
    and that check's tables (`challenger.verify_ledger`, `challenger.verify_periods`): each window root to its one
    session's ``reserve`` (None for a one-step open), ``open`` and ``end`` (None while it hasn't ended) entries, and each
    period's name to its ``lane``, ``open`` and ``close`` (None while it is open) entries, its ``windows`` (session ids,
    in order), its ``cap`` and its ``circuit``. `pins` is the host's pins log as infra published it (`pins_of`; None
    without one) and `main` the verifier's own first-parent history of main (``git rev-list --first-parent
    origin/main``; None without one): every session read from it must be run from a commit the log records and main's
    first-parent line holds (`challenger.pinned_session`)."""

    key: ChallengerKey
    lines: tuple[str, ...]
    windows: Mapping[str, Mapping[str, Any]]
    periods: Mapping[str, Mapping[str, Any]]
    pins: tuple[str, ...] | None = None
    main: tuple[str, ...] | None = None

    @classmethod
    def of(cls, heads: Sequence[Sequence[str]], key: ChallengerKey, verify: Any = lms_verify,
           pins: Sequence[Sequence[str]] | None = None, main: Sequence[str] | None = None) -> Ledger:
        """The ledger of `heads` (every published head, each its lines) and the pins log of `pins` (every published head
        of it, each its lines, `pins_of`), refused (`CoinError`) as core refuses them, its sessions held to `main`."""
        lines = ledger_head([list(h) for h in heads], key, verify)
        log = pins_of(pins)
        raw, ident = [line.encode() for line in lines], key.public[:16]

        def verifies(message: bytes, signature: bytes) -> bool:
            try:
                return verify(key, message, signature) is True
            except ValueError:
                return False

        return cls(key, tuple(lines), Ch.verify_ledger(raw, ident, verifies), Ch.verify_periods(raw, ident, verifies),
                   None if log is None else tuple(log), None if main is None else tuple(main))

    @classmethod
    def load(cls, heads: Sequence[str | Path], key: ChallengerKey, verify: Any = lms_verify,
             pins: Sequence[str | Path] | None = None, main: Sequence[str] | None = None) -> Ledger:
        """`of` the heads at `heads` (each a ``windows.jsonl``) and the pins log's at `pins` (each a ``pins.jsonl``,
        `pins_head`), its sessions held to `main`."""
        log = pins_head(pins)
        return cls.of([remote.ledger_lines(p) for p in heads], key, verify, None if log is None else [log], main)

    def window(self, root: str) -> Mapping[str, Any] | None:
        return self.windows.get(root)

    def pinned(self, window: Mapping[str, Any], verify: Any = lms_verify) -> Ch.Audited:
        """The session this ledger opens for `window`, run from a commit its pins log records and its `main`'s
        first-parent line holds (`challenger.pinned_session`); `CoinError` naming why not."""
        return pinned_session(list(self.lines), dict(window), self.key, _listed(self.pins), _listed(self.main), verify)


def _listed(lines: Sequence[str] | None) -> list[str] | None:
    return None if lines is None else list(lines)


#: ``fetch(session)``: the session's tree on the auditor's side (the gate's ``fetch SESSION``, unpacked: its
#: ``request.json`` and ``out/``), never the prover's bundle
Fetch = Callable[[str], "str | Path | None"]


def evidence(ledger: Ledger, root: str, fetch: Fetch, verify: Any = lms_verify
             ) -> tuple[dict[str, Any], LiveCommitment, dict[str, tuple[bytes, ...]]]:
    """The files of the session `ledger` opens for window root `root`, from its ``out/`` as `fetch` gives it
    (`remote.session_files`), the coin commitment its challenger signed and the roots its signed release log shows it
    received, as core reads a session run from main's code (`Ledger.pinned`: `challenger.pinned_session` under the
    ledger's pins log, then `challenger.session_named`); `CoinError` naming why."""
    window = {"root": root}
    # TODO: `Ch.credited_session(lines, window, key, descriptor, verify)` here once core has it.
    audited = ledger.pinned(window, verify)
    tree = fetch(audited.session)
    if tree is None:
        raise CoinError(f"session {audited.session} has no tree on the auditor's side (the gate's fetch)")
    files = remote.session_files(Path(tree) / "out")
    trusted, log = session_named(audited, ledger.key, files, verify)
    return files, trusted, Ch.by_family(trusted, log)


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
    ``out/`` as `fetch` gives it, read as core reads a session run from main's code, `evidence`, and checked as
    `trusted_rounds_reason` checks a session's: its salt by `core_check.salt_of_record`, S the ledger's reserve's
    session and R its open's root, and its rounds by `core_check.rounds`, both under the ledger's pins log and `main`); None when it
    is. `registered` is the weights' root the period registered and `commit` the window's Commit root as the verifier
    holds it, which the window's must be. A Pearl-C pass's (``scheme`` one of the kernel schemes' ``pearl-c`` family) session
    is held to its salt session instead (`salt_session_reason`). Fails closed."""
    if commits.get("coin_source") != "challenger":
        return ("the window's coins were held in the serving process, not drawn by the auditor device's challenger (the pouw "
                "setting 'challenger'), so no session binds them")
    if commits.get("scheme") in kernel_names("pearl-c"):
        return salt_session_reason(commits, ledger, fetch, registered, commit, verify)
    why, window = held(commits.get("session"), ledger, commit)
    if why is not None or window is None:
        return why or "no window of the auditor's ledger holds the window's session"
    if registered is not None and commits.get("weights_root") != bytes(registered).hex():
        return "the window's weights' root is not the one the period registered"
    sid, root = window["open"]["session"], Ch.root_of(window["open"])
    try:
        files, trusted, _received = evidence(ledger, root, fetch, verify)
    except CoinError as e:
        return f"session {sid} is not trusted: {e}"
    return _window_reason(commits, ledger.key, list(ledger.lines), _listed(ledger.pins), _listed(ledger.main),
                          {"root": root}, files, trusted, sid, verify, draw_files(commits, ledger, fetch, verify))


def salt_session_reason(commits: Mapping[str, Any], ledger: Ledger, fetch: Fetch, registered: bytes | None = None,
                        commit: str | None = None, verify: Any = lms_verify) -> str | None:
    """Why a Pearl-C pass (its device run's commitment `commits`) isn't held to its salt session, the one whose salt
    coin follows the pass's salt epoch's root R = `salt.epoch_open_root` of its weights' root W (``weights_root``)
    and its epoch's number k (``pass_index``, an integer in [0, 2^64)) rather than a window's open root
    (statement: Lean `Properties.Pouw.LiveSalt.Accepted`); None when it is. R is recomputed here, never read from the
    pass. The session is the one the auditor's `ledger` opened on R, ended closed and complete under the commitment the
    pass names (`held`, its ``session``: id, ledger ``seq``, R and commitment, never its hello's ``secret``; `commit`
    and `registered`, when the verifier holds them, must be W), reserved with the salt coin alone
    (`window.SALT_FAMILIES`) and having released that coin and nothing else. Its coins are read as core reads a session
    (`evidence`), never from the pass: the commitment its challenger signed, of that layout (`window.salt_epoch_reason`)
    and the pass's ``coins``, and its log, showing R alone as the root its salt coin followed. Under that commitment the
    pass's ``salt_opening`` is ``pouw/salt`` coin 0 opened after R (`challenger.live_reason`), and its ``salt`` is
    `audit.salt_of` of that coin over W. Fails closed."""
    w = commits.get("weights_root")
    if not (isinstance(w, str) and _ROOT.match(w)):
        return "the pass's weights' root is not 128 lowercase hex, so no session was opened on it"
    if commit is not None and commit != w:
        return f"the pass's weights' root {w[:16]}... is not {str(commit)[:16]}..., the one the verifier holds"
    if registered is not None and w != bytes(registered).hex():
        return "the pass's weights' root is not the one the period registered"
    try:
        root = PS.epoch_open_root(bytes.fromhex(w), commits.get("pass_index"))
    except DomainError as e:
        return f"the pass's pass_index names no salt epoch: {e}"
    if isinstance(commits.get("session"), Mapping) and "secret" in commits["session"]:
        return "the pass's session names its hello's secret, which a pass never records: a refused pass"
    why, window = held(commits.get("session"), ledger, root.hex())
    if why is not None or window is None:
        return why or "no window of the auditor's ledger holds the pass's session"
    sid, reserve, end = window["open"]["session"], window["reserve"], window["end"]["body"]
    families = None if reserve is None else reserve["body"].get("families")
    salt_only = [list(f) for f in Wd.SALT_FAMILIES]
    if json.dumps(families) != json.dumps(salt_only):
        return f"session {sid} was reserved with {families}, not the salt coin alone {salt_only}: a refused pass"
    if json.dumps(end.get("released"), sort_keys=True) != json.dumps({Wd.SALT: 1}):
        return f"session {sid} released {end.get('released')}, not its one salt coin: a refused pass"
    try:
        _, trusted, received = evidence(ledger, root.hex(), fetch, verify)
    except CoinError as e:
        return f"session {sid} is not trusted: {e}"
    if (why := Wd.salt_epoch_reason(trusted)) is not None:
        return f"session {sid}'s signed coins: {why}"
    if dict(received) != {Wd.SALT: (root,)}:
        return f"session {sid}'s signed log does not show the epoch root alone as the root its salt coin followed"
    try:
        claimed = LiveCommitment.from_json(commits.get("coins"))
        opening = LiveOpening.from_json(commits.get("salt_opening"))
    except CoinError as e:
        return f"the pass's coins are malformed ({e})"
    if claimed != trusted:
        return (f"the pass's coins {claimed.root.hex()[:16]}... are not the coin commitment session {sid}'s challenger "
                f"signed ({trusted.root.hex()[:16]}...)")
    if (why := live_reason(trusted, opening, Wd.SALT, 0, root)) is not None:
        return f"the pass's salt coin: {why}"
    if salt_of(trusted, opening.coin, bytes.fromhex(w)).hex() != commits.get("salt"):
        return "the pass's salt is not audit.salt_of of its session's salt coin over its weights' root"
    return None


def _incomplete(ledger: Ledger, outcome: Mapping[str, Any]) -> str:
    if outcome["root"] is None:
        return "reserved and never opened"
    end = ledger.windows[outcome["root"]]["end"]
    if end is None:
        return f"Commit {outcome['root']}, never ended"
    return f"Commit {outcome['root']}, ended {end['body'].get('ended')!r}, released {end['body'].get('released')}"


def period_reason(ledger: Ledger, period: str, accepted: Sequence[str], presented: Sequence[str] | None = None,
                  verify: Any = lms_verify) -> list[str]:
    """Why the auditor's `ledger` doesn't credit the lane's period `period`, given `accepted` (the session ids of its
    windows that `session_reason` accepted, each with its sample session's, which the close lists after it: a sample session
    no accepted window drew from is one opened for a coin the prover then dropped), one reason each; [] when it does.
    Refused: a period the ledger has no period-open for, or no period-close (its windows aren't all known); an accepted
    window that isn't one of the period's (its lane's reserves and one-step opens between its period-open and its
    period-close); a close that isn't complete (no window, or a window never opened, never ended, unclosed or short of
    its coins: every window counts); and
    a window of it with no accepted window, which ended complete, so released every coin: a window abandoned after its
    coins, which a prover keeps only when it liked them. With `presented`, the window roots the verifier is shown, in
    order, also core's rules of record: `credit.period` (exactly the close's windows, in its order, the close complete)
    and `credit.committed_k` (the open committed ``draw_k``), each refusal as ``code: why``; a period's credit itself is
    `credit.period_credit`'s."""
    p = ledger.periods.get(period)
    if p is None:
        return [f"the auditor's ledger has no period {period}"]
    out = _period_reasons(ledger, period, p, accepted)
    if presented is not None:
        lines = list(ledger.lines)
        why = CR.period(ledger.key.to_json(), lines, period, list(presented), verify) or CR.committed_k(lines, period)[1]
        out += [] if why is None else [f"{why[0]}: {why[1]}"]
    return out


def _period_reasons(ledger: Ledger, period: str, p: Mapping[str, Any], accepted: Sequence[str]) -> list[str]:
    out = [f"session {s}, an accepted window, is not one of period {period}'s windows ({p['windows']}, lane {p['lane']}'s "
           "between its period-open and its period-close)" for s in accepted if s not in p["windows"]]
    if p["close"] is None:
        return out + [(f"period {period} has no period-close in the auditor's ledger, so its windows aren't all known: "
                       "an open period credits nothing")]
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


def draw_k(layout: Any, eps: Fraction = DRAW_EPS, credited: Fraction = EXPECTED_CREDIT) -> int:
    """The K the proof system commits as ``draw_k`` in a period's open, before sampling, for windows of `layout` (the
    period's declared Program's, what `credit.layout_of` reads from its descriptor):
    `window.k_prime` at `eps` and core's δ (`window.DELTA`) over the window's work W, the credited work the share
    `credited` of it (⌊credited·W⌋, the fraction the service expects credited, unknown at the open), at most the
    window's proof blocks (core's work sampling design samples at most every proof block). It states no ε: the verifier reports what each window's
    sample established (`credit.established`). ValueError when core's `credit.strata` refuses the layout."""
    _ss, why = CR.strata(layout)
    if why is not None:
        raise ValueError(f"{why[0]}: {why[1]}")
    cw = Wd.call_work(layout)
    work = sum(size * w for size, w in cw)
    part = Fraction(credited)
    return min(Wd.k_prime(Fraction(eps), Wd.DELTA, part.numerator * work // part.denominator, work, 1),
               sum(size for size, _w in cw), Wd.MAX_K)


__all__ = ["DRAW_EPS", "EXPECTED_CREDIT", "Fetch", "Ledger", "NO_DRAW_SESSION", "REPLY_FIELDS", "SCHEME", "SESSION_FIELDS",
           "address", "carried", "draw_families", "draw_files", "draw_k", "draw_reason", "draw_session_reason", "evidence",
           "families", "held", "lms_verify", "period_reason", "pinned_key", "pins_head", "pins_of", "salt_session_reason",
           "session_of", "session_reason", "trusted_rounds_reason"]
