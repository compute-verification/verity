"""H1's coins from the proof service's challenger on its auditor host (`verity.core.protocols.sampled_proofs.remote`), and the
verifier's check that a served window's coins are that challenger's.

* **Serving** (the pouw setting ``challenger``, ``HOST:PORT``): `session` connects once per window
  (`remote.RemoteChallenger.connect`) to a challenger started over a live epoch's layout, scheme `SCHEME` and
  `families` (S) as ``FAMILIES`` (`benchmarks/live_challenger/run.sh`). The epoch's salt is its salt coin, opened after the
  weights' root (`audit.salt_of`), and round s's coin opens after round s's root (`pouw_h1.H1Run`); `finish` closes the
  session once the window has finished. The challenger is auditor-trusted (`trusted-nebius-host`): it runs on a machine
  the prover doesn't control, and the serving process reaches it through ``ssh -R`` on its own 127.0.0.1.
* **Checking** (`trusted_rounds_reason`): the verifier reads the challenger's directory from the challenger's own run
  (``research fetch RUN --all``, its ``out/``), never from the prover's bundle, under its own challenger key (`pinned_key`):
  the coin commitment the challenger signed and the roots its release log shows it received (`remote.trusted_session`).
  The window's coins must be that commitment's, its salt coin must have followed the weights' root and its rounds must be
  the challenger's logged roots in order, each coin opened after its root (`pouw_h1_requests.rounds_reason`). A window
  whose coins came from a challenger the prover holds (the in-process ``coins`` and ``rounds`` of tests, or a host of
  the prover's) is refused, since the verifier's key signed no record of it. The window's draw coin must have been opened
  once, after `pouw_h1_requests.draw_root` over its manifest, last receipts and every round's root and coin: a draw
  opened early, from another family, or never (a session closed at `finish` with no draw) is refused.
* **Sessions** (RT4-INT-17, the pouw setting ``session``): a lane opens a session on infra's challenger gate for a window,
  a period and the served Commit's root (``open --period P --commit HEX ...``, `benchmarks/live_challenger/auditor.py` on
  vy-challenger-1), and the serving window carries the gate's reply (``commits["session"]``: its ``session`` and ``seq``,
  with the ``period`` and ``commit`` it named; `session_of`). The gate appends the window's ``open`` entry to its ledger,
  signed under the challenger key, before the session's first coin, and refuses a second one for the window; the
  session's ``end`` entry seals the SHA-512 of each file of its ``out/``. `Ledger.verify` is infra's ``verify_ledger``
  over the published ledger under the verifier's own key (`pinned_key`, its pin from config), and refuses any ledger with
  two sessions for one window or one session for two. `session_reason` holds a window to the one session the ledger
  opened for its period and Commit, ended and closed, reading that session's tree from the gate's ``fetch`` (never the
  prover's bundle), each file the one its end entry sealed, its key the verifier's and its record signed for the open
  entry's host and layout; `period_reason` refuses a period with a session that released coins and has no accepted
  window (a window abandoned after its coins), or one that never ended.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import salt_of
from verity.core.protocols.sampled_proofs import remote
from verity.core.protocols.sampled_proofs.challenger import (ChallengerKey, ChallengerRecord, CoinError, LiveCommitment,
                                                             LiveOpening, LiveRecord, live_reason)
from verity.ml.commitments.identity import identity_digest_sha512

from verity_vllm.protocol_options import pouw_h1_requests as RQ

SCHEME = Wd.LIVE_SCHEME
#: infra's challenger ledger (`benchmarks/live_challenger/auditor.py`): its entries' schema, and the role and tag of what
#: an entry's ``sig`` signs (`entry_message`)
LEDGER_SCHEMA = "verity/challenger-auditor/v1"
LEDGER_ROLE = "verity/challenger-auditor-record/v1"
ENTRY_TAG = "verity/challenger-auditor/entry/v1"
#: the files of a session's ``out/`` the verifier reads, each of which its end entry must seal
SEALED = (remote.RECORD, remote.KEY, remote.RELEASES)
#: the pouw setting ``session``: the gate's reply and the window it named, and what else the reply may carry
SESSION_FIELDS = ("session", "seq", "period", "commit")
REPLY_FIELDS = ("host", "port", "peer")
_COMMIT = re.compile(r"^[0-9a-f]{128}$")
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
    """The pouw setting ``session``: None, or the gate's reply to the window's ``open`` (its ``session`` and ledger
    ``seq``, and optionally the challenger's ``host``, ``port`` and ``peer``) with the window it named (``period`` and the
    served Commit's root ``commit``, 128 hex); ValueError otherwise."""
    if raw is None:
        return None
    ok = (isinstance(raw, Mapping) and set(SESSION_FIELDS) <= set(raw) <= {*SESSION_FIELDS, *REPLY_FIELDS}
          and isinstance(raw["session"], str) and bool(raw["session"]) and type(raw["seq"]) is int and raw["seq"] >= 0
          and isinstance(raw["period"], str) and bool(raw["period"])
          and isinstance(raw["commit"], str) and bool(_COMMIT.match(raw["commit"])))
    if not ok:
        raise ValueError("pouw setting 'session' is the challenger gate's reply for the window, {'session', 'seq'} and "
                         "optionally 'host', 'port' and 'peer', with the 'period' and 'commit' (the served Commit's root, "
                         f"128 lowercase hex) its open named, not {raw!r}")
    return dict(raw)


def refusals(scheme: str, settings: Any, stage: str) -> list[str]:
    out = []
    if settings.session is None:
        out.append(f"{scheme}: the pouw setting 'challenger' needs 'session', the challenger gate's reply for the window's "
                   "period and Commit ({'session', 'seq', 'period', 'commit'}), which the window carries and the verifier "
                   "checks against the auditor's ledger")
    if settings.coins is not None or settings.rounds is not None:
        out.append(f"{scheme}: the window's coins come from one source: the pouw setting 'challenger' (the auditor host's "
                   "salt and round coins) or, in tests, 'coins' and 'rounds' held in this process, not both")
    if stage == "capture":
        out.append(f"{scheme}: a capture records nothing, so it opens no session on the auditor host's challenger (the pouw "
                   "setting 'challenger' is a served window's)")
    return out


def session(entry: str, at: str, steps: int, root: bytes) -> tuple[Any, LiveCommitment, LiveOpening, LiveRecord, bytes]:
    """The window's session on the challenger at `at`: a live epoch's coins with at least `steps` round coins, its salt coin
    opened after the weights' root `root`. Returns the session, its coin commitment, the salt coin's opening, the
    `LiveRecord` of its round coins and the epoch's salt; on a refusal the session is closed and ValueError (a layout other
    than a live window's) or `CoinError` (the challenger's) raised."""
    client = remote.RemoteChallenger.connect(at)
    try:
        c = client.commitment
        why = Wd.live_epoch_reason(c)
        if why is None and c.family(Wd.ROUND) < steps:
            why = (f"its {c.family(Wd.ROUND)} round coins are fewer than the window's {steps} rounds (start it with SCHEME "
                   f"{SCHEME} and FAMILIES {families(steps)})")
        if why is not None:
            raise ValueError(f"{entry}: the challenger at {at} is not a live window's: {why}")
        opening = LiveRecord(client, Wd.SALT).record(root)
    except BaseException:
        _close(client)
        raise
    return client, c, opening, LiveRecord(client, Wd.ROUND), salt_of(c, opening.coin, root)


def _close(client: Any) -> str | None:
    try:
        client.close()
    except CoinError as e:
        return str(e)
    return None


def finish(run: Any, client: Any) -> None:
    """The window's release: `run.finish()`, then its session closed; how it closed goes in ``details.challenger``."""
    try:
        run.finish()
    finally:
        if client is not None:
            why = _close(client)
            run.details["challenger"] = {"closed": why is None, **({"reason": why} if why else {})}


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


def trusted_rounds_reason(commits: Mapping[str, Any], directory: str | Path, key: ChallengerKey,
                          verify: Any = lms_verify) -> str | None:
    """Why the window's coins (its H1 commitment `commits`: ``coins``, ``round_coins``, ``salt_opening``, ``weights_root``,
    ``steps`` and the round log ``rounds``) aren't the verifier's own challenger's, read from `directory` under `key`; None
    when they are. Fails closed: a missing, malformed or unsigned record or an unreadable release log trusts nothing."""
    try:
        trusted, received = remote.trusted_session(directory, key, verify)
    except CoinError as e:
        return f"the challenger's session is not trusted: {e}"
    for name in ("round_coins", "coins"):
        try:
            c = LiveCommitment.from_json(commits.get(name))
        except CoinError as e:
            return f"the window's {name} are not a live coin commitment ({e}), so not the verifier's challenger's"
        if c != trusted:
            return (f"the window's {name} {c.root.hex()[:16]} are not the coin commitment the verifier's challenger signed "
                    f"({trusted.root.hex()[:16]}): coins from a challenger the verifier doesn't hold")
    why = Wd.live_epoch_reason(trusted)
    if why is not None:
        return why
    try:
        root, salt = bytes.fromhex(commits["weights_root"]), LiveOpening.from_json(commits.get("salt_opening"))
    except (CoinError, KeyError, TypeError, ValueError) as e:
        return f"the window's salt coin or weights' root is malformed ({e})"
    if received.get(Wd.SALT) != (root,):
        return "the challenger's log does not show the weights' root as the one root its salt coin followed"
    why = live_reason(trusted, salt, Wd.SALT, 0, root)
    if why is not None:
        return f"the window's salt coin: {why}"
    log = list(commits.get("rounds") or ())
    if len(log) != commits.get("steps"):
        return f"the window's round log has {len(log)} rounds, not its {commits.get('steps')} steps"
    why = RQ.rounds_reason(log, trusted)
    if why is not None:
        return why
    if received.get(Wd.ROUND) != tuple(bytes.fromhex(x["root"]) for x in log):
        return "the window's round roots are not the roots the challenger's log shows it received, in order"
    return draw_reason(commits, trusted, received.get(Wd.DRAW, ()))


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
    ENTRY_TAG, entry without sig)]), each [x] prefixed with its length (u64), under the key whose identifier is `I`."""
    if len(I) != 16:
        raise CoinError(f"an LMS identifier is 16 bytes, not {len(I)}")
    body = {k: v for k, v in entry.items() if k != "sig"}
    return hashlib.sha512(_field(LEDGER_ROLE.encode()) + I + _field(identity_digest_sha512(ENTRY_TAG, body))).digest()


def _window_of(entry: Mapping[str, Any]) -> tuple[str, str]:
    w = entry["body"]["window"]
    if not (isinstance(w["period"], str) and isinstance(w["commit"], str)):
        raise TypeError(f"a window is a period and a Commit's root, both strings, not {w!r}")
    return w["period"], w["commit"]


@dataclass(frozen=True)
class Ledger:
    """Infra's challenger ledger (`LEDGER_SCHEMA`), verified under the verifier's challenger `key`: each window
    ``(period, commit)`` to its one session's ``open`` entry and its ``end`` entry (None while it hasn't ended)."""

    key: ChallengerKey
    windows: Mapping[tuple[str, str], Mapping[str, Any]]

    @classmethod
    def verify(cls, lines: Sequence[bytes], key: ChallengerKey, verify: Any = lms_verify) -> Ledger:
        """The ledger of `lines` (its raw lines, as published), refused (`CoinError`) at its first entry that isn't entry
        i of a ledger chained to its predecessor, signed under `key`: an open under another key's pin, a second open for a
        window or a session, an end with no open or a second end, or another kind. Fails closed: one bad entry trusts
        nothing."""
        pin = hashlib.sha512(key.public).hexdigest()
        windows: dict[tuple[str, str], dict[str, Any]] = {}
        by_session: dict[str, tuple[str, str]] = {}
        prev = _ZERO
        for i, line in enumerate(lines):
            try:
                e = json.loads(line)
                kind, sid, sig = e["kind"], e["session"], bytes.fromhex(e["sig"])
                if not isinstance(e["body"], dict):
                    raise TypeError(f"an entry's body is an object, not {e['body']!r}")
                chained = e["schema"] == LEDGER_SCHEMA and type(e["seq"]) is int and e["seq"] == i and e["prev"] == prev
                w = _window_of(e) if kind == "open" else None
            except (KeyError, TypeError, ValueError) as err:
                raise CoinError(f"ledger entry {i} is malformed ({err})") from err
            if not chained:
                raise CoinError(f"ledger entry {i} is not entry {i} of a {LEDGER_SCHEMA} ledger chained to its predecessor")
            if not verify(key, entry_message(key.public[:16], e), sig):
                raise CoinError(f"ledger entry {i}'s signature doesn't verify under the verifier's challenger key")
            if kind == "open":
                if e["body"].get("key_sha512") != pin:
                    raise CoinError(f"ledger entry {i} opens session {sid} under the key pin {e['body'].get('key_sha512')}"
                                    f", not the verifier's {pin[:16]}...")
                if w in windows:
                    raise CoinError(f"ledger entry {i} opens a second session ({sid}, after "
                                    f"{windows[w]['open']['session']}) for Commit {w[1]} in period {w[0]}: one window, one "
                                    "session")
                if sid in by_session:
                    raise CoinError(f"ledger entry {i} opens session {sid} for a second window")
                windows[w], by_session[sid] = {"open": e, "end": None}, w
            elif kind == "end":
                if sid not in by_session or windows[by_session[sid]]["end"] is not None:
                    raise CoinError(f"ledger entry {i} ends session {sid}, which has no open entry or has ended")
                windows[by_session[sid]]["end"] = e
            else:
                raise CoinError(f"ledger entry {i} is of kind {kind!r}, not open or end")
            prev = hashlib.sha512(line).hexdigest()
        return cls(key, windows)

    @classmethod
    def load(cls, path: str | Path, key: ChallengerKey, verify: Any = lms_verify) -> Ledger:
        try:
            data = Path(path).read_bytes()
        except OSError as e:
            raise CoinError(f"no challenger ledger at {path} ({e})") from e
        return cls.verify([line for line in data.split(b"\n") if line], key, verify)

    def window(self, period: str, commit: str) -> Mapping[str, Any] | None:
        return self.windows.get((period, commit))


#: ``fetch(session)``: the session's tree on the auditor's side (the gate's ``fetch SESSION``, unpacked: its
#: ``request.json`` and ``out/``), never the prover's bundle
Fetch = Callable[[str], "str | Path | None"]


def _sealed(path: Path) -> str:
    return hashlib.sha512(path.read_bytes()).hexdigest()


def evidence(window: Mapping[str, Any], fetch: Fetch, key: ChallengerKey) -> Path:
    """The ``out/`` of `window`'s session as `fetch` gives it, refused (`CoinError`) unless the session ended, each file its
    end entry seals is the one it sealed (`SEALED` among them), its key is `key` (the one the ledger's open pins) and its
    record is signed for the open entry's host and layout (scheme, epoch, families); the record's signature and release
    log are `remote.trusted_session`'s."""
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
        record = ChallengerRecord.from_json(json.loads((out / remote.RECORD).read_text(encoding="utf-8")))
    except (OSError, ValueError) as e:
        raise CoinError(f"session {sid}'s key or record is malformed ({e})") from e
    if theirs != key:
        raise CoinError(f"session {sid}'s key is not the verifier's challenger key, the one its open entry pins")
    if record.host != body.get("host"):
        raise CoinError(f"session {sid}'s challenger record is signed for host {record.host}, not {body.get('host')}, the "
                        "one its open entry names")
    layout = json.loads(json.dumps({k: record.commitment.to_json()[k] for k in ("scheme", "epoch", "families")}))
    if layout != {k: body.get(k) for k in layout}:
        raise CoinError(f"session {sid}'s coin commitment is {layout}, not the layout its open entry names")
    return out


def session_reason(commits: Mapping[str, Any], ledger: Ledger, fetch: Fetch, registered: bytes | None = None,
                   commit: str | None = None, verify: Any = lms_verify) -> str | None:
    """Why the window (its H1 commitment `commits`) isn't held to the one session the auditor's `ledger` opened for its
    period and Commit, ended with its client closing it, with that session's coins (`trusted_rounds_reason` over the
    session's ``out/`` as `fetch` gives it, `evidence`); None when it is. `registered` is the weights' root the period
    registered and `commit` the window's Commit root as the verifier holds it, which the window's must be. Fails closed."""
    if commits.get("coin_source") != "challenger":
        return ("the window's coins were held in the serving process, not drawn by the auditor host's challenger (the pouw "
                "setting 'challenger'), so no session binds them")
    try:
        given = session_of(commits.get("session"))
    except ValueError as e:
        return f"the window's challenger session is malformed ({e})"
    if given is None:
        return "the window carries no challenger session (the pouw setting 'session'), so no ledger entry binds its coins"
    sid, period = given["session"], given["period"]
    if commit is not None and given["commit"] != commit:
        return f"the window's session is for Commit {given['commit']}, not the window's Commit {commit}"
    window = ledger.window(period, given["commit"])
    if window is None:
        return (f"the auditor's ledger has no session for Commit {given['commit']} in period {period}, so the window's "
                "coins are no session's of record")
    opened, end = window["open"], window["end"]
    if opened["session"] != sid:
        return (f"the window's session {sid} is not {opened['session']}, the one the auditor's ledger opened for Commit "
                f"{given['commit']} in period {period}")
    if opened["seq"] != given["seq"]:
        return f"the window's session {sid} is ledger entry {given['seq']} in its reply, but entry {opened['seq']} opened it"
    if end is None:
        return f"session {sid} never ended (the auditor's ledger has its open entry and no end): a refused window"
    if end["body"].get("closed") is not True:
        return f"session {sid} ended without its client closing it ({end['body'].get('ended')}): a refused window"
    if registered is not None and commits.get("weights_root") != bytes(registered).hex():
        return "the window's weights' root is not the one the period registered"
    try:
        out = evidence(window, fetch, ledger.key)
    except CoinError as e:
        return f"session {sid}'s evidence is not trusted: {e}"
    return trusted_rounds_reason(commits, out, ledger.key, verify)


def period_reason(ledger: Ledger, period: str, accepted: Sequence[str], fetch: Fetch,
                  verify: Any = lms_verify) -> list[str]:
    """Each session the auditor's `ledger` opened in `period` that released a coin and has no window among `accepted` (the
    session ids of the windows `session_reason` accepted): a window abandoned after its coins, which a prover keeps only
    when it liked them. Each one refuses the period. A session that never ended, or whose evidence is unreadable, is
    refused too (fails closed); one that released nothing is not."""
    out = []
    for (p, commit), window in ledger.windows.items():
        sid = window["open"]["session"]
        if p != period or sid in accepted:
            continue
        try:
            _c, received = remote.trusted_session(evidence(window, fetch, ledger.key), ledger.key, verify)
        except CoinError as err:
            out.append(f"session {sid} (Commit {commit}): its evidence is not trusted ({err}), so nothing shows it "
                       "released no coin")
            continue
        released = sum(len(r) for r in received.values())
        if released:
            out.append(f"session {sid} (Commit {commit}) released {released} coin"
                       f"{'' if released == 1 else 's'} and has no accepted window: a window abandoned after its coins")
    return out


__all__ = ["ENTRY_TAG", "Fetch", "LEDGER_ROLE", "LEDGER_SCHEMA", "Ledger", "REPLY_FIELDS", "SCHEME", "SEALED",
           "SESSION_FIELDS", "address", "draw_reason", "entry_message", "evidence", "families", "finish", "lms_verify",
           "pinned_key", "period_reason", "refusals", "session", "session_of", "session_reason", "trusted_rounds_reason"]
