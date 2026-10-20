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
* **Sessions** (RT4-INT-17, the pouw setting ``session``): a lane requests a session from infra for a Commit and a
  period and gets its id and the challenger's port; the serving window carries the id (``commits["session"]``). Infra's
  launcher writes the auditor's record of each session (`Record`: its id, Commit, period, the challenger's run, the
  SHA-512 of its key and the pinned commit it runs) before its first coin, and refuses a second session for one window.
  `session_reason` holds a window to the one session the record names for its Commit and period, reading that session's
  directory from the record's run and its key under the record's pin; `period_reason` refuses a period with a session
  that released coins and has no accepted window (a window abandoned after its coins). The record here is a stub of
  infra's (`RECORD_SCHEMA`), the seam until infra posts the real one.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import salt_of
from verity.core.protocols.sampled_proofs import remote
from verity.core.protocols.sampled_proofs.challenger import (ChallengerKey, CoinError, LiveCommitment, LiveOpening,
                                                             LiveRecord, live_reason)

from verity_vllm.protocol_options import pouw_h1_requests as RQ

SCHEME = Wd.LIVE_SCHEME
#: the stub of infra's auditor record of challenger sessions (RT4-INT-17's seam)
RECORD_SCHEMA = "verity/challenger-sessions/stub-v0"
#: each session's fields in the auditor's record
RECORD_FIELDS = ("session", "commit", "period", "run", "key_sha512", "source_commit")


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


def _hex64(raw: Any) -> bool:
    try:
        return isinstance(raw, str) and len(bytes.fromhex(raw)) == 64
    except ValueError:
        return False


def session_of(raw: Any) -> dict[str, Any] | None:
    """The pouw setting ``session``: None, or the session infra gave this window, ``{"id", "commit", "period"}`` and, when
    infra returns it, the coin commitment's ``root`` (hex); ValueError otherwise."""
    if raw is None:
        return None
    ok = (isinstance(raw, Mapping) and set(raw) <= {"id", "commit", "period", "root"}
          and all(isinstance(raw.get(k), str) and raw.get(k) for k in ("id", "commit"))
          and isinstance(raw.get("period"), (str, int)) and not isinstance(raw.get("period"), bool)
          and ("root" not in raw or _hex64(raw["root"])))
    if not ok:
        raise ValueError("pouw setting 'session' is the challenger session infra gave the window: {'id', 'commit', "
                         f"'period'}} and optionally its coin commitment's 'root' (128 hex), not {raw!r}")
    return dict(raw)


def refusals(scheme: str, settings: Any, stage: str) -> list[str]:
    out = []
    if settings.session is None:
        out.append(f"{scheme}: the pouw setting 'challenger' needs 'session', the session infra gave the window for its "
                   "Commit and period ({'id', 'commit', 'period'}), which the window carries and the verifier checks "
                   "against the auditor's record")
    if settings.coins is not None or settings.rounds is not None:
        out.append(f"{scheme}: the window's coins come from one source: the pouw setting 'challenger' (the auditor host's "
                   "salt and round coins) or, in tests, 'coins' and 'rounds' held in this process, not both")
    if stage == "capture":
        out.append(f"{scheme}: a capture records nothing, so it opens no session on the auditor host's challenger (the pouw "
                   "setting 'challenger' is a served window's)")
    return out


def session(entry: str, at: str, steps: int, root: bytes, given: Mapping[str, Any] | None = None
            ) -> tuple[Any, LiveCommitment, LiveOpening, LiveRecord, bytes]:
    """The window's session on the challenger at `at`: a live epoch's coins with at least `steps` round coins, its salt coin
    opened after the weights' root `root`. `given` is the session infra gave the window (`session_of`): when it names its
    coin commitment's root, the challenger at `at` must be that one. Returns the session, its coin commitment, the salt
    coin's opening, the `LiveRecord` of its round coins and the epoch's salt; on a refusal the session is closed and
    ValueError (a layout other than a live window's, or another session's) or `CoinError` (the challenger's) raised."""
    client = remote.RemoteChallenger.connect(at)
    try:
        c = client.commitment
        why = Wd.live_epoch_reason(c)
        if why is None and c.family(Wd.ROUND) < steps:
            why = (f"its {c.family(Wd.ROUND)} round coins are fewer than the window's {steps} rounds (start it with SCHEME "
                   f"{SCHEME} and FAMILIES {families(steps)})")
        if why is None and given and "root" in given and c.root.hex() != given["root"].lower():
            why = (f"its coin commitment {c.root.hex()[:16]}... is not session {given['id']}'s "
                   f"({given['root'][:16]}...), the one infra gave the window")
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
    return lms.verify(p, key.public, message, lms.Signature.decode(p, signature))


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


@dataclass(frozen=True)
class Record:
    """The auditor's record of challenger sessions, a stub of infra's (`RECORD_SCHEMA`): each session's id, the Commit and
    period it is for, the challenger run whose directory holds its record and release log, the SHA-512 of the
    challenger's public key (the pin, read here and never from code) and the commit the challenger ran, and optionally
    its coin commitment's ``root``. Infra writes an entry before the session's first coin."""

    sessions: tuple[Mapping[str, Any], ...]

    @classmethod
    def of(cls, raw: Any) -> Record:
        """`raw` as the record; CoinError on any other form (fails closed: a malformed record trusts nothing)."""
        if not (isinstance(raw, Mapping) and raw.get("schema") == RECORD_SCHEMA and isinstance(raw.get("sessions"), list)):
            raise CoinError(f"the auditor's session record is not a {RECORD_SCHEMA} record")
        for e in raw["sessions"]:
            if not (isinstance(e, Mapping) and set(RECORD_FIELDS) <= set(e) <= {*RECORD_FIELDS, "root"}
                    and all(isinstance(e[k], str) and e[k] for k in RECORD_FIELDS if k != "period")
                    and isinstance(e["period"], (str, int)) and not isinstance(e["period"], bool)
                    and _hex64(e["key_sha512"]) and ("root" not in e or _hex64(e["root"]))):
                raise CoinError(f"the auditor's session record has a malformed session ({e!r})")
        return cls(tuple(dict(e) for e in raw["sessions"]))

    @classmethod
    def load(cls, path: str | Path) -> Record:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise CoinError(f"no auditor's session record at {path} ({e})") from e
        return cls.of(raw)

    def window(self, commit: str, period: Any) -> list[Mapping[str, Any]]:
        return [e for e in self.sessions if e["commit"] == commit and e["period"] == period]


#: ``fetch(run)``: the directory of the challenger run `run` on the auditor's side (``research fetch RUN --all``'s
#: ``out/``), never the prover's bundle
Fetch = Callable[[str], "str | Path | None"]


def _evidence(entry: Mapping[str, Any], fetch: Fetch, verify: Any) -> tuple[Path, ChallengerKey]:
    directory = fetch(entry["run"])
    if directory is None:
        raise CoinError(f"session {entry['session']}'s challenger run {entry['run']} has no directory on the auditor's side")
    return Path(directory), pinned_key(Path(directory) / remote.KEY, entry["key_sha512"])


def session_reason(commits: Mapping[str, Any], record: Record, fetch: Fetch, registered: bytes | None = None,
                   verify: Any = lms_verify) -> str | None:
    """Why the window (its H1 commitment `commits`) isn't held to the one session the auditor's `record` names for its
    Commit and period, with that session's coins (`trusted_rounds_reason`, under the record's key pin, from the record's
    run as `fetch` resolves it); None when it is. `registered` is the weights' root the period registered, which the
    window's must be. Fails closed."""
    if commits.get("coin_source") != "challenger":
        return ("the window's coins were held in the serving process, not drawn by the auditor host's challenger (the pouw "
                "setting 'challenger'), so no session binds them")
    given = commits.get("session")
    if not isinstance(given, Mapping) or not {"id", "commit", "period"} <= set(given):
        return "the window carries no challenger session (the pouw setting 'session'), so no record binds its coins"
    window = record.window(given["commit"], given["period"])
    if not window:
        return (f"the auditor's record has no session for Commit {given['commit']} in period {given['period']}, so the "
                "window's coins are no session's of record")
    if len(window) > 1:
        return (f"the auditor's record has {len(window)} sessions for Commit {given['commit']} in period "
                f"{given['period']}: one window, one session")
    entry = window[0]
    if entry["session"] != given["id"]:
        return (f"the window's session {given['id']} is not {entry['session']}, the one the auditor's record names for "
                f"Commit {given['commit']} in period {given['period']}")
    if sum(e["session"] == entry["session"] for e in record.sessions) > 1:
        return f"the auditor's record names session {entry['session']} for more than one window"
    if registered is not None and commits.get("weights_root") != bytes(registered).hex():
        return "the window's weights' root is not the one the period registered"
    try:
        directory, key = _evidence(entry, fetch, verify)
    except CoinError as e:
        return f"session {entry['session']}'s evidence is not trusted: {e}"
    if "root" in entry:
        try:
            c = LiveCommitment.from_json(commits.get("coins"))
        except CoinError as e:
            return f"the window's coins are not a live coin commitment ({e})"
        if c.root.hex() != entry["root"].lower():
            return f"the window's coin commitment is not session {entry['session']}'s in the auditor's record"
    return trusted_rounds_reason(commits, directory, key, verify)


def period_reason(record: Record, period: Any, accepted: Sequence[str], fetch: Fetch,
                  verify: Any = lms_verify) -> list[str]:
    """Each session of `period` in the auditor's `record` that released a coin and has no window among `accepted` (the
    session ids of the windows `session_reason` accepted): a window abandoned after its coins, which a prover keeps only
    when it liked them. Each one refuses the period. A session whose evidence is unreadable is refused too (fails closed);
    one that released nothing is not."""
    out = []
    for e in record.sessions:
        if e["period"] != period or e["session"] in accepted:
            continue
        try:
            directory, key = _evidence(e, fetch, verify)
            _c, received = remote.trusted_session(directory, key, verify)
        except CoinError as err:
            out.append(f"session {e['session']} (Commit {e['commit']}): its evidence is not trusted ({err}), so nothing "
                       "shows it released no coin")
            continue
        released = sum(len(r) for r in received.values())
        if released:
            out.append(f"session {e['session']} (Commit {e['commit']}) released {released} coin"
                       f"{'' if released == 1 else 's'} and has no accepted window: a window abandoned after its coins")
    return out


__all__ = ["Fetch", "RECORD_FIELDS", "RECORD_SCHEMA", "Record", "SCHEME", "address", "draw_reason", "families", "finish",
           "lms_verify", "pinned_key", "period_reason", "refusals", "session", "session_of", "session_reason",
           "trusted_rounds_reason"]
