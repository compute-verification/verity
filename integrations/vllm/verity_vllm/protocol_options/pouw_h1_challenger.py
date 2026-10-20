"""H1's coins from the proof service's challenger on its auditor host (`verity.core.protocols.sampled_proofs.remote`), and the
verifier's check that a served window's coins are that challenger's.

* **Serving** (the pouw setting ``challenger``, ``HOST:PORT``): `session` connects once per window
  (`remote.RemoteChallenger.connect`) to a challenger started over a live epoch's layout, scheme `SCHEME` and
  `families` (S) as ``FAMILIES`` (`benchmarks/live_challenger/run.sh`). The epoch's salt is its salt coin, opened after the
  weights' root (`audit.salt_of`), and round s's coin opens after round s's root (`pouw_h1.H1Run`); `finish` closes the
  session once the window has finished. The challenger is auditor-trusted (`trusted-nebius-host`): it runs on a machine
  the prover doesn't control, and the serving process reaches it through ``ssh -R`` on its own 127.0.0.1.
* **Checking** (`trusted_rounds_reason`): the verifier reads the challenger's directory from the challenger's own run
  (``research fetch RUN --all``, its ``out/``), never from the prover's bundle, under its own challenger key (`pinned_key`),
  and the auditor's signed ledger, which opens and ends the one session of the window (`remote.trusted_session`): the coin
  commitment the challenger signed and the roots its signed release log shows it received. The window's coins must be that commitment's, its salt coin must have followed the
  weights' root, and its rounds pass PoUW's round rule of record (`core_check.rounds`, Lean `PouwCheck.roundsCheck`):
  each coin opened after its root, the roots the challenger's log shows in order. A window whose coins came from a
  challenger the prover holds (the in-process ``coins`` and ``rounds`` of tests, or a host of the prover's), whose log
  no one signed, or whose ledger is signed under another key, is cut before the session's end or opens a second session
  for the window, is refused.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import salt_of
from verity.core.protocols.sampled_proofs import remote
from verity.core.protocols.sampled_proofs.challenger import (ChallengerKey, CoinError, LiveCommitment, LiveOpening,
                                                             LiveRecord, live_reason)

SCHEME = Wd.LIVE_SCHEME


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


def refusals(scheme: str, settings: Any, stage: str) -> list[str]:
    out = []
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
    `LiveRecord` of its round coins and the epoch's salt; on a refusal the session is closed and ValueError (a layout
    other than a live window's) or `CoinError` (the challenger's) raised."""
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


def trusted_rounds_reason(commits: Mapping[str, Any], directory: str | Path, ledger: str | Path, window: Mapping[str, Any],
                          key: ChallengerKey, verify: Any = lms_verify) -> str | None:
    """Why the window's coins (its H1 commitment `commits`: ``coins``, ``round_coins``, ``salt_opening``, ``weights_root``,
    ``steps`` and the round log ``rounds``) aren't the verifier's own challenger's, read from `directory`, the session the
    auditor's signed ledger at `ledger` (infra's ``windows.jsonl``) opens and ends for `window` (``{commit, period}``),
    under `key`, the challenger key the deployment pins; None when they are. Fails closed: a missing, malformed or
    unsigned record, log or ledger, a ledger signed under another key or older than the session's end, or a window it
    opens no one session for, trusts nothing (`remote.trusted_session`)."""
    try:
        trusted, received = remote.trusted_session(directory, ledger, window, key, verify)
        lines, files = remote.ledger_lines(ledger), remote.session_files(directory)
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
    if [x.get("round") for x in log] != list(range(len(log))):
        return "the round log is not rounds 0, 1, ... in order"
    refused = CC.rounds(key.to_json(), lines, window, files, [x.get("root") for x in log], [x.get("coin") for x in log],
                        verify)
    return None if refused is None else f"the window's rounds: {refused[1]}"


__all__ = ["SCHEME", "address", "families", "finish", "lms_verify", "pinned_key", "refusals", "session",
           "trusted_rounds_reason"]
