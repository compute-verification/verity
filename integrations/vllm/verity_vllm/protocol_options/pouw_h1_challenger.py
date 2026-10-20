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
  the served Commit's root (``open --commit HEX --peer NAME --scheme S --families JSON``,
  `benchmarks/live_challenger/auditor.py` on vy-challenger-1, under the lane's own gate key), and the serving window
  carries the gate's answer (``commits["session"]``: its ``session``, ``seq``, ``root`` and coin ``commitment``, without
  its ``secret``; `session_of`, `carried`). The serving side's hello carries the answer's secret, and it refuses a
  challenger whose coin commitment isn't the answer's before its first coin (`connect`). The gate appends the window's
  ``open`` entry (v2: ``window.root``) to its ledger, signed under the challenger key, before the session's first coin,
  and refuses a second one for the root, ever; the session's ``end`` entry seals the SHA-512 of each file of its
  ``out/``, the coins each family released, whether that was all of them (``complete``), the leaf its record was signed
  at and its commitment's root. `Ledger.verify` is infra's ``verify_ledger`` over one published head under the
  verifier's own key (`pinned_key`, its pin from config): chained, no v1 entry after a v2, its LMS leaves strictly
  increasing, each record's leaf between its session's open and end and no entry's, one session a root and one root a
  session; `Ledger.newest` reads every published head and refuses two that don't extend one another. `session_reason`
  holds a window to the one session the ledger opened for its Commit's root, ended, closed and complete with the
  window's commitment, reading that session's tree from the gate's ``fetch`` (never the prover's bundle), each file the
  one its end entry sealed, its key the verifier's, its record signed for the open entry's host and layout at the end's
  record leaf, and its release log the end's counts; `period_reason` refuses a period with a session that released coins
  and has no accepted window (a window abandoned after its coins), or one that never ended. The ledger has no period
  yet: a period's sessions are those opened between its ``period-open`` and ``period-close`` entries (PRF-108), which
  `period_reason` takes as a range of ledger seqs.
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
#: infra's challenger ledger (`benchmarks/live_challenger/auditor.py`): its entries' schema (v2, after the host's v1
#: entries in one chain), and the role and tag by schema of what an entry's ``sig`` signs (`entry_message`)
LEDGER_V1 = "verity/challenger-auditor/v1"
LEDGER_SCHEMA = "verity/challenger-auditor/v2"
LEDGER_ROLE = "verity/challenger-auditor-record/v1"
ENTRY_TAGS = {LEDGER_V1: "verity/challenger-auditor/entry/v1", LEDGER_SCHEMA: "verity/challenger-auditor/entry/v2"}
#: the files of a session's ``out/`` the verifier reads, each of which its end entry must seal
SEALED = (remote.RECORD, remote.KEY, remote.RELEASES)
#: the pouw setting ``session``: the gate's open answer (``commitment`` from ``status SESSION`` when the answer's is
#: null), and what else the answer may carry; the window carries it without its ``secret`` (`carried`)
SESSION_FIELDS = ("session", "seq", "root", "commitment")
REPLY_FIELDS = ("host", "port", "peer", "by", "secret")
_ROOT = re.compile(r"^[0-9a-f]{128}$")
_SECRET = re.compile(r"^[0-9a-f]{64}$")
_HEX = re.compile(r"^(?:[0-9a-f]{2})+$")
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
    the challenger's ``host``, ``port`` and ``peer``, the opener ``by`` and the hello's ``secret``, 64 lowercase hex);
    ValueError otherwise."""
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
                         f"'host', 'port', 'peer', 'by' and 'secret' (64 lowercase hex), not {raw!r}")
    return dict(raw)


def carried(given: Mapping[str, Any]) -> dict[str, Any]:
    """The session a window or a Commit carries: the gate's answer `given` without its hello ``secret``."""
    return {k: v for k, v in given.items() if k != "secret"}


def refusals(scheme: str, settings: Any, stage: str) -> list[str]:
    out = []
    if settings.session is None:
        out.append(f"{scheme}: the pouw setting 'challenger' needs 'session', the challenger gate's open answer for the "
                   "window's Commit ({'session', 'seq', 'root', 'commitment'}), which the window carries and the verifier "
                   "checks against the auditor's ledger")
    if settings.coins is not None or settings.rounds is not None:
        out.append(f"{scheme}: the window's coins come from one source: the pouw setting 'challenger' (the auditor host's "
                   "salt and round coins) or, in tests, 'coins' and 'rounds' held in this process, not both")
    if stage == "capture":
        out.append(f"{scheme}: a capture records nothing, so it opens no session on the auditor host's challenger (the pouw "
                   "setting 'challenger' is a served window's)")
    return out


def connect(at: str, given: Mapping[str, Any]) -> Any:
    """The session the gate's answer `given` (`session_of`) opened, on the challenger at `at`: its hello carries the
    answer's ``secret`` when it has one, and a challenger whose coin commitment isn't the answer's is closed and refused
    (ValueError) before its first coin, since another process's challenger on that port would serve other coins."""
    secret = given.get("secret")
    if secret is None:
        client = remote.RemoteChallenger.connect(at)
    else:
        client = remote.RemoteChallenger.connect(at, bytes.fromhex(secret))
    got = client.commitment.root.hex()
    if got != given["commitment"]:
        _close(client)
        raise ValueError(f"the challenger at {at} commits to coins {got[:16]}..., not {given['commitment'][:16]}..., the "
                         f"commitment the gate's answer gives session {given['session']}: refused before its first coin")
    return client


def session(entry: str, at: str, steps: int, root: bytes,
            given: Mapping[str, Any]) -> tuple[Any, LiveCommitment, LiveOpening, LiveRecord, bytes]:
    """The window's session on the challenger at `at`, the one the gate's answer `given` opened (`connect`): a live epoch's
    coins with at least `steps` round coins, its salt coin opened after the weights' root `root`. Returns the session, its
    coin commitment, the salt coin's opening, the `LiveRecord` of its round coins and the epoch's salt; on a refusal the
    session is closed and ValueError (another commitment than the answer's, or a layout other than a live window's) or
    `CoinError` (the challenger's) raised."""
    try:
        client = connect(at, given)
    except ValueError as e:
        raise ValueError(f"{entry}: {e}") from e
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
    ENTRY_TAGS[schema], entry without sig)]), each [x] prefixed with its length (u64), under the key whose identifier is
    `I`."""
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


def _prover_of(opened: Mapping[str, Any]) -> Any:
    return opened["body"].get("by" if opened["schema"] == LEDGER_SCHEMA else "lane")


@dataclass(frozen=True)
class Ledger:
    """Infra's challenger ledger (`LEDGER_SCHEMA`, after the host's `LEDGER_V1` entries), verified under the verifier's
    challenger `key`: each window root to its one session's ``open`` entry and its ``end`` entry (None while it hasn't
    ended)."""

    key: ChallengerKey
    windows: Mapping[str, Mapping[str, Any]]

    @classmethod
    def verify(cls, lines: Sequence[bytes], key: ChallengerKey, verify: Any = lms_verify) -> Ledger:
        """The ledger of `lines` (one head's raw lines, as published), infra's ``verify_ledger`` under `key`, refused
        (`CoinError`) at its first entry that isn't entry i of a v1-then-v2 ledger chained to its predecessor and signed
        under `key` at a leaf after the last entry's: an open of no root or under another key's pin, a second open for a
        root or a session, an end with no open or a second end, a v2 end whose record leaf isn't between its open's and
        its end's (or is another end's or an entry's), or another kind. Fails closed: one bad entry trusts nothing."""
        pin = hashlib.sha512(key.public).hexdigest()
        windows: dict[str, dict[str, Any]] = {}
        opened_at: dict[str, int] = {}
        by_session: dict[str, str] = {}
        leaves: set[int] = set()
        records: dict[int, int] = {}
        prev, last, v2 = _ZERO, -1, False
        for i, line in enumerate(lines):
            try:
                e = json.loads(line)
                if type(e) is not dict:
                    raise TypeError(f"an entry is an object, not {e!r}")
                kind, sid, body, q = e["kind"], e["session"], e["body"], leaf_of(e["sig"])
                if not (isinstance(sid, str) and isinstance(body, dict)):
                    raise TypeError("an entry has a session (a string) and a body (an object)")
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
            if kind == "open":
                try:
                    root = root_of(e)
                except ValueError as err:
                    raise CoinError(f"ledger entry {i} opens session {sid} for no window ({err})") from err
                if body.get("key_sha512") != pin:
                    raise CoinError(f"ledger entry {i} opens session {sid} under the key pin {body.get('key_sha512')}"
                                    f", not the verifier's {pin[:16]}...")
                if root in windows:
                    raise CoinError(f"ledger entry {i} opens a second session ({sid}, after "
                                    f"{windows[root]['open']['session']}) for root {root}: one window, one session")
                if sid in by_session:
                    raise CoinError(f"ledger entry {i} opens session {sid} for a second window")
                windows[root], opened_at[root], by_session[sid] = {"open": e, "end": None}, q, root
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
                raise CoinError(f"ledger entry {i} is of kind {kind!r}, not open or end")
            leaves.add(q)
            prev, last = hashlib.sha512(line).hexdigest(), q
        if reused := sorted(leaves & set(records)):
            raise CoinError(f"ledger entry {records[reused[0]]}'s record leaf {reused[0]} is also an entry's leaf")
        return cls(key, windows)

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


def evidence(window: Mapping[str, Any], fetch: Fetch, key: ChallengerKey, verify: Any = lms_verify) -> Path:
    """The ``out/`` of `window`'s session as `fetch` gives it, refused (`CoinError`) unless the session ended, each file its
    end entry seals is the one it sealed (`SEALED` among them), its key is `key` (the one the ledger's open pins) and its
    record is signed for the open entry's host and layout (scheme, epoch, families); for a v2 end, the record's
    commitment root and LMS leaf are the end's ``commitment`` and ``record_leaf``, and the coins its release log shows each
    family released (`remote.trusted_session`, which also checks the record's signature under `key`) are the end's
    ``released``."""
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
        _c, received = remote.trusted_session(out, key, verify)
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
    (`trusted_rounds_reason` over the session's ``out/`` as `fetch` gives it, `evidence`); None when it is. `registered`
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
    return trusted_rounds_reason(commits, out, ledger.key, verify)


def period_reason(ledger: Ledger, accepted: Sequence[str], fetch: Fetch, prover: str | None = None,
                  seqs: range | None = None, verify: Any = lms_verify) -> list[str]:
    """Each session of the period that released a coin and has no window among `accepted` (the session ids of the windows
    `session_reason` accepted): a window abandoned after its coins, which a prover keeps only when it liked them. Each one
    refuses the period. A session that never ended, or whose evidence is unreadable, is refused too (fails closed); one
    that released nothing is not. The period's sessions are those the auditor's `ledger` opened for `prover` (a v2 open's
    ``by``, a v1 open's ``lane``; every prover's when None) at a ledger seq in `seqs`: the entries between the period's
    ``period-open`` and ``period-close`` (PRF-108), the whole ledger when None, since the ledger has no period entry
    yet."""
    out = []
    for commit, window in ledger.windows.items():
        opened = window["open"]
        sid = opened["session"]
        if (sid in accepted or (seqs is not None and opened["seq"] not in seqs)
                or (prover is not None and _prover_of(opened) != prover)):
            continue
        try:
            _c, received = remote.trusted_session(evidence(window, fetch, ledger.key, verify), ledger.key, verify)
        except CoinError as err:
            out.append(f"session {sid} (Commit {commit}): its evidence is not trusted ({err}), so nothing shows it "
                       "released no coin")
            continue
        released = sum(len(r) for r in received.values())
        if released:
            out.append(f"session {sid} (Commit {commit}) released {released} coin"
                       f"{'' if released == 1 else 's'} and has no accepted window: a window abandoned after its coins")
    return out


__all__ = ["ENTRY_TAGS", "Fetch", "LEDGER_ROLE", "LEDGER_SCHEMA", "LEDGER_V1", "Ledger", "REPLY_FIELDS", "SCHEME", "SEALED",
           "SESSION_FIELDS", "address", "carried", "connect", "draw_reason", "entry_message", "evidence", "families",
           "finish", "held", "leaf_of", "lines_of", "lms_verify", "pinned_key", "period_reason", "refusals", "root_of",
           "session", "session_of", "session_reason", "trusted_rounds_reason"]
