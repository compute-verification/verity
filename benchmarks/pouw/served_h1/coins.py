"""A served window's live coins: where they come from, and the verifier's check of them from the challenger's own run.

* **Off the node** (`Session`: its id, the HOST:PORT its challenger listens on, the secret its hello carries and the
  coin commitment's root the gate's open answered): the proof service's challenger on its auditor host
  (`sampled_proofs.remote.RemoteChallenger`), one session a window, from infra's gate in two steps; lanes don't start
  challenger sessions themselves. `reserve` (``ssh GATE reserve`` with the peer node and the window's layout) gives the
  window's session id; the reserve counts toward its lane's open period, and one not opened by its ``expires_utc`` is
  that period's refused window. `open_window` (``ssh GATE open --session S --root R``) opens it on the window's root R,
  which the caller gives (128 lowercase hex digits, no other open's, ever), and gives the port its challenger listens on
  at the peer's 127.0.0.1, the session's secret, which only that answer carries and nothing here prints, and the root of
  the coin commitment its challenger signed; the challenger opens the salt coin only after R. The gated served window
  of record is vllm's H1 path, `integrations/vllm/verity_vllm/protocols/pouw/h1/__init__.py` (#3053), which holds its
  salt, round-0 request table, R, session S and requests_root rules (#2950, PRF-132); serve.py refuses ``--gate``.
  ``coins.py open`` reserves and opens a session by hand, for development and timing rows only, on `stand_in_root` of
  its session, a stand-in and never a window's root, passed as ``--challenger HOST:PORT``, ``--session ID``,
  ``--secret-file F`` and ``--commitment HEX`` (``$POUW_CHALLENGER``, ``$POUW_CHALLENGER_SESSION``,
  ``$POUW_CHALLENGER_SECRET_FILE``, ``$POUW_CHALLENGER_COMMITMENT``); an address without its session id, either without
  the other, or a secret or commitment without both, is refused. The hello carries the secret
  (``RemoteChallenger.connect(address, secret)``, infra's hello secret, RT4-INT-29). The session's layout is the
  window's live epoch (`layout`: `window.LIVE_SCHEME`, the window's epoch, `window.live_families(rounds)`), and the
  client refuses a commitment of any other, or of another root than the gate's open answered.
  Each coin opens there, after the root the serving loop sends, and the challenger logs both before the coin leaves its
  host. `RoundPipe` sends a window's round roots without waiting on each coin (a ring of steps whose units wait on their
  round's coin), every coin still opened after its own root and checked as it comes back. The prover's record says only ``coins=off-node session ID at HOST:PORT (unchecked)``: what it writes about its
  coins is the prover's.
* **In-process** (no address): `window.live_challenger` in the serving process, on the prover's node. It stays for CPU
  tests, and every run that uses it carries `IN_PROCESS`, ``coins=in-process (not certified)``.

`check` is the verifier's side. It reads the session as infra's gate publishes it (``fetch SESSION``: the challenger's
out/, with record.json, key.json, releases.jsonl, hellos.jsonl and challenger.json, its signed close) and every head of
the auditor's signed ledger it was shown (``ssh GATE ledger``, or infra's published `challenger-ledger head`), never the
prover's bundle. The session is core's to establish, the one check the auditor host and every verifier run
(`remote.trusted_session`, through `audited_window`), under the challenger's key from its deployment record
(``--key-record``, infra's challenger-key/v1, never the ledger's key_sha512; no key is pinned in code): the ledger's
heads, signatures, chain and entries; the one session it opens on the window's root, binding its reserve and carrying
its request, by its prover, inside its minutes and its period; its end, closed by its client; the session's record,
close, release log and refused hellos the files that end sealed (`challenger.session_named`); and its open's pin a
commit the host's pins log records (``--pins``, infra's published `challenger-pins head`) on the verifier's own
first-parent history of main (``--main``, `challenger.pinned_session`), so a session the log doesn't record is refused
though this check credits nothing. On top of it `check` holds the manifest's session and window to that session's, the open's
layout to the signed commitment's, and its period to the verifier's; then the salt coin after the window's root, round
s's coin after round s's root for every round the challenger logged, and the draw coin after the root the manifest
names. The key record, the window's root and its period are the verifier's to name; while it names any not, the coins
are ``off-node HOST (checked; key, root, period unanchored)`` (under key.json's own key, the manifest's root, any
period), and only with all three are they ``off-node HOST (checked)``.

    python3 coins.py families --rounds S                        # the challenger's FAMILIES for a window of S rounds
    python3 coins.py open --gate USER@HOST --peer NODE --epoch E --rounds S [--minutes M] --secret-file F
    python3 coins.py check --challenger-out SESSION_DIR --ledger HEAD [--ledger HEAD ...] --pins LOG --main FILE
        --manifest OUT/manifest.json [--key-record KEY] [--period P] [--root R]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs import remote as R

CHALLENGER_ENV, SESSION_ENV, SECRET_ENV = "POUW_CHALLENGER", "POUW_CHALLENGER_SESSION", "POUW_CHALLENGER_SECRET_FILE"
COMMITMENT_ENV, GATE_ENV = "POUW_CHALLENGER_COMMITMENT", "POUW_CHALLENGER_GATE"
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "config"))
from verity_config.settings import of  # noqa: E402

settings = of(__file__)
IN_PROCESS = "in-process (not certified)"
STAND_IN_TAG = b"verity/served-h1/stand-in-root/v1"
STAND_IN = "a stand-in (coins.stand_in_root of the session), never a window's root"
SSH = ("ssh", "-o", "BatchMode=yes")


@dataclass(frozen=True)
class Session:
    """A challenger session as infra's gate opened it: its id, the HOST:PORT its challenger listens on here, the secret
    its hello carries (None for a challenger that takes none), and the root (hex) of the coin commitment the gate's open
    answered, which the hello's must be (None: not held)."""
    id: str
    address: str
    secret: bytes | None = field(default=None, repr=False)
    commitment: str | None = None


def read_secret(path: str | os.PathLike) -> bytes:
    text = Path(path).read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise Ch.CoinError(f"{path}: a session's secret is 64 lowercase hex digits")
    return bytes.fromhex(text)


def write_secret(path: str | os.PathLike, secret: bytes) -> None:
    """`secret` into a new file at `path`, mode 600."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(secret.hex() + "\n")


def session(address: str | None = None, sid: str | None = None, secret_file: str | None = None,
            commitment: str | None = None) -> Session | None:
    """The window's challenger session: `address`, `sid`, `secret_file` and `commitment`, else ``$POUW_CHALLENGER``,
    ``$POUW_CHALLENGER_SESSION``, ``$POUW_CHALLENGER_SECRET_FILE`` and ``$POUW_CHALLENGER_COMMITMENT``; None (none) for
    the in-process challenger.  An address without its id, or a secret or commitment without both, is refused."""
    at = address or settings.get(CHALLENGER_ENV)
    sid = sid or settings.get(SESSION_ENV)
    secret_file = secret_file or settings.get(SECRET_ENV)
    commitment = commitment or settings.get(COMMITMENT_ENV)
    if (at is None) != (sid is None) or ((secret_file or commitment) is not None and at is None):
        raise Ch.CoinError(f"a challenger session is an id and an address from infra's gate (and its secret), not "
                           f"{sid or at or secret_file or commitment!r} alone (--challenger and --session, or "
                           f"${CHALLENGER_ENV} and ${SESSION_ENV}, with --secret-file or ${SECRET_ENV})")
    if commitment is not None and not re.fullmatch(r"[0-9a-f]{128}", commitment):
        raise Ch.CoinError(f"a coin commitment's root is 128 lowercase hex digits, not {commitment[:20]!r}…")
    return None if at is None else Session(sid, at, None if secret_file is None else read_secret(secret_file), commitment)


def layout(epoch: int, rounds: int) -> dict:
    """A live window's coin layout, the challenger session's SCHEME, EPOCH and FAMILIES."""
    return {"scheme": Wd.LIVE_SCHEME, "epoch": epoch, "families": [list(f) for f in Wd.live_families(rounds)]}


def ssh_args(key: str | None = None) -> list[str]:
    return [*SSH, *(["-o", "IdentitiesOnly=yes", "-i", key] if key else [])]


def gate_call(gate: str, args: list[str], ssh: Sequence[str] = SSH, run: Callable[..., Any] = subprocess.run) -> dict:
    """The gate's answer to `args` (``ssh GATE VERB ...``, quoted for the gate's shell words); a refusal raises."""
    p = run([*ssh, gate, shlex.join(args)], capture_output=True, text=True)
    try:
        got = json.loads(p.stdout)
    except ValueError:
        got = None
    if p.returncode or not isinstance(got, dict) or "refused" in got:
        why = got.get("refused") if isinstance(got, dict) else None
        raise Ch.CoinError(f"infra's gate refused {args[0]}: "
                           f"{why or p.stderr.strip() or p.stdout.strip() or f'exit {p.returncode}'}")
    return got


def reserve(gate: str, peer: str, epoch: int, rounds: int, minutes: int | None = None, for_lane: str | None = None,
            ssh: Sequence[str] = SSH, run: Callable[..., Any] = subprocess.run) -> dict:
    """A window's session reserved at infra's gate `gate` (``ssh GATE reserve``), its layout the window's (`layout` of
    `epoch` and `rounds`) and its challenger to listen on `peer`'s 127.0.0.1, where the window's client runs; `for_lane`
    is vy-admin's (``--for``).  The gate's answer, {session, seq, peer, by, for?, expires_utc, period}; a refusal (the
    period's cap among them) raises.  Nothing serves until `open_window`."""
    lay = layout(epoch, rounds)
    args = ["reserve", "--peer", peer, "--scheme", lay["scheme"],
            "--families", json.dumps(lay["families"], separators=(",", ":")), "--epoch", str(epoch)]
    args += ["--minutes", str(minutes)] if minutes else []
    args += ["--for", for_lane] if for_lane else []
    got = gate_call(gate, args, ssh, run)
    if not isinstance(got.get("session"), str) or not {"seq", "expires_utc"} <= got.keys():
        raise Ch.CoinError(f"infra's gate answered reserve without a session: {json.dumps(got)}")
    return got


def open_window(gate: str, sid: str, root: bytes, ssh: Sequence[str] = SSH,
                run: Callable[..., Any] = subprocess.run) -> tuple[Session, dict]:
    """Reserved session `sid` opened at infra's gate on the window's root `root` (``ssh GATE open --session S --root
    R``), after which alone its challenger opens the salt coin: the session (its challenger at the peer's
    127.0.0.1:PORT, the secret its hello carries, which only this answer holds, and the root of the coin commitment its
    challenger signed, which its hello must answer) and the gate's answer without the secret.  A refusal (no reserve, a
    second open, another lane's reserve, an expired one, a closed period, a root an open already has) raises, and so
    does an open whose challenger signed no commitment (its unit stopped first)."""
    if len(root) != 64:
        raise ValueError(f"a window's root is 64 bytes, not {len(root)}")
    got = gate_call(gate, ["open", "--session", sid, "--root", root.hex()], ssh, run)
    secret = got.pop("secret", None)
    if (got.get("session") != sid or got.get("root") != root.hex() or type(got.get("port")) is not int
            or not (isinstance(secret, str) and re.fullmatch(r"[0-9a-f]{64}", secret))):
        raise Ch.CoinError(f"infra's gate answered the open of {sid} with another session, root or port, or no secret: "
                           f"{json.dumps(got)}")
    if not re.fullmatch(r"[0-9a-f]{128}", str(got.get("commitment"))):
        raise Ch.CoinError(f"infra's gate opened {sid}, and its challenger signed no coin commitment (commitment "
                           f"{got.get('commitment')!r}): it serves no window; `status {sid}` says why")
    return Session(sid, f"127.0.0.1:{got['port']}", bytes.fromhex(secret), got["commitment"]), got


def stand_in_root(sid: str) -> bytes:
    """A development window's root for session `sid`: SHA-512(`STAND_IN_TAG` ‖ SHA-512(sid)), a stand-in and never a
    window's (a served window's is `salt.window_open_root`); its session's alone, as an open's root must be, so a check
    recomputes it from the session id."""
    return hashlib.sha512(STAND_IN_TAG + hashlib.sha512(sid.encode()).digest()).digest()


def layout_reason(commitment: Ch.LiveCommitment, epoch: int, rounds: int) -> str | None:
    """Why `commitment` is not the coin commitment of a live window of `rounds` rounds at `epoch`; None when it is."""
    want = layout(epoch, rounds)
    got = {"scheme": commitment.scheme, "epoch": commitment.epoch, "families": [list(f) for f in commitment.families]}
    if got != want:
        return f"the challenger committed to {json.dumps(got)}, not this window's {json.dumps(want)}"
    return None


class Coins:
    """A window's live coins from `challenger` (`LiveChallenger`'s commitment and open: a `RemoteChallenger`, or the
    in-process `LiveChallenger`), `source` "off-node" or "in-process": its salt and draw coins, and its rounds through
    `LiveRecord` (`rounds`)."""

    def __init__(self, challenger: Any, source: str, session: Session | None = None) -> None:
        self.challenger, self.source, self.session = challenger, source, session
        self.commitment: Ch.LiveCommitment = challenger.commitment
        self.rounds = Ch.LiveRecord(challenger, Wd.ROUND)
        self.label = (f"off-node session {session.id} at {session.address} (unchecked)" if session is not None
                      else IN_PROCESS)

    def salt(self, after: bytes) -> Ch.LiveOpening:
        return self.challenger.open(Wd.SALT, 0, after)

    def draw(self, after: bytes) -> Ch.LiveOpening:
        return self.challenger.open(Wd.DRAW, 0, after)

    def close(self) -> None:
        """End an off-node session: the challenger signs and writes its close (challenger.json), then answers closed."""
        if self.source == "off-node":
            self.challenger.close()

    def record(self) -> dict:
        s = self.session
        return {"source": self.source, "label": self.label, "challenger": s and s.address, "session": s and s.id}


class RoundPipe:
    """A window's round coins without a wait on each, after the rounds `coins.rounds` already recorded: `send(root)`
    records the next round's root, after which alone the challenger opens its coin, and `opening(i)` is round i's coin
    once it is back (`wait=False`: None until then).  Off the node the opens go out on the session's one stream without
    waiting on the answers, since `remote.serve` answers each after its own root and in order; a thread of the pipe's
    reads them back and holds each to `live_reason` (a refusal, a broken stream or a bad opening ends the session and
    refuses every later coin: a retry never redraws).  In process each opens at once.  `roots` and `openings` are
    `LiveRecord`'s, `reason` its `order_reason`, and `sent_at` / `got_at` each round's times on `clock`; `draw` opens the
    window's draw coin with its time on the same clock."""

    def __init__(self, coins: Coins, clock: Callable[[], float] = time.monotonic) -> None:
        rec = coins.rounds
        self.coins, self.commitment, self.label = coins, coins.commitment, rec.label
        self.roots, self.openings = list(rec.roots), list(rec.openings)
        self._clock = clock
        self.sent_at: dict[int, float] = {}
        self.got_at: dict[int, float] = {}
        self._got = dict(enumerate(self.openings))
        self._error: str | None = None
        self._closed = False
        self._pending: collections.deque = collections.deque()
        self._cv = threading.Condition()
        self.remote = coins.source == "off-node"
        self._thread = None
        if self.remote:
            self._thread = threading.Thread(target=self._read, name="round-coins", daemon=True)
            self._thread.start()

    def _fail(self, why: str) -> None:
        with self._cv:
            if self._error is None:
                self._error = why
            self._cv.notify_all()
        if self.remote and self.coins.challenger._ended is None:
            self.coins.challenger._end(why)

    def _read(self) -> None:
        wire = self.coins.challenger._wire
        while True:
            with self._cv:
                while not self._pending and not self._closed and self._error is None:
                    self._cv.wait()
                if not self._pending:
                    return
                i, root = self._pending[0]
            try:
                reply = wire.recv()
            except (Ch.CoinError, OSError) as e:
                return self._fail(f"the stream broke ({e})")
            if reply.get("type") == "refused" and sorted(reply) == ["reason", "type"]:
                return self._fail(f"the challenger refused: {reply['reason']}")
            if reply.get("type") != "opening" or sorted(reply) != ["opening", "type"]:
                return self._fail("the challenger did not answer with 'opening'")
            try:
                o = Ch.LiveOpening.from_json(reply["opening"])
                why = Ch.live_reason(self.commitment, o, self.label, i, root)
            except Ch.CoinError as e:
                why = str(e)
            if why is not None:
                return self._fail(f"the challenger's {self.label} coin {i}: {why}")
            with self._cv:
                self._pending.popleft()
                self._got[i] = o
                self.openings.append(o)
                self.got_at[i] = self._clock()
                self._cv.notify_all()

    def send(self, root: bytes) -> int:
        """Record round i's root (the next), its coin opened after it; i."""
        root = bytes(root)
        with self._cv:
            if self._error is not None:
                raise Ch.CoinError(f"the challenger's session ended ({self._error}): a retry never redraws")
            if self._closed:
                raise Ch.CoinError("the round coins are closed")
            if len(self.roots) == self.commitment.family(self.label):
                raise Ch.CoinError(f"every {self.label} coin of the epoch is open")
            i = len(self.roots)
            self.roots.append(root)
            self.sent_at[i] = self._clock()
            if self.remote:
                self._pending.append((i, root))
                self._cv.notify_all()
        if not self.remote:
            o = self.coins.challenger.open(self.label, i, root)
            with self._cv:
                self._got[i] = o
                self.openings.append(o)
                self.got_at[i] = self._clock()
            return i
        try:
            self.coins.challenger._wire.send({"type": "open", "label": self.label, "index": i, "after": root.hex()})
        except OSError as e:
            self._fail(f"the stream broke ({e})")
            raise Ch.CoinError(f"the stream broke ({e})") from e
        return i

    def opening(self, i: int, wait: bool = True) -> Ch.LiveOpening | None:
        with self._cv:
            while i not in self._got and self._error is None and wait:
                if i >= len(self.roots):
                    raise Ch.CoinError(f"{self.label} coin {i}'s root is not recorded")
                self._cv.wait()
            if i in self._got:
                return self._got[i]
            if self._error is not None:
                raise Ch.CoinError(f"the challenger's session ended ({self._error}): a retry never redraws")
            return None

    def draw(self, after: bytes) -> tuple[Ch.LiveOpening, float]:
        """The window's draw coin opened after `after` once every sent round's coin is back, and the time it was asked
        for on `clock`, read under the lock `send` records each round's root and time under and held through the draw:
        every round sent before the draw has an earlier time and every round sent after it a later one, whatever thread
        sends it."""
        with self._cv:
            while self._pending and self._error is None:
                self._cv.wait()
            if self._error is not None:
                raise Ch.CoinError(f"the challenger's session ended ({self._error}): a retry never redraws")
            if self._closed:
                raise Ch.CoinError("the round coins are closed")
            at = self._clock()
            return self.coins.draw(after), at

    def reason(self) -> str | None:
        return Ch.order_reason(self.commitment, self.label, self.roots, self.openings)

    def close(self) -> None:
        """Every sent round's coin back, then the session closed (`Coins.close`)."""
        with self._cv:
            while self._pending and self._error is None:
                self._cv.wait()
            self._closed = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join()
        if self._error is None:
            self.coins.close()


def open_coins(at: Session | None, epoch: int, rounds: int, entropy: Callable[[int], bytes] | None = None,
               connect: Callable[..., Any] | None = None, wait_s: float = R.TUNNEL_WAIT_S,
               clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> Coins:
    """The window's coins: the challenger of session `at`, its hello carrying the session's secret when it has one, its
    commitment held to `layout` and to the root the gate's open answered when `at` holds one, or with no session the
    in-process challenger over `entropy` (the Lean verifier of record's coins by default).  The client connects once
    the session's tunnel listens (`remote.connect_when_listening` over `connect`, for up to `wait_s`: a caller that knows
    the window's coin deadline passes the time left to it)."""
    if at is None:
        if entropy is None:
            from verity.core.protocols.pouw import verifier
            entropy = verifier.coins
        return Coins(Wd.live_challenger(epoch, rounds, entropy), "in-process")
    client = R.connect_when_listening(at.address, at.secret, connect, wait_s, clock, sleep)
    why = layout_reason(client.commitment, epoch, rounds)
    if why is None and at.commitment is not None and client.commitment.root.hex() != at.commitment:
        why = (f"session {at.id}'s challenger answered the hello with commitment {client.commitment.root.hex()[:16]}…, "
               f"not the {at.commitment[:16]}… infra's gate opened it with")
    if why is not None:
        client.close()
        raise Ch.CoinError(why)
    return Coins(client, "off-node", at)


def lms_verify(key: Ch.ChallengerKey, message: bytes, signature: bytes) -> bool:
    from verity.ml.crypto import lms
    p = lms.Params(64, key.W, key.H)
    try:
        return lms.verify(p, key.public, message, lms.Signature.decode(p, signature))
    except ValueError:
        return False


def _openings(docs: list[dict]) -> list[Ch.LiveOpening]:
    return [Ch.LiveOpening.from_json(d) for d in docs]


def published(directory: str | os.PathLike) -> Path:
    """The challenger's out/ of a session as infra's gate publishes it (``fetch SESSION``): `directory`'s out/, beside
    its reserve's request.json and its open's open.json, or `directory` when it is that out/."""
    d = Path(directory)
    return d / "out" if (d / "out").is_dir() else d


def audited_window(directory: str | os.PathLike, heads: Sequence[str | os.PathLike], root: bytes, key: Ch.ChallengerKey,
                   pins: Any, main: Any
                   ) -> tuple[Ch.LiveCommitment, dict[str, tuple[bytes, ...]], Ch.Audited, dict[str, dict]]:
    """Window `root`'s session as core's verifier takes it from the session `directory` (`published`) and every head of
    the auditor's ledger infra published, `heads`, under the challenger key `key` (`remote.trusted_session`): the coin
    commitment the challenger signed, the roots its signed release log shows it received, the session the ledger opens
    on `root`, run from a commit the host's pins log `pins` (its lines, the one infra published) records on the
    verifier's own first-parent history of main `main` (`challenger.pinned_session`), and that session's entries in the
    ledger by kind.  Raises `CoinError` naming why the ledger, the session or its pin is refused."""
    window = {"root": root.hex()}
    commitment, roots = R.trusted_session(published(directory), heads, window, key, lms_verify)
    ledger = Ch.ledger_head([R.ledger_lines(p) for p in heads], key, lms_verify)
    audited = Ch.pinned_session(ledger, window, key, pins, main, lms_verify)
    entries = {e["kind"]: e for e in map(json.loads, ledger) if e["session"] == audited.session}
    return commitment, roots, audited, entries


def ledger_info(audited: Ch.Audited, entries: dict[str, dict]) -> dict:
    """What the auditor's ledger says of a window's session (`audited_window`): its id and prover, its entries' seqs,
    whether its end is complete (every family released all its coins), its refused hellos, and the period holding it
    with that period's cap and circuit."""
    return {"session": audited.session, "prover": audited.prover,
            **{k: entries[k]["seq"] if k in entries else None for k in Ch.SESSION_KINDS},
            "end_complete": entries["end"]["body"].get("complete"), "refused_hellos": audited.refused_hellos,
            "period": audited.period, "cap": audited.cap, "circuit": audited.circuit}


def window_reasons(audited: Ch.Audited, entries: dict[str, dict], commitment: Ch.LiveCommitment, sid: Any,
                   window: dict, root: bytes, period: str | None) -> dict[str, str | None]:
    """Why window `root`'s session as the auditor's ledger names it (`audited_window`) is not the one the manifest names
    (`sid`, its `window`), the open's layout isn't `commitment`'s, or its period isn't `period` when the verifier names
    one: by name, None where it holds.  The rest of the ledger's rules are core's (`challenger.audited_session`)."""
    lay = {"scheme": commitment.scheme, "epoch": commitment.epoch, "families": [list(f) for f in commitment.families]}
    asked = {k: entries["open"]["body"].get(k) for k in lay}
    return {
        "session": None if sid == audited.session else
        f"the manifest's session is {sid!r}, not {audited.session}, the one the auditor's ledger opens on the window's "
        f"root {root.hex()[:16]}…",
        "window": None if window.get("root") in (None, root.hex()) and window.get("period") in (None, audited.period)
        else f"the manifest's window {json.dumps(window)} is not the session's (root {root.hex()[:16]}…, period "
             f"{audited.period!r})",
        "layout": None if asked == lay else
        f"session {audited.session}'s open asked for {json.dumps(asked)}, not the signed commitment's {json.dumps(lay)}",
        "period": None if period is None or audited.period == period else
        f"session {audited.session} is a window of period {audited.period!r}, not {period!r}",
    }


def key_record(path: str | os.PathLike) -> Ch.ChallengerKey:
    """The challenger's key from its deployment record at `path` (infra's challenger-key/v1, read by core's
    `challenger.key_of_record`), the key a verifier holds a session to, never the session's own key.json nor the
    ledger's key_sha512."""
    return Ch.key_of_record(json.loads(Path(path).read_text()))


def check(directory: str | os.PathLike, coins: dict, heads: Sequence[str | os.PathLike], pins: Any, main: Any,
          rounds: list[bytes], salts: dict[int, str] | None = None, draw_after: bytes | None = None,
          key: Ch.ChallengerKey | None = None, period: str | None = None, root: bytes | None = None) -> dict:
    """The verifier's check of a window's coins against the session `directory` as infra's gate publishes it
    (`published`) and every head of the auditor's ledger infra published, `heads`, read from the challenger's own run,
    with the host's pins log infra published `pins` (its lines) and the verifier's own first-parent history of main
    `main`, which the session's pin must be on (`audited_window`).
    `coins` is the manifest's ``coins`` item (the openings the prover received, the session id and the window's root and
    period), and the rest is what the verifier recomputes from the window's committed roots: the round roots `rounds` in
    order, round r's unit salt `salts[r]` (hex) where it holds them, and `draw_after`, the root the draw coin follows
    (None: the window opened no draw); `key` is the challenger's key from its deployment record (`key_record`), `period`
    the window's period and `root` the window's root, each None while the verifier names none (`root` then the
    manifest's).  It holds: key.json is `key`; core's verifier takes the session the ledger opens on the window's root
    under it (`audited_window`); that session and its period are the manifest's, its open asked for the signed layout,
    and its period is `period` (`window_reasons`); the window's commitment is the record's, a live epoch's; the salt coin
    opened after the window's root; the challenger received exactly `rounds`, and round r's coin opened once after round
    r's root (`rounds_reason` over the challenger's roots); each salt is `h1tile.live_salt` of the opened coin; the draw
    coin opened after `draw_after`, or none opened.  ``ok`` only when all hold."""
    from verity.core.protocols.pouw.circuit import h1tile as HT
    d = published(directory)
    sid, window = coins.get("session"), coins.get("window") or {}
    out: dict[str, Any] = {"ok": False, "coins": "refused", "challenger_out": str(d), "session": sid}
    try:
        theirs = Ch.ChallengerKey.from_json(json.loads((d / R.KEY).read_text()))
        host = Ch.ChallengerRecord.from_json(json.loads((d / R.RECORD).read_text())).host
    except (OSError, ValueError) as e:
        return {**out, "reason": f"the session's key or record is unreadable: {e}"}
    out.update(host=host, key_sha512=hashlib.sha512(theirs.public).hexdigest(), window=window or None)
    if key is not None and theirs != key:
        return {**out, "reason": f"the session's key is sha512 {out['key_sha512'][:16]}..., not the deployment record's "
                                 f"{hashlib.sha512(key.public).hexdigest()[:16]}..."}
    loose = [name for name, anchor in (("key", key), ("root", root), ("period", period)) if anchor is None]
    key = key or theirs
    if root is None and not re.fullmatch(r"[0-9a-f]{128}", str(window.get("root"))):
        return {**out, "reason": "the verifier names no window root, and the manifest's window none"}
    root = root if root is not None else bytes.fromhex(window["root"])
    try:
        commitment, roots, audited, entries = audited_window(d, heads, root, key, pins, main)
        mine = Ch.LiveCommitment.from_json(coins.get("commitment"))
        salt = _openings([coins["salt_coin"]])
        openings = _openings([r["opening"] for r in coins.get("rounds", [])])
        draw = _openings([coins["draw_coin"]]) if coins.get("draw_coin") else []
    except (KeyError, TypeError, Ch.CoinError) as e:
        return {**out, "reason": f"{type(e).__name__}: {e}"}
    got = roots[Wd.ROUND]
    bad_salts = sorted(r for r, s in (salts or {}).items()
                       if not (0 <= r < min(len(got), len(openings)))
                       or HT.live_salt(commitment, openings[r].coin, r, got[r]).hex() != s)
    reasons = {
        **window_reasons(audited, entries, commitment, sid, window, root, period),
        "commitment": None if mine == commitment else "the window's coin commitment is not the one the challenger signed",
        "epoch": Wd.live_epoch_reason(commitment),
        "salt": (None if roots[Wd.SALT] == (root,) else "the challenger's salt root is not the window's root")
        or Ch.order_reason(commitment, Wd.SALT, roots[Wd.SALT], salt),
        "rounds": (None if got == tuple(rounds) else
                   f"the challenger received {len(got)} round roots, not the window's {len(rounds)} committed ones")
        or Wd.rounds_reason(commitment, got, openings),
        "salts": None if not bad_salts else f"rounds {bad_salts[:8]}'s unit salts are not their opened coins' live_salt",
        "draw": (None if roots[Wd.DRAW] == (() if draw_after is None else (draw_after,)) else
                 "the challenger's draw roots are not the window's")
        or (Ch.order_reason(commitment, Wd.DRAW, roots[Wd.DRAW], draw) if draw_after is not None else None),
    }
    out.update(reasons=reasons, ledger=ledger_info(audited, entries), root=root.hex(), rounds=len(got), draws=len(roots[Wd.DRAW]),
               commitment=commitment.root.hex())
    out["ok"] = all(v is None for v in reasons.values())
    out["coins"] = (f"off-node {host} (checked{'; ' + ', '.join(loose) + ' unanchored' if loose else ''})"
                    if out["ok"] else "refused")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("families", help="the challenger session's FAMILIES for a window of --rounds rounds")
    f.add_argument("--rounds", type=int, required=True)
    o = sub.add_parser("open", help="reserve a development window's session at infra's gate and open it on its stand-in "
                                    "root; prints the gate's answers (never the secret) and the client's env")
    o.add_argument("--gate", default=settings.get(GATE_ENV), help=f"infra's gate, USER@HOST (default ${GATE_ENV})")
    o.add_argument("--ssh-key", help="the lane's own gate key (default: ssh's own); a key the gate reads only is refused")
    o.add_argument("--peer", required=True, help="the node the window's client runs on, as the gate names it")
    o.add_argument("--epoch", type=int, required=True)
    o.add_argument("--rounds", type=int, required=True)
    o.add_argument("--minutes", type=int)
    o.add_argument("--for", dest="for_lane", help="vy-admin's: the lane a window is reserved for")
    o.add_argument("--secret-file", required=True,
                   help="a new file the session's secret is written to (mode 600), which the client's hello carries")
    c = sub.add_parser("check", help="a window's coins against the challenger's own run")
    c.add_argument("--challenger-out", required=True, help="the session as infra's gate publishes it (request.json, "
                   "open.json and out/), or its out/")
    c.add_argument("--ledger", action="append", required=True, help="a head of the auditor's ledger as infra published "
                   "it (ssh GATE ledger, or its published challenger-ledger head); once for each head the verifier holds")
    c.add_argument("--pins", type=Path, required=True, help="the challenger host's pins log as infra published it, a "
                   "line a pin (its `challenger-pins head` label)")
    c.add_argument("--main", type=Path, required=True, help="the verifier's own `git rev-list --first-parent "
                   "origin/main`, a commit a line")
    c.add_argument("--manifest", required=True, help="the window's manifest.json")
    c.add_argument("--key-record", help="the challenger's key record from its deployment (challenger-key/v1, an art: "
                   "fetched with `research data fetch`)")
    c.add_argument("--period", help="the window's period, as the verifier knows it")
    c.add_argument("--root", help="the window's root as the verifier recomputes it (128 hex digits; default the "
                   "manifest's, unanchored)")
    a = ap.parse_args(argv)
    if a.cmd == "families":
        print(json.dumps(layout(0, a.rounds)["families"], separators=(",", ":")))
        return 0
    if a.cmd == "open":
        if not a.gate:
            ap.error(f"open needs --gate or ${GATE_ENV}")
        if Path(a.secret_file).exists():
            ap.error(f"--secret-file {a.secret_file} exists: the session's secret goes to a new file")
        ssh = ssh_args(a.ssh_key)
        got = reserve(a.gate, a.peer, a.epoch, a.rounds, a.minutes, a.for_lane, ssh)
        print(json.dumps({"reserve": got}), flush=True)
        try:
            at, opened = open_window(a.gate, got["session"], stand_in_root(got["session"]), ssh)
        except Ch.CoinError as e:
            raise Ch.CoinError(f"session {got['session']} is reserved and its open failed ({e}); a reserve unopened by "
                               f"{got['expires_utc']} is a refused window of period {got.get('period')!r}") from e
        write_secret(a.secret_file, at.secret)
        print(json.dumps({"open": opened, "root_is": STAND_IN}), flush=True)
        print(f"{CHALLENGER_ENV}={at.address} {SESSION_ENV}={at.id} {SECRET_ENV}={a.secret_file} "
              f"{COMMITMENT_ENV}={at.commitment}")
        return 0
    import importlib.util
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    spec = importlib.util.spec_from_file_location("served_h1_serve", here / "serve.py")
    SV = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(SV)
    m = json.loads(Path(a.manifest).read_text())
    key = key_record(a.key_record) if a.key_record else None
    pins = [x for x in a.pins.read_text().split("\n") if x]
    got = check(a.challenger_out, m.get("coins", {}), a.ledger, pins, R.main_lines(a.main), **SV.coins_expect(m),
                key=key, period=a.period, root=bytes.fromhex(a.root) if a.root else None)
    print(json.dumps(got, indent=1))
    return 0 if got["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
