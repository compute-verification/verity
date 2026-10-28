"""A5's draw from the auditor's challenger (`a5.py --challenger`): one challenger session for the TP step, opened on its TP run root
of record, from whose coins each rank's registration draws once (`sampled_proofs.coins.ChallengerCoins` over a
`remote.RemoteChallenger`), and the evidence its verifier checks every draw's coins against once the session has ended.

* **The session.** ``local``: a development session on this machine, run by the auditor host's own service
  (`benchmarks/live_challenger/auditor.py`'s reserve, admit and run, then `serve.py`, with no tunnel) on a scratch root under an
  LMS key made there at `LOCAL_PARAMS`, never under the run's out/: its ledger and the session's out/ are what the challenger host
  writes, and its key is this run's alone, so its draws show the path, not an auditor's coins. Its pins log pins main's
  first-parent commit this tree is built on (`on_main`), as pin.sh pins the challenger host only to such a commit; its code is this
  tree's. ``loopback:HOST:PORT``: the same development session on the challenger party's host
  (`challenger_party.py`, `Loopback`), asked for its open and finish over loopback TCP. ``gate:USER@HOST``: the challenger host through infra's gate with the lane's gate key: `reserve` with this machine as its
  peer, `open` on the TP run root (`served_h1.coins.open_window`), the client on the peer's 127.0.0.1:PORT with the session's
  secret, and once the session has ended its `status`, `fetch`, `ledger` and `pins`; the verifier's key is then the
  deployment's record (`served_h1.coins.key_record`), never the session's key.json. The layout is `SCHEME` with one family
  `FAMILY`, sized from the draws before anything opens (`families`): one coin, then each rank's `need`, at least
  `COINS_PER_RANK` coins. The session opens once every rank has passed its checks before the draw and planned it
  (`Draws.plan`) and the layout covers every need (`reserve_reason`). Otherwise nothing opens and every rank's draw is
  refused, so a refusal before the draw never spends the root's one session.
* **The draws** (`Draws`, in a `multiprocessing` manager's process, so every rank's process reads the one client): first one
  coin after the window's root itself, before any registration (`pre`: the negatives' coin opened before the registration),
  then each rank's draw once, by its own `service.Stream` on `RankCoins`: the stream registers the rank's window, the coins
  commit to the registration's digest (`ChallengerCoins.commit`), and the stream's `select` reads them, in the rank's turn
  on the one client until `drawn` returns its segment. Once every rank has drawn or won't (`skip`), the client closes the
  session. `Replay` reads an honest draw's bytes again, for the negatives' streams.
* **The evidence** (`Local.finish` or `Gate.finish`, on a thread of the run's process once the session closed): the session's out/,
  the auditor's ledger and the host's pins log, copied to ``out/challenger/``, the key, the verifier's own copy of main's
  first-parent history (`start`'s ``main``, never the host's or the prover's), `remote.trusted_session`'s check of the files the
  ledger's end sealed and `challenger.pinned_session`'s of the open's pin. `evidence` is one draw's `{key, ledger, main, pins,
  segment, session}` (`coins.coins_reason`'s, and the Lean run's), and `negatives` its altered copies.
* **The verifier's own coins** (`a5.py --coins verifier`): no session. Each rank's `service.Stream` draws on `VerifierCoins`, the
  service's default coins (`verifier.coins`: the Lean verifier of record's `IO.getRandomBytes`, assumption A3), read by its
  `select` once the stream has received the rank's registration and written its receipt. The stream's coin source is then
  the verifier, its draws carry no evidence, and the Lean's run takes none (`coin_source` verifier). No auditor's device
  draws them: they show the record path, not an auditor's coins. `verifier_negatives` are the refusals that apply to them.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import secrets
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from multiprocessing import get_context
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any

from verity.core.protocols.sampled_proofs import challenger as Ch, coins as CC, remote as R, service as S
from verity.core.protocols.sampled_proofs.one_stage import draw as D, verifier as V
from verity.ml.circuits.program import Program
from verity.ml.crypto import lms

TREE = Path(__file__).resolve().parents[2]
SCHEME = "verity/one-stage/a5-draw/v1"
FAMILY = "draw"
COINS_PER_RANK = 16
LOCAL_PARAMS = (4, 5)
LOCAL_HOST, LOCAL_PEER = "local-challenger", "local"
MINUTES = 240


def _module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def served_h1():
    """PoUW's served window's coins (`benchmarks/pouw/served_h1/coins.py`): the gate's calls and the key record."""
    return _module("a5_served_h1_coins", TREE / "benchmarks/pouw/served_h1/coins.py")


def need(rule: str, program: Program, query: Mapping[str, Any], population: int,
         work_table: Mapping[str, Any] | None = None) -> int:
    """The most coins one rank's draw reads, known from its window before anything opens. The window gives the stream's law:
    the rule over its N (`service.selection_law`), expanded over its program and query (`draw.derive`) under the work table
    the rank registers with (a ``work`` law's), as `Stream.register` expands it. The stream's own sampler (`service.select_draw`) draws that law on bytes that reject no
    word (zeros: a zero word is below every largest multiple), and the need is what it reads plus its longest stream once
    more. `coins.draw_first` reads each stream (the law's one, or one per stratum) once, 2 k width(n) + 64 bytes. It reads
    a stream again only when the stream's words run out, which takes k + 1 of them rejected, each with probability below
    2^-64 (`coins.OwnRandomness`). The margin covers one such rerun, so a draw reads past its need only after two."""
    selection = S.Law(rule)
    law = D.derive(program, query, S.selection_law(selection, population), work_table)
    reads: list[int] = []

    def zeros(n: int) -> bytes:
        reads.append(n)
        return bytes(n)
    S.select_draw(selection, law, population, zeros)
    return -(-(sum(reads) + max(reads, default=0)) // Ch.COIN_BYTES)


def families(needs: Mapping[int, int]) -> list[list[Any]]:
    """The session's layout for the draws `needs` (rank -> `need`): the family `FAMILY`, with the coin after the window's
    root (`Draws.pre`) and each rank's need, at least `COINS_PER_RANK` coins."""
    return [[FAMILY, 1 + sum(max(COINS_PER_RANK, n) for n in needs.values())]]


def reserve_reason(layout: list, needs: Mapping[int, int]) -> str | None:
    """Why the challenger's reserve `layout` can't cover the coin after the window's root and every rank's draw (`needs`,
    rank -> `need`), naming the need and the reserve; None when it can."""
    want = 1 + sum(needs.values())
    why = Ch.layout_problem(SCHEME, 0, tuple(tuple(f) for f in layout))
    if why is not None:
        return f"the draws need {want} {FAMILY} coins, and the challenger can't reserve {layout}: {why}"
    have = dict(tuple(f) for f in layout).get(FAMILY, 0)
    if have < want:
        per_rank = ", ".join(f"rank {r} {n}" for r, n in sorted(needs.items()))
        return (f"the draws need {want} {FAMILY} coins (1 after the window's root, then {per_rank}), and the reserve "
                f"{layout} holds {have}")
    return None


def on_main(main: list[str]) -> str:
    """The commit of main's first-parent line `main` (newest first) this tree is built on: the newest one in its history (a
    branch that merged another branch's commit before it landed has a merge base with main off the line); else this tree's
    HEAD, whose session `pinned_session` then refuses."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(TREE), *args], capture_output=True, text=True).stdout.strip()
    built = set(git("rev-list", "HEAD").split())
    head = git("rev-parse", "HEAD")
    return next((c for c in main if c in built), head if len(head) == 40 else "0" * 40)


def lines(path: Path) -> list[str]:
    return [x for x in path.read_text().split("\n") if x]


#: the proof service's rule for a client of a session the gate has just opened (`remote.connect_when_listening`)
TUNNEL_WAIT_S, connect_when_listening = R.TUNNEL_WAIT_S, R.connect_when_listening


class Draws:
    """The step's coins over the window root `window_root` for every rank of `ranks`. Each rank plans its draw (`plan`).
    Once every rank has planned or ended (`wait_planned`), the session either opens (`connect`: its client, one
    `ChallengerCoins` for every rank) or is refused (`refuse`). The session's evidence follows once published. A rank's
    draw holds the client from its `commit` to its `drawn` (its turn), so no other rank's coins open between its commitment
    and its reads."""

    def __init__(self, window_root: str, ranks: list[int]) -> None:
        self._root = window_root
        self._lock, self._left, self._closed, self._ready = threading.Lock(), set(ranks), threading.Event(), threading.Event()
        self._unplanned, self._needs, self._planned = set(ranks), {}, threading.Event()
        self._opened, self._refused = threading.Event(), None
        self._client = self._coins = self._pre = None
        self._turn, self._current, self._segment = threading.Lock(), None, None
        self._ended: str | None = None
        self._evidence: dict | None = None
        if not ranks:
            self._planned.set()

    def plan(self, rank: int, coins: int) -> None:
        """Rank `rank`'s draw reads at most `coins` coins (`need`). It returns once the session has opened, and raises
        `CoinError` when none opened (`refuse`)."""
        with self._lock:
            if rank not in self._unplanned:
                raise Ch.CoinError(f"rank {rank} has planned its draw, or ended: a rank plans once, before its draw")
            self._needs[rank] = coins
            self._settled(rank)
        self._opened.wait()
        if self._refused is not None:
            raise Ch.CoinError(self._refused)

    def _settled(self, rank: int) -> None:
        self._unplanned.discard(rank)
        if not self._unplanned:
            self._planned.set()

    def wait_planned(self) -> dict[int, int]:
        """Once every rank has planned its draw or ended: each planned rank's coins (a rank that ended first has none)."""
        self._planned.wait()
        return dict(self._needs)

    def connect(self, address: str, secret: str | None, commitment: str | None) -> None:
        """The opened session's client at `address` (its hello with `secret`, its commitment the gate's open's `commitment`
        when given), then the one coin after the window's root itself, before any rank registers. Raises `CoinError`,
        refusing every rank's draw, when either fails."""
        client = None
        try:
            client = connect_when_listening(address, None if secret is None else bytes.fromhex(secret))
            if commitment is not None and client.commitment.root.hex() != commitment:
                raise Ch.CoinError(f"the challenger's hello answered commitment {client.commitment.root.hex()[:16]}…, not "
                                   f"the gate's {commitment[:16]}…")
            coins = CC.ChallengerCoins(Ch.LiveRecord(client, FAMILY))
            pre = coins.commit(bytes.fromhex(self._root))
            coins(Ch.COIN_BYTES)
        except Exception as e:  # every waiting rank's plan must end, whatever failed
            if client is not None:
                try:
                    client.close()
                except Ch.CoinError:
                    pass
            self.refuse(f"the challenger's session opened, and its client failed: {e}")
            raise Ch.CoinError(str(e)) from e
        self._client, self._coins, self._pre = client, coins, pre.to_json()
        self._opened.set()
        with self._lock:
            if not self._left and self._current is None:
                self._close()

    def refuse(self, why: str) -> None:
        """No session for the draws (`why`): every rank's `plan` and `commit` raise it."""
        self._refused = why
        self._opened.set()
        self._closed.set()

    def commit(self, rank: int, after: str) -> None:
        """Rank `rank`'s turn, once: the coins after its registration's digest `after` (hex), which its reads take until
        its `drawn`."""
        if self._refused is not None:
            raise Ch.CoinError(self._refused)
        if self._coins is None:
            raise Ch.CoinError(f"rank {rank} commits before the session opened: a rank plans its draw first (Draws.plan)")
        self._turn.acquire()
        self._current, self._segment = rank, None
        try:
            with self._lock:
                if rank not in self._left:
                    raise Ch.CoinError(f"rank {rank} has drawn, or won't: a rank's registration draws once")
                self._left.discard(rank)
            self._segment = self._coins.commit(bytes.fromhex(after))
        except BaseException:
            self.drawn(rank)
            raise

    def read(self, rank: int, n: int) -> bytes:
        if self._current != rank or self._segment is None:
            raise Ch.CoinError(f"rank {rank} reads coins only in its turn, after its commitment")
        return self._coins(n)

    def drawn(self, rank: int) -> dict | None:
        """The end of rank `rank`'s turn: its segment (None when it committed none). No-op outside its turn."""
        if self._current != rank:
            return None
        segment = None if self._segment is None else self._segment.to_json()
        self._current, self._segment = None, None
        self._turn.release()
        with self._lock:
            if not self._left:
                self._close()
        return segment

    def skip(self, rank: int) -> None:
        """Rank `rank` draws nothing (refused before its draw, or its process failed), ending its turn if it holds one: no-op
        once it has drawn."""
        self.drawn(rank)
        with self._lock:
            self._left.discard(rank)
            if rank in self._unplanned:
                self._settled(rank)
            if not self._left and self._current is None and self._client is not None:
                self._close()

    def _close(self) -> None:
        if self._closed.is_set():
            return
        try:
            if self._client is not None:
                self._client.close()
        except Ch.CoinError as e:
            self._ended = str(e)
        self._closed.set()

    def pre(self) -> dict:
        return self._pre

    def wait_closed(self) -> str | None:
        """Once the client has closed the session: None, or why its close failed."""
        self._closed.wait()
        return self._ended

    def publish(self, evidence: dict) -> None:
        self._evidence = evidence
        self._ready.set()

    def evidence(self) -> dict:
        """The session's evidence once it has ended (`Session.finish`): {key, ledger, session, trusted} or {error}, each
        with the `plan` it opened on (`start`)."""
        self._ready.wait()
        return self._evidence


class _Manager(BaseManager):
    pass


_Manager.register("Draws", Draws)


class RankCoins(CC.ChallengerCoins):
    """Rank `rank`'s coins of the step's session (`draws`, the `Draws` proxy) as its `service.Stream` reads them: a
    `ChallengerCoins`, so the stream's coin source is the challenger, whose commitment and reads go to the one client in
    `Draws`, in the rank's turn until `drawn`. `read` is every byte it read, `need` the coins it planned (`plan`)."""

    def __init__(self, draws: Any, rank: int) -> None:
        self._draws, self._rank, self.read, self.need = draws, rank, bytearray(), None

    def plan(self, coins: int) -> None:
        """The draw reads at most `coins` coins (`need`): once every rank has planned, the session opens (`Draws.plan`)."""
        self.need = coins
        self._draws.plan(self._rank, coins)

    def commit(self, after: bytes) -> None:
        self._draws.commit(self._rank, bytes(after).hex())

    def __call__(self, n: int) -> bytes:
        b = self._draws.read(self._rank, n)
        self.read.extend(b)
        return b

    def drawn(self) -> dict | None:
        return self._draws.drawn(self._rank)


class Replay(CC.ChallengerCoins):
    """The bytes an honest draw read from the challenger, read again in order: a negative's `service.Stream` on them draws
    the honest draw's units, so its outcome differs from the honest one in its coins' evidence alone."""

    def __init__(self, read: bytes) -> None:
        self._read, self._at = bytes(read), 0

    def __call__(self, n: int) -> bytes:
        if self._at + n > len(self._read):
            raise Ch.CoinError("the replayed draw reads past the honest draw's bytes")
        self._at += n
        return self._read[self._at - n:self._at]


class VerifierCoins:
    """The verifier's own live coins as a rank's `service.Stream` reads them (`source`, default `verifier.coins`: the Lean's
    `IO.getRandomBytes`, the stream's default coins). Not a `ChallengerCoins`, so the stream's coin source is the verifier.
    `read` is every byte it read."""

    def __init__(self, source: Callable[[int], bytes] = V.coins) -> None:
        self.source, self.read = source, bytearray()

    def __call__(self, n: int) -> bytes:
        b = self.source(n)
        self.read.extend(b)
        return b


class OwnReplay:
    """The bytes an honest draw read from the verifier's own coins, read again in order, as the verifier's own (no
    `ChallengerCoins`): a negative's stream on them draws the honest draw's units. `at` is how many it has read."""

    def __init__(self, read: bytes) -> None:
        self._read, self.at = bytes(read), 0

    def __call__(self, n: int) -> bytes:
        if self.at + n > len(self._read):
            raise Ch.CoinError("the replayed draw reads past the honest draw's bytes")
        self.at += n
        return self._read[self.at - n:self.at]


#: Evidence of the challenger's form that a prover might attach to a draw from the verifier's own coins.
FOREIGN_EVIDENCE = {"key": None, "ledger": [], "main": [], "pins": [], "segment": None, "session": {}}
SELECT_COINS = ("select", ["coins"])


def lean_side(audit: dict | None) -> dict | None:
    """The Lean's side of an outcome's audit record where its whole run (`Run.run`) disagreed: its step and codes when
    it refused one."""
    return ((((audit or {}).get("checks") or {}).get("run") or {}).get("detail") or {}).get("disagree", {}).get("lean")


def verifier_negatives(stream_of: Callable[[Callable[[int], bytes]], S.Stream],
                       register: Callable[[S.Stream], S.Registration], honest: S.Stream, coins: VerifierCoins,
                       draw: S.Draw, outcome: Callable[[S.Stream, S.Draw, dict | None], tuple[S.Outcome, dict | None]],
                       window: str, closure: dict[int, list[int]] | None = None) -> dict[str, dict]:
    """The refusals that apply to a draw `draw` from the verifier's own `coins` on stream `honest`, once its outcome is
    final. `stream_of(coins)` is a fresh stream of the same Spec on `coins`, `register(stream)` receives the honest
    registration on it, and `outcome(stream, draw, coins)` delivers the honest sessions and decides (the outcome, its
    audit record); `window` is the registration's window root, and `closure` the closure map the honest draw was selected
    under (`Stream.select`), which every select here passes too.

    * ``verifier-coins-claimed-as-challenger``: the honest draw's bytes passed off as the auditor's challenger's coins
      (`Replay`) with no evidence: `coins.coins_reason` refuses, and the Lean's run refuses at its select (`coins`).
    * ``verifier-coins-with-evidence``: the honest draw's bytes as the verifier's own with evidence of the challenger's
      form attached: a draw from the verifier's own coins carries none, and the Lean's run refuses at its select
      (`coins`).
    * ``verifier-coins-redraw``: a second select on the honest registration: the stream refuses it (`drawn`), and reads
      no coin; nothing redraws.
    * ``verifier-coins-before-the-registration``: a select on a registration the stream never received: refused
      (`unknown`) before any coin is read, so the verifier's coins are read only after the stream's receipt.

    -> {name: {refused, ...}}, refused only where both the stream (or the Lean's run) and, for the evidence, Python
    refuse for that reason."""
    neg: dict[str, dict] = {}
    read, after = bytes(coins.read), bytes.fromhex(draw.registration.digest)
    for name, replay, evidence in (("verifier-coins-claimed-as-challenger", Replay(read), None),
                                   ("verifier-coins-with-evidence", OwnReplay(read), FOREIGN_EVIDENCE)):
        s = stream_of(replay)
        try:
            d = s.select(register(s), closure=closure)
            if d.draw != draw.draw:
                raise ValueError(f"the replayed draw {d.units} is not the honest draw {draw.units}")
            got, audit = outcome(s, d, evidence)
            accepted, side = got.accepted, lean_side(audit)
        except (S.Refused, S.Pending, ValueError, Ch.CoinError) as e:
            accepted, side = None, {"refused": _refusal(e)}
        at = ((side or {}).get("step"), (side or {}).get("codes"))
        row = {"accepted": accepted, "lean": side, "source": s.coin_source}
        if isinstance(replay, Replay):
            row["python"] = CC.coins_reason(evidence, window, after, read, lms_verify)
        neg[name] = {"refused": accepted is False and at == SELECT_COINS and row.get("python", "") is not None, **row}
    neg["verifier-coins-redraw"] = stream_refusal(lambda: honest.select(draw.registration, closure=closure), "drawn",
                                                  lambda: len(coins.read))
    fresh = OwnReplay(read)
    neg["verifier-coins-before-the-registration"] = stream_refusal(
        lambda: stream_of(fresh).select(draw.registration, closure=closure), "unknown", lambda: fresh.at)
    return neg


def stream_refusal(call: Callable[[], Any], code: str, read_count: Callable[[], int]) -> dict:
    """`call()`, a stream's select, refused for `code` alone with no coin read (`read_count()` unchanged)."""
    before = read_count()
    try:
        call()
        codes = None
    except S.Refused as e:
        codes = list(e.verdict.codes)
    read_now = read_count() - before
    return {"refused": codes == [code] and read_now == 0, "stream": codes, "coins_read": read_now}


def _refusal(e: Exception) -> dict:
    return e.verdict.to_json() if isinstance(e, S.Refused) else {"pending": e.args[0] if e.args else None} \
        if isinstance(e, S.Pending) else {"stream": str(e)}


class Local:
    """A development session on this machine (module docstring), its auditor host's root in scratch `root`, pinned on the
    verifier's copy of main's first-parent line `main`."""

    def __init__(self, root: Path, main: list[str], by: str = "circuits") -> None:
        A = _module("a5_auditor", TREE / "benchmarks" / "live_challenger" / "auditor.py")
        self.A, self.by = A, by
        self.host = A.Host(root=root / "var", peers=root / "peers", name=LOCAL_HOST, pins=root / "pins.jsonl")
        (root / "var" / "sessions").mkdir(parents=True)
        (root / "peers").write_text(f"{LOCAL_HOST} research@127.0.0.1\n{LOCAL_PEER} research@127.0.0.1\n")
        (root / "coin-ports").write_text(f"{LOCAL_PEER} {A.PORTS.start}-{A.PORTS.stop - 1}\n")
        self.pin = on_main(main)
        self.host.pins.write_text(A.pin_line(self.pin, main[0] if main else self.pin, A.utc(datetime.now(timezone.utc))) + "\n")
        self.params = lms.Params(64, *LOCAL_PARAMS)
        self.signer = A.key_signer(self.host, self.params, workers=1)
        self.thread: threading.Thread | None = None
        self.session: str | None = None

    def _serve(self, session: str) -> None:
        A, host = self.A, self.host
        body = A.sessions_of(A.read_ledger(host.ledger))[session]["open"]["body"]
        argv = A.serve_argv(host, session, body)
        argv[argv.index("--port") + 1] = "0"
        argv += ["--params", ",".join(map(str, LOCAL_PARAMS)), "--keygen-workers", "1"]
        A.run(host, session, signer=self.signer, pinned=self.pin, now=lambda: datetime.now(timezone.utc), argv=argv)

    def open(self, root: str, layout: list, minutes: int = MINUTES, scheme: str = SCHEME) -> tuple[str, str, str]:
        """Reserve and open the session on `root` with the coins `layout` of `scheme`, as the gate does: (address, secret,
        commitment)."""
        A, host, now = self.A, self.host, datetime.now(timezone.utc)
        hooks = dict(by=self.by, now=now, own=lambda p: None, token=secrets.token_hex(2),
                     reserve=lambda s: A.reserve(host, s, signer=self.signer, pinned=self.pin, now=now))
        rc, out = A.gate(["reserve", "--peer", LOCAL_PEER, "--scheme", scheme, "--families", json.dumps(layout),
                          "--minutes", str(minutes)], host, **hooks)
        if rc:
            raise Ch.CoinError(f"the local gate refused the reserve: {out.decode().strip()}")
        self.session = json.loads(out)["session"]

        def start(s: str) -> None:
            self.thread = threading.Thread(target=self._serve, args=(s,), daemon=True)
            self.thread.start()

        def alive(_: str) -> bool:
            return self.thread is not None and self.thread.is_alive()
        rc, out = A.gate(["open", "--session", self.session, "--root", root], host, **hooks, secret=secrets.token_bytes(32),
                         admit=lambda s: A.admit(host, s, signer=self.signer, pinned=self.pin, now=now, active=alive),
                         start=start, active=alive, ready=lambda s: A.await_commitment(host, s, alive))
        got = json.loads(out)
        if rc or got.get("commitment") is None:
            raise Ch.CoinError(f"the local gate refused the open, or its challenger signed no commitment: {out.decode().strip()}")
        address = host.sessions / self.session / "out" / "address.json"
        # serve.py writes the file in place, its newline last: read it only once it is whole
        while not (address.exists() and address.read_text().endswith("\n")) and alive(self.session):
            time.sleep(0.1)
        return json.loads(address.read_text())["address"], got["secret"], got["commitment"]

    def finish(self, dest: Path) -> tuple[Path, Path, Path, Ch.ChallengerKey]:
        """Once the session has ended (its end entry in the ledger): its out/, the ledger and the pins log copied to `dest`, and
        its key."""
        assert self.thread is not None and self.session is not None
        self.thread.join()
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.host.sessions / self.session / "out", dest / "session", dirs_exist_ok=True)
        shutil.copyfile(self.host.ledger, dest / "windows.jsonl")
        shutil.copyfile(self.host.pins, dest / "pins.jsonl")
        key = Ch.ChallengerKey(self.params.W, self.params.H, (self.host.key / "public").read_bytes())
        return dest / "session", dest / "windows.jsonl", dest / "pins.jsonl", key


class Gate:
    """The challenger host's session through infra's gate `gate` (USER@HOST) with the lane's gate key `ssh_key`, its challenger
    listening on `peer`'s 127.0.0.1 (this machine's name in the host's /etc/vy/peers), and the challenger key the deployment
    records at `key_record`; `main` is the verifier's copy of main's first-parent line."""

    def __init__(self, gate: str, ssh_key: str | None, peer: str, key_record: str, main: list[str]) -> None:
        self.H1 = served_h1()
        self.gate, self.peer, self.ssh = gate, peer, self.H1.ssh_args(ssh_key)
        self.key = self.H1.key_record(key_record)
        self.main = main
        self.session: str | None = None

    def _raw(self, words: str) -> bytes:
        p = subprocess.run([*self.ssh, self.gate, words], capture_output=True)
        if p.returncode:
            raise Ch.CoinError(f"infra's gate refused {words.split()[0]}: {p.stderr.decode().strip() or p.stdout.decode().strip()}")
        return p.stdout

    def open(self, root: str, layout: list, minutes: int = MINUTES, scheme: str = SCHEME) -> tuple[str, str, str]:
        """Reserve and open the root's one session, of `scheme`, once the host's pins log pins, in its last line, a commit on
        `main`: a session from any other tree gives coins `pinned_session` refuses, and the gate opens a root once, ever."""
        pins = [x for x in self._raw("pins").decode().split("\n") if x]
        pin = json.loads(pins[-1]).get("commit") if pins else None
        if pin not in self.main:
            raise Ch.CoinError(f"the challenger host's pins log pins {pin} last, which is not on main's first-parent line in the "
                               f"verifier's copy ({len(self.main)} commits from {self.main[0][:10] if self.main else None}): "
                               "repin the host to a commit on it (vy-challenger-pin), or refresh --main, before spending the "
                               "root's one session")
        got = self.H1.gate_call(self.gate, ["reserve", "--peer", self.peer, "--scheme", scheme, "--families",
                                            json.dumps(layout, separators=(",", ":")), "--minutes", str(minutes)], self.ssh)
        self.session = got["session"]
        session, _ = self.H1.open_window(self.gate, self.session, bytes.fromhex(root), self.ssh)
        return session.address, session.secret.hex(), session.commitment

    def finish(self, dest: Path, minutes: int = MINUTES) -> tuple[Path, Path, Path, Ch.ChallengerKey]:
        """Once the gate's ledger has the session's end: its fetched directory, the ledger and the host's pins log in `dest`, and
        the key."""
        assert self.session is not None
        deadline = time.monotonic() + 60 * (minutes + 5)
        while self.H1.gate_call(self.gate, ["status", self.session], self.ssh).get("end") is None:
            if time.monotonic() > deadline:
                raise Ch.CoinError(f"session {self.session} has no end entry in the gate's ledger after {minutes + 5} minutes")
            time.sleep(10)
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(self._raw(f"fetch {self.session}"))) as tar:
            tar.extractall(dest, filter="data")
        (dest / "windows.jsonl").write_bytes(self._raw("ledger"))
        (dest / "pins.jsonl").write_bytes(self._raw("pins"))
        return self.H1.published(dest / self.session), dest / "windows.jsonl", dest / "pins.jsonl", self.key


class Loopback:
    """The challenger party's host on this machine (`challenger_party.Host`, `verity run CONFIG --party challenger`'s)
    at `address` (HOST:PORT): `Local`'s session, asked for over one JSON line each way. Its key comes
    from the party itself, so its coins are a development session's, not an auditor's (`simulated`)."""

    def __init__(self, address: str) -> None:
        host, port = address.rsplit(":", 1)
        self.address, self.session = (host, int(port)), None

    def _ask(self, req: dict) -> dict:
        with socket.create_connection(self.address) as s, s.makefile("rwb") as f:
            f.write((json.dumps(req) + "\n").encode())
            f.flush()
            got = json.loads(f.readline())
        if "refused" in got:
            raise Ch.CoinError(f"the challenger party refused {req['op']}: {got['refused']}")
        return got

    def open(self, root: str, layout: list, minutes: int = MINUTES, scheme: str = SCHEME) -> tuple[str, str, str]:
        got = self._ask({"op": "open", "root": root, "layout": layout, "minutes": minutes, "scheme": scheme})
        self.session = got["session"]
        return got["address"], got["secret"], got["commitment"]

    def finish(self, dest: Path) -> tuple[Path, Path, Path, Ch.ChallengerKey]:
        got = self._ask({"op": "finish", "dest": str(dest)})
        return Path(got["session"]), Path(got["ledger"]), Path(got["pins"]), Ch.ChallengerKey.from_json(got["key"])


def lms_verify(key: Ch.ChallengerKey, message: bytes, signature: bytes) -> bool:
    p = lms.Params(64, key.W, key.H)
    try:
        return lms.verify(p, key.public, message, lms.Signature.decode(p, signature))
    except ValueError:
        return False


def opened_pin(ledger: list[str], session: str) -> str | None:
    """The commit the open of `session` in the auditor's ledger `ledger` names as its host's tree (its `pinned`)."""
    for line in ledger:
        e = json.loads(line)
        if e.get("kind") == "open" and e.get("session") == session:
            return e["body"].get("pinned")
    return None


def start(challenger: str, root: str, ranks: list[int], dest: Path, main: str | Path, *, ssh_key: str | None = None,
          peer: str | None = None, key_record: str | None = None,
          reserve: Callable[[Mapping[int, int]], list] = families):
    """The step's session on `root` (``local`` or ``gate:USER@HOST``, module docstring) and its draws' manager: (the manager, a
    `Draws` proxy, and the thread that opens the session and publishes its evidence). The thread waits until every rank of
    `ranks` has planned its draw (`Draws.plan`), then opens the session with the layout `reserve` gives their needs
    (`families`). It finishes the session into `dest` once the client has closed it. It opens nothing, and refuses every
    rank's draw, when a rank ended before planning (refused before its draw, or failed) or `reserve_reason` refuses the
    layout. The evidence's `plan` is the needs, the layout and the session's id (None when none opened). `main` is the
    verifier's own copy of main's first-parent history (`remote.main_lines`'s file: ``git rev-list --first-parent
    origin/main``)."""
    first_parent = R.main_lines(main)
    if challenger == "local":
        session = Local(Path(tempfile.mkdtemp(prefix="a5-challenger-")), first_parent)
    elif challenger.startswith("loopback:"):
        session = Loopback(challenger.removeprefix("loopback:"))
    elif challenger.startswith("gate:") and peer and key_record:
        session = Gate(challenger.removeprefix("gate:"), ssh_key, peer, key_record, first_parent)
    else:
        raise SystemExit(f"--challenger is local, loopback:HOST:PORT or gate:USER@HOST (with --peer and --challenger-key), "
                         f"not {challenger!r}")
    manager = _Manager(ctx=get_context("spawn"))
    manager.start()
    draws = manager.Draws(root, ranks)

    def finish() -> None:
        needs = draws.wait_planned()
        plan: dict[str, Any] = {"needs": {str(r): n for r, n in sorted(needs.items())}, "families": None, "session": None}
        try:
            missing = sorted(set(ranks) - set(needs))
            if missing:
                raise Ch.CoinError(f"rank(s) {missing} ended before planning their draw (refused before it, or failed)")
            layout = plan["families"] = reserve(needs)
            why = reserve_reason(layout, needs)
            if why is not None:
                raise Ch.CoinError(why)
        except Exception as e:  # every waiting rank's plan must end, whatever refused
            draws.refuse(f"{e}: the root's one session was not opened")
            draws.publish({"error": f"the challenger's session was not opened: {e}", "plan": plan})
            return
        try:
            address, secret, commitment = session.open(root, layout)
        except Exception as e:
            draws.refuse(f"opening the challenger's session failed: {e}")
            draws.publish({"error": f"opening the challenger's session failed: {e}", "plan": plan})
            return
        plan["session"] = session.session
        try:
            draws.connect(address, secret, commitment)
        except Exception as e:
            draws.publish({"error": f"the challenger's session {session.session} gave no client: {e}", "plan": plan})
            return
        ended = draws.wait_closed()
        try:
            directory, ledger, pins_log, key = session.finish(dest)
            commitment_, roots = R.trusted_session(directory, [ledger], {"root": root}, key, lms_verify)
            led, pins = R.ledger_lines(ledger), lines(pins_log)
            try:
                Ch.pinned_session(led, {"root": root}, key, pins, first_parent, lms_verify)
                pinned = None
            except Ch.CoinError as e:
                pinned = str(e)
            draws.publish({"key": key.to_json(), "ledger": led, "main": first_parent, "pins": pins,
                           "session": R.session_files(directory),
                           "trusted": {"commitment": commitment_.root.hex(), "released": {k: len(v) for k, v in roots.items()},
                                       "close": ended, "ledger_sha512": hashlib.sha512(ledger.read_bytes()).hexdigest(),
                                       "pin": opened_pin(led, session.session), "pins_lines": len(pins),
                                       "main": {"tip": first_parent[0] if first_parent else None, "commits": len(first_parent)},
                                       "pinned_session": pinned}, "plan": plan})
        except (Ch.CoinError, OSError, ValueError, KeyError) as e:
            draws.publish({"error": f"the challenger's session {session.session} gave no evidence: {e}", "plan": plan})
    finisher = threading.Thread(target=finish, daemon=True)
    finisher.start()
    return manager, draws, finisher


def evidence(published: dict, segment: dict) -> dict | None:
    """One draw's coins' evidence, `coins.coins_reason`'s {key, ledger, main, pins, segment, session}; None when the session
    gave none."""
    if "error" in published:
        return None
    return {"key": published["key"], "ledger": published["ledger"], "main": published["main"], "pins": published["pins"],
            "segment": segment, "session": published["session"]}


def negatives(ev: dict, pre: dict) -> dict[str, dict | None]:
    """`ev` altered as a prover would alter it: a tampered opening, the coin opened before the registration (`pre`'s, its
    `after` rewritten to the registration's) added in front, a segment that skips a coin released after the registration,
    the verifier's main without any commit the pins log pins (the session's pin off main's first-parent line), no pins log,
    no evidence at all."""
    s = ev["segment"]
    earlier = [{**o, "after": s["after"]} for o in pre["openings"]]
    logged = {json.loads(line)["commit"] for line in ev["pins"]}
    return {
        "coins-tampered-opening": {**ev, "segment": {**s, "openings": [{**s["openings"][0], "coin": "00" * 64},
                                                                       *s["openings"][1:]]}},
        "coins-opened-before-the-registration": {**ev, "segment": {**s, "openings": earlier + s["openings"],
                                                                   "start": s["start"] - len(earlier)}},
        "coins-segment-skips-a-released-coin": {**ev, "segment": {**s, "openings": s["openings"][1:], "start": s["start"] + 1}},
        "coins-pin-off-main": {**ev, "main": [c for c in ev["main"] if c not in logged]},
        "coins-no-pins-log": {**ev, "pins": None},
        "coins-no-evidence": None,
    }
