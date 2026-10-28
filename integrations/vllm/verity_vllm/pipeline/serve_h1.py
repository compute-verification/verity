"""serve-openai's PoUW H1 arm (``--protocols pouw:eps1-sm120-h1fx1``): one H1 window a Commit window, tied to the server's
own Commit.

* **One Commit.** H1 builds no committer: its tap goes on the server's rows-sha512 collector (`acquire.h1_commit.tap`),
  which already acquires each linear's x_q and s_a (the bench row's int8 source) and each step's input ids (its step-input
  port), so ROOTS1 holds H1's A tables and input ids to the values the served window commits.
* **One window a Commit window.** `serve_commit.CommitWindows` opens the arm once the committer is installed, so H1's
  hooks sit on top of the committer's, and closes it first thing at the window's close, before the committer comes off.
  The H1 window's T is the engine's token budget (``--max-num-batched-tokens``) and its S the Commit window's steps
  (``--window-steps``); a window that closes early (idle, ``--window-gb``, ``--window-requests``) closes H1's with its
  remaining steps empty (`pouw_h1.H1Run.close`).
* **Once a server.** The period's declared Program over the served linears at (T, S) and its digest
  (`pouw_h1_served.declared`), and the weights' ``b/<K>`` registered under it (`pouw_h1_served.register_weights`), with a
  key of this process's: no auditor's registration (``model-key``) is read yet. The Program holds one batch of K's tile
  template a call (S × the linears), its tile units (`units`, S × slot(T)/32 × Σn/32) the batch's members, so its
  descriptor, declaring it and its digest grow with the calls, not the units. The server logs both counts and both
  times. The digest is hashed as the descriptor is written, ``h1/`` `PROGRAM` beside `PROGRAM_META` (`served.write_program`),
  and ``served_verdicts.py program`` writes the same file without serving, so the period-open of it (``served_verdicts.py
  period open``) goes out before the server starts, ahead of window 0's reserve at startup.
* **Coins.** With ``--challenger-gate CMD`` and ``--challenger-peer NAME`` (`sessions_of`), the auditor host's
  challenger's, one session a window through infra's gate (`GateSessions`, `commit_coins.ChallengerGate`): window 0's session is
  reserved (scheme `pouw_h1_challenger.SCHEME`, `pouw_h1_challenger.families` (S)) at startup (`arm_of`) and window
  k+1's while window k serves, off the engine loop (`GateSessions.take`); round 0 opens it on the window's open root
  (`pouw_h1_window.Window`), reached on this machine's 127.0.0.1 at the port the gate's open answers once the auditor
  host's ``ssh -R`` listens there (looked for every `LISTEN_S`), and the window's release waits for its end entry. The
  gate's open answers a port no session without an end entry holds, so a window whose session opens while the window
  before is still releasing takes a port of its own. A reserve joins its lane's open ``period``, if one is open, so one
  is opened before the server starts for window 0's to join; and a reserve made ahead that no window opens (the server
  stopped first, window 0's included) would leave its ``period`` incomplete, so the next window's is reserved ahead
  only while the last reserve joined none, and one no window opened is ``unused`` in the summary. Without a gate, the
  epoch's in-process coins, one epoch a window (`pouw_h1_settings.from_engine`): such a window credits nothing
  (`pouw_h1.IN_PROCESS`). ``--protocols-config`` sets no session (``SESSIONED``).
* **Ready before it listens.** `H1Windows.ready`, before serve-openai listens, does what the first block's first step
  would otherwise do before its first token: the weights' roots (`pouw_h1_hooks.weight_roots`, which each install
  reuses), the call tables' GPU hasher built (`pouw_h1_tables_device.available`), every z/y kernel a step of up to T
  rows runs compiled (`pouw_h1_hooks.warm_zy`), the Lean verifier of record built (`verifier.executable`, once a
  process: in a fresh tree a minute's build, which in-process coins otherwise run at the first one's draw), and the wait
  for that block's reserve (window 0's, which `arm_of` makes while the Program is declared). Round 0's open stays at the block's first admission: its open root binds that round's
  requests, and the call tables are bound under the salt the open draws.
* **Off the engine loop.** At a window's close the loop takes only H1's hooks and tap off; its release (the empty
  steps' calls, the manifest's commit, the draw coin, the session's close and the wait for its end entry) runs on
  ``closer``, one thread in window order, while the next window serves. Each window still has its own session,
  coins and transcript, and its outcome is recorded in window order once its release ends.
* **Its draw, for the draw-gated release** (``--retain device --seal --release-after pouw``, `serve_release`). Once a
  window's draw coin is open, its release puts H1's draw of the window under `DRAWER` (`H1Windows.drawn`): the proof
  blocks core draws (`credit.drawn` over the window Program's layout, as `pouw_h1_challenger.drawn_of` draws them), each
  read as the Commit's values it opens (`drawn_reads`): its call's x_q and s_a rows of its row tile at the engine step its
  step maps to. With a gate session its stream is the window's open root and its K its period-open's ``draw_k``; with
  in-process coins (no open root, no period, nothing credited) its calls_root and the proof service's K
  (`pouw_h1_challenger.draw_k`). A window whose draw can't be formed (it never opened, its close failed, no draw coin,
  no period) refuses its release by name, so its seal fails visibly and never waits.

Each window's outcome (`H1Windows.close`) is its ``windows.jsonl`` line's ``pouw_h1``: ROOTS1's verdict, the requests'
admission and gaps, the credit gate, the coins, the commitment's calls root, its gate session (the reserve, the open's
port, how the session closed and its end entry) and where its time went (``seconds``); a window whose H1 failed to open
or close carries its error there instead, and the Commit serves on. ``serve.json``'s ``protocols`` is
`H1Windows.describe`, ``summary.json``'s ``pouw_h1`` `H1Windows.summary` with the windows that failed.
``--protocols-config`` is the pouw settings the server doesn't set itself, ``{"pouw": {...}}`` (`config_of`). Its ``unit``
and ``ks`` shape the declared Program: ``{"pouw": {"unit": "j10r", "ks": [2048]}}`` declares J10-R's tile unit
(`pouw_h1_served.unit_of`) over the linears at K 2,048 alone, the form a period's guarantee covers (core's
`period_check.form`), and the linears at any other K serve on their arm unrecorded, uncredited as attention is. Unset,
the Program is v1's over every linear."""
from __future__ import annotations

import bisect
import dataclasses
import functools
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

from verity_vllm.config import option
from verity_vllm.protocols.pouw.h1 import PROGRAM, PROGRAM_META, SCHEME

ENTRY = f"pouw:{SCHEME}"
#: the pouw settings the server sets for each window itself
OWN = ("rows", "steps", "runner", "served", "transcript_dir", "coins", "rounds")
#: the pouw settings of a gate session a window, which serve-openai sets itself from ``--challenger-gate`` (`GateSessions`)
SESSIONED = ("challenger", "session", "gate", "period")
#: the challenger as the window reaches it: this machine's 127.0.0.1, its port the one the gate's open answers
#: (`pouw_h1_window.at_port`; the gate's 7811..7899)
AT = "127.0.0.1:7811"
#: how many times, a second apart, a session's end entry is polled for, and for how many seconds its ``ssh -R`` port is
#: looked for here
POLLS = 300
#: how often a session's ``ssh -R`` port is looked for here once its open answers
LISTEN_S = 0.1
#: a session reserved ahead is opened only while its reserve has this long left before it expires (the gate refuses an
#: expired reserve's open); else its window reserves another
MARGIN = timedelta(minutes=5)
#: the subprotocol H1's draw of each window is put under in serve-openai's draw-gated release (``--release-after``)
DRAWER = "pouw"
#: an s_a's bytes a row (f32)
S_A_BYTES = 4


def _cuda_device() -> int | None:
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    return int(torch.cuda.current_device())


def _on_device(device: int | None) -> None:
    if device is not None:
        sys.modules["torch"].cuda.set_device(device)


def worker(name: str) -> ThreadPoolExecutor:
    """One thread, named ``name``, that runs what it is handed (``submit``) in turn, on the CUDA device current here."""
    return ThreadPoolExecutor(1, thread_name_prefix=name, initializer=_on_device, initargs=(_cuda_device(),))


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _expires(reserved: Mapping[str, Any]) -> datetime | None:
    at = reserved.get("expires_utc")
    return datetime.strptime(at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC) if isinstance(at, str) else None


@dataclasses.dataclass(frozen=True, kw_only=True)
class GateOptions:
    """serve-openai's ``--challenger-gate`` and ``--challenger-peer`` (its ``gate`` group, `sessions_of`)."""
    challenger_gate: str | None = option("--challenger-gate", default=None, metavar="CMD",
                                         help="with --protocols: the auditor host's challenger gate, its command as one string "
                                              "(e.g. 'ssh -i KEY -o BatchMode=yes research@HOST'); each H1 window reserves a "
                                              "session of its own there and draws its coins from it (serve_h1.GateSessions); "
                                              "without it they are in-process and credit nothing")
    challenger_peer: str | None = option("--challenger-peer", default=None, metavar="NAME",
                                         help="with --challenger-gate: this machine's name in the auditor host's peers, where its "
                                              "challenger reaches the server (ssh -R)")
    h1_prover: str | None = option("--h1-prover", default=None, metavar="DIR",
                                   help="with --protocols: keep each H1 window's prover state (its salt key and each call's "
                                        "rows, pouw_h1_prover) and its live@<s> rows (pouw_h1_live.dump) in DIR/w<k>/, "
                                        "owner-only, for proving its drawn units and rounds after the close; never a "
                                        "published path")
    h1_model: str | None = option("--h1-model", default=None, metavar="ROLE",
                                  help="with --challenger-gate: the auditor's registration role of the served model (a role "
                                       "of the pinned tree's manifests/checkpoints-int8.json, the gate's models.MANIFESTS); "
                                       "its b/<K> are committed under the salt key the gate's "
                                       "model-key hands for the window Program's circuit, so the windows' weights root is the "
                                       "registry's, the root the period signs; without it the key is local and credits nothing")
    h1_weights_root: str | None = option("--h1-weights-root", default=None, metavar="HEX",
                                         help="with --h1-model: the registration's weights root (the period's signed one), "
                                              "required when the role is registered more than once under the circuit")


def model_key_of(gate: Any, model: str, weights_root: str | None) -> Callable[[str], tuple[bytes, str]]:
    """The salt key of ``model``'s auditor registration under a window Program's circuit, from the gate's ``model-key``
    (`auditor.gate_model_key`), with the registry's weights root it opens: a callable of the circuit's digest."""
    def key(circuit: str) -> tuple[bytes, str]:
        words = ["model-key", "--model", model, "--circuit", circuit]
        if weights_root:
            words += ["--weights-root", weights_root]
        answer = gate.call(*words)
        return bytes.fromhex(answer["key"]), str(answer["weights_root"])
    return key


def config_of(value: str | None) -> dict[str, Any]:
    """``--protocols-config``: a JSON object, or a JSON file, ``{"pouw": {...}}``; empty for None. `UsageError` on any key
    the server sets itself (`OWN`), one a gate session sets (`SESSIONED`, from ``--challenger-gate``), or a value
    `pouw.Settings` refuses."""
    from verity_vllm.config import UsageError
    from verity_vllm.protocols import load_settings
    from verity_vllm.protocols import pouw as PW

    if not value:
        return {}
    try:
        doc = load_settings(json.loads(value) if value.lstrip().startswith("{") else value)
    except (OSError, ValueError) as e:
        raise UsageError(f"--protocols-config {value[:80]}: {e}") from None
    if stray := sorted(set(doc) - {"pouw"}):
        raise UsageError(f"--protocols-config: {stray} are not the enabled protocols' (pouw)")
    pouw = doc.get("pouw", {})
    if own := sorted(set(pouw) & set(OWN)):
        raise UsageError(f"--protocols-config: the pouw settings {own} are the server's own (T, S, its runner, the window's "
                         "served tables and coins, one epoch a window)")
    if sessioned := sorted(set(pouw) & set(SESSIONED)):
        raise UsageError(f"--protocols-config: the pouw settings {sessioned} need one gate session a window, which serve-openai "
                         "reserves itself: give --challenger-gate and --challenger-peer")
    try:
        PW.Settings.of(pouw)
    except ValueError as e:
        raise UsageError(f"--protocols-config: {e}") from None
    return doc


def refusal(opts: Any) -> str | None:
    """Why ``opts`` can't serve ``--protocols`` (None when they can, or none is asked for); `config_of`'s `UsageError` when
    it refuses ``--protocols-config``."""
    after = getattr(getattr(opts, "retention", None), "release_after", "") or ""
    if DRAWER in (p.strip() for p in after.split(",")) and opts.protocols != ENTRY:
        return (f"--release-after {DRAWER} waits for PoUW H1's draw of each window, which only its arm puts: it needs "
                f"--protocols {ENTRY}")
    gate, peer = opts.gate.challenger_gate, opts.gate.challenger_peer
    if bool(gate) != bool(peer):
        return ("--challenger-gate and --challenger-peer go together: the gate's command and this machine's name in the "
                "auditor host's peers, where its challenger reaches the server (ssh -R)")
    if gate and not opts.protocols:
        return f"--challenger-gate draws PoUW H1's coins: it needs --protocols {ENTRY}"
    if opts.gate.h1_prover and not opts.protocols:
        return f"--h1-prover keeps PoUW H1's prover state: it needs --protocols {ENTRY}"
    if opts.protocols_config and not opts.protocols:
        return "--protocols-config without --protocols"
    if not opts.protocols:
        return None
    if opts.protocols != ENTRY:
        return f"--protocols {opts.protocols}: serve-openai serves {ENTRY} (PoUW H1) only"
    if not opts.commit or opts.scheme != "rows-sha512" or not opts.required_manifest:
        return (f"--protocols {ENTRY} taps the server's rows-sha512 Commit (its int8 source and step-input port): it needs "
                "--commit, --scheme rows-sha512 and the bench row's --required-manifest")
    if opts.max_num_batched_tokens % 32:
        return f"--protocols {ENTRY}: H1's T, --max-num-batched-tokens {opts.max_num_batched_tokens}, is a multiple of 32"
    config_of(opts.protocols_config)
    return None


def units(shapes: Any, rows: int, steps: int) -> int:
    """The tile units of the window Program `served.declared` builds over ``shapes`` (each ``(n, K)``) at T ``rows`` and S
    ``steps``: the members of its batches, one a call (S × the linears), so the descriptor, the declare and the digest
    grow with the calls, not the units."""
    from verity.core.protocols.pouw.circuit import h1window as HW

    from verity_vllm.protocols.pouw import h1 as PH

    r, c = PH.TILE
    return int(steps) * (HW.slot(int(rows), PH.TILE) // r) * sum(int(n) // c for n, _k in shapes)


def drawn_units(commits: Mapping[str, Any], layout: Mapping[str, Any], k: int, stream: str) -> list[int]:
    """The proof blocks a closed window's draw coin draws at K ``k`` (`credit.drawn` over the window Program's
    ``layout``, `credit.layout_of`'s, on stream ``stream``, registration the draw root and receipt its receipts, as
    `pouw_h1_challenger.drawn_of` draws them), from its commitment ``commits``. ValueError when it opened no draw coin or
    core refuses the draw."""
    from verity.core.protocols.pouw import credit as CR
    from verity.core.protocols.sampled_proofs.challenger import LiveOpening

    draw = commits.get("draw") or {}
    if draw.get("coin") is None:
        raise ValueError("the window opened no draw coin" + (f" ({draw['refused']})" if draw.get("refused") else ""))
    got, why = CR.drawn(LiveOpening.from_json(draw["coin"]).coin.hex(), stream, draw.get("root"),
                        {"receipts": draw.get("receipts")}, dict(layout), k)
    if why is not None:
        raise ValueError(f"{why[0]}: {why[1]}")
    return list(got)


def drawn_reads(tap: Any, layout: Mapping[str, Any], drawn: Sequence[int]) -> list[tuple[int, str, int, int]]:
    """What the proof blocks ``drawn`` of the window ``layout`` open of the server's Commit, as the draw-gated release takes
    them (`serve_release.Read`: engine step, member, byte lo, hi): unit u, tile t of call c, opens A rows R·I .. R·I + R - 1
    of its call, I = t // (n/C) its row block (`h1window.table_reads`), which are the Commit's x_q (K bytes a row) and s_a
    rows of the call's site at the engine step its step maps to, as H1's tap ``tap`` registered them (``where``), under
    the member names the Commit acquired them by (``acquired``), cut to the rows it acquired; a call of an empty step
    opens none. KeyError on a drawn call the Commit registered no rows of."""
    (r, cw), m = layout["tile"], int(layout["m"])
    ks = {name: (int(k), int(n)) for name, k, n in layout["linears"]}
    firsts, at = [0], 0
    for lin, _blk in layout["calls"]:
        at += (m // r) * (ks[lin][1] // cw)
        firsts.append(at)
    calls: dict[int, tuple[int | None, str, int]] = {}
    for w in tap.where.values():
        if "calls" in w:                                   # an A table's rows; a round's ids are no call's
            for c, j, s, n in zip(w["calls"], w["steps"], w["sites"], w["rows"]):
                calls[int(c)] = (j, s, int(n))
    out: set[tuple[int, str, int, int]] = set()
    for u in drawn:
        c = bisect.bisect_right(firsts, u) - 1
        if not 0 <= c < len(layout["calls"]):
            raise KeyError(f"unit {u} is no unit of the window's {firsts[-1]}")
        k, n = ks[layout["calls"][c][0]]
        i = (u - firsts[c]) // (n // cw)
        if c not in calls:
            raise KeyError(f"unit {u}'s call {c}: the Commit registered no rows of it, so what it opens can't be kept")
        j, site, rows = calls[c]
        lo, hi = r * i, min(r * (i + 1), rows)
        if j is None or hi <= lo:
            continue
        x_q, s_a = (tap.acquired[(int(j), site, port)] for port in ("0", "1"))
        out |= {(int(j), x_q, lo * k, hi * k), (int(j), s_a, lo * S_A_BYTES, hi * S_A_BYTES)}
    return sorted(out)


def listening(port: int) -> bool:
    """Whether a socket on this machine listens on TCP `port`, as the kernel's tables say (``/proc/net/tcp``, ``tcp6``;
    state 0A): a probe connect would take the place of the session's one client."""
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            rows = [r.split() for r in Path(table).read_text().splitlines()[1:]]
        except OSError:
            continue
        if any(int(r[1].rsplit(":", 1)[1], 16) == port and r[3] == "0A" for r in rows):
            return True
    return False


class GateSessions:
    """H1's coins from the auditor host's challenger, one session a window, through infra's gate `gate` (a
    `commit_coins.ChallengerGate`) as peer `peer`, for windows of S `steps` rounds. `take` is a window's session, `open` (the
    pouw setting ``gate``) opens it on the window's open root and `ended` waits for its end entry. ``held`` is each
    session reserved, as the gate answered it, under the window it was reserved for; one no window opened says why
    (``unused``). With ``ahead`` (a `worker`), a window's session is reserved there before it opens (`ahead_of`, `take`).
    `wait` sleeps between polls, `listens` is `listening` and `now` the UTC time a reserve's expiry is read against
    (tests)."""

    def __init__(self, gate: Any, peer: str, steps: int, *, wait: Callable[[float], Any] = time.sleep,
                 listens: Callable[[int], bool] = listening, ahead: Any = None,
                 now: Callable[[], datetime] = _utc_now) -> None:
        from verity_vllm.protocols.pouw.h1 import window as PHW

        self.gate, self.peer, self.steps, self.wait, self.listens = gate, peer, int(steps), wait, listens
        self.ahead, self.now = ahead, now
        self._open = PHW.gate_of(gate.call)
        self.held: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._ahead: dict[int, Future] = {}

    def reserve(self, window: int) -> dict[str, Any]:
        """Window `window`'s session reserved (``reserve --peer PEER --scheme SCHEME --families families(S)``): the gate's
        reserve answer as the pouw setting ``session`` takes it (`pouw_h1_window.reserved_of`). `CoinError` when the gate
        refuses."""
        from verity_vllm.protocols.pouw.h1 import challenger as PHC
        from verity_vllm.protocols.pouw.h1 import window as PHW

        answer = self.gate.call("reserve", "--peer", self.peer, "--scheme", PHC.SCHEME, "--families",
                                PHC.families(self.steps))
        reserved = PHW.reserved_of({k: v for k, v in answer.items() if k in ("session", *PHW.RESERVE_FIELDS)})
        with self._lock:
            self.held.append({"window": window, **reserved})
        print(f"[serve-h1] window {window}: reserved session {reserved['session']} on {self.peer} "
              f"(period {reserved.get('period')})", flush=True)
        return reserved

    def ahead_of(self, window: int) -> None:
        """Window `window`'s session reserved on ``ahead``, so its open finds it (`take`); nothing without ``ahead``."""
        if self.ahead is not None and window not in self._ahead:
            self._ahead[window] = self.ahead.submit(self.reserve, window)

    def settled(self, window: int) -> dict[str, Any] | None:
        """Window `window`'s reserve made ahead, once it ends: the session it reserved, or None when none was made or it
        failed (`take` reserves at the window's open then)."""
        fut = self._ahead.get(window)
        if fut is None or fut.exception() is not None:
            return None
        return fut.result()

    def take(self, window: int) -> dict[str, Any]:
        """Window `window`'s session: the one reserved ahead for it, waited for if its reserve is still in flight, unless
        that reserve failed or expires within `MARGIN`, else one reserved now (`reserve`; `CoinError` when the gate
        refuses). Then, while that reserve joined no period, the next window's is reserved ahead (`ahead_of`)."""
        got, fut = None, self._ahead.pop(window, None)
        if fut is not None and (e := fut.exception()) is not None:
            print(f"[serve-h1] window {window}: its reserve ahead failed ({type(e).__name__}: {e}); reserving now", flush=True)
        elif fut is not None:
            got = fut.result()
            expires = _expires(got)
            if expires is not None and expires - self.now() < MARGIN:
                self._unused(got["session"], f"its reserve expires at {got['expires_utc']}, before its window could open it")
                got = None
        if got is None:
            got = self.reserve(window)
        if got.get("period") is None:
            self.ahead_of(window + 1)
        return got

    def held_of(self, session: str) -> dict[str, Any]:
        with self._lock:
            return next((h for h in self.held if h["session"] == session), {})

    def _unused(self, session: str, why: str) -> None:
        self.held_of(session)["unused"] = why

    def drain(self) -> None:
        """Every reserve made ahead ended, the ones no window took marked ``unused``."""
        if self.ahead is not None:
            self.ahead.shutdown(wait=True)
        for window, fut in sorted(self._ahead.items()):
            if fut.exception() is None:                  # a reserve ahead that failed holds no session
                self._unused(fut.result()["session"], f"reserved ahead for window {window}, which never opened")
        self._ahead.clear()

    def open(self, session: str, root: str) -> dict[str, Any]:
        """`pouw_h1_window.Gate`: ``open --session SESSION --root ROOT`` (its commitment from ``status`` when the answer's
        is null), answered once the answer's port listens here (the auditor host's ``ssh -R``, looked for every
        `LISTEN_S` for `POLLS` seconds; ``listen_polls`` the looks it took)."""
        answer = self._open(session, root)
        port, polls, looks = answer.get("port"), None, round(POLLS / LISTEN_S)
        if type(port) is int:
            for i in range(looks):
                if self.listens(port):
                    polls = i
                    break
                self.wait(LISTEN_S)
        self.held_of(session).update(port=port, listen_polls=polls)
        print(f"[serve-h1] session {session} opened on {root[:16]}…: port {port} "
              + (f"listening after {polls} polls of {LISTEN_S} s" if polls is not None
                 else f"not listening after {POLLS} s"), flush=True)
        return answer

    def ended(self, session: str) -> dict[str, Any]:
        """Session `session` once the gate's ``status`` shows its end entry (polled `POLLS` times), or that it was never
        opened: ``opened``, and its end's ``rc``, ``closed`` and ``complete`` (None while it has none); ``error`` when
        the gate refuses the status."""
        s: Mapping[str, Any] = {}
        try:
            for _ in range(POLLS):
                s = self.gate.call("status", session)
                if s.get("end") is not None or s.get("open") is None:
                    break
                self.wait(1.0)
        except (ValueError, OSError) as e:
            return {"opened": None, "error": f"{type(e).__name__}: {e}"[:300]}
        end = s.get("end")
        body = end.get("body") if isinstance(end, Mapping) else None
        return {"opened": s.get("open") is not None,
                **{k: body.get(k) if isinstance(body, Mapping) else None for k in ("rc", "closed", "complete")}}


def sessions_of(opts: Any) -> GateSessions | None:
    """The server's `GateSessions` from ``--challenger-gate`` and ``--challenger-peer`` (`refusal` holds them together),
    for windows of ``--window-steps``, each reserved ahead on a `worker` of its own; None without a gate."""
    if not opts.gate.challenger_gate:
        return None
    from verity_vllm.pipeline import commit_coins as CC

    return GateSessions(CC.ChallengerGate(opts.gate.challenger_gate), opts.gate.challenger_peer, opts.window_steps,
                        ahead=worker("h1-reserve"))


class H1Windows:
    """The arm over ``model`` (the served `verity_vllm.LLM`'s) and the server's committer ``com``: T ``rows`` and S
    ``steps`` a window, ``config`` the pouw settings of ``--protocols-config``, ``engine`` the built ``vllm.LLM`` the
    runner and each window's coins come from, ``out`` where each window's transcript goes (``h1/w<k>/``). ``key`` is the
    weights' registration key (a fresh one when None); ``unit`` the Program's tile unit at each K (`h1window.tile_unit`
    when None); ``sessions`` each window's session on the auditor host's challenger (`GateSessions`; None: in-process
    coins); ``prover`` the directory each window's prover state and live rows go in (``--h1-prover``, `prover_of`;
    None: kept nowhere, so no drawn unit or round of it can be proved). ``closer`` (a `worker`) runs each window's
    release off the engine loop, and `close` hands back its outcome's `Future`; None: on the loop, the outcome itself.
    ``draws`` is serve-openai's draw-gated release (`serve_release.Draws`), into which each window's release puts its
    draw under `DRAWER` (`drawn`) when it waits for one. ``clock`` times each part."""

    def __init__(self, model: Any, com: Any, rows: int, steps: int, *, engine: Any, out: Path,
                 config: Mapping[str, Any] | None = None, key: bytes | None = None, unit: Any = None,
                 sessions: GateSessions | None = None, prover: str | Path | None = None, closer: Any = None,
                 draws: Any = None, model_key: Callable[[str], tuple[bytes, str]] | None = None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        from verity_vllm.protocols import pouw
        from verity_vllm.protocols.pouw.h1 import hooks as PHK
        from verity_vllm.protocols.pouw.h1 import served as SV

        self.model, self.com, self.rows, self.steps, self.engine = model, com, int(rows), int(steps), engine
        self.out, self.config, self.unit, self.sessions = Path(out), dict(config or {}), unit, sessions
        self.prover = None if prover is None else Path(prover)
        self.closer, self.clock = closer, clock
        self.draws: Any = None
        self._layout: dict[str, Any] | None = None
        self._period_k: dict[str, int] = {}
        pw = pouw.Settings.of(self.config.get("pouw", {}))
        self.layers = PHK.layers(model, pouw.linear_types(), pw.ks)
        if unit is None:
            self.unit = unit = SV.unit_of(pw.unit)
        if not self.layers:
            raise ValueError(f"{ENTRY}: the served model has no int8 linear H1 records (serve an int8 precision)")
        shapes = [tuple(int(d) for d in m.weight.shape) for m in self.layers]
        modules = {w: f"linear {w}" for w in range(len(self.layers))} | pouw._named(model, self.layers)
        print(f"[serve-h1] declaring the window Program: {self.steps} steps of {len(shapes)} linears at T {self.rows}, "
              f"{self.steps * len(shapes)} calls, {units(shapes, self.rows, self.steps)} tile units", flush=True)
        t0 = time.perf_counter()
        self.program = SV.declared(shapes, modules, self.rows, self.steps, unit=unit)
        t1 = time.perf_counter()
        self.digest = SV.write_program(self.program, SV.declared_layout(shapes, modules, self.rows, self.steps),
                                    self.out / "h1", rows=self.rows, steps=self.steps)
        print(f"[serve-h1] declared in {t1 - t0:.1f} s, its digest {self.digest[:16]} and {PROGRAM} in "
              f"{time.perf_counter() - t1:.1f} s", flush=True)
        registered = None
        if model_key is not None:
            key, registered = model_key(self.digest)
        self.weights = SV.register_weights(self.digest, shapes, SV.weights_of(self.layers), key or os.urandom(32))
        self.weights_key = "local" if registered is None else "auditor"
        if registered is not None and self.weights.root().hex() != registered:
            raise ValueError(f"{ENTRY}: the served weights committed under the auditor's key give root "
                             f"{self.weights.root().hex()[:16]}, not the registration's {registered[:16]}: the served "
                             f"checkpoint or the window Program's linears differ from the registered ones")
        self.tap: Any = None
        self.comp: Any = None
        self.run: Any = None
        self.window: int | None = None
        self.held: dict[str, Any] | None = None
        self.opened: dict[str, float] = {}
        self.outcomes: list[dict[str, Any]] = []
        self.releasing = 0
        self.readied: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def ready(self) -> dict[str, Any]:
        """What window 0's first step would otherwise do before its first token, done now (the module's "Ready before it
        listens"): the weights' roots computed, the call tables' GPU hasher built, the z/y kernels compiled, the Lean
        verifier of record built (the in-process coins' entropy and each release's verdicts) and window 0's reserve ended
        (`GateSessions.settled`, made here unless made ahead already); ``readied``, with where each part's time went
        (``seconds``)."""
        from verity.core.protocols.pouw import verifier
        from verity_vllm.protocols.pouw.h1 import hooks as PHK
        from verity_vllm.protocols.pouw.h1 import tables_device as TD

        t0 = self.clock()
        PHK.weight_roots(self.layers)
        t1 = self.clock()
        tables = "gpu" if TD.available() else "off"
        t2 = self.clock()
        kernels = PHK.warm_zy(self.layers, self.rows)
        t3 = self.clock()
        verifier.executable()
        t4 = self.clock()
        session = None
        if self.sessions is not None:
            self.sessions.ahead_of(0)
            session = (self.sessions.settled(0) or {}).get("session")
        t5 = self.clock()
        self.readied = {"tables": tables, "kernels": kernels, "session": session,
                        "seconds": {"roots": round(t1 - t0, 4), "tables": round(t2 - t1, 4), "kernels": round(t3 - t2, 4),
                                    "lean": round(t4 - t3, 4), "reserve": round(t5 - t4, 4)}}
        print(f"[serve-h1] ready: weight roots in {t1 - t0:.1f} s, the call tables' GPU hasher {tables} in {t2 - t1:.1f} s, "
              f"{kernels} z/y kernels compiled in {t3 - t2:.1f} s, the Lean verifier built in {t4 - t3:.1f} s"
              + ("" if self.sessions is None else f", window 0's session {session or 'not reserved (its open reserves)'}"
                                                  f" after {t5 - t4:.1f} s"), flush=True)
        return self.readied

    def describe(self) -> dict[str, Any]:
        """serve.json's ``protocols``: the entry, T and S, the period's Program digest and its descriptor's file
        (`served.write_program`), the weights' root, where the coins come from (``challenger``, with the gate's peer, or
        ``in-process``), where each window's session is reserved (``ahead``: window 0's at startup and window k+1's while
        window k serves, outside a period), where its release runs and what was made ready before the server listened
        (``ready``, `ready`)."""
        coins = {"coins": "in-process"} if self.sessions is None else {"coins": "challenger", "peer": self.sessions.peer}
        return {"entry": ENTRY, "rows": self.rows, "steps": self.steps, "program": self.digest,
                "program_file": f"h1/{PROGRAM}",
                "weights_root": self.weights.root().hex(), "weights_key": self.weights_key, **coins,
                "reserve": "ahead" if self.sessions is not None and self.sessions.ahead is not None else "at the open",
                "draw": f"put as {DRAWER} into the draw-gated release" if self.draws is not None else None,
                "release": "on the loop" if self.closer is None else "off the loop",
                "linears": len(self.layers), "config": dict(self.config), "ready": self.readied}

    def open(self, window: int) -> None:
        """Window ``window``'s H1, on the open Commit window: its session (with ``sessions``, `GateSessions.take`), the
        tap on the committer, a fresh `Served`, and the option installed with the session's coins, else the epoch
        ``window``'s."""
        from verity_vllm import protocols as PO
        from verity_vllm.acquire import h1_commit as HC
        from verity_vllm.protocols import pouw
        from verity_vllm.protocols.pouw.h1 import served as SV

        if self.comp is not None:
            raise RuntimeError(f"{ENTRY}: window {window} opens while window {self.window}'s H1 is open")
        t0, coins, held = self.clock(), {}, None
        if self.sessions is not None:
            reserved = self.sessions.take(window)
            coins, held = {"challenger": AT, "session": reserved, "gate": self.sessions.open}, \
                self.sessions.held_of(reserved["session"])
        t1 = self.clock()
        self.tap = HC.tap(self.com, self.model, self.layers, range(self.steps))
        prover = self.prover_of(window)
        served = SV.Served(unit=self.unit, weights=self.weights, tap=self.tap, window=os.urandom(64),
                           program=self.program, program_digest=self.digest,
                           prover=None if prover is None else str(prover))
        settings = {**self.config.get("pouw", {}), "rows": self.rows, "steps": self.steps, "served": served,
                    "transcript_dir": str(self.out / "h1" / f"w{window}"), **coins}
        ctx = PO.Context(stage="serve", enabled=(ENTRY,), settings={"pouw": settings})
        try:
            ctx = PO.from_engine(dataclasses.replace(ctx, filled=PO.engine_fills(ctx)), self.engine, epoch=window)
            self.comp = PO.install(self.model, ctx)
        except BaseException:
            self.tap.untap()
            self.tap = None
            raise
        self.run, self.window, self.held = pouw.h1_run(ENTRY), window, held
        self.opened = {"reserve": round(t1 - t0, 4), "install": round(self.clock() - t1, 4)}

    def prover_of(self, window: int) -> Path | None:
        """Window ``window``'s prover directory, ``prover/w<k>`` (owner-only, `pouw_h1_prover.private`), or None
        without ``prover``."""
        from verity_vllm.protocols.pouw.h1 import prover as PV

        return None if self.prover is None else PV.private(PV.private(self.prover) / f"w{window}")

    def close(self, window: int) -> dict[str, Any] | Future | None:
        """Window ``window``'s H1 closed: on the loop its hooks and the tap come off the engine and the committer; then
        its release (`_release`), on ``closer`` when there is one, whose outcome this hands back as a `Future`, else here,
        and the outcome itself; here too when its hooks refused to come off, so nothing off the loop patches the engine.
        None when no H1 window is open; then, if window ``window`` never opened one, its draw-gated release is refused."""
        if self.comp is None:
            if self.draws is not None and window != self.window:
                self.draws.refuse(window, f"{ENTRY}: window {window}'s H1 window never opened, so it has no draw")
            return None
        comp, run, tap, held, opened = self.comp, self.run, self.tap, self.held, self.opened
        self.comp = self.run = self.tap = self.held = None
        t0, unhooked = self.clock(), None
        try:
            comp.unhook()
        except (ValueError, RuntimeError, OSError) as e:          # the window's failure; its release still runs
            unhooked = f"{type(e).__name__}: {e}"
        finally:
            tap.untap()
        t1 = self.clock()
        off = self.closer is not None and unhooked is None
        job = functools.partial(self._release, window, comp, run, held, {**opened, "close_on_loop": round(t1 - t0, 4)},
                                unhooked, t1, off)
        if not off:
            return job()
        with self._lock:
            self.releasing += 1
        return self.closer.submit(job)

    def _release(self, window: int, comp: Any, run: Any, held: dict[str, Any] | None, seconds: dict[str, float],
                 error: str | None, queued: float, off: bool = False) -> dict[str, Any]:
        """Window ``window``'s release: its option closed (the empty steps' calls, its manifest committed, the draw coin
        opened and its session closed), with ``draws`` its draw put (`drawn`), and, with ``sessions``, its end entry
        waited for; its outcome, recorded in ``outcomes``, with ``error`` when any of it failed. ``off``: on ``closer``,
        one of the windows ``releasing``."""
        t0 = t1 = self.clock()
        end = drawn = None
        try:
            try:
                comp.close()
            except (ValueError, RuntimeError, OSError) as e:      # the window's failure, its outcome's error
                error = error or f"{type(e).__name__}: {e}"
            finally:
                t1 = self.clock()
                if self.draws is not None:
                    drawn = self.drawn(window, run, held, error)
                if held:
                    end = self.sessions.ended(held["session"])
            seconds = {**seconds, "queued": round(t0 - queued, 4), "release": round(t1 - t0, 4),
                       "ended": round(self.clock() - t1, 4)}
            out = self._outcome(window, run, held, end, seconds)
            if drawn is not None:
                out["draw"] = drawn
            if error is not None:
                out["error"] = f"close: {error}"[:500]
            with self._lock:
                self.outcomes.append(out)
            return out
        finally:
            if off:
                with self._lock:
                    self.releasing -= 1

    def drawing(self, draws: Any) -> None:
        """Put each window's draw into ``draws`` (`serve_release.Draws`) from now on, when it waits for `DRAWER`'s."""
        self.draws = draws if draws is not None and DRAWER in draws.protocols else None

    def drawn(self, window: int, run: Any, held: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
        """Window ``window``'s draw put into ``draws`` under `DRAWER` once its draw coin is open: the units its coin draws
        (`drawn_units`, at `draw_of`'s K and stream) and the Commit's values they open (`drawn_reads`). Its record for the
        outcome's ``draw``; ``refused`` when the window's release was refused instead (its close failed, or the draw
        can't be formed or put), naming why."""
        try:
            if error is not None:
                raise ValueError(f"its close failed ({error})")
            k, stream, source = self.draw_of(run, held)
            units = drawn_units(run.commits, self.layout(), k, stream)
            reads = drawn_reads(run.served.tap, self.layout(), units)
            self.draws.put(window, DRAWER, reads)
        except (ValueError, KeyError, IndexError, TypeError, RuntimeError, OSError) as e:   # a seal never waits on a draw not coming
            why = f"{ENTRY}: window {window}'s draw: {type(e).__name__}: {e}"[:500]
            self.draws.refuse(window, why)
            return {"refused": why}
        return {"k": k, "k_from": source, "units": len(units), "reads": len(reads),
                "bytes": sum(hi - lo for _s, _n, lo, hi in reads)}

    def layout(self) -> dict[str, Any]:
        """The window Program's layout as core's credit reads it from its descriptor (`credit.layout_of`), once."""
        if self._layout is None:
            from verity.core.protocols.pouw import credit as CR

            from verity_vllm.protocols.pouw.h1 import served as SV

            layout, why = CR.layout_of(SV.descriptor(self.program))
            if why is not None:
                raise ValueError(f"the window Program's layout: {why[0]}: {why[1]}")
            self._layout = layout
        return self._layout

    def draw_of(self, run: Any, held: dict[str, Any] | None) -> tuple[int, str, str]:
        """The K a window is drawn at, its stream, and where that K comes from. A gate session's window: its period-open's
        ``draw_k`` (`credit.committed_k` of the gate's ledger, read once a period) and its open root. An in-process
        window has neither, and credits nothing: the proof service's K (`pouw_h1_challenger.draw_k`) and its calls_root.
        ValueError when the session joined no period or the ledger commits it no K."""
        from verity.core.protocols.pouw import credit as CR

        from verity_vllm.protocols.pouw.h1 import challenger as PHC

        if held is None:
            return PHC.draw_k(self.layout()), run.commits["calls_root"], "the proof service's draw_k (in-process coins)"
        period = held.get("period")
        if period is None:
            raise ValueError(f"session {held.get('session')} joined no period, so no period-open committed the K its "
                             "window is drawn with")
        if period not in self._period_k:
            try:
                k, why = CR.committed_k(self.sessions.gate.ledger(), period)
            except StopIteration:
                k, why = None, ("budget", f"the gate's ledger has no period-open of {period}")
            if why is not None:
                raise ValueError(f"{why[0]}: {why[1]}")
            self._period_k[period] = k
        return self._period_k[period], (run.commits.get("session") or {}).get("root"), f"period {period}'s open"

    def _outcome(self, window: int, run: Any, held: dict[str, Any] | None, end: dict[str, Any] | None,
                 seconds: dict[str, float]) -> dict[str, Any]:
        """Window ``window``'s outcome. With ``prover``, a window that served a step and registered its live tables has
        its rounds' live@<s> rows dumped to ``prover/w<k>/live`` (`pouw_h1_live.dump`), its outcome's ``live_dump``."""
        from verity_vllm.protocols.pouw.h1 import live as LV

        d, c = run.details, run.commits
        live_dump = None
        if self.prover is not None and run.served is not None and run.served.live and d.get("steps_served"):
            try:
                live_dump = LV.dump(run, self.prover_of(window) / "live")
            except (KeyError, ValueError, OSError) as e:
                live_dump = {"error": f"{type(e).__name__}: {e}"[:300]}
        roots1, gate = d.get("roots1"), d.get("gate", {})
        return {"entry": ENTRY, "window": window, "rows": self.rows, "steps": self.steps,
                "steps_served": d.get("steps_served"), "roots1": roots1, "requests": d.get("requests"),
                "static": d.get("static"), "u": d.get("u"), "refused": run.refused[:8],
                "gate": {k: gate.get(k) for k in ("coins", "gate_passed", "refused", "padded", "units")},
                "coin_source": c.get("coin_source"), "calls_root": c.get("calls_root"), "weights_root": c.get("weights_root"),
                "served_refused": list(run.served.refused[:8]) if run.served is not None else None, "live_dump": live_dump,
                "tied": bool(roots1["tied"]) if roots1 is not None and d.get("steps_served") else None,
                "session": None if not held else {**held, "unopened": d.get("unopened"),
                                                  "closed": d.get("challenger"), "end": end},
                "seconds": seconds}

    def drain(self) -> None:
        """Every window's release ended and its outcome recorded, and every session reserved ahead (`GateSessions.drain`)."""
        if self.closer is not None:
            print(f"[serve-h1] stopping: {self.releasing} windows still releasing", flush=True)
            self.closer.shutdown(wait=True)
        if self.sessions is not None:
            self.sessions.drain()

    def summary(self) -> dict[str, Any]:
        """summary.json's ``pouw_h1``: windows closed, tied and refused at ROOTS1 (a window that served no step, whose
        tables H1 never registers, is neither), the rows the gate passed, the windows whose release hasn't ended
        (``releasing``) and the sessions reserved that no window opened (``unused``)."""
        with self._lock:
            outs, releasing = list(self.outcomes), self.releasing
        unused = [] if self.sessions is None else [h["session"] for h in list(self.sessions.held) if "unused" in h]
        return {"windows": len(outs), "tied": sum(o["tied"] is True for o in outs),
                "untied": [o["window"] for o in outs if o["tied"] is False][:20],
                "gate_passed_rows": sum(int(o["gate"]["gate_passed"]["rows"]) for o in outs
                                        if o["gate"]["gate_passed"] is not None),
                "releasing": releasing, "unused": unused[:20]}


def arm_of(opts: Any, llm: Any, com: Any, out: Path) -> H1Windows | None:
    """The server's arm under ``--protocols`` over the served ``llm`` and its committer ``com`` (None without): T
    ``--max-num-batched-tokens`` and S ``--window-steps`` a window, each window's session through `sessions_of` (window
    0's reserved while the Program is declared), its prover state in ``--h1-prover`` and its release on a `worker` of its
    own, made ready (`H1Windows.ready`)."""
    if not opts.protocols:
        return None
    sessions = sessions_of(opts)
    if sessions is not None:
        sessions.ahead_of(0)
    model_key = None
    if sessions is not None and opts.gate.h1_model:
        model_key = model_key_of(sessions.gate, opts.gate.h1_model, opts.gate.h1_weights_root)
    arm = H1Windows(llm.model, com, opts.max_num_batched_tokens, opts.window_steps, engine=llm.vllm, out=out,
                    config=config_of(opts.protocols_config), sessions=sessions, prover=opts.gate.h1_prover,
                    closer=worker("h1-release"), model_key=model_key)
    arm.ready()
    return arm


__all__ = ["AT", "ENTRY", "GateOptions", "GateSessions", "H1Windows", "LISTEN_S", "MARGIN", "OWN", "POLLS", "PROGRAM",
           "PROGRAM_META", "SCHEME", "SESSIONED", "arm_of", "config_of", "listening", "model_key_of", "refusal", "sessions_of",
           "units",
           "worker"]
