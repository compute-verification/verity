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
  key of this process's: no auditor's registration (``model-key``) is read yet. The Program holds a node for each tile
  unit (`units`), S × slot(T)/32 × Σn/32, and declaring it and its digest take time linear in them before the server
  listens: about 7 minutes for Llama-3.2-1B at T 512 and S 16, about 3 minutes a step at T 2048. The server logs both.
* **Coins.** With ``--challenger-gate CMD`` and ``--challenger-peer NAME`` (`sessions_of`), the auditor host's
  challenger's, one session a window through infra's gate (`GateSessions`, `commit_coins.Gate`): the window's open
  reserves it (scheme `pouw_h1_challenger.SCHEME`, `pouw_h1_challenger.families` (S)), round 0 opens it on the window's
  open root (`pouw_h1_window.Window`), reached on this machine's 127.0.0.1 at the port the gate's open answers once the
  auditor host's ``ssh -R`` listens there, and the window's close waits for its end entry, so the next window's
  session takes that port only after this one's tunnel has gone. A reserve joins its lane's open period, if one is
  open. Without a gate, the epoch's in-process coins, one epoch a window (`pouw_h1_settings.from_engine`): such a
  window credits nothing (`pouw_h1.IN_PROCESS`). ``--protocols-config`` sets no session (``SESSIONED``).

Each window's outcome (`H1Windows.close`) is its ``windows.jsonl`` line's ``pouw_h1``: ROOTS1's verdict, the requests'
admission and gaps, the credit gate, the coins, the commitment's calls root and its gate session (the reserve, the open's
port, how the session closed and its end entry); a window whose H1 failed to open or close
carries its error there instead, and the Commit serves on. ``serve.json``'s ``protocols`` is `H1Windows.describe`,
``summary.json``'s ``pouw_h1`` `H1Windows.summary` with the windows that failed. ``--protocols-config`` is the pouw
settings the server doesn't set itself, ``{"pouw": {...}}`` (`config_of`)."""
from __future__ import annotations

import dataclasses
import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from verity_vllm.protocols.pouw.h1 import SCHEME

ENTRY = f"pouw:{SCHEME}"
#: the pouw settings the server sets for each window itself
OWN = ("rows", "steps", "runner", "served", "transcript_dir", "coins", "rounds")
#: the pouw settings of a gate session a window, which serve-openai sets itself from ``--challenger-gate`` (`GateSessions`)
SESSIONED = ("challenger", "session", "gate", "period")
#: the challenger as the window reaches it: this machine's 127.0.0.1, its port the one the gate's open answers
#: (`pouw_h1_window.at_port`; the gate's 7811..7899)
AT = "127.0.0.1:7811"
#: how many times, a second apart, a session's state is polled: its ``ssh -R`` port listening here, its end entry
POLLS = 300


def config_of(value: str | None) -> dict[str, Any]:
    """``--protocols-config``: a JSON object, or a JSON file, ``{"pouw": {...}}``; empty for None. `UsageError` on any key
    the server sets itself (`OWN`), one it can't serve yet (`SESSIONED`), or a value `pouw.Settings` refuses."""
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
    """Why ``opts`` can't serve ``--protocols`` (None when they can, or none is asked for)."""
    gate, peer = getattr(opts, "challenger_gate", None), getattr(opts, "challenger_peer", None)
    if bool(gate) != bool(peer):
        return ("--challenger-gate and --challenger-peer go together: the gate's command and this machine's name in the "
                "auditor host's peers, where its challenger reaches the server (ssh -R)")
    if gate and not opts.protocols:
        return f"--challenger-gate draws PoUW H1's coins: it needs --protocols {ENTRY}"
    if not opts.protocols:
        return None
    if opts.protocols != ENTRY:
        return f"--protocols {opts.protocols}: serve-openai serves {ENTRY} (PoUW H1) only"
    if not opts.commit or opts.scheme != "rows-sha512" or not opts.required_manifest:
        return (f"--protocols {ENTRY} taps the server's rows-sha512 Commit (its int8 source and step-input port): it needs "
                "--commit, --scheme rows-sha512 and the bench row's --required-manifest")
    if opts.max_num_batched_tokens % 32:
        return f"--protocols {ENTRY}: H1's T, --max-num-batched-tokens {opts.max_num_batched_tokens}, is a multiple of 32"
    return None


def units(shapes: Any, rows: int, steps: int) -> int:
    """The tile units of the window Program `served.declared` builds over ``shapes`` (each ``(n, K)``) at T ``rows`` and S
    ``steps``: one node of its body each, so declaring it and its digest cost time linear in them."""
    from verity.core.protocols.pouw.circuit import h1window as HW

    from verity_vllm.protocols.pouw import h1 as PH

    r, c = PH.TILE
    return int(steps) * (HW.slot(int(rows), PH.TILE) // r) * sum(int(n) // c for n, _k in shapes)


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
    `commit_coins.Gate`) as peer `peer`, for windows of S `steps` rounds. `reserve` is a window's reserve, `open` (the
    pouw setting ``gate``) opens it on the window's open root and `ended` waits for its end entry. ``held`` is each
    window's session as the gate answered it. `wait` sleeps between polls and `listens` is `listening` (tests)."""

    def __init__(self, gate: Any, peer: str, steps: int, *, wait: Callable[[float], Any] = time.sleep,
                 listens: Callable[[int], bool] = listening) -> None:
        from verity_vllm.protocols.pouw.h1 import window as PHW

        self.gate, self.peer, self.steps, self.wait, self.listens = gate, peer, int(steps), wait, listens
        self._open = PHW.gate_of(gate.call)
        self.held: list[dict[str, Any]] = []

    def reserve(self, window: int) -> dict[str, Any]:
        """Window `window`'s session reserved (``reserve --peer PEER --scheme SCHEME --families families(S)``): the gate's
        reserve answer as the pouw setting ``session`` takes it (`pouw_h1_window.reserved_of`). `CoinError` when the gate
        refuses."""
        from verity_vllm.protocols.pouw.h1 import challenger as PHC
        from verity_vllm.protocols.pouw.h1 import window as PHW

        answer = self.gate.call("reserve", "--peer", self.peer, "--scheme", PHC.SCHEME, "--families",
                                PHC.families(self.steps))
        reserved = PHW.reserved_of({k: v for k, v in answer.items() if k in ("session", *PHW.RESERVE_FIELDS)})
        self.held.append({"window": window, **reserved})
        print(f"[serve-h1] window {window}: reserved session {reserved['session']} on {self.peer} "
              f"(period {reserved.get('period')})", flush=True)
        return reserved

    def open(self, session: str, root: str) -> dict[str, Any]:
        """`pouw_h1_window.Gate`: ``open --session SESSION --root ROOT`` (its commitment from ``status`` when the answer's
        is null), answered once the answer's port listens here (the auditor host's ``ssh -R``, polled `POLLS` times)."""
        answer = self._open(session, root)
        port, polls = answer.get("port"), None
        if type(port) is int:
            for i in range(POLLS):
                if self.listens(port):
                    polls = i
                    break
                self.wait(1.0)
        held = next((h for h in self.held if h["session"] == session), {})
        held.update(port=port, listen_polls=polls)
        print(f"[serve-h1] session {session} opened on {root[:16]}…: port {port} "
              + (f"listening after {polls} polls" if polls is not None else f"not listening after {POLLS} polls"),
              flush=True)
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
        body = (end.get("body") or {}) if isinstance(end, Mapping) else {}
        return {"opened": s.get("open") is not None, **{k: body.get(k) for k in ("rc", "closed", "complete")}}


def sessions_of(opts: Any) -> GateSessions | None:
    """The server's `GateSessions` from ``--challenger-gate`` and ``--challenger-peer`` (`refusal` holds them together),
    for windows of ``--window-steps``; None without a gate."""
    if not getattr(opts, "challenger_gate", None):
        return None
    from verity_vllm.pipeline import commit_coins as CC

    return GateSessions(CC.Gate(opts.challenger_gate), opts.challenger_peer, opts.window_steps)


class H1Windows:
    """The arm over ``model`` (the served `verity_vllm.LLM`'s) and the server's committer ``com``: T ``rows`` and S
    ``steps`` a window, ``config`` the pouw settings of ``--protocols-config``, ``engine`` the built ``vllm.LLM`` the
    runner and each window's coins come from, ``out`` where each window's transcript goes (``h1/w<k>/``). ``key`` is the
    weights' registration key (a fresh one when None); ``unit`` the Program's tile unit at each K (`h1window.tile_unit`
    when None); ``sessions`` each window's session on the auditor host's challenger (`GateSessions`; None: in-process
    coins)."""

    def __init__(self, model: Any, com: Any, rows: int, steps: int, *, engine: Any, out: Path,
                 config: Mapping[str, Any] | None = None, key: bytes | None = None, unit: Any = None,
                 sessions: GateSessions | None = None) -> None:
        from verity.core.protocols.sampled_proofs.one_stage import partition as P

        from verity_vllm.protocols import pouw
        from verity_vllm.protocols.pouw.h1 import hooks as PHK
        from verity_vllm.protocols.pouw.h1 import served as SV

        self.model, self.com, self.rows, self.steps, self.engine = model, com, int(rows), int(steps), engine
        self.out, self.config, self.unit, self.sessions = Path(out), dict(config or {}), unit, sessions
        self.layers = PHK.layers(model, pouw.linear_types())
        if not self.layers:
            raise ValueError(f"{ENTRY}: the served model has no int8 linear H1 records (serve an int8 precision)")
        shapes = [tuple(int(d) for d in m.weight.shape) for m in self.layers]
        modules = {w: f"linear {w}" for w in range(len(self.layers))} | pouw._named(model, self.layers)
        print(f"[serve-h1] declaring the window Program: {self.steps} steps of {len(shapes)} linears at T {self.rows}, "
              f"{units(shapes, self.rows, self.steps)} tile units", flush=True)
        t0 = time.perf_counter()
        self.program = SV.declared(shapes, modules, self.rows, self.steps, unit=unit)
        t1 = time.perf_counter()
        self.digest = P.program_digest(self.program)
        print(f"[serve-h1] declared in {t1 - t0:.1f} s, its digest {self.digest[:16]} in {time.perf_counter() - t1:.1f} s",
              flush=True)
        self.weights = SV.register_weights(self.digest, shapes, SV.weights_of(self.layers), key or os.urandom(32))
        self.tap: Any = None
        self.comp: Any = None
        self.run: Any = None
        self.window: int | None = None
        self.outcomes: list[dict[str, Any]] = []

    def describe(self) -> dict[str, Any]:
        """serve.json's ``protocols``: the entry, T and S, the period's Program digest, the weights' root and where the
        coins come from (``challenger``, with the gate's peer, or ``in-process``)."""
        coins = {"coins": "in-process"} if self.sessions is None else {"coins": "challenger", "peer": self.sessions.peer}
        return {"entry": ENTRY, "rows": self.rows, "steps": self.steps, "program": self.digest,
                "weights_root": self.weights.root().hex(), "weights_key": "local", **coins,
                "linears": len(self.layers), "config": dict(self.config)}

    def open(self, window: int) -> None:
        """Window ``window``'s H1, on the open Commit window: its session reserved (with ``sessions``), the tap on the
        committer, a fresh `Served`, and the option installed with the session's coins, else the epoch ``window``'s."""
        from verity_vllm import protocols as PO
        from verity_vllm.acquire import h1_commit as HC
        from verity_vllm.protocols import pouw
        from verity_vllm.protocols.pouw.h1 import served as SV

        if self.comp is not None:
            raise RuntimeError(f"{ENTRY}: window {window} opens while window {self.window}'s H1 is open")
        coins = {} if self.sessions is None else {"challenger": AT, "session": self.sessions.reserve(window),
                                                  "gate": self.sessions.open}
        self.tap = HC.tap(self.com, self.model, self.layers, range(self.steps))
        served = SV.Served(unit=self.unit, weights=self.weights, tap=self.tap, window=os.urandom(64),
                           program=self.program, program_digest=self.digest)
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
        self.run, self.window = pouw.h1_run(ENTRY), window

    def close(self, window: int) -> dict[str, Any] | None:
        """Window ``window``'s H1 closed and released (its hooks off, its manifest committed, the draw coin opened, its
        session closed and, with ``sessions``, its end entry waited for), the tap off the committer; its outcome. None
        when no H1 window is open."""
        if self.comp is None:
            return None
        run, held, end = self.run, self.sessions and self.sessions.held[-1], None
        try:
            self.comp.close()
        finally:
            self.comp = self.run = None
            self.tap.untap()
            if held:
                end = self.sessions.ended(held["session"])
        d, c = run.details, run.commits
        roots1, gate = d.get("roots1"), d.get("gate", {})
        out = {"entry": ENTRY, "window": window, "rows": self.rows, "steps": self.steps,
               "steps_served": d.get("steps_served"), "roots1": roots1, "requests": d.get("requests"),
               "static": d.get("static"), "u": d.get("u"), "refused": run.refused[:8],
               "gate": {k: gate.get(k) for k in ("coins", "gate_passed", "refused", "padded", "units")},
               "coin_source": c.get("coin_source"), "calls_root": c.get("calls_root"), "weights_root": c.get("weights_root"),
               "served_refused": list(run.served.refused[:8]) if run.served is not None else None,
               "tied": bool(roots1["tied"]) if roots1 is not None and d.get("steps_served") else None,
               "session": None if not held else {**held, "unopened": d.get("unopened"),
                                                 "closed": d.get("challenger"), "end": end}}
        self.outcomes.append(out)
        return out

    def summary(self) -> dict[str, Any]:
        """summary.json's ``pouw_h1``: windows closed, tied and refused at ROOTS1 (a window that served no step, whose
        tables H1 never registers, is neither), and the rows the gate passed."""
        return {"windows": len(self.outcomes), "tied": sum(o["tied"] is True for o in self.outcomes),
                "untied": [o["window"] for o in self.outcomes if o["tied"] is False][:20],
                "gate_passed_rows": sum(int(o["gate"]["gate_passed"]["rows"]) for o in self.outcomes
                                        if o["gate"]["gate_passed"] is not None)}


__all__ = ["AT", "ENTRY", "GateSessions", "H1Windows", "OWN", "POLLS", "SCHEME", "SESSIONED", "config_of", "listening",
           "refusal", "sessions_of", "units"]
