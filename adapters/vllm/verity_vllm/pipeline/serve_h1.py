"""serve-openai's PoUW H1 arm (``--protocols pouw:eps1-sm120-h1fx1``): one H1 window a Commit period, tied to the server's
own Commit.

* **One Commit.** H1 builds no committer: its tap goes on the server's rows-sha512 collector (`acquire.h1_commit.tap`),
  which already acquires each linear's x_q and s_a (the bench row's int8 source) and each step's input ids (its step-input
  port), so ROOTS1 holds H1's A tables and input ids to the values the served window commits.
* **One window a Commit period.** `serve_commit.CommitWindows` opens the arm once the committer is installed, so H1's
  hooks sit on top of the committer's, and closes it first thing at the window's close, before the committer comes off.
  The H1 window's T is the engine's token budget (``--max-num-batched-tokens``) and its S the Commit period's steps
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
  challenger's, one session a window through infra's gate (`GateSessions`, `commit_coins.ChallengerGate`): window 0's
  session is reserved (scheme `pouw_h1_challenger.SCHEME`, `pouw_h1_challenger.families` (S)) at startup (`arm_of`) and
  window k+1's while window k serves, off the engine loop (`GateSessions.take`), each with its sampling session,
  reserved withheld, beside it (`pouw_h1_challenger.draw_families`); round 0 opens it on the window's open root
  (`pouw_h1_window.Window`), reached on this machine's 127.0.0.1 at the port the gate's open answers once the auditor
  host's ``ssh -R`` listens there (looked for every `LISTEN_S`), and the sampling session opens on the sample root once
  the prover's commitment is in (`pouw_h1_window.DrawSession`), its coin commitment released only at its close; the
  release waits for both end entries. The gate's open answers a port no session without an end entry holds, so a window
  whose session opens while the window before is still releasing takes a port of its own. A reserve joins its lane's
  open ``period``, if one is open, so one is opened before the server starts for window 0's to join; and a reserve made
  ahead that no window opens (the server stopped first, window 0's included) would leave its ``period`` incomplete, so
  the next window's is reserved ahead only while the last reserve joined none, and one no window opened is ``unused`` in
  the summary. Without a gate, the epoch's in-process coins, one epoch a window (`pouw_h1_settings.from_engine`): such a
  window credits nothing (`pouw_h1.IN_PROCESS`). ``--protocols-config`` sets no session (``SESSIONED``).
* **Ready before it listens.** `H1Windows.ready`, before serve-openai listens, does what the first block's first step
  would otherwise do before its first token: the weights' roots (`pouw_h1_hooks.weight_roots`, which each install
  reuses), the call tables' GPU hasher built (`pouw_h1_tables_device.available`), every z/y kernel a step of up to T
  rows runs compiled (`pouw_h1_hooks.warm_zy`), the Lean verifier built (`verifier.executable`, once a
  process: in a fresh tree a minute's build, which in-process coins otherwise run at the first one's sample), and the wait
  for that block's reserve (window 0's, which `arm_of` makes while the Program is declared). Round 0's open stays at the block's first admission: its open root binds that round's
  requests, and the call tables are bound under the salt the open samples.
* **Off the engine loop.** At a window's close the loop takes only H1's hooks and tap off; its release (the empty
  steps' calls, the manifest's commit, the sampling coin, the session's close and the wait for its end entry) runs on
  ``closer``, one thread in window order, while the next window serves. Each window still has its own session,
  coins and transcript, and its outcome is recorded in window order once its release ends. With ``--h1-prover`` each
  served call's prover state is copied off the device and written by the server's `pouw_h1_prover.Writer`, whose ring
  holds a window's bytes at most, and the window's release waits for its files: one not written is the window's error,
  ``kept`` in its outcome otherwise.
* **Its sample, for the draw-gated release** (``--retain device --seal --release-after pouw``, `serve_release`). Once a
  window's sampling coin is open, its release puts H1's sample of the window under `DRAWER` (`H1Windows.drawn`): the proof
  blocks core samples (`credit.drawn` over the window Program's layout, as `pouw_h1_challenger.drawn_of` samples them), each
  read as the Commit's values it opens (`drawn_reads`): its call's x_q and s_a rows of its row tile at the engine step its
  step maps to. With a gate session its stream is the window's open root and its K its period-open's ``draw_k``; with
  in-process coins (no open root, no period, nothing credited) its calls_root and the proof system's K
  (`pouw_h1_challenger.draw_k`). A window whose sample can't be formed (it never opened, its close failed, no sampling coin,
  no period) refuses its release by name, so its seal fails visibly and never waits.
* **Its Commit record, tied to the audited window** (``--seal --h1-prover --h1-window-commit CMD``, `serve_tie`). Its
  release, once H1 has closed the window and before its sample coin, waits for the window's seal, writes the window's
  registration W and runs CMD, V3c's window-commit, whose record R reuses H1's tables from W; R is the window's Commit
  record in the tap's place (`pouw_h1_commit.bind`), its outcome's ``commit_record``. It never runs with
  ``--release-after pouw``, whose seal waits for this release's draw.

Each window's outcome (`H1Windows.close`) is its ``windows.jsonl`` line's ``pouw_h1``: ROOTS1's verdict, the requests'
admission and gaps, the credit gate, the coins, the commitment's calls root, its gate session (the reserve, the open's
port, how the session closed and its end entry), its receipts' calls and time (``receipts``, ``--h1-receipts``) and
where its time went (``seconds``); a window whose H1 failed to open or close carries its error there instead, and the
Commit serves on. ``serve.json``'s ``protocols`` is `H1Windows.describe`, ``summary.json``'s ``pouw_h1``
`H1Windows.summary` with the windows that failed.
``--protocols-config`` is the pouw settings the server doesn't set itself, ``{"pouw": {...}}`` (`config_of`). Its ``unit``
and ``ks`` shape the declared Program: ``{"pouw": {"unit": "j10r", "ks": [2048]}}`` declares J10-R's tile unit
(`pouw_h1_served.unit_of`) over the linears at K 2,048 alone, the form a period's guarantee covers (core's
`period_check.form`), and the linears at any other K serve on their arm unrecorded, uncredited as attention is. Unset,
the Program is v1's over every linear."""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from verity_vllm.pipeline.serve_h1_draw import drawn_reads, drawn_units, units
from verity_vllm.pipeline.serve_h1_gate import LISTEN_S, MARGIN, POLLS, GateSessions, listening
from verity_vllm.config import option
from verity_vllm.protocols.pouw.h1 import PROGRAM, PROGRAM_META, SCHEME

ENTRY = f"pouw:{SCHEME}"
#: the pouw settings the server sets for each window itself
OWN = ("rows", "steps", "runner", "served", "transcript_dir", "coins", "rounds", "commit_record")
#: the pouw settings of a gate session a window, which serve-openai sets itself from ``--challenger-gate`` (`GateSessions`)
SESSIONED = ("challenger", "session", "gate", "period")
#: the challenger as the window reaches it: this machine's 127.0.0.1, its port the one the gate's open answers
#: (`pouw_h1_window.at_port`; the gate's 7811..7899)
AT = "127.0.0.1:7811"
#: the subprotocol H1's sample of each window is put under in serve-openai's draw-gated release (``--release-after``)
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
    h1_receipts: bool = option("--h1-receipts", default=False, action="store_true",
                               help="with --protocols: each H1 window registers each call's receipt@<c>, the receipts a "
                                    "lanes-form K's combine reads (pouw_h1_receipts.Receipts.serving), computed inline")
    h1_model: str | None = option("--h1-model", default=None, metavar="ROLE",
                                  help="with --challenger-gate: the auditor's registration role of the served model (a role "
                                       "of the pinned tree's manifests/checkpoints-int8.json, the gate's models.MANIFESTS); "
                                       "its b/<K> are committed under the salt key the gate's "
                                       "model-key hands for the window Program's circuit, so the windows' weights root is the "
                                       "registry's, the root the period signs; without it the key is local and credits nothing")
    h1_weights_root: str | None = option("--h1-weights-root", default=None, metavar="HEX",
                                         help="with --h1-model: the registration's weights root (the period's signed one), "
                                              "required when the role is registered more than once under the circuit")
    h1_window_commit: str | None = option("--h1-window-commit", default=None, metavar="CMD",
                                          help="with --protocols, --seal and --h1-prover: at each H1 window's close, once it "
                                               "is sealed and before its sample coin, its registration W is written and CMD "
                                               "run as `bash -c CMD _ SERVE_DIR K W_FILE R_FILE PROVER_DIR` (V3c's "
                                               "window-commit, writing R, the audited circuit's registration reusing H1's "
                                               "tables from W); R is the window's Commit record (serve_tie)")


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
                         "served tables, Commit record and coins, one epoch a window)")
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
    if (opts.gate.h1_prover or opts.gate.h1_receipts or opts.gate.h1_window_commit) and not opts.protocols:
        return f"--h1-prover, --h1-receipts and --h1-window-commit are PoUW H1's window's: they need --protocols {ENTRY}"
    if opts.gate.h1_window_commit and not (getattr(opts, "seal", False) and opts.gate.h1_prover):
        return "--h1-window-commit commits the sealed window from H1's prover state: it needs --seal and --h1-prover"
    if opts.gate.h1_window_commit and DRAWER in (p.strip() for p in after.split(",")):
        return (f"--h1-window-commit waits for the window's seal before its sample coin, and --release-after {DRAWER} holds "
                "the seal until that coin's draw: give one of them")
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
    None: kept nowhere, so no sampled proof block or round of it can be proved), the calls' state written by ``writer`` (a
    `pouw_h1_prover.Writer` whose ring holds the window's bytes, `pouw_h1_prover.kept_bytes`), whose files a window's
    release waits for: a call's not written is its error, so its sample is refused, and the outcome's ``kept`` otherwise
    (`pouw_h1_prover.Writer.wait`); ``receipts`` each window's `Served` registers each call's ``receipt@<c>``
    (``--h1-receipts``, `pouw_h1_receipts.Receipts.serving`). ``closer`` (a `worker`) runs each window's
    release off the engine loop, and `close` hands back its outcome's `Future`; None: on the loop, the outcome itself.
    ``draws`` is serve-openai's draw-gated release (`serve_release.Draws`), into which each window's release puts its
    sample under `DRAWER` (`drawn`) when it waits for one. ``tie`` (`serve_tie.WindowTie`, ``--h1-window-commit``) names
    each window's Commit record at its close, once the window is sealed (`sealed`) and its prover state written (`kept`),
    so its release runs on ``closer``. ``clock`` times each part."""

    def __init__(self, model: Any, com: Any, rows: int, steps: int, *, engine: Any, out: Path,
                 config: Mapping[str, Any] | None = None, key: bytes | None = None, unit: Any = None,
                 sessions: GateSessions | None = None, prover: str | Path | None = None, closer: Any = None,
                 draws: Any = None, model_key: Callable[[str], tuple[bytes, str]] | None = None,
                 clock: Callable[[], float] = time.perf_counter, receipts: bool = False, tie: Any = None) -> None:
        from verity_vllm.protocols import pouw
        from verity_vllm.protocols.pouw import h1 as PH
        from verity_vllm.protocols.pouw.h1 import hooks as PHK
        from verity_vllm.protocols.pouw.h1 import prover as PV
        from verity_vllm.protocols.pouw.h1 import served as SV

        self.model, self.com, self.rows, self.steps, self.engine = model, com, int(rows), int(steps), engine
        self.out, self.config, self.unit, self.sessions = Path(out), dict(config or {}), unit, sessions
        self.prover = None if prover is None else Path(prover)
        self.closer, self.clock, self.receipts, self.tie = closer, clock, bool(receipts), tie
        if tie is not None and closer is None:
            raise ValueError(f"{ENTRY}: a window's tie waits for its seal, which comes after its close: its release runs "
                             "off the loop, on closer")
        self.draws: Any = None
        self._kept: dict[int, Any] = {}
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
        self.writer = None if prover is None else PV.Writer(self.steps * sum(
            PV.kept_bytes(self.rows, k, PH.units(k, n, self.rows)) for n, k in shapes))
        self.outcomes: list[dict[str, Any]] = []
        self.releasing = 0
        self.readied: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def ready(self) -> dict[str, Any]:
        """What window 0's first step would otherwise do before its first token, done now (the module's "Ready before it
        listens"): the weights' roots computed, the call tables' GPU hasher built, the z/y kernels compiled (with
        ``receipts`` noise.cu loaded too), the Lean verifier built (the in-process coins' entropy and each
        release's verdicts) and window 0's reserve ended (`GateSessions.settled`, made here unless made ahead already);
        ``readied``, with where each part's time went (``seconds``)."""
        from verity.protocols.pouw.verifier import verifier
        from verity_vllm.protocols.pouw.h1 import hooks as PHK, receipts as RC
        from verity_vllm.protocols.pouw.h1 import tables_device as TD

        t0 = self.clock()
        PHK.weight_roots(self.layers)
        t1 = self.clock()
        tables = "gpu" if TD.available() else "off"
        t2 = self.clock()
        kernels = PHK.warm_zy(self.layers, self.rows)
        if self.receipts:                                   # loads noise.cu (`receipts.loaded`), which every window reuses
            RC.Receipts.serving(None, None, self.layers)
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
                "prover_ring": None if self.writer is None else self.writer.bound,
                "linears": len(self.layers), "config": dict(self.config), "ready": self.readied, "receipts": self.receipts,
                "commit_record": "the tap's" if self.tie is None else {"named": "at the close", **self.tie.describe()}}

    def open(self, window: int) -> None:
        """Window ``window``'s H1, on the open Commit period: its session (with ``sessions``, `GateSessions.take`), the
        tap on the committer, a fresh `Served`, and the option installed with the session's coins, else the epoch
        ``window``'s."""
        from verity_vllm import protocols as PO
        from verity_vllm.acquire import h1_commit as HC
        from verity_vllm.protocols import pouw
        from verity_vllm.protocols.pouw.h1 import receipts as RC, served as SV

        if self.comp is not None:
            raise RuntimeError(f"{ENTRY}: window {window} opens while window {self.window}'s H1 is open")
        t0, coins, held = self.clock(), {}, None
        if self.sessions is not None:
            reserved = self.sessions.take(window)
            coins, held = {"challenger": AT, "session": reserved, "draw_session": self.sessions.draw_session(reserved["session"]),
                           "gate": self.sessions.open}, self.sessions.held_of(reserved["session"])
        t1 = self.clock()
        self.tap = HC.tap(self.com, self.model, self.layers, range(self.steps))
        prover = self.prover_of(window)
        served = SV.Served(unit=self.unit, weights=self.weights, tap=self.tap,
                           window=None if self.sessions is not None else os.urandom(64), program=self.program,
                           program_digest=self.digest, prover=None if prover is None else str(prover))
        settings = {**self.config.get("pouw", {}), "rows": self.rows, "steps": self.steps, "served": served,
                    "transcript_dir": str(self.out / "h1" / f"w{window}"), **coins}
        if self.tie is not None:
            settings["commit_record"] = self.tie.hook(window, prover, None if self.writer is None else
                                                      functools.partial(self.kept, window))
        ctx = PO.Context(stage="serve", enabled=(ENTRY,), settings={"pouw": settings})
        if self.writer is not None:
            self.writer.open(prover)
        try:
            served.receipts = RC.Receipts.serving(served, None, self.layers) if self.receipts else None
            ctx = PO.from_engine(dataclasses.replace(ctx, filled=PO.engine_fills(ctx)), self.engine, epoch=window)
            self.comp = PO.install(self.model, ctx)
        except BaseException:
            self.tap.untap()
            self.tap = None
            if self.writer is not None:
                self.writer.wait(prover)
            raise
        self.run, self.window, self.held = pouw.h1_run(ENTRY), window, held
        if served.receipts is not None:
            served.receipts.bind(self.run)
        self.opened = {"reserve": round(t1 - t0, 4), "install": round(self.clock() - t1, 4)}

    def prover_of(self, window: int) -> Path | None:
        """Window ``window``'s prover directory, ``prover/w<k>`` (owner-only, `pouw_h1_prover.private`), or None
        without ``prover``."""
        from verity_vllm.protocols.pouw.h1 import prover as PV

        return None if self.prover is None else PV.private(PV.private(self.prover) / f"w{window}")

    def kept(self, window: int) -> dict[str, int]:
        """Window ``window``'s prover state once written (`pouw_h1_prover.Writer.wait`, which closes its directory, so it is
        waited for once, by its tie's command or else its release): its calls, bytes and waits. OSError as the writer's."""
        if window not in self._kept:
            try:
                self._kept[window] = self.writer.wait(self.prover_of(window))
            except OSError as e:
                self._kept[window] = e
        if isinstance(got := self._kept[window], OSError):
            raise got
        return got

    def sealed(self, window: int, line: Mapping[str, Any]) -> None:
        """`serve_commit.CommitWindows`' seal line of window ``window``, or why it has none: its tie's, with a tie."""
        if self.tie is not None:
            self.tie.sealed(window, line)

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
            if self.tie is not None:
                self.tie.refuse(window, f"its hooks refused to come off ({unhooked}), so it is released on the engine "
                                        "loop, which seals it after")
            return job()
        with self._lock:
            self.releasing += 1
        return self.closer.submit(job)

    def _release(self, window: int, comp: Any, run: Any, held: dict[str, Any] | None, seconds: dict[str, float],
                 error: str | None, queued: float, off: bool = False) -> dict[str, Any]:
        """Window ``window``'s release: its option closed (the empty steps' calls, its manifest committed, the sampling coin
        opened and its session closed), with ``draws`` its sample put (`drawn`), and, with ``sessions``, its end entry
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
                kept = None
                if self.writer is not None:
                    try:
                        kept = self.kept(window)
                    except OSError as e:                        # its prover state not all written: the window's failure
                        error = error or f"its prover state: {e}"
                    self._kept.pop(window, None)
                if self.tie is not None:
                    self.tie.bound(window, run)
                t1 = self.clock()
                if self.draws is not None:
                    drawn = self.drawn(window, run, held, error)
                if held:
                    end = self.sessions.ended(held["session"])
                    if isinstance(held.get("draw"), dict):
                        held["draw"]["end"] = self.sessions.ended(held["draw"]["session"])
            seconds = {**seconds, "queued": round(t0 - queued, 4), "release": round(t1 - t0, 4),
                       "ended": round(self.clock() - t1, 4)}
            out = self._outcome(window, run, held, end, seconds)
            if drawn is not None:
                out["draw"] = drawn
            if kept is not None:
                out["kept"] = kept
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
        """Put each window's sample into ``draws`` (`serve_release.Draws`) from now on, when it waits for `DRAWER`'s."""
        self.draws = draws if draws is not None and DRAWER in draws.protocols else None

    def drawn(self, window: int, run: Any, held: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
        """Window ``window``'s sample put into ``draws`` under `DRAWER` once its sampling coin is open: the units its coin samples
        (`drawn_units`, at `draw_of`'s K and stream) and the Commit's values they open (`drawn_reads`). Its record for the
        outcome's ``draw``; ``refused`` when the window's release was refused instead (its close failed, or the sample
        can't be formed or put), naming why."""
        try:
            if error is not None:
                raise ValueError(f"its close failed ({error})")
            k, stream, source = self.draw_of(run, held)
            units = drawn_units(run.commits, self.layout(), k, stream)
            reads = drawn_reads(run.served.tap, self.layout(), units, S_A_BYTES)
            self.draws.put(window, DRAWER, reads)
        except (ValueError, KeyError, IndexError, TypeError, RuntimeError, OSError) as e:   # a seal never waits on a sample not coming
            why = f"{ENTRY}: window {window}'s draw: {type(e).__name__}: {e}"[:500]
            self.draws.refuse(window, why)
            return {"refused": why}
        return {"k": k, "k_from": source, "units": len(units), "reads": len(reads),
                "bytes": sum(hi - lo for _s, _n, lo, hi in reads)}

    def layout(self) -> dict[str, Any]:
        """The window Program's layout as core's credit reads it from its descriptor (`credit.layout_of`), once."""
        if self._layout is None:
            from verity.protocols.pouw.verifier import credit as CR

            from verity_vllm.protocols.pouw.h1 import served as SV

            layout, why = CR.layout_of(SV.descriptor(self.program))
            if why is not None:
                raise ValueError(f"the window Program's layout: {why[0]}: {why[1]}")
            self._layout = layout
        return self._layout

    def draw_of(self, run: Any, held: dict[str, Any] | None) -> tuple[int, str, str]:
        """The K a window is sampled at, its stream, and where that K comes from. A gate session's window: its period-open's
        ``draw_k`` (`credit.committed_k` of the gate's ledger, read once a period) and its open root. An in-process
        window has neither, and credits nothing: the proof system's K (`pouw_h1_challenger.draw_k`) and its calls_root.
        ValueError when the session joined no period or the ledger commits it no K."""
        from verity.protocols.pouw.verifier import credit as CR

        from verity_vllm.protocols.pouw.h1 import challenger as PHC

        if held is None:
            return PHC.draw_k(self.layout()), run.commits["calls_root"], "the proof system's draw_k (in-process coins)"
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
        session = run.commits.get("session")                # None: no gate session carried the run's coins
        return self._period_k[period], session.get("root") if session else None, f"period {period}'s open"

    def _outcome(self, window: int, run: Any, held: dict[str, Any] | None, end: dict[str, Any] | None,
                 seconds: dict[str, float]) -> dict[str, Any]:
        """Window ``window``'s outcome. With ``prover``, a window that served a step and registered its live tables has
        its rounds' live@<s> rows dumped to ``prover/w<k>/live`` (`pouw_h1_live.dump`), its outcome's ``live_dump``;
        with ``receipts``, the calls its producer computed and their time, its ``receipts``."""
        from verity_vllm.protocols.pouw.h1 import live as LV

        d, c = run.details, run.commits
        live_dump = None
        if self.prover is not None and run.served is not None and run.served.live and d.get("steps_served"):
            try:
                live_dump = LV.dump(run, self.prover_of(window) / "live")
            except (KeyError, ValueError, OSError) as e:
                live_dump = {"error": f"{type(e).__name__}: {e}"[:300]}
        roots1, gate, rc = d.get("roots1"), d.get("gate", {}), getattr(run.served, "receipts", None) if self.receipts else None
        return {"entry": ENTRY, "window": window, "rows": self.rows, "steps": self.steps,
                "steps_served": d.get("steps_served"), "roots1": roots1, "requests": d.get("requests"),
                "static": d.get("static"), "u": d.get("u"), "refused": run.refused[:8],
                "gate": {k: gate.get(k) for k in ("coins", "gate_passed", "refused", "padded", "units")},
                "coin_source": c.get("coin_source"), "calls_root": c.get("calls_root"), "weights_root": c.get("weights_root"),
                "served_refused": list(run.served.refused[:8]) if run.served is not None else None, "live_dump": live_dump,
                "tied": bool(roots1["tied"]) if roots1 is not None and d.get("steps_served") else None,
                "session": None if not held else {**held, "unopened": d.get("unopened"),
                                                  "closed": d.get("challenger"), "end": end},
                "receipts": None if rc is None else {"calls": rc.calls, "seconds": round(rc.seconds, 4)},
                "commit_record": d.get("commit_record"), "seconds": seconds}

    def drain(self) -> None:
        """Every window's release ended and its outcome recorded (a tie's window still waiting for a seal refused: the
        server's sealer has closed), every session reserved ahead (`GateSessions.drain`), and the prover state's writer
        closed."""
        if self.tie is not None:
            self.tie.stop()
        if self.closer is not None:
            print(f"[serve-h1] stopping: {self.releasing} windows still releasing", flush=True)
            self.closer.shutdown(wait=True)
        if self.sessions is not None:
            self.sessions.drain()
        if self.writer is not None:
            self.writer.close()

    def summary(self) -> dict[str, Any]:
        """summary.json's ``pouw_h1``: windows closed, tied and refused at ROOTS1 (a window that served no step, whose
        tables H1 never registers, is neither), the rows the gate passed, the windows whose release hasn't ended
        (``releasing``) and the sessions reserved that no window opened (``unused``)."""
        with self._lock:
            outs, releasing = list(self.outcomes), self.releasing
        unused = [] if self.sessions is None else [d["session"] for h in list(self.sessions.held)
                                                   for d in (h, h.get("draw")) if d and "unused" in d]
        return {"windows": len(outs), "tied": sum(o["tied"] is True for o in outs),
                "untied": [o["window"] for o in outs if o["tied"] is False][:20],
                "gate_passed_rows": sum(int(o["gate"]["gate_passed"]["rows"]) for o in outs
                                        if o["gate"]["gate_passed"] is not None),
                "releasing": releasing, "unused": unused[:20]}


def arm_of(opts: Any, llm: Any, com: Any, out: Path, flush: Callable[[], Any] | None = None) -> H1Windows | None:
    """The server's arm under ``--protocols`` over the served ``llm`` and its committer ``com`` (None without): T
    ``--max-num-batched-tokens`` and S ``--window-steps`` a window, each window's session through `sessions_of` (window
    0's reserved while the Program is declared), its prover state in ``--h1-prover``, its Commit record named by
    ``--h1-window-commit`` (`serve_tie.WindowTie`, after ``flush``, the advice trace's drain) and its release on a
    `worker` of its own, made ready (`H1Windows.ready`)."""
    from verity_vllm.pipeline.serve_tie import WindowTie

    if not opts.protocols:
        return None
    tie = WindowTie(opts.gate.h1_window_commit, out, flush=flush) if opts.gate.h1_window_commit else None
    sessions = sessions_of(opts)
    if sessions is not None:
        sessions.ahead_of(0)
    model_key = None
    if sessions is not None and opts.gate.h1_model:
        model_key = model_key_of(sessions.gate, opts.gate.h1_model, opts.gate.h1_weights_root)
    arm = H1Windows(llm.model, com, opts.max_num_batched_tokens, opts.window_steps, engine=llm.vllm, out=out,
                    config=config_of(opts.protocols_config), sessions=sessions, prover=opts.gate.h1_prover,
                    closer=worker("h1-release"), model_key=model_key, receipts=opts.gate.h1_receipts, tie=tie)
    arm.ready()
    return arm


__all__ = ["AT", "ENTRY", "GateOptions", "GateSessions", "H1Windows", "LISTEN_S", "MARGIN", "OWN", "POLLS", "PROGRAM",
           "PROGRAM_META", "SCHEME", "SESSIONED", "arm_of", "config_of", "listening", "model_key_of", "refusal", "sessions_of",
           "units", "worker"]
