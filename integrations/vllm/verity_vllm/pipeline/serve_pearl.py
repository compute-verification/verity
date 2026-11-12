"""serve-openai's Pearl+ arm (``--protocols pouw:pearl-c-h100-v1-h3``): every linear but the LM head served as Pearl+'s FP8
product, one Pearl+ epoch a Commit window, under the server's own Commit, whose window circuit is cut at those linears.

* **The window's composition.** `serve_commit.CommitWindows` opens the arm once the committer is installed and closes it
  first at the window's close. The arm installs, at stage ``commit``, ``pouw:pearl-c-h100-v1-h3`` beside
  ``sampled-proofs:rows-sha512``, the window's circuit cut at Pearl+'s sites (the sampled-proofs setting ``cut``,
  `protocol_options.refusals`). Pearl+'s wrap of each linear's ``quant_method.apply`` sits under the committer's forward
  hooks, so the Commit commits each linear's x and the y Pearl+ returns: the values window-commit's cut reads
  (`window_class.PearlTables`).
* **One epoch a window.** Each install commits the weights under a fresh hm96 salt key (`pouw.commit_weights`), forms B̃
  and takes the epoch's salt coin after the weights' root. Each engine step's forward ends one pass at the engine's
  model runner (the pouw setting ``runner``, `pearl_c.passes.hook_steps`), whose ``execute_model`` the committer patched
  before the window's install, retained in ``pearl/w<k>/pass-NNNNN/`` (owner-only: its manifest keeps the salt key).
  So the window's passes are its Commit's steps in order, round s is its step s, call c of a round its c-th linear (for
  the 1B, layer by layer qkv, o, gate_up and down), and no warm-up pass is in the epoch.
* **Coins.** ``coins(k)`` gives the pouw settings window k's salt coin comes from. With ``--challenger-gate``,
  ``--challenger-peer`` and ``--challenger-lane`` it is a live salt-only session of window k's own on the auditor host's
  challenger (`live`): reserved at the window's open under the lane (`pouw_h1_window.reserve_salt`, joining the lane's
  open period), opened by the install on its salt epoch's root and its salt coin drawn from it
  (`pouw_h1_window.salt_session`). Without them it is an in-process challenger of epoch k over the Lean verifier of
  record's entropy (`verity.core.protocols.pouw.coins.Challenger`, `verifier.coins`), which credits nothing.
* **Its close**, on the engine loop: the hooks off, the option closed (the last pass's files waited for,
  `PearlCDeviceRun.finish`), then ``pearl/w<k>/coins.json``, the epoch's coin commitment (`verify_run.epoch_coins`), and
  ``served.json`` (`pearl_c.window.SERVED_SCHEMA`): the window's passes with calls, each timed, with its manifest's path,
  and its coins (`pearl_c.window.coins_of`), which ``served_passes.py retain --window-record W`` and ``verify --tie W``
  read. A window whose passes with calls aren't its Commit's steps, each with every linear's call, is the window's error.
  The window's seal is the Commit's (``windows/w<k>/``); ``pearl/`` is outside it.
* **The weights' rows** are written once for the served model, before the server listens (`PearlWindows.ready`,
  ``pearl/rows/``), and every window's passes link them (the pouw setting ``kept_rows``): a pass's ``weights/`` holds the
  files its manifest names, as when each window wrote them. A weight's tree is under its window's salt key, so each
  window writes its own.
* **An epoch made ahead.** A window's epoch is made (`pouw.prepare`) in the install's own order: its coins,
  the weights committed under its fresh key, the salt coin after their root, then the device run under that salt. With
  in-process coins it is made ahead of the window: window 0's before the server listens (`PearlWindows.ready`), window
  k+1's at window k's close, on the loop, once window k's epoch is freed; the window's open then only hooks the model
  onto it (the pouw setting ``prepared``). With a live session the same happens under ``--challenger-ahead`` alone,
  since a session made ahead that no window takes is closed unused when the server stops, and an unused session of a
  lane's period refuses the period: without the flag a live window's epoch is made at its open. No session is made
  ahead while the lane has no open period, which it would join none of: that window's epoch is left to its open. At
  most one epoch is ever ahead, and a window must close within its session's minutes of the session's open. An epoch
  that can't be made ahead is its window's visible failure at its open, as when the open made it, and is not made again.

Each window's outcome (`PearlWindows.close`) is its ``windows.jsonl`` line's ``pouw_pearl`` (`PearlWindows.outcome_key`),
``serve.json``'s ``protocols`` `PearlWindows.describe`, ``summary.json``'s ``commit.pouw_pearl`` `PearlWindows.summary`.
``--protocols-config`` holds the pouw settings of `CONFIG` alone, ``{"pouw": {"device": "cuda", "kernel": DIR, "cubin":
FILE}}`` (`config_of`; ``served_passes.py arm`` writes a spec arm's)."""
from __future__ import annotations

import dataclasses
import gc
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from verity.ml.protocols.pouw.schemes import pearl_c as C
from verity.ml.protocols.pouw.schemes.pearl_c_device import H100

from verity_vllm.log import init_logger

logger = init_logger(__name__)

#: pearl_c_sm90's scheme, one of `pearl_c.SCHEMES`: v1 forming on the device record of an H100 or H200, -h3 hashing
SCHEME = C.PearlC(forming="v1", device=H100, hashing="h3").name
ENTRY = f"pouw:{SCHEME}"
#: the precision the arm serves at: the BF16 checkpoint's own weights, which Pearl+ reads unquantized
PRECISION = "bfloat16"
#: the pouw settings ``--protocols-config`` may give; the server sets the rest (the transcript dir, the coins) per window
CONFIG = ("device", "kernel", "cubin", "schedule")
#: where each window's epoch goes under ``--out``: ``pearl/w<k>/``
DIR = "pearl"
#: where the served model's weight rows are kept under `DIR`, beside the windows
ROWS = "rows"
#: the subprotocol whose draw serve-openai's draw-gated release waits for (`serve_h1.DRAWER`), which Pearl+ never puts
DRAWER = "pouw"
#: serve-openai's gate options a live salt session takes, all three or none (`refusal`, `live`)
GATE = ("challenger_gate", "challenger_peer", "challenger_lane")
#: ... and the one that makes each live window's epoch, its session with it, ahead of the window (`PearlWindows.ahead`)
AHEAD = "challenger_ahead"
#: where a live session's challenger is reached: this node's 127.0.0.1, at the port the gate's open answers
AT = "127.0.0.1:0"


def config_of(value: str | None) -> dict[str, Any]:
    """``--protocols-config``: a JSON object, or a JSON file, ``{"pouw": {...}}`` of `CONFIG`'s keys; `UsageError` on any
    other key, the server's own among them, or a value `pouw.Settings` refuses."""
    from verity_vllm.config import UsageError
    from verity_vllm.protocols import load_settings
    from verity_vllm.protocols import pouw as PW

    try:
        doc = load_settings(json.loads(value) if value and value.lstrip().startswith("{") else value)
    except (OSError, ValueError) as e:
        raise UsageError(f"--protocols-config {str(value)[:80]}: {e}") from None
    if stray := sorted(set(doc) - {"pouw"}):
        raise UsageError(f"--protocols-config: {stray} are not Pearl+'s (pouw)")
    pouw = doc.get("pouw", {})
    if stray := sorted(set(pouw) - set(CONFIG)):
        raise UsageError(f"--protocols-config: the pouw settings {stray} are not Pearl+'s to give ({', '.join(CONFIG)}): "
                         "the server sets each window's transcript dir and coins itself")
    try:
        PW.Settings.of(pouw)
    except ValueError as e:
        raise UsageError(f"--protocols-config: {e}") from None
    return {"pouw": dict(pouw)}


def context(config: Mapping[str, Any], pouw: Mapping[str, Any] | None = None) -> Any:
    """The window's composition at stage ``commit``: Pearl+ beside sampled proofs, the circuit cut at Pearl+'s sites, its
    pouw settings ``config``'s with ``pouw`` (the window's transcript dir and coins) over them."""
    from verity_vllm import protocols as PO
    from verity_vllm.protocols import spot as SP

    return PO.Context(stage="commit", enabled=PO.parse(f"{ENTRY},{SP.NAME}:{SP.DEFAULT_SCHEME}"),
                      settings={"pouw": {**config.get("pouw", {}), **(pouw or {})}, SP.NAME: {PO.CUT: ENTRY}})


def refusal(opts: Any) -> str | None:
    """Why ``opts`` can't serve ``--protocols`` `ENTRY` (None when they can, or another arm is asked for); `config_of`'s
    `UsageError` when it refuses ``--protocols-config``."""
    from verity_vllm import protocols as PO

    if opts.protocols != ENTRY:
        return None
    after = getattr(getattr(opts, "retention", None), "release_after", "") or ""
    if DRAWER in (p.strip() for p in after.split(",")):
        return (f"--release-after {DRAWER} waits for PoUW H1's draw of each window, and Pearl+ puts none: its verifier's "
                "coin is drawn after its window record W (served_passes.py verify)")
    g = opts.gate
    if h1 := [f"--{f.name.replace('_', '-')}" for f in dataclasses.fields(g)
              if f.name not in (*GATE, AHEAD) and getattr(g, f.name)]:
        return (f"{', '.join(h1)}: PoUW H1's, not Pearl+'s, whose salt coin comes from a live salt-only session "
                "(--challenger-gate, --challenger-peer, --challenger-lane) or in-process (serve_pearl)")
    if len({bool(getattr(g, f)) for f in GATE}) > 1:
        return ("--challenger-gate, --challenger-peer and --challenger-lane go together under Pearl+: the gate's command, "
                "this machine's name in the auditor host's peers, and the lane whose gate key the command holds, whose open "
                "period each window's salt-only session joins")
    if getattr(g, AHEAD) and not g.challenger_gate:
        return ("--challenger-ahead makes each window's live session ahead of the window: it needs --challenger-gate, "
                "--challenger-peer and --challenger-lane (in-process coins are always made ahead)")
    if opts.precision != PRECISION:
        return (f"--protocols {ENTRY} serves the BF16 checkpoint's linears as Pearl+'s FP8 product: it needs --precision "
                f"{PRECISION}, not {opts.precision}")
    if not opts.commit or opts.scheme != "rows-sha512" or not opts.required_manifest:
        return (f"--protocols {ENTRY} runs under the server's rows-sha512 Commit, whose window circuit is cut at its "
                "linears: it needs --commit, --scheme rows-sha512 and the row's --required-manifest")
    if bad := PO.refusals(context(config_of(opts.protocols_config))):
        return f"--protocols {ENTRY}: " + "; ".join(bad)
    return None


def in_process(window: int) -> dict[str, Any]:
    """Window ``window``'s coins held in this process: epoch ``window``'s challenger over the Lean verifier of record's
    entropy, which credits nothing."""
    from verity.core.protocols.pouw import verifier
    from verity.core.protocols.pouw.coins import Challenger

    return {"coins": Challenger(window, verifier.coins)}


@dataclasses.dataclass(frozen=True)
class Live:
    """Each window's coins from a live salt-only session of its own on the auditor host's challenger, through infra's
    ``gate`` (holding lane ``lane``'s gate key) as peer ``peer``: window k's reserved when it is called (``reserve
    --peer PEER``, Pearl-C's salt coin alone, epoch k: `pouw_h1_window.reserve_salt`, joining the lane's open period,
    refused before reserving while that period is H1's), which the install opens on its salt epoch's root and draws the
    salt coin from (`pouw_h1_window.salt_session`), through the settings ``challenger`` (`AT`), ``session`` (the reserve
    answer), ``gate`` and ``period``. A refusal is `CoinError` (a ValueError), the window's failure."""

    gate: Any
    peer: str
    lane: str

    def __call__(self, window: int, ledger: list[str] | None = None) -> dict[str, Any]:
        from verity_vllm.protocols.pouw.h1 import window as PHW

        g = self.gate
        reserved, period = PHW.reserve_salt(g.call, g.ledger if ledger is None else lambda: ledger, self.lane, self.peer,
                                            window)
        logger.info("window %d: reserved salt-only session %s on %s (lane %s, period %s)", window, reserved.get("session"),
                    self.peer, self.lane, None if period is None else period.get("period"))
        return {"challenger": AT, "session": reserved, "gate": PHW.gate_of(g.call), "period": period}

    def ahead(self, window: int) -> dict[str, Any] | None:
        """Window ``window``'s coins for an epoch made ahead of the window (`PearlWindows.make_ahead`), or None while
        the lane has no open period: a session reserved then would join none, where the one its window's open reserves
        joins the period opened meanwhile.  The reserve is held to the ledger read here."""
        from verity_vllm.protocols.pouw.h1 import window as PHW

        ledger = self.gate.ledger()
        return None if PHW.open_period(ledger, self.lane) is None else self(window, ledger)


def live(gate: str, peer: str, lane: str, run: Any = None) -> Live:
    """`Live` coins through infra's gate ``gate``, its command as one string; ``run`` replaces the gate's subprocess
    (tests)."""
    from verity_vllm.pipeline.commit_coins import ChallengerGate

    return Live(ChallengerGate(gate, run), peer, lane)


@dataclasses.dataclass
class Made:
    """Window ``window``'s epoch as `PearlWindows.make` made it, starting at ``at`` on the arm's clock: the ``epoch``
    (`pearl_c.epoch.Prepared`), the coin settings it was made with (``coins``; a live session's reserve is their
    ``session``) and each part's ``seconds``; or, made ahead, why it couldn't be (``error``), which is the window's
    failure at its open."""

    window: int
    at: float
    epoch: Any = None
    coins: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    seconds: Mapping[str, float] = dataclasses.field(default_factory=dict)
    error: Exception | None = None


def _session(made: Made) -> str:
    """The live session ``made``'s epoch holds, for a log line: nothing for in-process coins."""
    reserved = made.coins.get("session")
    if reserved is None:
        return ""
    return f" (live session {reserved['session']}, reserved and opened; unused, it refuses its lane's open period)"


class PearlWindows:
    """The arm over ``model`` (the served `verity_vllm.LLM`'s), its model ``runner`` and the server's committer ``com``:
    ``config`` the pouw settings of ``--protocols-config`` (`config_of`), ``out`` the serve dir each window's epoch goes
    under (``pearl/w<k>/``), ``run_id`` the run its ``served.json`` names, ``coins`` each window's coin settings
    (`in_process` when None). ``ahead``: each window's epoch is made ahead of its open (`ready`, `close`); None is
    in-process coins' own answer, yes, and given coins', no. ``clock`` times each part."""

    #: the window line's and the summary's key for this arm's outcomes
    outcome_key = "pouw_pearl"

    def __init__(self, model: Any, runner: Any, com: Any, *, out: Path, config: Mapping[str, Any], run_id: str = "local",
                 coins: Callable[[int], Mapping[str, Any]] | None = None, ahead: bool | None = None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        from verity_vllm.protocols import pouw

        self.model, self.runner, self.com, self.out, self.run_id = model, runner, com, Path(out).resolve(), str(run_id)
        self.config = {"pouw": dict(config.get("pouw", {}))}
        self.coins, self.given = coins or in_process, coins is not None
        self.ahead = (not self.given) if ahead is None else bool(ahead)
        self.made: Made | None = None                    # the one epoch made ahead, until its window opens
        self.aheads = {"made": 0, "taken": 0, "unused": 0, "failed": []}
        self.clock = clock
        self.linears = sum(isinstance(m, pouw.linear_types()) for m in model.modules())
        if not self.linears:
            raise ValueError(f"{ENTRY}: the served model has no linear for Pearl+ to serve")
        self.comp: Any = None
        self.window: int | None = None
        self.reserved: Any = None
        self.opened: dict[str, float] = {}
        self.outcomes: list[dict[str, Any]] = []
        self.readied: dict[str, Any] | None = None
        self.rows: Mapping[str, Any] | None = None

    def ready(self) -> dict[str, Any]:
        """What window 0's open would otherwise do first, done before the server listens: the kernel loaded
        (`pearl_c.Kernel.load`, once a process), for in-process coins the Lean verifier of record built
        (`verifier.executable`, their entropy), the weights' rows written once for every window's passes to link
        (`pouw.keep_rows`, in `dir_of` `ROWS`), and under ``ahead`` window 0's epoch made (`make_ahead`); ``readied``,
        with each part's time."""
        from verity.core.protocols.pouw import verifier
        from verity_vllm import protocols as PO
        from verity_vllm.protocols import pouw
        from verity_vllm.protocols.pouw import pearl_c as PPD

        pw, t0 = self.config["pouw"], self.clock()
        PPD.Kernel.load(pw["kernel"], pw.get("cubin"))
        t1 = self.clock()
        if not self.given:
            verifier.executable()
        t2 = self.clock()
        self.rows = pouw.keep_rows(SCHEME, PO.Target(self.model, context(self.config)), self.dir_of(ROWS))
        t3 = self.clock()
        epoch = {"epoch": self.make_ahead(0)} if self.ahead else {}
        self.readied = {"seconds": {"kernel": round(t1 - t0, 4), "lean": round(t2 - t1, 4), "rows": round(t3 - t2, 4),
                                    **epoch},
                        "rows": {"files": len(self.rows), "bytes": sum(e["bytes"] for _, e in self.rows.values())}}
        print(f"[serve-pearl] ready: the kernel loaded in {t1 - t0:.1f} s"
              + ("" if self.given else f", the Lean verifier built in {t2 - t1:.1f} s")
              + f", {len(self.rows)} weights' rows kept in {t3 - t2:.1f} s"
              + (f", window 0's epoch made in {epoch['epoch']:.1f} s" if epoch else ""), flush=True)
        return self.readied

    def describe(self) -> dict[str, Any]:
        """serve.json's ``protocols``: the entry, the window's composition and its cut, where each window's epoch goes,
        where its coins come from, the linears a pass calls, the config and what `ready` did."""
        ctx = context(self.config)
        return {"entry": ENTRY, "composition": list(ctx.enabled), "cut": ENTRY, "stage": ctx.stage,
                "epoch": f"{DIR}/w<k>/", "rows": f"{DIR}/{ROWS}/", "coins": "given" if self.given else "in-process",
                "ahead": self.ahead, "release": "on the loop",
                "linears": self.linears, "config": self.config, "sealed": f"no: {DIR}/ is outside the window's seal",
                "ready": self.readied}

    def dir_of(self, window: int | str) -> Path:
        """Window ``window``'s epoch directory, ``pearl/w<k>/`` under ``out``, owner-only; `ROWS`' is ``pearl/rows/``."""
        d = self.out / DIR / (window if isinstance(window, str) else f"w{window}")
        for p in (d.parent, d):
            p.mkdir(mode=0o700, exist_ok=True)
            p.chmod(0o700)
        return d

    def make(self, window: int, ahead: bool = False) -> Made | None:
        """Window ``window``'s epoch made, the model untouched (`pouw.prepare`): its coins (``coins``: a live session
        reserved, or the in-process challenger), then the weights committed under its key, the salt coin after their
        root and the device run under that salt, with each part's seconds.  None, with nothing made, where it is asked
        for ``ahead`` of the window and its coins are not to be had ahead yet (`Live.ahead`)."""
        from verity_vllm import protocols as PO
        from verity_vllm.protocols import pouw

        t0 = self.clock()
        coins = (getattr(self.coins, "ahead", self.coins) if ahead else self.coins)(window)
        if coins is None:
            return None
        coins = dict(coins)
        t1 = self.clock()
        kept = {} if self.rows is None else {"kept_rows": self.rows}
        epoch = pouw.prepare(SCHEME, PO.Target(self.model, context(self.config, {**kept, **coins})))
        return Made(window, t0, epoch, coins, {"coins": round(t1 - t0, 4), **epoch.seconds})

    def make_ahead(self, window: int) -> float:
        """Window ``window``'s epoch made now, ahead of its open, which takes it (``made``); the seconds it took. One it
        can't make is kept as that window's failure, raised at its open as when the open made it; one whose coins are
        not to be had ahead yet (`make`) is left to that open."""
        self.drop()
        t0 = self.clock()
        try:
            self.made = self.make(window, ahead=True)
            if self.made is None:
                logger.info("window %d: its epoch is left to its open: its lane has no open period for a session "
                            "reserved now to join", window)
            else:
                self.aheads["made"] += 1
                logger.info("window %d: its epoch made ahead in %.1f s%s", window, self.clock() - t0, _session(self.made))
        except (ValueError, RuntimeError, OSError) as e:
            self.made = Made(window, t0, error=e)
            self.aheads["failed"] = [*self.aheads["failed"], window][:20]
            logger.warning("window %d: its epoch can't be made ahead: %s: %s", window, type(e).__name__, str(e)[:300])
        return round(self.clock() - t0, 4)

    def drop(self) -> None:
        """The epoch made ahead that no window took, closed: its run finished and its live session closed, unused."""
        made, self.made = self.made, None
        if made is not None and made.epoch is not None:
            self.aheads["unused"] += 1
            logger.info("window %d: its epoch made ahead is closed unused, no window having opened on it%s", made.window,
                        _session(made))
            try:
                made.epoch.close()
            except (ValueError, RuntimeError, OSError) as e:
                logger.warning("window %d: its unused epoch's close: %s: %s", made.window, type(e).__name__, str(e)[:300])

    def open(self, window: int) -> None:
        """Window ``window``'s epoch, on the open Commit window: the one made ahead for it, else made now (`make`), and
        the window's composition installed over it (`context`, the pouw setting ``prepared``), which hooks the model
        onto its run, its passes retained in `dir_of`.  ``opened``: the open's seconds on the loop (``install``), of
        them the hooks' (``hooks``), what making the epoch took wherever it was made (``epoch``), and for one made
        ahead how long before the open its making began, a live session's reserve first (``ahead``)."""
        from verity_vllm import protocols as PO

        if self.comp is not None:
            raise RuntimeError(f"{ENTRY}: window {window} opens while window {self.window}'s epoch is open")
        t0 = self.clock()
        made, ahead = self.made, {}
        if made is not None and made.window == window:
            self.made = None
            if made.error is not None:
                raise made.error
            self.aheads["taken"] += 1
            ahead = {"ahead": round(t0 - made.at, 4)}
        else:
            self.drop()
            made = self.make(window)
        t1 = self.clock()
        kept = {} if self.rows is None else {"kept_rows": self.rows}
        ctx = context(self.config, {"transcript_dir": str(self.dir_of(window)), "runner": self.runner, **kept, **made.coins,
                                    "prepared": made.epoch})
        try:
            self.comp = PO.install(self.model, ctx)
        except BaseException:
            made.epoch.close()             # nothing once the install took it, which closes its session itself
            raise
        self.window, self.reserved = window, made.coins.get("session")
        t2 = self.clock()
        self.opened = {"install": round(t2 - t0, 4), "hooks": round(t2 - t1, 4), "epoch": dict(made.seconds), **ahead}

    def close(self, window: int) -> dict[str, Any] | None:
        """Window ``window``'s epoch closed, on the loop: its hooks off, its option closed, its ``coins.json`` and
        ``served.json`` written (`record`); its outcome, with ``error`` when any of it failed. None when no epoch is open.
        Under ``ahead`` the next window's epoch is then made (`make_ahead`), its seconds the outcome's ``next``. The
        outcome's ``seconds`` are the open's (`open`), the close's parts and where its passes' went (``passes``,
        `pearl_c.parts.summary`: the first pass's parts and the later ones' summed)."""
        from verity_vllm.protocols.pouw.pearl_c import parts as PPA

        if self.comp is None:
            return None
        comp, reserved, seconds = self.comp, self.reserved, dict(self.opened)
        self.comp = self.reserved = None
        steps, error, t0 = int(self.com._step_index), None, self.clock()
        try:
            comp.unhook()
        except (ValueError, RuntimeError, OSError) as e:          # the window's failure; its option still closes
            error = f"unhook: {type(e).__name__}: {e}"
        t1 = self.clock()
        try:
            comp.close()
        except (ValueError, RuntimeError, OSError) as e:
            error = error or f"close: {type(e).__name__}: {e}"
        t2 = self.clock()
        opt = comp.get("pouw")
        try:
            out = self.record(window, opt, steps, reserved, error)
        except (KeyError, OSError, ValueError) as e:
            out = {"entry": ENTRY, "window": window, "steps": steps, "error": f"record: {type(e).__name__}: {e}"[:500]}
        passes = PPA.summary(opt.details.get("seconds", ()))
        del comp, opt
        gc.collect()                       # the epoch's device buffers (B̃, the pass buffers) freed before the next is made
        t3 = self.clock()
        after = {"next": self.make_ahead(window + 1)} if self.ahead else {}
        out["seconds"] = {**seconds, "unhook": round(t1 - t0, 4), "release": round(t2 - t1, 4), "record": round(t3 - t2, 4),
                          "passes": passes, **after}
        self.outcomes.append(out)
        return out

    def record(self, window: int, opt: Any, steps: int, reserved: Any = None, error: str | None = None) -> dict[str, Any]:
        """Window ``window``'s ``coins.json`` and ``served.json`` from its closed option ``opt``, whose Commit committed
        ``steps`` steps, and its outcome: ``error`` (``error`` given, or the passes with calls not the Commit's steps, a
        pass short of a linear's call or without its manifest) when it isn't one Pearl+ round a step."""
        from verity_vllm.protocols.pouw.pearl_c import window as PWR

        d, c = self.dir_of(window), opt.commitment()
        passes = [p for p in PWR.served_passes(opt.details["passes"]) if p["units"]]
        bad = ([] if len(passes) == steps else
               [f"its passes with calls are {len(passes)} and its Commit's steps {steps}, so round s isn't step s"])
        bad += [f"pass {p['pass']} has {p['units']} calls, not one a linear ({self.linears})" for p in passes
                if p["units"] != self.linears][:4]
        bad += [f"pass {p['pass']} retained no manifest" for p in passes if not p.get("manifest")][:4]
        error = error or ("; ".join(bad) or None)
        (d / "coins.json").write_text(json.dumps(c["coins"]) + "\n")
        doc = {"schema": PWR.SERVED_SCHEMA, "entry": ENTRY, "run": self.run_id, "window": window, "retained": True,
               "commit_steps": steps, "passes": passes, "passes_with_calls": len(passes),
               "coins": PWR.coins_of(opt, reserved), "ok": error is None, "error": error}
        (d / "served.json").write_text(json.dumps(doc, indent=1) + "\n")
        out = {"entry": ENTRY, "window": window, "steps": steps, "passes": len(passes),
               "calls": sum(p["units"] for p in passes), "salt": c.get("salt"), "weights_root": c.get("weights_root"),
               "coin_source": doc["coins"]["source"], "served": str(d.relative_to(self.out) / "served.json")}
        return out if error is None else {**out, "error": error[:500]}

    def drawing(self, draws: Any) -> None:
        """Pearl+ puts no draw into serve-openai's draw-gated release (`refusal` refuses ``--release-after pouw``): its
        verifier's coin is drawn after W."""

    def drain(self) -> None:
        """Each window's release ran on the loop at its close; the epoch made ahead that no window took is closed
        (`drop`)."""
        self.drop()

    def summary(self) -> dict[str, Any]:
        """summary.json's ``pouw_pearl``: the windows closed, those whose epoch is one Pearl+ round a step, and their
        passes and calls; under ``ahead``, the epochs made ahead, those a window took, those closed unused, and the
        windows whose epoch couldn't be made ahead."""
        ok = [o for o in self.outcomes if "error" not in o]
        return {"windows": len(self.outcomes), "served": len(ok), "passes": sum(o["passes"] for o in ok),
                "calls": sum(o["calls"] for o in ok), **({"ahead": dict(self.aheads)} if self.ahead else {})}


def arm_of(opts: Any, llm: Any, com: Any, out: Path) -> PearlWindows | None:
    """The server's arm under ``--protocols`` `ENTRY` over the served ``llm``, its model runner and its committer ``com``
    (None without), its coins a live salt-only session a window under ``--challenger-gate`` (`live`), made ahead of
    the window only under ``--challenger-ahead``, else in-process and always made ahead; made ready
    (`PearlWindows.ready`)."""
    from verity_vllm.engine.handle import model_runner

    if opts.protocols != ENTRY:
        return None
    g = opts.gate
    coins = live(g.challenger_gate, g.challenger_peer, g.challenger_lane) if g.challenger_gate else None
    arm = PearlWindows(llm.model, model_runner(llm.vllm), com, out=out, config=config_of(opts.protocols_config),
                       run_id=opts.run_id, coins=coins, ahead=bool(getattr(g, AHEAD)) if coins is not None else None)
    arm.ready()
    return arm


__all__ = ["AT", "CONFIG", "DIR", "DRAWER", "ENTRY", "GATE", "PRECISION", "ROWS", "SCHEME", "PearlWindows", "arm_of",
           "config_of", "context", "in_process", "live", "refusal"]
