"""Pearl+ as a resource scheme (`verity.protocols.resource_accounting.scheme`), registered as ``pearl-plus``.

Its verifier is `verity/pouw/pearl-plus/v1` (`verifier/lean/pearlplus/PouwPearlPlus/Protocol.lean`). A tile that passes
the tile rule costs its credited cells at the record's kept width |K| (`Credit.lean`'s `creditAtAssumption`), so a
block's cost reads the record's kept width (``K_KEPT``, `UnitRecord.k_kept`) beside the tile's TM × TN, not the
Definition's K, which is the unit's padded width. The cost is of a tile whose every cell is credited: filler, excluded and
given-up rows credit none (`UnitRecord.credited`).

On a served run (`attach`) it computes every linear block with its device run (`verity.hooks.ComputeBlockWith`), the
workload's `PearlCDeviceRun` (`serving.tensors.pearl_c_device`), over the weight operands the run registers
(``memory``): one epoch a Commit phase (`PhaseOpened` to `PhaseClosed`), or one for the server's life where no phase
arrives (`Attached.step`); outside an epoch a block is the workload's own. It ends the run's pass once each step's blocks
are issued (``steps``, `StepComputed`). It provides the ``cut`` (`Scheme.provides`): a phase's circuit is cut at the
linears it computes, whose calls commit their A rows and y. Its settings are
`SETTINGS`; ``tie-uniform`` lets a uniform sampling rate stand under the tie to its tables, with the weaker tie its
record states.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity import hooks as H
from verity.protocols.resource_accounting.compute.pouw.prover import workload
from verity.protocols.resource_accounting.scheme import Block, Charge, Order, Scheme, Term, register

OPTION = "pouw:pearl-c-h100-v1-h3"
#: the weight formats its kernel reads, quantizing them itself; a quantized (FP8) weight is refused, not misread
READS = ("bf16", "fp16", "fp32")


def tile_macs(shape: Mapping[str, int]) -> int:
    return shape["TM"] * shape["TN"] * shape["K_KEPT"]


#: Its settings (`Scheme.settings`), each a default of its type: ``kernel`` the kernel's library (required on a served
#: run), ``cubin`` its cubin, ``schedule`` the schedule each pass runs under (`serving.SCHEDULES`),
#: ``transcript_dir`` where passes are retained (a phase's under ``w<phase>/``, owner-only, with its ``served.json``),
#: the salt's source, ``coins`` (``in-process``: a challenger held in this
#: process, crediting nothing) or ``challenger`` (the auditor device's, HOST:PORT), whose sessions the launcher's
#: ``ctx["gate"]``, the gate in its own words, opens: ``session``, a reserve answer the gate already gave, opened by the
#: first epoch, and ``lane`` and ``peer``, under which each later epoch reserves its own (`Attached.reserve`); and
#: ``tie-uniform``
SETTINGS: dict[str, Any] = {"tie-uniform": False, "kernel": "", "cubin": "", "schedule": "serial", "transcript_dir": "",
                            "coins": "", "challenger": "", "session": {}, "lane": "", "peer": ""}
COINS = ("", "in-process")


def check_settings(own: Mapping[str, Any]) -> str | None:
    from verity.protocols.resource_accounting.compute.pouw.prover import serving as SV

    if own["schedule"] not in SV.SCHEDULES:
        return f"schedule is one of {', '.join(SV.SCHEDULES)}, not {own['schedule']!r}"
    if own["coins"] not in COINS:
        return f"coins is 'in-process' (a challenger held in this process) or unset, not {own['coins']!r}"
    if own["coins"] and own["challenger"]:
        return "coins and challenger are both set: the salt has one source"
    host, sep, port = own["challenger"].rpartition(":")
    if own["challenger"] and not (sep and host and port.isdecimal()):
        return f"challenger is the auditor device's challenger as HOST:PORT, not {own['challenger']!r}"
    if bool(own["lane"]) != bool(own["peer"]):
        return "lane and peer go together: the lane whose gate key reserves, and this machine's name in the auditor's peers"
    if (own["session"] or own["lane"]) and not own["challenger"]:
        return "session, lane and peer are a challenger's: they reserve or name its sessions"
    if own["challenger"] and not (own["session"] or own["lane"]):
        return "a challenger needs a session (the gate's reserve answer) or lane and peer to reserve each epoch's"
    if own["session"] and not isinstance(own["session"].get("session"), str):
        return f"session is the gate's reserve answer, {{'session': SESSION_ID, ...}}, not {own['session']!r}"
    return None


#: The transforms its device run applies to a checkpoint's weights before committing them: none.  The keyed 8-block
#: rotation (`registration.ROTATION_8`) is not built on the device, so it registers only a curated unrotated checkpoint,
#: and no curated list is wired.
SERVED_TRANSFORMS: tuple[str, ...] = ()
#: What its commitment says of weights that are not registered (`registration`)
UNREGISTERED = "nothing about which weights were served: the checkpoint is not registered"


def registration(scheme: Any, refused: type[Exception]) -> dict[str, Any]:
    """What it claims of the weights it serves, the commitment's ``registration``: registered weights only when
    `SERVED_TRANSFORMS` pass the registered-weights rule (``scheme.check_registration``, which raises ``refused``);
    otherwise the claim says it is about nothing (`UNREGISTERED`) and names the refusal, and serving goes on, since the
    salt is still the challenger's live coin."""
    out: dict[str, Any] = {"transforms": list(SERVED_TRANSFORMS), "curated": False}
    try:
        scheme.check_registration(SERVED_TRANSFORMS, False, False)
    except refused as e:
        return {**out, "registered": False, "claims": UNREGISTERED, "refused": str(e)}
    return {**out, "registered": True}


#: The file in the scheme's private directory (``ctx["private"]``) that holds the last salt epoch number it spent
EPOCH_FILE = "epoch"


def spend_epoch(private: str | os.PathLike[str]) -> int:
    """The next salt epoch's number (`salt.epoch_open_root`'s k), written to ``private``'s `EPOCH_FILE`, synced with its
    directory, before it is returned: a killed server's number is never handed out again, so no later server opens a
    session on a root already used (the gate's one session a root); a number may be skipped."""
    path = Path(private) / EPOCH_FILE
    try:
        text = path.read_text()
    except FileNotFoundError:
        k = 0
    else:
        if not re.fullmatch(r"(0|[1-9][0-9]*)\n", text) or int(text) >= 2**64 - 1:
            raise ValueError(f"pearl-plus: {path} holds {text[:40]!r}, not the last salt epoch number spent below 2^64 - 1: "
                             "refused rather than risk a number already used")
        k = int(text) + 1
    tmp = path.with_name(f"{EPOCH_FILE}.next")
    with open(tmp, "w") as f:
        f.write(f"{k}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return k


def requests(ctx: Any) -> tuple[H.RequireProperty, ...]:
    return (H.RequireProperty("eager", True, "its kernel at each linear block is not graph-capturable"),
            H.RequireProperty("world", 1, "its passes and transcript are per rank, which is not wired"))


class Attached:
    """Pearl+ on one served run's exchange: the run's declaration (``declared``), its linear weights (``memory``), each
    step's ``steps`` events, and every linear block computed by the device run (`kernel`). What the adapter reads of it
    (a report, `verity.protocols.resource_accounting.scheme`): ``commitment`` the run's commitment, which the composition's record hashes, ``record``
    and ``summary`` beside it, and `release`, which finishes the run once serving ends."""

    def __init__(self, ctx: Mapping[str, Any]) -> None:
        from verity.protocols.resource_accounting.scheme import settings

        self.own: Mapping[str, Any] = settings(PEARL_PLUS, ctx.get("settings") or {})
        self.gate: Any = ctx.get("gate")
        self.private: Any = ctx.get("private")
        self.ledger: Any = ctx.get("ledger")
        if ctx.get("stage") == "capture":
            raise NotImplementedError("pearl-plus at a Match capture (ctx['stage'] 'capture') records rather than commits, "
                                      "and what it records of each call is not settled (vllm, TOP-441): its device run has "
                                      "no record-only mode")
        if not self.own["coins"] and not self.own["challenger"]:
            raise ValueError("pearl-plus: neither 'coins' nor 'challenger' is among its settings (ctx['settings']): the salt is "
                             "the challenger's salt coin, opened after the weights' root, and there is no beacon")
        if self.own["challenger"] and not callable(self.gate):
            raise ValueError("pearl-plus: with a 'challenger', its launcher passes ctx['gate'], the auditor device's "
                             "challenger gate in its own words, gate(*words) -> answer (commit_coins.ChallengerGate.call)")
        if self.own["challenger"] and self.private is None:
            raise ValueError("pearl-plus: with a 'challenger', its launcher passes ctx['private'], the scheme's private "
                             "directory: the salt epoch's number must survive a restart, or a later server would open a "
                             "session on a root already used")
        if self.own["lane"] and not callable(self.ledger):
            raise ValueError("pearl-plus: with a 'lane', its launcher passes ctx['ledger'], the gate's ledger as lines "
                             "(commit_coins.ChallengerGate.ledger): each epoch's reserve joins the lane's open period")
        if not self.own["kernel"]:
            raise ValueError("pearl-plus: no 'kernel' among its settings (ctx['settings']): its run computes on the device, through "
                             "the kernel's library")
        self.declared: H.Declared | None = None
        self.weights: dict[int, H.Handle] | None = None
        self.run: Any = None
        self.session: Any = None
        self.reserved: dict[str, Any] | None = None
        self.epochs = 0
        self.phase: int | None = None
        self.phased = False
        self.first_step = 0
        self.life: dict[str, Any] | None = None
        self.phases: dict[int, dict[str, Any]] = {}
        self.busy: dict[int, int] = {}
        self.computed: set[int] = set()

    def declare(self, event: H.Declared) -> None:
        self.declared = event

    def registered(self, event: H.RegisteredValues) -> None:
        self.weights = self.keyed(event)

    def keyed(self, event: H.RegisteredValues) -> dict[int, H.Handle]:
        """The linear weights by block key (``operands``, the key `BlockDue` carries), every one committed before the
        first block."""
        values = dict(event.values)
        if not event.operands:
            raise ValueError("pearl-plus: the registered values name no weight operand (RegisteredValues.operands is "
                             "empty), so there is no linear weight to commit")
        if missing := sorted(set(event.operands) - set(values)):
            raise ValueError(f"pearl-plus: weight operands {missing} are not among the registered values")
        return {key: values[name] for name, key in event.operands.items()}

    def step(self, event: Any) -> None:
        """Its epochs, keyed on whether phase events arrive: one epoch a phase, opened at `PhaseOpened` and finished at
        `PhaseClosed`, its committed root taken from `PhaseCommitted` whenever it comes, after the next phase has opened
        too; with no phase, one epoch for the server's life, opened at the first step with tokens and finished at
        `release`. A pass ends at each `StepComputed`, once the step's blocks are all issued. A step that computed and
        ended without one is refused: under a multiprocess executor the adapter emits no `StepComputed`, and no pass
        would end."""
        if isinstance(event, H.PhaseOpened):
            if self.run is not None and not self.phased:
                raise RuntimeError(f"pearl-plus: phase {event.phase} opens at step {event.step}, after steps outside any "
                                   "phase opened its epoch for the server's life: its committer opens phase 0 before "
                                   "the first step")
            if self.phase is not None:
                raise RuntimeError(f"pearl-plus: phase {event.phase} opens while phase {self.phase} is open")
            if self.weights is None:
                raise RuntimeError(f"pearl-plus: phase {event.phase} opens before its linear weights registered")
            self.phased, self.phase, self.first_step = True, event.phase, event.step
            self.run = self.open(self.weights, self.own)
            return
        if isinstance(event, H.PhaseClosed):
            if self.phase != event.phase:
                raise RuntimeError(f"pearl-plus: phase {event.phase} closes, and the open phase is {self.phase}")
            self.phases[event.phase] = {**self.close(event.step - self.first_step + 1), "first_step": self.first_step,
                                        "last_step": event.step,
                                        "program": event.program.hex(), "root": None}
            self.phase = None
            return
        if isinstance(event, H.PhaseCommitted):
            if event.phase not in self.phases:
                raise RuntimeError(f"pearl-plus: phase {event.phase}'s root comes before the phase closed")
            if self.phases[event.phase]["root"] is not None:
                raise RuntimeError(f"pearl-plus: phase {event.phase}'s root comes twice")
            self.phases[event.phase]["root"] = event.root.hex()
            return
        if isinstance(event, H.StepBegan) and self.run is None and not self.phased and event.slots \
                and self.weights is not None:
            self.run = self.open(self.weights, self.own)
        if self.run is None:
            return
        if isinstance(event, H.StepBegan):
            if (limit := self.run.details.get("max_step_tokens")) is not None and event.slots > limit:
                raise RuntimeError(f"pearl-plus: a step of {event.slots} tokens is over one pass ({limit} tokens a "
                                   f"step); serve with max_num_batched_tokens at most {limit}")
            self.busy[event.step] = event.slots
        elif isinstance(event, H.StepComputed):
            self.run.end_pass(step=True)
            self.computed.add(event.step)
        elif isinstance(event, H.StepEnded):
            if self.busy.pop(event.step, 0) and event.step not in self.computed:
                raise RuntimeError(f"pearl-plus: step {event.step} computed and ended with no StepComputed, so its pass "
                                   "never ends (a multiprocess executor emits none); serve with a uniprocess executor")
            self.computed.discard(event.step)

    def kernel(self, due: H.BlockDue, original: Callable[[Mapping[str, H.Handle] | None], H.Handle]) -> H.Handle:
        if due.format not in READS:
            raise ValueError(f"pearl-plus: block {due.block}'s weight is {due.format}, and its kernel reads {READS}")
        if self.weights is None:
            raise RuntimeError(f"pearl-plus: block {due.block} is due before the run registered its linear weights")
        if self.run is None:
            return original(None)
        return self.run.gemm(due.key, due.inputs["x"], due.inputs["weight"], due.inputs.get("bias"))

    def commitment(self) -> Mapping[str, Any]:
        """The epoch's commitment for the server's life; with phases, each phase's by number, with its first and last
        step, its Program's digest and its committed root (None until `PhaseCommitted`)."""
        if self.phased:
            return {"phases": {str(p): {k: v for k, v in e.items() if k != "summary"} for p, e in self.phases.items()}}
        return dict(self.life["commitment"]) if self.life else dict(self.run.commits) if self.run is not None else {}

    def record(self) -> Mapping[str, Any]:
        return {"entry": OPTION, "schedule": self.own["schedule"], "tie-uniform": self.own["tie-uniform"]}

    def summary(self) -> Mapping[str, Any]:
        if self.phased:
            return {"phases": {str(p): e["summary"] for p, e in self.phases.items()}}
        return dict(self.life["summary"]) if self.life else dict(self.run.details) if self.run is not None else {}

    def release(self) -> None:
        """The open epoch finished when serving stops: the server's life's, or a phase that never closed, recorded
        with ``closed`` False."""
        if self.run is None:
            return
        if self.phase is None:
            self.life = self.close()
            return
        self.phases[self.phase] = {**self.close(), "first_step": self.first_step, "last_step": None, "program": None,
                                   "root": None, "closed": False}
        self.phase = None

    def close(self, steps: int | None = None) -> dict[str, Any]:
        """The open epoch's device run finished (its session closed with it) and freed, before another opens; what it
        committed and its details. A retained epoch's record is written beside its passes (`served`)."""
        import gc

        from verity.protocols.resource_accounting.compute.pouw.prover import serving as SV

        run, session = self.run, self.session
        self.run = self.session = None
        if session is None:
            run.finish()
        else:
            SV.finish_window(run, session)
        if getattr(run, "retain_dir", None) is not None:
            self.served(run, steps)
        out = {"commitment": dict(run.commits), "summary": dict(run.details), **({"reserved": self.reserved}
                                                                                   if self.reserved else {})}
        self.reserved = None
        del run
        gc.collect()
        return out

    def reserve(self, k: int) -> dict[str, Any]:
        """Salt epoch ``k``'s session, reserved before it opens, since a reserve opens once: the first epoch's the
        ``session`` setting, if given; every other one reserved at the gate under ``lane`` as ``peer``, joining the
        lane's open period (`serving.reserve_salt`, epoch ``k``). The reserve answer and its period go in the epoch's
        record."""
        from verity.protocols.resource_accounting.compute.pouw.prover import serving as SV

        if self.epochs == 1 and self.own["session"]:
            reserved, period = dict(self.own["session"]), None
        elif self.own["lane"]:
            reserved, period = SV.reserve_salt(self.gate, self.ledger, self.own["lane"], self.own["peer"], k)
        else:
            raise ValueError(f"pearl-plus: salt epoch {k} needs a session of its own, and the 'session' setting was the "
                             "first epoch's (a reserve opens once): give 'lane' and 'peer' to reserve one an epoch")
        self.reserved = {"session": reserved, "period": period}
        return reserved

    def served(self, run: Any, steps: int | None) -> None:
        """A retained epoch's ``served.json`` (`serving.SERVED_SCHEMA`, as ``served_passes.py retain --window-record`` and
        ``verify --tie`` read it) and ``coins.json`` (its coin commitment) in its retained directory: its passes with
        calls (`serving.served_passes`), its coins (`serving.coins_of`), and ``error`` when they aren't one round a step
        of its phase's ``steps``, each with one call a linear and a manifest."""
        import json

        from verity.protocols.resource_accounting.compute.pouw.prover import serving as SV

        d, linears = Path(run.retain_dir), len(self.weights or {})
        passes = [p for p in SV.served_passes(run.details["passes"]) if p["units"]]
        bad = ([] if steps is None or len(passes) == steps else
               [f"its passes with calls are {len(passes)} and its phase's steps {steps}, so round s isn't step s"])
        bad += [f"pass {p['pass']} has {p['units']} calls, not one a linear ({linears})" for p in passes
                if p["units"] != linears][:4]
        bad += [f"pass {p['pass']} retained no manifest" for p in passes if not p.get("manifest")][:4]
        held = SimpleNamespace(commitment=lambda: run.commits, info=lambda: run.details)
        doc = {"schema": SV.SERVED_SCHEMA, "entry": OPTION, "window": self.phase, "retained": True, "commit_steps": steps,
               "passes": passes, "passes_with_calls": len(passes),
               "coins": SV.coins_of(held, (self.reserved or {}).get("session")), "ok": not bad,
               "error": "; ".join(bad) or None}
        (d / "coins.json").write_text(json.dumps(run.commits["coins"]) + "\n")
        (d / "served.json").write_text(json.dumps(doc, indent=1) + "\n")

    def open(self, weights: Mapping[int, H.Handle], own: Mapping[str, Any]) -> Any:
        """The device run over ``weights``, under Pearl+'s own settings ``own`` (``kernel``, ``cubin``, ``schedule``,
        ``transcript_dir``, and ``coins`` or ``challenger``, whose session `reserve` gives and the launcher's gate
        opens): their B rows salted with fresh bytes each epoch (the served checkpoint is not registered), their
        roots committed before any salt, the salt coin opened on their root (with ``challenger``, one
        session an epoch, closed at its `close`, on an epoch number first spent in ``ctx["private"]``, `spend_epoch`; else
        the in-process coins of this process's epoch count), and the run's pass limit, retained passes and ``registration``,
        what it claims of the weights it serves."""
        from verity.protocols.resource_accounting.compute.pouw.prover import serving as SV
        from verity.protocols.resource_accounting.compute.pouw.prover.serving.tensors import pearl_c_device

        device, name, weights = pearl_c_device(), OPTION.partition(":")[2], dict(weights)
        self.epochs += 1
        if bad := SV.pearl_c_refusals(name, {w: tuple(t.shape) for w, t in weights.items()}):
            raise ValueError(f"pearl-plus: its weights are refused before any commitment: {'; '.join(bad)}")
        kernel = SV.PearlCKernel.load(own["kernel"], own["cubin"] or None)
        salts = {w: b"".join(SV.fresh_salts(t.shape[0])) for w, t in weights.items()}
        roots = device.weight_roots(kernel, weights, name, weight_salts=salts)
        k, carried = None, None
        if own["challenger"]:
            k = spend_epoch(self.private)
            given = SimpleNamespace(challenger=own["challenger"], session=self.reserve(k), gate=SV.gate_of(self.gate))
            self.session, coins, opening, _, carried, k = SV.salt_session(OPTION, given, SV.weights_root(roots), True,
                                                                          epoch=k)
        else:
            held = SV.in_process_coins(self.epochs - 1)
            coins, opening = held.commitment, held.open("salt", SV.weights_root(roots))
        with SV.closed_on_error(self.session):
            scheme = SV.pearl_c_scheme(name)
            run = device.PearlCDeviceRun(scheme, OPTION, kernel, weights, roots, coins, opening, pass_index=k,
                                         schedule=own["schedule"], weight_salts=salts)
        if self.session is not None:
            run.commits.update(coin_source=SV.H1_CHALLENGER, session=carried)
        run.commits["registration"] = registration(scheme, SV.DomainError)
        run.details["max_step_tokens"] = device.step_tokens(run, SV.PEARL_C_CALL_CAPACITY)
        if own.get("transcript_dir"):
            run.retain_dir = Path(own["transcript_dir"]) / (f"w{self.phase}" if self.phased else "")
            for d in {Path(own["transcript_dir"]), run.retain_dir}:
                d.mkdir(mode=0o700, parents=True, exist_ok=True)
                d.chmod(0o700)
            if not run.cur.matmuls:
                run.begin()
        return run


def served_record(d: str | os.PathLike[str]) -> dict[str, Any]:
    """A served run's report as vLLM's scheme plugin writes it (``out/schemes/pearl-plus/``: ``configuration.json``,
    ``summary.json``, ``commitment.json``), read as the served verdicts' Pearl+ piece reads ``served.json``: its
    ``arm``, ``passes`` (`pearl_c.window.served_passes`), ``coins`` (`pearl_c.window.coins_of`, the gate's reserve answer
    its ``session`` setting), pass limit and pass ends. What only the load's driver measures (``generate_s``, the
    requests and their tokens, ``ok``) is not in it, and a pass's ``manifest`` is there only where it was retained."""
    import json

    from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c import window as PW

    configuration, summary, commitment = (json.loads((Path(d) / f"{n}.json").read_text())
                                          for n in ("configuration", "summary", "commitment"))
    own = configuration.get("settings") or {}
    held = SimpleNamespace(commitment=lambda: commitment, info=lambda: summary)
    return {"arm": configuration["record"]["entry"], "retained": bool(own.get("transcript_dir")),
            "max_step_tokens": summary.get("max_step_tokens"), "pass_ends": summary.get("pass_ends"),
            "passes": PW.served_passes(summary.get("passes") or []), "coins": PW.coins_of(held, own.get("session") or None)}


def attach(exchange: Any, ctx: Any) -> Attached:
    on = Attached(ctx)
    exchange.subscribe(PEARL_PLUS.name, "declared", on.declare)
    exchange.subscribe(PEARL_PLUS.name, "memory", on.registered)
    exchange.subscribe(PEARL_PLUS.name, "steps", on.step)
    exchange.request(PEARL_PLUS.name, H.ComputeBlockWith(on.kernel))
    return on


PEARL_PLUS = register(Scheme(
    name="pearl-plus",
    messages=workload.MESSAGES,
    protocol="verity/pouw/pearl-plus/v1",
    resource="compute",
    blocks=(Block("PearlPlusTile_v1", tile_macs), Block("PearlCTileRead_v1", tile_macs)),
    assumptions=("Pouw.PearlC.Assumptions.TTOutTilePearlCH100Rev1",),
    log=(Order("weightsRoot", "pouw/salt"), Order("commitment", "pouw/draw")),
    charges=(Charge("inert-work", "rho"), Charge("sample", "epsilon"), Charge("over-bound-tiles", "over_cap_draws"),
             Charge("excluded-rows", "give_up")),
    delta_scheme=Term("Pouw.PearlC.δPearlC", ("q", "N"), lambda q, N: Fraction(q + N, 2**128)),
    over="399 × the MACs credited: each passed tile's credited cells times the record's kept width |K|",
    under="400 × the MACs spent: the prover's cost in the cost model, priced at a credited MAC's honest cost (TT_OUT's T "
          "at g₀ = 1/400)",
    settings=SETTINGS,
    check_settings=check_settings,
    option=OPTION,
    hooks=("blocks", "declared", "memory", "steps"),
    provides=("cut",),
    requires={"one-labelling": "its own pass manifests, tied to the window record's pearl-a/, pearl-y/ and pearl-b/ values",
              "log-order": "its own coin sessions: the salt's opened on the weights' root, the sample's on the run's "
                           "commitment",
              "one-design": "its own tile sampling by ticket weight"},
    requests=requests,
    attach=attach,
))
