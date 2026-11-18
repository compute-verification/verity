"""Pearl+ as a resource scheme (`verity.protocols.resource_accounting.scheme`), registered as ``pearl-plus``.

Its verifier is `verity/pouw/pearl-plus/v1` (`verifier/lean/pearlplus/PouwPearlPlus/Protocol.lean`). A tile that passes
the tile rule costs its credited cells at the record's kept width |K| (`Credit.lean`'s `creditAtAssumption`), so a
block's cost reads the record's kept width (``K_KEPT``, `UnitRecord.k_kept`) beside the tile's TM × TN, not the
Definition's K, which is the unit's padded width. The cost is of a tile whose every cell is credited: filler, excluded and
given-up rows credit none (`UnitRecord.credited`).

On a served run (`attach`) it computes every linear block with its device run (`verity.hooks.ComputeBlockWith`), the
workload's `PearlCDeviceRun` (`serving.tensors.pearl_c_device`), opened once the run registers its weight operands
(``memory``), and ends the run's pass once each step's blocks are issued (``steps``, `StepComputed`). Its settings are
`SETTINGS`; ``tie-uniform`` lets a uniform sampling rate stand under the tie to its tables, with the weaker tie its
record states.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity import hooks as H
from verity.protocols.resource_accounting.compute.pouw.prover import workload
from verity.protocol import Channel, Role
from verity.protocols.resource_accounting.scheme import VERIFIER, Block, Charge, Lacks, Order, Scheme, Term, register

OPTION = "pouw:pearl-c-h100-v1-h3"
#: the weight formats its kernel reads, quantizing them itself; a quantized (FP8) weight is refused, not misread
READS = ("bf16", "fp16", "fp32")


def tile_macs(shape: Mapping[str, int]) -> int:
    return shape["TM"] * shape["TN"] * shape["K_KEPT"]


#: Its settings (`Scheme.settings`), each a default of its type: ``kernel`` the kernel's library (required on a served
#: run), ``cubin`` its cubin, ``schedule`` the schedule each pass runs under (`serving.SCHEDULES`),
#: ``transcript_dir`` where passes are retained, the salt's source, ``coins`` (``in-process``: a challenger held in this
#: process, crediting nothing) or ``challenger`` (the auditor device's, HOST:PORT) with ``session`` (the gate's reserve
#: answer, which the launcher's ``ctx["gate"]`` opens), and ``tie-uniform``
SETTINGS: dict[str, Any] = {"tie-uniform": False, "kernel": "", "cubin": "", "schedule": "serial", "transcript_dir": "",
                            "coins": "", "challenger": "", "session": {}}
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
    if bool(own["challenger"]) != bool(own["session"]):
        return "challenger and session go together: the session is the gate's reserve answer for that challenger"
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


def requests(ctx: Any) -> tuple[H.RequireProperty, ...]:
    return (H.RequireProperty("eager", True, "its kernel at each linear block is not graph-capturable"),
            H.RequireProperty("world", 1, "its passes and transcript are per rank, which is not wired"))


class Attached:
    """Pearl+ on one served run's exchange: the run's declaration (``declared``), its linear weights (``memory``), each
    step's ``steps`` events, and every linear block computed by the device run (`kernel`). What the adapter reads of it
    (`verity.hooks.Commits`): ``commitment`` the run's commitment, which the composition's record hashes, ``record``
    and ``summary`` beside it, and `release`, which finishes the run once serving ends."""

    def __init__(self, ctx: Mapping[str, Any]) -> None:
        from verity.protocols.resource_accounting.scheme import settings

        self.own: Mapping[str, Any] = settings(PEARL_PLUS, ctx.get("settings") or {})
        self.gate: Any = ctx.get("gate")
        if not self.own["coins"] and not self.own["challenger"]:
            raise ValueError("pearl-plus: neither 'coins' nor 'challenger' is among its settings (ctx['settings']): the salt is "
                             "the challenger's salt coin, opened after the weights' root, and there is no beacon")
        if self.own["challenger"] and not callable(self.gate):
            raise ValueError("pearl-plus: with a 'challenger', its launcher passes ctx['gate'], gate(session, root): the "
                             "gate's open of the reserved session on the salt epoch's root (serving.h1.window.Gate)")
        if not self.own["kernel"]:
            raise ValueError("pearl-plus: no 'kernel' among its settings (ctx['settings']): its run computes on the device, through "
                             "the kernel's library")
        self.declared: H.Declared | None = None
        self.run: Any = None
        self.session: Any = None
        self.busy: dict[int, int] = {}
        self.computed: set[int] = set()

    def declare(self, event: H.Declared) -> None:
        self.declared = event

    def registered(self, event: H.RegisteredValues) -> None:
        self.run = self.open(self.keyed(event), self.own)

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
        """A pass ends at each `StepComputed`, once the step's blocks are all issued. A step that computed and ended
        without one is refused: under a multiprocess executor the adapter emits no `StepComputed`, and no pass would
        end."""
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
        if self.run is None:
            raise RuntimeError(f"pearl-plus: block {due.block} is due before the run registered its linear weights")
        return self.run.gemm(due.key, due.inputs["x"], due.inputs["weight"], due.inputs.get("bias"))

    def commitment(self) -> Mapping[str, Any]:
        return dict(self.run.commits) if self.run is not None else {}

    def record(self) -> Mapping[str, Any]:
        return {"entry": OPTION, "schedule": self.own["schedule"], "tie-uniform": self.own["tie-uniform"]}

    def summary(self) -> Mapping[str, Any]:
        return dict(self.run.details) if self.run is not None else {}

    def release(self) -> None:
        if self.run is None:
            return
        if self.session is None:
            self.run.finish()
        else:
            from verity.protocols.resource_accounting.compute.pouw.prover import serving as SV
            SV.finish_window(self.run, self.session)

    def open(self, weights: Mapping[int, H.Handle], own: Mapping[str, Any]) -> Any:
        """The device run over ``weights``, under Pearl+'s own settings ``own`` (``kernel``, ``cubin``, ``schedule``,
        ``transcript_dir``, and ``coins`` or ``challenger`` with its ``session``, which the launcher's gate opens):
        their roots committed before any salt, the salt coin opened on their root (with ``challenger``, one session
        for the server's life, closed at `release`; else the in-process coins), and the run's pass limit, retained
        passes and ``registration``, what it claims of the weights it serves."""
        from verity.protocols.resource_accounting.compute.pouw.prover import serving as SV
        from verity.protocols.resource_accounting.compute.pouw.prover.serving.tensors import pearl_c_device

        device, name, weights = pearl_c_device(), OPTION.partition(":")[2], dict(weights)
        if bad := SV.pearl_c_refusals(name, {w: tuple(t.shape) for w, t in weights.items()}):
            raise ValueError(f"pearl-plus: its weights are refused before any commitment: {'; '.join(bad)}")
        kernel = SV.PearlCKernel.load(own["kernel"], own["cubin"] or None)
        roots = device.weight_roots(kernel, weights, name)
        k, carried = None, None
        if own["challenger"]:
            given = SimpleNamespace(challenger=own["challenger"], session=dict(own["session"]), gate=self.gate)
            self.session, coins, opening, _, carried, k = SV.salt_session(OPTION, given, SV.weights_root(roots), True)
        else:
            held = SV.in_process_coins(0)
            coins, opening = held.commitment, held.open("salt", SV.weights_root(roots))
        with SV.closed_on_error(self.session):
            scheme = SV.pearl_c_scheme(name)
            run = device.PearlCDeviceRun(scheme, OPTION, kernel, weights, roots, coins, opening, pass_index=k,
                                         schedule=own["schedule"])
        if self.session is not None:
            run.commits.update(coin_source=SV.H1_CHALLENGER, session=carried)
        run.commits["registration"] = registration(scheme, SV.DomainError)
        run.details["max_step_tokens"] = device.step_tokens(run, SV.PEARL_C_CALL_CAPACITY)
        if own.get("transcript_dir"):
            run.retain_dir = Path(own["transcript_dir"])
            if not run.cur.matmuls:
                run.begin()
        return run


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
    requires={"one-labelling": "its own pass manifests, tied to the window record's pearl-a/, pearl-y/ and pearl-b/ values",
              "log-order": "its own coin sessions: the salt's opened on the weights' root, the sample's on the run's "
                           "commitment",
              "one-design": "its own tile sampling by ticket weight"},
    requests=requests,
    attach=attach,
    parties=(Role("challenger", "auditor", needs=("host",)), Role("workers", "developer", needs=("serve",)), VERIFIER),
    channels=(Channel("challenger", "workers", ("PearlSalt",), carries_coins=True),),
    colocated=(),
    needs=frozenset({"windows"}),
    depends=(),
    record="work",
    symbol=Lacks("the record's work row is H1's (W_cert, in work); Pearl+'s certified quantity is its owner's to name"),
    record_unit=Lacks("the record's work row is H1's (W_cert, in work); Pearl+'s certified quantity is its owner's to "
                      "name"),
    title=Lacks("the record's work row is titled for H1 (PoUW); Pearl+'s is its owner's to name"),
    record_owner="compute-accounting",
))
