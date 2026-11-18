"""Pearl+ as a resource scheme (`verity.protocols.resource_accounting.scheme`), registered as ``pearl-plus``.

Its verifier is `verity/pouw/pearl-plus/v1` (`verifier/lean/pearlplus/PouwPearlPlus/Protocol.lean`). A tile that passes
the tile rule costs its credited cells at the record's kept width |K| (`Credit.lean`'s `creditAtAssumption`), so a
block's cost reads the record's kept width (``K_KEPT``, `UnitRecord.k_kept`) beside the tile's TM × TN, not the
Definition's K, which is the unit's padded width. The cost is of a tile whose every cell is credited: filler, excluded and
given-up rows credit none (`UnitRecord.credited`).

On a served run (`attach`) it computes every linear block with its device run (`verity.hooks.ComputeBlockWith`), the
workload's `PearlCDeviceRun` (`serving.tensors.pearl_c_device`), opened once the run registers its linear weights
(``memory``), and ends the run's pass at each step's end (``steps``); its one setting, ``tie-uniform``, lets a uniform
sampling rate stand under the tie to its tables, with the weaker tie its record states.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity import hooks as H
from verity.protocols.resource_accounting.compute.pouw.prover import workload
from verity.protocols.resource_accounting.scheme import Block, Charge, Order, Scheme, register

OPTION = "pouw:pearl-c-h100-v1-h3"
#: the weight formats its kernel reads, quantizing them itself; a quantized (FP8) weight is refused, not misread
READS = ("bf16", "fp16", "fp32")


def tile_macs(shape: Mapping[str, int]) -> int:
    return shape["TM"] * shape["TN"] * shape["K_KEPT"]


def requests(ctx: Any) -> tuple[H.RequireProperty, ...]:
    return (H.RequireProperty("eager", True, "its kernel at each linear block is not graph-capturable"),
            H.RequireProperty("world", 1, "its passes and transcript are per rank, which is not wired"),
            H.RequireProperty("async_steps", False, "it ends a pass at each StepEnded, which under async steps comes after "
                              "the next step's forward"))


class Attached:
    """Pearl+ on one served run's exchange: the run's declaration (``declared``), its linear weights (``memory``), each
    step's end (``steps``), and every linear block computed by the device run (`kernel`)."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.declared: H.Declared | None = None
        self.run: Any = None
        self.session: Any = None

    def declare(self, event: H.Declared) -> None:
        self.declared = event

    def registered(self, event: H.RegisteredValues) -> None:
        self.run = self.open(*self.inputs(event))

    def inputs(self, event: H.RegisteredValues) -> tuple[dict[int, H.Handle], Mapping[str, Any]]:
        """The linear weights by block key, and Pearl+'s own settings: ``kernel``, ``cubin``, ``schedule``,
        ``transcript_dir``, and ``coins`` or ``challenger`` with its ``session`` and ``gate``."""
        raise NotImplementedError(
            "pearl-plus: vllm builds the two inputs its run opens on: each linear weight's block key in RegisteredValues "
            "(the key BlockDue carries), and Pearl+'s own settings in attach's ctx (kernel, cubin, schedule, "
            "transcript_dir, and coins or challenger with session and gate)")

    def step(self, event: Any) -> None:
        if self.run is None:
            return
        if isinstance(event, H.StepBegan) and (limit := self.run.details.get("max_step_tokens")) is not None \
                and event.slots > limit:
            raise RuntimeError(f"pearl-plus: a step of {event.slots} tokens is over one pass ({limit} tokens a step); "
                               f"serve with max_num_batched_tokens at most {limit}")
        if isinstance(event, H.StepEnded):
            self.run.end_pass(step=True)

    def kernel(self, due: H.BlockDue, original: Callable[[Mapping[str, H.Handle] | None], H.Handle]) -> H.Handle:
        if due.format not in READS:
            raise ValueError(f"pearl-plus: block {due.block}'s weight is {due.format}, and its kernel reads {READS}")
        if self.run is None:
            raise RuntimeError(f"pearl-plus: block {due.block} is due before the run registered its linear weights")
        return self.run.gemm(due.key, due.inputs["x"], due.inputs["weight"], due.inputs.get("bias"))

    def open(self, weights: Mapping[int, H.Handle], own: Mapping[str, Any]) -> Any:
        """The device run over ``weights``: their roots committed before any salt, the salt coin opened on their root
        (the challenger's session with ``challenger``, else ``coins``), and the run's pass limit and retained passes."""
        from verity.protocols.resource_accounting.compute.pouw.prover import serving as SV
        from verity.protocols.resource_accounting.compute.pouw.prover.serving.tensors import pearl_c_device

        device, name, weights = pearl_c_device(), OPTION.partition(":")[2], dict(weights)
        kernel = SV.PearlCKernel.load(own["kernel"], own.get("cubin"))
        roots = device.weight_roots(kernel, weights, name)
        k, carried = None, None
        if own.get("challenger") is not None:
            self.session, coins, opening, _, carried, k = SV.salt_session(OPTION, SimpleNamespace(**own),
                                                                          SV.weights_root(roots), True)
        else:
            coins, opening = own["coins"].commitment, own["coins"].open("salt", SV.weights_root(roots))
        with SV.closed_on_error(self.session):
            run = device.PearlCDeviceRun(SV.pearl_c_scheme(name), OPTION, kernel, weights, roots, coins, opening,
                                         pass_index=k, **({"schedule": own["schedule"]} if "schedule" in own else {}))
        if self.session is not None:
            run.commits.update(coin_source=SV.H1_CHALLENGER, session=carried)
        run.details["max_step_tokens"] = device.step_tokens(run, SV.PEARL_C_CALL_CAPACITY)
        if own.get("transcript_dir"):
            run.retain_dir = Path(own["transcript_dir"])
            if not run.cur.matmuls:
                run.begin()
        return run


def attach(exchange: Any, ctx: Any) -> None:
    on = Attached(ctx)
    exchange.subscribe(PEARL_PLUS.name, "declared", on.declare)
    exchange.subscribe(PEARL_PLUS.name, "memory", on.registered)
    exchange.subscribe(PEARL_PLUS.name, "steps", on.step)
    exchange.request(PEARL_PLUS.name, H.ComputeBlockWith(on.kernel))


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
    settings={"tie-uniform": False},
    option=OPTION,
    hooks=("blocks", "declared", "memory", "steps"),
    requires={"one-labelling": "its own pass manifests, tied to the window record's pearl-a/, pearl-y/ and pearl-b/ values",
              "log-order": "its own coin sessions: the salt's opened on the weights' root, the sample's on the run's "
                           "commitment",
              "one-design": "its own tile sampling by ticket weight"},
    requests=requests,
    attach=attach,
))
