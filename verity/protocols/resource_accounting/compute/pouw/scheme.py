"""Pearl+ as a resource scheme (`verity.protocols.resource_accounting.scheme`), registered as ``pearl-plus``.

Its verifier is `verity/pouw/pearl-plus/v1` (`verifier/lean/pearlplus/PouwPearlPlus/Protocol.lean`). A tile that passes
the tile rule costs its credited cells at the record's kept width |K| (`Credit.lean`'s `creditAtAssumption`), so a
block's cost reads the record's kept width (``K_KEPT``, `UnitRecord.k_kept`) beside the tile's TM × TN, not the
Definition's K, which is the unit's padded width. The cost is of a tile whose every cell is credited: filler, excluded and
given-up rows credit none (`UnitRecord.credited`).

On a served run (`attach`) it computes every linear block with its device run (`verity.hooks.ComputeBlockWith`), opened
once the run registers its linear weights (``memory``), and ends the run's pass at each step's end (``steps``); its one
setting, ``tie-uniform``, lets a uniform sampling rate stand under the tie to its tables, with the weaker tie its record
states.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from verity import hooks as H
from verity.protocols.resource_accounting.scheme import Block, Charge, Order, Scheme, register

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

    def __init__(self) -> None:
        self.declared: H.Declared | None = None
        self.run: Any = None

    def declare(self, event: H.Declared) -> None:
        self.declared = event

    def registered(self, event: H.RegisteredValues) -> None:
        self.run = self.open({n: v for n, v in event.values if n in event.linear})

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

    def open(self, weights: Mapping[str, H.Handle]) -> Any:
        raise NotImplementedError(
            "pearl-plus: opening its device run on the exchange needs the device run handed in to verity "
            "(PearlCDeviceRun and its weight roots are verity_torch's; compute-accounting builds the hand-in), and from "
            "vllm each linear weight's block key in RegisteredValues (the salt is opened on every weight's root before "
            "the first block), Pearl+'s settings (kernel, cubin, coins or challenger, schedule, transcript dir) in "
            "Declared.settings, and a request that hands its commitment and release to the run's record (hooks.Reports)")


def attach(exchange: Any, ctx: Any) -> None:
    on = Attached()
    exchange.subscribe(PEARL_PLUS.name, "declared", on.declare)
    exchange.subscribe(PEARL_PLUS.name, "memory", on.registered)
    exchange.subscribe(PEARL_PLUS.name, "steps", on.step)
    exchange.request(PEARL_PLUS.name, H.ComputeBlockWith(on.kernel))


PEARL_PLUS = register(Scheme(
    name="pearl-plus",
    protocol="verity/pouw/pearl-plus/v1",
    resource="compute",
    blocks=(Block("PearlPlusTile_v1", tile_macs), Block("PearlCTileRead_v1", tile_macs)),
    assumptions=("Pouw.PearlC.Assumptions.TTOutTilePearlCH100Rev1",),
    log=(Order("weightsRoot", "pouw/salt"), Order("commitment", "pouw/draw")),
    charges=(Charge("inert-work", "rho"), Charge("sample", "epsilon"), Charge("over-bound-tiles", "over_cap_draws"),
             Charge("excluded-rows", "give_up")),
    settings={"tie-uniform": False},
    option="pouw:pearl-c-h100-v1-h3",
    hooks=("blocks", "declared", "memory", "steps"),
    requires={"one-labelling": "its own pass manifests, tied to the window record's pearl-a/, pearl-y/ and pearl-b/ values",
              "log-order": "its own coin sessions: the salt's opened on the weights' root, the sample's on the run's "
                           "commitment",
              "one-design": "its own tile sampling by ticket weight"},
    requests=requests,
    attach=attach,
))
