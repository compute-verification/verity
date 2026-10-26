"""What the arguments read about one training algorithm on one model: the interface between `families` and `classes`.

A family (pretraining, LoRA, DiLoCo, ...) at an instance (a model and a run) gives one `Profile`. A bound or an attack
reads the fields its theorem or construction names, and puts the algorithm outside its class when one is `None`
(unknown) or its premise fails. Field names follow the Lean statements' symbols, given beside each.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    #: `κ = 2 |M| / FLOPs`: the counted products' share of the algorithm's FLOPs (`α₀ = κ/2` per FLOP)
    kappa: float
    #: bits of the state the algorithm trains: a block that can read it whole trains unthrottled
    state_bits: float
    #: compute to reach the run's target, in multiples of the reference algorithm's (`m`; `math.inf`: never reaches it)
    efficiency: float = 1.0

    # Weights and their updates
    #: `b`: bits a block pays per parameter whose update it completes (`UpdateHardness`)
    b: float | None = None
    #: `b_w`: bits a small subcircuit pays per weight it uses (`WeightsIncompressible`)
    b_w: float | None = None
    #: `c_w`, `c_a`: the chained costs, per weight and per activation (`CoverTrainingChained`, `CoveringBoundTwoCosts`)
    c_w: float | None = None
    c_a: float | None = None

    # Activations and values
    #: `b_a`: bits per novel activation read (`WeightsIncompressible`'s activation side, `ThreadedThrottled`)
    b_a: float | None = None
    #: `f`: bits per value read (`ValueReadHardness`, `ActivationReads`)
    f: float | None = None
    #: `a`: an accumulation entry's width (`ActivationReads`, `psRate`)
    width: float | None = None
    #: `β_x`: fresh bits a block pays per counted activation it uses, at a rate that survives replay (the draft's
    #: Proposition 3; measured on exact blocks)
    beta_x: float | None = None

    # The keyed token structure (`Keyed`)
    #: `d`: legs of a token's accumulation chain
    legs: int | None = None
    #: `aOut`: counted products per activation value
    fanout: float | None = None
    #: `c`: counted products per parameter per token (1 forward, 2 with Dgrad on the forward's tokens)
    per_param: float | None = None
    #: `V`: vocabulary (embedding columns)
    vocab: int | None = None
    #: `bcol`: bits per embedding column entry (`ColumnsIncompressible`)
    bcol: float | None = None

    # Threads (`Threading`, `ThreadedThrottled`)
    #: `d`: novel values of a thread a small subcircuit may read before `e` stops binding
    thread_d: int | None = None
    #: `e`: counted multiplications per value read from a thread
    thread_e: float | None = None
    #: `c`: the reuse constant in `Λ = max(e, c S / (4 d b_w))`
    thread_c: float | None = None
