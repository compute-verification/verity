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

    # Counted sets: a class whose counted products are another set than `κ`'s names its `κ` here (`classes.counted`)
    #: `κ` of the forward products alone, each (weight, token) pair once (Kimi K2: `α₀ = 0.147`, so 0.294)
    kappa_forward: float | None = None
    #: `κ` of the weight-gradient products alone (the accumulators of E1's Corollary 4a)
    kappa_wgrad: float | None = None

    # Depth (`MulBlocks.Upstream`, `ActivationChainedAt`) and layers
    #: `L`: layers a token's computation passes through
    layers: int | None = None
    #: distinct updated weights one token's computation uses in one layer's MLP, the products between the layer's
    #: attention output and the next layer's input, the least over layers (`Upstream`'s `D` per layer, the attention
    #: left out, which only lowers it; Kimi K2: 4·10⁸)
    layer_weights: float | None = None
    #: bits of one layer's whole update: a block that reads it computes that layer's products from token ids (E1)
    layer_bits: float | None = None
    #: the block size up to which the `Threading` certificate is checked (`ReferenceThreaded`: `S ≤ 2·10⁷`)
    thread_k: float | None = None
    #: `n`: an accumulation entry's width (`Cube.Wide`; kv_b's 512 in Kimi K2)
    entry_width: float | None = None

    # Prices (`ParameterHardPricedAt`)
    #: `Σ_b p_b / |𝓜*|`: the chained blocks' mean weight price, bits
    price: float | None = None
    #: `t`: the price ceiling, `p i ≤ t` for every weight
    price_max: float | None = None

    # E1's drafts
    #: `γ`: the splitting margin of premise `H_γ^t` (Proposition 3′)
    gamma: float | None = None
    #: `β_v`: fresh bits per value a block reads at the output map (Proposition 6b)
    beta_v: float | None = None
    #: `d`: an output map's width, the products one value read reaches (Proposition 6b)
    map_width: float | None = None
    #: `d_in`: the narrowest output map's input width, which sets Proposition 6b's cap `k*`
    map_width_min: float | None = None
    #: `h_eff`: bits per accumulated partial sum a block must hold (Corollary 4a)
    h_eff: float | None = None
    #: `b_ps`: bits per weight-gradient accumulator a block reads (Corollary 4a)
    b_ps: float | None = None
    #: `T_step`: tokens per optimizer step (Corollary 4a's `n_E = M / T_step`)
    step_tokens: float | None = None

    # Constructions (`attacks`)
    #: `β_x` at the margin: fresh bits per extra activation occurrence a tile reads (the tile, E1)
    beta_x_marginal: float | None = None
    #: bits per partial sum a tile hands on
    psum_bits: float | None = None
    #: bits of state one token's computation carries between blocks (`U_token`, `benchmarks/nci`): its updated weights
    #: and, inside an optimizer step, the gradient accumulators a block hands on
    token_state_bits: float | None = None
    #: the reference circuit's FLOPs per token
    flops_per_token: float | None = None
    #: bits per token crossing a shard boundary (crossings times the cut's width)
    cut_bits: float | None = None
    #: novel data bits per token
    token_bits: float | None = None
    #: FLOPs to decode the whole update's description inside a block (0 if stored plainly)
    decode_flops: float | None = None
    #: whether the updated state splits across shards (False: a description such as seeds, read whole)
    state_splits: bool | None = None
    #: bits of state the largest contiguous slice of a token's computation holds whose weights every token uses (all of
    #: `token_state_bits` for a dense model; in a MoE, the run from one routed layer's output to the next one's input):
    #: a slice of many tokens holds no more than its own share, and only its two ends cross
    shared_state_bits: float | None = None
    #: bits a tile reads per updated weight (an achievable code: the stored width, or a measured rate)
    tile_weight_bits: float | None = None
    #: a tile's most input coordinates (rows of `Wᵀ`) and output slots that every token of the tile shares
    tile_inputs: float | None = None
    tile_outputs: float | None = None
