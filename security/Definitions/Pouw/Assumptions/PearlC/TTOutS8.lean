module

public import Definitions.Pouw.Assumptions.PearlC.TTOutW1

/-! Pearl-C's TT_OUT per tile on the H100 split into the device model `W1CompleteH100`, falsified on the H200, and the conjecture `TTOutTilePearlCH100Rev1S8`, both open premises and not named assumptions. -/

@[expose] public section

namespace Pouw.PearlC

variable {Q R S : Type}

/-- **`CM'` dominates `CM` within every bound on the tiles `TR`**: for every in-domain workload, every cost `T` and
query count `q`, every adversary of `CM` within `T` and `q` has an adversary of `CM'` within `T` and `q` whose good
tiles' credit exceeds every threshold `c` with at least the probability, over the oracle and the salt, that the
adversary's does. TT_OUT per tile bounds exactly that probability at `c = T/(1 − g₀)`, so TT_OUT per tile at `CM'`
gives it at `CM` (`Pouw.PearlC.ttOutTile_of_dominatedWithin`). -/
def CostModel.DominatedWithin [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : Layout → Prop) (CM CM' : CostModel Q R S) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (T : ℝ) (q : ℕ) (A : Adversary CM), A.Bounded U T q →
    ∃ A' : Adversary CM', A'.Bounded U T q ∧ ∀ c : ℝ,
      pr (fun x : (Q → R) × S => c < ∑ g ∈ goodTiles P TR U x.1 x.2 (A.trace U x.1 x.2),
          TR.credit x.1 x.2 U g (actOf (A.trace U x.1 x.2) ((TR.tiling U.layout).unit g))) ≤
        pr (fun x : (Q → R) × S => c < ∑ g ∈ goodTiles P TR U x.1 x.2 (A'.trace U x.1 x.2),
          TR.credit x.1 x.2 U g (actOf (A'.trace U x.1 x.2) ((TR.tiling U.layout).unit g)))

/-- **The Hopper INT8 setting at the adversary's best case**: `W1Device.s8H100Tensor` with a SHA-512 compression and a
first fetch free. Neither price is on a data sheet (the H100 whitepaper's Tables 1 and 3 and the H200 data sheet give
tensor-core and FP32 rates and clocks, not a compression's or a word's first read), and the audited hardware and its
vendor are untrusted, so neither is taken at a measured rate. W1's cost only grows with each: at any `sha, fetch ≥ 0`,
`s8H100Tensor sha fetch` is simulated by this setting (`Pouw.PearlC.W1.w1Priced_s8H100Tensor_simulatedBy_best`), so a
bound here holds at every price. -/
def W1.W1Device.s8H100Best : W1.W1Device := W1.W1Device.s8H100Tensor 0 0

end Pouw.PearlC

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- `w1-complete-h100` (open premise, falsified on the H200 by the timed loop of FP32 beside the tensor pipe; falsified
by a program on an H100 or an H200, within a cost and a query count, that earns more good-tile credit at the H100 record
per second at the tensor clock, rev1, with higher probability than every W1 program on `W1Device.s8H100Best` within
that cost and count). **W1 is complete for the H100's and the H200's credited work**: the card's cost model `CM` reaches `H` only through counted queries, and W1 on
`W1Device.s8H100Best` with the salt's bytes `salt` dominates it on rev1's audit tiles at the H100 record per second at
the tensor clock within every bound (`CostModel.DominatedWithin`, at `pearlCProtocolDevRev1 devH100Tensor sem ρ`,
`pearlCTilesDevRev1 devH100Tensor sem ρ` and `pearlCDomainDev devH100Tensor`). Its prices are the data sheets' per
second at the tensor clock: NVIDIA's H100 Tensor Core GPU Architecture whitepaper V1.04, Tables 1 and 3, and the H200
Tensor Core GPU data sheet (`W1.W1Prices.h100Tensor`, the SM pipes at `h100TensorMHz / h100SmMHz` of their per-clock
prices), with a compression and a first fetch at 0. -/
def W1CompleteH100 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (salt : S → List ℕ) (sem : PearlCSem Q R S)
    (ρ : ℚ) (CM : CostModel Q R S) : Prop :=
  CM.Queries ∧ CM.DominatedWithin (pearlCProtocolDevRev1 devH100Tensor sem ρ) (pearlCTilesDevRev1 devH100Tensor sem ρ)
    (pearlCDomainDev devH100Tensor) (W1.w1Priced W1.W1Device.s8H100Best salt)

/-- `tt-out-tile-pearl-c-h100-rev1` (open premise, conjectured; refuted by a W1 program on `W1Device.s8H100Best` that outputs a set of
good tiles' words for less than `(1 − g₀)` of their credit). **TT_OUT(1/400) rev1 per audit tile at the H100 record
per second at the tensor clock, on the H100 with INT8**: `TTOutTilePearlCDevRev1W1` on `W1Device.s8H100Best` at
`devH100Tensor`, the adversary's best case for the prices no data sheet gives. -/
def TTOutTilePearlCH100Rev1S8 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (salt : S → List ℕ)
    (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTilePearlCDevRev1W1 W1.W1Device.s8H100Best salt devH100Tensor sem ρ

end Pouw.PearlC.Assumptions
