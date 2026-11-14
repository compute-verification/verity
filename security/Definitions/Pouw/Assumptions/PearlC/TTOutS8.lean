module

public import Definitions.Pouw.Assumptions.PearlC.TTOutW1

/-!
# Pearl-C's TT_OUT on the H100, from a device model and a conjecture (open premises)

`TTOutTilePearlCH100Rev1 CM` (`Definitions.Pouw.Assumptions.PearlC.TTOutRev1`) is TT_OUT per audit tile over an
abstract cost model: which `CM` it holds at is prose, so it mixes a claim about the card with a claim about the
programs that run on it. This module splits it in two. Neither part is a named assumption: both are open premises,
listed under `open` in the lock, and the guarantees that take them are conditional on them.

* **The device model, `w1-complete-h100`** (`W1CompleteH100`): the card's cost model `CM` reaches the oracle only
  through counted queries, and W1 on `W1Device.s8H100Best` (the H100's and the H200's tensor-core and SM-pipe rates,
  INT8 included, per second at the tensor clock, with a compression and a first fetch free) dominates it within every
  bound on the tiles of `devH100Tensor`, the H100 record per second at the tensor clock (`CostModel.DominatedWithin`):
  for every cost `T` and query count `q`, every adversary of `CM` at an in-domain workload within `T` and `q` has an
  adversary of W1 within `T` and `q` whose good tiles earn more than any given credit with at least the same
  probability over the oracle and the salt. It is the data sheets' reading: NVIDIA's H100 Tensor Core GPU Architecture
  whitepaper V1.04, Tables 1 and 3, and the H200 Tensor Core GPU data sheet (`W1.W1Prices.h100Tensor`). Its SM pipes
  are priced per second at the 1830 MHz tensor clock beside a 1980 MHz SM clock (`h100TensorMHz`, `h100SmMHz`), so it
  holds whether a mixed load keeps both pipes at one clock or each at its own. It is falsified by a measurement: a
  program on the card within a cost and a query count that earns more good-tile credit, with higher probability, than
  every W1 program within that cost and count. It is falsified on the H200: the timed loop of FP32 beside the tensor
  pipe refutes both data-sheet readings of an FP32 instruction's price, the per-clock 32 and the tensor-clock 976/33,
  so the lock marks it falsified under `shape.conjectures`.
* **The conjecture, `tt-out-tile-pearl-c-h100-rev1`** (`TTOutTilePearlCH100Rev1S8`): TT_OUT per audit tile, rev1, at
  `devH100Tensor`, over W1 on that setting. It is about W1 programs only, and it is falsified by one: a W1 program
  that outputs a set of good tiles' words for less than `(1 − g₀)` of their credit. The standing falsification test is
  the census's known skip classes replayed against rev1's credit at that record
  (`benchmarks/pouw/tests/test_tt_out_tile_h100_rev1.py`). TT_OUT rev1 itself is refuted by the proofs red team's
  rating.

Together they give TT_OUT per tile at `CM` on `devH100Tensor` (`Pouw.PearlC.ttOutTilePearlCH100Rev1_of_s8`, in
`Proofs.Pouw.PearlC.S8H100Rev1Gap`): the abstract form's statement on that record, whose credit is below `devH100`'s,
so they do not give the abstract form at `devH100` itself. The converse does not hold: the pair says more than TT_OUT
at `CM`, since the conjecture is stated over W1, INT8 fragments and free compressions and fetches included, whatever
`CM` is. The device model asks only what that implication uses: the bound an adversary is held to and the chance its
good tiles earn a given credit. It does not compare runs, costs or calls oracle by oracle and salt by salt, so a card
program that leaves a unit blank, or that tests the salt with an instruction W1 lacks or prices higher (a one-bit
tensor MMA, the SFU, FP64) and stops early on some salts, refutes it only if it earns more than every W1 program within
its bound. `CM.Queries` holds a card program's run and calls to its counted queries, not its cost. A program that reads
`H` beyond those queries and discards what it read changes nothing W1 must match; one whose cost moves with the oracle
is held to its bound at its costliest oracle and salt, and refutes the device model if no W1 program within that bound
earns as much.

It sits beside the other TT_OUT forms over W1 (`Definitions.Pouw.Assumptions.PearlC.TTOutW1`) and also holds the two
definitions the premises read beside W1's and the game's: the domination within every bound
`CostModel.DominatedWithin` and the setting `W1.W1Device.s8H100Best`.
-/

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
      pr (fun x : (Q → R) × S => c < ∑ g ∈ goodTiles P TR U x.1 x.2 (A.transcript U x.1 x.2),
          TR.credit x.1 x.2 U g (actOf (A.transcript U x.1 x.2) ((TR.tiling U.layout).unit g))) ≤
        pr (fun x : (Q → R) × S => c < ∑ g ∈ goodTiles P TR U x.1 x.2 (A'.transcript U x.1 x.2),
          TR.credit x.1 x.2 U g (actOf (A'.transcript U x.1 x.2) ((TR.tiling U.layout).unit g)))

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
