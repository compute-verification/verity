module

public import Definitions.Pouw.Assumptions.PearlC.TTOutW1

/-!
# Pearl-C's TT_OUT on the H100, from a device model and a conjecture

`TTOutTilePearlCH100Rev1 CM` (`Definitions.Pouw.Assumptions.PearlC.TTOutRev1`) is TT_OUT per audit tile over an
abstract cost model: which `CM` it holds at is prose, so it mixes a claim about the card with a claim about the
programs that run on it. By ruling (8 Oct) a mixed statement is split, and each part is named:

* **The device model, `w1-complete-h100`** (`W1CompleteH100`): the card's cost model `CM` reaches the oracle only
  through counted queries, and W1 on `W1Device.s8H100Best` (the H100's and the H200's tensor-core and FP32 rates, INT8
  included, with a compression and a first fetch free) dominates it on the H100 record's tiles
  (`CostModel.DominatedBy`): for every adversary of `CM` at an in-domain workload, some W1 program, run on that
  adversary's activations and preprocessed state, costs no more and makes no more oracle calls on every oracle and
  salt, and its good tiles earn at least the credit the adversary's do. It is the data sheets' reading: NVIDIA's H100
  Tensor Core GPU Architecture whitepaper V1.04, Table 1, and the H200 Tensor Core GPU data sheet (`W1.W1Prices.h100`).
  It is falsified by a measurement: a program on the card that earns more good-tile credit, for its cost, than every
  W1 program at that cost.
* **The conjecture, `tt-out-tile-pearl-c-h100-rev1`** (`TTOutTilePearlCH100Rev1S8`): TT_OUT per audit tile, rev1, at the
  H100 record, over W1 on that setting. It is about W1 programs only, and it is falsified by one: a W1 program that
  outputs a set of good tiles' words for less than `(1 − g₀)` of their credit. The standing falsification test is the
  census's known skip classes replayed against rev1's credit
  (`benchmarks/pouw/tests/test_tt_out_tile_h100_rev1.py`).

Together they give the abstract form at `CM` (`Pouw.PearlC.ttOutTilePearlCH100Rev1_of_s8`, in
`Proofs.Pouw.PearlC.S8H100Rev1Gap`). The converse does not hold: the pair says more than the abstract form, since the
conjecture is stated over W1, INT8 fragments and free compressions and fetches included, whatever `CM` is. The device
model asks only what that implication uses. It does not ask W1 to reproduce the card's transcript, so a program of `CM`
that leaves a unit blank, or that reads `H` beyond its counted queries, does not refute it. A blank unit earns no
credit, and the credit of a good tile is all that is compared.

It is a module of its own under `Definitions.Assumptions` that reads only PoUW's definitions, so the PoUW statements
that take these import it without the rest of the named assumptions. It also holds the two definitions they read
beside W1's and the game's: the domination `CostModel.DominatedBy` and the setting `W1.W1Device.s8H100Best`.
-/

@[expose] public section

namespace Pouw.PearlC

variable {Q R S : Type}

/-- **`CM'` dominates `CM` on the tiles `TR`**: for every in-domain workload and every adversary of `CM`, some program
of `CM'`, run on the adversary's activations and its preprocessed state, costs no more and makes no more oracle calls
on every oracle and salt, and the credit of the good tiles it commits is at least that of the adversary's. Each such
adversary is then one of `CM'` within the same bounds that earns no less, so TT_OUT per tile at `CM'` gives it at `CM`
(`Pouw.PearlC.ttOutTile_of_dominatedBy`). -/
def CostModel.DominatedBy (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (CM CM' : CostModel Q R S) :
    Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ A : Adversary CM, ∃ p' : CM'.Prog, ∀ (H : Q → R) (s : S),
    CM'.cost p' H U A.act (A.pre.eval H) s ≤ CM.cost A.online H U A.act (A.pre.eval H) s ∧
      CM'.calls p' H U A.act (A.pre.eval H) s ≤ CM.calls A.online H U A.act (A.pre.eval H) s ∧
      ∑ g ∈ goodTiles P TR U H s (A.transcript U H s),
          TR.credit H s U g (actOf (A.transcript U H s) ((TR.tiling U.layout).unit g)) ≤
        ∑ g ∈ goodTiles P TR U H s (CM'.run p' H U A.act (A.pre.eval H) s),
          TR.credit H s U g (actOf (CM'.run p' H U A.act (A.pre.eval H) s) ((TR.tiling U.layout).unit g))

/-- **The Hopper INT8 setting at the adversary's best case**: `W1Device.s8H100` with a SHA-512 compression and a first
fetch free. Neither price is on a data sheet (the H100 whitepaper's Table 1 and the H200 data sheet give tensor-core and
FP32 rates, not a compression's or a word's first read), and the audited hardware and its vendor are untrusted, so
neither is taken at a measured rate. W1's cost only grows with each: at any `sha, fetch ≥ 0`, `s8H100 sha fetch` is
simulated by this setting (`Pouw.PearlC.W1.w1Priced_s8H100_simulatedBy_best`), so a bound here holds at every price. -/
def W1.W1Device.s8H100Best : W1.W1Device := W1.W1Device.s8H100 0 0

end Pouw.PearlC

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- `w1-complete-h100` (measured; falsified by a program on an H100 or an H200 that earns more good-tile credit at the
H100 record, rev1, for its cost than every W1 program on `W1Device.s8H100Best` at that cost). **W1 is complete for the
H100's and the H200's credited work**: the card's cost model `CM` reaches `H` only through counted queries, and W1 on
`W1Device.s8H100Best` with the salt's bytes `salt` dominates it on rev1's audit tiles at the H100 record
(`CostModel.DominatedBy`, at `pearlCProtocolDevRev1 devH100 sem ρ`, `pearlCTilesDevRev1 devH100 sem ρ` and
`pearlCDomainDev devH100`). Its prices are the data sheets' per SM clock: NVIDIA's H100 Tensor Core GPU Architecture
whitepaper V1.04, Table 1, and the H200 Tensor Core GPU data sheet (`W1.W1Prices.h100`), with a compression and a first
fetch at 0. -/
def W1CompleteH100 (salt : S → List ℕ) (sem : PearlCSem Q R S) (ρ : ℚ) (CM : CostModel Q R S) : Prop :=
  CM.Queries ∧ CM.DominatedBy (pearlCProtocolDevRev1 devH100 sem ρ) (pearlCTilesDevRev1 devH100 sem ρ)
    (pearlCDomainDev devH100) (W1.w1Priced W1.W1Device.s8H100Best salt)

/-- `tt-out-tile-pearl-c-h100-rev1` (conjectured; refuted by a W1 program on `W1Device.s8H100Best` that outputs a set of
good tiles' words for less than `(1 − g₀)` of their credit). **TT_OUT(1/400) rev1 per audit tile at the H100 record,
on the H100 with INT8**: `TTOutTilePearlCDevRev1W1` on `W1Device.s8H100Best`, the adversary's best case for the
prices no data sheet gives. -/
def TTOutTilePearlCH100Rev1S8 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (salt : S → List ℕ)
    (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTilePearlCDevRev1W1 W1.W1Device.s8H100Best salt devH100 sem ρ

end Pouw.PearlC.Assumptions
