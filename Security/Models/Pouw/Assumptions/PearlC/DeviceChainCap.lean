import Models.Pouw.Assumptions.PearlC.TTOut

/-!
# The cap on the chain's share (trusted; definitions and the named `Prop`)

`pearlCProtocolDev` admits a unit whose debit is at most `ρ·creditDev`. `creditDev`'s A′·F_B, forming and peel terms don't
shrink with the unit's width n, so that cap admits a larger flagged share of the chain on narrow units: up to
`ρ·creditDev/(m·k·n)`, 0.214% at n = 64 and k = 8,192 for ρ = 1/1,000, where the aligned-spike family R = 64 (0.12–0.14%
flagged) passes. The chain-share cap compares the same debit with `ρ` times the chain's credited work alone
(`chainCreditDev`: its MACs and promotion adds), so an admitted unit has at most a `ρ` share of its atom-words flagged at
every shape (`chainCap_flagged_le`). Everything else, the credit and the tiles' fixed credit `(1 − ρ)·creditDev·share`
included, is `pearlCProtocolDev`'s and `pearlCTilesDev`'s.

It is stated at any device record. The panel's line is v2 (`devSm120v2`, G = 0) at ρ = 1/1,000. At G > 0 the same
construction on rev1's protocol is the analogue, not staged.
-/

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

/-- The chain's credited work of one unit at a device: `m·k·n` MACs and its promotion adds, as `creditDev` counts them. -/
def chainCreditDev (d : PearlCDevice) (s : Shape) : ℚ :=
  let mkn : ℚ := (s.m : ℚ) * s.k * s.n
  mkn + (if d.G = 0 then 0 else mkn * d.prices.add / (32 * d.G))

variable {Q R S : Type}

/-- **Pearl-C's protocol at a device record with the cap on the chain's share**: `pearlCProtocolDev` with the cap
`debitDev ≤ ρ·chainCreditDev`. -/
noncomputable def pearlCProtocolDevChainCap (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Protocol Q R S :=
  { pearlCProtocolDev d sem ρ with
    capOK := fun H s U u act => debitDev d (sem.unitDebitDev d H s U u act) ≤ ρ * chainCreditDev d (U.layout.shape u) }

/-- **Pearl-C's audit tiles with the cap on the chain's share**: `pearlCTilesDev` with a tile's debit capped at
`ρ·chainCreditDev` times its credited rows' share. A good tile's credit stays `(1 − ρ)·creditDev·share`. -/
noncomputable def pearlCTilesDevChainCap (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : TileRules Q R S :=
  { pearlCTilesDev d sem ρ with
    capOK := fun H s U g act =>
      let tl := auditTiling 64 64 U.layout
      let sh := U.layout.shape (tl.unit g)
      sem.tileDebitDev d H s U (tl.unit g) (sem.passRows sh act (tl.rows g)) (tl.cols g) act ≤
        ρ * chainCreditDev d sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) }

end Pouw.PearlC

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- **TT_OUT(1/400) at a device record with the cap on the chain's share**: `TTOut` at
`pearlCProtocolDevChainCap d sem ρ`, `pearlCDomainDev d`, `γ₀ = 1/400` and `εPearlC`. It is `TTOutPearlCDev` with
fewer units credited: a unit counts only if its debit is at most `ρ` times its chain's credited work, which bounds its
flagged share of atom-words by `ρ` at every width (`chainCap_flagged_le`). It asserts less than `TTOutPearlCDev` at the
same `ρ` (`ttOutPearlCDevChainCap_of_cap`). The game, charging rule, error, debit, falsifiers and cost model are
`TTOutPearlCDev`'s: `CM` charges a tensor-core instruction its whole fragment shape, never per output word. The panel's
instance is `devSm120v2` at `ρ = 1/1,000` (`tt-out/pearl-c-sm120-unpromoted-chaincap1000`). -/
def TTOutPearlCDevChainCap [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOut CM (pearlCProtocolDevChainCap d sem ρ) (pearlCDomainDev d) (1 / 400) εPearlC

/-- **TT_OUT(1/400) per audit tile with the cap on the chain's share**: `TTOutTile` at
`pearlCProtocolDevChainCap d sem ρ` and `pearlCTilesDevChainCap d sem ρ`. It asserts less than `TTOutTilePearlCDev` at
the same `ρ` (`ttOutTilePearlCDevChainCap_of_cap`): the same fixed credit on fewer tiles. -/
def TTOutTilePearlCDevChainCap [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTile CM (pearlCProtocolDevChainCap d sem ρ) (pearlCTilesDevChainCap d sem ρ) (pearlCDomainDev d) (1 / 400)
    εPearlC

end Pouw.PearlC.Assumptions
