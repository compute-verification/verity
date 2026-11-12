module

public import Definitions.Pouw.PearlC.DeviceWithin
public import Definitions.Pouw.PearlC.Instance
public import Definitions.Pouw.Assumptions.PearlC.TTOutW1
public import Definitions.Pouw.PearlC.FlockTile.Audit

/-!
# Pearl-C's served form at every device record (trusted)

The served protocol, v1's checks under rev1 with the honest kernel's `W_ref` (`pearlCProtocolDevRev1K`), stated once
over every device record. Each statement takes the W1 setting `dv` and the device record `d` as parameters, as the
conjectures it rests on do, and asks one thing of the record: `KernelWithin d ρ c Ss ω`
(`Definitions.Pouw.PearlC.DeviceWithin`), on the shapes `Ss` its domain holds. Its gap is `1 − (1 − 1/400)/ω`, with no
hardware figure in it; the gap at a named record is the statement at that record's `ω`.

* `GapDevRev1K`: per unit, from TT_OUT rev1 at W1 (`TTOutPearlCDevRev1W1`).
* `SampledDevRev1K`: per audit tile, from TT_OUT rev1 per audit tile at W1 (`TTOutTilePearlCDevRev1W1`).
* `HiddenFlockDevRev1K`: per audit tile under Flock's hidden audit over a TileRead session `F` (`F.audit`), at
  `t·δzk(q)` more error than the sampled audit, given T3 at the session (`TileReadSound`) and the binding
  (`TileReadBinds`), the two open obligations of the Flock tile audit's statements.

Each proof is `Pouw.SecurityProofs.PearlC.G : Pouw.Guarantees.PearlC.G` (`Proofs.Pouw.PearlC.DevGap`).
-/

@[expose] public section

namespace Pouw.Guarantees.PearlC

open Pouw.PearlC Pouw.PearlC.Assumptions Pouw.PearlC.FlockTile

/-- **The served form's gap per unit, at every device record.** For every W1 setting `dv`, salt bytes `salt`, device
record `d`, semantics `sem`, inert-work share `ρ`, kernel forming `c` and shapes `Ss`: if the honest kernel's `W_ref` is
within `ω` of the credit left after the inert-work share on `Ss` (`KernelWithin`), TT_OUT rev1 at W1 gives `G_g` on
the domain restricted to `Ss` at `g = 1 − (1 − 1/400)/ω`. -/
def GapDevRev1K : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : Pouw.PearlC.W1.W1Device)
    (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (Ss : Pouw.PearlC.Shape → Prop) (ω : ℚ),
    KernelWithin d ρ c Ss ω → TTOutPearlCDevRev1W1 dv salt d sem ρ →
    Pouw.PearlC.GGap (Pouw.PearlC.W1.w1Priced dv salt) (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss)
      ((1 - (1 - 1 / 400) / ω : ℚ) : ℝ) δPearlC

/-- **The served form's gap per audit tile, at every device record**: `GapDevRev1K`'s statement for the sampled audit,
from TT_OUT rev1 per audit tile at W1. -/
def SampledDevRev1K : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : Pouw.PearlC.W1.W1Device)
    (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (Ss : Pouw.PearlC.Shape → Prop) (ω : ℚ),
    KernelWithin d ρ c Ss ω → TTOutTilePearlCDevRev1W1 dv salt d sem ρ →
    Pouw.PearlC.GGapSampled (Pouw.PearlC.W1.w1Priced dv salt) (pearlCProtocolDevRev1K d sem ρ c)
      (pearlCTilesDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss) ((1 - (1 - 1 / 400) / ω : ℚ) : ℝ) δPearlC

/-- **The served form's gap per audit tile under Flock's hidden audit, at every device record**: `SampledDevRev1K`'s
statement at `F.audit` for a TileRead session `F`, at `t·δzk(q)` more error, given `0 ≤ δzk`, T3 at the session
(`TileReadSound`, to be discharged by Flock's soundness of the TileRead statement, whose CR hypotheses are `F.Cls`) and
the binding (`TileReadBinds`). It says something about Flock only at the session whose `game` is TileRead's live session
and whose `Cls` is T3's CR hypotheses: at a session with an empty `Cls`, `F.audit`'s class is empty and both obligations
hold with `Bad` everything. -/
def HiddenFlockDevRev1K : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : Pouw.PearlC.W1.W1Device)
    (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (Ss : Pouw.PearlC.Shape → Prop) (ω : ℚ)
    (F : TileReadSession Q R S) (δzk : ℕ → ℝ),
    KernelWithin d ρ c Ss ω → (∀ q, 0 ≤ δzk q) → TTOutTilePearlCDevRev1W1 dv salt d sem ρ →
    TileReadSound (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss) F δzk →
    TileReadBinds (pearlCProtocolDevRev1K d sem ρ c) (pearlCTilesDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss) F →
    Pouw.PearlC.GGapHidden (Pouw.PearlC.W1.w1Priced dv salt) (pearlCProtocolDevRev1K d sem ρ c)
      (pearlCTilesDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss) F.audit ((1 - (1 - 1 / 400) / ω : ℚ) : ℝ) δPearlC δzk

end Pouw.Guarantees.PearlC
