module

public import Proofs.Pouw.PearlC.TileCheckWords
public import Definitions.Pouw.Assumptions.PearlC.TTOutRowSeed

/-!
# The plain `pc8` tile check's words under rev1 and its row caps

`TileCheck.ComputesWords` reads a protocol's ticket and useful words and its weight rule, and a tile rule's tiling and
row rule. It never reads credit, cap or `W_ref`, so it moves between protocols and tile rules that agree on those
(`TileCheck.ComputesWords.transfer`). Rev1 (`pearlCProtocolDevRev1`, `pearlCTilesDevRev1`) and its caps per row
(`pearlCProtocolDevRev1RowCapAt`, `pearlCTilesDevRev1RowCapAt`) change only credit, cap and `W_ref`, so the H100's words
theorem holds at them as it is: `tileCheck8_computesWords_h100Rev1` at the protocol and tiles `TTOutTilePearlCH100Rev1`
reads, and `tileCheck8_computesWords_h100Rev1RowCapAt` at those of the `-h3` sampled g.
-/

@[expose] public section

namespace Pouw.PearlC

variable {Q R S : Type}

/-- `ComputesWords` moves to any protocol and tile rules with the same words, weight rule, tiling and row rule, on any
smaller domain. -/
theorem TileCheck.ComputesWords.transfer {TC : TileCheck Q R S} {P P' : Protocol Q R S} {TR TR' : TileRules Q R S}
    {D D' : Layout → Prop} (h : TC.ComputesWords P TR D) (hD : ∀ L, D' L → D L) (hticket : P'.ticket = P.ticket)
    (huseful : P'.useful = P.useful) (hweight : P'.fixedOK = P.fixedOK) (htiling : TR'.tiling = TR.tiling)
    (hrow : TR'.rowOK = TR.rowOK) : TC.ComputesWords P' TR' D' := by
  obtain ⟨_, _, _, _, _, _, _⟩ := P
  obtain ⟨_, _, _, _, _, _, _⟩ := P'
  obtain ⟨_, _, _, _, _, _⟩ := TR
  obtain ⟨_, _, _, _, _, _⟩ := TR'
  dsimp only at hticket huseful hweight htiling hrow
  subst hticket huseful hweight htiling hrow
  intro U hU H s g hg c X hacc
  exact h U ⟨hU.1, hD _ hU.2⟩ H s g hg c X hacc

end Pouw.PearlC

namespace Pouw.PearlC.WholeCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.TileCircuit Pouw.PearlC.RowCircuit

variable {Q R S : Type}

/-- **L3's words for the plain `pc8` Program at the H100's record under rev1**: `tileCheck8_computesWords_h100`, since
rev1 changes only credit, cap and `W_ref`. -/
theorem tileCheck8_computesWords_h100Rev1 (ρ : ℚ)
    (noise : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Noise (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k Params.pi.r) :
    (tileCheck8 coreOps hopperE4m3K32 noise).ComputesWords
      (pearlCProtocolDevRev1 devH100 (pc8Sem coreOps hopperE4m3K32 noise) ρ)
      (pearlCTilesDevRev1 devH100 (pc8Sem coreOps hopperE4m3K32 noise) ρ) (fun L => pearlCDomainDev devH100 L ∧
        ∀ u < L.N, tileSide (L.shape u).m ∣ (L.shape u).m ∧ tileSide (L.shape u).n ∣ (L.shape u).n) :=
  (tileCheck8_computesWords_h100 Costs.adopted ρ noise).transfer (fun _ h => h) rfl rfl rfl rfl rfl

/-- **L3's words for the plain `pc8` Program at the H100's record under rev1, with the tile cap `ρ` and the row cap
`ρr`**: `tileCheck8_computesWords_h100` again, since the caps per row change only the cap. -/
theorem tileCheck8_computesWords_h100Rev1RowCapAt (ρ ρr : ℚ)
    (noise : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Noise (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k Params.pi.r) :
    (tileCheck8 coreOps hopperE4m3K32 noise).ComputesWords
      (pearlCProtocolDevRev1RowCapAt devH100 (pc8Sem coreOps hopperE4m3K32 noise) ρ ρr)
      (pearlCTilesDevRev1RowCapAt devH100 (pc8Sem coreOps hopperE4m3K32 noise) ρ ρr) (fun L => pearlCDomainDev devH100 L ∧
        ∀ u < L.N, tileSide (L.shape u).m ∣ (L.shape u).m ∧ tileSide (L.shape u).n ∣ (L.shape u).n) :=
  (tileCheck8_computesWords_h100 Costs.adopted ρ noise).transfer (fun _ h => h) rfl rfl rfl rfl rfl

end Pouw.PearlC.WholeCircuit
