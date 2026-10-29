import Definitions.Pouw.Lead.Model
import Proofs.Pouw.Lead.CoinBranch.Kept

/-!
# Coin branching at tile grain

At each coin value the execution's run is its own path, so a prover that branches on the coins pays, on each path, a
write for every distinct committed tile its memory didn't start with; revisits (a tile written twice) only add writes.
-/

namespace Pouw.SecurityProofs.Lead.CoinBranch

open Pouw.Lead Pouw.Dimension.Pointwise

/-- **Per path, distinct tiles cost writes**: if every committed tile is held by the run at some
point (tile grain, `TileGrain` at a coin value with no egress, and `Accepted`'s holds), the committed tiles' distinct
values number at most the run's whole-tile writes plus the committed tiles its memory starts with (`tableHits`).
Reads `run`, `writes` and `Step.apply` (`Pouw.Dimension.Pointwise`), as `run_chunks_le` does at word grain. -/
theorem distinct_le_writes {L : Pouw.Lead.Layout} (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (σ : Strategy L)
    (stream : CoinStream L) (ω : Coins L)
    (hgrain : ∀ y, (σ.exec ω).holds y → ∃ k a, run (σ.init stream ω) ((σ.exec ω).steps.take k) a 0 = y)
    (hall : ∀ rb < L.rowBlocks, ∀ cb < L.colBlocks, ∀ t : Fin L.S, (σ.exec ω).holds (tileOf L κ ω rb cb t)) :
    ((tileIdx L).image fun x => tileOf L κ ω x.1 x.2.1 x.2.2).card ≤
      writes (σ.exec ω).steps + tableHits L κ σ stream ω := by
  classical
  have hR : ∀ x ∈ tileIdx L, ∃ k a, run (σ.init stream ω) ((σ.exec ω).steps.take k) a 0 =
      tileOf L κ ω x.1 x.2.1 x.2.2 := fun x hx => by
    simp only [tileIdx, Finset.mem_product, Finset.mem_range] at hx
    exact hgrain _ (hall _ hx.1 _ hx.2.1 _)
  unfold tableHits
  refine (image_card_le (σ.init stream ω) (σ.exec ω).steps (tileIdx L)
    (fun x => tileOf L κ ω x.1 x.2.1 x.2.2) hR).trans
    (Nat.add_le_add_left (Finset.card_le_card fun x hx => ?_) _)
  obtain ⟨hx, a, ha⟩ := Finset.mem_filter.1 hx
  refine Finset.mem_filter.2 ⟨hx, ?_⟩
  cases a with
  | inl a => exact Or.inl ⟨a, ha⟩
  | inr c => exact Or.inr ⟨c, ha⟩

end Pouw.SecurityProofs.Lead.CoinBranch
