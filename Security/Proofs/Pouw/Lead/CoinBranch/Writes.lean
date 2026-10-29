import Definitions.Pouw.Lead.Model

/-!
# Coin branching at tile grain

At each coin value the execution's run is its own path, so a prover that branches on the coins pays, on each path, a
write for every distinct committed tile its memory didn't start with; revisits (a tile written twice) only add writes.
-/

namespace Pouw.SecurityProofs.Lead.CoinBranch

open Pouw.Lead Pouw.Dimension.Pointwise

/-- **Per path, distinct tiles cost writes**: if the run holds every committed tile whole at some point (tile grain,
`TileGrain` at a coin value with no egress whose execution hashes the honest words), the committed tiles' distinct
values number at most the run's whole-tile writes plus the committed tiles its memory starts with (`tableHits`).
Reads `run`, `writes` and `Step.apply` (`Pouw.Dimension.Pointwise`), as `run_chunks_le` does at word grain. -/
theorem distinct_le_writes {L : Pouw.Lead.Layout} (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (σ : Strategy L)
    (stream : CoinStream L) (ω : Coins L)
    (hrun : ∀ rb < L.rowBlocks, ∀ cb < L.colBlocks, ∀ t : Fin L.S,
      ∃ k a, run (σ.init stream ω) ((σ.exec ω).steps.take k) a 0 = tileOf L κ ω rb cb t) :
    ((tileIdx L).image fun x => tileOf L κ ω x.1 x.2.1 x.2.2).card ≤
      writes (σ.exec ω).steps + tableHits L κ σ stream ω := by
  sorry

end Pouw.SecurityProofs.Lead.CoinBranch
