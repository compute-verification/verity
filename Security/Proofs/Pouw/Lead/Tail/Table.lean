import Proofs.Pouw.Lead.PStar.Table

/-!
# The tables' tail

`table_tail`: a strategy's memory, fixed before the coins, or the coin stream holds some committed tile on at most a
`tableTail` share of the coins. It is `PStar.memHolds_card_le`'s union bound over the call's tiles and steps with the
per-value shares before rounding: a tile at a step is one of the `mem + 1` values of the pre-coin tables and zero on
at most `2^−168` of the coins each (`PStar.call_tile_pstar`) and one of the `mem` streamed tiles on at most `2^−105`
each (`PStar.call_tile_stream`).
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Pouw.Lead

/-- **The tables' tail**, on any layout of whole tiles with 7-bit lanes in 64-row coin tiles. -/
theorem table_tail {L : Pouw.Lead.Layout} (he : L.e = 7) (htile : L.tile = 64) (hm : L.m = 16 * L.rowBlocks)
    (hn : L.n = 8 * L.colBlocks) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (σ : Strategy L) (stream : CoinStream L) :
    Pouw.pr (fun ω : Coins L => 0 < tableHits L κ σ stream ω) ≤ (tableTail L : ℝ) := by
  sorry

end Pouw.SecurityProofs.Lead.Tail
