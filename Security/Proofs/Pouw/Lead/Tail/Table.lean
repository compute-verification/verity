import Proofs.Pouw.Lead.PStar.Table

/-!
# The tables' tail

`table_tail`: a strategy's memory, fixed before the coins, or the coin stream holds some committed tile on at most a
`tableTail` share of the coins. It is `PStar.memHolds_card_le`'s union bound over the call's tiles and steps with the
per-value shares before rounding (`memHolds_card_le_fine`): a tile at a step is one of the `mem + 1` values of the
pre-coin tables and zero on at most `2^−168` of the coins each (`PStar.call_tile_pstar`) and one of the `mem` streamed
tiles on at most `2^−105` each (`PStar.call_tile_stream`). A coin value with a table hit has at least one, so the
share is at most the hits' mean (`PStar.sum_tableHits`).
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Finset Pouw.Lead Pouw.SecurityProofs.Lead.PStar

section Memory

variable {L : Pouw.Lead.Layout} {rb cb : ℕ}

open Classical in
/-- **A memory holds a tile rarely**, before rounding: a tile at any step is held on at most a
`(mem + 1) · 2^−168 + mem · 2^−105` share of the coins. -/
theorem memHolds_card_le_fine (he : L.e = 7) (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m)
    (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (σ : Strategy L)
    (stream : CoinStream L) :
    2 ^ 168 * (univ.filter (MemHolds L κ σ stream rb cb t)).card ≤
      (L.mem + 1 + L.mem * 2 ^ 63) * Fintype.card (Coins L) := by
  set V : Finset (Fin tileWords → ℤ) := (range L.mem).image σ.pre ∪ {0}
  set A : (Fin tileWords → ℤ) → Finset (Coins L) := fun y =>
    univ.filter fun ω => stepTile hrb hcb t κ ω = unpos y
  set B : ℕ → Finset (Coins L) := fun c => univ.filter fun ω => stepTile hrb hcb t κ ω = streamTile stream c ω
  have hzero : ∀ ω c, L.mem ≤ c → stream.val ω c = 0 := fun ω c hc => funext fun p => by
    simp only [CoinStream.val, stream.supp c p hc, Option.elim, Pi.zero_apply]
  have hsub : univ.filter (MemHolds L κ σ stream rb cb t) ⊆ V.biUnion A ∪ (range L.mem).biUnion B := by
    intro ω hω
    have hA : ∀ y ∈ V, y = tileOf L κ ω rb cb t → ω ∈ V.biUnion A ∪ (range L.mem).biUnion B := fun y hy h =>
      mem_union_left _ (mem_biUnion.mpr ⟨y, hy, mem_filter.mpr ⟨mem_univ _, by rw [h, unpos_tileOf]⟩⟩)
    rcases (mem_filter.mp hω).2 with ⟨a, ha⟩ | ⟨c, hc⟩
    · by_cases hlt : a < L.mem
      · exact hA _ (mem_union_left _ (mem_image_of_mem _ (mem_range.mpr hlt))) ha
      · rw [σ.pre_supp a (not_lt.mp hlt)] at ha
        exact hA 0 (mem_union_right _ (mem_singleton_self _)) ha
    · by_cases hlt : c < L.mem
      · refine mem_union_right _ (mem_biUnion.mpr ⟨c, mem_range.mpr hlt, mem_filter.mpr ⟨mem_univ _, ?_⟩⟩)
        rw [← unpos_tileOf hrb hcb t κ ω, ← hc]
        rfl
      · rw [hzero ω c (not_lt.mp hlt)] at hc
        exact hA 0 (mem_union_right _ (mem_singleton_self _)) hc
  have hV : V.card ≤ L.mem + 1 := by
    refine (card_union_le _ _).trans ?_
    rw [card_singleton]
    exact Nat.add_le_add_right (card_image_le.trans (card_range _).le) 1
  have hAy : ∀ y, 2 ^ 168 * (A y).card ≤ Fintype.card (Coins L) := fun y =>
    call_tile_pstar he htile hrb hcb t κ (unpos y)
  have hBc : ∀ c, 2 ^ 168 * (B c).card ≤ 2 ^ 63 * Fintype.card (Coins L) := fun c => by
    calc 2 ^ 168 * (B c).card = 2 ^ 63 * (2 ^ 105 * (B c).card) := by ring
      _ ≤ 2 ^ 63 * Fintype.card (Coins L) := Nat.mul_le_mul_left _ (call_tile_stream he htile hrb hcb t κ stream c)
  calc 2 ^ 168 * (univ.filter (MemHolds L κ σ stream rb cb t)).card
      ≤ 2 ^ 168 * (∑ y ∈ V, (A y).card + ∑ c ∈ range L.mem, (B c).card) :=
        Nat.mul_le_mul_left _ ((card_le_card hsub).trans ((card_union_le _ _).trans
          (Nat.add_le_add card_biUnion_le card_biUnion_le)))
    _ = ∑ y ∈ V, 2 ^ 168 * (A y).card + ∑ c ∈ range L.mem, 2 ^ 168 * (B c).card := by
        rw [mul_add, mul_sum, mul_sum]
    _ ≤ ∑ _y ∈ V, Fintype.card (Coins L) + ∑ _c ∈ range L.mem, 2 ^ 63 * Fintype.card (Coins L) :=
        Nat.add_le_add (sum_le_sum fun y _ => hAy y) (sum_le_sum fun c _ => hBc c)
    _ = V.card * Fintype.card (Coins L) + L.mem * (2 ^ 63 * Fintype.card (Coins L)) := by
        rw [sum_const, sum_const, card_range, smul_eq_mul, smul_eq_mul]
    _ ≤ (L.mem + 1) * Fintype.card (Coins L) + L.mem * (2 ^ 63 * Fintype.card (Coins L)) :=
        Nat.add_le_add_right (Nat.mul_le_mul_right _ hV) _
    _ = (L.mem + 1 + L.mem * 2 ^ 63) * Fintype.card (Coins L) := by ring

end Memory

/-- **The tables' tail**, on any layout of whole tiles with 7-bit lanes in 64-row coin tiles. -/
theorem table_tail {L : Pouw.Lead.Layout} (he : L.e = 7) (htile : L.tile = 64) (hm : L.m = 16 * L.rowBlocks)
    (hn : L.n = 8 * L.colBlocks) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (σ : Strategy L) (stream : CoinStream L) :
    Pouw.pr (fun ω : Coins L => 0 < tableHits L κ σ stream ω) ≤ (tableTail L : ℝ) := by
  classical
  have hN : 2 ^ 168 * ∑ ω : Coins L, tableHits L κ σ stream ω ≤
      L.rowBlocks * L.colBlocks * L.S * ((L.mem + 1 + L.mem * 2 ^ 63) * Fintype.card (Coins L)) := by
    rw [sum_tableHits, mul_sum]
    calc _ ≤ ∑ _x ∈ tileIdx L, (L.mem + 1 + L.mem * 2 ^ 63) * Fintype.card (Coins L) := sum_le_sum fun x hx => by
          simp only [tileIdx, mem_product, mem_range] at hx
          exact memHolds_card_le_fine he htile (by omega) (by omega) x.2.2 κ σ stream
      _ = _ := by
          rw [sum_const, smul_eq_mul, tileIdx, card_product, card_product, card_range, card_range, card_univ,
            Fintype.card_fin]
          ring
  have hcard : (univ.filter fun ω : Coins L => 0 < tableHits L κ σ stream ω).card ≤
      ∑ ω : Coins L, tableHits L κ σ stream ω :=
    calc _ = ∑ _ω ∈ univ.filter fun ω : Coins L => 0 < tableHits L κ σ stream ω, 1 := card_eq_sum_ones _
      _ ≤ ∑ ω ∈ univ.filter fun ω : Coins L => 0 < tableHits L κ σ stream ω, tableHits L κ σ stream ω :=
          sum_le_sum fun ω hω => (mem_filter.mp hω).2
      _ ≤ ∑ ω : Coins L, tableHits L κ σ stream ω := sum_le_sum_of_subset (filter_subset _ _)
  have hq : (2 : ℚ) ^ 168 * (univ.filter fun ω : Coins L => 0 < tableHits L κ σ stream ω).card ≤
      L.rowBlocks * L.colBlocks * L.S * ((L.mem + 1 + L.mem * 2 ^ 63) * Fintype.card (Coins L)) := by
    exact_mod_cast (Nat.mul_le_mul_left _ hcard).trans hN
  have hq' : ((univ.filter fun ω : Coins L => 0 < tableHits L κ σ stream ω).card : ℚ) ≤
      tableTail L * Fintype.card (Coins L) := by
    have hT : tableTail L * Fintype.card (Coins L) =
        L.rowBlocks * L.colBlocks * L.S * ((L.mem + 1 + L.mem * 2 ^ 63) * Fintype.card (Coins L)) / 2 ^ 168 := by
      unfold tableTail
      field_simp
    rw [hT, le_div_iff₀ (by positivity)]
    linarith
  have hr : ((univ.filter fun ω : Coins L => 0 < tableHits L κ σ stream ω).card : ℝ) ≤
      (tableTail L : ℝ) * Fintype.card (Coins L) := by
    exact_mod_cast hq'
  unfold Pouw.pr
  rcases Nat.eq_zero_or_pos (Fintype.card (Coins L)) with h0 | hpos
  · rw [h0, Nat.cast_zero, div_zero]
    unfold tableTail
    positivity
  rw [div_le_iff₀ (by exact_mod_cast hpos)]
  convert hr using 4

end Pouw.SecurityProofs.Lead.Tail
