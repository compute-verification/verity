import Definitions.Pouw.Lead.Gap
import Proofs.Pouw.Lead.PStar.Lanes
import Proofs.Pouw.Lead.PStar.Stream

/-!
# The coin lanes' guess, and tables against the committed tiles

`pStar_eq`: the exact `p*` of `blc4a-t64-e7`'s three lanes under `-rot`'s weight `(127, 1, 1)` with the tag bit, 130
of `2 · 128³`, so `65 / 2^21`. `table_le`: a memory fixed before the coins, or a coin stream of raw coins, holds a
committed tile rarely: at most `tableGap` in words averaged over the coins.
-/

namespace Pouw.SecurityProofs.Lead.PStar

open Finset Pouw.Lead

/-- A coin lane's value from its draw `u`: `2u − 127`. -/
def lane (u : Fin 128) : ℤ := 2 * (u : ℤ) - 127

theorem lane_mem (u : Fin 128) : lane u ∈ coins := by
  have hu := u.isLt
  refine mem_coins.mpr ⟨(u : ℤ) - 64, by omega, by omega, ?_⟩
  simp only [lane]
  ring

theorem lane_injective : Function.Injective lane := fun u u' h => by
  simp only [lane] at h
  exact Fin.ext (by omega)

theorem exists_lane {x : ℤ} (hx : x ∈ coins) : ∃ u, lane u = x := by
  obtain ⟨k, h1, h2, rfl⟩ := mem_coins.mp hx
  refine ⟨⟨(k + 64).toNat, by omega⟩, ?_⟩
  simp only [lane]
  omega

set_option linter.constructorNameAsVariable false in
/-- The statement's draws of a step's lanes and tag bit are `rotLane 1 0`'s on an even row. -/
theorem pStar_fiber_card (v : ℤ) :
    (univ.filter fun x : Fin 128 × Fin 128 × Fin 128 × Fin 2 =>
        127 * lane x.1 + lane x.2.1 + lane x.2.2.1 + 1 + (x.2.2.2 : ℤ) = v).card =
      (Pouw.Dimension.Alphabet.fiber (rotLane 1 0) ({0, 1} : Finset ℤ) triples v).card := by
  refine card_bij (fun x _ => (((x.2.2.2 : ℕ) : ℤ), lane x.1, lane x.2.1, lane x.2.2.1)) (fun x hx => ?_)
    (fun x _ y _ h => ?_) (fun p hp => ?_)
  · simp only [mem_filter, mem_univ, true_and] at hx
    rw [Pouw.Dimension.Alphabet.fiber, mem_filter, mem_product, mem_triples]
    refine ⟨⟨?_, lane_mem _, lane_mem _, lane_mem _⟩, ?_⟩
    · have := x.2.2.2.isLt
      simp only [mem_insert, mem_singleton]
      omega
    · rw [← hx]
      simp only [rotLane, rotSum, evenSum, Nat.zero_mod, ite_true]
      ring
  · simp only [Prod.mk.injEq] at h
    obtain ⟨hb, h0, h1, h2⟩ := h
    exact Prod.ext (lane_injective h0) (Prod.ext (lane_injective h1)
      (Prod.ext (lane_injective h2) (Fin.ext (by omega))))
  · obtain ⟨b, s0, s1, s2⟩ := p
    rw [Pouw.Dimension.Alphabet.fiber, mem_filter, mem_product, mem_triples] at hp
    obtain ⟨⟨hb, h0, h1, h2⟩, hv⟩ := hp
    obtain ⟨u0, rfl⟩ := exists_lane h0
    obtain ⟨u1, rfl⟩ := exists_lane h1
    obtain ⟨u2, rfl⟩ := exists_lane h2
    have hb' : b = 0 ∨ b = 1 := by simpa only [mem_insert, mem_singleton] using hb
    refine ⟨(u0, u1, u2, ⟨b.toNat, by omega⟩), ?_, ?_⟩
    · simp only [mem_filter, mem_univ, true_and]
      rw [← hv]
      simp only [rotLane, rotSum, evenSum, Nat.zero_mod, ite_true]
      have : (((⟨b.toNat, by omega⟩ : Fin 2) : ℕ) : ℤ) = b := by simp only; omega
      rw [this]
      ring
    · simp only [Prod.mk.injEq, and_true]
      omega

/-- **The exact `p*`**: a value of `127 s₀ + s₁ + s₂ + 1 + b` is taken by at most 130 of the
`2 · 128³` draws, and one value by exactly 130 (`p* = 65 / 2^21`). -/
theorem pStar_eq :
    (∀ v : ℤ, (Finset.univ.filter fun x : Fin 128 × Fin 128 × Fin 128 × Fin 2 =>
        127 * lane x.1 + lane x.2.1 + lane x.2.2.1 + 1 + (x.2.2.2 : ℤ) = v).card ≤ 130) ∧
      ∃ v : ℤ, (Finset.univ.filter fun x : Fin 128 × Fin 128 × Fin 128 × Fin 2 =>
        127 * lane x.1 + lane x.2.1 + lane x.2.2.1 + 1 + (x.2.2.2 : ℤ) = v).card = 130 := by
  refine ⟨fun v => ?_, 1 + 127, ?_⟩
  · rw [pStar_fiber_card]
    exact rotLane_fiber_le 1 0 v
  · rw [pStar_fiber_card]
    exact le_antisymm (rotLane_fiber_le 1 0 _) (rotLane_fiber_attained 1 0)

/-! ## A tile against a memory -/

section Memory

variable {L : Pouw.Lead.Layout} {rb cb : ℕ}

/-- A tile's words row by row. -/
def unpos (y : Fin tileWords → ℤ) : Fin 16 → Fin 8 → ℤ := fun r q => y (tilePos r q)

theorem unpos_tileOf (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n) (t : Fin L.S)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) :
    unpos (tileOf L κ ω rb cb t) = stepTile hrb hcb t κ ω := by
  funext r q
  have hp : (tilePos r q : ℕ) = 8 * r + q := rfl
  simp only [unpos, tileOf, stepTile]
  rw [dite_eq_left (by rw [hp]; omega)]
  congr 1 <;> refine Fin.ext ?_ <;> simp only [tileRow, tileCol, hp] <;> omega

/-- A strategy's memory holds a tile's words: a pre-coin table's tile or a streamed one. -/
def MemHolds (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (σ : Strategy L) (stream : CoinStream L)
    (rb cb : ℕ) (t : Fin L.S) (ω : Coins L) : Prop :=
  (∃ a, σ.pre a = tileOf L κ ω rb cb t) ∨ ∃ c, stream.val ω c = tileOf L κ ω rb cb t

open Classical in
/-- **A memory holds a tile rarely**: a tile at any step is among the at most `mem + 1` values of the pre-coin tables
(zero included) on at most a `2^−168` share of the coins each, and is one of the `mem` streamed tiles on at most a
`2^−105` share each, so on at most a `(2 · mem + 1) · 2^−100` share. -/
theorem memHolds_card_le (he : L.e = 7) (htile : L.tile = 64) (hrb : 16 * rb + 16 ≤ L.m) (hcb : 8 * cb + 8 ≤ L.n)
    (t : Fin L.S) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (σ : Strategy L) (stream : CoinStream L) :
    2 ^ 100 * (univ.filter (MemHolds L κ σ stream rb cb t)).card ≤ (2 * L.mem + 1) * Fintype.card (Coins L) := by
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
  have hAy : ∀ y, 2 ^ 100 * (A y).card ≤ Fintype.card (Coins L) := fun y =>
    (Nat.mul_le_mul_right _ (Nat.pow_le_pow_right (by norm_num) (by norm_num))).trans
      (call_tile_pstar he htile hrb hcb t κ (unpos y))
  have hBc : ∀ c, 2 ^ 100 * (B c).card ≤ Fintype.card (Coins L) := fun c =>
    (Nat.mul_le_mul_right _ (Nat.pow_le_pow_right (by norm_num) (by norm_num))).trans
      (call_tile_stream he htile hrb hcb t κ stream c)
  calc 2 ^ 100 * (univ.filter (MemHolds L κ σ stream rb cb t)).card
      ≤ 2 ^ 100 * (∑ y ∈ V, (A y).card + ∑ c ∈ range L.mem, (B c).card) :=
        Nat.mul_le_mul_left _ ((card_le_card hsub).trans ((card_union_le _ _).trans
          (Nat.add_le_add card_biUnion_le card_biUnion_le)))
    _ = ∑ y ∈ V, 2 ^ 100 * (A y).card + ∑ c ∈ range L.mem, 2 ^ 100 * (B c).card := by
        rw [mul_add, mul_sum, mul_sum]
    _ ≤ ∑ _y ∈ V, Fintype.card (Coins L) + ∑ _c ∈ range L.mem, Fintype.card (Coins L) :=
        Nat.add_le_add (sum_le_sum fun y _ => hAy y) (sum_le_sum fun c _ => hBc c)
    _ = V.card * Fintype.card (Coins L) + L.mem * Fintype.card (Coins L) := by
        rw [sum_const, sum_const, card_range, smul_eq_mul, smul_eq_mul]
    _ ≤ (L.mem + 1) * Fintype.card (Coins L) + L.mem * Fintype.card (Coins L) :=
        Nat.add_le_add_right (Nat.mul_le_mul_right _ hV) _
    _ = (2 * L.mem + 1) * Fintype.card (Coins L) := by ring

open Classical in
/-- Summed over the coins, the table hits are the coins on which each committed tile is held. -/
theorem sum_tableHits (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (σ : Strategy L) (stream : CoinStream L) :
    ∑ ω : Coins L, tableHits L κ σ stream ω =
      ∑ x ∈ tileIdx L, (univ.filter (MemHolds L κ σ stream x.1 x.2.1 x.2.2)).card := by
  simp only [tableHits, MemHolds, card_filter]
  rw [Finset.sum_comm]
  exact sum_congr rfl fun x _ => sum_congr rfl fun ω _ => by congr

/-- **Tables rarely hold a committed tile**, on any layout of whole tiles with 7-bit lanes in 64-row coin tiles. -/
theorem sum_tableHits_le (he : L.e = 7) (htile : L.tile = 64) (hm : L.m = 16 * L.rowBlocks)
    (hn : L.n = 8 * L.colBlocks) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (σ : Strategy L) (stream : CoinStream L) :
    ∑ ω : Coins L, (tileWords * tableHits L κ σ stream ω : ℚ) ≤ Fintype.card (Coins L) * tableGap L := by
  classical
  have hN : 2 ^ 100 * ∑ ω : Coins L, tableHits L κ σ stream ω ≤
      L.rowBlocks * L.colBlocks * L.S * ((2 * L.mem + 1) * Fintype.card (Coins L)) := by
    rw [sum_tableHits, mul_sum]
    calc _ ≤ ∑ _x ∈ tileIdx L, (2 * L.mem + 1) * Fintype.card (Coins L) := sum_le_sum fun x hx => by
          simp only [tileIdx, mem_product, mem_range] at hx
          exact memHolds_card_le he htile (by omega) (by omega) x.2.2 κ σ stream
      _ = _ := by
          rw [sum_const, smul_eq_mul, tileIdx, card_product, card_product, card_range, card_range, card_univ,
            Fintype.card_fin]
          ring
  have hNq : (2 : ℚ) ^ 100 * ∑ ω : Coins L, (tableHits L κ σ stream ω : ℚ) ≤
      (L.rowBlocks * L.colBlocks * L.S : ℚ) * ((2 * L.mem + 1) * Fintype.card (Coins L)) := by
    exact_mod_cast hN
  set H := ∑ ω : Coins L, (tableHits L κ σ stream ω : ℚ)
  have hH : H ≤ (L.rowBlocks * L.colBlocks * L.S : ℚ) * ((2 * L.mem + 1) * Fintype.card (Coins L)) / 2 ^ 100 := by
    rw [le_div_iff₀ (by positivity)]
    linarith
  rw [← mul_sum, tableGap, show ((tileWords : ℕ) : ℚ) = 128 by norm_num [tileWords]]
  calc (128 : ℚ) * H
      ≤ 128 * ((L.rowBlocks * L.colBlocks * L.S : ℚ) * ((2 * L.mem + 1) * Fintype.card (Coins L)) / 2 ^ 100) :=
        mul_le_mul_of_nonneg_left hH (by norm_num)
    _ = Fintype.card (Coins L) *
          ((L.rowBlocks * L.colBlocks * 128 : ℚ) * (L.S * (2 * L.mem + 1 : ℚ) / 2 ^ 100)) := by
        ring

end Memory

/-- **Tables rarely hold a committed tile**: for any data, every strategy's pre-coin tables
and the certifier's coin stream, the committed tiles its memory starts with, in words and averaged over the coins,
are at most `tableGap`. Every step carries the coin lanes, so a tile at any step is a given value with chance at most
`2^-168` (`call_tile_pstar`: each column's even and odd rows fix its three lanes) and a streamed tile with chance at most
`2^-105` (`call_tile_stream`: at least five columns free of the stream's lanes); a union over the memory's
`2 · mem + 1` values (`memHolds_card_le`) gives `(2 · mem + 1) · 2^-100` a tile and step, which is `tableGap`.
Reads `coinTerm`, `laneTerm`, `word`, `tileOf` and `CoinStream`. -/
theorem table_le (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (σ : Strategy (lead T m n K)) (stream : CoinStream (lead T m n K)) :
    ∑ ω : Coins (lead T m n K), (tileWords * tableHits (lead T m n K) κ σ stream ω : ℚ) ≤
      Fintype.card (Coins (lead T m n K)) * tableGap (lead T m n K) := by
  refine sum_tableHits_le rfl rfl ?_ ?_ κ σ stream
  · show padRows m = 16 * ((padRows m + 15) / 16)
    unfold padRows
    omega
  · show padCols n = 8 * ((padCols n + 7) / 8)
    unfold padCols
    omega

end Pouw.SecurityProofs.Lead.PStar
