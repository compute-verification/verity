import Definitions.Pouw.Lead.Gap
import Proofs.Pouw.Lead.Tags.Tile

/-!
# The lead's repeated tiles, by pairs

A call's repeated tiles are at most its ordered pairs of distinct tiles that agree (`card_sub_card_image_le`), and
each pair agrees on a share of the coins bounded by the coins that separate it (`pair_le`):

* `tile_lane_card_le`: a tile against one at an earlier step, or at its step in another coin tile or column block, by
  the tile's own coin lane at its step, `2^−7` a column;
* `tile_rows_card_le` (`Tile`): two tiles at one step, column block and coin tile, by the difference of their first
  rows' tags (`rowTag_block_ne`, for any number of rows) against each column's tag weights, `7^−1` a column.

`repeats_le_of`: for an inner dimension of at least 1 and at most `2^40` tiles a call, the repeats in words and
averaged over the coins are at most `coinBranchGap`; `repeats_le_caps` for at most `2^15` rows, `2^20` columns and an
inner dimension of at most `2^16`.
-/

namespace Pouw.SecurityProofs.Lead.Tags

open Finset Pouw.Lead Pouw.Dimension.Tag

/-! ## Repeats by colliding pairs -/

/-- A set's elements less its image's are at most its ordered pairs of distinct elements with one value. -/
theorem card_sub_card_image_le {α β : Type} [DecidableEq α] [DecidableEq β] (A : Finset α) (f : α → β) :
    A.card - (A.image f).card ≤ ∑ x ∈ A, ((A.erase x).filter fun y => f y = f x).card := by
  set B := A.filter fun x => ∃ y ∈ A.erase x, f y = f x
  have hB : B ⊆ A := filter_subset _ _
  have h1 : (A \ B).card ≤ (A.image f).card := by
    refine card_le_card_of_injOn f (by intro a ha; exact mem_image_of_mem f (mem_sdiff.1 ha).1) ?_
    intro a ha b hb hab
    by_contra hne
    have ha' := mem_sdiff.1 ha
    exact ha'.2 (mem_filter.2 ⟨ha'.1, b, mem_erase.2 ⟨fun h => hne h.symm, (mem_sdiff.1 hb).1⟩, hab.symm⟩)
  have h2 := card_sdiff_add_card_eq_card hB
  have h3 : B.card ≤ ∑ x ∈ A, ((A.erase x).filter fun y => f y = f x).card := by
    calc B.card = ∑ _x ∈ B, 1 := by simp
      _ ≤ ∑ x ∈ B, ((A.erase x).filter fun y => f y = f x).card := sum_le_sum fun x hx => by
          obtain ⟨y, hy, hfy⟩ := (mem_filter.1 hx).2
          exact card_pos.2 ⟨y, mem_filter.2 ⟨hy, hfy⟩⟩
      _ ≤ _ := sum_le_sum_of_subset hB
  omega

variable {L : Pouw.Lead.Layout}

/-- **Summed over the coins, the repeats** are at most, over ordered pairs of distinct tiles, the coins where they
agree. -/
theorem sum_repeats_le (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) :
    ∑ ω : Coins L, repeats L κ ω ≤ ∑ x ∈ tileIdx L, ∑ y ∈ (tileIdx L).erase x,
      (univ.filter fun ω : Coins L => tileOf L κ ω y.1 y.2.1 y.2.2 = tileOf L κ ω x.1 x.2.1 x.2.2).card := by
  calc ∑ ω : Coins L, repeats L κ ω ≤ ∑ ω : Coins L, ∑ x ∈ tileIdx L,
        (((tileIdx L).erase x).filter fun y => tileOf L κ ω y.1 y.2.1 y.2.2 = tileOf L κ ω x.1 x.2.1 x.2.2).card :=
        sum_le_sum fun ω _ => card_sub_card_image_le _ _
    _ = _ := by
        simp only [card_filter]
        rw [sum_comm]
        exact sum_congr rfl fun x _ => sum_comm

/-! ## Each pair's separating coins -/

/-- A tile reads no coin lane of step `t₀` outside its own coin tile and column block, nor of a later step. -/
theorem tileOf_updU (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (htile : L.tile = 64) (t₀ : Fin L.S) (k₀ : Fin L.tiles) (ℓ₀ : Fin 3) {cb rb' cb' : ℕ} {t' : Fin L.S}
    (h : t' < t₀ ∨ cb' ≠ cb ∨ rb' / 4 ≠ k₀) (ω : Coins L) (v : Fin L.n → Fin (2 ^ L.e)) (p : Fin tileWords) :
    tileOf L κ ((blockU L t₀ k₀ ℓ₀ (colsOf L cb)).upd ω v) rb' cb' t' p = tileOf L κ ω rb' cb' t' p := by
  have hp : (p : ℕ) < 128 := p.2
  unfold tileOf
  split_ifs with hc
  · have := word_updU κ (by omega) t₀ k₀ ℓ₀ (colsOf L cb) ω v ⟨16 * rb' + (p : ℕ) / 8, hc.1⟩ t'
      ⟨8 * cb' + (p : ℕ) % 8, hc.2⟩
    rw [ite_eq_right ?_] at this
    · linarith
    rintro ⟨hj, hk, ht⟩
    have hj' : (8 * cb' + (p : ℕ) % 8) / 8 = cb := (mem_filter.1 hj).2
    have hk' : (16 * rb' + (p : ℕ) / 8) / L.tile = k₀ := congrArg Fin.val hk
    rw [htile] at hk'
    rcases h with h | h | h
    · exact absurd ht (not_le.2 h)
    · exact h (by omega)
    · exact h (by omega)
  · rfl

/-- **Through a tile's own coin lane**: a tile at step `t` and one at an earlier step, or at step `t` in another coin
tile or column block, agree on at most a `2^−e|J|` share of the coins, `J` the first tile's columns. -/
theorem tile_lane_card_le (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (htile : L.tile = 64) {rb rb' cb cb' : ℕ}
    {t t' : Fin L.S} (hrb : rb < L.rowBlocks) (h : t' < t ∨ cb' ≠ cb ∨ rb' / 4 ≠ rb / 4) :
    (univ.filter fun ω => tileOf L κ ω rb cb t = tileOf L κ ω rb' cb' t').card * (2 ^ L.e) ^ (colsOf L cb).card ≤
      Fintype.card (Coins L) := by
  have hr := row_lt hrb
  have ht0 : 0 < L.tile := by omega
  set k₀ := rowTile ht0 ⟨16 * rb, hr⟩
  have hk₀ : (k₀ : ℕ) = rb / 4 := by
    show 16 * rb / L.tile = rb / 4
    rw [htile]; omega
  set J := colsOf L cb
  have hJ : ∀ j ∈ J, (j : ℕ) / 8 = cb := fun j hj => (mem_filter.1 hj).2
  refine (Nat.mul_le_mul_right _ (card_le_card fun ω hω => ?_)).trans
    (agreeU_card_le t k₀ 0 J (fun ω j => word L κ ω ⟨16 * rb, hr⟩ t j)
      (fun ω j => tileOf L κ ω rb' cb' t' ⟨(j : ℕ) % 8, pos_lt j⟩)
      (fun _ => rotWeight (16 * rb) 0 * 2) (fun _ _ => by have := rotWeight_ne_zero (16 * rb) 0; omega)
      (fun ω v j hj => by
        have a := word_updU κ ht0 t k₀ 0 J ω v ⟨16 * rb, hr⟩ t j
        rw [ite_eq_left ⟨hj, rfl, le_rfl⟩] at a
        have hg : (((blockU L t k₀ 0 J).get ω j : ℕ) : ℤ) = ((ω.2.2.2 t k₀ 0 j : ℕ) : ℤ) := rfl
        rw [hg]
        linear_combination a)
      fun ω v j _ => tileOf_updU κ htile t k₀ 0 (by rw [hk₀]; exact h) ω v _)
  refine mem_filter.2 ⟨mem_univ _, fun j hj => ?_⟩
  rw [← tileOf_col κ ω hr t (hJ j hj)]
  exact congrFun (mem_filter.1 hω).2 _

/-- Equal balanced base-13 digits are equal values modulo `13⁴`. -/
theorem bal13_eq_mod {z z' : ℤ} (h : bal13 z = bal13 z') : (z + 14280) % 28561 = (z' + 14280) % 28561 := by
  have e0 := congrFun h 0
  have e1 := congrFun h 1
  have e2 := congrFun h 2
  have e3 := congrFun h 3
  simp only [bal13, Fin.val_zero, Fin.val_one, Fin.val_two, pow_zero, pow_one] at e0 e1 e2 e3
  norm_num at e0 e1 e2 e3
  omega

/-- **The row blocks of a coin tile have distinct first-row tags**, for any number of rows: rows `16 rb` and
`16 rb'` of one 64-row coin tile have tag values `8 rb + 1` and `8 rb' + 1`, at most 24 apart. -/
theorem rowTag_block_ne {rb rb' : ℕ} (h : rb ≠ rb') (h4 : rb / 4 = rb' / 4) :
    ∃ q, (rowTag (16 * rb) - rowTag (16 * rb')) q ≠ 0 := by
  by_contra hc
  push Not at hc
  have he : ∀ r, rowTag (16 * r) = bal13 (((16 * r) / 2 + 1 : ℕ) : ℤ) := fun r => by
    unfold rowTag; rw [ite_eq_left (by omega)]
  have := bal13_eq_mod ((he rb).symm.trans ((funext fun q => sub_eq_zero.1 (hc q)).trans (he rb')))
  push_cast at this
  omega

/-! ## The lead -/

/-- A pair's share of the coins: `7^−8` for tiles at one step, column block and coin tile, and `2^−56` for any. -/
def pairShare (L : Pouw.Lead.Layout) (x y : ℕ × ℕ × Fin L.S) : ℚ :=
  (if y.2.1 = x.2.1 ∧ y.2.2 = x.2.2 ∧ y.1 / 4 = x.1 / 4 then 1 / 7 ^ 8 else 0) + 1 / 2 ^ 56

theorem pairShare_nonneg (L : Pouw.Lead.Layout) (x y : ℕ × ℕ × Fin L.S) : 0 ≤ pairShare L x y := by
  unfold pairShare; split_ifs <;> positivity

/-- **Each pair of tiles** of a layout of 64-row coin tiles, 7-bit coin lanes and whole column blocks agrees on at most
its `pairShare` of the coins. -/
theorem pair_le (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (htile : L.tile = 64) (he : L.e = 7)
    (hn8 : L.n = 8 * L.colBlocks) {x y : ℕ × ℕ × Fin L.S} (hx : x ∈ tileIdx L) (hy : y ∈ (tileIdx L).erase x) :
    ((univ.filter fun ω : Coins L => tileOf L κ ω y.1 y.2.1 y.2.2 = tileOf L κ ω x.1 x.2.1 x.2.2).card : ℚ) ≤
      Fintype.card (Coins L) * pairShare L x y := by
  obtain ⟨rb, cb, t⟩ := x
  obtain ⟨rb', cb', t'⟩ := y
  obtain ⟨hne, hy'⟩ := mem_erase.1 hy
  simp only [tileIdx, mem_product, mem_range, mem_univ, and_true] at hx hy'
  have hcols : ∀ c, c < L.colBlocks → (colsOf L c).card = 8 := fun c hc => colsOf_card (by omega)
  have hN : (0 : ℚ) ≤ Fintype.card (Coins L) := Nat.cast_nonneg _
  have of_mul : ∀ c D : ℕ, 0 < D → c * D ≤ Fintype.card (Coins L) →
      (c : ℚ) ≤ Fintype.card (Coins L) * (1 / D) := by
    intro c D hD h
    rw [mul_one_div, le_div_iff₀ (by exact_mod_cast hD)]
    exact_mod_cast h
  have hs0 : (0 : ℚ) ≤ if cb' = cb ∧ t' = t ∧ rb' / 4 = rb / 4 then 1 / 7 ^ 8 else 0 := by
    split_ifs <;> positivity
  have hlane2 : (2 ^ L.e) ^ (colsOf L cb).card = 2 ^ 56 := by rw [hcols cb hx.2, he]; norm_num
  have hlane2' : (2 ^ L.e) ^ (colsOf L cb').card = 2 ^ 56 := by rw [hcols cb' hy'.2, he]; norm_num
  have lane : ((univ.filter fun ω : Coins L => tileOf L κ ω rb' cb' t' = tileOf L κ ω rb cb t).card : ℚ) ≤
      Fintype.card (Coins L) * (1 / (2 ^ 56 : ℕ)) →
      ((univ.filter fun ω : Coins L => tileOf L κ ω rb' cb' t' = tileOf L κ ω rb cb t).card : ℚ) ≤
        Fintype.card (Coins L) * pairShare L (rb, cb, t) (rb', cb', t') := fun h => by
    refine h.trans (mul_le_mul_of_nonneg_left ?_ hN)
    unfold pairShare
    push_cast
    linarith
  by_cases hlt : t < t'
  · refine lane (of_mul _ _ (by positivity) ?_)
    rw [← hlane2']
    exact tile_lane_card_le κ htile hy'.1 (Or.inl hlt)
  by_cases hgt : t' < t
  · refine lane (of_mul _ _ (by positivity) ?_)
    rw [← hlane2, filter_congr fun ω _ => eq_comm]
    exact tile_lane_card_le κ htile hx.1 (Or.inl hgt)
  have htt : t' = t := le_antisymm (not_lt.1 hlt) (not_lt.1 hgt)
  subst htt
  by_cases hl : cb' ≠ cb ∨ rb' / 4 ≠ rb / 4
  · refine lane (of_mul _ _ (by positivity) ?_)
    rw [← hlane2']
    exact tile_lane_card_le κ htile hy'.1 (Or.inr (hl.imp (fun h e => h e.symm) fun h e => h e.symm))
  push Not at hl
  obtain ⟨hcb, h4⟩ := hl
  subst hcb
  have hrb : rb' ≠ rb := fun h => hne (by rw [h])
  refine (of_mul _ (7 ^ 8) (by positivity) ?_).trans (mul_le_mul_of_nonneg_left ?_ hN)
  · rw [← hcols cb' hy'.2]
    exact tile_rows_card_le κ hy'.1 hx.1 (rowTag_block_ne hrb h4)
  · unfold pairShare
    rw [ite_eq_left ⟨rfl, rfl, h4⟩]
    push_cast
    linarith

theorem tileIdx_card (L : Pouw.Lead.Layout) : (tileIdx L).card = L.rowBlocks * L.colBlocks * L.S := by
  simp only [tileIdx, card_product, card_range, card_univ, Fintype.card_fin, mul_assoc]

/-- A filter's share sum: `c` on each element meeting `p`. -/
theorem sum_ite_le {α : Type} (s : Finset α) (p : α → Prop) [DecidablePred p] (c : ℚ) (hc : 0 ≤ c) {M : ℕ}
    (hM : (s.filter p).card ≤ M) : ∑ y ∈ s, (if p y then c else 0) ≤ M * c := by
  rw [← sum_filter, sum_const, nsmul_eq_mul]
  exact mul_le_mul_of_nonneg_right (by exact_mod_cast hM) hc

/-- **A layout's repeats** of 64-row coin tiles, 7-bit coin lanes and whole MMA tiles, for at most `2^40` tiles, an
inner dimension of at least 1 and `28 S ≤ K + 34` (`stepsFor`'s): in words and averaged over the coins, at most
`coinBranchGap`. -/
theorem repeats_le_layout (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (htile : L.tile = 64) (he : L.e = 7)
    (hLm : L.m = 16 * L.rowBlocks) (hLn : L.n = 8 * L.colBlocks) (htiles : L.rowBlocks * L.colBlocks * L.S ≤ 2 ^ 40)
    (hS28 : 28 * L.S ≤ L.K + 34) (hK : 1 ≤ L.K) :
    ∑ ω : Coins L, (tileWords * repeats L κ ω : ℚ) ≤ Fintype.card (Coins L) * coinBranchGap L := by
  set N := (Fintype.card (Coins L) : ℚ)
  have hN : 0 ≤ N := Nat.cast_nonneg _
  set R := L.rowBlocks with hR
  set C := L.colBlocks with hC
  have hA : (tileIdx L).card = R * C * L.S := tileIdx_card L
  have hblock : ∀ x ∈ tileIdx L,
      ((tileIdx L).filter fun y => y.2.1 = x.2.1 ∧ y.2.2 = x.2.2 ∧ y.1 / 4 = x.1 / 4).card ≤ 4 := by
    intro x _
    refine (card_le_card_of_injOn (fun y => y.1 % 4) (t := range 4) (fun y _ => by
      simp only [coe_range, Set.mem_Iio]; omega) ?_).trans (card_range 4).le
    intro y hy y' hy' h
    simp only [coe_filter, Set.mem_ofPred_eq] at hy hy'
    have h1 : y.1 = y'.1 := by have := hy.2.2.2; have := hy'.2.2.2; simp only at h; omega
    exact Prod.ext h1 (Prod.ext (hy.2.1.trans hy'.2.1.symm) (hy.2.2.1.trans hy'.2.2.1.symm))
  have hsum : ∀ x ∈ tileIdx L, ∑ y ∈ tileIdx L, pairShare L x y ≤ 4 / 7 ^ 8 + R * C * L.S / 2 ^ 56 := by
    intro x hx
    unfold pairShare
    rw [sum_add_distrib, sum_const, nsmul_eq_mul, hA]
    have a := sum_ite_le (tileIdx L) (fun y => y.2.1 = x.2.1 ∧ y.2.2 = x.2.2 ∧ y.1 / 4 = x.1 / 4) (1 / 7 ^ 8)
      (by positivity) (hblock x hx)
    push_cast at a ⊢
    linarith
  have h1 : ∑ ω : Coins L, (tileWords * repeats L κ ω : ℚ) = 128 * ((∑ ω : Coins L, repeats L κ ω : ℕ) : ℚ) := by
    rw [Nat.cast_sum, mul_sum]; norm_num [tileWords]
  have h2 : ((∑ ω : Coins L, repeats L κ ω : ℕ) : ℚ) ≤
      ∑ x ∈ tileIdx L, ∑ y ∈ tileIdx L, N * pairShare L x y := by
    refine (Nat.cast_le.2 (sum_repeats_le κ)).trans ?_
    push_cast
    refine sum_le_sum fun x hx => (sum_le_sum fun y hy => pair_le κ htile he hLn hx hy).trans ?_
    exact sum_le_sum_of_subset_of_nonneg (erase_subset _ _) fun y _ _ => mul_nonneg hN (pairShare_nonneg L x y)
  have h3 : ∑ x ∈ tileIdx L, ∑ y ∈ tileIdx L, N * pairShare L x y ≤
      (R * C * L.S : ℕ) * (N * (4 / 7 ^ 8 + R * C * L.S / 2 ^ 56)) := by
    rw [← hA, ← nsmul_eq_mul, ← sum_const]
    exact sum_le_sum fun x hx => by rw [← mul_sum]; exact mul_le_mul_of_nonneg_left (hsum x hx) hN
  have hbr : (4 : ℚ) / 7 ^ 8 + R * C * L.S / 2 ^ 56 ≤ 4 / 7 ^ 8 + 2 ^ 40 / 2 ^ 56 := by
    have : ((R * C * L.S : ℕ) : ℚ) ≤ 2 ^ 40 := by exact_mod_cast htiles
    push_cast at this
    gcongr
  have hS28' : 28 * (L.S : ℚ) ≤ L.K + 34 := by exact_mod_cast hS28
  have hK0 : (1 : ℚ) ≤ L.K := by exact_mod_cast hK
  have hkey : 128 * (L.S : ℚ) * (4 / 7 ^ 8 + R * C * L.S / 2 ^ 56) ≤ 84 / 100000 * (4 * L.K) := by
    have := mul_le_mul_of_nonneg_left hbr (by positivity : (0 : ℚ) ≤ 128 * L.S)
    refine this.trans ?_
    norm_num
    linarith
  have hgap : coinBranchGap L = 84 / 100000 * (16 * R * (8 * C) * L.K / 32) := by
    unfold coinBranchGap stockRegs
    rw [hLm, hLn]
    push_cast
    rfl
  rw [h1, hgap]
  calc 128 * ((∑ ω : Coins L, repeats L κ ω : ℕ) : ℚ)
      ≤ 128 * ((R * C * L.S : ℕ) * (N * (4 / 7 ^ 8 + R * C * L.S / 2 ^ 56))) := by
        gcongr; exact h2.trans h3
    _ = N * (R * C) * (128 * L.S * (4 / 7 ^ 8 + R * C * L.S / 2 ^ 56)) := by push_cast; ring
    _ ≤ N * (R * C) * (84 / 100000 * (4 * L.K)) := by gcongr
    _ = N * (84 / 100000 * (16 * R * (8 * C) * L.K / 32)) := by ring

/-- **The lead's repeats**, for an inner dimension of at least 1 and at most `2^40` tiles a call: in words and averaged
over the coins, at most `coinBranchGap`. -/
theorem repeats_le_of (T m n K : ℕ)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ) (hK : 1 ≤ K)
    (htiles : (tileIdx (lead T m n K)).card ≤ 2 ^ 40) :
    ∑ ω : Coins (lead T m n K), (tileWords * repeats (lead T m n K) κ ω : ℚ) ≤
      Fintype.card (Coins (lead T m n K)) * coinBranchGap (lead T m n K) := by
  exact repeats_le_layout κ rfl rfl (by simp only [Layout.rowBlocks, lead, padRows]; omega)
    (by simp only [Layout.colBlocks, lead, padCols]; omega) (tileIdx_card _ ▸ htiles)
    (by simp only [lead, stepsFor]; omega) hK

/-- **The lead's repeats** for at most `2^15` rows, `2^20` columns and an inner dimension from 1 to `2^16`. -/
theorem repeats_le_caps (T m n K : ℕ)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ) (hm : m ≤ 2 ^ 15)
    (hn : n ≤ 2 ^ 20) (hK : 1 ≤ K) (hK' : K ≤ 2 ^ 16) :
    ∑ ω : Coins (lead T m n K), (tileWords * repeats (lead T m n K) κ ω : ℚ) ≤
      Fintype.card (Coins (lead T m n K)) * coinBranchGap (lead T m n K) := by
  refine repeats_le_of T m n K κ hK ?_
  rw [tileIdx_card]
  simp only [Layout.rowBlocks, Layout.colBlocks, lead, padRows, padCols, stepsFor]
  have hR : (16 * ((m + 15) / 16) + 15) / 16 ≤ 2048 := by omega
  have hC : (8 * ((n + 7) / 8) + 7) / 8 ≤ 131072 := by omega
  have hS : 1 + (K - 21 + 27) / 28 ≤ 2342 := by omega
  calc _ ≤ 2048 * 131072 * 2342 := Nat.mul_le_mul (Nat.mul_le_mul hR hC) hS
    _ ≤ 2 ^ 40 := by norm_num

end Pouw.SecurityProofs.Lead.Tags
