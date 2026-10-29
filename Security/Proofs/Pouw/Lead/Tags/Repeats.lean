import Proofs.Pouw.Lead.Tags.Tile

/-!
# The lead's repeated tiles, by pairs

A call's repeated tiles are at most its ordered pairs of distinct tiles that agree (`card_sub_card_image_le`), and
each pair agrees on a share of the coins bounded by the coins that separate it:

* `tile_lane_card_le`: a tile at step `t ≥ 1` against one at an earlier step, or at step `t` in another coin tile or
  column block, by the tile's step-`t` coin lane, `2^−7` a column;
* `tile_cross_card_le`: two tiles in different column blocks, by two of the first tile's rows, whose tags differ by
  the unit digit and one of which has a nonzero higher digit, against each column's tag weights, `7^−2` a column;
* `tile_agree_card_le` (`Tile`): any two tiles at different places, by a row tag, `7^−1` a column.

`repeats_le_of`: for at most 28,560 rows, `2^20` columns and an inner dimension from 4,096 to 65,536, the repeats in
words and averaged over the coins are at most `coinBranchGap`.
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

/-- A tile's row `r` holds its block's columns: position `8 r + j mod 8` is row `16 rb + r`'s word in column `j`. -/
theorem tileOf_at (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) {rb cb r : ℕ} (hr : r < 16)
    (hrb : 16 * rb + r < L.m) (t : Fin L.S) {j : Fin L.n} (hj : (j : ℕ) / 8 = cb) :
    tileOf L κ ω rb cb t ⟨8 * r + (j : ℕ) % 8, by unfold tileWords; omega⟩ = word L κ ω ⟨16 * rb + r, hrb⟩ t j := by
  have h1 : 16 * rb + (8 * r + (j : ℕ) % 8) / 8 = 16 * rb + r := by omega
  have h2 : 8 * cb + (8 * r + (j : ℕ) % 8) % 8 = j := by
    have := Nat.div_add_mod (j : ℕ) 8; rw [hj] at this; omega
  unfold tileOf
  rw [dite_eq_left ⟨by simp only [h1]; exact hrb, by simp only [h2]; exact j.isLt⟩]
  congr 1 <;> exact Fin.ext (by simp only [h1, h2])

/-- A tile reads no coin lane of step `t₀` outside its own coin tile and column block, nor of a later step. -/
theorem tileOf_updU (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (htile : L.tile = 64) {t₀ : Fin L.S}
    (h0 : (t₀ : ℕ) ≠ 0) (k₀ : Fin L.tiles) (ℓ₀ : Fin 3) {cb rb' cb' : ℕ} {t' : Fin L.S}
    (h : t' < t₀ ∨ cb' ≠ cb ∨ rb' / 4 ≠ k₀) (ω : Coins L) (v : Fin L.n → Fin (2 ^ L.e)) (p : Fin tileWords) :
    tileOf L κ ((blockU L t₀ k₀ ℓ₀ (colsOf L cb)).upd ω v) rb' cb' t' p = tileOf L κ ω rb' cb' t' p := by
  have hp : (p : ℕ) < 128 := p.2
  unfold tileOf
  split_ifs with hc
  · have := word_updU κ (by omega) h0 k₀ ℓ₀ (colsOf L cb) ω v ⟨16 * rb' + (p : ℕ) / 8, hc.1⟩ t'
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

/-- **Through a tile's own coin lane**: a tile at step `t ≥ 1` and one at an earlier step, or at step `t` in another
coin tile or column block, agree on at most a `2^−e|J|` share of the coins, `J` the first tile's columns. -/
theorem tile_lane_card_le (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (htile : L.tile = 64) {rb rb' cb cb' : ℕ}
    {t t' : Fin L.S} (hrb : rb < L.rowBlocks) (h0 : (t : ℕ) ≠ 0) (h : t' < t ∨ cb' ≠ cb ∨ rb' / 4 ≠ rb / 4) :
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
        have a := word_updU κ ht0 h0 k₀ 0 J ω v ⟨16 * rb, hr⟩ t j
        rw [ite_eq_left ⟨hj, rfl, le_rfl⟩] at a
        have hg : (((blockU L t k₀ 0 J).get ω j : ℕ) : ℤ) = ((ω.2.2.2 t k₀ 0 j : ℕ) : ℤ) := rfl
        rw [hg]
        linear_combination a)
      fun ω v j _ => tileOf_updU κ htile h0 k₀ 0 (by rw [hk₀]; exact h) ω v _)
  refine mem_filter.2 ⟨mem_univ _, fun j hj => ?_⟩
  rw [← tileOf_col κ ω hr t (hJ j hj)]
  exact congrFun (mem_filter.1 hω).2 _

theorem card_ne_ne : ∀ a : Fin 4, a ≠ 0 → Fintype.card {e : Fin 4 // e ≠ 0 ∧ e ≠ a} = 2 := by decide

/-- **At most `7^2` of a column's draws meet two targets** when two tags have a nonzero minor at the unit digit and
digit `a`. -/
theorem col2_card_le (d₁ d₂ : Fin 4 → ℤ) {a : Fin 4} (ha : a ≠ 0) (hmin : d₁ 0 * d₂ a - d₁ a * d₂ 0 ≠ 0)
    (c₁ c₂ : ℤ) :
    (univ.filter fun g : Fin 4 → Fin 7 => tagTerm 3 d₁ g = c₁ ∧ tagTerm 3 d₂ g = c₂).card ≤ 49 := by
  let f : (Fin 4 → Fin 7) → ({e : Fin 4 // e ≠ 0 ∧ e ≠ a} → Fin 7) := fun g e => g e
  have hsplit : ∀ (d : Fin 4 → ℤ) (g : Fin 4 → Fin 7), tagTerm 3 d g =
      d 0 * ((g 0 : ℤ) - 3) + d a * ((g a : ℤ) - 3) + ∑ e ∈ (univ.erase 0).erase a, d e * ((g e : ℤ) - 3) := by
    intro d g
    unfold tagTerm
    rw [← add_sum_erase _ _ (mem_univ 0), ← add_sum_erase _ _ (mem_erase.2 ⟨ha, mem_univ a⟩), add_assoc]
  have hinj : Set.InjOn f (univ.filter fun g : Fin 4 → Fin 7 => tagTerm 3 d₁ g = c₁ ∧ tagTerm 3 d₂ g = c₂) := by
    intro g hg g' hg' hfg
    simp only [coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at hg hg'
    have hoff : ∀ e, e ≠ 0 → e ≠ a → g e = g' e := fun e h1 h2 => congrFun hfg ⟨e, h1, h2⟩
    have hrest : ∀ d : Fin 4 → ℤ, ∑ e ∈ (univ.erase 0).erase a, d e * ((g e : ℤ) - 3) =
        ∑ e ∈ (univ.erase 0).erase a, d e * ((g' e : ℤ) - 3) := fun d =>
      sum_congr rfl fun e he => by
        rw [hoff e (ne_of_mem_erase (mem_of_mem_erase he)) (ne_of_mem_erase he)]
    have e1 := hg.1.trans hg'.1.symm
    have e2 := hg.2.trans hg'.2.symm
    rw [hsplit, hsplit d₁ g', hrest] at e1
    rw [hsplit, hsplit d₂ g', hrest] at e2
    have h0 : (d₁ 0 * d₂ a - d₁ a * d₂ 0) * ((g 0 : ℤ) - g' 0) = 0 := by linear_combination d₂ a * e1 - d₁ a * e2
    have ha' : (d₁ 0 * d₂ a - d₁ a * d₂ 0) * ((g a : ℤ) - g' a) = 0 := by linear_combination d₁ 0 * e2 - d₂ 0 * e1
    have g0 : g 0 = g' 0 := Fin.ext (by
      have := (mul_eq_zero.1 h0).resolve_left hmin; omega)
    have ga : g a = g' a := Fin.ext (by
      have := (mul_eq_zero.1 ha').resolve_left hmin; omega)
    funext e
    by_cases e0 : e = 0
    · rw [e0, g0]
    by_cases ea : e = a
    · rw [ea, ga]
    exact hoff e e0 ea
  have hcard : Fintype.card {e : Fin 4 // e ≠ 0 ∧ e ≠ a} = 2 := card_ne_ne a ha
  calc _ ≤ (univ : Finset ({e : Fin 4 // e ≠ 0 ∧ e ≠ a} → Fin 7)).card :=
        card_le_card_of_injOn f (by intro _ _; exact mem_univ _) hinj
    _ = 49 := by rw [card_univ, Fintype.card_fun, hcard, Fintype.card_fin]; norm_num

/-- **Through two rows' tags**: two forms that move with column `j`'s tag weights by tags with a nonzero minor at the
unit digit and digit `a` meet targets that read no tag weight of `J` on every column of `J` for at most a `7^−2|J|`
share of the coins. -/
theorem agreeG2_card_le (J : Finset (Fin L.n)) (F₁ F₂ y₁ y₂ : Coins L → Fin L.n → ℤ) (d₁ d₂ : Fin 4 → ℤ)
    {a : Fin 4} (ha : a ≠ 0) (hmin : d₁ 0 * d₂ a - d₁ a * d₂ 0 ≠ 0)
    (hF₁ : ∀ ω v, ∀ j ∈ J, F₁ ((blockG L J).upd ω v) j - F₁ ω j =
      tagTerm 3 d₁ (v j) - tagTerm 3 d₁ ((blockG L J).get ω j))
    (hF₂ : ∀ ω v, ∀ j ∈ J, F₂ ((blockG L J).upd ω v) j - F₂ ω j =
      tagTerm 3 d₂ (v j) - tagTerm 3 d₂ ((blockG L J).get ω j))
    (hy₁ : ∀ ω v, ∀ j ∈ J, y₁ ((blockG L J).upd ω v) j = y₁ ω j)
    (hy₂ : ∀ ω v, ∀ j ∈ J, y₂ ((blockG L J).upd ω v) j = y₂ ω j) :
    (univ.filter fun ω => ∀ j ∈ J, F₁ ω j = y₁ ω j ∧ F₂ ω j = y₂ ω j).card * 49 ^ J.card ≤
      Fintype.card (Coins L) := by
  classical
  have key := (blockG L J).card_le (fun _ => 3)
    (fun j r g => F₁ r j + tagTerm 3 d₁ g = y₁ r j ∧ F₂ r j + tagTerm 3 d₂ g = y₂ r j) (M := 49) fun j hj r => by
      refine le_of_eq_of_le (congrArg card (filter_congr fun g _ => ?_))
        (col2_card_le d₁ d₂ ha hmin (y₁ r j - F₁ r j) (y₂ r j - F₂ r j))
      constructor <;> rintro ⟨h1, h2⟩ <;> constructor <;> linarith
  have h0 : ∀ d : Fin 4 → ℤ, tagTerm 3 d (fun _ : Fin 4 => (3 : Fin 7)) = 0 := fun d => by simp [tagTerm]
  have hset : (univ.filter fun ω => ∀ j ∈ J, F₁ ω j = y₁ ω j ∧ F₂ ω j = y₂ ω j) = univ.filter fun ω => ∀ j ∈ J,
      F₁ ((blockG L J).upd ω fun _ => fun _ => 3) j + tagTerm 3 d₁ ((blockG L J).get ω j) =
          y₁ ((blockG L J).upd ω fun _ => fun _ => 3) j ∧
        F₂ ((blockG L J).upd ω fun _ => fun _ => 3) j + tagTerm 3 d₂ ((blockG L J).get ω j) =
          y₂ ((blockG L J).upd ω fun _ => fun _ => 3) j := by
    refine filter_congr fun ω _ => forall₂_congr fun j hj => ?_
    have h1 := hF₁ ω (fun _ => fun _ => 3) j hj
    have h2 := hF₂ ω (fun _ => fun _ => 3) j hj
    rw [h0] at h1 h2
    rw [hy₁ ω _ j hj, hy₂ ω _ j hj]
    constructor <;> rintro ⟨e1, e2⟩ <;> constructor <;> linarith
  rw [hset]
  have hB : Fintype.card (Fin 4 → Fin 7) = 49 * 49 := by
    rw [Fintype.card_fun, Fintype.card_fin, Fintype.card_fin]; norm_num
  rw [hB] at key
  refine Nat.le_of_mul_le_mul_left ?_ (by positivity : 0 < 49 ^ J.card)
  calc 49 ^ J.card * (_ * 49 ^ J.card) = _ * (49 * 49) ^ J.card := by rw [mul_pow, mul_left_comm]
    _ ≤ _ := key
    _ = _ := by ring

/-- Two consecutive values without a carry differ in the unit digit alone. -/
theorem bal13_succ {w : ℕ} (hw : (w + 14280) % 13 ≠ 12) (q : Fin 4) :
    bal13 ((w + 1 : ℕ) : ℤ) q = bal13 (w : ℤ) q + if q = 0 then 1 else 0 := by
  unfold bal13
  fin_cases q <;> simp <;> omega

/-- From 7 to 14,280 a value has a nonzero digit above the unit digit. -/
theorem bal13_high {w : ℕ} (h1 : 7 ≤ w) (h2 : w ≤ 14280) : ∃ a : Fin 4, a ≠ 0 ∧ bal13 (w : ℤ) a ≠ 0 := by
  by_contra hc
  push Not at hc
  have e1 := hc 1 (by decide)
  have e2 := hc 2 (by decide)
  have e3 := hc 3 (by decide)
  simp only [bal13] at e1 e2 e3
  norm_num at e1 e2 e3
  omega

/-- **Every row block below 28,560 rows has two rows whose tags span two digits**: rows `16 rb + r` and
`16 rb + r + 2`, whose tags differ by the unit digit, the first with a nonzero digit `a` above it. -/
theorem rowTag_pair {rb : ℕ} (hrb : 16 * rb + 15 < 28560) : ∃ r : ℕ, r + 2 < 16 ∧ ∃ a : Fin 4, a ≠ 0 ∧
    (∀ q, rowTag (16 * rb + (r + 2)) q = rowTag (16 * rb + r) q + if q = 0 then 1 else 0) ∧
      rowTag (16 * rb + r) a ≠ 0 := by
  have hev : ∀ r, r % 2 = 0 → rowTag (16 * rb + r) = bal13 (((16 * rb + r) / 2 + 1 : ℕ) : ℤ) := fun r hr => by
    unfold rowTag; rw [ite_eq_left (by omega)]
  by_cases hw : (8 * rb + 7 + 14280) % 13 = 12
  · refine ⟨10, by norm_num, ?_⟩
    have hz : (16 * rb + 10) / 2 + 1 = 8 * rb + 6 := by omega
    have hz' : (16 * rb + (10 + 2)) / 2 + 1 = 8 * rb + 6 + 1 := by omega
    obtain ⟨a, ha, hne⟩ := bal13_high (w := 8 * rb + 6) (by omega) (by omega)
    refine ⟨a, ha, fun q => ?_, ?_⟩
    · rw [hev _ (by omega), hev _ (by omega), hz, hz']
      exact bal13_succ (by omega) q
    · rw [hev _ (by omega), hz]; exact hne
  · refine ⟨12, by norm_num, ?_⟩
    have hz : (16 * rb + 12) / 2 + 1 = 8 * rb + 7 := by omega
    have hz' : (16 * rb + (12 + 2)) / 2 + 1 = 8 * rb + 7 + 1 := by omega
    obtain ⟨a, ha, hne⟩ := bal13_high (w := 8 * rb + 7) (by omega) (by omega)
    refine ⟨a, ha, fun q => ?_, ?_⟩
    · rw [hev _ (by omega), hev _ (by omega), hz, hz']
      exact bal13_succ hw q
    · rw [hev _ (by omega), hz]; exact hne

/-- **Through two rows' tags**: tiles in different column blocks agree on at most a `7^−2|J|` share of the coins, `J`
the first tile's columns, below 28,560 rows. -/
theorem tile_cross_card_le (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (hm : L.m ≤ 28560) {rb rb' cb cb' : ℕ}
    (t t' : Fin L.S) (hrow : 16 * rb + 16 ≤ L.m) (hcb : cb' ≠ cb) :
    (univ.filter fun ω => tileOf L κ ω rb cb t = tileOf L κ ω rb' cb' t').card * 49 ^ (colsOf L cb).card ≤
      Fintype.card (Coins L) := by
  obtain ⟨r, hr2, a, ha, hpair, hne⟩ := rowTag_pair (rb := rb) (by omega)
  have hi₁ : 16 * rb + r < L.m := by omega
  have hi₂ : 16 * rb + (r + 2) < L.m := by omega
  set J := colsOf L cb
  have hJ : ∀ j ∈ J, (j : ℕ) / 8 = cb := fun j hj => (mem_filter.1 hj).2
  have hJ' : ∀ j ∈ J, (j : ℕ) / 8 ≠ cb' := fun j hj h => hcb (h.symm.trans (hJ j hj))
  have hmin : rowTag (16 * rb + r) 0 * rowTag (16 * rb + (r + 2)) a -
      rowTag (16 * rb + r) a * rowTag (16 * rb + (r + 2)) 0 ≠ 0 := by
    rw [hpair a, hpair 0, ite_eq_right ha, ite_eq_left rfl]
    intro h
    exact hne (by linarith)
  refine (Nat.mul_le_mul_right _ (card_le_card fun ω hω => ?_)).trans
    (agreeG2_card_le J (fun ω j => word L κ ω ⟨16 * rb + r, hi₁⟩ t j)
      (fun ω j => word L κ ω ⟨16 * rb + (r + 2), hi₂⟩ t j)
      (fun ω j => tileOf L κ ω rb' cb' t' ⟨8 * r + (j : ℕ) % 8, by unfold tileWords; omega⟩)
      (fun ω j => tileOf L κ ω rb' cb' t' ⟨8 * (r + 2) + (j : ℕ) % 8, by unfold tileWords; omega⟩)
      (rowTag (16 * rb + r)) (rowTag (16 * rb + (r + 2))) ha hmin
      (fun ω v j hj => by rw [word_updG, ite_eq_left hj]) (fun ω v j hj => by rw [word_updG, ite_eq_left hj])
      (fun ω v j _ => tileOf_updG κ hJ' ω v t' _) fun ω v j _ => tileOf_updG κ hJ' ω v t' _)
  have hω' := (mem_filter.1 hω).2
  refine mem_filter.2 ⟨mem_univ _, fun j hj => ⟨?_, ?_⟩⟩
  · rw [← tileOf_at κ ω (by omega) hi₁ t (hJ j hj), hω']
  · rw [← tileOf_at κ ω (by omega) hi₂ t (hJ j hj), hω']

end Pouw.SecurityProofs.Lead.Tags
